import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

class skyRopeConfig(PretrainedConfig):
    model_type = "skyRope"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size",6400)
        self.bos_token_id = kwargs.get("bos_token_id",1)
        self.eos_token_id = kwargs.get("eos_token_id",2)
        self.flash_attn = kwargs.get("flash_attn",True)
        self.num_attention_heads = kwargs.get("num_attention_heads",8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads",4)
        self.head_dim = kwargs.get("head_dim",self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act","silu")
        self.intermediate_size = kwargs.get("intermediate_size",math.ceil(hidden_size * math.pi / 64) * 64)
        self.original_max_position_embeddings = kwargs.get("original_max_position_embeddings",2048)
        self.max_positions = kwargs.get("max_position_embeddings",32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps",1e-6)
        self.rope_theta = kwargs.get("rope_theta",1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings",True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling",True)
        self.rope_scaling = {
            "beta_fast":32,
            "beta_slow":1,
            "factor":16,
            "original_max_position_embeddings":2048,
            "attention_factor":1.0,
            "type":"yarn"
        } if self.inference_rope_scaling else None
        self.num_experts = kwargs.get("num_experts",4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok",1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size",self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob",True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef",5e-4)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim:int, eps:float=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self,x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self,x):
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim:int, end:int=int(32 * 1024), rope_base:float = 1e6, rope_scaling:dict = None):
    freqs,attn_factor = 1.0 / (rope_base ** (torch.arange(0,dim,2)[:(dim//2)].float()/dim)),1.0
    if rope_scaling is not None:
        orig_max,factor,beta_fast,beta_slow,attn_factor = (
            rope_scaling.get("original_max_position_embeddings",2048),
            rope_scaling.get("factor",16),
            rope_scaling.get("beta_fast",32.0),
            rope_scaling.get("beta_slow",1.0),
            rope_scaling.get("attn_factor",1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b:(dim * math.log(orig_max / (2 * math.pi * b))) / (2 * math.log(rope_base))
            low,high = max(math.floor(inv_dim(beta_fast))),min(math.ceil(inv_dim(beta_slow)),dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2,device=freqs.device).float() - low) / max(high-low,0.001),0,1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end,device=freqs.device)
    freqs = torch.outer(t,freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs),torch.sin(freqs)],dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs),torch.cos(freqs)],dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]),dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x:torch.Tensor, n_rep:int) -> torch.Tensor:
    bs,slen,num_key_value_heads,head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:,:,:,None,:].expand(bs,slen,num_key_value_heads,n_rep,head_dim).reshape(bs,slen,num_key_value_heads*n_rep,head_dim))

class SimpleSWCache:
    def __init__(self, window: int):
        self.w = window
        self.keys = None
        self.values = None
        self.cum_len = 0
        self.overlap_ca = None
        self.buffer_kv = None
        self.buffer_gate = None
        self.compressed_kv = None

    def update(self, kv: torch.Tensor):
        b, s, h, d = kv.shape
        self.cum_len += s
        if self.keys is None:
            self.keys = kv
            self.values = kv
        else:
            self.keys = torch.cat([self.keys, kv], dim=1)[:, :, -self.w+1:, :]
            self.values = torch.cat([self.values, kv], dim=1)[:, :, -self.w+1:, :]
        full_kv = torch.cat([self.keys, kv], dim=1)
        return full_kv

    def reset(self):
        self.keys = None
        self.values = None
        self.cum_len = 0
        self.overlap_ca = None
        self.buffer_kv = None
        self.buffer_gate = None
        self.compressed_kv = None

class BaseAttention(nn.Module):
    def __init__(self, config: skyRopeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.eps = config.rms_norm_eps
        self.dropout = config.dropout

        self.q_proj = nn.Linear(self.hidden_size , self.num_attention_heads * self.head_dim,bias=False)
        self.k_proj = nn.Linear(self.hidden_size , self.num_key_value_heads * self.head_dim,bias=False)
        self.v_proj = nn.Linear(self.hidden_size , self.num_key_value_heads * self.head_dim,bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim , self.hidden_size,bias=False)
        self.q_norm = RMSNorm(self.head_dim , eps=self.eps)
        self.k_norm = RMSNorm(self.head_dim , eps=self.eps)
        self.v_norm = RMSNorm(self.head_dim , eps=self.eps)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.flash_attn = config.flash_attn
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and self.flash_attn
        self.sliding_window = config.max_positions // 4

    def _common_qkv_rope(self, x, position_embeddings):
        b, s, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(b, s, self.n_local_heads, self.head_dim)
        xk = xk.view(b, s, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(b, s, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        return xq, xk, xv

    def _attn_calc(self, xq, xk, xv, attention_mask, seq_len):
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)
        if (self.flash
                and (seq_len > 1)
                and (attention_mask is None or torch.all(attention_mask == 1))):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2,-1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                q_len = xq.size(2)
                kv_len = xk.size(-1)
                offset = kv_len - q_len
                mask = torch.ones(q_len, kv_len, device=scores.device, dtype=torch.bool)
                mask.triu_(offset + 1)
                scores[:, :, mask] = -1e9
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1,2).reshape(xq.shape[0] ,seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output

class SWAAttention(BaseAttention):
    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self._common_qkv_rope(x, position_embeddings)
        sw_cache: SimpleSWCache = past_key_value

        if sw_cache is not None:
            xk = sw_cache.update(xk)
            xv = sw_cache.update(xv)
        past_kv = sw_cache if use_cache else None
        out = self._attn_calc(xq, xk, xv, attention_mask, seq_len)
        return out, past_kv

class CSAAttention(BaseAttention):
    def __init__(self, config):
        super().__init__(config)
        self.compress_rate = 4
        self.index_topk = 6
        self.comp_kv = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.comp_gate = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.comp_pos_bias = nn.Parameter(torch.zeros(self.compress_rate, 2 * self.head_dim))
        self.comp_norm = RMSNorm(self.head_dim)
        self.index_q = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.index_weight = nn.Linear(config.hidden_size, 1, bias=False)

    def _compress_chunk(self, chunk_kv, chunk_gate, cache: SimpleSWCache):
        b, n_win, r, _ = chunk_kv.shape
        dh = self.head_dim
        new_kv = chunk_kv.new_zeros((b, n_win, 2*r, dh))
        new_gate = chunk_gate.new_full((b, n_win, 2*r, dh), -1e9)
        new_kv[:, :, r:] = chunk_kv[..., dh:]
        new_gate[:, :, r:] = chunk_gate[..., dh:]
        if n_win > 1:
            new_kv[:, 1:, :r] = chunk_kv[:, :-1, :, :dh]
            new_gate[:, 1:, :r] = chunk_gate[:, :-1, :, :dh]
        if cache.overlap_ca is not None:
            new_kv[:, 0, :r] = cache.overlap_ca
            new_gate[:, 0, :r] = cache.overlap_ca
        gate_soft = new_gate.softmax(dim=2)
        compressed = self.comp_norm((new_kv * gate_soft).sum(dim=2)).unsqueeze(-2)
        cache.overlap_ca = chunk_kv[:, -1, :, :dh].detach()
        return compressed

    def _index_mask(self, q_idx, comp_kv, x):
        comp_flat = comp_kv.mean(dim=-2)
        q_flat = q_idx.squeeze(-2)
        score = torch.matmul(q_flat.float(), comp_flat.transpose(-1, -2).float())
        w = self.index_weight(x).squeeze(-1)[:, :, None]
        score = F.relu(score) * w
        topk = min(self.index_topk, comp_flat.shape[1])
        top_idx = score.topk(topk, dim=-1).indices
        mask = torch.full_like(score, -1e9).scatter(-1, top_idx, 0.0)
        return mask

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        b, s, _ = x.shape
        xq, xk, xv = self._common_qkv_rope(x, position_embeddings)
        sw_cache: SimpleSWCache = past_key_value
        k_base = sw_cache.update(xk) if sw_cache is not None else xk
        v_base = sw_cache.update(xv) if sw_cache is not None else xv

        comp_kv_feat = self.comp_kv(x)
        comp_gate_raw = self.comp_gate(x)
        repeat_num = (s + self.compress_rate - 1) // self.compress_rate
        bias_full = self.comp_pos_bias.repeat(repeat_num, 1)[:s]
        comp_gate_feat = comp_gate_raw + bias_full
        usable = (s // self.compress_rate) * self.compress_rate
        chunk_kv = comp_kv_feat[:, :usable, :].view(b, -1, self.compress_rate, 2 * self.head_dim)
        chunk_gate = comp_gate_feat[:, :usable, :].view(b, -1, self.compress_rate, 2 * self.head_dim)
        compressed = self._compress_chunk(chunk_kv, chunk_gate, sw_cache)
        comp_expand = compressed.expand(-1, -1, self.n_local_kv_heads, -1)
        k_full = torch.cat([k_base, comp_expand], dim=1)
        v_full = torch.cat([v_base, comp_expand], dim=1)

        q_idx = self.index_q(x).view(b, s, 1, self.head_dim)
        index_mask = self._index_mask(q_idx, compressed, x)
        if attention_mask is not None:
            attn_mask_exp = attention_mask.unsqueeze(-1).repeat(1, 1, index_mask.size(-1))
            full_mask = torch.cat([attn_mask_exp, index_mask], dim=-1)
        else:
            full_mask = index_mask

        past_kv = sw_cache if use_cache else None
        out = self._attn_calc(xq, k_full, v_full, full_mask, s)
        return out, past_kv

class HCAAttention(BaseAttention):
    def __init__(self, config):
        super().__init__(config)
        self.compress_rate = 128
        self.comp_kv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.comp_gate = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.comp_pos_bias = nn.Parameter(torch.zeros(self.compress_rate, self.head_dim))
        self.comp_norm = RMSNorm(self.head_dim)

    def _compress_chunk(self, chunk_kv, chunk_gate):
        gate_soft = chunk_gate.softmax(dim=2)
        compressed = self.comp_norm((chunk_kv * gate_soft).sum(dim=2)).unsqueeze(-2)
        return compressed

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        b, s, _ = x.shape
        xq, xk, xv = self._common_qkv_rope(x, position_embeddings)
        sw_cache: SimpleSWCache = past_key_value
        k_base = sw_cache.update(xk) if sw_cache is not None else xk
        v_base = sw_cache.update(xv) if sw_cache is not None else xv

        comp_kv_feat = self.comp_kv(x)
        comp_gate_raw = self.comp_gate(x)
        repeat_num = (s + self.compress_rate - 1) // self.compress_rate
        bias_full = self.comp_pos_bias.repeat(repeat_num, 1)[:s]
        comp_gate_feat = comp_gate_raw + bias_full
        usable = (s // self.compress_rate) * self.compress_rate
        chunk_kv = comp_kv_feat[:, :usable, :].view(b, -1, self.compress_rate, self.head_dim)
        chunk_gate = comp_gate_feat[:, :usable, :].view(b, -1, self.compress_rate, self.head_dim)
        compressed = self._compress_chunk(chunk_kv, chunk_gate)
        comp_expand = compressed.expand(-1, -1, self.n_local_kv_heads, -1)
        k_full = torch.cat([k_base, comp_expand], dim=1)
        v_full = torch.cat([v_base, comp_expand], dim=1)

        past_kv = sw_cache if use_cache else None
        out = self._attn_calc(xq, k_full, v_full, attention_mask, s)
        return out, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: skyRopeConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config: skyRopeConfig, intermediate_size: int = None):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        intermediate_size = intermediate_size or config.moe_intermediate_size
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, intermediate_size, bias=False),
                ACT2FN[config.hidden_act],
                nn.Linear(intermediate_size, config.hidden_size, bias=False)
            ) for _ in range(self.num_experts)
        ])
        self.aux_loss = torch.tensor(0.)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1,hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = torch.where(mask.any(dim=-1))[0]
                weight = topk_weight[mask].unsqueeze(-1)
                y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.num_experts).float().mean((0,1))
            self.aux_loss = (load * scores.mean((0,1))).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class skyRopeBlock(nn.Module):
    def __init__(self, layer_id: int, config: skyRopeConfig, attn_type="swa"):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if attn_type == "swa":
            self.attention = SWAAttention(config)
        elif attn_type == "csa":
            self.attention = CSAAttention(config)
        elif attn_type == "hca":
            self.attention = HCAAttention(config)
        else:
            raise ValueError("attn_type only support swa / csa / hca")
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.attention(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.output_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class skyRopeModel(nn.Module):
    def __init__(self, config: skyRopeConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        attn_type_list = ["csa","csa","swa","swa","swa","swa","csa","csa"]
        self.layers = nn.ModuleList([
            skyRopeBlock(l, config, attn_type=attn_type_list[l])
            for l in range(self.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.original_max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
            )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if past_key_values is not None and not isinstance(past_key_values, list):
            past_key_values = None
        past_key_values = past_key_values or [SimpleSWCache(self.config.max_positions // 4) for _ in range(len(self.layers))]
        start_pos = past_key_values[0].cum_len if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        if self.freqs_cos[0,0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling
            )
            self.freqs_cos = freqs_cos.to(hidden_states.device)
            self.freqs_sin = freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        present_key_values = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present_key_value = layer(
                hidden_states,
                position_embeddings,
                past_key_value,
                use_cache,
                attention_mask
            )
            present_key_values.append(present_key_value)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, present_key_values, aux_loss

class skyRopeForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = skyRopeConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: skyRopeConfig = None):
        self.config = config or skyRopeConfig()
        super().__init__(self.config)
        self.model = skyRopeModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_value, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits= self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x = logits[..., :-1 , :].contiguous()
            y = labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.shape[-1]), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer:
            streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0].cum_len if (past_key_values is not None and len(past_key_values) > 0) else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float("inf")
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break
        if streamer: streamer.end()
        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids

































































