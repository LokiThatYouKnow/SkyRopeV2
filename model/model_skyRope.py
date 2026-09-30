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
        # 环境变量兜底（未设置时行为完全不变）：换词表/MoE 形状时下游脚本无需改动
        import os as _os
        _ei = lambda k, d: int(_os.environ[k]) if _os.environ.get(k) else d
        self.vocab_size = kwargs.get("vocab_size",_ei("SKYROPE_VOCAB",6400))
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
        self.num_experts = kwargs.get("num_experts",_ei("SKYROPE_NUM_EXPERTS",4))
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok",_ei("SKYROPE_TOP_K",1))
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size",_ei("SKYROPE_MOE_INTERMEDIATE",self.intermediate_size))
        self.norm_topk_prob = kwargs.get("norm_topk_prob",True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef",5e-4)
        self.attn_type_list = kwargs.get("attn_type_list",None)
        # MoE 增强开关（默认 = 旧行为）
        self.balance_mode = kwargs.get("balance_mode","aux")        # aux | aux_free
        self.moe_impl = kwargs.get("moe_impl",_os.environ.get("SKYROPE_MOE_IMPL","loop"))  # loop | permute
        self.n_shared_experts = kwargs.get("n_shared_experts",0)    # 常驻共享专家数
        self.aux_free_gamma = kwargs.get("aux_free_gamma",1e-3)     # aux-free bias 更新步长
        # 先进组件开关（默认全关 = 旧行为）
        self.num_nextn_predict_layers = kwargs.get("num_nextn_predict_layers",0)  # MTP 深度
        self.mtp_loss_weight = kwargs.get("mtp_loss_weight",0.3)
        self.z_loss_coef = kwargs.get("z_loss_coef",0.0)            # logit 正则（稳定性）
        self.logit_softcap = kwargs.get("logit_softcap",0.0)        # Gemma2 式 logit 软上限
        self.mla_kv_lora_rank = kwargs.get("mla_kv_lora_rank",256)  # MLA KV 隐向量维度
        self.mla_rope_dim = kwargs.get("mla_rope_dim",32)           # MLA 解耦 RoPE 维度

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
    """滑动窗口 KV cache：只保留最近 (window - 1) 个历史位置，拼上当前步即为完整 KV。

    注意：keys/values 分开存储（旧实现两个字段共用一个 keys，且首步会把 KV 拼成两份）。
    """
    def __init__(self, window: int):
        self.w = window
        self.keys = None
        self.values = None
        self.cum_len = 0
        self.overlap_kv = None
        self.overlap_gate = None
        self.compressed_kv = None

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """传入当前步的 k / v，返回 (完整 k, 完整 v)，并把窗口内历史写回 cache"""
        if self.keys is None:
            full_k, full_v = k, v
        else:
            full_k = torch.cat([self.keys, k], dim=1)
            full_v = torch.cat([self.values, v], dim=1)
        keep = max(self.w - 1, 1)
        self.keys = full_k[:, -keep:, :, :]
        self.values = full_v[:, -keep:, :, :]
        self.cum_len += k.size(1)
        return full_k, full_v

    def reset(self):
        self.keys = None
        self.values = None
        self.cum_len = 0
        self.overlap_kv = None
        self.overlap_gate = None
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
        xq, xk, xv = self.q_norm(xq), self.k_norm(xk), self.v_norm(xv)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        return xq, xk, xv

    def _base_mask_bias(self, x, attention_mask, kb_len, q_len):
        """k_base 部分的加性 mask：右下对齐因果 + padding（0=保留，-1e9=屏蔽）→ [b, q_len, kb_len]"""
        b = x.size(0)
        bias = x.new_zeros((b, 1, kb_len), dtype=torch.float32)
        if attention_mask is not None:
            keep = attention_mask[:, -kb_len:] if attention_mask.size(1) > kb_len else attention_mask
            bias = bias + (1.0 - keep.float())[:, None, :] * -1e9
        bias = bias.expand(-1, q_len, -1)
        cm = torch.ones(q_len, kb_len, device=x.device, dtype=torch.bool).triu_(kb_len - q_len + 1)
        return bias.masked_fill(cm, -1e9)

    def _chunk_causal_bias(self, n_comp, q_len, start, device):
        """压缩 chunk c 覆盖绝对位置 [r*c, r*c+r-1]，只允许 query 看到"严格早于自己"的 chunk。

        这样做的两个理由：
          1. 避免 query 通过压缩表示看到自己所在 chunk 里的未来 token（训练泄漏）；
          2. 保证"整段前向"与"增量解码"看到的压缩集合一致（否则可见范围会随序列总长度变化）。
        返回 (forbid [q_len, n_comp] bool, bias [q_len, n_comp] 加性)
        """
        r = self.compress_rate
        qpos = torch.arange(start, start + q_len, device=device).unsqueeze(1)          # [q, 1]
        cend = (torch.arange(n_comp, device=device).unsqueeze(0) + 1) * r - 1          # [1, n_comp]
        forbid = cend >= qpos
        bias = torch.zeros(q_len, n_comp, device=device, dtype=torch.float32).masked_fill(forbid, -1e9)
        return forbid, bias

    def _attn_calc(self, xq, xk, xv, attention_mask, seq_len, causal=None):
        """统一的注意力计算（可走 SDPA 快速路径）。

        causal=None 时用 self.is_causal；CSA/HCA 因为把压缩 token 追加在 KV 尾部，
        causal mask 必须由它们自己构造（见 _chunk_causal_bias），故传 causal=False。

        attention_mask 支持两种形式：
          - 2D padding mask [b, L]（0/1，1 有效）：按 key 侧对齐到 KV 的**最后** L 个位置；
          - 3D 加性 mask [b, q_len, kv_len]（0=保留，负值=屏蔽）：CSA 的 index mask 用这种。
        """
        causal = self.is_causal if causal is None else causal
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)
        b, q_len, kv_len = xq.size(0), xq.size(2), xk.size(2)
        offset = kv_len - q_len

        bias = None
        if attention_mask is not None:
            m = attention_mask
            if m.dim() == 3:
                if m.size(-1) != kv_len:
                    raise ValueError(f'attention mask 最后一维({m.size(-1)}) 与 kv_len({kv_len}) 不一致')
                bias = m.float().unsqueeze(1)
            else:
                keep = m[:, -kv_len:] if m.size(1) > kv_len else m
                bias = (1.0 - keep.float()).view(b, 1, 1, -1) * -1e9
                if keep.size(1) < kv_len:  # 多出来的压缩位不屏蔽
                    bias = torch.cat([bias.new_zeros(b, 1, 1, kv_len - keep.size(1)), bias], dim=-1)

        # SDPA 的 is_causal 是"左上对齐"，只有 q_len == kv_len 时才等价于右下对齐的因果 mask；
        # q_len == 1（增量解码）时因果 mask 恒为全可见，必须显式关掉 is_causal。
        need_mask = (bias is not None) or (causal and q_len > 1 and offset != 0)
        if not need_mask:
            if self.flash:
                output = F.scaled_dot_product_attention(
                    xq, xk, xv,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=causal and q_len == kv_len)
            else:
                scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
                if causal:
                    cm = torch.ones(q_len, kv_len, device=xq.device, dtype=torch.bool).triu_(offset + 1)
                    scores = scores.masked_fill(cm, -1e9)
                output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        else:
            combined = xq.new_zeros((b, 1, q_len, kv_len), dtype=torch.float32)
            if causal:
                cm = torch.ones(q_len, kv_len, device=xq.device, dtype=torch.bool).triu_(offset + 1)
                combined = combined.masked_fill(cm, -1e9)
            if bias is not None:
                combined = combined + bias
            if xq.dtype in (torch.float16, torch.bfloat16):   # -1e9 直接转 fp16 会溢出成 -inf
                combined = combined.clamp(min=float(torch.finfo(xq.dtype).min))
            if self.flash:
                output = F.scaled_dot_product_attention(
                    xq, xk, xv,
                    attn_mask=combined.to(xq.dtype),
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False)
            else:
                scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) + combined
                output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1,2).reshape(b, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output

class SWAAttention(BaseAttention):
    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self._common_qkv_rope(x, position_embeddings)
        sw_cache: SimpleSWCache = past_key_value

        if use_cache and sw_cache is not None:
            xk, xv = sw_cache.update(xk, xv)
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
        neg = -1e9 if chunk_gate.dtype == torch.float32 else float(torch.finfo(chunk_gate.dtype).min)
        new_gate = chunk_gate.new_full((b, n_win, 2*r, dh), neg)   # fp16 放不下 -1e9，按 dtype 取安全下界
        new_kv[:, :, r:] = chunk_kv[..., dh:]
        new_gate[:, :, r:] = chunk_gate[..., dh:]
        if n_win > 1:
            new_kv[:, 1:, :r] = chunk_kv[:, :-1, :, :dh]
            new_gate[:, 1:, :r] = chunk_gate[:, :-1, :, :dh]
        if cache is not None and n_win > 0 and cache.overlap_kv is not None:
            new_kv[:, 0, :r] = cache.overlap_kv
            new_gate[:, 0, :r] = cache.overlap_gate
        gate_soft = new_gate.softmax(dim=2)
        compressed = self.comp_norm((new_kv * gate_soft).sum(dim=2)).unsqueeze(-2)
        if cache is not None and n_win > 0:
            cache.overlap_kv = chunk_kv[:, -1, :, :dh].detach()
            cache.overlap_gate = chunk_gate[:, -1, :, :dh].detach()
        return compressed

    def _index_mask(self, q_idx, comp_kv, x, forbid=None):
        comp_flat = comp_kv.mean(dim=-2)
        q_flat = q_idx.squeeze(-2)
        score = torch.matmul(q_flat.float(), comp_flat.transpose(-1, -2).float())
        w = self.index_weight(x).squeeze(-1)[:, :, None]
        score = F.relu(score) * w
        if forbid is not None:                       # 未来 chunk 先压到 -1e9，避免 topk 占名额
            score = score.masked_fill(forbid.unsqueeze(0), -1e9)
        topk = min(self.index_topk, comp_flat.shape[1])
        top_idx = score.topk(topk, dim=-1).indices
        mask = torch.full_like(score, -1e9).scatter(-1, top_idx, 0.0)
        return mask

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        b, s, _ = x.shape
        xq, xk, xv = self._common_qkv_rope(x, position_embeddings)
        sw_cache: SimpleSWCache = past_key_value
        if use_cache and sw_cache is not None:
            k_base, v_base = sw_cache.update(xk, xv)
            start = sw_cache.cum_len - s          # 本步第一个 query 的绝对位置
        else:
            k_base, v_base = xk, xv
            start = 0

        comp_kv_feat = self.comp_kv(x)
        comp_gate_raw = self.comp_gate(x)
        repeat_num = (s + self.compress_rate - 1) // self.compress_rate
        bias_full = self.comp_pos_bias.repeat(repeat_num, 1)[:s]
        comp_gate_feat = comp_gate_raw + bias_full
        usable = (s // self.compress_rate) * self.compress_rate
        chunk_kv = comp_kv_feat[:, :usable, :].view(b, -1, self.compress_rate, 2 * self.head_dim)
        chunk_gate = comp_gate_feat[:, :usable, :].view(b, -1, self.compress_rate, 2 * self.head_dim)
        compressed = self._compress_chunk(chunk_kv, chunk_gate, sw_cache)
        # 增量解码时把历史压缩 token 拼回来（预填充直接用自己的压缩结果，保留梯度）
        if use_cache and sw_cache is not None:
            cached = sw_cache.compressed_kv
            if cached is None:
                all_comp = compressed
            else:
                all_comp = torch.cat([cached, compressed], dim=1) if compressed.size(1) else cached
            sw_cache.compressed_kv = all_comp.detach()
        else:
            all_comp = compressed
        comp_expand = all_comp.expand(-1, -1, self.n_local_kv_heads, -1)
        k_full = torch.cat([k_base, comp_expand], dim=1)
        v_full = torch.cat([v_base, comp_expand], dim=1)

        q_idx = self.index_q(x).view(b, s, 1, self.head_dim)
        forbid, chunk_bias = self._chunk_causal_bias(all_comp.size(1), s, start, x.device)
        index_mask = self._index_mask(q_idx, all_comp, x, forbid) + chunk_bias    # [b, s, n_comp] 加性
        base_bias = self._base_mask_bias(x, attention_mask, k_base.size(1), s)    # [b, s, k_base] 加性
        full_mask = torch.cat([base_bias, index_mask], dim=-1)                    # [b, s, kv_len]

        past_kv = sw_cache if use_cache else None
        out = self._attn_calc(xq, k_full, v_full, full_mask, s, causal=False)
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
        if use_cache and sw_cache is not None:
            k_base, v_base = sw_cache.update(xk, xv)
            start = sw_cache.cum_len - s
        else:
            k_base, v_base = xk, xv
            start = 0

        comp_kv_feat = self.comp_kv(x)
        comp_gate_raw = self.comp_gate(x)
        repeat_num = (s + self.compress_rate - 1) // self.compress_rate
        bias_full = self.comp_pos_bias.repeat(repeat_num, 1)[:s]
        comp_gate_feat = comp_gate_raw + bias_full
        usable = (s // self.compress_rate) * self.compress_rate
        chunk_kv = comp_kv_feat[:, :usable, :].view(b, -1, self.compress_rate, self.head_dim)
        chunk_gate = comp_gate_feat[:, :usable, :].view(b, -1, self.compress_rate, self.head_dim)
        compressed = self._compress_chunk(chunk_kv, chunk_gate)
        if use_cache and sw_cache is not None:      # 压缩上下文跨步持久化（与 CSA 一致）
            cached = sw_cache.compressed_kv
            if cached is None:
                all_comp = compressed
            else:
                all_comp = torch.cat([cached, compressed], dim=1) if compressed.size(1) else cached
            sw_cache.compressed_kv = all_comp.detach()
        else:
            all_comp = compressed
        comp_expand = all_comp.expand(-1, -1, self.n_local_kv_heads, -1)
        k_full = torch.cat([k_base, comp_expand], dim=1)
        v_full = torch.cat([v_base, comp_expand], dim=1)

        full_mask = self._base_mask_bias(x, attention_mask, k_base.size(1), s)
        if all_comp.size(1):
            _, chunk_bias = self._chunk_causal_bias(all_comp.size(1), s, start, x.device)
            full_mask = torch.cat([full_mask, chunk_bias.unsqueeze(0).expand(b, -1, -1)], dim=-1)

        past_kv = sw_cache if use_cache else None
        out = self._attn_calc(xq, k_full, v_full, full_mask, s, causal=False)
        return out, past_kv

class MLAttention(BaseAttention):
    """MLA（DeepSeek-V2 风格 Multi-head Latent Attention）。

    KV 先压到低秩隐向量再投影出 K/V，并额外带一路解耦 RoPE（只作用于 rope 分量）。
    训练（prefill）成本与 GQA 相当或略高；收益主要体现在推理时 KV cache 可只存隐向量。
    注：本实现的 cache 里存的是投影后的 K/V（功能正确但不省显存），真正的 latent cache 待接。
    """
    def __init__(self, config):
        super().__init__(config)
        self.kv_lora = getattr(config, 'mla_kv_lora_rank', 256)
        self.rope_dim = min(getattr(config, 'mla_rope_dim', 32), self.head_dim // 2)
        self.nope_dim = self.head_dim - self.rope_dim
        h = config.hidden_size
        self.q_nope = nn.Linear(h, self.n_local_heads * self.nope_dim, bias=False)
        self.q_rope = nn.Linear(h, self.n_local_heads * self.rope_dim, bias=False)
        self.kv_down = nn.Linear(h, self.kv_lora + self.rope_dim, bias=False)
        self.kv_up = nn.Linear(self.kv_lora, self.n_local_kv_heads * (self.nope_dim + self.head_dim), bias=False)
        # 解耦 RoPE 用自己的一套频率表（维度为 rope_dim），按绝对位置切片
        rc, rs = precompute_freqs_cis(dim=self.rope_dim, end=max(config.max_positions, 4096),
                                      rope_base=config.rope_theta, rope_scaling=None)
        self.register_buffer('rope_cos', rc, persistent=False)
        self.register_buffer('rope_sin', rs, persistent=False)

    def _rot(self, t, cos, sin):
        """t: [b,s,h,d] 或 [b,s,d]；cos/sin: [s,d]"""
        if t.dim() == 4:
            cos, sin = cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)
        else:
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        half = t.shape[-1] // 2
        rot = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
        return (t * cos + rot * sin).to(t.dtype)

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        b, s, _ = x.shape
        start = past_key_value.cum_len if past_key_value is not None else 0   # 绝对位置（cache 维护）
        cos, sin = self.rope_cos[start:start + s], self.rope_sin[start:start + s]
        q = torch.cat([self.q_nope(x).view(b, s, self.n_local_heads, self.nope_dim),
                       self._rot(self.q_rope(x).view(b, s, self.n_local_heads, self.rope_dim), cos, sin)], dim=-1)
        c = self.kv_down(x)
        c_kv, k_rope = c[..., :self.kv_lora], c[..., self.kv_lora:]
        k_rope = self._rot(k_rope, cos, sin)                       # MQA：所有 KV 头共享
        kv = self.kv_up(c_kv).view(b, s, self.n_local_kv_heads, self.nope_dim + self.head_dim)
        k_nope, v = kv[..., :self.nope_dim], kv[..., self.nope_dim:]
        k = torch.cat([k_nope, k_rope.unsqueeze(2).expand(-1, -1, self.n_local_kv_heads, -1)], dim=-1)
        sw_cache: SimpleSWCache = past_key_value
        if use_cache and sw_cache is not None:
            k, v = sw_cache.update(k, v)
        out = self._attn_calc(q, k, v, attention_mask, s)
        return out, (sw_cache if use_cache else None)


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
    """MoE FFN（可切换负载均衡方式与前向实现，默认保持旧行为）。

    balance_mode:
      'aux'      —— 旧行为：load-balancing 辅助损失加进主 loss（router_aux_loss_coef）
      'aux_free' —— DeepSeek-V3 式无辅助损失负载均衡：每个专家一个 bias，只参与 top-k 选择，
                    按负载误差做 sign 更新，不往主 loss 里注入梯度
    moe_impl:
      'loop'     —— 旧行为：逐专家 mask 扫描
      'permute'  —— 先按专家排序再连续切片，去掉逐专家的全量扫描
    n_shared_experts: 常驻共享专家数量（0=关闭），对所有 token 生效
    """
    def __init__(self, config: skyRopeConfig, intermediate_size: int = None):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.balance_mode = getattr(config, 'balance_mode', 'aux')
        self.impl = getattr(config, 'moe_impl', 'loop')
        self.num_shared = getattr(config, 'n_shared_experts', 0)
        intermediate_size = intermediate_size or config.moe_intermediate_size

        def make_expert():
            return nn.Sequential(
                nn.Linear(config.hidden_size, intermediate_size, bias=False),
                ACT2FN[config.hidden_act],
                nn.Linear(intermediate_size, config.hidden_size, bias=False)
            )

        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([make_expert() for _ in range(self.num_experts)])
        self.shared_experts = nn.ModuleList([make_expert() for _ in range(self.num_shared)])
        if self.balance_mode == 'aux_free':
            self.register_buffer('expert_bias', torch.zeros(self.num_experts), persistent=False)
        # 统计用：累计每个专家被选中的次数（不参与 state_dict）
        self.register_buffer('expert_load', torch.zeros(self.num_experts), persistent=False)
        self.aux_loss = torch.tensor(0.)

    def _dispatch(self, x_flat, topk_idx, topk_weight):
        y = torch.zeros_like(x_flat)
        if self.impl == 'loop':
            for i, expert in enumerate(self.experts):
                mask = (topk_idx == i)
                if mask.any():
                    token_idx = torch.where(mask.any(dim=-1))[0]
                    weight = topk_weight[mask].unsqueeze(-1).to(x_flat.dtype)
                    y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)
            return y
        # permute：按专家排序后连续切片，避免逐专家 O(T*k) 的掩码扫描
        k = topk_idx.size(-1)
        flat_expert = topk_idx.reshape(-1)
        order = torch.argsort(flat_expert, stable=True)
        token_ids = torch.div(order, k, rounding_mode='floor')
        counts = torch.bincount(flat_expert, minlength=self.num_experts)
        flat_w = topk_weight.reshape(-1)
        start = 0
        for i, expert in enumerate(self.experts):
            n = int(counts[i].item())
            if n == 0:
                continue
            ids = token_ids[start:start + n]
            w = flat_w[order[start:start + n]].unsqueeze(-1).to(x_flat.dtype)
            y.index_add_(0, ids, expert(x_flat[ids]) * w)
            start += n
        return y

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        logits = self.gate(x_flat)
        scores = F.softmax(logits, dim=-1)
        # aux-free 的 bias 只影响"选谁"，不影响门控权重
        select_scores = logits + self.expert_bias if self.balance_mode == 'aux_free' else logits
        _, topk_idx = torch.topk(select_scores, k=self.num_experts_per_tok, dim=-1, sorted=False)
        topk_weight = scores.gather(-1, topk_idx)
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = self._dispatch(x_flat, topk_idx, topk_weight)
        for shared in self.shared_experts:
            y = y + shared(x_flat)
        if self.training:
            with torch.no_grad():
                self.expert_load += F.one_hot(topk_idx, self.num_experts).float().sum(dim=(0, 1))
            if self.balance_mode == 'aux' and self.config.router_aux_loss_coef > 0:
                load = F.one_hot(topk_idx, self.num_experts).float().mean((0, 1))
                self.aux_loss = (load * scores.mean((0, 1))).sum() * self.config.num_experts * self.config.router_aux_loss_coef
            else:
                self.aux_loss = scores.new_zeros(1).squeeze()
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

    @torch.no_grad()
    def update_router_bias(self, gamma: float = None):
        """aux-free 均衡：被超选的专家降 bias，欠载的升 bias（每步调用一次）"""
        if self.balance_mode != 'aux_free':
            return
        total = float(self.expert_load.sum())
        if total <= 0:
            return
        share = self.expert_load / total              # 归一化被选比例，总和为 1
        target = 1.0 / self.num_experts
        gamma = getattr(self.config, 'aux_free_gamma', 1e-3) if gamma is None else gamma
        self.expert_bias += gamma * torch.sign(target - share)
        self.expert_load.zero_()

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
        elif attn_type == "mla":
            self.attention = MLAttention(config)
        else:
            raise ValueError("attn_type only support swa / csa / hca / mla")
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
        attn_type_list = config.attn_type_list or ["csa","csa","swa","swa","swa","swa","csa","csa"]
        self.layers = nn.ModuleList([
            skyRopeBlock(l, config, attn_type=attn_type_list[l % len(attn_type_list)])
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
        window = self.config.max_positions // 4
        past_key_values = [c if isinstance(c, SimpleSWCache) else SimpleSWCache(window)
                           for c in (past_key_values or [None] * len(self.layers))]
        start_pos = past_key_values[0].cum_len
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

class MTPModule(nn.Module):
    """多 token 预测（DeepSeek-V3 式）：concat(RMSNorm(h_t), Embed(x_{t+k})) → 一层 block → 共享 head 预测 x_{t+k+1}"""
    def __init__(self, config: skyRopeConfig, layer_id: int):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.block = skyRopeBlock(layer_id, config, attn_type='swa')
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, h, next_emb, position_embeddings):
        x = self.proj(torch.cat([self.norm(h), next_emb], dim=-1))
        x, _ = self.block(x, position_embeddings, None, False, None)
        return self.final_norm(x)


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
        self.mtp_layers = nn.ModuleList([
            MTPModule(self.config, self.config.num_hidden_layers + i)
            for i in range(self.config.num_nextn_predict_layers)])
        self.post_init()

    def _mtp_loss(self, hidden_states, input_ids, labels):
        """第 k 层 MTP 用 h_t 与 x_{t+k} 预测 x_{t+k+1}（位置对齐见下方切片）"""
        total = hidden_states.new_zeros(1).squeeze()
        emb = self.model.embed_tokens
        s = input_ids.size(1)
        for k, mtp in enumerate(self.mtp_layers, start=1):
            n = s - k - 1
            if n <= 1:
                continue
            h = hidden_states[:, :n]
            next_emb = emb(input_ids[:, k:k + n])
            pos = (self.model.freqs_cos[:n], self.model.freqs_sin[:n])
            logits = self.lm_head(mtp(h, next_emb, pos))
            if self.config.logit_softcap:
                cap = self.config.logit_softcap
                logits = cap * torch.tanh(logits / cap)
            tgt = labels[:, k + 1:k + 1 + n]
            total = total + F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                            tgt.reshape(-1), ignore_index=-100) * self.config.mtp_loss_weight
        return total

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_value, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits= self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.logit_softcap:                      # Gemma2 式 logit 软上限（稳定性）
            cap = self.config.logit_softcap
            logits = cap * torch.tanh(logits / cap)
        loss, mtp_loss = None, None
        if labels is not None:
            x = logits[..., :-1 , :].contiguous()
            y = labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.shape[-1]), y.view(-1), ignore_index=-100)
            if self.config.z_loss_coef:                    # logit 正则，抑制 logit 爆炸
                z = torch.logsumexp(x.float(), dim=-1)
                loss = loss + self.config.z_loss_coef * (z ** 2).mean()
            if len(self.mtp_layers):
                mtp_loss = self._mtp_loss(hidden_states, input_ids, labels)
        out = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_value, hidden_states=hidden_states)
        out.mtp_loss = mtp_loss                            # 单独返回，验证集 loss 仍只算下一 token
        return out

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

































































