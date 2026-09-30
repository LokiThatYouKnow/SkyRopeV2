"""skyRope 模型冒烟测试（重构后回归用）

覆盖:
  1. 前向/反向能跑通（MoE 与非 MoE）
  2. 滑动窗口 KV cache 的增量解码与整段前向数值一致（纯 SWA 层）
  3. 2D padding mask 与无 mask 等价、3D 加性 mask(CSA) 不崩
  4. MoE 真的在训练：aux_loss>0、router/experts 都有梯度
  5. 真实配置单步显存/耗时（估算训练时长）

用法: python scripts/smoke_test.py [--batch_size 16] [--seq_len 340] [--device cuda]
"""
import os
import sys
import copy
import time
import argparse
import warnings

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from model.model_skyRope import skyRopeConfig, skyRopeForCausalLM, SWAAttention, MOEFeedForward

warnings.filterwarnings('ignore')


def tiny_cfg(**kw):
    base = dict(hidden_size=64, num_hidden_layers=8, vocab_size=256,
                max_position_embeddings=1024, num_attention_heads=4, num_key_value_heads=2)
    base.update(kw)
    return skyRopeConfig(**base)


def all_swa(model):
    """把每层注意力都换成 SWA 层（用于验证 mask/cache 的数值正确性）"""
    src = copy.deepcopy(model.model.layers[2])
    for i in range(len(model.model.layers)):
        model.model.layers[i] = copy.deepcopy(src)
    return model


def t1_cache_equivalence():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = all_swa(skyRopeForCausalLM(cfg)).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    with torch.no_grad():
        full = model(ids).logits[:, -1, :]
        out = model(ids[:, :31], past_key_values=None, use_cache=True)
        caches = out.past_key_values
        step = model(ids[:, 31:], past_key_values=caches, use_cache=True).logits[:, -1, :]
    diff = (full - step).abs().max().item()
    print(f'[1] 增量解码 vs 整段前向: max|diff| = {diff:.3e}  {"OK" if diff < 1e-4 else "FAIL"}')
    return diff < 1e-4


def t1b_full_model_decode():
    """各层型下"增量解码"都应与"整段前向"一致（同一位置 logits 相同）。

    注意：CSA/HCA 只压缩"完整的" chunk，所以预填充长度需对齐 chunk 边界
    （否则最后一个不完整 chunk 没有压缩表示，两边 KV 集合本身不同）。
    """
    ok = True
    for name, types in [('8层 CSA+SWA(默认)', None), ('全 CSA', ['csa'] * 8),
                        ('全 SWA', ['swa'] * 8), ('全 HCA', ['hca'] * 8)]:
        torch.manual_seed(0)
        # window = max_positions // 4，必须大于序列长度，否则滑窗截断本来就会让两者不同
        cfg = tiny_cfg(max_position_embeddings=4096, attn_type_list=types)
        model = skyRopeForCausalLM(cfg).eval()
        # 预填充长度对齐到最大压缩率(128)的倍数，保证压缩集合一致
        n_pre = 256
        ids = torch.randint(0, cfg.vocab_size, (1, n_pre + 1))
        with torch.no_grad():
            full = model(ids).logits[:, -1, :]
            pre = model(ids[:, :n_pre], past_key_values=None, use_cache=True)
            assert pre.past_key_values is not None and pre.past_key_values[0] is not None, '前向没有返回 KV cache'
            dec = model(ids[:, n_pre:], past_key_values=pre.past_key_values, use_cache=True).logits[:, -1, :]
        diff = (full - dec).abs().max().item()
        ok &= diff < 1e-3
        print(f'[1b] {name:18s} 增量解码 vs 整段前向: max|diff| = {diff:.3e}  {"OK" if diff < 1e-3 else "FAIL"}')
    return ok


def t1c_moe_impl_equiv():
    """MoE 的 loop / permute 两种调度实现必须数值等价"""
    import copy
    torch.manual_seed(0)
    cfg = tiny_cfg(use_moe=True, num_experts=4, num_experts_per_tok=2)
    a = MOEFeedForward(cfg).eval()
    b = copy.deepcopy(a); b.impl = 'permute'
    h = torch.randn(4, 64, cfg.hidden_size)
    with torch.no_grad():
        d = float((a(h) - b(h)).abs().max())
    print(f'[1c] MoE loop vs permute 输出 max|diff| = {d:.3e}  {"OK" if d < 1e-5 else "FAIL"}')
    return d < 1e-5


def t2_mask_equiv():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = all_swa(skyRopeForCausalLM(cfg)).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 24))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        a = model(ids).logits
        b = model(ids, attention_mask=mask).logits
        m2 = mask.clone(); m2[0, :3] = 0
        c = model(ids, attention_mask=m2).logits
    d1 = (a - b).abs().max().item()
    ok = d1 < 1e-4 and torch.isfinite(c).all().item()
    print(f'[2] padding mask: 全1 mask vs 无 mask max|diff| = {d1:.3e}; 带0 mask 输出有限 = {torch.isfinite(c).all().item()}  {"OK" if ok else "FAIL"}')
    return ok


def t3_moe_grad(device):
    torch.manual_seed(0)
    cfg = tiny_cfg(use_moe=True, num_experts=4, num_experts_per_tok=1)
    model = skyRopeForCausalLM(cfg).to(device).train()
    ids = torch.randint(0, cfg.vocab_size, (4, 32)).to(device)
    res = model(ids, labels=ids)
    loss = res.loss + res.aux_loss
    loss.backward()
    gate = model.model.layers[0].mlp.gate.weight.grad
    exp0 = model.model.layers[0].mlp.experts[0][0].weight.grad
    n_all = sum(1 for l in model.model.layers if isinstance(l.mlp, MOEFeedForward) for _ in l.mlp.experts)
    n_expert_grad = sum(1 for l in model.model.layers if isinstance(l.mlp, MOEFeedForward)
                        for e in l.mlp.experts if e[0].weight.grad is not None and e[0].weight.grad.abs().sum() > 0)
    gsum = 0.0 if gate is None else float(gate.abs().sum())
    ok = float(res.aux_loss) > 0 and gate is not None and gsum > 0 and n_expert_grad > 0
    print(f'[3] MoE: loss={float(res.loss):.4f} aux_loss={float(res.aux_loss):.2e} '
          f'router_grad_sum={gsum:.3e} 有梯度专家={n_expert_grad}/{n_all}  {"OK" if ok else "FAIL"}')
    if n_expert_grad < n_all:
        print(f'    注意: {n_all - n_expert_grad} 个专家本轮没收到 token（DDP 多卡需 find_unused_parameters）')
    return ok


def params(model, cfg):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    if not cfg.use_moe:
        return total, total
    n_routed, n_active = cfg.num_experts, cfg.num_experts_per_tok
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    base = total - expert * n_routed
    return total, base + expert * n_active


def t4_real_step(device, batch_size, seq_len, layers, use_moe):
    torch.manual_seed(0)
    cfg = skyRopeConfig(hidden_size=768, num_hidden_layers=layers, vocab_size=6400,
                        max_position_embeddings=seq_len, use_moe=use_moe,
                        num_experts=4, num_experts_per_tok=1)
    model = skyRopeForCausalLM(cfg).to(device).train()
    total, active = params(model, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    ids = torch.randint(0, 6400, (batch_size, seq_len), device=device)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    t0 = time.perf_counter()
    with ctx:
        res = model(ids, labels=ids)
        loss = res.loss + res.aux_loss
    loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    tok = batch_size * seq_len
    print(f'[4] 真实配置 layers={layers} moe={use_moe} bs={batch_size} seq={seq_len}: '
          f'params {total:.1f}M(激活 {active:.1f}M)  单步 {dt*1000:.0f}ms  {tok/dt:,.0f} tok/s  '
          f'峰值显存 {peak:.1f} GiB  {"OK" if peak < 11 else "OOM风险"}')
    del model, opt, ids, res, loss
    torch.cuda.empty_cache()
    return dt, peak, tok


def t5_generate(device):
    cfg = tiny_cfg()
    model = all_swa(skyRopeForCausalLM(cfg)).eval().to(device)
    ids = torch.randint(1, cfg.vocab_size, (1, 8), device=device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=8, eos_token_id=None)
    ok = out.shape == (1, 16)
    print(f'[5] generate: {tuple(ids.shape)} -> {tuple(out.shape)}  {"OK" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--seq_len', type=int, default=340)
    p.add_argument('--layers', type=int, default=8)
    args = p.parse_args()
    device = args.device
    print(f'device={device} torch={torch.__version__}')
    results = [t1_cache_equivalence(), t1b_full_model_decode(), t1c_moe_impl_equiv(),
               t2_mask_equiv(), t3_moe_grad(device), t5_generate(device)]
    t4_real_step(device, args.batch_size, args.seq_len, args.layers, True)
    t4_real_step(device, args.batch_size, args.seq_len, args.layers, False)
    print('\n=== 结果:', 'ALL PASS' if all(results) else 'HAS FAILURE', '===')
