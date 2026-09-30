"""A/B 基准：在固定 token 预算下对比优化器 / 数据格式 / MoE 组件。

用法:
  python scripts/ab_bench.py --suite speed                 # 各配置 100 步，比吞吐/显存/稳定性
  python scripts/ab_bench.py --suite quality --tokens 2000000
  python scripts/ab_bench.py --configs base,packed,muon

输出: 表格 + scripts/ab_results.json
"""
import os
import sys
import json
import time
import argparse
import warnings

# Windows + 非 ASCII 路径/环境变量会让 triton 读环境时报 UnicodeDecodeError，这里先净化
for _k in list(os.environ):
    try:
        os.environ[_k].encode('utf-8')
    except Exception:                                        # noqa: BLE001
        os.environ.pop(_k, None)
for _k in ('TEMP', 'TMP', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR'):
    _v = os.environ.get(_k, '')
    if any(ord(c) > 127 for c in _v):
        os.environ[_k] = r'C:\tmp\skyrope'
os.makedirs(os.environ.get('TORCHINDUCTOR_CACHE_DIR', r'C:\tmp\skyrope'), exist_ok=True)

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from model.model_skyRope import skyRopeConfig, skyRopeForCausalLM, MOEFeedForward
from trainer.optim_utils import build_optimizer, build_lr_lambda, set_lr_mult
from dataset.lm_dataset import PackedPretrainDataset

warnings.filterwarnings('ignore')

BASE = dict(hidden_size=768, num_hidden_layers=8, use_moe=True, num_experts=4,
            num_experts_per_tok=1, vocab_size=6400)

# name -> 组件开关
CONFIGS = {
    'base':       dict(opt='adamw',       sched='cosine', data='pad', impl='loop',    balance='aux'),
    'fused':      dict(opt='adamw_fused', sched='cosine', data='pad', impl='loop',    balance='aux'),
    'packed':     dict(opt='adamw',       sched='cosine', data='packed', impl='loop',  balance='aux'),
    'permute':    dict(opt='adamw',       sched='cosine', data='pad', impl='permute', balance='aux'),
    'pd_muon':    dict(opt='muon_batched', sched='cosine', data='pad', impl='permute', balance='aux'),
    'muon':       dict(opt='muon',        sched='cosine', data='pad', impl='loop',    balance='aux'),
    'wsd':        dict(opt='adamw',       sched='wsd',    data='pad', impl='loop',    balance='aux'),
    'auxfree':    dict(opt='adamw',       sched='cosine', data='pad', impl='loop',    balance='aux_free'),
    'wd0.1':      dict(opt='adamw',       sched='cosine', data='pad', impl='loop',    balance='aux', weight_decay=0.1),
    'top2_half':  dict(opt='adamw',       sched='cosine', data='pad', impl='loop',    balance='aux',
                       num_experts_per_tok=2, moe_intermediate_size=1216),
    'shared':     dict(opt='adamw',       sched='cosine', data='pad', impl='loop',    balance='aux',
                       n_shared_experts=1, moe_intermediate_size=1300),
    'compile':    dict(opt='adamw',       sched='cosine', data='pad', impl='permute', balance='aux', compile=True),
    'swa_only':   dict(opt='adamw',       sched='cosine', data='pad', impl='loop',    balance='aux', attn_type_list=['swa'] * 8),
    'csa_only':   dict(opt='adamw',       sched='cosine', data='pad', impl='loop',    balance='aux', attn_type_list=['csa'] * 8),
    'base_s7':    dict(opt='adamw',       sched='cosine', data='pad', impl='loop',    balance='aux', seed=7),
    # 等墙钟时间对比（LR 恒定，避免不同 step 数导致调度不可比）
    'time_base':  dict(opt='adamw',       sched='const',  data='pad', impl='loop',    balance='aux'),
    'time_packed': dict(opt='adamw',      sched='const',  data='packed', impl='loop', balance='aux'),
    'time_muon':  dict(opt='muon_batched', sched='const', data='pad', impl='loop', balance='aux'),
    'time_muon_slow': dict(opt='muon',    sched='const',  data='pad', impl='permute', balance='aux'),
    'muon_b':     dict(opt='muon_batched', sched='cosine', data='pad', impl='loop',   balance='aux'),
}
STACK = dict(opt='muon_batched', sched='cosine', data='packed', impl='permute', balance='aux',
             num_experts_per_tok=2, moe_intermediate_size=1216)
CONFIGS.update({
    'stack':        dict(STACK),
    'stack_mtp':    dict(STACK, num_nextn_predict_layers=1),
    'stack_stab':   dict(STACK, z_loss_coef=1e-4, logit_softcap=30.0),
    'stack_ns3':    dict(STACK, ns_steps=3),
    'stack_mla':    dict(STACK, attn_type_list=['mla'] * 8),
    'stack_deep':   dict(STACK, hidden_size=512, num_hidden_layers=16),
    'stack_8x':     dict(STACK, num_experts=8, num_experts_per_tok=3, moe_intermediate_size=810),
    'stack_shared': dict(STACK, num_experts_per_tok=1, moe_intermediate_size=1216, n_shared_experts=1),
    'stack24k':     dict(STACK, vocab_size=24000),
})
TIME_CONFIGS = {
    # 组合候选（等墙钟时间对比，恒定 LR）
    'time_packed_muon': dict(opt='muon_batched', sched='const', data='packed', impl='permute', balance='aux'),
    'time_packed_top2': dict(opt='adamw',        sched='const', data='packed', impl='permute', balance='aux',
                             num_experts_per_tok=2, moe_intermediate_size=1216),
    'time_stack':       dict(opt='muon_batched', sched='const', data='packed', impl='permute', balance='aux',
                             num_experts_per_tok=2, moe_intermediate_size=1216),
}

CONFIGS.update(TIME_CONFIGS)

ROUND3_SUITE = ['stack', 'stack_mtp', 'stack_stab', 'stack_ns3', 'stack_mla', 'stack_deep',
                'stack_8x', 'stack_shared']
SPEED_SUITE = list(CONFIGS.keys())
QUALITY_SUITE = ['base', 'base_s7', 'packed', 'muon', 'auxfree', 'top2_half']


def load_texts(path, start, count):
    out = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if len(out) >= count:
                break
            line = line.strip()
            if line:
                out.append(json.loads(line)['text'])
    return out


def tokenize_pad(texts, tok, seq_len):
    ids_all, lab_all = [], []
    for t in texts:
        ids = tok(t, add_special_tokens=False, max_length=seq_len - 2, truncation=True).input_ids
        ids = [tok.bos_token_id] + ids + [tok.eos_token_id]
        ids = ids + [tok.pad_token_id] * (seq_len - len(ids))
        lab = [i if i != tok.pad_token_id else -100 for i in ids]
        ids_all.append(ids)
        lab_all.append(lab)
    return (torch.tensor(ids_all, dtype=torch.long), torch.tensor(lab_all, dtype=torch.long))


def build_data(args, tok):
    t0 = time.perf_counter()
    train_texts = load_texts(args.data, 0, args.train_docs)
    val_texts = load_texts(args.data, args.train_docs, args.val_docs)
    data = {'val': tokenize_pad(val_texts, tok, args.seq_len)}
    data['pad'] = tokenize_pad(train_texts, tok, args.seq_len)
    packed = PackedPretrainDataset(tokenizer=tok, max_length=args.seq_len, texts=train_texts)
    p_ids = torch.tensor(packed.bins, dtype=torch.long)
    p_lab = p_ids.clone()
    p_lab[p_ids == tok.pad_token_id] = -100
    data['packed'] = (p_ids, p_lab)
    print(f'数据就绪 ({time.perf_counter()-t0:.0f}s): pad={tuple(data["pad"][0].shape)} '
          f'packed={tuple(data["packed"][0].shape)} val={tuple(data["val"][0].shape)}')
    return data


def real_tokens(labels):
    return int((labels != -100).sum())


@torch.no_grad()
def evaluate(model, val, device, bs=8):
    model.eval()
    ids, lab = val
    total, n = 0.0, 0
    for i in range(0, min(len(ids), bs * 24), bs):
        x = ids[i:i + bs].to(device)
        y = lab[i:i + bs].to(device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = model(x, labels=y)
        total += float(out.loss) * real_tokens(y)
        n += real_tokens(y)
    model.train()
    return total / max(n, 1)


def run_config(name, spec, args, data, tok, device):
    over = dict(BASE)
    over['max_position_embeddings'] = args.seq_len
    over['use_moe'] = True
    for k in ('num_experts', 'num_experts_per_tok', 'moe_intermediate_size', 'n_shared_experts',
              'attn_type_list', 'balance_mode', 'moe_impl', 'hidden_size', 'num_hidden_layers',
              'num_nextn_predict_layers', 'mtp_loss_weight', 'z_loss_coef', 'logit_softcap',
              'mla_kv_lora_rank', 'mla_rope_dim'):
        if k in spec:
            over[k] = spec[k]
    over['balance_mode'] = spec.get('balance', 'aux')
    over['moe_impl'] = spec.get('impl', 'loop')
    cfg = skyRopeConfig(**over)

    seed = spec.get('seed', args.seed)
    torch.manual_seed(seed)
    model = skyRopeForCausalLM(cfg).to(device).train()
    opt, opt_desc = build_optimizer(model, kind=spec.get('opt', 'adamw'), lr=args.lr,
                                    muon_lr=spec.get('muon_lr', 0.02),
                                    weight_decay=spec.get('weight_decay', 0.0),
                                    ns_steps=spec.get('ns_steps', 5))
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_active = n_total
    if cfg.use_moe:
        expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
        n_active = n_total - expert * cfg.num_experts + expert * cfg.num_experts_per_tok

    if spec.get('compile'):
        try:
            model = torch.compile(model)
        except Exception as e:                                  # noqa: BLE001
            print(f'  [compile 失败] {e}')
            return dict(name=name, error=f'compile: {e}')

    ids, lab = data[spec.get('data', 'pad')]
    steps_cap = args.steps
    lr_fn = build_lr_lambda(spec.get('sched', 'cosine'),
                            warmup_steps=int(args.steps * 0.05) if spec.get('sched') in ('wsd', 'warmup_cosine') else 0,
                            total_steps=steps_cap)
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(1234 + seed))
    bs = args.batch_size
    n_moe = [m for m in model.modules() if isinstance(m, MOEFeedForward)]
    load_total = torch.zeros(cfg.num_experts)
    losses, gnorms, tokens, n_steps, nan = [], [], 0, 0, 0
    first_step_delta = None
    before = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    ptr = 0
    while n_steps < steps_cap:
        if ptr + bs > len(order):
            ptr = 0
        idx = order[ptr:ptr + bs]
        ptr += bs
        x = ids[idx].to(device, non_blocking=True)
        y = lab[idx].to(device, non_blocking=True)
        set_lr_mult(opt, lr_fn(n_steps))
        if first_step_delta is None:
            before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            res = model(x, labels=y)
            loss = res.loss + (res.aux_loss if cfg.use_moe else 0) + (getattr(res, 'mtp_loss', None) or 0)
        if not torch.isfinite(loss):
            nan += 1
            break
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        gnorms.append(float(gn))
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for m in n_moe:                       # 先统计本步负载
                load_total += m.expert_load.cpu()
                m.expert_load.zero_()
        if cfg.balance_mode == 'aux_free':        # 再按负载误差调整选择 bias
            for m in n_moe:
                m.update_router_bias()
        if first_step_delta is None:
            first_step_delta = measure_updates(model, before, spec.get('opt', 'adamw'))
            before = None
        losses.append(float(loss))
        tokens += real_tokens(y)
        n_steps += 1
        if args.tokens and tokens >= args.tokens:
            break
        if args.wall_seconds and (time.perf_counter() - t0) >= args.wall_seconds:
            break
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    val_loss = evaluate(model, data['val'], device) if not args.no_eval else float('nan')
    share = load_total / max(float(load_total.sum()), 1e-9)
    res = dict(name=name, seed=seed, steps=n_steps, real_tokens=tokens, wall_s=wall,
               tok_per_s=tokens / wall, ms_per_step=wall / max(n_steps, 1) * 1000,
               peak_gib=torch.cuda.max_memory_allocated() / 2**30,
               params_m=n_total, active_m=n_active,
               train_loss=sum(losses[-max(1, len(losses) // 5):]) / max(1, len(losses[-max(1, len(losses) // 5):])),
               val_loss=val_loss, grad_norm=sum(gnorms) / max(len(gnorms), 1), nan=nan,
               moe_load_max=float(share.max()), moe_load_min=float(share.min()),
               moe_imbalance=float(share.max() / max(float(share.min()), 1e-9)),
               update_rel=first_step_delta)
    print(f'  {name:11s} steps={n_steps:4d} tok/s={res["tok_per_s"]:9,.0f} {res["ms_per_step"]:6.0f}ms/step '
          f'peak={res["peak_gib"]:.1f}G val={val_loss:.4f} train={res["train_loss"]:.4f} '
          f'gnorm={res["grad_norm"]:.3f} moe不平衡={res["moe_imbalance"]:.2f} NaN={nan}')
    if first_step_delta:
        print(f'              首步相对更新量: {first_step_delta}')
    del model, opt
    torch.cuda.empty_cache()
    return res


def measure_updates(model, before, kind):
    """首步 ||ΔW||/||W|| 分组成对：检查 Muon/AdamW 的更新尺度是否合理"""
    out = {}
    for n, p in model.named_parameters():
        if n not in before:
            continue
        base_norm = float(before[n].float().norm())
        if base_norm < 1e-6:            # 零初始化参数(如 comp_pos_bias)相对量无意义
            continue
        d = (p.detach() - before[n]).float()
        rel = float(d.norm() / base_norm)
        if 'embed_tokens' in n or 'lm_head' in n:
            g = 'embed/head'
        elif '.gate.' in n:
            g = 'router'
        elif p.ndim == 2 and min(p.shape) >= 32:
            g = 'hidden2d(Muon组)'
        else:
            g = '1d/thin'
        out.setdefault(g, []).append(rel)
    return {k: round(sum(v) / len(v), 6) for k, v in out.items()}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--suite', default='speed', choices=['speed', 'quality'])
    p.add_argument('--repeats', type=int, default=1, help='交错重复次数（笔记本 GPU 会热降频，用中位数比较）')
    p.add_argument('--seed', type=int, default=42, help='模型初始化 + 数据顺序种子（用于估计噪声）')
    p.add_argument('--no_eval', action='store_true', help='跳过验证集（纯测速）')
    p.add_argument('--wall_seconds', type=int, default=0, help='墙钟预算(秒)：等时间对比用（配合 sched=const）')
    p.add_argument('--configs', default='')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--tokens', type=int, default=0, help='真实 token 预算，达到即停（quality 套件用）')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--seq_len', type=int, default=340)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--train_docs', type=int, default=20000)
    p.add_argument('--val_docs', type=int, default=512)
    p.add_argument('--data', default='dataset/pretrain_t2t_mini.jsonl')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.suite == 'quality' and args.tokens == 0:
        args.tokens = 2_000_000
        args.steps = 4000
    names = [c for c in (args.configs.split(',') if args.configs else
                         (SPEED_SUITE if args.suite == 'speed' else QUALITY_SUITE)) if c]
    tok = AutoTokenizer.from_pretrained('model')
    data = build_data(args, tok)
    results, per_config = [], {n: [] for n in names}
    for rep in range(args.repeats):
        if args.repeats > 1:
            print(f'--- 第 {rep+1}/{args.repeats} 轮 ---')
        for name in names:
            print(f'=== {name} ===')
            try:
                r = run_config(name, CONFIGS[name], args, data, tok, args.device)
                results.append(r)
                if 'error' not in r:
                    per_config[name].append(r)
            except Exception as e:                               # noqa: BLE001
                import traceback; traceback.print_exc()
                r = dict(name=name, error=str(e)[:200])
                results.append(r)
    if args.repeats > 1:
        import statistics as _st
        print('\n=== 交错重复的中位数（抗热降频）===')
        for n in names:
            rs = per_config[n]
            if not rs:
                continue
            med = lambda k: _st.median([x[k] for x in rs])
            print(f'{n:11s} ms/step={med("ms_per_step"):7.1f}  tok/s={med("tok_per_s"):9,.0f}  '
                  f'peak={med("peak_gib"):.1f}G  (n={len(rs)})')
    with open('scripts/ab_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print('\n=== 汇总 ===')
    for r in results:
        if 'error' in r:
            print(f'{r["name"]:11s} ERROR: {r["error"]}')
        else:
            print(f'{r["name"]:11s} tok/s={r["tok_per_s"]:9,.0f} peak={r["peak_gib"]:.1f}G '
                  f'val={r["val_loss"]:.4f} moe不平衡={r["moe_imbalance"]:.2f} 参数={r["params_m"]:.1f}M')
