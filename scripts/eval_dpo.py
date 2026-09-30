"""DPO 留出集评估：偏好准确率 = P(margin > 0)，margin = β[(logp_policy−logp_ref)(chosen) − (…)(rejected)]

用法:
  set SKYROPE_* 对齐环境变量后:
  python scripts/eval_dpo.py --policy out/dpo_768_moe.pth --ref out/full_sft_768_moe.pth --n 500
  python scripts/eval_dpo.py --policy out/full_sft_768_moe.pth --ref out/full_sft_768_moe.pth --n 500   # 基线应≈50%
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model.model_skyRope import skyRopeConfig
from model.model_vlm import VLMConfig
from trainer.train_dpo import DPODataset, compute_log_probs


def build(ckpt, device):
    cfg = skyRopeConfig(use_moe=True, max_position_embeddings=1024)
    from model.model_skyRope import skyRopeForCausalLM
    m = skyRopeForCausalLM(cfg)
    sd = torch.load(ckpt, map_location='cpu')
    r = m.load_state_dict(sd, strict=False)
    if r.missing_keys or r.unexpected_keys:
        print(f'  [警告] {ckpt} 缺失 {len(r.missing_keys)} / 多余 {len(r.unexpected_keys)} 键')
    return m.half().eval().to(device)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--policy', required=True)
    p.add_argument('--ref', required=True)
    p.add_argument('--data', default='dataset/dpo_holdout.jsonl')
    p.add_argument('--n', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--max_seq_len', type=int, default=1024)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()

    tok_path = os.environ.get('SKYROPE_TOKENIZER', 'model_tok24k')
    if not os.path.isdir(tok_path):        # 环境变量给的是相对 trainer/ 的路径，这里按项目根解析
        alt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.basename(tok_path))
        if os.path.isdir(alt):
            tok_path = alt
    print(f'分词器: {tok_path}')
    tok = AutoTokenizer.from_pretrained(tok_path)
    ds = DPODataset(a.data, tok, max_length=a.max_seq_len)
    ds.samples = ds.samples.select(range(min(a.n, len(ds.samples)))) if hasattr(ds.samples, 'select') else ds.samples
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0)
    print(f'留出集 {len(ds)} 对 | policy={os.path.basename(a.policy)} ref={os.path.basename(a.ref)}')
    policy, ref = build(a.policy, a.device), build(a.ref, a.device)

    wins = tot = 0
    margins = []
    direct = []          # 直接偏好：policy 自己是否给 chosen 更高 logp（随机水平=50%）
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for batch in dl:
            cid = batch['chosen_input_ids'].to(a.device); cl = batch['chosen_labels'].to(a.device)
            rid = batch['rejected_input_ids'].to(a.device); rl = batch['rejected_labels'].to(a.device)
            pc = compute_log_probs(policy, cid, cl).float(); pr = compute_log_probs(policy, rid, rl).float()
            rc = compute_log_probs(ref, cid, cl).float(); rr = compute_log_probs(ref, rid, rl).float()
            margin = a.beta * ((pc - rc) - (pr - rr))
            wins += int((margin > 0).sum()); tot += margin.numel()
            margins.append(margin.cpu())
            direct.append((pc - pr).cpu())
    allm = torch.cat(margins)
    alld = torch.cat(direct)
    print(f'DPO 隐式奖励 margin 准确率 = {wins/tot*100:.2f}%  ({wins}/{tot})')
    print(f'margin: 均值 {allm.mean():+.4f}  中位 {allm.median():+.4f}')
    print(f'直接偏好准确率(>50% 即学到偏好) = {(alld>0).float().mean()*100:.2f}%')
    print(f'直接 logp 差: 均值 {alld.mean():+.4f}  (chosen 比 rejected 高多少)')
