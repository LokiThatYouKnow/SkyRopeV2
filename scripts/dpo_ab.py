"""DPO 超参 A/B + 留出集偏好评估（自动串行跑完并汇总）。

流程: 基线评估(full_sft vs full_sft，应≈50%) → 各 lr 短跑(3000 对=1500 步) → 各自留出集评估 → 汇总
用法: python scripts/dpo_ab.py    （需已设置 SKYROPE_* 对齐环境变量）
"""
import datetime
import json
import os
import re
import subprocess
import sys
import time

ROOT = r'D:\SkyRope\SkyRopeV2'
TRAINER = os.path.join(ROOT, 'trainer')
LOG = os.path.join(ROOT, 'out', 'dpo_ab.log')
env = dict(os.environ)
env.update(SKYROPE_TOKENIZER='../model_tok24k', SKYROPE_VOCAB='24000', SKYROPE_NUM_EXPERTS='4',
           SKYROPE_TOP_K='2', SKYROPE_MOE_INTERMEDIATE='1216', SKYROPE_MOE_IMPL='permute',
           PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONUNBUFFERED='1')


def log(m):
    line = datetime.datetime.now().strftime('%H:%M:%S') + '  ' + str(m)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + chr(10))


def eval_dpo(policy, ref, n=300, bs=4, tag=''):
    cmd = [sys.executable, '-X', 'utf8', 'scripts/eval_dpo.py', '--policy', policy, '--ref', ref,
           '--n', str(n), '--batch_size', str(bs), '--beta', '0.1']
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    out = (r.stdout or '') + (r.stderr or '')
    acc = None
    m = re.search(r'margin 准确率 = ([0-9.]+)%', out)
    if m:
        acc = float(m.group(1))
    log(f'  [eval {tag}] acc={acc}  ' + ' | '.join(l.strip() for l in out.splitlines()[-3:]))
    return acc


results = []
log('=== 基线评估：SFT 模型 作为 policy 与 ref（理论应≈50%） ===')
base_acc = eval_dpo('out/full_sft_768_moe.pth', 'out/full_sft_768_moe.pth', tag='baseline')
results.append(dict(name='baseline(sft)', lr=0, beta=0.1, acc=base_acc))

for lr, beta in [(1e-5, 0.1), (5e-5, 0.1), (2e-4, 0.1)]:
    name = f'lr{lr:g}_b{beta:g}'
    w = f'dpo_ab_{name}'
    log(f'=== 训练 {name} (3000 对 / 1500 步) ===')
    args = [sys.executable, '-X', 'utf8', '-u', 'train_dpo.py', '--model_type', 'llm', '--use_moe', '1',
            '--data_path', '../dataset/dpo_ab_train.jsonl', '--from_weight', 'full_sft', '--from_resume', '0',
            '--epochs', '1', '--batch_size', '2', '--accumulation_steps', '4', '--max_seq_len', '1024',
            '--dpo_beta', str(beta), '--learning_rate', str(lr),
            '--save_weight', w, '--save_interval', '100000', '--log_interval', '250', '--num_workers', '2']
    t0 = time.time()
    with open(os.path.join(ROOT, 'out', f'{name}.log'), 'w', encoding='utf-8') as fo, \
         open(os.path.join(ROOT, 'out', f'{name}.err'), 'w', encoding='utf-8') as fe:
        p = subprocess.run(args, cwd=TRAINER, env=env, stdout=fo, stderr=fe)
    dt = time.time() - t0
    tail = ''
    try:
        with open(os.path.join(ROOT, 'out', f'{name}.log'), encoding='utf-8', errors='ignore') as f:
            tail = ''.join(f.readlines()[-3:])
    except OSError:
        pass
    log(f'  训练结束 rc={p.returncode} 用时 {dt/60:.1f}min | ' + ' '.join(tail.split())[:160])
    ck = os.path.join(ROOT, 'out', f'{w}_768_moe.pth')
    acc = eval_dpo(ck, 'out/full_sft_768_moe.pth', tag=name) if os.path.exists(ck) else None
    results.append(dict(name=name, lr=lr, beta=beta, acc=acc, sec=round(dt)))

log('=== 汇总（留出集偏好准确率） ===')
for r in sorted(results, key=lambda x: -(x['acc'] or 0)):
    log(f"  {r['name']:18s} lr={r['lr']:<8g} beta={r['beta']}  准确率={r['acc']}%")
with open(os.path.join(ROOT, 'out', 'dpo_ab_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
log('=== A/B 完成 ===')
