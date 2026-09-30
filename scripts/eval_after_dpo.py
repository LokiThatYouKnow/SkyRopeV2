"""等 DPO2 训练进程结束，自动跑留出集偏好评估并写结论。用法: python scripts/eval_after_dpo.py <pid> <ckpt>"""
import datetime
import os
import re
import subprocess
import sys
import time

ROOT = r'D:\SkyRope\SkyRopeV2'
LOG = os.path.join(ROOT, 'out', 'dpo2_eval.txt')
pid = int(sys.argv[1])
ckpt = sys.argv[2]
env = dict(os.environ)
env.update(SKYROPE_TOKENIZER='../model_tok24k', SKYROPE_VOCAB='24000', SKYROPE_NUM_EXPERTS='4',
           SKYROPE_TOP_K='2', SKYROPE_MOE_INTERMEDIATE='1216', SKYROPE_MOE_IMPL='permute',
           PYTHONIOENCODING='utf-8', PYTHONUTF8='1')


def log(m):
    line = datetime.datetime.now().strftime('%m-%d %H:%M:%S') + '  ' + str(m)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + chr(10))


def alive(p):
    try:
        os.kill(p, 0)
        return True
    except OSError:
        return False


TRAIN_LOG = os.path.join(ROOT, 'out', 'dpo2.log')
MARK = '[2/2](7500/7500)'
log(f'等待训练完成标记 {MARK} ...')
deadline = time.time() + 6 * 3600
while time.time() < deadline:
    try:
        with open(TRAIN_LOG, encoding='utf-8', errors='ignore') as f:
            tail = ''.join(f.readlines()[-40:])
    except OSError:
        tail = ''
    if MARK in tail:
        break
    time.sleep(60)
log('检测到完成标记（或超时），开始评估')

w = os.path.join(ROOT, 'out', os.path.basename(ckpt))
if not os.path.exists(w):
    log('找不到权重文件: ' + w)
    sys.exit(1)
log('权重: %s (%.1f MB, %s)' % (os.path.basename(w), os.path.getsize(w) / 1e6,
                                datetime.datetime.fromtimestamp(os.path.getmtime(w)).strftime('%m-%d %H:%M')))

for tag, policy, ref in [('SFT 基线', 'out/full_sft_768_moe.pth', 'out/full_sft_768_moe.pth'),
                         ('DPO1(1ep/17k对)', 'out/dpo_768_moe.pth', 'out/full_sft_768_moe.pth'),
                         ('DPO2(2ep/15k对)', 'out/dpo2_768_moe.pth', 'out/full_sft_768_moe.pth')]:
    if not os.path.exists(os.path.join(ROOT, policy)):
        log(f'跳过 {tag}（缺 {policy}）')
        continue
    r = subprocess.run([sys.executable, '-X', 'utf8', 'scripts/eval_dpo.py', '--policy', policy,
                        '--ref', ref, '--n', '1000', '--batch_size', '4'],
                       cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    out = (r.stdout or '') + (r.stderr or '')
    keep = [l.strip() for l in out.splitlines() if '准确率' in l or 'logp 差' in l or 'margin:' in l]
    log(f'--- {tag} ---')
    for l in keep:
        log('   ' + l)
log('=== 完成 ===')
