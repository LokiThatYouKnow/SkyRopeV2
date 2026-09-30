"""等 SFT 跑完后自动接 DPO（独立进程，不受会话影响）。日志: out/pipeline_dpo.log"""
import datetime
import os
import re
import subprocess
import sys
import time

ROOT = r'D:\SkyRope\SkyRopeV2'
SFT_LOG = os.path.join(ROOT, 'out', 'sft24k_run2.log')
PIPE_LOG = os.path.join(ROOT, 'out', 'pipeline_dpo.log')
DPO_LOG = os.path.join(ROOT, 'out', 'dpo24k.log')


def log(msg):
    line = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + str(msg)
    with open(PIPE_LOG, 'a', encoding='utf-8') as f:
        f.write(line + chr(10))


def tail(path, n=60):
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            return ''.join(f.readlines()[-n:])
    except OSError:
        return ''


log('=== 等待 SFT 完成 (113215/113215) ===')
deadline = time.time() + 10 * 3600
done = False
while time.time() < deadline:
    if re.search(r'\(113215/113215\)', tail(SFT_LOG)):
        done = True
        break
    time.sleep(60)
if not done:
    log('未检测到 SFT 完成标记，放弃自动接 DPO')
    sys.exit(1)
log('SFT 完成标记已出现，等最终权重落盘 ...')

w = os.path.join(ROOT, 'out', 'full_sft_768_moe.pth')
for _ in range(20):
    if os.path.exists(w) and time.time() - os.path.getmtime(w) < 900:
        log('SFT 权重已落盘: ' + datetime.datetime.fromtimestamp(os.path.getmtime(w)).strftime('%Y-%m-%d %H:%M:%S'))
        break
    time.sleep(30)
if not os.path.exists(w):
    log('找不到 full_sft_768_moe.pth，放弃')
    sys.exit(1)

env = dict(os.environ)
env.update(SKYROPE_TOKENIZER='../model_tok24k', SKYROPE_VOCAB='24000', SKYROPE_NUM_EXPERTS='4',
           SKYROPE_TOP_K='2', SKYROPE_MOE_INTERMEDIATE='1216', SKYROPE_MOE_IMPL='permute',
           PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONUNBUFFERED='1')
args = [sys.executable, '-X', 'utf8', '-u', 'train_dpo.py',
        '--model_type', 'llm', '--use_moe', '1', '--data_path', '../dataset/dpo.jsonl',
        '--from_weight', 'full_sft', '--from_resume', '0', '--epochs', '1', '--batch_size', '2',
        '--accumulation_steps', '4', '--max_seq_len', '1024', '--dpo_beta', '0.1',
        '--save_weight', 'dpo', '--save_interval', '500', '--log_interval', '50', '--num_workers', '2']
log('启动 DPO ...')
with open(DPO_LOG, 'w', encoding='utf-8') as fo, open(DPO_LOG + '.err', 'w', encoding='utf-8') as fe:
    p = subprocess.Popen(args, cwd=os.path.join(ROOT, 'trainer'), env=env, stdout=fo, stderr=fe,
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS)
log('DPO 已启动 pid=%d' % p.pid)

time.sleep(900)
if p.poll() is None:
    log('DPO 运行中，日志尾部:')
    for ln in tail(DPO_LOG, 8).splitlines():
        log('  ' + ln)
    log('=== 驱动结束，DPO 继续后台运行 ===')
else:
    log('DPO 提前退出 rc=%s，日志尾部:' % p.returncode)
    for ln in tail(DPO_LOG, 25).splitlines():
        log('  ' + ln)
    for ln in tail(DPO_LOG + '.err', 15).splitlines():
        log('  ERR ' + ln)
