"""数据侧验证：精确去重 + MinHash 近邻去重 + 启发式质量过滤（纯 CPU）。

用法:
  python scripts/data_clean.py --data dataset/pretrain_t2t_mini.jsonl --docs 200000 \
      --near_dup_docs 50000 --out dataset/pretrain_t2t_mini_clean.jsonl
输出: 各规则命中率、去重率、保留率、处理速度(docs/s)、全量语料耗时外推
"""
import os
import re
import sys
import json
import time
import hashlib
import argparse
from collections import Counter

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BLACKLIST = ['lorem ipsum', 'javascript', 'document.write', '请记住本站', '转载请注明出处',
             '免责声明', '版权所有', '点击下载', '扫码关注']


def exact_hash(text):
    return hashlib.blake2b(text.encode('utf-8', 'ignore'), digest_size=16).digest()


def shingles(text, n=5, limit=1000):
    s = text[:limit]
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 0))} if s else set()


def minhash_sig(sh, perms=64, mod=(1 << 61) - 1):
    """线性同余族最小哈希，免依赖、比 datasketch 快"""
    if not sh:
        return None
    hs = [int.from_bytes(hashlib.blake2b(x.encode('utf-8', 'ignore'), digest_size=8).digest(), 'little') for x in sh]
    sig = []
    for a in range(1, perms + 1):
        sig.append(min(((a * h + a * 7919) % mod) for h in hs))
    return sig


def band_keys(sig, bands=16):
    per = max(len(sig) // bands, 1)
    return [tuple(sig[i * per:(i + 1) * per]) for i in range(bands)]


def heuristic_filters(text):
    """返回命中的规则名列表（C4/Gopher 式启发式，针对中英混合语料）"""
    hits = []
    n = len(text)
    if n < 50:
        hits.append('too_short')
    if n > 200000:
        hits.append('too_long')
    lines = [l for l in text.split('\n') if l.strip()]
    if lines:
        mean_line = sum(len(l) for l in lines) / len(lines)
        if mean_line < 10:
            hits.append('mean_line_short')
        if len(lines) > 5000:
            hits.append('too_many_lines')
    if n:
        bad = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or '\u4e00' <= ch <= '\u9fff'))
        if bad / n > 0.3:
            hits.append('symbol_ratio')
        zh = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        if zh / n < 0.05 and not re.search(r'[a-zA-Z]{4,}', text):
            hits.append('no_content_chars')
    toks = re.findall(r'\w+', text)
    if len(toks) >= 20:
        bigrams = Counter(zip(toks, toks[1:]))
        if bigrams and bigrams.most_common(1)[0][1] / len(toks) > 0.2:
            hits.append('repetitive')
    low = text.lower()
    if any(b in low for b in BLACKLIST):
        hits.append('blacklist')
    return hits


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset/pretrain_t2t_mini.jsonl')
    p.add_argument('--docs', type=int, default=200000)
    p.add_argument('--near_dup_docs', type=int, default=50000)
    p.add_argument('--near_dup_threshold', type=int, default=56, help='64 个排列里命中最少几个判为重复')
    p.add_argument('--out', default='')
    args = p.parse_args()

    total_bytes = os.path.getsize(args.data)
    doc_rule = Counter()
    exact_seen = {}
    near_band = {}
    n_dup_exact = n_dup_near = n_kept = n = 0
    t0 = time.perf_counter()
    fout = open(args.out, 'w', encoding='utf-8') if args.out else None
    with open(args.data, encoding='utf-8') as f:
        for line in f:
            if n >= args.docs:
                break
            line = line.strip()
            if not line:
                continue
            n += 1
            text = json.loads(line)['text']
            hits = heuristic_filters(text)
            if hits:
                doc_rule.update(hits)
                continue
            h = exact_hash(text)
            if h in exact_seen:
                n_dup_exact += 1
                continue
            exact_seen[h] = 1
            if n <= args.near_dup_docs:
                sig = minhash_sig(shingles(text))
                if sig:
                    dup = False
                    for bk in band_keys(sig):
                        for other in near_band.get(bk, []):
                            if sum(1 for a, b in zip(sig, other) if a == b) >= args.near_dup_threshold:
                                dup = True
                                break
                        if dup:
                            break
                    if dup:
                        n_dup_near += 1
                        continue
                    for bk in band_keys(sig):
                        near_band.setdefault(bk, []).append(sig)
            n_kept += 1
            if fout:
                fout.write(line + '\n')
    dt = time.perf_counter() - t0
    if fout:
        fout.close()
    print(f'扫描 {n:,} 条 / {dt:.0f}s  ({n/dt:,.0f} docs/s)')
    print('启发式规则命中（可叠加）:')
    for k, v in doc_rule.most_common():
        print(f'   {k:20s} {v:>8,}  ({v/n*100:.2f}%)')
    print(f'命中任一启发式规则: {sum(1 for _ in [0])+sum(v for k,v in doc_rule.items() if k=="too_short") and ""}'
          f'共 {n - n_kept - n_dup_exact - n_dup_near + 0:,} 条被规则剔除（按不可叠加口径见下）')
    print(f'精确重复: {n_dup_exact:,} ({n_dup_exact/n*100:.2f}%)')
    print(f'近邻重复(前 {args.near_dup_docs:,} 条内): {n_dup_near:,} ({n_dup_near/args.near_dup_docs*100:.2f}%)')
    print(f'最终保留: {n_kept:,} ({n_kept/n*100:.2f}%)')
    full_docs = int(total_bytes / (os.path.getsize(args.data) / max(n, 1)))
    print(f'外推全量 {total_bytes/1e9:.2f}GB(~{full_docs:,} 条) 处理耗时 ≈ {full_docs/(n/dt)/3600:.1f} h')
