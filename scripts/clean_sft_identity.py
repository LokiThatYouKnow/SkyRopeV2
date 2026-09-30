"""sft_t2t.jsonl 清洗：身份字符串替换 + 质量检查（流式，内存恒定）。

用法:
  python scripts/clean_sft_identity.py --src dataset/sft_t2t.jsonl --dst dataset/sft_t2t_clean.jsonl
替换规则（大小写不敏感）:
  MiniMind  -> skyRope      (含 minimind / MINIMIND / MiniMind 等)
  JingyaoGong/Jingyao Gong/gongjy -> liangzhen
输出: 替换计数、质量报告（解析失败/结构异常/空回复/重复/长度分布）
"""
import argparse
import hashlib
import json
import re
import time
from collections import Counter

RULES = [
    (re.compile('minimind', re.I), 'skyRope'),
    (re.compile(r'jingyao\s*gong', re.I), 'liangzhen'),
    (re.compile('gongjy', re.I), 'liangzhen'),
]
WATCH = [re.compile('jingyao', re.I), re.compile('gong', re.I)]

stats = Counter()
hits = Counter()
watch = Counter()


def sub(value):
    if isinstance(value, str):
        for pat, rep in RULES:
            value, n = pat.subn(rep, value)
            if n:
                hits[pat.pattern] += n
        for pat in WATCH:
            if pat.search(value):
                watch[pat.pattern] += 1
        return value
    if isinstance(value, dict):
        return {k: sub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sub(v) for v in value]
    return value


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='dataset/sft_t2t.jsonl')
    ap.add_argument('--dst', default='dataset/sft_t2t_clean.jsonl')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dedup', type=int, default=0, help='1=丢弃重复行')
    ap.add_argument('--drop_empty', type=int, default=0, help='1=丢弃 assistant 内容为空的样本')
    a = ap.parse_args()

    seen = set()
    n = dup = 0
    len_sum = turns = 0
    lens = []
    t0 = time.time()
    with open(a.src, encoding='utf-8') as f, open(a.dst, 'w', encoding='utf-8') as g:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                stats['blank_line'] += 1
                continue
            n += 1
            try:
                obj = json.loads(line)
            except Exception:
                stats['json_error'] += 1
                g.write(line + '\n')
                continue
            convs = obj.get('conversations')
            if not isinstance(convs, list) or not convs:
                stats['no_conversations'] += 1
            else:
                roles = [c.get('role') for c in convs if isinstance(c, dict)]
                stats['role_' + roles[0]] += 1 if roles else 0
                if 'assistant' not in roles:
                    stats['no_assistant'] += 1
                if roles and roles[0] == 'assistant':
                    stats['starts_with_assistant'] += 1
                has_empty_assistant = any(
                    not (c.get('content') or '').strip() for c in convs
                    if isinstance(c, dict) and c.get('role') == 'assistant')
                if has_empty_assistant:
                    stats['empty_assistant_content'] += 1
                    if a.drop_empty:
                        stats['dropped_empty'] += 1
                        continue
                turns += roles.count('assistant')
                if any(isinstance(c, dict) and c.get('tools') for c in convs):
                    stats['has_tools'] += 1
                if any(isinstance(c, dict) and c.get('tool_calls') for c in convs):
                    stats['has_tool_calls'] += 1
                if any(isinstance(c, dict) and (c.get('reasoning_content') or '').strip() for c in convs):
                    stats['has_reasoning'] += 1
                L = sum(len(c.get('content') or '') for c in convs if isinstance(c, dict))
                len_sum += L
                lens.append(L)
            h = hashlib.blake2b(line.encode('utf-8', 'ignore'), digest_size=8).digest()
            if h in seen:
                dup += 1
                if a.dedup:
                    stats['dropped_dup'] += 1
                    continue
            else:
                seen.add(h)
            g.write(json.dumps(sub(obj), ensure_ascii=False) + '\n')
            if n % 500000 == 0:
                print(f'  ... {n:,} 行  {time.time()-t0:.0f}s  替换命中 {sum(hits.values()):,}', flush=True)
            if a.limit and n >= a.limit:
                break

    lens.sort()
    print('\n===== 清洗报告 =====')
    print(f'总行数 {n:,} | 去重后 {n-dup:,} | 重复 {dup:,} ({dup/max(n,1)*100:.2f}%) | 耗时 {time.time()-t0:.0f}s')
    print('替换命中（次数）:')
    for pat, rep in RULES:
        print(f'   {pat.pattern:20s} -> {rep:10s}  {hits[pat.pattern]:,}')
    print('其他疑似身份词（仅统计未替换）:')
    for pat in WATCH:
        print(f'   {pat.pattern:20s} 出现在 {watch[pat.pattern]:,} 行的某个字符串里')
    print('质量:')
    for k in ['json_error', 'no_conversations', 'no_assistant', 'empty_assistant_content',
              'starts_with_assistant', 'has_tools', 'has_tool_calls', 'has_reasoning', 'blank_line']:
        if stats[k]:
            print(f'   {k:24s} {stats[k]:,} ({stats[k]/max(n,1)*100:.2f}%)')
    if lens:
        q = lambda p: lens[int(len(lens) * p)]
        print(f'   内容长度: mean {len_sum/max(n,1):.0f}  median {q(.5)}  p90 {q(.9)}  p99 {q(.99)}  max {lens[-1]}')
    print(f'   平均 assistant 轮数 {turns/max(n,1):.1f}')
    print(f'输出: {a.dst}')
