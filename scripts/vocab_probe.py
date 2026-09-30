"""词表扩容可行性探针：对比不同 BPE 词表大小对同一语料的压缩率。

用法: python scripts/vocab_probe.py --train_docs 100000 --vocabs 16000,24000,32000
输出: 每种词表的 tokens/文档、相对 6400 的压缩比、seq340 下的截断率、embedding 参数量
"""
import os
import sys
import json
import time
import argparse
import statistics as st

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import AutoTokenizer


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


def train_bpe(texts, vocab_size, special_tokens):
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tk.train_from_iterator(texts, trainer=trainer)
    return tk


def stats(tk, texts, seq_len=340):
    lens = []
    for t in texts:
        enc = tk.encode(t)
        lens.append(len(enc.ids if hasattr(enc, 'ids') else enc))   # 原生 tokenizers vs HF tokenizer 返回类型不同
    lens.sort()
    trunc = sum(1 for L in lens if L + 2 > seq_len) / len(lens)
    return dict(mean=st.mean(lens), median=lens[len(lens) // 2], p90=lens[int(len(lens) * .9)],
                max_tokens=lens[-1], trunc_rate=trunc)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset/pretrain_t2t_mini.jsonl')
    p.add_argument('--train_docs', type=int, default=100000)
    p.add_argument('--eval_docs', type=int, default=2000)
    p.add_argument('--vocabs', default='16000,24000,32000')
    p.add_argument('--hidden_size', type=int, default=768)
    p.add_argument('--seq_len', type=int, default=340)
    args = p.parse_args()

    base_tk = AutoTokenizer.from_pretrained('model')
    specials = list(base_tk.get_added_vocab().keys())
    print(f'现有词表: {len(base_tk)} (含 {len(specials)} 个 special token)')
    train_texts = load_texts(args.data, 0, args.train_docs)
    eval_texts = load_texts(args.data, args.train_docs, args.eval_docs)
    print(f'训练语料 {len(train_texts)} 条, 评测语料 {len(eval_texts)} 条')

    base_stats = stats(base_tk, eval_texts, args.seq_len)
    ref = base_stats['mean']
    rows = [dict(vocab=len(base_tk), kind='现有(已训练)', **base_stats,
                 embed_m=len(base_tk) * args.hidden_size / 1e6, ratio=1.0,
                 step_saving=0.0, train_s=0.0)]
    print(f'\n{"词表":>8} {"tokens/文档":>12} {"压缩比":>8} {"截断率":>8} {"embed参数":>10} {"训练耗时":>9}')
    print(f'{len(base_tk):>8} {base_stats["mean"]:>12.1f} {1.0:>8.3f} {base_stats["trunc_rate"]*100:>7.1f}% '
          f'{base_tk.vocab_size*args.hidden_size/1e6:>9.1f}M {"-":>9}')

    for vs in [int(v) for v in args.vocabs.split(',')]:
        t0 = time.perf_counter()
        tk = train_bpe(train_texts, vs, specials)
        dt = time.perf_counter() - t0
        s = stats(tk, eval_texts, args.seq_len)
        embed_m = vs * args.hidden_size / 1e6
        row = dict(vocab=vs, kind='新训练', **s, embed_m=embed_m, ratio=s['mean'] / ref,
                   step_saving=1 - s['mean'] / ref, train_s=dt)
        rows.append(row)
        print(f'{vs:>8} {s["mean"]:>12.1f} {row["ratio"]:>8.3f} {s["trunc_rate"]*100:>7.1f}% '
              f'{embed_m:>9.1f}M {dt:>8.0f}s')
    with open('scripts/vocab_probe_results.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print('\n结论: 相对 6400，tokens/文档 降低 X% ⟹ 相同语料下 step 数同比例减少（省算力）')
