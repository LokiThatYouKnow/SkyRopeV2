"""全量语料预分词 → uint16 分片（供 memmap 装箱数据集使用）。

按**字节区间**切分任务：父进程只扫一遍文件记录偏移，worker 各自 seek 读取，
不把文本 pickle 过去 —— 全量 8.3GB 语料下内存占用恒定。

用法:
  python scripts/pretokenize.py --data dataset/pretrain_t2t.jsonl --tok model_tok24k \
      --out dataset/tokens_full_24k --docs_per_shard 200000 --workers 8
产物: <out>/shard_XXXX.npy (uint16 token，文档间插入 eos) + <out>/meta.json
"""
import os, sys, json, time, argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

_TK = None
_SIZE = 0
_EOS = 0


def _init(tok_path):
    global _TK, _SIZE, _EOS
    from transformers import AutoTokenizer
    _TK = AutoTokenizer.from_pretrained(tok_path)
    _SIZE, _EOS = len(_TK), _TK.eos_token_id


def _work(task):
    idx, path, start, end, out_dir = task
    ids_out = []
    ext = _TK.eos_token_id
    with open(path, 'rb') as f:
        f.seek(start)
        data = f.read(end - start)
    for line in data.split(b'\n'):
        line = line.strip()
        if not line:
            continue
        text = json.loads(line.decode('utf-8'))['text']
        ids_out.extend(_TK(text, add_special_tokens=False).input_ids)
        ids_out.append(ext)
    arr = np.asarray(ids_out, dtype=np.uint16)
    p = os.path.join(out_dir, f'shard_{idx:04d}.npy')
    np.save(p, arr)
    return p, int(arr.size)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='dataset/pretrain_t2t.jsonl')
    ap.add_argument('--tok', default='model_tok24k')
    ap.add_argument('--out', default='dataset/tokens_full_24k')
    ap.add_argument('--docs_per_shard', type=int, default=200000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max_docs', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoTokenizer as _AT
    tk_main = _AT.from_pretrained(a.tok)
    _SIZE, _EOS = len(tk_main), tk_main.eos_token_id
    print(f'分词器 {a.tok}: size={_SIZE} eos_id={_EOS}', flush=True)

    t0 = time.perf_counter()
    tasks, idx, n, start = [], 0, 0, 0
    with open(a.data, 'rb') as f:                      # 只扫偏移，不读内容进内存
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                if pos > start:
                    tasks.append((idx, a.data, start, pos, a.out)); idx += 1
                break
            n += 1
            if a.max_docs and n >= a.max_docs:
                tasks.append((idx, a.data, start, f.tell(), a.out)); idx += 1
                break
            if n % a.docs_per_shard == 0:
                tasks.append((idx, a.data, start, f.tell(), a.out)); idx += 1
                start = f.tell()
    print(f'扫描完成 {n:,} 条 → {len(tasks)} 个分片 ({time.perf_counter()-t0:.0f}s), workers={a.workers}', flush=True)

    total, done = 0, 0
    ex = ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.tok,))
    it = iter(tasks)
    pending = {}

    def submit_next():
        try:
            t = next(it)
        except StopIteration:
            return False
        pending[ex.submit(_work, t)] = t[0]
        return True

    for _ in range(a.workers * 2):                     # 有界窗口，避免一次性提交撑爆内存
        if not submit_next():
            break
    while pending:
        finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
        for fut in finished:
            pending.pop(fut)
            p, size = fut.result()
            total += size
            done += 1
            print(f'  [{done}/{len(tasks)}] {os.path.basename(p)} {size:,} tok  '
                  f'累计 {total/1e9:.3f}B  {time.perf_counter()-t0:.0f}s', flush=True)
            submit_next()
    ex.shutdown()
    with open(os.path.join(a.out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'tokenizer': a.tok, 'tokenizer_size': _SIZE, 'total_tokens': total,
                   'shards': len(tasks), 'docs': n, 'eos_token_id': _EOS}, f)
    print(f'完成: {total/1e9:.3f}B tokens, 耗时 {time.perf_counter()-t0:.0f}s, 输出 {a.out}', flush=True)
