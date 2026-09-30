"""对话能力实测：加载权重 → 按 chat template 生成 → 打印回复 + 客观指标。

用法（需先设 SKYROPE_* 对齐环境变量）:
  python scripts/chat_test.py --ckpt out/full_sft_768_moe.pth
  python scripts/chat_test.py --ckpt out/dpo2_768_moe.pth --tag dpo2
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from collections import Counter
from transformers import AutoTokenizer
from model.model_skyRope import skyRopeConfig, skyRopeForCausalLM

PROMPTS = [
    ('身份', '你是谁？是谁开发的？'),
    ('常识', '中国的四大发明是什么？简单说说。'),
    ('推理', '一个笼子里有鸡和兔共 10 只，脚共 28 只。鸡和兔各几只？请给出推理过程。'),
    ('指令跟随', '用三点说明为什么要早睡，每点不超过 15 个字。'),
    ('创作', '写一首关于秋天的五言绝句。'),
    ('格式化', '把下面这句话翻译成英文，只输出译文：今天天气很好。'),
    ('多轮', '我前面问了什么？'),
    ('英文', 'Explain in two sentences why the sky is blue.'),
    ('长文本', '请用大约 150 字介绍杭州这座城市。'),
    ('拒答/边界', '你能告诉我明天的比特币价格吗？'),
]


def repetition_ratio(text, n=4):
    toks = list(text)
    if len(toks) < n * 2:
        return 0.0
    grams = [''.join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    c = Counter(grams)
    return max(c.values()) / len(grams)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='out/full_sft_768_moe.pth')
    p.add_argument('--tag', default='')
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()

    tok_path = os.environ.get('SKYROPE_TOKENIZER', 'model_tok24k')
    if not os.path.isdir(tok_path):
        alt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           os.path.basename(tok_path))
        if os.path.isdir(alt):
            tok_path = alt
    tok = AutoTokenizer.from_pretrained(tok_path)
    cfg = skyRopeConfig(use_moe=True, max_position_embeddings=2048)
    model = skyRopeForCausalLM(cfg)
    r = model.load_state_dict(torch.load(a.ckpt, map_location='cpu'), strict=False)
    print(f'权重 {os.path.basename(a.ckpt)} | 缺失 {len(r.missing_keys)} 多余 {len(r.unexpected_keys)} '
          f'| 参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    model = model.half().eval().to(a.device)

    ends_eos = reps = 0
    total_new = 0
    t_all = time.time()
    rows = []
    for name, q in PROMPTS:
        history = [{'role': 'user', 'content': q}]      # 每题独立单轮上下文（避免前文串扰）
        prompt = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors='pt').input_ids.to(a.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=a.max_new_tokens, temperature=a.temperature,
                                 top_p=a.top_p, top_k=a.top_k, eos_token_id=tok.eos_token_id)
        dt = time.time() - t0
        new = out[0][ids.shape[1]:]
        text = tok.decode(new, skip_special_tokens=True).strip()
        total_new += new.numel()
        ok = tok.eos_token_id in new.tolist()
        ends_eos += ok
        rr = repetition_ratio(text)
        reps += rr > 0.25
        rows.append((name, q, text, ok, new.numel()))
    print(f'\n=== 汇总 ({a.tag or os.path.basename(a.ckpt)}) ===')
    for name, q in PROMPTS:
        pass
    print(f'\n=== 逐题速览 ({a.tag}) ===')
    for name, q, text, ok, ln in rows:
        print(f'  [{name:6s}] {text[:70].replace(chr(10), " ")}  {"|EOS" if ok else "|截断"} ({ln}tok)')
    print(f'题目 {len(PROMPTS)} | 正常以 EOS 结束 {ends_eos}/{len(PROMPTS)} | '
          f'重复率>0.25 的 {reps}/{len(PROMPTS)} | 生成 {total_new} tok | '
          f'总耗时 {time.time()-t_all:.0f}s')
