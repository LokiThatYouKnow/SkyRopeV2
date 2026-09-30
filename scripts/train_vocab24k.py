"""训练 24k 词表并做可用性验证（CPU）。产物: model_tok24k/"""
import os, sys, json, time, argparse
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import AutoTokenizer, PreTrainedTokenizerFast

p = argparse.ArgumentParser()
p.add_argument('--data', default='dataset/pretrain_t2t_mini.jsonl')
p.add_argument('--docs', type=int, default=60000)
p.add_argument('--vocab', type=int, default=24000)
p.add_argument('--out', default='model_tok24k')
args = p.parse_args()

base = AutoTokenizer.from_pretrained('model')
specials = list(base.get_added_vocab().keys())
texts = []
with open(args.data, encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= args.docs: break
        line = line.strip()
        if line: texts.append(json.loads(line)['text'])
t0 = time.perf_counter()
tk = Tokenizer(models.BPE())
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(texts, trainer=trainers.BpeTrainer(
    vocab_size=args.vocab, special_tokens=specials,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
print(f'训练完成 {time.perf_counter()-t0:.0f}s, 词表 {tk.get_vocab_size()}')

os.makedirs(args.out, exist_ok=True)
tk.save(os.path.join(args.out, 'tokenizer.json'))
extra = [t for t in specials if t not in ('<|im_start|>', '<|im_end|>', '<|endoftext|>')]
hf = PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.out, 'tokenizer.json'),
                             bos_token='<|im_start|>', eos_token='<|im_end|>',
                             pad_token='<|endoftext|>', unk_token='<|endoftext|>',
                             additional_special_tokens=extra)
hf.chat_template = base.chat_template
hf.save_pretrained(args.out)

# 验证
chk = AutoTokenizer.from_pretrained(args.out)
probe = ['给我生成一首有关秋天的诗歌。', 'Hello world, this is a tokenizer test 123.', '你好，世界！\n换行测试。']
for t in probe:
    ids = chk(t).input_ids
    assert chk.decode(ids, skip_special_tokens=True).replace(' ', '') == t.replace(' ', ''), (t, chk.decode(ids))
print('解码往返: OK')
for name in ['<|im_start|>', '<|im_end|>', '<|image_pad|>', '<tool_call>', '<think>']:
    assert chk.convert_tokens_to_ids(name) == base.convert_tokens_to_ids(name), name
print(f'special token id 与旧词表一致: OK (bos={chk.bos_token_id}, eos={chk.eos_token_id}, image_pad={chk.convert_tokens_to_ids("<|image_pad|>")})')
full = sum(len(chk(t).input_ids) for t in texts[:2000]) / 2000
old = sum(len(base(t).input_ids) for t in texts[:2000]) / 2000
print(f'压缩率: 6400→{old:.1f} tok/doc, {args.vocab}→{full:.1f} tok/doc (比值 {full/old:.3f})')
print('产物:', args.out)
