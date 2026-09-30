"""给下游训练脚本补齐"与预训练对齐"的开关：MoE 形状 + 分词器/词表。

不改它们的优化器与调度（保持各自默认），只保证模型结构能对上 checkpoint。
用法: python scripts/patch_downstream.py
"""
import re, sys, ast

ARGS = '''    parser.add_argument('--num_experts', default=4, type=int, help="MoE 专家总数")
    parser.add_argument('--num_experts_per_tok', default=1, type=int, help="每个 token 激活的专家数")
    parser.add_argument('--moe_intermediate_size', default=0, type=int, help="专家隐层维度，0=与 dense 相同")
    parser.add_argument('--moe_impl', default='loop', choices=['loop', 'permute'], help="MoE 前向实现")
    parser.add_argument('--tokenizer_path', default='../model', type=str, help="分词器目录（换词表时必改）")
    parser.add_argument('--vocab_size', default=0, type=int, help="词表大小，0=自动读取")
'''

FILES = ['trainer/train_dpo.py', 'trainer/train_lora.py', 'trainer/train_grpo.py', 'trainer/train_agent.py']
for path in FILES:
    src = open(path, encoding='utf-8').read()
    if 'align-patched' in src:
        print(f'{path}: 已打过补丁，跳过'); continue
    orig = src
    # 1) argparse：在 --use_moe 行后插入
    m = re.search(r"^( *parser\.add_argument\('--use_moe'.*)$", src, re.M)
    if not m:
        print(f'{path}: ! 找不到 --use_moe 参数行'); continue
    src = src[:m.end()] + '\n' + ARGS.rstrip('\n') + src[m.end():]
    # 2) 词表自动对齐 + moe_kwargs（插在 model_config 之前）
    m2 = re.search(r"^ *model_config = (skyRopeConfig|VLMConfig)\(", src, re.M)
    if not m2:
        print(f'{path}: ! 找不到 model_config 构造'); continue
    pre = '''    if args.vocab_size == 0:
        from transformers import AutoTokenizer as _AT
        args.vocab_size = len(_AT.from_pretrained(args.tokenizer_path))
    Logger(f'词表: {args.vocab_size} ({args.tokenizer_path})')
    _moe_kwargs = dict(vocab_size=args.vocab_size, num_experts=args.num_experts,
                       num_experts_per_tok=args.num_experts_per_tok, moe_impl=args.moe_impl)
    if args.moe_intermediate_size > 0:
        _moe_kwargs['moe_intermediate_size'] = args.moe_intermediate_size
'''
    src = src[:m2.start()] + pre + src[m2.start():]
    # 3) 所有 skyRopeConfig(/VLMConfig( 调用注入 **moe_kwargs
    src = re.sub(r"((?:skyRopeConfig|VLMConfig)\([^)]*?use_moe=bool\(args\.use_moe\))", r"\1, **_moe_kwargs", src)
    # 4) init_model 传 tokenizer_path
    src = re.sub(r"init_model\(model_config, from_weight=args\.from_weight, device=args\.device\)",
                 "init_model(model_config, from_weight=args.from_weight, device=args.device,\n                                      tokenizer_path=args.tokenizer_path)", src)
    src = re.sub(r"init_vlm_model\(model_config, from_weight=args\.from_weight, device=args\.device, freeze_llm=args\.freeze_llm\)",
                 "init_vlm_model(model_config, from_weight=args.from_weight, device=args.device,\n                                                      freeze_llm=args.freeze_llm, tokenizer_path=args.tokenizer_path)", src)
    src = src.replace('if __name__ == "__main__":', '# align-patched\nif __name__ == "__main__":', 1)
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f'{path}: ! 补丁后语法错误 {e}'); continue
    open(path, 'w', encoding='utf-8').write(src)
    print(f'{path}: OK (注入 {src.count("_moe_kwargs")} 处, tokenizer_path {src.count("tokenizer_path=args.tokenizer_path")} 处)')
