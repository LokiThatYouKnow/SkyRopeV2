"""
统一预训练脚本 — 支持 LLM (纯文本) 和 VLM (图文多模态)
通过 --model_type llm|vlm 一键切换

LLM 示例:
  python train_pretrain.py --model_type llm --data_path ../dataset/pretrain_t2t_mini.jsonl

VLM 示例:
  python train_pretrain.py --model_type vlm --from_weight pretrain --freeze_llm 2 --data_path ../dataset/pretrain_i2t.parquet
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_skyRope import skyRopeConfig
from model.model_vlm import VLMConfig
from dataset.lm_dataset import PretrainDataset, VLMDataset
from trainer.trainer_utils import (get_lr, Logger, is_main_process, vlm_checkpoint,
                                    init_distributed_mode, setup_seed, SkipBatchSampler,
                                    init_model, init_vlm_model, vlm_collate_fn)
from trainer.optim_utils import build_optimizer, build_lr_lambda, set_lr_mult
from model.model_skyRope import MOEFeedForward

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step
        # 学习率调度（默认 cosine 与旧实现逐位一致，见 optim_utils.build_lr_lambda）
        set_lr_mult(optimizer, lr_fn(epoch * iters + step))

        if is_vlm:
            input_ids, labels, pixel_values = batch
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()} if isinstance(pixel_values, dict) else pixel_values.to(args.device)
            with autocast_ctx:
                res = model(input_ids, labels=labels, pixel_values=pixel_values)
        else:
            input_ids, labels = batch
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            with autocast_ctx:
                res = model(input_ids, labels=labels)

        loss = (res.loss + res.aux_loss) / args.accumulation_steps
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if args.balance_mode == 'aux_free':      # 每步按负载误差调整路由选择 bias
                for m in moe_modules:
                    m.update_router_bias()

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, eta: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if model_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{model_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("vision_encoder.")}
            torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp)
            vlm_checkpoint(model_config, weight=args.save_weight, model=model, optimizer=optimizer,
                           scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict, clean_state_dict

        if is_vlm:
            del input_ids, labels, pixel_values, res, loss
        else:
            del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="skyRope Unified Pretrain (LLM / VLM)")
    # 通用参数
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size (LLM=32, VLM=16)")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构 (1=开启)")
    parser.add_argument('--num_experts', default=4, type=int, help="MoE 专家总数")
    parser.add_argument('--num_experts_per_tok', default=1, type=int, help="每个 token 激活的专家数 (top-k)")
    parser.add_argument('--moe_intermediate_size', default=0, type=int, help="专家隐层维度，0=与 dense FFN 相同")
    parser.add_argument('--router_aux_loss_coef', default=5e-4, type=float, help="MoE 负载均衡辅助损失系数")
    parser.add_argument('--attn_type_list', default='', type=str, help="逗号分隔层型，如 csa,csa,swa,swa (默认 8 层方案)")
    parser.add_argument('--n_shared_experts', default=0, type=int, help="常驻共享专家数 (0=关闭)")
    parser.add_argument('--balance_mode', default='aux', choices=['aux', 'aux_free'], help="MoE 负载均衡方式")
    parser.add_argument('--moe_impl', default='loop', choices=['loop', 'permute'], help="MoE 前向实现")
    # A/B 验证过的组件开关
    parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'adamw_fused', 'muon', 'muon_batched'], help="优化器")
    parser.add_argument('--muon_lr', default=0.02, type=float, help="Muon 组学习率（其余参数用 --learning_rate）")
    parser.add_argument('--weight_decay', default=0.0, type=float, help="权重衰减（预训练建议 0.1）")
    parser.add_argument('--lr_schedule', default='cosine', choices=['cosine', 'warmup_cosine', 'wsd', 'const'], help="学习率调度")
    parser.add_argument('--warmup_ratio', default=0.02, type=float, help="warmup 占总步数比例 (warmup_cosine/wsd)")
    parser.add_argument('--packing', default=0, type=int, choices=[0, 1], help="1=内存内装箱（小语料）")
    parser.add_argument('--shard_dir', default='', type=str, help="预分词分片目录（全量语料用，自动启用 memmap 装箱）")
    parser.add_argument('--tokenizer_path', default='../model', type=str, help="分词器目录（换词表时必改）")
    parser.add_argument('--vocab_size', default=0, type=int, help="词表大小，0=从分词器自动读取")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，none=从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="skyRope-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速")
    # 模式切换
    parser.add_argument('--model_type', default='llm', type=str, choices=['llm', 'vlm'], help="模型类型: llm=纯文本, vlm=多模态")
    parser.add_argument('--freeze_llm', default=2, type=int, choices=[0, 1, 2], help="VLM冻结策略 (0=全训练, 1=解冻首尾层, 2=仅训练proj)")
    args = parser.parse_args()

    is_vlm = (args.model_type == 'vlm')

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    if args.vocab_size == 0:                       # 自动对齐分词器词表，避免 embedding 越界
        from transformers import AutoTokenizer as _AT
        args.vocab_size = len(_AT.from_pretrained(args.tokenizer_path))
    Logger(f'词表: {args.vocab_size} ({args.tokenizer_path})')
    moe_kwargs = dict(vocab_size=args.vocab_size,
                      num_experts=args.num_experts, num_experts_per_tok=args.num_experts_per_tok,
                      router_aux_loss_coef=args.router_aux_loss_coef,
                      n_shared_experts=args.n_shared_experts,
                      balance_mode=args.balance_mode, moe_impl=args.moe_impl)
    if args.moe_intermediate_size > 0:
        moe_kwargs['moe_intermediate_size'] = args.moe_intermediate_size
    if args.attn_type_list:
        moe_kwargs['attn_type_list'] = [t.strip() for t in args.attn_type_list.split(',') if t.strip()]
    if is_vlm:
        model_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                                 max_position_embeddings=args.max_seq_len, use_moe=bool(args.use_moe), **moe_kwargs)
    else:
        model_config = skyRopeConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                                     max_position_embeddings=args.max_seq_len, use_moe=bool(args.use_moe), **moe_kwargs)
    ckp_data = vlm_checkpoint(model_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        tag = 'VLM' if is_vlm else 'LLM'
        wandb_run_name = f"skyRope-{tag}-Pretrain-E{args.epochs}-BS{args.batch_size}-LR{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    if is_vlm:
        model, tokenizer, preprocess = init_vlm_model(model_config, from_weight=args.from_weight, device=args.device,
                                                      freeze_llm=args.freeze_llm, tokenizer_path=args.tokenizer_path)
        train_ds = VLMDataset(args.data_path, tokenizer, preprocess,
                              image_special_token=model_config.image_special_token,
                              image_token_len=model_config.image_token_len,
                              max_length=model_config.max_position_embeddings)
    else:
        model, tokenizer = init_model(model_config, from_weight=args.from_weight, device=args.device,
                                      tokenizer_path=args.tokenizer_path)
        if args.shard_dir:
            from dataset.lm_dataset import MemmapPackedPretrainDataset
            train_ds = MemmapPackedPretrainDataset(args.shard_dir, max_length=args.max_seq_len)
            Logger(f'memmap 装箱数据集: {len(train_ds):,} 块 × {args.max_seq_len} token '
                   f'= {len(train_ds) * args.max_seq_len / 1e9:.2f}B token')
        elif args.packing:
            from dataset.lm_dataset import PackedPretrainDataset
            train_ds = PackedPretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
        else:
            train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # MoE 自检：--use_moe 1 必须真的挂上 MOEFeedForward，否则直接报错而不是静默训 dense
    from model.model_skyRope import MOEFeedForward
    n_moe = sum(1 for m in model.modules() if isinstance(m, MOEFeedForward))
    if args.use_moe and n_moe == 0:
        raise RuntimeError('--use_moe 1 但模型里没有 MOEFeedForward，请检查配置')
    Logger(f'MoE: {"开启" if n_moe else "关闭"} (MoE 层={n_moe}, 专家数={model_config.num_experts}, '
           f'top-{model_config.num_experts_per_tok}, aux_coef={model_config.router_aux_loss_coef})')
    Logger(f'层型: {[type(l.attention).__name__.replace("Attention", "") for l in model.model.layers]}')

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer, opt_desc = build_optimizer(model, kind=args.optimizer, lr=args.learning_rate,
                                          muon_lr=args.muon_lr, weight_decay=args.weight_decay)
    Logger(f'优化器: {opt_desc}')
    moe_modules = [m for m in model.modules() if isinstance(m, MOEFeedForward)]

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    collate_fn = vlm_collate_fn if is_vlm else None
    total_steps = args.epochs * max(len(train_ds) // args.batch_size, 1)
    lr_fn = build_lr_lambda(args.lr_schedule,
                            warmup_steps=int(total_steps * args.warmup_ratio),
                            total_steps=total_steps)
    Logger(f'调度: {args.lr_schedule} (total_steps≈{total_steps}, warmup≈{int(total_steps * args.warmup_ratio)})  '
           f'数据: {"memmap-packed" if args.shard_dir else ("packed" if args.packing else "padded")}  '
           f'wd={args.weight_decay}')
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=collate_fn)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理 ==========
    if dist.is_initialized(): dist.destroy_process_group()
