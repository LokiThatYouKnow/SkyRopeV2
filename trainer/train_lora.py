"""
LoRA (Low-Rank Adaptation) 微调脚本 — 支持 LLM 和 VLM
使用 model_lora.py 中的 apply_lora / save_lora / merge_lora

训练流程:
  LLM:  python train_lora.py --data_path ../dataset/lora_medical.jsonl --from_weight full_sft
  VLM:  python train_lora.py --data_path ../dataset/sft_i2t.parquet --from_weight sft_vlm --model_type vlm
  合并: python train_lora.py --merge_only --lora_path ../out/lora.pth --from_weight full_sft --merge_save_path ../out/merged.pth
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
from model.model_lora import apply_lora, save_lora, load_lora, merge_lora
from dataset.lm_dataset import SFTDataset, VLMDataset
from trainer.trainer_utils import (get_lr, Logger, is_main_process, vlm_checkpoint,
                                    init_distributed_mode, setup_seed, SkipBatchSampler,
                                    init_model, init_vlm_model, vlm_collate_fn)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if args.model_type == 'vlm':
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

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, lr: {current_lr:.8f}, eta: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "learning_rate": current_lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            lora_path = f'{args.save_dir}/{args.save_weight}_lora.pth'
            save_lora(raw_model, lora_path)
            Logger(f'LoRA权重已保存: {lora_path}')
            config_to_use = vlm_config if args.model_type == 'vlm' else lm_config
            vlm_checkpoint(config_to_use, weight=args.save_weight, model=model, optimizer=optimizer,
                           scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        del loss, res
        if args.model_type == 'vlm':
            del input_ids, labels, pixel_values
        else:
            del input_ids, labels

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="skyRope LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='lora', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=50, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="../dataset/lora_medical.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，none=从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument('--lora_rank', default=16, type=int, help="LoRA秩（rank）")
    parser.add_argument('--model_type', default='llm', type=str, choices=['llm', 'vlm'], help="模型类型")
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1, 2], help="VLM冻结策略（仅vlm模式）")
    parser.add_argument('--merge_only', action='store_true', help="仅执行LoRA合并而不训练")
    parser.add_argument('--lora_path', default='', type=str, help="LoRA权重路径（合并时使用）")
    parser.add_argument('--merge_save_path', default='', type=str, help="合并后模型保存路径")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="SkyRope-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速")
    args = parser.parse_args()

    # ========== 仅合并模式 ==========
    if args.merge_only:
        if not args.lora_path or not args.merge_save_path:
            raise ValueError("合并模式需要指定 --lora_path 和 --merge_save_path")
        if args.model_type == 'vlm':
            vlm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
            model, _, _ = init_vlm_model(vlm_config, from_weight=args.from_weight, device=args.device)
        else:
            lm_config = skyRopeConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
            model, _ = init_model(lm_config, from_weight=args.from_weight, device=args.device)
        merge_lora(model, args.lora_path, args.merge_save_path)
        Logger(f'LoRA合并完成，模型已保存至: {args.merge_save_path}')
        exit(0)

    # ========== 训练模式 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)

    if args.model_type == 'vlm':
        vlm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, max_position_embeddings=args.max_seq_len, use_moe=bool(args.use_moe))
        ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None
    else:
        lm_config = skyRopeConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, max_position_embeddings=args.max_seq_len, use_moe=bool(args.use_moe))
        ckp_data = vlm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"SkyRope-LoRA-{args.model_type}-Rank{args.lora_rank}-LR{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 初始化模型
    if args.model_type == 'vlm':
        model, tokenizer, preprocess = init_vlm_model(vlm_config, from_weight=args.from_weight, device=args.device, freeze_llm=args.freeze_llm)
        train_ds = VLMDataset(args.data_path, tokenizer, preprocess, image_special_token=vlm_config.image_special_token, image_token_len=vlm_config.image_token_len, max_length=vlm_config.max_position_embeddings)
    else:
        model, tokenizer = init_model(lm_config, from_weight=args.from_weight, device=args.device)
        train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # 应用LoRA
    raw_model = model
    apply_lora(raw_model, rank=args.lora_rank)
    # 冻结非LoRA参数
    for name, param in raw_model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    Logger(f'Trainable Params (LoRA): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    collate_fn = vlm_collate_fn if args.model_type == 'vlm' else None
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    if dist.is_initialized(): dist.destroy_process_group()
