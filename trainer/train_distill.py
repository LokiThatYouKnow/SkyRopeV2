"""
知识蒸馏训练脚本 — 支持 LLM 和 VLM
使用 Teacher 模型的 logits 作为软标签，结合 KL 散度 + 硬标签交叉熵损失

训练示例:
  LLM: python train_distill.py --teacher_weight pretrain --from_weight none --data_path ../dataset/sft_t2t_mini.jsonl
  VLM: python train_distill.py --teacher_weight sft_vlm --from_weight pretrain_vlm --data_path ../dataset/sft_i2t.parquet --model_type vlm

蒸馏策略:
  - 软损失: KL(teacher_logits || student_logits) * temperature^2
  - 硬损失: CrossEntropy(student_logits, labels)
  - 总损失: alpha * 硬损失 + (1 - alpha) * 软损失
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_skyRope import skyRopeConfig
from model.model_vlm import VLMConfig
from dataset.lm_dataset import SFTDataset, VLMDataset
from trainer.trainer_utils import (get_lr, Logger, is_main_process, vlm_checkpoint,
                                    init_distributed_mode, setup_seed, SkipBatchSampler,
                                    init_model, init_vlm_model, vlm_collate_fn)

warnings.filterwarnings('ignore')


def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
    """
    Args:
        student_logits: [B, S, V]
        teacher_logits: [B, S, V]
        labels: [B, S], -100 for ignored positions
        temperature: 蒸馏温度
        alpha: 硬标签损失权重
    """
    # 硬损失 (CrossEntropy)
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.shape[-1]),
        labels.view(-1),
        ignore_index=-100
    )

    # 软损失 (KL Divergence)
    mask = (labels != -100).float().unsqueeze(-1)  # [B, S, 1]
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)  # [B, S]
    soft_loss = (kl_loss * mask.squeeze(-1)).sum() / mask.sum() * (temperature ** 2)

    return alpha * hard_loss + (1 - alpha) * soft_loss, hard_loss, soft_loss


def train_epoch(epoch, loader, iters, teacher_model, start_step=0, wandb=None):
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
                student_out = model(input_ids, labels=None, pixel_values=pixel_values)
            with torch.no_grad():
                teacher_out = teacher_model(input_ids, labels=None, pixel_values=pixel_values)
        else:
            input_ids, labels = batch
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            with autocast_ctx:
                student_out = model(input_ids, labels=None)
            with torch.no_grad():
                teacher_out = teacher_model(input_ids, labels=None)

        student_logits = student_out.logits[..., :-1, :].contiguous()
        teacher_logits = teacher_out.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        total_loss, hard_loss, soft_loss = distillation_loss(
            student_logits, teacher_logits, shift_labels,
            temperature=args.distill_temperature, alpha=args.distill_alpha
        )
        aux_loss = student_out.aux_loss if student_out.aux_loss is not None else torch.tensor(0.0, device=args.device)
        loss = (total_loss + aux_loss) / args.accumulation_steps
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
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, hard: {hard_loss.item():.4f}, soft: {soft_loss.item():.4f}, lr: {current_lr:.8f}, eta: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "hard_loss": hard_loss.item(), "soft_loss": soft_loss.item(), "learning_rate": current_lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            config_to_use = vlm_config if args.model_type == 'vlm' else lm_config
            moe_suffix = '_moe' if config_to_use.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{config_to_use.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            vlm_checkpoint(config_to_use, weight=args.save_weight, model=model, optimizer=optimizer,
                           scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del loss, total_loss, hard_loss, soft_loss, student_out, teacher_out, student_logits, teacher_logits, shift_labels
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
    parser = argparse.ArgumentParser(description="skyRope Knowledge Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='distill', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="学生模型基于哪个权重，none=从头开始")
    parser.add_argument('--teacher_weight', default='full_sft', type=str, help="教师模型权重名称")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument('--model_type', default='llm', type=str, choices=['llm', 'vlm'], help="模型类型")
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1, 2], help="VLM冻结策略（仅vlm模式）")
    parser.add_argument('--distill_temperature', default=3.0, type=float, help="蒸馏温度")
    parser.add_argument('--distill_alpha', default=0.5, type=float, help="硬标签损失权重（0-1，越大越偏硬标签）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="SkyRope-Distill", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速")
    args = parser.parse_args()

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
        wandb_run_name = f"SkyRope-Distill-{args.model_type}-T{args.distill_temperature}-A{args.distill_alpha}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 初始化学生模型和教师模型
    if args.model_type == 'vlm':
        model, tokenizer, preprocess = init_vlm_model(vlm_config, from_weight=args.from_weight, device=args.device, freeze_llm=args.freeze_llm)
        teacher_model, _, _ = init_vlm_model(vlm_config, from_weight=args.teacher_weight, device=args.device, freeze_llm=2)
        teacher_model = teacher_model.eval().requires_grad_(False)
        train_ds = VLMDataset(args.data_path, tokenizer, preprocess, image_special_token=vlm_config.image_special_token, image_token_len=vlm_config.image_token_len, max_length=vlm_config.max_position_embeddings)
    else:
        model, tokenizer = init_model(lm_config, from_weight=args.from_weight, device=args.device)
        teacher_model, _ = init_model(lm_config, from_weight=args.teacher_weight, device=args.device)
        teacher_model = teacher_model.eval().requires_grad_(False)
        train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

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
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, 0, wandb)

    if dist.is_initialized(): dist.destroy_process_group()
