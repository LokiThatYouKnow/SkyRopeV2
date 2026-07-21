"""
DPO (Direct Preference Optimization) 训练脚本 — 支持 LLM 和 VLM
使用偏好对数据 (chosen/rejected) 进行直接偏好优化

训练示例:
  LLM: python train_dpo.py --from_weight full_sft --data_path ../dataset/dpo.jsonl
  VLM: python train_dpo.py --from_weight sft_vlm --data_path ../dataset/dpo.jsonl --model_type vlm

DPO 损失:
  loss = -log(sigma(beta * (log_p_chosen/ref_chosen - log_p_rejected/ref_rejected))))
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from model.model_skyRope import skyRopeConfig
from model.model_vlm import VLMConfig
from dataset.lm_dataset import pre_processing_chat
from trainer.trainer_utils import (get_lr, Logger, is_main_process, vlm_checkpoint,
                                    init_distributed_mode, setup_seed, SkipBatchSampler,
                                    init_model, init_vlm_model)
from datasets import load_dataset, Features, Value

warnings.filterwarnings('ignore')


class DPODataset(Dataset):
    """DPO 偏好对数据集，支持 text-only 和多模态"""
    def __init__(self, jsonl_path, tokenizer, max_length=1024, model_type='llm'):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        features = Features({
            'chosen': [{'role': Value('string'), 'content': Value('string')}],
            'rejected': [{'role': Value('string'), 'content': Value('string')}],
        })
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)

    def __len__(self):
        return len(self.samples)

    def _format_and_tokenize(self, conversations):
        conversations = pre_processing_chat(conversations)
        prompt = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False
        )
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成 labels (只对 assistant 回复计算损失)
        bos_id = self.tokenizer(f'{self.tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        eos_id = self.tokenizer(f'{self.tokenizer.eos_token}\n', add_special_tokens=False).input_ids

        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(bos_id)] == bos_id:
                start = i + len(bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(eos_id)] == eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen_ids, chosen_labels = self._format_and_tokenize(sample['chosen'])
        rejected_ids, rejected_labels = self._format_and_tokenize(sample['rejected'])
        return {
            'chosen_input_ids': chosen_ids,
            'chosen_labels': chosen_labels,
            'rejected_input_ids': rejected_ids,
            'rejected_labels': rejected_labels,
        }


def compute_log_probs(model, input_ids, labels, pixel_values=None):
    """计算模型在 labels 位置上的 log probabilities"""
    kwargs = {}
    if pixel_values is not None:
        kwargs['pixel_values'] = pixel_values
    outputs = model(input_ids, labels=None, **kwargs)
    logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100).float()
    return (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)


def train_epoch(epoch, loader, iters, ref_model, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        chosen_ids = batch['chosen_input_ids'].to(args.device)
        chosen_labels = batch['chosen_labels'].to(args.device)
        rejected_ids = batch['rejected_input_ids'].to(args.device)
        rejected_labels = batch['rejected_labels'].to(args.device)

        with autocast_ctx:
            # Policy model log probs
            policy_chosen_logp = compute_log_probs(model, chosen_ids, chosen_labels)
            policy_rejected_logp = compute_log_probs(model, rejected_ids, rejected_labels)

            # Reference model log probs
            with torch.no_grad():
                ref_chosen_logp = compute_log_probs(ref_model, chosen_ids, chosen_labels)
                ref_rejected_logp = compute_log_probs(ref_model, rejected_ids, rejected_labels)

            # DPO loss
            chosen_ratio = policy_chosen_logp - ref_chosen_logp
            rejected_ratio = policy_rejected_logp - ref_rejected_logp
            logits_diff = args.dpo_beta * (chosen_ratio - rejected_ratio)
            loss = -F.logsigmoid(logits_diff).mean()
            loss = loss / args.accumulation_steps

        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            accuracy = (logits_diff > 0).float().mean().item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, acc: {accuracy:.4f}, lr: {current_lr:.8f}, eta: {eta_min:.1f}min')
            if wandb: wandb.log({"dpo_loss": current_loss, "accuracy": accuracy, "learning_rate": current_lr})

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
                           epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del chosen_ids, chosen_labels, rejected_ids, rejected_labels
        del policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="skyRope DPO Training")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument('--model_type', default='llm', type=str, choices=['llm', 'vlm'], help="模型类型")
    parser.add_argument('--dpo_beta', default=0.1, type=float, help="DPO温度参数beta")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="SkyRope-DPO", help="wandb项目名")
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
        wandb_run_name = f"SkyRope-DPO-{args.model_type}-Beta{args.dpo_beta}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 初始化 Policy 模型和 Reference 模型
    if args.model_type == 'vlm':
        model, tokenizer, _ = init_vlm_model(vlm_config, from_weight=args.from_weight, device=args.device)
        ref_model, _, _ = init_vlm_model(vlm_config, from_weight=args.from_weight, device=args.device)
    else:
        model, tokenizer = init_model(lm_config, from_weight=args.from_weight, device=args.device)
        ref_model, _ = init_model(lm_config, from_weight=args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len, model_type=args.model_type)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, 0, wandb)

    if dist.is_initialized(): dist.destroy_process_group()
