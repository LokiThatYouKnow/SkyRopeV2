"""
统一 Agent (Tool-Use / Function-Calling) 训练脚本 — 支持 LLM 和 VLM
通过 --model_type llm|vlm 一键切换，使用 GRPO + 工具执行反馈进行强化学习

LLM 示例:
  python train_agent.py --model_type llm --from_weight full_sft --data_path ../dataset/agent_rl.jsonl

VLM 示例:
  python train_agent.py --model_type vlm --from_weight sft_vlm --freeze_llm 1 --data_path ../dataset/agent_rl.jsonl
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import json
import re
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_skyRope import skyRopeConfig
from model.model_vlm import VLMConfig
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (Logger, is_main_process, vlm_checkpoint,
                                    init_distributed_mode, setup_seed, SkipBatchSampler,
                                    init_model, init_vlm_model, LMForRewardModel)
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps

warnings.filterwarnings('ignore')


def parse_tool_calls(response: str):
    """解析模型回复中的 <tool_call>...</tool_call>"""
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, response, re.DOTALL)
    tool_calls = []
    for match in matches:
        try:
            tool_calls.append(json.loads(match))
        except json.JSONDecodeError:
            continue
    return tool_calls


def simulate_tool_execution(tool_call: dict) -> str:
    """模拟工具执行（可替换为真实工具）"""
    name = tool_call.get('name', '')
    arguments = tool_call.get('arguments', {})

    if name == 'calculator':
        try:
            expr = arguments.get('expression', '0')
            allowed = set('0123456789+-*/().% ')
            if all(c in allowed for c in expr):
                return str(eval(expr))
        except Exception:
            return 'Error: invalid expression'

    elif name == 'search':
        query = arguments.get('query', '')
        return f'Search results for "{query}": No results found (simulated).'

    elif name == 'weather':
        city = arguments.get('city', '')
        return f'Weather in {city}: 22°C, partly cloudy (simulated).'

    elif name == 'get_current_time':
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return f'Tool "{name}" executed (simulated).'


def calculate_agent_rewards(prompts, responses, reward_model=None):
    """Agent 训练奖励: 格式正确性 + 工具调用合理性 + 回答质量 + 可选 Reward 模型"""
    rewards = torch.zeros(len(responses), device=args.device)

    for idx, (prompt, response) in enumerate(zip(prompts, responses)):
        reward = 0.0
        clean_response = response.strip()

        if '</think>' in clean_response:
            think_content, answer_content = clean_response.split('</think>', 1)
            if 20 <= len(think_content.strip()) <= 500:
                reward += 0.5
            answer = answer_content.strip()
        else:
            answer = clean_response

        has_tool_call = '<tool_call>' in answer and '</tool_call>' in answer
        if has_tool_call:
            tool_calls = parse_tool_calls(answer)
            if tool_calls:
                reward += 1.0
                for tc in tool_calls:
                    if tc.get('name') and tc.get('arguments'):
                        reward += 0.5
                        break
            else:
                reward -= 0.5

        if 20 <= len(clean_response) <= 1500:
            reward += 0.25
        elif len(clean_response) < 20:
            reward -= 0.5

        toks = re.findall(r"\w+|[^\w\s]", clean_response.lower())
        if len(toks) > 0:
            unique_ratio = len(set(toks)) / len(toks)
            if unique_ratio < 0.4:
                reward -= 0.5

        rewards[idx] = reward

    if reward_model is not None:
        with torch.no_grad():
            for idx, (prompt, response) in enumerate(zip(prompts, responses)):
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                try:
                    score = reward_model.get_score(messages, response)
                    rewards[idx] += max(min(score, 2.0), -2.0)
                except Exception:
                    pass

    return rewards


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device)

        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(outputs)
            aux_loss = res.aux_loss if model_config.use_moe else torch.tensor(0.0, device=args.device)
            logits = res.logits[:, :-1, :]
            per_token_logps = F.log_softmax(logits, dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1)[:, -completion_ids.size(1):]

        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(ref_model, outputs, completion_ids.size(1))

        rewards = calculate_agent_rewards(prompts, completions, reward_model).to(args.device)

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(min(len(prompts), 2)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('=' * 80)
                Logger(prompts[i])
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"--- gen[{j}] reward={rewards[idx].item():.4f} ---")
                    Logger(completions[idx])
                Logger('=' * 80)

        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        is_eos = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
        per_token_loss1 = ratio * advantages.unsqueeze(1)
        per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
        per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if is_main_process() and step % args.save_interval == 0:
                rollout_engine.update_policy(model)

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, PolicyLoss: {policy_loss_val:.4f}, '
                   f'AvgLen: {avg_len_val:.2f}, LR: {current_lr:.8f}')
            if wandb and is_main_process():
                wandb.log({"reward": avg_reward_val, "policy_loss": policy_loss_val,
                           "avg_response_len": avg_len_val, "learning_rate": current_lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if model_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{model_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            vlm_checkpoint(model_config, weight=args.save_weight, model=model, optimizer=optimizer,
                           epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="skyRope Unified Agent Training (LLM / VLM)")
    # 通用参数
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='agent', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=5, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/agent_rl.jsonl", help="Agent训练数据路径")
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.05, help="KL惩罚系数")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO PPO clip epsilon")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="skyRope-Agent", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.5, help="按概率开启thinking")
    # 模式切换
    parser.add_argument('--model_type', default='llm', type=str, choices=['llm', 'vlm'], help="模型类型")
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1, 2], help="VLM冻结策略")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8996", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_agent", help="SGLang共享存储路径")
    args = parser.parse_args()

    is_vlm = (args.model_type == 'vlm')

    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    if is_vlm:
        model_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                                 max_position_embeddings=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    else:
        model_config = skyRopeConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                                     max_position_embeddings=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = vlm_checkpoint(model_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        tag = 'VLM' if is_vlm else 'LLM'
        wandb_run_name = f"skyRope-{tag}-Agent-E{args.epochs}-LR{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    base_weight = args.from_weight
    if is_vlm:
        model, tokenizer, _ = init_vlm_model(model_config, base_weight, device=args.device, freeze_llm=args.freeze_llm)
        ref_model, _, _ = init_vlm_model(model_config, base_weight, device=args.device, freeze_llm=args.freeze_llm)
    else:
        model, tokenizer = init_model(model_config, base_weight, device=args.device)
        ref_model, _ = init_model(model_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    reward_model = None
    if os.path.exists(args.reward_model_path):
        try:
            reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
            Logger(f'Reward模型加载成功: {args.reward_model_path}')
        except Exception as e:
            Logger(f'Reward模型加载失败，使用规则奖励: {e}')

    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine, policy_model=model, tokenizer=tokenizer,
        device=args.device, autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url, sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )

    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=model_config.max_position_embeddings, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    if is_main_process(): rollout_engine.update_policy(model)

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb)
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb)

    if dist.is_initialized(): dist.destroy_process_group()
