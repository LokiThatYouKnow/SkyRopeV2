"""
Trainer utilities — 统一的模型初始化、checkpoint、数据加载工具
支持 LLM (skyRopeForCausalLM) 和 VLM (skyRopeVLM) 两种模型类型
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel
from model.model_skyRope import skyRopeForCausalLM
from model.model_vlm import skyRopeVLM


def get_model_params(model, config, ignore_patterns=['vision_encoder']):
    """打印模型参数量统计"""
    def should_count(n):
        return not any(p in n for p in ignore_patterns)

    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    n_routed = getattr(config, 'num_experts', 0)
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)

    expert = sum(p.numel() for n, p in model.named_parameters()
                 if 'mlp.experts.0.' in n and should_count(n)) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters()
                        if 'mlp.shared_experts.0.' in n and should_count(n)) / 1e6

    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)

    if n_routed > 0 and active < total:
        print(f'Model Params: {total:.2f}M (Active: {active:.2f}M)')
    else:
        print(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """Cosine learning rate schedule with warmup (10% ~ 100%)"""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """初始化分布式训练环境，返回 local_rank"""
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model',
               save_dir='../out', device='cuda'):
    """初始化 LLM 模型 (skyRopeForCausalLM)"""
    tokenizer_path = os.environ.get('SKYROPE_TOKENIZER', tokenizer_path)   # 换词表时可用环境变量统一指定
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = skyRopeForCausalLM(lm_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=device)
            model.load_state_dict(weights, strict=False)
            Logger(f'Loaded weights from: {weight_path}')
        else:
            Logger(f'Warning: weight file not found: {weight_path}')

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


def init_vlm_model(vlm_config, from_weight='pretrain_vlm', tokenizer_path='../model',
                   vision_model_path='../model/siglip2-base-p32-256-ve',
                   save_dir='../out', device='cuda', freeze_llm=0):
    """
    初始化 VLM 模型 (skyRopeVLM)

    freeze_llm 策略:
      0 = 全参数可训练（适合大规模 VLM SFT）
      1 = 冻结中间层，仅训练首尾层 + vision_proj（推荐用于 VLM SFT）
      2 = 完全冻结 LLM backbone，仅训练 vision_proj（推荐用于 VLM 预训练对齐阶段）
    """
    tokenizer_path = os.environ.get('SKYROPE_TOKENIZER', tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = skyRopeVLM(vlm_config, vision_model_path)

    # 加载 LLM 权重（如果有的话）
    if from_weight != 'none':
        moe_suffix = '_moe' if vlm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=device)
            model.load_state_dict(weights, strict=False)
            Logger(f'Loaded weights from: {weight_path}')
        else:
            Logger(f'Warning: weight file not found: {weight_path}')

    # 设置训练策略
    # Step 1: 默认冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    if freeze_llm == 0:
        # 全参数可训练（除 vision_encoder 外）
        for name, param in model.named_parameters():
            if "vision_encoder" not in name:
                param.requires_grad = True
    elif freeze_llm == 1:
        # 训练 vision_proj + LLM 首尾层
        for name, param in model.named_parameters():
            if "vision_proj" in name:
                param.requires_grad = True
        last_idx = vlm_config.num_hidden_layers - 1
        for name, param in model.model.named_parameters():
            if "layers.0." in name or f'layers.{last_idx}.' in name:
                param.requires_grad = True
        # 同时训练 embedding 和 norm
        for name, param in model.model.named_parameters():
            if 'embed_tokens' in name or 'norm.' in name:
                param.requires_grad = True
    elif freeze_llm == 2:
        # 仅训练 vision_proj（阶段一对齐）
        for name, param in model.named_parameters():
            if "vision_proj" in name:
                param.requires_grad = True

    get_model_params(model, vlm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')

    preprocess = model.processor
    return model.to(device), tokenizer, preprocess


def init_model_by_type(config, model_type='llm', from_weight='pretrain', device='cuda',
                       freeze_llm=0, **kwargs):
    """
    统一的模型初始化入口，自动区分 LLM / VLM

    Args:
        config: skyRopeConfig 或 VLMConfig
        model_type: 'llm' 或 'vlm'
        from_weight: 权重名称
        device: 设备
        freeze_llm: VLM 冻结策略（仅 model_type='vlm' 时生效）
    Returns:
        (model, tokenizer) 或 (model, tokenizer, preprocess) for VLM
    """
    if model_type == 'vlm':
        return init_vlm_model(config, from_weight=from_weight, device=device,
                              freeze_llm=freeze_llm, **kwargs)
    else:
        return init_model(config, from_weight=from_weight, device=device, **kwargs)


def vlm_checkpoint(config, weight='pretrain', model=None, optimizer=None, epoch=0, step=0,
                   wandb=None, save_dir='../checkpoints', **kwargs):
    """
    统一的 checkpoint 保存/加载函数
    - 当 model 不为 None 时: 保存 checkpoint
    - 当 model 为 None 时: 加载 checkpoint

    额外 kwargs 支持: scaler, scheduler (需实现 state_dict() 的对象)
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_suffix = '_moe' if config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{config.hidden_size}{moe_suffix}.pth'
    resume_path = f'{save_dir}/{weight}_{config.hidden_size}{moe_suffix}_resume.pth'

    if model is not None:
        # ========== 保存模式 ==========
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()

        # 清理 vision_encoder（不保存冻结的视觉编码器权重）
        clean_state_dict = {k: v for k, v in state_dict.items()
                            if not k.startswith("vision_encoder.")}
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        # 获取 wandb_id
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 构建 resume 数据
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict() if optimizer else None,
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id,
        }

        # 保存额外的有状态对象 (scaler, scheduler 等)
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

        del state_dict, clean_state_dict, resume_data
        torch.cuda.empty_cache()

    else:
        # ========== 加载模式 ==========
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                saved_step = ckp_data['step']
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}->{current_ws})，'
                       f'原step({saved_step})已自动转化为新step({ckp_data["step"]})')
            return ckp_data
        return None


def vlm_collate_fn(batch):
    """
    VLM 批次整理函数
    处理 (input_ids, labels, pixel_data) 三元组，
    pixel_data 可能是 dict (HuggingFace processor 输出) 或 tensor
    """
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])

    # pixel_data 可能是 dict 或 tensor
    pixel_data = [b[2] for b in batch]
    sample = pixel_data[0]

    if hasattr(sample, 'keys') and callable(getattr(sample, 'keys', None)):
        # HuggingFace processor dict 格式
        pixel_values = {k: torch.cat([d[k] for d in pixel_data], dim=0) for k in sample.keys()}
    else:
        # 纯 tensor 格式
        pixel_values = torch.stack([d for d in pixel_data])

    return input_ids, labels, pixel_values


class SkipBatchSampler(Sampler):
    """支持跳过指定数量 batch 的采样器（用于续训恢复）"""
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    """外部 Reward 模型包装器，用于 GRPO / RLHF 训练的奖励计算"""
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        """计算对话-回复对的奖励分数"""
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)
