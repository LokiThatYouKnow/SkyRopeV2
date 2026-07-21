#!/bin/bash
set -e

# ============================================================
# skyRope 训练入口 — 所有脚本统一通过 --model_type llm|vlm 切换
#
# 用法:
#   docker run --gpus all -v ./dataset:/data:ro skyrope pretrain
#   docker run --gpus all -v ./dataset:/data:ro skyrope sft --model_type vlm
# ============================================================

MODE=${1:-help}
shift 2>/dev/null || true

echo "╔══════════════════════════════════════════════╗"
echo "║       skyRope Training Container            ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Mode:     ${MODE}"
echo "║  Time:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "║  Host:     $(hostname)"
echo "║  GPUs:     $(nvidia-smi -L 2>/dev/null | wc -l) detected"
echo "╚══════════════════════════════════════════════╝"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU detected)"

cd /workspace

case "$MODE" in
    # ============ 统一训练入口（通过 --model_type llm|vlm 切换）============
    pretrain)
        echo ">>> 预训练 (train_pretrain.py)  model_type=${MODEL_TYPE:-llm}"
        exec python trainer/train_pretrain.py --model_type ${MODEL_TYPE:-llm} "$@"
        ;;
    sft)
        echo ">>> SFT微调 (train_sft.py)  model_type=${MODEL_TYPE:-llm}"
        exec python trainer/train_sft.py --model_type ${MODEL_TYPE:-llm} "$@"
        ;;
    lora)
        echo ">>> LoRA微调 (train_lora.py)  model_type=${MODEL_TYPE:-llm}"
        exec python trainer/train_lora.py --model_type ${MODEL_TYPE:-llm} "$@"
        ;;
    distill)
        echo ">>> 知识蒸馏 (train_distill.py)  model_type=${MODEL_TYPE:-llm}"
        exec python trainer/train_distill.py --model_type ${MODEL_TYPE:-llm} "$@"
        ;;
    dpo)
        echo ">>> DPO偏好优化 (train_dpo.py)  model_type=${MODEL_TYPE:-llm}"
        exec python trainer/train_dpo.py --model_type ${MODEL_TYPE:-llm} "$@"
        ;;
    grpo)
        echo ">>> GRPO强化学习 (train_grpo.py)  model_type=${MODEL_TYPE:-llm}"
        exec python trainer/train_grpo.py --model_type ${MODEL_TYPE:-llm} "$@"
        ;;
    agent)
        echo ">>> Agent工具调用 (train_agent.py)  model_type=${MODEL_TYPE:-llm}"
        exec python trainer/train_agent.py --model_type ${MODEL_TYPE:-llm} "$@"
        ;;

    # ============ 工具 ============
    tokenizer)
        echo ">>> Tokenizer训练"
        exec python trainer/train_tokenizer.py "$@"
        ;;
    convert)
        echo ">>> 模型格式转换"
        exec python scripts/convert_vlm.py "$@"
        ;;
    eval)
        echo ">>> 模型评估"
        exec python eval_vlm.py "$@"
        ;;
    web)
        echo ">>> Web Demo"
        exec python scripts/web_demo_vlm.py "$@"
        ;;
    bash|shell)
        echo ">>> 交互Shell"
        exec /bin/bash "$@"
        ;;
    python)
        exec python "$@"
        ;;
    help|--help|-h)
        cat << 'EOF'
用法: docker run ... skyrope <MODE> [参数...]

所有训练脚本统一通过 --model_type 切换 LLM/VLM:

  MODEL_TYPE=llm   (默认) 纯文本语言模型
  MODEL_TYPE=vlm          多模态视觉语言模型

训练模式:
  pretrain       预训练        推荐: LLM先 → VLM (freeze_llm=2)
  sft            SFT微调        推荐: LLM→VLM (freeze_llm=1)
  lora           LoRA微调       支持 LLM/VLM
  distill        知识蒸馏       支持 LLM/VLM
  dpo            DPO偏好优化    支持 LLM/VLM
  grpo           GRPO强化学习   支持 LLM/VLM
  agent          Agent工具调用  支持 LLM/VLM

工具模式:
  tokenizer      训练分词器
  convert        模型格式转换
  eval           模型评估
  web            启动Web Demo

训练流程:
  # LLM 全流程
  pretrain --model_type llm --from_weight none
  sft      --model_type llm --from_weight pretrain
  lora     --model_type llm --from_weight full_sft
  dpo      --model_type llm --from_weight full_sft
  grpo     --model_type llm --from_weight full_sft
  agent    --model_type llm --from_weight full_sft

  # VLM 全流程 (先拿到 LLM 权重)
  pretrain --model_type vlm --from_weight pretrain --freeze_llm 2
  sft      --model_type vlm --from_weight pretrain --freeze_llm 1
  lora     --model_type vlm --from_weight full_sft
  grpo     --model_type vlm --from_weight full_sft

环境变量:
  MODEL_TYPE=llm|vlm    模型类型切换 (默认 llm)
  EPOCHS=3              训练轮数
  BATCH_SIZE=16         batch大小
  LR=5e-5               学习率
  FREEZE_LLM=1          VLM冻结策略
  FROM_WEIGHT=pretrain  基础权重
  WANDB_PROJECT=xxx     wandb项目名

示例:
  docker compose up -d pretrain
  MODEL_TYPE=vlm docker compose up -d sft
  docker compose --profile debug run shell
EOF
        exit 0
        ;;
    *)
        echo "未知模式: ${MODE}"
        echo "使用 '$0 help' 查看帮助"
        exit 1
        ;;
esac
