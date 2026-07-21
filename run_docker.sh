#!/bin/bash
# ============================================================
# skyRopeLLM - 一键构建和运行脚本（云服务器用）
#
# 用法:
#   chmod +x run_docker.sh
#   ./run_docker.sh build                    # 构建镜像
#   ./run_docker.sh pretrain                 # 启动预训练
#   ./run_docker.sh pretrain_vlm             # 启动VLM预训练
#   ./run_docker.sh sft_vlm                  # 启动SFT微调
#   ./run_docker.sh grpo                     # 启动GRPO
#   ./run_docker.sh shell                    # 进入容器调试
#   ./run_docker.sh logs pretrain            # 查看日志
#   ./run_docker.sh stop pretrain            # 停止某个训练
# ============================================================
set -e

IMAGE_NAME="skyrope-llm:latest"
COMPOSE_FILE="docker-compose.yml"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERR]${NC}   $*"; exit 1; }

check_nvidia() {
    if ! command -v nvidia-smi &>/dev/null; then
        err "未检测到 nvidia-smi，请先安装 NVIDIA 驱动"
    fi
    if ! docker info 2>/dev/null | grep -q 'Runtimes.*nvidia'; then
        warn "Docker 未配置 nvidia runtime，请安装 nvidia-container-toolkit"
        warn "参考: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
}

build() {
    log "开始构建镜像: ${IMAGE_NAME}"
    check_nvidia
    docker build \
        --build-arg TORCH_VERSION="${TORCH_VERSION:-2.6.0}" \
        --build-arg TORCH_CUDA="${TORCH_CUDA:-cu124}" \
        --build-arg CUDA_VERSION="${CUDA_VERSION:-12.4.0}" \
        -t "${IMAGE_NAME}" \
        .
    log "构建完成! 镜像大小:"
    docker image ls "${IMAGE_NAME}"
}

ensure_dirs() {
    mkdir -p ./dataset ./output ./checkpoints
}

start() {
    local mode=$1
    shift
    log "启动训练: ${mode}"
    ensure_dirs
    docker compose -f "${COMPOSE_FILE}" up -d "${mode}" "$@"
}

stop() {
    local mode=$1
    log "停止: ${mode}"
    docker compose -f "${COMPOSE_FILE}" stop "${mode}"
}

logs() {
    local mode=$1
    docker compose -f "${COMPOSE_FILE}" logs -f "${mode}" --tail=50
}

shell() {
    log "进入交互Shell..."
    ensure_dirs
    docker compose -f "${COMPOSE_FILE}" --profile debug run --rm shell
}

cleanup() {
    warn "清理所有容器和镜像?"
    read -p "确认 (y/N): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        docker compose -f "${COMPOSE_FILE}" down --remove-orphans
        docker rmi "${IMAGE_NAME}" 2>/dev/null || true
        log "清理完成"
    fi
}

CMD=${1:-help}
shift 2>/dev/null || true

case "$CMD" in
    build)       build "$@";;
    pretrain|pretrain_vlm|sft_vlm|grpo)
                 start "$CMD" "$@"
                 warn "查看日志: $0 logs ${CMD}"
                 ;;
    stop)        stop "$@";;
    logs)        logs "$@";;
    shell|bash)  shell "$@";;
    clean)       cleanup "$@";;
    status)      docker compose -f "${COMPOSE_FILE}" ps;;
    help|--help|-h)
        cat << 'EOF'
skyRopeLLM Docker 管理脚本

用法: ./run_docker.sh <COMMAND> [ARGS]

命令:
  build           构建 Docker 镜像
  pretrain        启动纯文本预训练
  pretrain_vlm    启动多模态预训练
  sft_vlm         启动多模态SFT
  grpo            启动GRPO强化学习
  stop <mode>     停止指定训练
  logs <mode>     查看训练日志
  shell           进入容器交互Shell
  status          查看容器状态
  clean           清理容器和镜像

环境变量 (可选):
  EPOCHS=10 BATCH_SIZE=64 LR=5e-4 ./run_docker.sh pretrain
  TORCH_VERSION=2.5.1 CUDA_VERSION=12.1.0 ./run_docker.sh build

自定义参数:
  通过 EXTRA_ARGS 环境变量传递:
  EXTRA_ARGS="--use_moe 1 --hidden_size 1024" ./run_docker.sh pretrain
EOF
        ;;
    *)
        err "未知命令: ${CMD}，使用 '$0 help' 查看帮助"
        ;;
esac
