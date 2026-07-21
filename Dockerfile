# ============================================================
# skyRopeLLM - 多任务训练镜像
# 支持的训练模式:
#   pretrain       - 纯文本预训练
#   pretrain_vlm   - 多模态预训练
#   sft_vlm        - 多模态SFT微调
#   grpo           - GRPO强化学习训练
#
# 构建:
#   docker build -t skyrope-llm:latest .
#
# 运行:
#   docker run --gpus all -v ./dataset:/data:ro -v ./output:/output skyrope-llm pretrain
# ============================================================

ARG CUDA_VERSION=12.4.0
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

LABEL maintainer="skyRope"
LABEL description="skyRopeLLM training image with CUDA, PyTorch, Flash-Attention"

# ============================================================
# 系统依赖
# ============================================================
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y --no-install-recommends \
    # 基础工具
    build-essential \
    ca-certificates \
    curl \
    git \
    git-lfs \
    wget \
    vim \
    htop \
    tmux \
    # 编译依赖
    cmake \
    ninja-build \
    pkg-config \
    # Python
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # 图像处理（Pillow/SigLIP需要）
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # 音频/视频编解码
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # 大文件排序/压缩
    pigz \
    pbzip2 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# ============================================================
# 升级 pip / setuptools
# ============================================================
RUN python3 -m pip install --no-cache-dir -U \
    pip \
    setuptools \
    wheel

# ============================================================
# PyTorch (CUDA 12.4 对应 torch >= 2.5)
# ============================================================
ARG TORCH_VERSION=2.6.0
ARG TORCH_CUDA=cu124

RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA}

# ============================================================
# Flash Attention 2 (编译安装, 可选)
# ============================================================
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV MAX_JOBS=4

RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "Flash-Attention 编译失败，将回退到 PyTorch SDPA"

# ============================================================
# 项目依赖
# ============================================================
COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ============================================================
# 额外训练工具
# ============================================================
RUN pip install --no-cache-dir \
    gpustat \
    nvitop \
    orjson \
    accelerate \
    deepspeed \
    && echo "SGLang 如需使用请手动安装: pip install 'sglang[all]'"

# ============================================================
# 目录结构和 symlink（兼容训练脚本中的相对路径）
# ============================================================
RUN mkdir -p /workspace /data /output /checkpoints \
    # 训练脚本以 /workspace 为 CWD, "../model"→/model, "../dataset"→/dataset, "../out"→/out, "../checkpoints"→/checkpoints
    # symlink 桥接这些相对路径
    && ln -sf /workspace/model /model \
    && ln -sf /data /dataset \
    && ln -sf /output /out

# ============================================================
# 创建非 root 用户并授权
# ============================================================
RUN groupadd -r skyrope && useradd -r -g skyrope -m -s /bin/bash skyrope \
    && chown -R skyrope:skyrope /workspace /data /output /checkpoints

# ============================================================
# 环境变量
# ============================================================
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=8
ENV TORCH_HOME=/workspace/.cache/torch
ENV HF_HOME=/workspace/.cache/huggingface
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NCCL_IB_DISABLE=0
ENV NCCL_NET_GDR_LEVEL=2
ENV CUDA_MODULE_LOADING=LAZY

# ============================================================
# 复制项目代码
# ============================================================
WORKDIR /workspace
COPY --chown=skyrope:skyrope . /workspace/

USER skyrope
ENTRYPOINT ["/workspace/scripts/entrypoint.sh"]
CMD ["pretrain"]
