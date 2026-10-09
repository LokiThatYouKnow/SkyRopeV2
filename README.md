# SkyRopeV2 · 从零构建的 LLM / VLM 训练框架

> **单卡消费级显卡，用一套代码从零训出 LLM 与 VLM：自研 SWA / CSA / HCA / MLA 四种注意力 + MoE 双负载均衡 + Muon 优化器；等墙钟时间下 val loss 比 AdamW 基线低 0.98，吞吐 21,188 tok/s（+75%）、显存仅 6.2 GB。**
>
> 不用 Trainer 黑盒：自己实现 Transformer 核心、注意力、MoE、优化器与完整训练管线；一套代码 `--model_type llm|vlm` 切换纯文本与多模态。

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76b900)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

---

## 一、项目简介

SkyRopeV2 是一个**从零实现**的大语言模型 / 多模态模型训练框架。它不是对 HuggingFace Trainer 的封装，而是自己写了模型定义、注意力机制、MoE 路由、优化器、分布式训练与数据管线，只保留与 HuggingFace `PretrainedConfig` / `GenerationMixin` 的接口兼容，便于接入生态做评测与部署。

目标是把"论文里的先进组件"变成**可 A/B、可复现、可回滚**的工程实现：每个组件都在固定 token 预算 / 等墙钟时间下做过受控实验，结论落在 `scripts/AB_REPORT.md`。

**模型规模**：`hidden_size=768`、`num_hidden_layers=8`，dense 约 **140M** 参数；开启 MoE top-2 × 半宽专家后约 **80M** 激活参数。

---

## 二、核心亮点

### 1. 自研注意力家族（可逐层混合）
| 类型 | 说明 |
| --- | --- |
| `SWAAttention` | 滑动窗口注意力，配自研 **SimpleSWCache**（只保留 w−1 历史，推理时 KV 缓存恒定） |
| `CSAAttention` | 压缩注意力：每 4 个 token 压成 1 个压缩 token（跨块 overlap + 位置偏置门控），查询侧按 `ReLU(q·comp)·w` 选 top-6 压缩块 → 近程精确、长程摘要 |
| `HCAAttention` | 跨层交叉注意力，桥接不同层表征，增强信息复用与梯度流动 |
| `MLAttention` | 潜在注意力（MLA）分支，低秩压缩 KV |

通过 `--attn_type_list`（如 `csa,csa,swa,swa`）实现**逐层异构注意力**，而不是全模型一种注意力。

### 2. MoE：两种负载均衡 + 两种调度
- 专家总数 / top-k / 专家隐层维度 / 共享专家数全部可配
- **aux-loss** 与 **aux-free**（路由偏置动态调整）两种均衡方式
- **loop** 与 **permute** 两种前向调度（permute 把专家计算按 token 排序批量做）
- 支持从 dense 权重迁移：`--from_weight` + `--use_moe 1`

### 3. 优化器与学习率调度可切换
`adamw` / `adamw_fused` / `muon` / `muon_batched`；调度支持 `cosine` / `warmup_cosine` / `wsd` / `const`。
其中 **Muon 只作用于隐层 2D 矩阵**（Newton–Schulz 正交化 + 形状缩放），其余参数仍走 AdamW，支持 `--muon_lr` 单独设学习率。

### 4. 七种训练范式，一套入口
`pretrain` → `pretrain_vlm` → `sft` → `lora` → `distill` → `dpo` → `grpo` → `agent`
全部由 `scripts/entrypoint.sh` 分发，统一 `--model_type llm|vlm` 切换。

### 5. 多模态（VLM）
- 视觉端：**SigLIP2 ViT-B/32 (256)** + 双层 MLP 投影器（`LayerNorm → Linear → GELU → Linear`）
- 图像占位 token `<|image_pad|>` 被 64 个视觉 token 替换
- 冻结策略分阶段训练（先对齐、再指令微调），`--freeze_llm 0/1/2`
- 权重可一键转 HuggingFace 格式（`scripts/convert_vlm.py`）

### 6. 数据与词表工程
- 预分词分片 + **memmap 装箱**（`--shard_dir`），GB 级语料不用全量进内存
- 可选 **24k 词表**（自带 6.4k 与 24k 两套），降低 seq340 截断率
- 数据清洗 / 去重脚本：`scripts/data_clean.py`、`scripts/clean.py`

### 7. 训练即实验：内置 A/B 基准
`scripts/ab_bench.py` 支持固定 token 预算与等墙钟时间两种口径、交错重复取中位数、第二随机种子测噪声底。

---

## 三、实测数据（节选自 scripts/AB_REPORT.md）

**噪声底 ≈ ±0.018 val loss**（同配置换随机种子），下面所有提升都显著高于噪声。

**等 token 预算（300 万真实 token，bs16）：**

| 配置 | ms/step | val loss | 相对基线 | 参数量 |
| --- | --- | --- | --- | --- |
| base（AdamW + aux + loop） | 395 | 4.5525 | — | 140.1M |
| **Muon（Muon + AdamW 混合）** | 514 | **3.8534** | **−0.70** | 140.1M |
| **top-2 × 半宽专家** | 329 | **4.3300** | **−0.22** | **80.4M** |
| packed（序列装箱） | 273 | 4.8026 | +0.25 | 140.1M |
| aux-free 均衡 | 330 | 4.5428 | −0.01（噪声内） | 140.1M |
| csa_only | 286 | 4.5062 | −0.05 | 141.6M |

**等墙钟时间（7 分钟/组，真正决定训练预算的口径）：**

| 配置 | 真实 token | val loss | 相对基线 |
| --- | --- | --- | --- |
| time_base | 4.03M | 4.2964 | — |
| time_packed | 6.65M | 3.9258 | −0.37 |
| time_packed_top2 | 7.55M | 3.5594 | −0.74 |
| **time_stack（packed + Muon + top2 + permute）** | 5.45M | **3.3122** | **−0.98** |

**吞吐与显存（bs32 生产配置）**：`12,075 tok/s / 8.8 GB` → **`21,188 tok/s / 6.2 GB`**（+75% 吞吐、−30% 显存）

> 完整表格、方法与"为什么 packed 在等 token 下反而变差"的分析见 [`scripts/AB_REPORT.md`](scripts/AB_REPORT.md)。

---

## 四、目录结构

```
SkyRopeV2/
├── model/
│   ├── model_skyRope.py      # 核心：skyRopeConfig / RMSNorm / RoPE / 4 种注意力 / MoE / Block
│   ├── model_vlm.py          # skyRopeVLM + VLMConfig + MMVisionProjector
│   ├── model_lora.py         # LoRA 注入
│   ├── tokenizer.json        # 自带 6.4k 词表分词器
│   └── siglip2-base-p32-256-ve/   # 视觉编码器（体积大，需单独下载，见复现教程）
├── trainer/
│   ├── train_pretrain.py     # 预训练（LLM / VLM）
│   ├── train_sft.py          # 指令微调
│   ├── train_lora.py         # LoRA / QLoRA
│   ├── train_distill.py      # 知识蒸馏
│   ├── train_dpo.py          # DPO 偏好优化
│   ├── train_grpo.py         # GRPO 强化学习
│   ├── train_agent.py        # Agent 工具调用训练
│   ├── train_tokenizer.py    # 分词器训练（仅参考，不建议重训）
│   ├── optim_utils.py        # AdamW / fused AdamW / Muon / 调度器
│   ├── rollout_engine.py     # RL rollout 采样引擎
│   └── trainer_utils.py      # 通用工具（种子、参数量、日志）
├── scripts/
│   ├── entrypoint.sh         # 容器内统一训练入口
│   ├── ab_bench.py           # A/B 基准（speed / quality 两套）
│   ├── AB_REPORT.md          # 实验结果报告
│   ├── pretokenize.py        # 预分词 → 分片
│   ├── data_clean.py clean.py  clean_sft_identity.py   # 数据清洗
│   ├── smoke_test.py         # 全链路冒烟测试
│   ├── eval_dpo.py eval_after_dpo.py dpo_ab.py         # DPO 评测
│   ├── chat_test.py          # 命令行对话测试
│   ├── web_demo_vlm.py       # Gradio 多模态 Demo
│   ├── convert_vlm.py        # 转 HuggingFace 格式
│   ├── search.py             # 检索相关脚本
│   ├── train_vocab24k.py vocab_probe.py   # 24k 词表训练与探针
│   └── patch_downstream.py run_dpo_after_sft.py
├── dataset/
│   └── lm_dataset.py         # 数据集定义（jsonl / parquet）
├── eval_vlm.py               # VLM 评测 / 对话推理
├── requirements.txt
├── Dockerfile                # CUDA 12.4 + PyTorch 2.6 + Flash-Attention
├── docker-compose.yml        # 预训练 / SFT / GRPO 等编排
├── run_docker.sh             # 一键构建 + 启动训练
└── .env.example              # 训练超参环境变量模板
```

---

## 五、环境要求

| 项 | 要求 |
| --- | --- |
| Python | 3.11（Dockerfile 内置） |
| PyTorch | 2.6.0 + CUDA 12.4（`torch==2.6.0`、`torchvision`） |
| 显卡 | 单卡即可跑通（本项目在 **RTX 4090D 24G** 上训练）；多卡通过 DDP |
| 其他 | 见 `requirements.txt`（transformers 4.57.6、trl 0.13.0、peft 0.7.1、wandb / swanlab） |

---

## 六、快速开始

### 方式 A：Docker（推荐，云服务器）
```bash
git clone git@github.com:LokiThatYouKnow/SkyRopeV2.git
cd SkyRopeV2

# 1) 准备配置
cp .env.example .env
vim .env                       # 设 EPOCHS / BATCH_SIZE / LR / USE_MOE / GPU_COUNT ...

# 2) 构建镜像（约 15–30 分钟，含 Flash-Attention 编译）
chmod +x run_docker.sh
./run_docker.sh build

# 3) 启动训练（模式见下）
./run_docker.sh pretrain                  # 纯文本预训练
./run_docker.sh pretrain_vlm              # 多模态预训练
./run_docker.sh sft_vlm                   # 多模态 SFT
./run_docker.sh grpo                      # GRPO 强化学习
./run_docker.sh shell                     # 进容器调试
./run_docker.sh logs pretrain             # 看日志
```

### 方式 B：本地直接跑（需自备 CUDA 环境）
```bash
pip install -r requirements.txt
# 自行安装匹配 CUDA 版本的 torch / flash-attn
python trainer/train_pretrain.py --help
```

### 训练范式一览（容器内模式名）
`pretrain` · `sft` · `lora` · `distill` · `dpo` · `grpo` · `agent` · `tokenizer` · `convert` · `eval`

---

## 七、常用训练命令

```bash
# 预训练（LLM）
python trainer/train_pretrain.py --model_type llm \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --hidden_size 768 --num_hidden_layers 8 \
  --max_seq_len 340 --batch_size 32 --accumulation_steps 8 \
  --learning_rate 5e-4 --lr_schedule cosine \
  --optimizer adamw --save_weight pretrain

# 开启 MoE（top-2 + 共享专家 + permute 调度）
python trainer/train_pretrain.py --use_moe 1 --num_experts 4 --num_experts_per_tok 2 \
  --n_shared_experts 1 --balance_mode aux_free --moe_impl permute

# 换 Muon 优化器（仅隐层 2D 矩阵）
python trainer/train_pretrain.py --optimizer muon_batched --muon_lr 0.02

# 逐层异构注意力（前 4 层 CSA、后 4 层 SWA）
python trainer/train_pretrain.py --attn_type_list csa,csa,csa,csa,swa,swa,swa,swa

# 用预分词分片 + memmap 装箱（大语料）
python scripts/pretokenize.py --data dataset/pretrain_t2t.jsonl --tok model_tok24k --out dataset/tokens_full_24k
python trainer/train_pretrain.py --shard_dir dataset/tokens_full_24k --tokenizer_path model_tok24k

# VLM 预训练 / SFT（先下载 SigLIP2，见复现教程）
python trainer/train_pretrain.py --model_type vlm --batch_size 16 --from_weight none
python trainer/train_sft.py --model_type vlm --from_weight pretrain_vlm --data_path ../dataset/sft_i2t.parquet
```

---

## 八、推理与评估

```bash
# VLM 对话 / 评测
python eval_vlm.py --load_from model --weight sft_vlm --hidden_size 768 --num_hidden_layers 8 --use_moe 0

# Gradio 网页 Demo
python scripts/web_demo_vlm.py --load_from ./ --vision_model ../model/siglip2-base-p32-256-ve

# 命令行对话测试 / 全链路冒烟
python scripts/chat_test.py
python scripts/smoke_test.py

# 组件 A/B 基准
python scripts/ab_bench.py --suite speed  --tokens 3000000 --repeats 3
python scripts/ab_bench.py --suite quality --wall_seconds 420
```

---

## 九、数据格式

预训练 / SFT 使用 **jsonl（或 parquet）**，每行一条样本。SFT 对话数据形如：

```json
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

多模态样本在对话中插入图像占位符 `<|image_pad|>`，图像路径/字段由 `dataset/lm_dataset.py` 读取。
> 说明：仓库为控制体积**未包含数据集本体**（`.gitignore` 已排除 `dataset/*.jsonl`、`*.parquet`、预分词分片与权重）。请按 [`REPRODUCE.md`](REPRODUCE.md) 自备语料，或使用任意同结构数据。

---

## 十、已知限制

- `model_tok24k/`（24k 词表）与权重文件未入库，需自行训练或从发布渠道获取；`model/` 自带 6.4k 词表可直接用。
- 视觉编码器 `siglip2-base-p32-256-ve` 需单独下载（见复现教程），仓库只保留了转换脚本与说明。
- 预训练数据未包含，`AB_REPORT.md` 中的数字来自作者本地语料切片，换数据后请以自己的基线为准。
- 词表不建议重训：`train_tokenizer.py` 仅供学习，换词表会导致输出与社区权重不统一。

---

## 十一、致谢

- 视觉编码器基于 [google/siglip2-base-patch32-256](https://huggingface.co/google/siglip2-base-patch32-256)（Apache-2.0），本项目仅裁剪保留 Vision Encoder 部分并转 fp16。
- 训练 / 评测接口与 HuggingFace `transformers` / `trl` / `peft` 生态保持兼容。

## 十二、License

Apache License 2.0
