# SkyRopeV2 复现教程（从零跑通 LLM → VLM → 对齐 → 评测）

> 目标：在一台 **单卡 24G（RTX 4090 / 4090D）** 机器上，从零复现本项目的完整训练链路，
> 并能用 `scripts/ab_bench.py` 复现 `AB_REPORT.md` 里的对照结果。
> 全程只依赖本仓库 + 公开数据 + 公开视觉编码器。

---

## 0. 复现清单（照着勾）

- [ ] 环境：Python 3.11 / PyTorch 2.6 / CUDA 12.4（推荐直接用 Dockerfile）
- [ ] 代码：`git clone` + `cp .env.example .env`
- [ ] 视觉编码器：`model/siglip2-base-p32-256-ve/`（VLM 阶段必需）
- [ ] 数据：`dataset/pretrain_t2t_mini.jsonl` 等（仓库不含，需自备，格式见第 3 节）
- [ ] 阶段一：`pretrain` → `out/pretrain_768.pth`
- [ ] 阶段二：VLM `pretrain`（`MODEL_TYPE=vlm`）→ `out/pretrain_vlm_768.pth`
- [ ] 阶段三：`sft` → `out/sft_vlm_768.pth`
- [ ] 阶段四：`dpo` / `grpo` → `out/dpo_768.pth` 等
- [ ] 验证：`smoke_test.py` → `eval_vlm.py` → `web_demo_vlm.py` → `ab_bench.py`

---

## 1. 环境准备

### 1.1 硬件
| 场景 | 配置 | 说明 |
| --- | --- | --- |
| 最小可用 | 单卡 24G（4090 / 4090D） | 本项目主开发环境，bs16 约 4.4G 显存、bs32 约 8.8G |
| 推荐 | 多卡 + NVLink/高速 PCIe | 通过 DDP 扩展；`.env` 里 `GPU_COUNT=all/2/4/8` |
| 纯 CPU | 不推荐 | 只够跑 `smoke_test.py` 级别的形状校验 |

### 1.2 软件（方式 A：Docker，最省事）
`Dockerfile` 已经锁定：Ubuntu 22.04 + CUDA 12.4 + Python 3.11 + PyTorch 2.6(cu124) + Flash-Attention。
宿主机只需：
```bash
nvidia-smi                     # 确认驱动正常
docker --version
docker info | grep -i nvidia   # 需要 nvidia-container-toolkit
```

### 1.3 软件（方式 B：本地 Python）
```bash
conda create -n skyrope python=3.11 -y && conda activate skyrope
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# flash-attn 需与本机 CUDA/torch 版本匹配，编译较慢（可选，缺失时会退化到 SDPA）
pip install flash-attn --no-build-isolation
```

---

## 2. 获取代码与配置

```bash
git clone git@github.com:LokiThatYouKnow/SkyRopeV2.git
cd SkyRopeV2
cp .env.example .env
```

`.env` 关键项（训练脚本 / docker compose 共用）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `EPOCHS` | 2 | 训练轮数 |
| `BATCH_SIZE` | 32 | LLM 用 32，VLM 建议 16 |
| `LR` | 5e-4 | 初始学习率（A/B 显示 5e-4 已接近最优） |
| `ACCUM_STEPS` | 8 | 梯度累积 |
| `USE_MOE` | 1 | 1=开 MoE，0=dense |
| `FREEZE_LLM` | 2 | VLM 冻结策略：0/1/2 |
| `FROM_WEIGHT` | pretrain_vlm | 续训的基座权重前缀 |
| `GPU_COUNT` | all | 参加训练的卡数 |
| `MODEL_TYPE` | llm | `llm` / `vlm`，切换训练目标 |

---

## 3. 数据准备

### 3.1 期望格式
**预训练（纯文本）** `dataset/pretrain_t2t.jsonl`，每行一条：
```json
{"text": "……一段中文或英文语料……"}
```
**SFT / 对话** `dataset/sft_t2t.jsonl`：
```json
{"conversations": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}]}
```
**多模态** `dataset/sft_i2t.parquet`（或 jsonl）：对话中插入图像占位符 `<|image_pad|>`，并附带图像字段（读取逻辑见 `dataset/lm_dataset.py`）。

> ⚠️ 仓库**不包含数据集**（`.gitignore` 排除了 `dataset/*.jsonl`、`*.parquet`、`tokens_*`）。
> 想快速验证流程，可以先用 1–2 万条任意中文语料跑通，再换全量。

### 3.2 清洗（可选但建议）
```bash
python scripts/data_clean.py        # 通用清洗
python scripts/clean.py             # 去重 / 过滤
python scripts/clean_sft_identity.py  # SFT 身份相关清理
```

### 3.3 预分词 + 分片（大语料推荐）
```bash
python scripts/pretokenize.py \
  --data dataset/pretrain_t2t.jsonl \
  --tok model_tok24k \
  --out dataset/tokens_full_24k \
  --docs_per_shard 200000 --workers 8
```
产出物直接喂给 `--shard_dir`，训练时自动启用 **memmap 装箱**，不需要把 GB 级语料读进内存。

### 3.4 词表选择
| 词表 | 位置 | 说明 |
| --- | --- | --- |
| 6.4k（默认） | `model/tokenizer.json` | 仓库自带，开箱即用 |
| 24k | `model_tok24k/` | 压缩比更高、seq340 截断率更低，但仓库未包含，需自行训练（`scripts/train_vocab24k.py` + `scripts/vocab_probe.py` 评估） |

> 换词表必须同时改 `--tokenizer_path` 与 `--vocab_size`，且**不要混用**不同词表的权重。

---

## 4. 下载 / 构建视觉编码器（VLM 必需）

代码用 `SiglipVisionModel.from_pretrained(path)` 加载，所以目录里只要有
`config.json + model.safetensors + preprocessor_config.json` 即可。

**方式 1：从公开权重裁剪（推荐，可复现）**
```python
# tools/make_siglip2_ve.py
from transformers import SiglipModel, SiglipImageProcessor
SRC = "google/siglip2-base-patch32-256"
DST = "model/siglip2-base-p32-256-ve"

m = SiglipModel.from_pretrained(SRC)
m.vision_model.half().save_pretrained(DST)          # 只保留 Vision Encoder，fp16
SiglipImageProcessor.from_pretrained(SRC).save_pretrained(DST)
print("saved ->", DST)
```
```bash
python tools/make_siglip2_ve.py
# 目录应包含: config.json / model.safetensors / preprocessor_config.json
```

**方式 2：使用作者转换好的 ModelScope 版本**（与方式 1 等价，目录名保持
`model/siglip2-base-p32-256-ve`）。仓库内 `model/siglip2-base-p32-256-ve/README.md`
记录了来源与裁剪方式。

> 该目录已被 `.gitignore` 排除（`model.safetensors` 约 180 MB），所以克隆后必须自己放进去。

---

## 5. 阶段一：预训练（纯文本 LLM）

### Docker
```bash
./run_docker.sh build
./run_docker.sh pretrain            # 等价于 MODEL_TYPE=llm docker compose up -d pretrain
./run_docker.sh logs pretrain
```

### 本地
```bash
cd trainer
python train_pretrain.py --model_type llm \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_dir ../out --save_weight pretrain \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 340 \
  --batch_size 32 --accumulation_steps 8 --learning_rate 5e-4 \
  --lr_schedule cosine --optimizer adamw --dtype bfloat16
```

**产物**：`out/pretrain_768.pth`（命名规则 `{save_dir}/{prefix}_{hidden_size}{_moe}.pth`）

**想复现 A/B 里的"新配置"**（吞吐 21,188 tok/s）：
```bash
python train_pretrain.py --use_moe 1 --num_experts 4 --num_experts_per_tok 2 \
  --n_shared_experts 1 --moe_impl permute --balance_mode aux_free \
  --optimizer muon_batched --muon_lr 0.02 --packing 1 --batch_size 32
```

---

## 6. 阶段二：VLM 预训练

```bash
# 容器
MODEL_TYPE=vlm docker compose up -d pretrain && docker compose logs -f pretrain
# 或
./run_docker.sh pretrain_vlm
```
```bash
# 本地
cd trainer
python train_pretrain.py --model_type vlm \
  --data_path ../dataset/pretrain_i2t.parquet \
  --batch_size 16 --freeze_llm 2 --save_weight pretrain_vlm
```
**产物**：`out/pretrain_vlm_768.pth`

---

## 7. 阶段三：SFT（多模态指令微调）

```bash
MODEL_TYPE=vlm docker compose up -d sft
# 或
cd trainer && python train_sft.py --model_type vlm \
  --data_path ../dataset/sft_i2t.parquet \
  --from_weight pretrain_vlm --save_weight sft_vlm --freeze_llm 1
```
**产物**：`out/sft_vlm_768.pth`（A/B 报告里 DPO 的基线就是 `full_sft_768_moe.pth`）

---

## 8. 阶段四：偏好对齐（DPO / GRPO）

**DPO**（数据里成对出现 chosen / rejected）：
```bash
python trainer/train_dpo.py --model_type llm \
  --data_path ../dataset/dpo_train.jsonl \
  --from_weight full_sft --save_weight dpo
# 或直接跑封装脚本（SFT → DPO → 评测一条龙）
python scripts/run_dpo_after_sft.py
```
**GRPO**（可验证奖励：数学/代码等有标准答案的任务）：
```bash
python trainer/train_grpo.py --model_type llm --data_path ../dataset/rlaif.jsonl --save_weight grpo
```
**LoRA / 蒸馏 / Agent**：
```bash
python trainer/train_lora.py    --model_type llm --data_path ../dataset/lora_medical.jsonl
python trainer/train_distill.py --model_type llm
python trainer/train_agent.py   --model_type llm --data_path ../dataset/agent_rl.jsonl
```

**对齐效果自查**：
```bash
python scripts/eval_dpo.py            # 单次评测
python scripts/eval_after_dpo.py      # SFT / DPO1 / DPO2 三档对比
python scripts/dpo_ab.py              # DPO 数据配比 A/B
```

---

## 9. 验证与评测

```bash
# 1) 全链路冒烟（形状、前向、保存加载）
python scripts/smoke_test.py

# 2) VLM 对话 / 评测（从原生 .pth 加载）
python eval_vlm.py --load_from model --weight sft_vlm --hidden_size 768 --num_hidden_layers 8 --use_moe 0

# 3) 命令行对话
python scripts/chat_test.py --ckpt out/sft_vlm_768.pth

# 4) 网页 Demo（Gradio）
python scripts/web_demo_vlm.py --load_from ./ --vision_model ../model/siglip2-base-p32-256-ve

# 5) 转 HuggingFace 格式（部署 / 二次开发）
python scripts/convert_vlm.py
# 产物：out/{weight}_{hidden_size}[_moe]/ 下的 safetensors + tokenizer
```

**复现 AB_REPORT 的数字**：
```bash
python scripts/ab_bench.py --suite speed   --tokens 3000000 --repeats 3 --seed 42
python scripts/ab_bench.py --suite quality --wall_seconds 420
# 结果写入 scripts/ab_results.json，与 scripts/AB_REPORT.md 对照
```

---

## 10. 资源参考（来自本仓库实测，bs16 / 单卡）

| 配置 | ms/step | 真实 tok/s | 显存 |
| --- | --- | --- | --- |
| base | 335 | 9,320 | 4.4 GB |
| MoE permute 调度 | 275 | 11,348 | 4.4 GB |
| packed | 320 | 16,987 | 4.4 GB |
| 生产配置 bs32（旧） | 517 | 12,075 | 8.8 GB |
| **生产配置 bs32（stack）** | **515** | **21,188** | **6.2 GB** |

> 时间量级：300 万真实 token 的对照实验约 7 分钟；全量预训练按数据规模线性放大。

---

## 11. 常见问题（踩坑记录）

**Q1. 显存不够 / OOM**
依次尝试：① 降 `--batch_size`、提高 `--accumulation_steps` 保持等效 batch；② 开 `--packing 1`（吞吐高、显存不变）；③ 开梯度检查点（代码内可配）；④ `--dtype bfloat16`（默认）；⑤ VLM 阶段降 `--batch_size 16` 甚至 8。

**Q2. 为什么 `packed` 在等 token 预算下反而变差？**
装箱会改变样本分布与 padding 比例：等 token 数下 packed 的实际训练步数更少（552 vs 962 步），学习率调度没走完，所以 val loss 偏高；但**在等墙钟时间口径下 packed 大幅领先**（−0.37）。做决策要看等时间口径，报告里有完整讨论。

**Q3. Muon 怎么用？**
`--optimizer muon`（逐参数更新）或 `muon_batched`（批量 Newton–Schulz，更快）。Muon **只作用于隐层 2D 权重矩阵**，embedding / norm / 输出头仍走 AdamW；两组学习率分别由 `--muon_lr`（默认 0.02）与 `--learning_rate` 控制。注意它在等 token 下很强（−0.70），但每步更慢，要用等时间口径评估。

**Q4. MoE 开启了但没省显存？**
MoE 省的是**计算量**不是显存：所有专家参数都要驻留显存。top-2 × 半宽专家把参数量从 140M 降到 80.4M，才是显存/吞吐收益的来源（见报告表格）。

**Q5. `aux` 和 `aux_free` 怎么选？**
本仓库实测 aux-free 与 aux 在 300 万 token 规模下差异在噪声内（−0.01）；aux-free 不污染主损失、超参更少，长训更省心。

**Q6. flash-attn 装不上？**
不影响主流程（会退化到 PyTorch SDPA，速度略降）。编译需与本机 CUDA、torch 版本严格匹配；用 Dockerfile 可跳过该问题。

**Q7. 多卡没加速 / NCCL 报错**
先确认 `GPU_COUNT`；Infiniband 环境按 `.env` 设 `NCCL_IB_DISABLE=0`；容器内共享内存已在 compose 里设为 `shm_size: 16gb`（dat loader / NCCL 需要）。

**Q8. 权重文件对不上 / load_state_dict 报错**
命名必须严格匹配 `{prefix}_{hidden_size}{_moe}.pth`：dense → `out/pretrain_768.pth`，MoE → `out/pretrain_768_moe.pth`。用 MoE 权重时命令里要带 `--use_moe 1`，反之亦然。

**Q9. 能直接用 HuggingFace 生态加载吗？**
可以：先 `python scripts/convert_vlm.py` 转成 transformers 格式，再用 `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` 加载（`eval_vlm.py` 支持 `--load_from` 指向该目录）。

---

## 12. 最小可复现路径（30 分钟版）

只想确认"这套代码能跑"：
```bash
python scripts/smoke_test.py                                   # 模型/数据/前向 全绿
python trainer/train_pretrain.py --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --max_seq_len 340 --batch_size 8 --accumulation_steps 1 --epochs 1 --save_interval 50
python scripts/ab_bench.py --suite speed --tokens 300000 --repeats 1
```
三条命令分别验证：**实现正确性 → 训练闭环 → 组件对比可用性**。
