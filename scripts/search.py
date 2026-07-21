import os
import json
import pandas as pd
from pathlib import Path

# ========= 配置 =========
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_PATH = SCRIPT_DIR / "skyrope_data.json"
TARGET_KEYWORD = "skyrope"
CASE_SENSITIVE = False

# ========= 初始化 =========
result = {}


# ========= 安全检查关键词（任何格式都不崩溃）=========
def safe_check_keyword(content, keyword, case_sensitive):
    try:
        # 递归处理字典
        if isinstance(content, dict):
            for v in content.values():
                if safe_check_keyword(v, keyword, case_sensitive):
                    return True
            return False

        # 递归处理列表
        elif isinstance(content, list):
            for item in content:
                if safe_check_keyword(item, keyword, case_sensitive):
                    return True
            return False

        # 处理字符串
        elif isinstance(content, str):
            if case_sensitive:
                return keyword in content
            else:
                return keyword.lower() in content.lower()

        # 其他类型：数字、bool、None → 不检查
        else:
            return False
    except:
        # 坏数据、乱码、二进制 → 直接跳过不报错
        return False


# ========= 主逻辑 =========
if not DATASET_DIR.exists():
    print(f"❌ 错误：找不到 dataset 目录 {DATASET_DIR}")
    exit(1)

print(f"项目根目录：{PROJECT_ROOT}")
print(f"开始扫描目录：{DATASET_DIR}")
print(f"目标关键词：{TARGET_KEYWORD}（大小写敏感：{CASE_SENSITIVE}）\n")

# ====================== 处理 JSONL ======================
jsonl_files = list(DATASET_DIR.glob("*.jsonl"))
for file_path in jsonl_files:
    if not file_path.is_file():
        continue

    print(f"正在处理 JSONL：{file_path.name}")
    line_count = 0
    hit_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                continue  # 坏行直接跳过

            if safe_check_keyword(data, TARGET_KEYWORD, CASE_SENSITIVE):
                key = f"{file_path.name}:JSONL:第{line_count}条"
                result[key] = data
                hit_count += 1

    print(f"✅ 处理完成：{file_path.name}，命中 {hit_count} 条数据\n")

# ====================== 处理 PARQUET（修复编码崩溃）======================
parquet_files = list(DATASET_DIR.glob("*.parquet"))
for file_path in parquet_files:
    if not file_path.is_file():
        continue

    print(f"正在处理 PARQUET：{file_path.name}")
    hit_count = 0

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"⚠️ {file_path.name} 读取失败，跳过")
        continue

    # 逐行逐列安全检查
    for idx in df.index:
        row_hit = False

        for col in df.columns:
            try:
                val = df.at[idx, col]
                if safe_check_keyword(val, TARGET_KEYWORD, CASE_SENSITIVE):
                    row_hit = True
                    break
            except:
                continue  # 乱码/二进制直接跳过

        if row_hit:
            try:
                row_data = df.iloc[idx].to_dict()
                key = f"{file_path.name}:PARQUET:第{idx + 1}行"
                result[key] = row_data
                hit_count += 1
            except:
                continue

    print(f"✅ 处理完成：{file_path.name}，命中 {hit_count} 条数据\n")

# ========= 保存结果 =========
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("=" * 50)
print(f"扫描结束！总共命中 {len(result)} 条数据")
print(f"结果已保存到：{OUTPUT_PATH}")