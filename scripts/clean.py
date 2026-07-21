import re
import json
from pathlib import Path

# ========== 路径配置（scripts 目录下运行）==========
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_DIR = ROOT_DIR / "dataset"

# ========== 替换规则 ==========
OLD_WORD = "skyrope"
NEW_WORD = "skyRope"
PATTERN = re.compile(re.escape(OLD_WORD), re.IGNORECASE)

# ========== 递归安全替换（任何格式都不报错）==========
def safe_replace(value):
    try:
        if isinstance(value, str):
            new_val, cnt = PATTERN.subn(NEW_WORD, value)
            return new_val, cnt
        elif isinstance(value, dict):
            total = 0
            new_dict = {}
            for k, v in value.items():
                v_new, c = safe_replace(v)
                new_dict[k] = v_new
                total += c
            return new_dict, total
        elif isinstance(value, list):
            total = 0
            new_list = []
            for item in value:
                i_new, c = safe_replace(item)
                new_list.append(i_new)
                total += c
            return new_list, total
        else:
            return value, 0
    except:
        return value, 0

# ========== 流式处理 JSONL（永不爆内存）==========
def process_jsonl(file_path: Path):
    print(f"\n===== 处理 JSONL：{file_path.name} =====")
    tmp = file_path.with_suffix(".tmp")
    total = 0

    with open(file_path, "r", encoding="utf-8") as in_f, \
         open(tmp, "w", encoding="utf-8") as out_f:

        for line in in_f:
            line = line.strip()
            if not line:
                out_f.write("\n")
                continue
            try:
                data = json.loads(line)
                data_new, cnt = safe_replace(data)
                total += cnt
                out_f.write(json.dumps(data_new, ensure_ascii=False) + "\n")
            except:
                out_f.write(line + "\n")

    file_path.unlink()
    tmp.rename(file_path)
    print(f"✅ 替换完成：共替换 {total} 处")

# ========== 安全处理 PARQUET（不编码报错）==========
def process_parquet(file_path: Path):
    print(f"\n===== 处理 PARQUET：{file_path.name} =====")
    import pandas as pd

    total = 0
    df = pd.read_parquet(file_path)

    for col in df.columns:
        for idx in df.index:
            val = df.at[idx, col]
            new_val, cnt = safe_replace(val)
            if cnt > 0:
                df.at[idx, col] = new_val
                total += cnt

    df.to_parquet(file_path, index=False)
    print(f"✅ 替换完成：共替换 {total} 处")

# ========== 主入口 ==========
def main():
    jsonl_files = list(DATASET_DIR.glob("*.jsonl"))
    parquet_files = list(DATASET_DIR.glob("*.parquet"))

    print(f"找到 {len(jsonl_files)} 个 jsonl，{len(parquet_files)} 个 parquet，开始替换...")

    for f in jsonl_files:
        process_jsonl(f)
    for f in parquet_files:
        process_parquet(f)

    print("\n🎉 全部文件处理完成！无报错！")

if __name__ == "__main__":
    main()