#!/bin/bash
# 用重训后的校验器(v2适配器)重跑全部门控；第一轮附带分类指标，其余用空测试集跳过。
# 运行前：先激活你的 Python 环境（如 source venv/bin/activate），并把
#   unzip verifier_lora_v2.0.0.zip -d models/
# 解好；基座模型已在本地缓存时可 export HF_HUB_OFFLINE=1 走离线。
set -e
cd "$(dirname "$0")"
M="Qwen/Qwen2.5-7B-Instruct"
L="models/verifier_lora_v2"

# 复现产物默认另存，避免覆盖随仓库发布的参考输出；用 OUT_DIR=results 可就地覆盖
OUT="${OUT_DIR:-results/repro}"
mkdir -p "$OUT"
: > "$OUT/_empty_test.jsonl"

# 校验器的测试划分不随仓库发布（由构建脚本生成），缺失时从已归档的答案文件重建
[ -f data/verifier/test.jsonl ] || \
  python build_verifier_data.py results/answers_v2_structural_full.jsonl -o data/verifier/

run () {  # run <测试集> <答案文件> <输出>
  echo "=== GATE $3 ==="
  python eval_verifier.py "$1" --model "$M" --lora "$L" --backend hf \
         --gate "results/$2.jsonl" --dump-gate "$OUT/$3.jsonl"
}

run data/verifier/test.jsonl   answers_v2_structural_full gate_new_v2
run "$OUT/_empty_test.jsonl"   answers_v3_full            gate_new_v3
run "$OUT/_empty_test.jsonl"   answers_14b_v3_full        gate_new_14b
run "$OUT/_empty_test.jsonl"   answers_32b_v3_full        gate_new_32b
run "$OUT/_empty_test.jsonl"   answers_m3_v3ctx           gate_new_m3
echo "=== ALL GATES DONE ==="
