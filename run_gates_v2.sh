#!/bin/bash
# 用重训后的校验器(v2适配器)重跑全部门控；第一轮附带分类指标，其余用空测试集跳过。
set -e
cd ~/finrag/pilot_toolkit
source ~/finrag/venv/bin/activate
export HF_HUB_OFFLINE=1
M="Qwen/Qwen2.5-7B-Instruct"
L="models/verifier_lora_v2"
: > data/_empty_test.jsonl

run () {  # run <测试集> <答案文件> <输出>
  echo "=== GATE $3 ==="
  python eval_verifier.py "$1" --model "$M" --lora "$L" --backend hf \
         --gate "data/$2.jsonl" --dump-gate "data/$3.jsonl"
}

run data/verifier/test.jsonl answers_v2_structural_full gate_new_v2
run data/_empty_test.jsonl   answers_v3_full            gate_new_v3
run data/_empty_test.jsonl   answers_14b_v3_full        gate_new_14b
run data/_empty_test.jsonl   answers_32b_v3_full        gate_new_32b
run data/_empty_test.jsonl   answers_m3_v3ctx           gate_new_m3
echo "=== ALL GATES DONE ==="
