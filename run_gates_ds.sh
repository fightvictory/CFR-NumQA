#!/bin/bash
# 对 DeepSeek-V4-Pro 在我们v3上下文下的答案做门控，v1/v2 两个校验器各一次。
# 用途：把"验证器跨模型族迁移"从单一异构生成器扩展到两个不同厂商。
set -e
cd ~/finrag/pilot_toolkit
source ~/finrag/venv/bin/activate
export HF_HUB_OFFLINE=1
M="Qwen/Qwen2.5-7B-Instruct"
: > data/_empty_test.jsonl

for v in 1 2; do
  [ "$v" = "1" ] && L="models/verifier_lora" || L="models/verifier_lora_v2"
  echo "=== GATE deepseek v$v ($L) ==="
  python eval_verifier.py data/_empty_test.jsonl --model "$M" --lora "$L" --backend hf \
      --gate data/answers_ds_v3ctx.jsonl --dump-gate data/gate_ds_v$v.jsonl
done
echo "=== DEEPSEEK GATES DONE ==="
