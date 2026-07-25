#!/bin/bash
# 对 DeepSeek-V4-Pro 在我们v3上下文下的答案做门控，v1/v2 两个校验器各一次。
# 用途：把"验证器跨模型族迁移"从单一异构生成器扩展到两个不同厂商。
# 运行前：先激活你的 Python 环境（如 source venv/bin/activate），并把两个适配器解好
#   unzip verifier_lora_v1.0.0.zip -d models/   # -> models/verifier_lora
#   unzip verifier_lora_v2.0.0.zip -d models/   # -> models/verifier_lora_v2
# 基座模型已在本地缓存时可 export HF_HUB_OFFLINE=1 走离线。
set -e
cd "$(dirname "$0")"
M="Qwen/Qwen2.5-7B-Instruct"

# 复现产物默认另存，避免覆盖随仓库发布的参考输出；用 OUT_DIR=results 可就地覆盖
OUT="${OUT_DIR:-results/repro}"
mkdir -p "$OUT"
: > "$OUT/_empty_test.jsonl"

for v in 1 2; do
  [ "$v" = "1" ] && L="models/verifier_lora" || L="models/verifier_lora_v2"
  echo "=== GATE deepseek v$v ($L) ==="
  python eval_verifier.py "$OUT/_empty_test.jsonl" --model "$M" --lora "$L" --backend hf \
      --gate results/answers_ds_v3ctx.jsonl --dump-gate "$OUT/gate_ds_v$v.jsonl"
done
echo "=== DEEPSEEK GATES DONE ==="
