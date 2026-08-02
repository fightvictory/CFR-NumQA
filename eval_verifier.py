#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校验器评测（vLLM推理，支持零样本基线与LoRA适配器）。

  # 零样本基线（AWQ推理模型即可）
  python eval_verifier.py data/verifier/test.jsonl --model Qwen/Qwen2.5-7B-Instruct-AWQ

  # 训练后的校验器
  python eval_verifier.py data/verifier/test.jsonl \
      --model Qwen/Qwen2.5-7B-Instruct --lora models/verifier_lora

  # 端到端拦截模拟（对答案文件里test公司的真实预测做gate）
  加 --gate results/answers_v2_structural_full.jsonl

指标：判定准确率 / 正例通过率 / 负例拦截率 / 分错误类型拦截率；
gate模式：拦截前后 已答题准确率、幻觉率、有效拒答率。
"""
import argparse
import json
from collections import Counter

from build_verifier_data import (PROMPT_TMPL, TEST_COMPANIES, ctx_text,
                                 rec_company)
from eval_answers import ABSTAIN_RE, is_correct, is_grounded


def parse_verdict(text):
    t = text.strip()
    if t.startswith("不支持") or "不支持" in t[:6]:
        return "不支持"
    if "支持" in t[:6]:
        return "支持"
    return "不支持"  # 无法解析时保守拦截


def run_llm(prompts, args):
    if args.backend == "hf":
        return run_hf(prompts, args)
    from vllm import LLM, SamplingParams
    kw = {}
    if args.lora:
        kw = {"enable_lora": True, "max_lora_rank": 16}
    # 非量化 7B 权重 14.19 GiB，16 GiB 卡上 0.85 利用率留不出 KV cache（引擎启动即失败）。
    # 而适配器的 adapter_config 记录的基座正是非量化版；用 -AWQ 会让全部指标偏低
    # 约 3.5 pp（2026-08-01 实测），所以不能退回量化版了事——提高利用率才是正解。
    # enforce_eager：非量化基座下 torch inductor 的自动调优还要额外 1 GiB，
    # 14.19 GiB 权重 + 15.54 GiB 卡腾不出来。关掉图编译换取显存；本评测仅
    # 715 条，慢一点无所谓。
    llm = LLM(model=args.model, max_model_len=2560,
              gpu_memory_utilization=args.gpu_util,
              enforce_eager=args.eager, **kw)
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    convs = [[{"role": "user", "content": p}] for p in prompts]
    if args.lora:
        from vllm.lora.request import LoRARequest
        outs = llm.chat(convs, sp, lora_request=LoRARequest("verifier", 1, args.lora))
    else:
        outs = llm.chat(convs, sp)
    return [parse_verdict(o.outputs[0].text) for o in outs]


def run_hf(prompts, args):
    """transformers 4bit后端：16GB显存跑全量7B+LoRA的稳妥路径（与训练同款加载）。"""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True),
        device_map={"": 0}, attn_implementation="sdpa")
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    verdicts = []
    B = 8
    for i in range(0, len(prompts), B):
        batch = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False)
                 for p in prompts[i:i + B]]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            text = tok.decode(out[j][enc["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            verdicts.append(parse_verdict(text))
        if (i // B) % 10 == 0:
            print(f"  hf eval {i + len(batch)}/{len(prompts)}", flush=True)
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_file")
    # 默认基座必须与训练一致：适配器是 QLoRA 在 4bit NF4 上训的（--backend hf 那条
    # 路径）。先前默认 -AWQ 是另一种 4bit 方案，全部指标偏低约 3.5 pp 且不报任何错，
    # 照默认参数复现的人会以为论文数字造假（2026-08-01 实测发现）。
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--gate", default=None, help="答案文件：对test公司真实预测做拦截模拟")
    ap.add_argument("--gpu-util", type=float, default=0.85,
                    help="显存利用率；非量化基座在 16 GiB 卡上需 0.93 以上")
    ap.add_argument("--eager", action="store_true",
                    help="关闭 CUDA 图/编译以省显存；非量化基座在 16 GiB 卡上必需")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="hf")
    ap.add_argument("--dump-test", default=None,
                    help="把测试集判定明细写入 jsonl，供 tab:verifier 正面对账")
    ap.add_argument("--dump-gate", default=None, help="把gate明细写入jsonl（含verdict/correct/grounded）")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test_file, encoding="utf-8")]
    prompts = [r["prompt"] for r in rows]

    gate_recs = []
    if args.gate:
        for l in open(args.gate, encoding="utf-8"):
            rec = json.loads(l)
            comp = set(rec_company(rec).split("+"))
            if comp & TEST_COMPANIES and not ABSTAIN_RE.search(rec["prediction"]):
                gate_recs.append(rec)
                prompts.append(PROMPT_TMPL.format(
                    ctx=ctx_text(rec), q=rec["question"], a=rec["prediction"]))

    verdicts = run_llm(prompts, args)
    v_test, v_gate = verdicts[:len(rows)], verdicts[len(rows):]

    # ---- 校验器本身 ----
    n = len(rows)
    if n == 0:
        pass
    else:
        if args.dump_test:
            with open(args.dump_test, "w", encoding="utf-8") as fh:
                for r, v in zip(rows, v_test):
                    fh.write(json.dumps({
                        "qa_id": r.get("qa_id"), "type": r.get("type"),
                        "kind": r.get("kind"), "label": r["label"], "verdict": v,
                    }, ensure_ascii=False) + "\n")
        _report_cls(rows, v_test, args)

    # ---- 端到端拦截模拟 ----
    if gate_recs:
        if args.dump_gate:
            with open(args.dump_gate, "w", encoding="utf-8") as fh:
                for r, v in zip(gate_recs, v_gate):
                    fh.write(json.dumps({
                        "id": r["id"], "type": r["type"], "question": r["question"],
                        "gold": r["gold"], "prediction": r["prediction"],
                        "verdict": v, "correct": is_correct(r),
                        "grounded_pred": is_grounded(r),
                    }, ensure_ascii=False) + "\n")
        _report_gate(gate_recs, v_gate)


def _report_cls(rows, v_test, args):
    n = len(rows)
    ok = sum(1 for r, v in zip(rows, v_test) if v == r["label"])
    pos = [(r, v) for r, v in zip(rows, v_test) if r["label"] == "支持"]
    neg = [(r, v) for r, v in zip(rows, v_test) if r["label"] == "不支持"]
    print(f"\n== 校验器判定（{args.test_file}, n={n}) ==")
    print(f"总准确率      {ok/n:.1%}")
    print(f"正例通过率    {sum(v=='支持' for _,v in pos)/max(1,len(pos)):.1%}  (n={len(pos)})")
    print(f"负例拦截率    {sum(v=='不支持' for _,v in neg)/max(1,len(neg)):.1%}  (n={len(neg)})")
    kinds = Counter()
    kind_ok = Counter()
    for r, v in neg:
        kinds[r["kind"]] += 1
        kind_ok[r["kind"]] += (v == "不支持")
    print("分错误类型拦截率：")
    for k, c in kinds.most_common():
        print(f"  {k:<14} {kind_ok[k]/c:>6.1%}  (n={c})")

def _report_gate(gate_recs, v_gate):
    def stats(recs_kept):
        nn = len(recs_kept)
        acc = sum(is_correct(r) for r in recs_kept)
        hal = sum((not is_correct(r)) and (not is_grounded(r)) for r in recs_kept)
        return nn, acc, hal
    n0, acc0, hal0 = stats(gate_recs)
    kept = [r for r, v in zip(gate_recs, v_gate) if v == "支持"]
    blocked_correct = sum(is_correct(r) for r, v in zip(gate_recs, v_gate)
                          if v == "不支持")
    n1, acc1, hal1 = stats(kept)
    print(f"\n== 端到端拦截模拟（test公司非拒答预测 n={n0}）==")
    print(f"拦截前：已答准确率 {acc0/n0:.1%}，幻觉 {hal0} 条")
    if n1:
        print(f"拦截后：保留 {n1} 条，已答准确率 {acc1/n1:.1%}，幻觉 {hal1} 条")
    print(f"误拦（正确答案被拦）: {blocked_correct}/{acc0} = {blocked_correct/max(1,acc0):.1%}")
    print(f"拦截命中（错误答案被拦）: {(n0-n1)-blocked_correct}/{n0-acc0} = "
          f"{((n0-n1)-blocked_correct)/max(1,n0-acc0):.1%}")


if __name__ == "__main__":
    main()
