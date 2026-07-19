#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
答案评测脚本：准确率 + 数值幻觉率（无GPU也能跑）
    python eval_answers.py data/answers_structural.jsonl
    python eval_answers.py data/answers_naive.jsonl

指标定义（对齐计划书5.4节）：
  Accuracy    答案正确（数值题：量纲归一后相对误差<0.5%；比较题：公司名正确）
  Abstain     模型回答"无法从资料中确定"（拒答，不算幻觉）
  HAL幻觉率   答案错误 且 预测数值在检索到的资料中不存在（凭空捏造）
              ——参考FinReflectKG-HalluBench定义
  GroundedErr 答案错误 但 数值来自资料（检索到了错的位置/推理错，非捏造）
"""
import argparse
import json
import re

UNIT_SCALE = {"元": 1, "千元": 1e3, "万元": 1e4, "百万元": 1e6, "亿元": 1e8}
NUM_RE = re.compile(r"-?[\d,]+\.?\d*")
ABSTAIN_RE = re.compile(r"无法从资料中确定|无法确定|资料中(未|没有)")


def extract_number(text):
    """从文本中提取第一个数值及其量纲，返回换算成元/原值的浮点数。"""
    m = NUM_RE.search(text.replace(" ", ""))
    if not m:
        return None
    val = float(m.group().replace(",", ""))
    tail = text[m.end():m.end() + 4]
    for unit, scale in sorted(UNIT_SCALE.items(), key=lambda x: -len(x[0])):
        if tail.startswith(unit):
            return val * scale
    return val


def is_correct(rec):
    pred, gold, t = rec["prediction"], rec["gold"], rec["type"]
    if t == "cross_company":
        # 只要求答对公司名（且不能两个都说）
        companies = rec["meta"].get("companies", [])
        others = [c for c in companies if c != gold]
        return gold in pred and not any(c in pred.replace(gold, "") for c in others)
    if t == "yoy_compare":
        g = float(gold.rstrip("%"))
        # 匹配预测中的百分数（允许写成 -9.7% 或 下降9.70%）
        p = extract_number(pred)
        if p is None:
            return False
        if abs(p) > 1000:  # 明显不是百分数
            return False
        if re.search(r"下降|减少|降低", pred) and p > 0:
            p = -p
        return abs(p - g) < 0.06 or abs(abs(p) - abs(g)) < 0.06 and (p > 0) == (g > 0)
    # extraction：量纲归一后相对误差<0.5%
    gv, pv = extract_number(gold), extract_number(pred)
    if gv is None or pv is None:
        return False
    if gv == 0:
        return pv == 0
    return abs(pv - gv) / abs(gv) < 0.005


def numbers_in_context(rec):
    """检索资料中出现的所有数值（原始字符串规格化）。"""
    nums = set()
    for u in rec["retrieved"]:
        for m in NUM_RE.finditer(u["text"]):
            s = m.group().replace(",", "")
            if len(s.replace(".", "").replace("-", "")) >= 3:  # 忽略页码等短数字
                nums.add(s)
    return nums


def is_grounded(rec):
    """预测中的关键数值是否能在检索资料中找到（原样，忽略千分位）。"""
    m = NUM_RE.search(rec["prediction"].replace(" ", ""))
    if not m:
        return True  # 无数值输出（如公司名），不判捏造
    pred_num = m.group().replace(",", "")
    ctx_nums = numbers_in_context(rec)
    if pred_num in ctx_nums:
        return True
    # 计算题：结果由资料数值算出，检查是否可由任意两个资料数值组合得到（增长率）
    if rec["type"] == "yoy_compare":
        try:
            p = float(pred_num)
        except ValueError:
            return False
        vals = [float(s) for s in ctx_nums if abs(float(s)) > 1]
        for a in vals:
            for b in vals:
                if b and abs((a - b) / abs(b) * 100 - p) < 0.06:
                    return True
        return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers_file")
    ap.add_argument("--dump-errors", action="store_true", help="打印错误样例")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.answers_file, encoding="utf-8")]
    from collections import defaultdict
    stats = defaultdict(lambda: {"n": 0, "acc": 0, "abstain": 0, "hal": 0, "grounded_err": 0})

    for r in recs:
        for key in ("ALL", r["type"]):
            s = stats[key]
            s["n"] += 1
            if ABSTAIN_RE.search(r["prediction"]):
                s["abstain"] += 1
            elif is_correct(r):
                s["acc"] += 1
            elif is_grounded(r):
                s["grounded_err"] += 1
            else:
                s["hal"] += 1
                if args.dump_errors and key == "ALL":
                    print(f"[幻觉] {r['question']}\n  gold={r['gold']}  pred={r['prediction'][:80]}")

    print(f"\n{'':>14} {'n':>4} {'准确率':>8} {'拒答率':>8} {'HAL幻觉率':>9} {'有依据错误':>9}")
    for key in ["ALL"] + sorted(k for k in stats if k != "ALL"):
        s = stats[key]
        n = s["n"]
        print(f"{key:>14} {n:>4} {s['acc']/n:>8.1%} {s['abstain']/n:>8.1%} "
              f"{s['hal']/n:>9.1%} {s['grounded_err']/n:>9.1%}")


if __name__ == "__main__":
    main()
