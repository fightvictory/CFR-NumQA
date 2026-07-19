#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误归因脚本：把答案文件中每条记录归入细分错误类别（无GPU也能跑）
    python attribute_errors.py data/answers_structural.jsonl --dump

类别体系（论文错误分析表用）：
  correct              答对
  拒答类:
    abstain_miss       拒答且gold数值不在检索资料中 —— 检索未命中，合理拒答
    abstain_over       拒答但gold数值就在检索资料中 —— 过度拒答（生成端问题）
  同比题错误:
    yoy_sign_flip      幅度对但正负号反 —— 增减方向判断错
    yoy_arithmetic     两个gold操作数都在资料中，但算出的结果错 —— 纯算术错误
    yoy_wrong_operand  结果可由资料中某两个数算出，但不是gold操作数 —— 取错数
    yoy_miss           gold操作数不全在资料中 —— 检索缺失导致
    yoy_no_number      预测中无可解析数值
  抽取题错误:
    ext_unit_omit      数字串与gold完全一致但单位缺失/写错 —— 格式遗漏（非取错数）
    ext_unit_error     数值对但量纲差10^k —— 单位/量纲换算错
    ext_wrong_pos      预测数值在资料中但不是gold —— 取错位置
    ext_fabrication    预测数值不在资料中 —— 凭空捏造
    ext_no_number      预测中无可解析数值
  对比题错误:
    cmp_wrong_company  答成了另一家公司
    cmp_other          其他（两家都提/格式无法判定）
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict

from eval_answers import (ABSTAIN_RE, NUM_RE, extract_number, is_correct,
                          numbers_in_context)


def ctx_values(rec):
    """检索资料中的数值集合（float）。"""
    vals = set()
    for s in numbers_in_context(rec):
        try:
            vals.add(float(s))
        except ValueError:
            pass
    return vals


def in_ctx(v, ctx, tol=1e-6):
    return any(abs(v - c) <= tol * max(1, abs(v)) for c in ctx)


def gold_values(rec):
    """gold证据数值：yoy用meta.values两个操作数，extraction用meta.value。"""
    m = rec.get("meta", {})
    if rec["type"] == "yoy_compare":
        return [float(v) for v in m.get("values", [])]
    if m.get("value") is not None:
        return [float(m["value"])]
    g = extract_number(rec["gold"])
    return [g] if g is not None else []


def pred_percent(rec):
    """按eval_answers同款逻辑解析预测百分数（含'下降'语义取负）。"""
    p = extract_number(rec["prediction"])
    if p is None or abs(p) > 1000:
        return None
    if re.search(r"下降|减少|降低", rec["prediction"]) and p > 0:
        p = -p
    return p


def derivable_pairs(p, ctx):
    """预测百分数p能否由资料中两数(a-b)/|b|*100得到，返回命中的(a,b)对。"""
    pairs = []
    vals = [v for v in ctx if abs(v) > 1]
    for a in vals:
        for b in vals:
            if b and abs((a - b) / abs(b) * 100 - p) < 0.06:
                pairs.append((a, b))
    return pairs


def classify(rec):
    ctx = ctx_values(rec)
    golds = gold_values(rec)
    gold_in_ctx = bool(golds) and all(in_ctx(v, ctx) for v in golds)

    if ABSTAIN_RE.search(rec["prediction"]):
        return "abstain_over" if gold_in_ctx else "abstain_miss"
    if is_correct(rec):
        return "correct"

    t = rec["type"]
    if t == "yoy_compare":
        g = float(rec["gold"].rstrip("%"))
        p = pred_percent(rec)
        if p is None:
            return "yoy_no_number"
        if abs(abs(p) - abs(g)) < 0.06 and (p > 0) != (g > 0):
            return "yoy_sign_flip"
        if derivable_pairs(p, ctx):
            return "yoy_wrong_operand"
        if gold_in_ctx:
            return "yoy_arithmetic"
        return "yoy_miss"

    if t == "extraction":
        gv, pv = extract_number(rec["gold"]), extract_number(rec["prediction"])
        if pv is None:
            return "ext_no_number"
        # 数字串与gold完全一致但单位缺失/写错 —— 格式遗漏而非取错数
        mg = NUM_RE.search(rec["gold"].replace(" ", ""))
        mp = NUM_RE.search(rec["prediction"].replace(" ", ""))
        if mg and mp and mg.group().replace(",", "") == mp.group().replace(",", ""):
            return "ext_unit_omit"
        if gv:
            ratio = abs(pv) / abs(gv) if gv else 0
            if ratio > 0:
                log = math.log10(ratio)
                if abs(log - round(log)) < 0.01 and round(log) != 0:
                    return "ext_unit_error"
        # 预测的原始数值串是否在资料中
        m = NUM_RE.search(rec["prediction"].replace(" ", ""))
        if m and m.group().replace(",", "") in numbers_in_context(rec):
            return "ext_wrong_pos"
        return "ext_fabrication"

    # cross_company
    companies = rec.get("meta", {}).get("companies", [])
    others = [c for c in companies if c != rec["gold"]]
    if any(c in rec["prediction"] for c in others) and rec["gold"] not in rec["prediction"]:
        return "cmp_wrong_company"
    return "cmp_other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers_file")
    ap.add_argument("--dump", action="store_true", help="打印每个类别的样例")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.answers_file, encoding="utf-8")]
    by_cat = defaultdict(list)
    by_type = defaultdict(Counter)
    for r in recs:
        cat = classify(r)
        by_cat[cat].append(r)
        by_type[r["type"]][cat] += 1

    n = len(recs)
    print(f"\n== {args.answers_file}（{n}条） ==")
    order = ["correct", "abstain_miss", "abstain_over",
             "yoy_sign_flip", "yoy_arithmetic", "yoy_wrong_operand",
             "yoy_miss", "yoy_no_number",
             "ext_unit_omit", "ext_unit_error", "ext_wrong_pos", "ext_fabrication",
             "ext_no_number",
             "cmp_wrong_company", "cmp_other"]
    for cat in order:
        if cat in by_cat:
            print(f"  {cat:<20} {len(by_cat[cat]):>3}  ({len(by_cat[cat])/n:.1%})")

    print("\n  分题型：")
    for t, cnt in sorted(by_type.items()):
        tn = sum(cnt.values())
        detail = "  ".join(f"{c}={v}" for c, v in cnt.most_common())
        print(f"  {t:<14} n={tn:<4} {detail}")

    if args.dump:
        for cat in order:
            if cat in ("correct",) or cat not in by_cat:
                continue
            print(f"\n-- {cat} --")
            for r in by_cat[cat][:5]:
                print(f"  Q: {r['question']}")
                print(f"     gold={r['gold']}  pred={r['prediction'][:70]}")


if __name__ == "__main__":
    main()
