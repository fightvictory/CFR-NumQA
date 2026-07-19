#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨公司对比题badcase分析：区分检索失败 vs 模型比较错误（含单位盲区检查）
    python analyze_cmp.py data/answers_structural_unitfix_calc.jsonl
"""
import argparse
import json
import re

from eval_answers import ABSTAIN_RE, NUM_RE, UNIT_SCALE, is_correct

UNIT_AFTER_NUM = re.compile(r"(-?[\d,]+\.?\d*)\s*(亿元|百万元|万元|千元|元)?")


def numbers_with_scale(text):
    """提取文本中的(数值, 换算成元)对；无单位时按 千元/(千元)行标签 或 原值处理。"""
    out = []
    # 行标签自带单位如"营业收入（千元）"
    label_unit = None
    m = re.search(r"（(亿元|百万元|万元|千元|元)）|\((亿元|百万元|万元|千元|元)\)", text)
    if m:
        label_unit = m.group(1) or m.group(2)
    for mm in UNIT_AFTER_NUM.finditer(text.replace(" ", "")):
        try:
            v = float(mm.group(1).replace(",", ""))
        except ValueError:
            continue
        unit = mm.group(2) or label_unit
        out.append((v, v * UNIT_SCALE.get(unit, 1)))
    return out


def company_value_in_ctx(rec, company, gold_yuan):
    """该公司的gold数值（换算元后）是否出现在检索到的、来自该公司文档的单元里。"""
    for u in rec["retrieved"]:
        if company not in u["source"]:
            continue
        for _, yuan in numbers_with_scale(u["text"]):
            if gold_yuan and abs(yuan - gold_yuan) / abs(gold_yuan) < 0.01:
                return True
    return False


def units_differ(rec):
    """两家公司gold证据的计量单位是否不同（比较时需换算）。"""
    units = set()
    for ev in rec["gold_evidence"]:
        m = re.search(r"单位[:：]?\s*(?:人民币)?\s*(亿元|百万元|万元|千元|元)", ev.get("caption", ""))
        units.add(m.group(1) if m else re.search(r"（(千元|百万元)）", ev.get("row_label", "")) and "千元" or "元?")
    return len(units) > 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers_file")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.answers_file, encoding="utf-8")
            if json.loads(l)["type"] == "cross_company"]
    cats = {}
    for r in recs:
        companies = r["meta"]["companies"]
        golds = r["meta"].get("values_yuan", [])
        present = [company_value_in_ctx(r, c, g) for c, g in zip(companies, golds)]
        if ABSTAIN_RE.search(r["prediction"]):
            cat = "拒答_证据齐全" if all(present) else "拒答_证据缺失"
        elif is_correct(r):
            cat = "答对_证据齐全" if all(present) else "答对_证据不全(蒙对/常识)"
        else:
            if not all(present):
                cat = "答错_证据缺失(检索问题)"
            elif units_differ(r):
                cat = "答错_证据齐全_两家单位不同(疑似不换算直接比)"
            else:
                cat = "答错_证据齐全_同单位(纯比较错)"
        cats.setdefault(cat, []).append(r)

    print(f"跨公司题 {len(recs)} 条：")
    for cat, rs in sorted(cats.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:<38} {len(rs)}")
    print()
    for cat in [c for c in cats if c.startswith("答错")]:
        print(f"-- {cat} --")
        for r in cats[cat][:6]:
            gy = r["meta"].get("values_yuan", [])
            print(f"  Q: {r['question']}")
            print(f"     gold={r['gold']} pred={r['prediction'][:40]}  两家数值(元)={gy}")


if __name__ == "__main__":
    main()
