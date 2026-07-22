#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构造数值校验器（RQ2）训练数据。

输入：端到端答案文件（含真实检索上下文）
输出：train/val/test 三个jsonl，每条 = {"prompt", "label", "kind", "company", "qa_id"}
  label: "支持"   —— 候选答案正确 且 gold证据确实在上下文中
         "不支持" —— 候选答案错误，或答案对但上下文缺证据（groundedness口径）

负例按端到端错误归因的真实模式程序化生成：
  wrong_pos      抽取题换成上下文中另一个数（真实错误占比最高的抽取错）
  unit_err       同数字换错单位/量纲
  wrong_operand  同比题用上下文中错误的两数算增长率
  sign_flip      同比题正负号翻转
  wrong_company  对比题换成另一家公司
  fabricated     扰动数字（上下文中不存在）
  ungrounded     答案=gold但上下文缺证据 -> 不支持
另加模型真实预测：判对的为正例(model_correct)，判错且非拒答的为负例(model_error)。

划分：按公司划分，test公司的所有样本不进train（防同报告泄漏）。
用法：
    python build_verifier_data.py data/answers_v2_structural_full.jsonl -o data/verifier/
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from eval_answers import ABSTAIN_RE, NUM_RE, extract_number, is_correct
from attribute_errors import ctx_values, gold_values, in_ctx

random.seed(42)

TEST_COMPANIES = {"贵州茅台", "宁德时代", "工商银行", "迈瑞医疗", "伊利股份", "顺丰控股"}
VAL_COMPANIES = {"万科A", "紫金矿业", "恒瑞医药"}

PROMPT_TMPL = (
    "你是财报问答的数值校验员。根据下面的资料，判断候选答案是否正确且有资料依据。\n"
    "只有当候选答案与资料中的数值一致（含单位换算）、且确实回答了问题时，才判定"
    "\"支持\"；答案错误、单位错误、或资料中找不到依据时判定\"不支持\"。\n\n"
    "资料：\n{ctx}\n\n问题：{q}\n候选答案：{a}\n\n判定："
)


def fmt_num(v):
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.2f}"


def rec_company(rec):
    m = rec.get("meta", {})
    return m.get("company") or "+".join(m.get("companies", [])) or "unknown"


def split_of(company):
    parts = set(company.split("+"))
    if parts & TEST_COMPANIES:
        return "test"
    if parts & VAL_COMPANIES:
        return "val"
    return "train"


def ctx_text(rec):
    return "\n".join(f"[{i+1}] {u['text']}" for i, u in enumerate(rec["retrieved"]))


def gold_answer_str(rec):
    return rec["gold"]


def _norm(s):
    return re.sub(r"[（(].*?[)）]|\s", "", s or "")


def evidence_present(rec, ctx_vals):
    """gold证据是否完整进入上下文。

    比较题的gold是公司名而非数值，gold_values() 对其返回空列表；若直接按数值
    判定，整类比较题都会被标成"无依据"，其标准答案会被当作负例喂给校验器。
    比较题改按 gold_evidence 溯源（源文件公司名 + 行标签）判定双侧证据。
    """
    if rec["type"] == "cross_company":
        ev = rec.get("gold_evidence") or []
        if not ev:
            return False
        return all(
            any(e["source"].split("_")[1] in u["text"]
                and _norm(e["row_label"])[:6] in _norm(u["text"])
                for u in rec["retrieved"])
            for e in ev
        )
    golds = gold_values(rec)
    return bool(golds) and all(in_ctx(v, ctx_vals) for v in golds)


def make_negatives(rec, ctx_vals):
    """按题型生成扰动负例（候选答案字符串, kind）。"""
    negs = []
    t, gold = rec["type"], rec["gold"]
    m = rec.get("meta", {})
    if t == "extraction":
        gv = m.get("value")
        unit = m.get("unit", "")
        # wrong_pos: 上下文中另一个量级相近的数
        cands = [v for v in ctx_vals if abs(v) > 100 and gv and abs(v - gv) > abs(gv) * 0.01]
        if cands:
            v = random.choice(cands)
            negs.append((f"{fmt_num(v)}{unit}", "wrong_pos"))
        if gv:
            # unit_err: 数字不变换错单位
            wrong_unit = random.choice([u for u in ("元", "千元", "万元", "百万元", "亿元")
                                        if u != unit] or ["元"])
            negs.append((f"{fmt_num(gv)}{wrong_unit}", "unit_err"))
            # fabricated: 扰动数字（确保不在上下文）
            for _ in range(5):
                fake = gv * random.uniform(1.02, 1.30)
                if not in_ctx(fake, ctx_vals, tol=1e-4):
                    negs.append((f"{fmt_num(fake)}{unit}", "fabricated"))
                    break
    elif t == "yoy_compare":
        g = float(gold.rstrip("%"))
        # sign_flip
        negs.append((f"{-g:+.2f}%", "sign_flip"))
        # wrong_operand: 上下文任意两数的增长率（避开正确值）
        vals = [v for v in ctx_vals if abs(v) > 100]
        random.shuffle(vals)
        for a in vals[:20]:
            for b in vals[:20]:
                if b and abs(a) != abs(b):
                    p = (a - b) / abs(b) * 100
                    if 0.2 < abs(p - g) and abs(p) < 200:
                        negs.append((f"{p:+.2f}%", "wrong_operand"))
                        break
            if len([n for n in negs if n[1] == "wrong_operand"]):
                break
        # fabricated
        negs.append((f"{g + random.choice([3.7, -5.2, 11.4]):+.2f}%", "fabricated"))
    elif t == "cross_company":
        others = [c for c in m.get("companies", []) if c != gold]
        if others:
            negs.append((others[0], "wrong_company"))
    return negs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers_file")
    ap.add_argument("-o", "--out", default="data/verifier")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.answers_file, encoding="utf-8")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    buckets = {"train": [], "val": [], "test": []}
    stats = Counter()
    for rec in recs:
        company = rec_company(rec)
        split = split_of(company)
        ctx = ctx_text(rec)
        cvals = ctx_values(rec)
        grounded = evidence_present(rec, cvals)

        def emit(ans, label, kind):
            buckets[split].append({
                "prompt": PROMPT_TMPL.format(ctx=ctx, q=rec["question"], a=ans),
                "label": label, "kind": kind,
                "company": company, "qa_id": rec["id"], "type": rec["type"],
            })
            stats[f"{split}:{label}:{kind}"] += 1

        # 正例 / ungrounded负例：gold答案
        if grounded:
            emit(gold_answer_str(rec), "支持", "gold")
        else:
            emit(gold_answer_str(rec), "不支持", "ungrounded")

        # 扰动负例（随机抽最多2条，控制正负比并保持负例种类多样）
        negs = make_negatives(rec, cvals)
        for ans, kind in random.sample(negs, min(2, len(negs))):
            emit(ans, "不支持", kind)

        # 模型真实预测
        pred = rec["prediction"]
        if not ABSTAIN_RE.search(pred):
            if is_correct(rec):
                if grounded and pred.strip() != gold_answer_str(rec):
                    emit(pred, "支持", "model_correct")
            else:
                emit(pred, "不支持", "model_error")

    for split, rows in buckets.items():
        random.shuffle(rows)
        p = out / f"{split}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        pos = sum(1 for r in rows if r["label"] == "支持")
        print(f"{split}: {len(rows)} 条（支持 {pos} / 不支持 {len(rows)-pos}）-> {p}")

    print("\n负例构成：")
    kinds = Counter()
    for k, v in stats.items():
        kinds[k.split(":", 2)[2]] += v
    for k, v in kinds.most_common():
        print(f"  {k:<14} {v}")


if __name__ == "__main__":
    main()
