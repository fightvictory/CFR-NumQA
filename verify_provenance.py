#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""穷尽验证全部 1,016 条问答对的证据溯源，不抽样。

为什么改成机器做
----------------
首轮抽检有一列是人工判「证据支持?」，100 条全判 ✓。但这件事本身是可判定的：
gold 数值在不在所记 (source,page,table_id,row_label) 解析出的三元组里，是个
确定性检查，机器可以对全部 1,016 条做，比抽 100 条手判强得多。人工判断应该留给
机器判不了的事（问题是否有歧义）。

检查项
------
  P1 证据可解析：每条 evidence 都能在语料里定位到至少一个三元组
  P2 数值在证据内：抽取题的 gold 数值出现在所定位的三元组中
  P3 操作数齐备：同比题与对比题各有不少于两条证据（两个操作数缺一不可）
  P4 年份可核：所定位的三元组里能找到问题所问的年份

论文的证据覆盖率（主检索指标）与验证器的 provenance-based 监督，都以 evidence
字段正确为前提。这个脚本就是那个前提的证明。

用法：
    python verify_provenance.py            # 全部通过则 exit 0
    python verify_provenance.py --show 5   # 额外打印前 5 条失败样例
"""
import argparse
import collections
import json
import re
import sys


def digits(s):
    return re.sub(r"[^\d]", "", str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/qa_seed.jsonl")
    ap.add_argument("--corpus", default="data/corpus/structural.jsonl")
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    qas = [json.loads(l) for l in open(args.qa, encoding="utf-8")]
    idx = collections.defaultdict(list)
    for l in open(args.corpus, encoding="utf-8"):
        u = json.loads(l)
        if u.get("table_id"):
            idx[(u["source"], u["page"], u["table_id"])].append(u)

    fails = collections.defaultdict(list)

    for q in qas:
        evs = q.get("evidence", [])
        hits = []
        for ev in evs:
            key = (ev["source"], ev["page"], ev.get("table_id"))
            rl = ev.get("row_label", "")
            hits += [u["text"] for u in idx.get(key, []) if rl and rl in u["text"]]

        # P1 证据可解析
        if not hits:
            fails["P1 证据定位不到三元组"].append(q["id"])
            continue

        # P2 数值在证据内（只对抽取题：同比题的 gold 是增长率，不直接出现在表里；
        #    对比题的 gold 是公司名）
        if q["type"] == "extraction":
            d = digits(q.get("answer", ""))[:8]
            if d and not any(d in digits(h) for h in hits):
                fails["P2 gold 数值不在证据中"].append(q["id"])

        # P3 两操作数题的证据条数
        if q["type"] in ("yoy_compare", "cross_company") and len(evs) < 2:
            fails["P3 双操作数题证据不足两条"].append(q["id"])

        # P4 年份可核：问题所问年份应出现在三元组的「列标题=值」段
        m = q.get("meta", {})
        years = [str(y) for y in (m.get("years") or ([m["year"]] if m.get("year") else []))]
        if years:
            segs = [h.split("|", 1)[1] if "|" in h else h for h in hits]
            if not any(f"{y}年" in s for y in years for s in segs):
                fails["P4 证据里找不到所问年份"].append(q["id"])

    n = len(qas)
    by_type = collections.Counter(q["type"] for q in qas)
    print(f"证据溯源穷尽验证：{n} 条问答对"
          f"（{', '.join(f'{k} {v}' for k, v in sorted(by_type.items()))}）\n")
    checks = ["P1 证据定位不到三元组", "P2 gold 数值不在证据中",
              "P3 双操作数题证据不足两条", "P4 证据里找不到所问年份"]
    total = 0
    for c in checks:
        bad = fails.get(c, [])
        total += len(bad)
        mark = "✓" if not bad else "✗"
        print(f"  {mark} {c:<26} {len(bad)} 条")
        for i in bad[:args.show]:
            print(f"      {i}")
    print()
    if total == 0:
        print(f"全部 {n} 条通过。证据字段可作为覆盖率指标与验证器监督的依据。")
        return 0
    print(f"!! 共 {total} 处问题，需在发布前修正。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
