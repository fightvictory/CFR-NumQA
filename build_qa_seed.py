#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
种子问答对生成脚本（试点版）
从 report_parser.py 的解析结果中，基于财务指标模板生成三类问答对：
  1. extraction   抽取式：X公司2024年度的营业收入是多少？
  2. yoy_compare  同比多跳：X公司2024年营业收入比2023年增长了多少（%）？（答案程序化计算，可验证）
  3. cross_company 跨文档对比：2024年度A公司和B公司谁的营业收入更高？

每条问答附带证据溯源（文件/页码/表ID/原始三元组），供后续：
  - 检索评测（Precision@k 的 gold evidence）
  - 幻觉标注（答案是否可由证据支持）

用法：
    python build_qa_seed.py data/parsed/ -o data/qa_seed.jsonl
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

# 核心财务指标模板：行标签匹配模式 -> 标准指标名
INDICATORS = {
    "营业收入": re.compile(r"^营业(总)?收入"),
    "归母净利润": re.compile(r"^归属于(上市公司|本行|母公司)(股东|普通股股东)?的?净利润"),
    "基本每股收益": re.compile(r"^基本每股收益"),
    "总资产": re.compile(r"^(资产总额|总资产)"),
    "归母净资产": re.compile(r"^归属于(上市公司|本行|母公司)(股东|普通股股东)?的?(净资产|所有者权益|股东权益)"),
    "经营活动现金流量净额": re.compile(r"^经营活动产生的现金流量净额"),
    "加权平均净资产收益率": re.compile(r"^加权平均净资产收益率"),
}

YEAR_COL_RE = re.compile(r"^(20\d{2})年(度|末|12月31日)?$")
NON_NUM_CELL_RE = re.compile(r"^-?[\d,．.。%()（）\s]+$")
NUM_CLEAN_RE = re.compile(r"^-?[\d,]+(\.\d+)?$")
UNIT_RE = re.compile(r"(单位[：:]\s*)?(人民币)?\s*(百万元|万元|千元|亿元|元)")


def parse_number(s):
    s = s.strip().replace(",", "")
    if s.startswith("(") and s.endswith(")"):  # 会计负数
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None


def detect_unit(caption, row_label):
    """量纲优先取行标签内的（如'营业收入（千元）'），其次取表名内的单位说明。
    比率类指标（收益率/占比等）不是货币量纲，单位为%（人工抽检发现的badcase）。"""
    if re.search(r"率|占比|比例", row_label or ""):
        return "%"
    for src in (row_label, caption or ""):
        m = UNIT_RE.search(src)
        if m:
            return m.group(3)
    return "元"


def company_of(fname):
    """文件名约定：代码_公司名_标题.json"""
    parts = Path(fname).stem.split("_")
    return parts[1] if len(parts) >= 2 else Path(fname).stem


def harvest_facts(parsed_dir: Path):
    """从所有解析JSON中收割 (公司, 指标, 年度) -> {值, 单位, 证据} 事实表。"""
    facts = {}
    for f in sorted(parsed_dir.glob("*.json")):
        company = company_of(f.name)
        doc = json.loads(f.read_text(encoding="utf-8"))
        for page in doc["pages"]:
            for t in page["tables"]:
                if t.get("quality") == "low":
                    continue
                header = t["header"]
                year_cols = {j: m.group(1) for j, h in enumerate(header)
                             if h and (m := YEAR_COL_RE.match(h.replace(" ", "")))}
                if not year_cols:
                    continue
                for row in t["rows"]:
                    if not row:
                        continue
                    # 行标签：首列为空时回退到第一个非空、非纯数值单元格（合并单元格场景）
                    raw_label = row[0]
                    if not raw_label:
                        for c in row:
                            if c and not NON_NUM_CELL_RE.match(c):
                                raw_label = c
                                break
                    if not raw_label:
                        continue
                    label = raw_label.replace(" ", "")
                    for ind_name, pat in INDICATORS.items():
                        if not pat.match(label):
                            continue
                        for j, year in year_cols.items():
                            if j >= len(row) or not row[j]:
                                continue
                            val = parse_number(row[j])
                            if val is None:
                                continue
                            key = (company, ind_name, year)
                            # 同一事实可能出现在多处（摘要页优先=首次出现）
                            if key in facts:
                                continue
                            facts[key] = {
                                "value": val,
                                "raw": row[j],
                                "unit": detect_unit(t.get("caption_guess"), raw_label),
                                "evidence": {
                                    "source": doc["source"],
                                    "page": page["page"],
                                    "table_id": t["table_id"],
                                    "caption": t.get("caption_guess", ""),
                                    "row_label": raw_label,
                                },
                            }
    return facts


def gen_qa(facts, seed=42):
    rng = random.Random(seed)
    qas = []
    qid = 0

    def add(qtype, question, answer, evidence, meta=None):
        nonlocal qid
        qid += 1
        qas.append({
            "id": f"seed_{qid:04d}", "type": qtype,
            "question": question, "answer": answer,
            "evidence": evidence, "meta": meta or {},
        })

    # 1. 抽取式
    for (company, ind, year), f in facts.items():
        if f["raw"].endswith("%") or (f["unit"] == "元" and "每股" in ind):
            answer = f["raw"]
        else:
            answer = f"{f['raw']}{f['unit']}"
        add("extraction",
            f"{company}{year}年度的{ind}是多少？",
            answer,
            [f["evidence"]],
            {"indicator": ind, "year": year, "company": company,
             "value": f["value"], "unit": f["unit"]})

    # 2. 同比多跳（增长率程序化计算）
    by_ci = defaultdict(dict)
    for (company, ind, year), f in facts.items():
        by_ci[(company, ind)][year] = f
    for (company, ind), ymap in by_ci.items():
        years = sorted(ymap)
        for y0, y1 in zip(years, years[1:]):
            f0, f1 = ymap[y0], ymap[y1]
            if f0["unit"] != f1["unit"] or f0["value"] == 0 or "率" in ind:
                continue
            growth = (f1["value"] - f0["value"]) / abs(f0["value"]) * 100
            add("yoy_compare",
                f"{company}{y1}年{ind}相比{y0}年变动了百分之多少？",
                f"{growth:+.2f}%",
                [f1["evidence"], f0["evidence"]],
                {"indicator": ind, "years": [y0, y1], "company": company,
                 "values": [f0["value"], f1["value"]], "unit": f0["unit"]})

    # 3. 跨公司对比（需统一量纲后比较）
    UNIT_SCALE = {"元": 1, "千元": 1e3, "万元": 1e4, "百万元": 1e6, "亿元": 1e8}
    by_iy = defaultdict(list)
    for (company, ind, year), f in facts.items():
        by_iy[(ind, year)].append((company, f))
    for (ind, year), items in by_iy.items():
        if len(items) < 2 or "每股" in ind or "率" in ind:
            continue
        pairs = [(a, b) for i, a in enumerate(items) for b in items[i + 1:]]
        rng.shuffle(pairs)
        for (ca, fa), (cb, fb) in pairs[:3]:
            va = fa["value"] * UNIT_SCALE.get(fa["unit"], 1)
            vb = fb["value"] * UNIT_SCALE.get(fb["unit"], 1)
            winner = ca if va > vb else cb
            add("cross_company",
                f"{year}年度，{ca}和{cb}的{ind}谁更高？",
                winner,
                [fa["evidence"], fb["evidence"]],
                {"indicator": ind, "year": year,
                 "companies": [ca, cb], "values_yuan": [va, vb]})
    return qas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parsed_dir", help="report_parser.py 的输出目录")
    ap.add_argument("-o", "--out", default="data/qa_seed.jsonl")
    args = ap.parse_args()

    facts = harvest_facts(Path(args.parsed_dir))
    print(f"收割事实: {len(facts)} 条")
    qas = gen_qa(facts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for qa in qas:
            fh.write(json.dumps(qa, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"生成问答对: {len(qas)} 条 {dict(Counter(q['type'] for q in qas))}")
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
