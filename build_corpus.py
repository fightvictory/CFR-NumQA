#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建两套检索语料（索引单元），用于对比实验：
  A. naive     朴素基线：整页文本+表格拍平成纯文本，按固定长度切chunk（模拟常见RAG做法）
  B. structural 结构感知：文本块 + 表格线性化三元组（每条三元组独立成索引单元，
                保留 表名|行标签|列标题=值 的语义坐标）

每个索引单元带溯源元数据（source/page/table_id），与 qa_seed.jsonl 的 evidence 对齐，
用于计算检索命中（Recall@k）。

用法：
    python build_corpus.py data/parsed/ -o data/corpus/
输出：
    data/corpus/naive.jsonl       {"uid", "text", "source", "page", "table_id": null}
    data/corpus/structural.jsonl  {"uid", "text", "source", "page", "table_id"}
"""
import argparse
import json
import re
from pathlib import Path

CHUNK_SIZE = 400  # 朴素基线的chunk字符数
CHUNK_OVERLAP = 50

# 从表名caption解析计量单位，如"（货币单位：人民币百万元）""（单位：元 币种：人民币）"
CAPTION_UNIT_RE = re.compile(
    r"(?:货币)?单位[:：]\s*(?:人民币)?\s*(亿元|百万元|万元|千元|元|万股|千股|股)")
# 纯数值（允许千分位、负号、会计括号负数）才附单位
PURE_NUM_RE = re.compile(r"^\(?-?[\d,]+(?:\.\d+)?\)?$")
# 行标签/列标题含这些字样时不附单位（每股指标、比率、数量类，其值不是caption的货币单位）
NO_UNIT_HINT_RE = re.compile(r"每股|元/股|率|占比|比例|百分|倍|人数|户数|股数")


def attach_unit(tri):
    """把caption中的单位附着到三元组的数值上：
    '[表:关键指标（货币单位：人民币百万元）] 营业收入 | 2023年 = 164,699'
      -> '... = 164,699百万元'
    解决单位与数值分离导致模型回答缺单位、评测量纲判错的问题。"""
    head, sep, val = tri.rpartition(" = ")
    if not sep:
        return tri
    v = val.strip()
    if not PURE_NUM_RE.match(v):
        return tri
    m = CAPTION_UNIT_RE.search(head)
    if not m:
        return tri
    body = head.split("]", 1)[1] if "]" in head else head
    if NO_UNIT_HINT_RE.search(body):
        return tri
    return f"{head} = {v}{m.group(1)}"


def doc_context(doc):
    """文档级上下文前缀：公司名+报告名，注入每个索引单元（两套语料均加，保证对比公平）。"""
    parts = Path(doc["source"]).stem.split("_")
    company = parts[1] if len(parts) >= 2 else ""
    title = parts[2] if len(parts) >= 3 else ""
    return f"【{company} {title}】"


def naive_units(doc):
    """整页内容（文本+表格拍平）拼接后固定长度切分——典型的结构无关做法。"""
    units = []
    for page in doc["pages"]:
        parts = list(page["text_blocks"])
        for t in page["tables"]:
            # 拍平表格：所有单元格顺序拼接（模拟PDF直接抽文本的效果）
            parts.append(" ".join(t["header"]))
            for r in t["rows"]:
                parts.append(" ".join(c for c in r if c))
        full = "\n".join(parts)
        ctx = doc_context(doc)
        i = 0
        while i < len(full):
            chunk = full[i:i + CHUNK_SIZE]
            if chunk.strip():
                units.append({"text": ctx + chunk, "source": doc["source"],
                              "page": page["page"], "table_id": None})
            i += CHUNK_SIZE - CHUNK_OVERLAP
    return units


def structural_units(doc):
    """文本块 + 三元组各自成独立索引单元，表格保留语义坐标。"""
    units = []
    ctx = doc_context(doc)
    for page in doc["pages"]:
        for b in page["text_blocks"]:
            units.append({"text": ctx + b, "source": doc["source"],
                          "page": page["page"], "table_id": None})
        for t in page["tables"]:
            for tri in t["linearized"]:
                units.append({"text": ctx + attach_unit(tri), "source": doc["source"],
                              "page": page["page"], "table_id": t["table_id"]})
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parsed_dir")
    ap.add_argument("-o", "--out", default="data/corpus")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_naive, all_struct = [], []
    for f in sorted(Path(args.parsed_dir).glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        all_naive.extend(naive_units(doc))
        all_struct.extend(structural_units(doc))

    for name, units in [("naive", all_naive), ("structural", all_struct)]:
        p = out / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for i, u in enumerate(units):
                u["uid"] = f"{name}_{i:06d}"
                fh.write(json.dumps(u, ensure_ascii=False) + "\n")
        print(f"{name}: {len(units)} 个索引单元 -> {p}")


if __name__ == "__main__":
    main()
