#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
年报 PDF 结构化解析脚本（试点版）
核心思路：结构感知——表格不做朴素chunk切分，保留 表名/表头/行标签/单元格 关系，
并额外输出线性化三元组（表格上下文|行标签|列标题=值），供结构感知检索模块直接使用。

用法：
    python report_parser.py data/raw_pdfs/xxx.pdf -o data/parsed/
    python report_parser.py data/raw_pdfs/ -o data/parsed/          # 整个目录

依赖：pip install pdfplumber
输出：每份PDF一个 JSON：
{
  "source": "...pdf",
  "pages": [
    {"page": 1,
     "text_blocks": ["段落文本", ...],
     "tables": [
        {"table_id": "p1_t0",
         "caption_guess": "表格上方最近一行文本",
         "header": ["项目", "本期金额", "上期金额"],
         "rows": [["营业收入", "1,234,567.89", "1,100,000.00"], ...],
         "linearized": ["[表:caption] 营业收入 | 本期金额 = 1,234,567.89", ...]}
     ]}
  ]
}
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pdfplumber

NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?%?")


def clean_cell(c):
    if c is None:
        return ""
    return re.sub(r"\s+", " ", str(c)).strip()


def merge_multirow_header(rows, max_header_rows=2):
    """简单处理两行表头：若第2行大量为空/延续，与第1行拼接。"""
    if len(rows) < 2:
        return rows[0] if rows else [], rows[1:] if len(rows) > 1 else []
    r0, r1 = rows[0], rows[1]
    # 第一行数值占比低且第二行也非数值行 -> 可能是两行表头
    def numeric_ratio(row):
        vals = [c for c in row if c]
        if not vals:
            return 0.0
        return sum(bool(NUM_RE.fullmatch(c.replace(" ", ""))) for c in vals) / len(vals)

    if numeric_ratio(r1) < 0.3 and len(rows) > 2:
        header = []
        for a, b in zip(r0, r1):
            header.append((a + " " + b).strip() if b and b != a else a)
        return header, rows[2:]
    return r0, rows[1:]


def table_quality(header, rows):
    """低质量表检测：表头大面积为空的超宽表（复杂合并单元格），标记为low。"""
    if not header:
        return "low"
    empty_ratio = sum(1 for h in header if not h) / len(header)
    if len(header) >= 10 and empty_ratio > 0.5:
        return "low"
    if empty_ratio > 0.8:
        return "low"
    return "ok"


def linearize_table(caption, header, rows):
    """输出 '[表:caption] 行标签 | 列标题 = 值' 三元组，保留数值与其语义坐标的绑定。
    行标签回退：首列为空时，取该行第一个非数值的非空单元格。"""
    out = []
    prefix = f"[表:{caption}] " if caption else ""
    for row in rows:
        if not any(row):
            continue
        row_label = row[0]
        label_idx = 0
        if not row_label:
            for j, c in enumerate(row):
                if c and not NUM_RE.fullmatch(c.replace(" ", "")):
                    row_label, label_idx = c, j
                    break
        if not row_label:
            row_label = "(无行标签)"
        for j, val in enumerate(row):
            if j == label_idx or not val:
                continue
            col = header[j] if j < len(header) and header[j] else f"列{j}"
            out.append(f"{prefix}{row_label} | {col} = {val}")
    return out


UNIT_LINE_RE = re.compile(r"^[（(]?\s*(单位|货币单位|币种|人民币|金额单位)")
SKIP_CAPTION_RE = re.compile(r"年度报告(全文)?$|^第[一二三四五六七八九十]+节|^\d+\s*/\s*\d+$")


def guess_caption(page, table_bbox, text_lines):
    """取表格上方最近的有效文本行作为表名：跳过单位行、页眉、纯页码。
    若单位行存在，将其附加到caption后（保留量纲信息）。"""
    top = table_bbox[1]
    candidates = [(ln["top"], ln["text"].strip()) for ln in text_lines
                  if ln["bottom"] <= top and ln["text"].strip()]
    candidates.sort(key=lambda x: -x[0])  # 由近及远
    unit = ""
    caption = ""
    for _, text in candidates[:5]:  # 最多向上看5行
        if UNIT_LINE_RE.search(text):
            unit = unit or text
            continue
        if SKIP_CAPTION_RE.search(text):
            continue
        caption = text[:60]
        break
    if caption and unit:
        return f"{caption}（{unit.strip('（）()')}）"[:80]
    return caption or unit


def split_text_blocks(text, max_len=300):
    """按句号切分并聚合成不超过max_len字符的文本块，作为检索的文本单元。"""
    if not text.strip():
        return []
    sentences = re.split(r"(?<=[。！？；])", text)
    blocks, buf = [], ""
    for s in sentences:
        if buf and len(buf) + len(s) > max_len:
            blocks.append(buf)
            buf = s
        else:
            buf += s
    if buf.strip():
        blocks.append(buf)
    return [b.strip() for b in blocks if len(b.strip()) > 10]


def parse_pdf(pdf_path: Path, max_pages=None):
    doc = {"source": pdf_path.name, "pages": []}
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        for pno, page in enumerate(pages, start=1):
            text_lines = page.extract_text_lines() or []
            tables_found = page.find_tables()
            table_bboxes = [t.bbox for t in tables_found]

            # 文本块：排除落在表格bbox内的行，按空行聚合成段
            def in_table(ln):
                cy = (ln["top"] + ln["bottom"]) / 2
                return any(bb[1] <= cy <= bb[3] for bb in table_bboxes)

            page_text = "".join(ln["text"].strip() for ln in text_lines if not in_table(ln))
            blocks = split_text_blocks(page_text)

            page_tables = []
            for ti, t in enumerate(tables_found):
                raw = t.extract()
                if not raw:
                    continue
                rows = [[clean_cell(c) for c in r] for r in raw]
                header, body = merge_multirow_header(rows)
                caption = guess_caption(page, t.bbox, text_lines)
                quality = table_quality(header, body)
                page_tables.append({
                    "table_id": f"p{pno}_t{ti}",
                    "caption_guess": caption,
                    "quality": quality,
                    "header": header,
                    "rows": body,
                    "linearized": linearize_table(caption, header, body) if quality == "ok" else [],
                })

            if blocks or page_tables:
                doc["pages"].append({
                    "page": pno,
                    "text_blocks": blocks,
                    "tables": page_tables,
                })
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="PDF文件或目录")
    ap.add_argument("-o", "--out", default="data/parsed", help="JSON输出目录")
    ap.add_argument("--max-pages", type=int, default=None, help="每份PDF最多解析页数（试点可设50）")
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(inp.glob("*.pdf")) if inp.is_dir() else [inp]
    if not pdfs:
        sys.exit("未找到PDF文件")

    for p in pdfs:
        print(f"解析 {p.name} ...")
        try:
            doc = parse_pdf(p, max_pages=args.max_pages)
        except Exception as e:
            print(f"  [错误] {e}")
            continue
        n_tables = sum(len(pg["tables"]) for pg in doc["pages"])
        n_triples = sum(len(t["linearized"]) for pg in doc["pages"] for t in pg["tables"])
        out_path = out_dir / (p.stem + ".json")
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  页数 {len(doc['pages'])}, 表格 {n_tables}, 线性化三元组 {n_triples} -> {out_path.name}")


if __name__ == "__main__":
    main()
