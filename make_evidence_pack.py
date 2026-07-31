#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把抽检表引用到的年报页面抽出来，合成一份小体积的核验用 PDF。

为什么要这个
------------
标注表里约两成条目需要回查年报原文（表名读不出计量单位、或要确认上年数是否
追溯调整）。但 103 份源年报共 859 MB，整包发给外部标注者不现实，而且他们还得
在里面翻找。实际上 180 个条目只引用了几十页——抽出来即可。

产物
----
  引用页.pdf     只含被引用的页，按 (年报, 页码) 排序；每页顶部盖上引用它的条目 ID；
                 每个条目一条书签，可直接搜 seed_0416 跳转
  引用页索引.md  条目 ID -> 抽取包页码 -> 问题 -> 原始出处，便于按 ID 查表

需要 PyMuPDF（pilot_toolkit/venv 里有）：
    ../pilot_toolkit/venv/bin/python make_evidence_pack.py
"""
import argparse
import collections
import importlib.util
import json
import os
import sys

try:
    import fitz                      # PyMuPDF
except ImportError:
    sys.exit("!! 需要 PyMuPDF。试试 ../pilot_toolkit/venv/bin/python make_evidence_pack.py")


def load_items():
    """与 make_audit2.py 用同一套抽样逻辑，保证抽取包与标注表严格对应。"""
    spec = importlib.util.spec_from_file_location("m", "make_audit2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    qas = [json.loads(l) for l in open("data/qa_seed.jsonl", encoding="utf-8")]
    main = m.main_sample(qas)
    hard = [q for q, _ in m.hard_sample(qas, {q["id"] for q in main}, 80)]
    return {q["id"]: q for q in main + hard}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="../pilot_toolkit/data/raw_pdfs")
    ap.add_argument("-o", "--out", default="引用页.pdf")
    ap.add_argument("--index", default="引用页索引.md")
    args = ap.parse_args()

    items = load_items()

    # (年报, 页码) -> 引用它的条目
    cites = collections.defaultdict(list)
    for qid, q in items.items():
        for ev in q["evidence"]:
            cites[(ev["source"], ev["page"])].append(qid)

    keys = sorted(cites)
    out = fitz.open()
    missing, index = [], []

    for src, page in keys:
        path = os.path.join(args.pdf_dir, src)
        if not os.path.exists(path):
            missing.append(src)
            continue
        doc = fitz.open(path)
        if page < 1 or page > doc.page_count:
            missing.append(f"{src}@{page}")
            doc.close()
            continue
        out.insert_pdf(doc, from_page=page - 1, to_page=page - 1)   # evidence 的页码是 1 起
        doc.close()

        newno = out.page_count                     # 该页在抽取包里的页码（1 起）
        ids = sorted(set(cites[(src, page)]))
        pg = out[newno - 1]

        # 顶部盖一条浅色带，写上引用本页的条目 ID。放在页面最上沿的页边距里，
        # 年报正文一般不占这个位置；万一遮住页眉也只是页眉，不影响表格核验。
        # 换行而不截断：有 4 页被十几个条目共同引用，一行放不下。截断会让被截掉的
        # 条目 ID 在 PDF 里搜不到——标注者按 ID 找页就会扑空。
        r = pg.rect
        PER_LINE = 8
        lines = [" ".join(ids[i:i + PER_LINE]) for i in range(0, len(ids), PER_LINE)]
        band = fitz.Rect(0, 0, r.width, 14 + 11 * len(lines))
        pg.draw_rect(band, color=None, fill=(1, 0.94, 0.6), overlay=True)
        # 用拉丁字体而非 china-s：标记内容全是 ASCII，而 china-s 是 CID 字体、
        # 缺 ToUnicode 映射，盖上去只能看不能搜，标注者就没法 Ctrl+F 找条目 ID。
        for k, ln in enumerate(lines):
            pg.insert_text((8, 15 + 11 * k), (f"[{newno}] " if k == 0 else "      ") + ln,
                           fontsize=9, fontname="helv", color=(0.2, 0.15, 0), overlay=True)

        for qid in ids:
            q = items[qid]
            out.set_toc(out.get_toc() + [[1, f"{qid}  {q['question'][:28]}", newno]])
            index.append((qid, newno, q["question"], f"{src} @{page}"))

    # 64 页来自 55 份不同年报，每份都带着自己的完整字体子集，不做处理约 24 MB；
    # 子集化后只留实际用到的字形，降到 14 MB 左右，便于微信/邮件传给外部标注者。
    try:
        out.subset_fonts()
    except Exception as e:                      # 旧版 PyMuPDF 没有这个方法
        print(f"  （字体子集化跳过：{e}）")
    out.save(args.out, garbage=4, deflate=True, deflate_images=True,
             deflate_fonts=True, clean=True)
    size = os.path.getsize(args.out) / 1048576
    out.close()

    index.sort()
    with open(args.index, "w", encoding="utf-8") as fh:
        fh.write("# 引用页索引\n\n")
        fh.write(f"共 {len(keys)} 页，覆盖 {len(items)} 个条目。"
                 f"在 `引用页.pdf` 里可直接搜条目 ID，或用书签跳转。\n\n")
        fh.write("| 条目 ID | 引用页.pdf 第几页 | 问题 | 原始出处 |\n|---|---|---|---|\n")
        for qid, no, ques, src in index:
            fh.write(f"| {qid} | {no} | {ques} | {src} |\n")

    print(f"已生成 {args.out}（{len(keys)} 页，{size:.1f} MB）与 {args.index}")
    print(f"  覆盖条目 {len({i[0] for i in index})}/{len(items)}")
    if missing:
        print(f"  !! 缺 {len(missing)} 项：{missing[:5]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
