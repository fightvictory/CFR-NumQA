#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预审抽检表：用PyMuPDF独立提取PDF原页文本，核对答案/证据/年份。
输出每条的核验结果jsonl，FLAG项另存页面PNG供目视核验。"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import fitz

random.seed(42)
qas = [json.loads(l) for l in open("data/qa_seed.jsonl", encoding="utf-8")]
by_type = defaultdict(list)
for qa in qas:
    by_type[qa["type"]].append(qa)
sample = (random.sample(by_type["extraction"], 53)
          + random.sample(by_type["yoy_compare"], 38)
          + random.sample(by_type["cross_company"], 9))

PDF_DIR = Path("data/raw_pdfs")
OUT_IMG = Path("audit_flag_pages")
OUT_IMG.mkdir(exist_ok=True)

_page_cache = {}


def page_text_norm(source, page_no):
    key = (source, page_no)
    if key not in _page_cache:
        doc = fitz.open(PDF_DIR / source)
        t = doc[page_no - 1].get_text() if 0 < page_no <= len(doc) else ""
        doc.close()
        _page_cache[key] = re.sub(r"[\s,，]", "", t)
    return _page_cache[key]


def num_variants(v):
    out = {f"{v:.2f}", f"{v:.1f}", f"{v:g}"}
    if float(v) == int(v):
        out.add(str(int(v)))
    # 负数在PDF中可能以括号或特殊负号渲染，同时匹配绝对值数字串
    if v < 0:
        out |= num_variants(-v)
    return out


def value_on_page(v, text):
    return any(s in text for s in num_variants(v))


def label_on_page(label, text):
    # 去掉标签内的括号单位后缀（如"总资产（元）"->"总资产"）再匹配
    lab = re.sub(r"[（(][^（）()]*[)）]", "", label or "")
    lab = re.sub(r"\s", "", lab)
    return bool(lab) and lab[:10] in text


def render(source, page_no, tag):
    doc = fitz.open(PDF_DIR / source)
    pix = doc[page_no - 1].get_pixmap(dpi=110)
    p = OUT_IMG / f"{tag}.png"
    pix.save(p)
    doc.close()
    return str(p)


results = []
for i, qa in enumerate(sample):
    m = qa.get("meta", {})
    evs = qa["evidence"]
    checks, notes = [], []

    if qa["type"] == "extraction":
        ev = evs[0]
        text = page_text_norm(ev["source"], ev["page"])
        ok_v = value_on_page(m["value"], text)
        ok_l = label_on_page(ev.get("row_label"), text)
        checks = [ok_v, ok_l]
        if not ok_v:
            notes.append("答案数值未在原页找到")
        if not ok_l:
            notes.append("行标签未在原页找到")

    elif qa["type"] == "yoy_compare":
        vals = m.get("values", [])
        # evidence顺序与values顺序无固定对应（新报含两年数据），按证据页并集匹配
        union = "".join(page_text_norm(ev["source"], ev["page"]) for ev in evs)
        ok_vals = []
        for v in vals:
            scales = (1, 1e3, 1e4, 1e6, 1e8, 1e-3, 1e-4, 1e-6, 1e-8) if abs(v) >= 1e5 else (1,)
            ok_vals.append(any(value_on_page(v * s, union) for s in scales))
        ok_rate = False
        if len(vals) == 2 and vals[0]:
            rate = (vals[1] - vals[0]) / abs(vals[0]) * 100
            ok_rate = abs(rate - float(qa["answer"].rstrip("%"))) < 0.01
        checks = ok_vals + [ok_rate]
        if not all(ok_vals):
            notes.append("操作数未全部在原页找到")
        if not ok_rate:
            notes.append("增长率与操作数不一致")

    else:  # cross_company
        vals = m.get("values_yuan", [])
        ok_vals = []
        for ev, v in zip(evs, vals):
            text = page_text_norm(ev["source"], ev["page"])
            hit = any(value_on_page(v / s, text) for s in (1, 1e3, 1e4, 1e6, 1e8))
            ok_vals.append(hit)
        gold_idx = m["companies"].index(qa["answer"]) if qa["answer"] in m.get("companies", []) else -1
        ok_ans = gold_idx >= 0 and len(vals) == 2 and vals[gold_idx] == max(vals)
        checks = ok_vals + [ok_ans]
        if not all(ok_vals):
            notes.append("公司数值未全部在原页找到")
        if not ok_ans:
            notes.append("答案公司并非较大值")

    verdict = "PASS" if all(checks) else "FLAG"
    imgs = []
    if verdict == "FLAG":
        for j, ev in enumerate(evs[:2]):
            imgs.append(render(ev["source"], ev["page"], f"{i+1:03d}_{qa['id']}_{j}"))
    results.append({"row": i + 1, "id": qa["id"], "type": qa["type"],
                    "verdict": verdict, "notes": notes, "imgs": imgs,
                    "question": qa["question"], "answer": qa["answer"],
                    "meta": m, "evidence": evs})

Path("preaudit.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8")
n_pass = sum(r["verdict"] == "PASS" for r in results)
print(f"PASS {n_pass} / FLAG {100 - n_pass}")
for r in results:
    if r["verdict"] == "FLAG":
        print(f"  行{r['row']} {r['id']} [{r['type']}] {'; '.join(r['notes'])}")
        print(f"    Q: {r['question'][:40]}  A: {r['answer']}")
