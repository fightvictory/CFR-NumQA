#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
答案评测脚本：准确率 + 数值幻觉率（无GPU也能跑）
    python eval_answers.py results/answers_structural.jsonl
    python eval_answers.py results/answers_naive.jsonl

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
    """从文本中提取第一个数值及其量纲，返回换算成元/原值的浮点数。

    跳过年份词元（紧跟"年"的四位整数，如"2023年度"）：叙述式回答常以
    "公司2023年度…"开头，取第一个数字会误取年份而非答案数值。
    """
    t = text.replace(" ", "")
    for m in NUM_RE.finditer(t):
        raw = m.group()
        tail = t[m.end():m.end() + 4]
        if tail.startswith("年") and "." not in raw and len(raw.replace(",", "")) == 4:
            continue
        val = float(raw.replace(",", ""))
        for unit, scale in sorted(UNIT_SCALE.items(), key=lambda x: -len(x[0])):
            if tail.startswith(unit):
                return val * scale
        return val
    return None


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


# 长上下文实验的上下文就是整份年报（约 7.5 万字符）。若把它逐条写进结果文件，
# 同一份报告会被上千条记录重复存储，两个文件合计 553 MB——超过 GitHub 单文件
# 100 MB 上限，读者也得为核对一个数字下几百兆。因此这类记录只存文档标识，
# 上下文在评测时从 data/parsed/ 确定性重建。缓存按文档，重建一次即可。
_CTX_CACHE = {}


def _render_parsed(stem):
    """把解析后的年报渲染成线性文本。必须与 run_longctx.render_report 逐字一致，
    否则重建出的数值集合与原始运行不同，HAL 与 grounded 判定就会漂移。"""
    import os
    if stem not in _CTX_CACHE:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "parsed", stem + ".json")
        if not os.path.exists(path):
            _CTX_CACHE[stem] = None
        else:
            doc = json.load(open(path, encoding="utf-8"))
            out = [f"===== {doc['source']} ====="]
            for pg in doc["pages"]:
                out.append(f"\n--- 第 {pg['page']} 页 ---")
                for b in pg["text_blocks"]:
                    out.append(b)
                for t in pg["tables"]:
                    cap = t.get("caption_guess") or "(无表名)"
                    out.append(f"[表格 {t['table_id']}] {cap}")
                    if t["header"]:
                        out.append(" | ".join(t["header"]))
                    for r in t["rows"]:
                        if any(r):
                            out.append(" | ".join(r))
            _CTX_CACHE[stem] = "\n".join(out)
    return _CTX_CACHE[stem]


def _unit_text(u):
    """取该检索单元的文本；缺 text 时按 source 从 data/parsed/ 重建。

    重建失败必须硬失败。返回空串会让「上下文里没有任何数字」，于是所有答案
    都被判为无依据——准确率不受影响，幻觉率却会被显著抬高（实测 0.8% -> 5.8%），
    而且不报任何错。下载不完整的读者会得到错的数字并以为我们复现不了。
    """
    if "text" in u:
        return u["text"]
    stem = u.get("source", "")
    if stem.endswith(".pdf"):
        stem = stem[:-4]
    t = _render_parsed(stem)
    if t is None:
        raise FileNotFoundError(
            f"缺 data/parsed/{stem}.json，无法重建检索上下文。\n"
            f"长上下文实验的结果文件只存文档标识（整份年报逐条存会超 GitHub "
            f"单文件上限），评测时需要 data/parsed/ 才能还原。\n"
            f"请确认仓库完整检出，或从 Zenodo 归档获取 data/parsed/。")
    return t


def numbers_in_context(rec):
    """检索资料中出现的所有数值（原始字符串规格化）。"""
    nums = set()
    for u in rec["retrieved"]:
        for m in NUM_RE.finditer(_unit_text(u)):
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
