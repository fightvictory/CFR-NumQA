#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""计算两位标注者的一致性：一致率、Cohen's kappa、Gwet's AC1，附 bootstrap 置信区间。

为什么同时报 kappa 和 AC1
-------------------------
本数据集的抽检判定极度偏斜——随机样本上判定几乎全是 ✓。在这种边缘分布下
Cohen's kappa 会塌陷（kappa 悖论：一致率 99%，kappa 却可能低到 0.4 甚至无定义），
因为它把"两人都倾向于选同一个多数类"当成了偶然一致。Gwet's AC1 正是为这种情形
设计的，对偏斜边缘分布稳健。

  Gwet, K.L. (2008). Computing inter-rater reliability and its variance in the
  presence of high agreement. Br J Math Stat Psychol 61(1):29-48.

所以：随机样本报一致率（+AC1），难例样本报 kappa 与 AC1。只报 kappa 会低估真实
一致性，只报一致率又会被质疑没排除偶然一致——两个一起报才站得住。

用法：
    python agreement.py                       # 读默认的三份工作簿
    python agreement.py --a 甲.xlsx --b 乙.xlsx
"""
import argparse
import random
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
B = 10000
CATS = ("✓", "✗", "存疑")


def read_sheet(path, sheet_name=None):
    """读 xlsx 的一个页签，返回 {id: {"H":..., "I":..., "J":...}}。
    只认第 5 行起的数据（第 3 行表头、第 4 行示例）。"""
    try:
        z = zipfile.ZipFile(path)
    except FileNotFoundError:
        return None
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    sheets = [s.get("name") for s in wb.iter() if s.tag.endswith("}sheet")]
    if sheet_name is None:
        i = 1
    elif sheet_name in sheets:
        i = sheets.index(sheet_name) + 1
    else:
        return None
    root = ET.fromstring(z.read(f"xl/worksheets/sheet{i}.xml"))
    out = {}
    for r in root.findall(".//m:row", NS):
        n = int(r.get("r", 0))
        if n < 5:
            continue
        cells = {}
        for c in r.findall("m:c", NS):
            col = "".join(ch for ch in c.get("r", "") if ch.isalpha())
            t = c.find("m:is/m:t", NS)
            v = c.find("m:v", NS)
            cells[col] = ((t.text if t is not None else
                           (v.text if v is not None else "")) or "").strip()
        qid = cells.get("B", "")
        if qid.startswith("seed_"):
            out[qid] = {k: cells.get(k, "") for k in ("H", "I", "J")}
    return out


def p_a(pairs):
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def cohen_kappa(pairs, cats):
    n = len(pairs)
    obs = p_a(pairs)
    m1 = Counter(x for x, _ in pairs)
    m2 = Counter(y for _, y in pairs)
    exp = sum((m1[c] / n) * (m2[c] / n) for c in cats)
    if abs(1 - exp) < 1e-12:
        return None                      # 边缘分布退化，kappa 无定义
    return (obs - exp) / (1 - exp)


def gwet_ac1(pairs, cats):
    n = len(pairs)
    q = len(cats)
    if q < 2:
        return None
    obs = p_a(pairs)
    m1 = Counter(x for x, _ in pairs)
    m2 = Counter(y for _, y in pairs)
    pi = {c: (m1[c] + m2[c]) / (2 * n) for c in cats}
    exp = sum(pi[c] * (1 - pi[c]) for c in cats) / (q - 1)
    if abs(1 - exp) < 1e-12:
        return None
    return (obs - exp) / (1 - exp)


def boot_ci(pairs, fn, cats, seed=42):
    """对条目做有放回重采样，与论文其余 CI 口径一致（B=10000, seed=42）。"""
    random.seed(seed)
    n = len(pairs)
    vals = []
    for _ in range(B):
        s = [pairs[random.randrange(n)] for _ in range(n)]
        v = fn(s, cats)
        if v is not None:
            vals.append(v)
    if len(vals) < B * 0.5:              # 过半重采样退化 -> CI 无意义
        return None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def report(label, a, b, col, colname):
    ids = sorted(set(a) & set(b))
    pairs = [(a[i][col], b[i][col]) for i in ids
             if a[i][col] and b[i][col]]
    if not pairs:
        print(f"  {label} · {colname}: 尚无双方都已填写的条目")
        return None
    cats = tuple(c for c in CATS if any(c in p for p in pairs))
    obs = p_a(pairs)
    k = cohen_kappa(pairs, cats)
    ac = gwet_ac1(pairs, cats)
    d1, d2 = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)

    print(f"  {label} · {colname}   n={len(pairs)}")
    print(f"    边缘分布  A: {dict(d1)}    B: {dict(d2)}")
    print(f"    一致率    {obs*100:.1f}%  （{sum(1 for x,y in pairs if x==y)}/{len(pairs)}）")
    if k is None:
        print(f"    Cohen κ   无定义（只出现 {len(cats)} 个类别，边缘分布退化）")
    else:
        ci = boot_ci(pairs, cohen_kappa, cats)
        s = f"  95%CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "  （CI 因重采样退化未给出）"
        print(f"    Cohen κ   {k:.3f}{s}")
    if ac is None:
        print(f"    Gwet AC1  无定义")
    else:
        ci = boot_ci(pairs, gwet_ac1, cats)
        s = f"  95%CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "  （CI 因重采样退化未给出）"
        print(f"    Gwet AC1  {ac:.3f}{s}")
    dis = [(i, a[i][col], b[i][col]) for i in ids
           if a[i][col] and b[i][col] and a[i][col] != b[i][col]]
    if dis:
        print(f"    分歧 {len(dis)} 条：" + ", ".join(f"{i}({x}/{y})" for i, x, y in dis[:8])
              + (" …" if len(dis) > 8 else ""))
    print()
    return {"n": len(pairs), "pa": obs, "kappa": k, "ac1": ac, "dis": len(dis)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="抽检表_标注者A.xlsx")
    ap.add_argument("--b", default="抽检表_标注者B.xlsx")
    ap.add_argument("--a-main", default="../pilot_toolkit/抽检表_人工校验100条_预审版.xlsx",
                    help="标注者A首轮的主样本标注（单页签）")
    args = ap.parse_args()

    a_hard = read_sheet(args.a, "难例样本")
    a_recheck = read_sheet(args.a, "复核4条") or {}
    b_hard = read_sheet(args.b, "难例样本")
    b_main = read_sheet(args.b, "主样本100条")
    a_main = read_sheet(args.a_main)

    missing = [n for n, v in [(args.a, a_hard), (args.b, b_hard)] if v is None]
    if missing:
        sys.exit(f"!! 读不到：{missing}。先运行 make_audit2.py 生成工作簿。")

    if a_main and a_recheck:
        # 首轮标注针对的是修复前的数据，被修复过的条目以复核结果为准
        for i, v in a_recheck.items():
            if v["H"] or v["I"]:
                a_main[i] = v

    print("标注一致性（两位标注者独立判定）\n")
    print("难例样本 —— 刻意挑选可争议条目，系数在此才有变异可衡量")
    h1 = report("难例", a_hard, b_hard, "H", "答案正确?")
    h2 = report("难例", a_hard, b_hard, "I", "唯一合理答案?")

    if a_main and b_main:
        print("主样本 —— 分层随机 100 条，代表整体数据质量")
        m1 = report("主样本", a_main, b_main, "H", "答案正确?")
        m2 = report("主样本", a_main, b_main, "I", "唯一合理答案?")
    else:
        m1 = m2 = None
        print("主样本：标注者B 尚未填写（或找不到标注者A的首轮文件），跳过\n")

    done = [x for x in (h1, h2, m1, m2) if x]
    if done:
        print("可直接写进论文的一句：")
        if h1 and h1["kappa"] is not None:
            print(f"  难例样本上两位标注者的 Cohen κ = {h1['kappa']:.2f}"
                  f"（答案正确性）/ {h2['kappa']:.2f}（问题无歧义性）" if h2 and h2["kappa"] is not None
                  else f"  难例样本 Cohen κ = {h1['kappa']:.2f}")
        if m1:
            print(f"  随机样本一致率 {m1['pa']*100:.0f}% / {m2['pa']*100:.0f}%")
    else:
        print("尚无已完成的判定——两位标注者填好黄色列后再运行本脚本。")


if __name__ == "__main__":
    main()
