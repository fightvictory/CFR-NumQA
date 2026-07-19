#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap置信区间（论文统计严谨性用，无GPU纯本地计算）。
  python eval_bootstrap.py data/answers_v3_full.jsonl [...]        # 各文件指标95%CI
  python eval_bootstrap.py --diff A.jsonl B.jsonl                  # 配对差值CI（A-B）

方法：对1016条问答做有放回重采样（B=10000），percentile法95%CI。
配对差值在相同重采样索引上计算（配对bootstrap），CI不含0即p<0.05水平显著。
"""
import argparse
import json

import random

from eval_answers import ABSTAIN_RE, is_correct, is_grounded

B = 10000
random.seed(42)


def load_indicators(path):
    """每条 -> (correct, abstain, hal)，口径与eval_answers完全一致。"""
    out = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if ABSTAIN_RE.search(r["prediction"]):
            out.append((0, 1, 0))
        elif is_correct(r):
            out.append((1, 0, 0))
        elif is_grounded(r):
            out.append((0, 0, 0))
        else:
            out.append((0, 0, 1))
    return out


def pct_ci(samples):
    s = sorted(samples)
    return s[int(0.025 * len(s))], s[int(0.975 * len(s))]


def boot_means(ind, idx_sets):
    n = len(ind)
    accs, abss, hals = [], [], []
    for idx in idx_sets:
        a = c = h = 0
        for i in idx:
            c += ind[i][0]; a += ind[i][1]; h += ind[i][2]
        accs.append(c / n); abss.append(a / n); hals.append(h / n)
    return accs, abss, hals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--diff", action="store_true", help="两个文件的配对差值CI（前-后）")
    args = ap.parse_args()

    inds = [load_indicators(f) for f in args.files]
    n = len(inds[0])
    assert all(len(x) == n for x in inds), "配对要求相同条数与顺序"
    idx_sets = [[random.randrange(n) for _ in range(n)] for _ in range(B)]

    if args.diff:
        assert len(args.files) == 2
        (a1, b1, h1), (a2, b2, h2) = (boot_means(x, idx_sets) for x in inds)
        for name, s1, s2 in [("准确率", a1, a2), ("拒答率", b1, b2), ("HAL", h1, h2)]:
            d = [x - y for x, y in zip(s1, s2)]
            lo, hi = pct_ci(d)
            point = sum(x[0] for x in inds[0]) / n - sum(x[0] for x in inds[1]) / n \
                if name == "准确率" else sum(d) / len(d)
            sig = "显著" if lo > 0 or hi < 0 else "不显著"
            print(f"  Δ{name} = {sum(d)/len(d):+.1%}  95%CI [{lo:+.1%}, {hi:+.1%}]  {sig}")
        return

    print(f"{'文件':<42} {'准确率[95%CI]':<24} {'拒答率':<22} {'HAL':<20}")
    for f, ind in zip(args.files, inds):
        accs, abss, hals = boot_means(ind, idx_sets)
        pa = sum(x[0] for x in ind) / n
        pb = sum(x[1] for x in ind) / n
        ph = sum(x[2] for x in ind) / n
        (al, ah), (bl, bh), (hl, hh) = pct_ci(accs), pct_ci(abss), pct_ci(hals)
        print(f"{f:<42} {pa:.1%}[{al:.1%},{ah:.1%}]   {pb:.1%}[{bl:.1%},{bh:.1%}]   {ph:.1%}[{hl:.1%},{hh:.1%}]")


if __name__ == "__main__":
    main()
