#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
证据覆盖率评测：gold数值是否全部进入检索上下文（严口径，直接对应下游可答性）。
    python eval_context.py data/ctx_baseline.jsonl [more.jsonl ...]
"""
import json
import sys
from collections import defaultdict

from attribute_errors import ctx_values, gold_values, in_ctx


def coverage(path):
    stats = defaultdict(lambda: [0, 0])
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        golds = gold_values(r)
        if not golds:
            continue
        c = ctx_values(r)
        full = all(in_ctx(v, c) for v in golds)
        for key in ("ALL", r["type"]):
            stats[key][0] += full
            stats[key][1] += 1
    return stats


def main():
    files = sys.argv[1:]
    keys = ["ALL", "cross_company", "extraction", "yoy_compare"]
    print(f"{'证据全覆盖率':<38}" + "".join(f"{k:>15}" for k in keys))
    for f in files:
        s = coverage(f)
        row = "".join(f"{s[k][0]/max(1,s[k][1]):>14.1%} " if s[k][1] else f"{'-':>15}"
                      for k in keys)
        print(f"{f:<38}{row}")


if __name__ == "__main__":
    main()
