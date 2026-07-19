#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前沿商业模型对照基线（MiniMax-M3，OpenAI兼容API）。
复用已有检索上下文与评测口径，只把生成换成API调用；支持断点续传。

  export $(grep -v '^#' ~/.config/minimax.env | xargs)  # 或 source
  python run_api_baseline.py closedbook -o data/answers_m3_closedbook.jsonl
  python run_api_baseline.py naive      -o data/answers_m3_naive.jsonl
  python run_api_baseline.py v3ctx     -o data/answers_m3_v3ctx.jsonl

configs:
  closedbook  无检索闭卷
  naive       复用 answers_v2_naive_calc.jsonl 的朴素检索上下文 + 工具契约
  v3ctx       复用 answers_v3_full.jsonl 的全链路上下文 + 工具契约
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import requests

from run_e2e import (SYSTEM_PROMPT, CALC_INSTRUCTION, CMP_INSTRUCTION,
                     calc_postprocess, cmp_postprocess)

API = "https://api.minimaxi.com/v1/chat/completions"
MODEL = "MiniMax-M3"
KEY = os.environ.get("MINIMAX_API_KEY")
if not KEY:
    raise SystemExit("请先设置环境变量 MINIMAX_API_KEY（export MINIMAX_API_KEY=...）")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

CLOSEDBOOK_SYS = ("你是财报问答助手。请直接回答问题。回答尽量简短：数值题只给数值和单位；"
                  "比较题只给公司名；计算题给出计算结果保留两位小数。"
                  "如果不知道答案，回答：无法从资料中确定。")

usage_lock = Lock()
usage = {"prompt": 0, "completion": 0}


def call(messages, retries=5):
    for a in range(retries):
        try:
            r = requests.post(API, timeout=180,
                              headers={"Authorization": f"Bearer {KEY}",
                                       "Content-Type": "application/json"},
                              json={"model": MODEL, "messages": messages,
                                    "temperature": 0, "max_tokens": 3072})
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (a + 1))
                continue
            r.raise_for_status()
            d = r.json()
            with usage_lock:
                usage["prompt"] += d["usage"].get("prompt_tokens", 0)
                usage["completion"] += d["usage"].get("completion_tokens", 0)
            txt = d["choices"][0]["message"].get("content") or ""
            return THINK_RE.sub("", txt).strip()
        except requests.RequestException:
            time.sleep(5 * (a + 1))
    return "__API_FAILED__"


def build_record(rec, config):
    q = rec["question"]
    if config == "closedbook":
        msgs = [{"role": "system", "content": CLOSEDBOOK_SYS},
                {"role": "user", "content": q}]
    else:
        if rec["type"] == "yoy_compare":
            q = q + CALC_INSTRUCTION
        elif rec["type"] == "cross_company":
            q = q + CMP_INSTRUCTION
        ctx = "\n".join(f"[{i+1}] {u['text']}" for i, u in enumerate(rec["retrieved"]))
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"资料：\n{ctx}\n\n问题：{q}"}]
    raw = call(msgs)
    pred = raw
    if config != "closedbook":
        if rec["type"] == "yoy_compare":
            pred = calc_postprocess(raw)
        elif rec["type"] == "cross_company":
            pred = cmp_postprocess(raw, rec["question"],
                                   rec.get("meta", {}).get("companies", []))
    out = {"id": rec["id"], "type": rec["type"], "question": rec["question"],
           "gold": rec["gold"], "prediction": pred,
           "meta": rec.get("meta", {}),
           "retrieved": [] if config == "closedbook" else rec["retrieved"],
           "gold_evidence": rec["gold_evidence"], "raw_prediction": raw}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=["closedbook", "naive", "v3ctx"])
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    src = {"closedbook": "data/answers_v2_naive_calc.jsonl",
           "naive": "data/answers_v2_naive_calc.jsonl",
           "v3ctx": "data/answers_v3_full.jsonl"}[args.config]
    recs = [json.loads(l) for l in open(src, encoding="utf-8")]

    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")}
        print(f"断点续传：已完成 {len(done)} 条")
    todo = [r for r in recs if r["id"] not in done]

    write_lock = Lock()
    n_done = 0
    with out.open("a", encoding="utf-8") as fh:
        def work(rec):
            global n_done
            res = build_record(rec, args.config)
            with write_lock:
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                fh.flush()
                n_done += 1
                if n_done % 50 == 0:
                    print(f"{n_done}/{len(todo)} tokens(p/c)={usage['prompt']}/{usage['completion']}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))

    fails = sum(1 for l in open(out, encoding="utf-8")
                if "__API_FAILED__" in l)
    print(f"完成 {args.config} -> {out}; API失败 {fails} 条; "
          f"tokens prompt={usage['prompt']} completion={usage['completion']}")
