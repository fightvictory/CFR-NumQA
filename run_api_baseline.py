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

# 各家推理模型的思维链返回方式不同：MiniMax内联<think>标签，DeepSeek用独立字段。
# 注意：推理token计入max_tokens，给小了会拿到空content，故推理模型上限放宽。
PROVIDERS = {
    "minimax": dict(api="https://api.minimaxi.com/v1/chat/completions",
                    model="MiniMax-M3", key_env="MINIMAX_API_KEY",
                    max_tokens=3072, inline_think=True),
    "deepseek": dict(api="https://api.deepseek.com/v1/chat/completions",
                     model="deepseek-v4-pro", key_env="DEEPSEEK_API_KEY",
                     max_tokens=8192, inline_think=False),
}
PROV = {}
API = MODEL = KEY = None
MAX_TOKENS = 3072
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def init_provider(name):
    global PROV, API, MODEL, KEY, MAX_TOKENS
    PROV = PROVIDERS[name]
    API, MODEL = PROV["api"], PROV["model"]
    MAX_TOKENS = PROV["max_tokens"]
    KEY = os.environ.get(PROV["key_env"])
    if not KEY:
        raise SystemExit(f"请先设置环境变量 {PROV['key_env']}")

CLOSEDBOOK_SYS = ("你是财报问答助手。请直接回答问题。回答尽量简短：数值题只给数值和单位；"
                  "比较题只给公司名；计算题给出计算结果保留两位小数。"
                  "如果不知道答案，回答：无法从资料中确定。")

usage_lock = Lock()
usage = {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0,
         "truncated": 0, "empty": 0, "n": 0}


def call(messages, retries=5):
    for a in range(retries):
        try:
            r = requests.post(API, timeout=300,
                              headers={"Authorization": f"Bearer {KEY}",
                                       "Content-Type": "application/json"},
                              json={"model": MODEL, "messages": messages,
                                    "temperature": 0, "max_tokens": MAX_TOKENS})
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (a + 1))
                continue
            r.raise_for_status()
            d = r.json()
            ch = d["choices"][0]
            u = d.get("usage", {})
            with usage_lock:
                usage["n"] += 1
                usage["prompt"] += u.get("prompt_tokens", 0)
                usage["completion"] += u.get("completion_tokens", 0)
                usage["reasoning"] += (u.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens", 0)
                usage["cached"] += u.get("prompt_cache_hit_tokens", 0)
                if ch.get("finish_reason") == "length":
                    usage["truncated"] += 1
            txt = ch["message"].get("content") or ""
            if PROV["inline_think"]:
                txt = THINK_RE.sub("", txt)
            txt = txt.strip()
            if not txt:
                # 推理耗尽max_tokens导致无答案；计数以便发现上限设置过低
                with usage_lock:
                    usage["empty"] += 1
            return txt
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
    ap.add_argument("--provider", choices=list(PROVIDERS), default="minimax")
    ap.add_argument("--limit", type=int, default=0, help="只跑前N条（试点计量用）")
    args = ap.parse_args()
    init_provider(args.provider)

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
    if args.limit:
        todo = todo[:args.limit]

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
    n = max(1, usage["n"])
    print(f"\n完成 {args.config} ({args.provider}) -> {out}; API失败 {fails} 条")
    print(f"调用 {usage['n']} 次 | prompt={usage['prompt']} (缓存命中 {usage['cached']}) "
          f"completion={usage['completion']} 其中推理={usage['reasoning']}")
    print(f"均值/题: prompt {usage['prompt']/n:.0f}, completion {usage['completion']/n:.0f} "
          f"(推理 {usage['reasoning']/n:.0f}) | 截断 {usage['truncated']} 条, 空答案 {usage['empty']} 条")
