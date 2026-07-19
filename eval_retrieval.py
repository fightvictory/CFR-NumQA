#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索评测脚本：BM25（默认）/ 稠密检索（--dense，需本地装sentence-transformers）
在 qa_seed.jsonl 上评测两套语料的检索质量。

命中判定：检索到的单元与任一gold证据 同源文件且同页（表格题再要求table_id一致时用--strict）。
指标：Recall@k（k=1,3,5,10）——gold证据是否被检回。

用法：
    python eval_retrieval.py data/corpus/ data/qa_seed.jsonl                 # BM25
    python eval_retrieval.py data/corpus/ data/qa_seed.jsonl --dense         # bge稠密检索（本地）
依赖：pip install jieba rank_bm25
     稠密检索另需：pip install sentence-transformers torch
"""
import argparse
import json
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

KS = [1, 3, 5, 10]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def tokenize(s):
    return [w for w in jieba.lcut(s) if w.strip()]


class BM25Retriever:
    def __init__(self, units):
        self.units = units
        self.bm25 = BM25Okapi([tokenize(u["text"]) for u in units])

    def search(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.units[i] for i in idx]


class DenseRetriever:
    def __init__(self, units, model_name="BAAI/bge-small-zh-v1.5"):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.np = np
        self.units = units
        self.model = SentenceTransformer(model_name)
        self.emb = self.model.encode([u["text"] for u in units],
                                     normalize_embeddings=True,
                                     batch_size=128, show_progress_bar=True)

    def search(self, query, k):
        q = self.model.encode(["为这个句子生成表示以用于检索相关文章：" + query],
                              normalize_embeddings=True)
        scores = (self.emb @ q.T).ravel()
        idx = self.np.argsort(-scores)[:k]
        return [self.units[i] for i in idx]


def hit(unit, evidences, strict=False):
    for ev in evidences:
        if unit["source"].replace(".pdf", "") != ev["source"].replace(".pdf", ""):
            continue
        if unit["page"] != ev["page"]:
            continue
        if strict and ev.get("table_id") and unit.get("table_id") \
                and unit["table_id"] != ev["table_id"]:
            continue
        return True
    return False


def decompose_query(qa):
    """实体分解（oracle版）：多实体问题拆成 每公司+指标+年度 的子查询。
    试点用meta做oracle分解验证上限；正式方法中由LLM/规则从问题文本分解。"""
    m = qa.get("meta", {})
    companies = m.get("companies") or ([m["company"]] if m.get("company") else [])
    ind = m.get("indicator", "")
    year = m.get("year") or (m.get("years", [""])[-1])
    if len(companies) <= 1:
        return [qa["question"]]
    return [f"{c}{year}年{ind}" for c in companies]


def merged_search(retriever, qa, k, decompose=False):
    subqueries = decompose_query(qa) if decompose else [qa["question"]]
    if len(subqueries) == 1:
        return retriever.search(subqueries[0], k)
    per = retriever.search_multi(subqueries, k) if hasattr(retriever, "search_multi") \
        else [retriever.search(q, k) for q in subqueries]
    # 轮流合并（round-robin），去重
    merged, seen = [], set()
    for tier in zip(*per):
        for u in tier:
            if u["uid"] not in seen:
                seen.add(u["uid"])
                merged.append(u)
    return merged[:k]


def evaluate(retriever, qas, strict=False, decompose=False):
    recall = {k: 0 for k in KS}
    for qa in qas:
        results = merged_search(retriever, qa, max(KS), decompose)
        for k in KS:
            if any(hit(u, qa["evidence"], strict) for u in results[:k]):
                recall[k] += 1
    n = len(qas)
    return {f"Recall@{k}": round(recall[k] / n, 4) for k in KS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir")
    ap.add_argument("qa_file")
    ap.add_argument("--dense", action="store_true", help="使用bge稠密检索（需GPU/较大内存）")
    ap.add_argument("--strict", action="store_true", help="表格命中额外要求table_id一致")
    ap.add_argument("--by-type", action="store_true", help="按问题类型分组输出")
    ap.add_argument("--decompose", action="store_true", help="多实体问题做oracle实体分解后检索")
    args = ap.parse_args()

    qas = load_jsonl(args.qa_file)
    Retriever = DenseRetriever if args.dense else BM25Retriever
    method = "Dense(bge)" if args.dense else "BM25"

    print(f"评测: {method}, 问答数: {len(qas)}")
    for corpus_name in ["naive", "structural"]:
        units = load_jsonl(Path(args.corpus_dir) / f"{corpus_name}.jsonl")
        r = Retriever(units)
        res = evaluate(r, qas, args.strict, args.decompose)
        print(f"\n[{corpus_name}] ({len(units)} units) {res}")
        if args.by_type:
            from collections import defaultdict
            groups = defaultdict(list)
            for qa in qas:
                groups[qa["type"]].append(qa)
            for t, g in sorted(groups.items()):
                print(f"    {t:>14} (n={len(g)}): {evaluate(r, g, args.strict, args.decompose)}")


if __name__ == "__main__":
    main()
