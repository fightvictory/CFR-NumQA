#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统级基线（论文对比用，7B本地轻量实现）：
  closedbook  无检索闭卷回答（展示RAG价值）
  selfrag     Self-RAG风格：生成 -> 同模型自反思"是否被资料支持" -> 不支持则拒答
  crag        Corrective RAG风格：检索质量评估 -> 不充分则LLM改写查询重检 -> 合并再生成

统一使用 structural 语料 + 稠密检索 + oracle分解（与主方法检索起点对齐，
对比的是幻觉抑制机制本身）。输出schema与run_e2e一致，eval_answers/attribute_errors直接可用。

用法（训练机）：
  python run_baselines.py data/corpus/structural.jsonl data/qa_seed.jsonl \
      --baseline selfrag -o data/answers_bl_selfrag.jsonl
"""
import argparse
import json
from pathlib import Path

from run_e2e import (MODEL, EMB_MODEL, SYSTEM_PROMPT, TOP_K,
                     load_jsonl, decompose)

REFLECT_PROMPT = (
    "资料：\n{ctx}\n\n问题：{q}\n候选回答：{a}\n\n"
    "请判断候选回答是否完全由上述资料支持且正确。只输出一个词：支持 或 不支持。"
)
EVAL_CTX_PROMPT = (
    "资料：\n{ctx}\n\n问题：{q}\n\n"
    "请判断上述资料是否足以准确回答该问题。只输出一个词：充分 或 不充分。"
)
REWRITE_PROMPT = (
    "原始问题：{q}\n\n"
    "该问题在财报语料库中检索效果不佳。请改写为更适合检索财报表格的简短查询"
    "（保留公司名、年份、指标名，去掉疑问词），只输出一行改写后的查询。"
)


def build_dense_retriever(units):
    """bge稠密检索：语料GPU编码后转CPU numpy，查询编码走CPU，与vLLM共存。"""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL)
    emb = np.asarray(model.encode([u["text"] for u in units],
                                  normalize_embeddings=True,
                                  batch_size=256, show_progress_bar=True))
    model = model.to("cpu")
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    def search(query, k=TOP_K):
        q = model.encode(["为这个句子生成表示以用于检索相关文章：" + query],
                         normalize_embeddings=True)
        idx = np.argsort(-(emb @ q.T).ravel())[:k]
        return [units[i] for i in idx]
    return search


def retrieve(search, qa, k=TOP_K):
    subs = decompose(qa)
    if len(subs) == 1:
        return search(qa["question"], k)
    merged, seen = [], set()
    for tier in zip(*[search(s, k) for s in subs]):
        for u in tier:
            if u["uid"] not in seen:
                seen.add(u["uid"])
                merged.append(u)
    return merged  # 与主方法同用sub-quota（每子查询各自top-k）


def ctx_text(ctx):
    return "\n".join(f"[{i+1}] {u['text']}" for i, u in enumerate(ctx))


def chat_batch(llm, prompts, max_tokens=128, system=None):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    convs = []
    for p in prompts:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": p})
        convs.append(msgs)
    outs = llm.chat(convs, sp)
    return [o.outputs[0].text.strip() for o in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_file")
    ap.add_argument("qa_file")
    ap.add_argument("--baseline", required=True,
                    choices=["closedbook", "selfrag", "crag"])
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    qas = load_jsonl(args.qa_file)
    ABSTAIN = "无法从资料中确定。"

    contexts = [[] for _ in qas]
    search = None
    if args.baseline != "closedbook":
        units = load_jsonl(args.corpus_file)
        print(f"语料 {len(units)} units")
        search = build_dense_retriever(units)
        contexts = [retrieve(search, qa) for qa in qas]

    from vllm import LLM
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=0.8)

    if args.baseline == "closedbook":
        sys_p = ("你是财报问答助手。请直接回答问题。回答尽量简短：数值题只给数值和单位；"
                 "比较题只给公司名；计算题给出计算结果保留两位小数。"
                 "如果不知道答案，回答：无法从资料中确定。")
        preds = chat_batch(llm, [qa["question"] for qa in qas], system=sys_p)
        extras = [{} for _ in qas]

    elif args.baseline == "selfrag":
        # 第一步：常规RAG生成
        prompts = [f"资料：\n{ctx_text(c)}\n\n问题：{qa['question']}"
                   for qa, c in zip(qas, contexts)]
        answers = chat_batch(llm, prompts, system=SYSTEM_PROMPT)
        # 第二步：自反思
        refl = chat_batch(llm, [REFLECT_PROMPT.format(ctx=ctx_text(c), q=qa["question"], a=a)
                                for qa, c, a in zip(qas, contexts, answers)],
                          max_tokens=8)
        preds, extras = [], []
        for a, r in zip(answers, refl):
            supported = not r.strip().startswith("不支持")
            preds.append(a if supported else ABSTAIN)
            extras.append({"raw_prediction": a, "reflection": r.strip()[:20]})

    else:  # crag
        # 第一步：检索质量评估
        ev = chat_batch(llm, [EVAL_CTX_PROMPT.format(ctx=ctx_text(c), q=qa["question"])
                              for qa, c in zip(qas, contexts)], max_tokens=8)
        insufficient = [i for i, e in enumerate(ev) if e.strip().startswith("不充分")]
        print(f"检索评估：{len(insufficient)}/{len(qas)} 判为不充分，触发纠错")
        # 第二步：查询改写 + 重检索 + 合并
        if insufficient:
            rewrites = chat_batch(llm, [REWRITE_PROMPT.format(q=qas[i]["question"])
                                        for i in insufficient], max_tokens=48)
            for i, rw in zip(insufficient, rewrites):
                extra_units = search(rw.splitlines()[0][:80], TOP_K)
                seen = {u["uid"] for u in contexts[i]}
                contexts[i] = contexts[i] + [u for u in extra_units
                                             if u["uid"] not in seen]
        # 第三步：生成
        prompts = [f"资料：\n{ctx_text(c)}\n\n问题：{qa['question']}"
                   for qa, c in zip(qas, contexts)]
        preds = chat_batch(llm, prompts, system=SYSTEM_PROMPT)
        extras = [{"crag_eval": e.strip()[:10]} for e in ev]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for qa, c, p, ex in zip(qas, contexts, preds, extras):
            rec = {"id": qa["id"], "type": qa["type"],
                   "question": qa["question"], "gold": qa["answer"],
                   "prediction": p, "meta": qa.get("meta", {}),
                   "retrieved": [{"uid": u["uid"], "source": u["source"],
                                  "page": u["page"], "text": u["text"]} for u in c],
                   "gold_evidence": qa["evidence"], **ex}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"完成，答案写入 {out}")


if __name__ == "__main__":
    main()
