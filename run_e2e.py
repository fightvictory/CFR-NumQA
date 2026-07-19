#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端RAG问答脚本（训练机运行，需GPU）
流程：检索top-k -> 拼接上下文 -> vLLM本地生成 -> 保存答案（含检索溯源，供幻觉评测）

用法（在 ~/finrag/pilot_toolkit 下）：
    python run_e2e.py data/corpus/structural.jsonl data/qa_seed.jsonl -o data/answers_structural.jsonl
    python run_e2e.py data/corpus/naive.jsonl      data/qa_seed.jsonl -o data/answers_naive.jsonl

依赖：pip install vllm sentence-transformers jieba rank_bm25
显存：Qwen2.5-7B-Instruct-AWQ 约需 6-7GB，加bge与KV cache，16GB足够
如HuggingFace不通：export HF_ENDPOINT=https://hf-mirror.com
"""
import argparse
import json
import re
from pathlib import Path

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
EMB_MODEL = "BAAI/bge-small-zh-v1.5"
TOP_K = 5

SYSTEM_PROMPT = (
    "你是严谨的财报问答助手。只能根据提供的资料回答问题，禁止使用资料以外的知识。"
    "如果资料中找不到答案，必须回答：无法从资料中确定。"
    "回答尽量简短：数值题只给数值和单位；比较题只给公司名；"
    "计算题（如增长率）给出计算结果，保留两位小数。"
)


CALC_INSTRUCTION = (
    "\n\n这是同比计算题，不要自己计算增长率。"
    "请从资料中找出题目所问指标在两个年度的数值，严格按以下格式输出一行（不要输出其他内容）：\n"
    "旧值=<较早年度的数值>；新值=<较晚年度的数值>\n"
    "数值照抄资料原文（可含千分位逗号），不带单位。"
    "若任一年度的数值在资料中找不到，回答：无法从资料中确定。"
)

CALC_NUM_RE = re.compile(r"(-?[\d,，]+\.?\d*)")

CMP_INSTRUCTION = (
    "\n\n这是跨公司对比题，不要直接回答公司名。"
    "请从资料中分别找出两家公司该指标的数值，严格按以下格式输出一行（不要输出其他内容）：\n"
    "<公司名>=<数值><单位>；<公司名>=<数值><单位>\n"
    "数值和单位照抄资料原文（单位如 元/千元/百万元，注意有的单位写在表名或行名里）。"
    "若任一公司的数值在资料中找不到，回答：无法从资料中确定。"
)

CMP_UNIT_SCALE = {"元": 1, "千元": 1e3, "万元": 1e4, "百万元": 1e6, "亿元": 1e8}
CMP_PAIR_RE = re.compile(r"([^\s=；;，,]+)\s*[=＝]\s*(-?[\d,，]+\.?\d*)\s*(亿元|百万元|万元|千元|元)?")


def cmp_postprocess(raw, question, companies):
    """从'公司A=数值单位；公司B=数值单位'解析，量纲归一后由程序比较；解析失败原样返回。"""
    found = {}
    for name, num, unit in CMP_PAIR_RE.findall(raw):
        comp = next((c for c in companies if c in name), None)
        if comp is None or comp in found:
            continue
        try:
            v = float(num.replace(",", "").replace("，", ""))
        except ValueError:
            continue
        found[comp] = v * CMP_UNIT_SCALE.get(unit, 1)
    if len(found) != 2:
        return raw
    lower = bool(re.search(r"更低|更少|更小", question))
    pick = min(found, key=found.get) if lower else max(found, key=found.get)
    return pick


def calc_postprocess(raw):
    """从'旧值=X；新值=Y'解析操作数并程序化计算增长率；解析失败则原样返回。"""
    m_old = re.search(r"旧值\s*[=＝:：]\s*" + CALC_NUM_RE.pattern, raw)
    m_new = re.search(r"新值\s*[=＝:：]\s*" + CALC_NUM_RE.pattern, raw)
    if not (m_old and m_new):
        return raw
    try:
        old = float(m_old.group(1).replace(",", "").replace("，", ""))
        new = float(m_new.group(1).replace(",", "").replace("，", ""))
    except ValueError:
        return raw
    if old == 0:
        return raw
    return f"{(new - old) / abs(old) * 100:+.2f}%"


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


YEAR_RE = re.compile(r"(20\d\d)")


def unit_meta(units):
    """从source文件名解析每个索引单元的(公司, 年报年份)。"""
    metas = []
    for u in units:
        src = u["source"]
        parts = Path(src).stem.split("_")
        company = parts[1] if len(parts) >= 2 else ""
        m = YEAR_RE.search(parts[2] if len(parts) >= 3 else src)
        metas.append((company, int(m.group(1)) if m else 0))
    return metas


def query_filter_mask(query, metas, companies_all, np):
    """规则解析查询中的公司与年份（非oracle），返回候选池布尔掩码。
    年份y的数据也出现在y+1年报的同比列，故允许{y, y+1}。"""
    comps = [c for c in companies_all if c in query]
    years = set()
    for y in YEAR_RE.findall(query):
        years.add(int(y))
        years.add(int(y) + 1)
    if not comps and not years:
        return None
    mask = np.ones(len(metas), dtype=bool)
    if comps:
        cs = set(comps)
        mask &= np.array([m[0] in cs for m in metas])
    if years:
        mask &= np.array([m[1] in years for m in metas])
    if not mask.any():
        return None  # 过滤过狠时退回全库
    return mask


def build_retriever(units, hybrid=False, filter_meta=False):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer(EMB_MODEL)
    emb = model.encode([u["text"] for u in units], normalize_embeddings=True,
                       batch_size=256, show_progress_bar=True)

    metas = unit_meta(units) if filter_meta else None
    companies_all = sorted({m[0] for m in metas}) if metas else []

    bm25 = None
    if hybrid:
        import jieba
        from rank_bm25 import BM25Okapi
        print("构建BM25索引（jieba分词）...")
        bm25 = BM25Okapi([jieba.lcut(u["text"]) for u in units])

    def search(query, k=TOP_K):
        q = model.encode(["为这个句子生成表示以用于检索相关文章：" + query],
                         normalize_embeddings=True)
        dense = (emb @ q.T).ravel()
        mask = query_filter_mask(query, metas, companies_all, np) if filter_meta else None
        if mask is not None:
            dense = np.where(mask, dense, -1e9)
        if bm25 is None:
            idx = np.argsort(-dense)[:k]
            return [units[i] for i in idx]
        # RRF融合：1/(60+rank_dense) + 1/(60+rank_bm25)
        import jieba
        bs = np.asarray(bm25.get_scores(jieba.lcut(query)))
        if mask is not None:
            bs = np.where(mask, bs, -1e9)
        r_d = np.empty(len(dense)); r_d[np.argsort(-dense)] = np.arange(len(dense))
        r_b = np.empty(len(bs)); r_b[np.argsort(-bs)] = np.arange(len(bs))
        rrf = 1 / (60 + r_d) + 1 / (60 + r_b)
        idx = np.argsort(-rrf)[:k]
        return [units[i] for i in idx]
    return search


def decompose(qa):
    """与eval_retrieval.py一致的oracle实体分解（多实体题拆子查询）。"""
    m = qa.get("meta", {})
    companies = m.get("companies") or []
    if len(companies) < 2:
        return [qa["question"]]
    ind = m.get("indicator", "")
    year = m.get("year", "")
    return [f"{c}{year}年{ind}" for c in companies]


def decompose_auto(qa, companies_all):
    """非oracle分解：公司名从问题文本按语料公司表匹配，
    子查询=原问题去掉其他公司名（自然保留年份与指标词）。"""
    q = qa["question"]
    comps = [c for c in companies_all if c in q]
    if len(comps) < 2:
        return [q]
    subs = []
    for c in comps:
        s = q
        for o in comps:
            if o != c:
                s = s.replace(o, "")
        subs.append(s.replace("和", "").replace("与", ""))
    return subs


def retrieve_context(search, qa, k=TOP_K, sub_quota=False, decomp_fn=None):
    subs = decomp_fn(qa) if decomp_fn else decompose(qa)
    if len(subs) == 1:
        return search(qa["question"], k)
    merged, seen = [], set()
    for tier in zip(*[search(s, k) for s in subs]):
        for u in tier:
            if u["uid"] not in seen:
                seen.add(u["uid"])
                merged.append(u)
    # sub_quota: 每个子查询保留各自top-k（上下文最多len(subs)*k条），
    # 避免多实体题合并截断后每家公司只剩2-3条证据
    return merged if sub_quota else merged[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_file")
    ap.add_argument("qa_file")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--calc", action="store_true",
                    help="计算器增强：同比题模型只输出两个操作数，增长率由程序计算")
    ap.add_argument("--sub-quota", action="store_true",
                    help="多实体题每个子查询保留各自top-k，不合并截断")
    ap.add_argument("--filter-meta", action="store_true",
                    help="规则解析查询中的公司/年份，过滤候选池（非oracle）")
    ap.add_argument("--hybrid", action="store_true",
                    help="BM25+稠密RRF混合检索")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="只做检索并写出上下文，不加载生成模型（检索消融用）")
    ap.add_argument("--auto-decompose", action="store_true",
                    help="多实体分解用规则解析问题文本（非oracle meta）")
    args = ap.parse_args()

    units = load_jsonl(args.corpus_file)
    qas = load_jsonl(args.qa_file)
    print(f"语料 {len(units)} units, 问答 {len(qas)} 条")

    print("构建检索索引...")
    search = build_retriever(units, hybrid=args.hybrid, filter_meta=args.filter_meta)
    decomp_fn = None
    if args.auto_decompose:
        companies_all = sorted({m[0] for m in unit_meta(units)})
        decomp_fn = lambda qa: decompose_auto(qa, companies_all)
    contexts = [retrieve_context(search, qa, args.top_k, args.sub_quota, decomp_fn)
                for qa in qas]

    if args.retrieval_only:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for qa, ctx in zip(qas, contexts):
                fh.write(json.dumps({
                    "id": qa["id"], "type": qa["type"],
                    "question": qa["question"], "gold": qa["answer"],
                    "prediction": "", "meta": qa.get("meta", {}),
                    "retrieved": [{"uid": u["uid"], "source": u["source"],
                                   "page": u["page"], "text": u["text"]} for u in ctx],
                    "gold_evidence": qa["evidence"],
                }, ensure_ascii=False) + "\n")
        print(f"检索上下文已写入 {out}")
        return

    # 检索全部完成，释放bge模型和向量占用的显存，给vLLM腾地方
    del search
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    print(f"加载生成模型 {args.model} ...")
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=128)

    conversations = []
    for qa, ctx in zip(qas, contexts):
        ctx_text = "\n".join(f"[{i+1}] {u['text']}" for i, u in enumerate(ctx))
        question = qa["question"]
        if args.calc and qa["type"] == "yoy_compare":
            question += CALC_INSTRUCTION
        elif args.calc and qa["type"] == "cross_company":
            question += CMP_INSTRUCTION
        conversations.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"资料：\n{ctx_text}\n\n问题：{question}"},
        ])
    outputs = llm.chat(conversations, sp)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for qa, ctx, o in zip(qas, contexts, outputs):
            raw = o.outputs[0].text.strip()
            pred = raw
            rec = {
                "id": qa["id"], "type": qa["type"],
                "question": qa["question"], "gold": qa["answer"],
                "prediction": pred,
                "meta": qa.get("meta", {}),
                "retrieved": [{"uid": u["uid"], "source": u["source"],
                               "page": u["page"], "text": u["text"]} for u in ctx],
                "gold_evidence": qa["evidence"],
            }
            if args.calc and qa["type"] == "yoy_compare":
                rec["prediction"] = calc_postprocess(raw)
                rec["raw_prediction"] = raw
            elif args.calc and qa["type"] == "cross_company":
                rec["prediction"] = cmp_postprocess(
                    raw, qa["question"], qa.get("meta", {}).get("companies", []))
                rec["raw_prediction"] = raw
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"完成，答案写入 {out}")


if __name__ == "__main__":
    main()
