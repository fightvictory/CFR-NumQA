#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
长上下文直读基线（外审 07-31 A2.3）。

问题：60 页年报完全塞得进前沿模型的上下文窗口，那检索管线还值不值 +22.6 pp？
不做这个对照，该结论会被质疑。

做法：把问题所需年报的**完整解析内容**直接交给模型，不做任何检索。喂的是
解析后的内容而非 PDF 原文，这样与管线用的是同一批信息，隔离出来的才是
「检索 vs 不检索」，而不是「解析质量」。

**这是 oracle 文档选择**：直接按 gold 证据的来源挑报告，等于替模型免去了
「该读哪一份」。这对基线是有利的设定，我们有意让它有利——若模型拿到正确
文档仍答不对，结论才干净。

成本控制：同一份年报会被平均问 7 次，按文档分组连发即可吃满前缀缓存
（缓存命中价是未命中的 1/120）。每组先单发一条把缓存捂热，其余再并发。
思考模式关闭——这是「直接读文档能不能答对」的测试，不是推理能力测试，
且推理 token 按输出计费。

用法：
    export DEEPSEEK_API_KEY=...
    python run_longctx.py data/qa_seed.jsonl -o results/answers_longctx_ds.jsonl

断点续跑：输出文件已有的 id 会被跳过，直接重跑同一条命令即可。
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_e2e import (SYSTEM_PROMPT, CALC_INSTRUCTION, CMP_INSTRUCTION,
                     calc_postprocess, cmp_postprocess, load_jsonl)

API = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-pro"
# 模块相对路径：从任何工作目录调用都能定位，也让对账脚本可跨目录导入本模块
PARSED = Path(__file__).resolve().parent / "data" / "parsed"

usage = {"prompt": 0, "completion": 0, "cached": 0, "reasoning": 0,
         "calls": 0, "failed": 0}


def render_report(stem):
    """把解析后的年报渲染成线性文本。与索引用的是同一份解析产物。"""
    path = PARSED / (stem + ".json")
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
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
    return "\n".join(out)


YEAR_RE = __import__("re").compile(r"(20\d{2})")
_DOCS = None


def _all_docs():
    """全部报告的 (stem -> (公司, 年报年份))，与 run_e2e.unit_meta 同规则。"""
    global _DOCS
    if _DOCS is None:
        _DOCS = {}
        for f in PARSED.glob("*.json"):
            parts = f.stem.split("_")
            m = YEAR_RE.search(parts[2] if len(parts) >= 3 else f.stem)
            _DOCS[f.stem] = (parts[1] if len(parts) >= 2 else "",
                             int(m.group(1)) if m else 0)
    return _DOCS


def docs_for(qa, mode="oracle"):
    """该题需要读的报告。

    oracle: 按 gold 证据的来源直接挑（替模型免去「该读哪份」，对基线有利）。
    filter: 用与 run_e2e.query_filter_mask 同一套规则从问句解析公司与年份，
            这是部署时真正可得的信息。

    filter 模式下多一层兜底：年份筛空时退回「该公司全部年报」。原过滤器此时
    会退回全库（103 份 = 410 万 tokens，超 1M 窗口），且这是它在 88/1016 道题
    上静默失效的原因——那些题问 2020 年数字，而窗口只放宽到 {y, y+1}，
    语料里并无 2020/2021 年报。
    """
    if mode == "oracle":
        return tuple(sorted({e["source"].replace(".pdf", "")
                             for e in qa.get("evidence", [])}))
    docs = _all_docs()
    q = qa["question"]
    comps_all = sorted({v[0] for v in docs.values()})
    cs = [c for c in comps_all if c in q]
    ys = set()
    for y in YEAR_RE.findall(q):
        ys.add(int(y))
        ys.add(int(y) + 1)
    sel = [s for s, (c, y) in docs.items()
           if (not cs or c in cs) and (not ys or y in ys)]
    if not sel and cs:                       # 兜底：只按公司
        sel = [s for s, (c, _) in docs.items() if c in cs]
    return tuple(sorted(sel)) if sel else tuple(sorted(docs))


def call(key, messages, retries=5):
    # 惰性导入：渲染与分组不需要 HTTP 库，对账脚本只为核对 render_report
    # 而导入本模块时不应因缺 requests 而失败。
    import requests
    for i in range(retries):
        try:
            r = requests.post(
                API, timeout=300,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={"model": MODEL, "messages": messages,
                      "temperature": 0, "max_tokens": 512,
                      # 关思考：这是「直接读文档能不能答对」的测试，
                      # 且推理 token 按输出计费（¥6.18/M）。
                      "thinking": {"type": "disabled"}})
            if r.status_code != 200:
                if r.status_code in (429, 500, 502, 503) and i < retries - 1:
                    time.sleep(2 ** i)
                    continue
                return None, f"HTTP {r.status_code}: {r.text[:160]}"
            d = r.json()
            u = d.get("usage") or {}
            usage["prompt"] += u.get("prompt_tokens", 0)
            usage["completion"] += u.get("completion_tokens", 0)
            usage["cached"] += (u.get("prompt_tokens_details") or {}).get(
                "cached_tokens", 0)
            usage["reasoning"] += (u.get("completion_tokens_details") or {}).get(
                "reasoning_tokens", 0)
            usage["calls"] += 1
            return d["choices"][0]["message"].get("content", "").strip(), None
        except Exception as e:                      # noqa: BLE001
            if i == retries - 1:
                return None, str(e)[:160]
            time.sleep(2 ** i)
    return None, "retries exhausted"


def build_messages(qa, ctx):
    """报告正文放最前（可缓存的前缀），问题放最后。顺序不能颠倒，
    否则每题的前缀都不同，前缀缓存完全失效。"""
    instr = ""
    if qa["type"] == "yoy_compare":
        instr = "\n" + CALC_INSTRUCTION
    elif qa["type"] == "cross_company":
        instr = "\n" + CMP_INSTRUCTION
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"以下是完整年报内容：\n\n{ctx}\n\n"
                        f"====\n问题：{qa['question']}{instr}"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qa_file")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="冒烟用：按题型分层抽 N 题（每型 N/3），而非取前 N 条")
    ap.add_argument("--select", choices=["oracle", "filter"], default="oracle",
                    help="文档选择方式；filter 用规则解析公司/年份，非 oracle")
    ap.add_argument("--workers", type=int, default=4,
                    help="组内并发；每组的第一条仍单发以捂热缓存")
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("!! 未设置 DEEPSEEK_API_KEY")

    qas = load_jsonl(args.qa_file)
    if args.limit:
        # 分层抽样，不取前 N 条。起因（2026-08-01）：qa_seed 前 8 题恰好全是抽取题，
        # 冒烟因此绕过了同比/跨公司的后处理分支，bug 直到跑完全量（¥23）才暴露。
        # 冒烟的意义是覆盖代码分支，不是省时间。
        by = defaultdict(list)
        for q in qas:
            by[q["type"]].append(q)
        per = max(1, args.limit // len(by))
        picked = [q for t in sorted(by) for q in by[t][:per]]
        seen = {q["id"] for q in picked}
        for q in qas:                       # 不足 N 时按原序补足
            if len(picked) >= args.limit:
                break
            if q["id"] not in seen:
                picked.append(q)
        qas = picked
        print("冒烟分层抽样：" + "  ".join(
            f"{t}={sum(1 for q in qas if q['type'] == t)}" for t in sorted(by)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():                                # 断点续跑
        done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")}
        print(f"已有 {len(done)} 条，跳过")

    groups = defaultdict(list)
    for qa in qas:
        if qa["id"] not in done:
            groups[docs_for(qa, args.select)].append(qa)
    print(f"待跑 {sum(len(v) for v in groups.values())} 题，"
          f"分 {len(groups)} 个文档组")

    fh = out.open("a", encoding="utf-8")
    t0 = time.time()
    for gi, (stems, items) in enumerate(sorted(groups.items()), 1):
        parts = [render_report(s) for s in stems]
        if any(p is None for p in parts):
            print(f"  [跳过] 组 {stems} 缺解析文件")
            usage["failed"] += len(items)
            continue
        ctx = "\n\n".join(parts)

        def run_one(qa):
            pred, err = call(key, build_messages(qa, ctx))
            if pred is None:
                usage["failed"] += 1
                return None
            # 工具契约的后处理。prompt 里给了 CALC/CMP 指令让模型只输出操作数，
            # 就必须在这里把它算成最终答案——与 run_e2e 的 --calc 路径一致。
            # 漏掉这一步会让同比题与比较题全判错（2026-08-01 第一版正是如此，
            # 表面看是「长上下文直读在这两类上得 0 分」，实为评测侧的假象）。
            final = pred
            if qa["type"] == "yoy_compare":
                final = calc_postprocess(pred)
            elif qa["type"] == "cross_company":
                final = cmp_postprocess(pred, qa["question"],
                                        qa.get("meta", {}).get("companies", []))
            return {"id": qa["id"], "type": qa["type"],
                    "question": qa["question"], "gold": qa["answer"],
                    "prediction": final, "raw_prediction": pred,
                    "meta": qa.get("meta", {}),
                    # 只记文档标识，不写正文：整份年报约 7.5 万字符，逐条写会
                    # 让文件涨到几百 MB（超 GitHub 单文件上限）。评测时由
                    # eval_answers._render_parsed 从 data/parsed/ 确定性重建，
                    # 已验证重建结果与此处的 ctx 逐字一致。
                    # 记全部 stems——早先只记 stems[:1]，而正文跨多份，是 schema bug。
                    "retrieved": [{"uid": f"longctx_{s}", "source": s + ".pdf",
                                   "page": 0} for s in stems],
                    "gold_evidence": qa["evidence"]}

        # 第一条单发，把该组的前缀写进缓存；其余并发
        first = run_one(items[0])
        if first:
            fh.write(json.dumps(first, ensure_ascii=False) + "\n")
        if len(items) > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for rec in ex.map(run_one, items[1:]):
                    if rec:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        hit = usage["cached"] / max(usage["prompt"], 1)
        print(f"  [{gi}/{len(groups)}] {'+'.join(stems)[:46]} "
              f"{len(items)} 题 | 缓存命中 {hit:.0%} | "
              f"{time.time() - t0:.0f}s", flush=True)
    fh.close()

    miss = usage["prompt"] - usage["cached"]
    # 汇率 6.1 是按 2026-08-01 的实际账单反校准的：按 7.1 算得 ¥27.12，
    # 平台实扣 ¥23.29（DeepSeek 的人民币价并非美元价乘即期汇率）。
    cost = (miss * 0.435 + usage["cached"] * 0.003625
            + usage["completion"] * 0.87) / 1e6 * 6.1
    print(f"\n调用 {usage['calls']} 次，失败 {usage['failed']} 次")
    print(f"输入 {usage['prompt']:,}（缓存命中 {usage['cached']:,} = "
          f"{usage['cached'] / max(usage['prompt'], 1):.1%}）"
          f" 输出 {usage['completion']:,}（推理 {usage['reasoning']:,}）")
    print(f"约合 ¥{cost:.2f}（按 $0.435/$0.003625/$0.87 每 M，汇率 6.1，已按实账单校准）")
    print(f"答案写入 {out}")


if __name__ == "__main__":
    main()
