#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04-d: 对商业模型的正确答案施加8模式扰动，生成合成负样本，
供验证器门控，把跨厂商迁移证据从个位数真实错误扩到 n>=50。
输出记录保持原 retrieved 上下文，prediction 换成扰动后的错误答案。
"""
import json, sys, random
sys.path.insert(0, ".")
random.seed(7)
from eval_answers import is_correct
from attribute_errors import ctx_values
from build_verifier_data import make_negatives, rec_company, TEST_COMPANIES

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]

for tag, fn in [("glm","data/answers_glm_v3ctx.jsonl"),("gpt","data/answers_gpt_v3ctx.jsonl")]:
    recs = load(fn)
    out = []
    for r in recs:
        if not is_correct(r):        # 只扰动模型本来答对的题
            continue
        if not (set(rec_company(r).split("+")) & TEST_COMPANIES):  # 仅测试公司
            continue
        cvals = ctx_values(r)
        negs = make_negatives(r, cvals)
        for ans, kind in negs:
            nr = dict(r); nr["prediction"] = ans; nr["_synth_kind"] = kind
            nr["raw_prediction"] = ans
            out.append(nr)
    outp = f"data/synthneg_{tag}.jsonl"
    with open(outp,"w",encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    import collections
    kc = collections.Counter(r["_synth_kind"] for r in out)
    print(f"{tag}: {len(out)} 合成负样本 -> {outp}  分模式 {dict(kc)}")
