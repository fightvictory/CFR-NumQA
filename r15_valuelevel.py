#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R15（值级口径版）：用 Table 2 的精确覆盖判定重跑词表重叠分层。
覆盖 = 该题所有 gold 数值都出现在检索到的 top-5 上下文中（eval_context.py 同款）。
分层维度：问题 vs gold 表头(row_label+caption) 的 Jaccard 词重叠。
"""
import json, sys
from pathlib import Path
import jieba
from rank_bm25 import BM25Okapi
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, ".")
from attribute_errors import ctx_values, gold_values, in_ctx

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def tok(s): return [w for w in jieba.lcut(s) if w.strip() and len(w.strip())>1]

_dev = "mps" if torch.backends.mps.is_available() else "cpu"
_tok = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
_mdl = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5").to(_dev).eval()
def encode(texts, bs=256):
    single=isinstance(texts,str); texts=[texts] if single else texts
    out=[]
    for i in range(0,len(texts),bs):
        enc=_tok(texts[i:i+bs],padding=True,truncation=True,max_length=256,return_tensors="pt").to(_dev)
        with torch.no_grad(): v=_mdl(**enc).last_hidden_state[:,0]
        out.append(torch.nn.functional.normalize(v,dim=-1).cpu().numpy())
    a=np.vstack(out); return a[0] if single else a

units = load("data/corpus/structural.jsonl")
qa = load("data/qa_seed.jsonl")
bm25 = BM25Okapi([tok(u["text"]) for u in units])
emb = np.load("data/r15_emb.npy"); print(f"emb {emb.shape}", file=sys.stderr)

def rrf(rank_lists, k=60):
    sc={}
    for rl in rank_lists:
        for r,i in enumerate(rl): sc[i]=sc.get(i,0)+1.0/(k+r+1)
    return sorted(sc, key=lambda i:-sc[i])

def build_rec(q, idxs):
    return {"type":q["type"], "meta":q.get("meta",{}), "gold":q.get("answer",""),
            "retrieved":[{"text":units[i]["text"]} for i in idxs]}
def covered(q, idxs):
    rec=build_rec(q, idxs); g=gold_values(rec)
    return all(in_ctx(v, ctx_values(rec)) for v in g) if g else None

rows=[]
for q in qa:
    htext=" ".join((e.get("row_label","")+" "+e.get("caption","")) for e in q["evidence"])
    qw,hw=set(tok(q["question"])),set(tok(htext))
    jac=len(qw&hw)/len(qw|hw) if (qw|hw) else 0.0
    qv=encode(q["question"]); ds=emb@qv; bs=np.array(bm25.get_scores(tok(q["question"])))
    d5=list(np.argsort(-ds)[:5])
    h5=rrf([list(np.argsort(-ds)[:200]), list(np.argsort(-bs)[:200])])[:5]
    dc,hc=covered(q,d5),covered(q,h5)
    if dc is None: continue
    rows.append((q["type"],jac,dc,hc))

import collections
jacs=sorted(r[1] for r in rows); q33,q67=jacs[len(jacs)//3],jacs[2*len(jacs)//3]
strat=lambda j:"low" if j<=q33 else("mid" if j<=q67 else "high")
print(f"重叠分位 low<={q33:.3f} mid<={q67:.3f}", file=sys.stderr)
print(f"\n{'层':<6}{'n':>5}{'dense':>8}{'hybrid':>8}{'Δpp':>7}")
b=collections.defaultdict(list)
for t,j,d,h in rows: b[strat(j)].append((d,h))
for s in ["low","mid","high"]:
    v=b[s]; dc=sum(x[0] for x in v)/len(v); hc=sum(x[1] for x in v)/len(v)
    print(f"{s:<6}{len(v):>5}{dc:>7.1%}{hc:>7.1%}{(hc-dc)*100:>+6.1f}")
allv=[(d,h) for _,_,d,h in rows]; dc=sum(x[0] for x in allv)/len(allv); hc=sum(x[1] for x in allv)/len(allv)
print(f"{'ALL':<6}{len(allv):>5}{dc:>7.1%}{hc:>7.1%}{(hc-dc)*100:>+6.1f}  (Table2: 55.3/67.5 +12.2)")
byt=collections.defaultdict(lambda:[0,0,0])
for t,j,d,h in rows: byt[t][0]+=d;byt[t][1]+=h;byt[t][2]+=1
print()
for t,v in sorted(byt.items()): print(f"  {t:<14} dense {v[0]/v[2]:.1%} hybrid {v[1]/v[2]:.1%} Δ{(v[1]-v[0])/v[2]*100:+.1f}")
json.dump([{"type":t,"jac":j,"dense":d,"hyb":h} for t,j,d,h in rows], open("data/r15_valuelevel.json","w"))
