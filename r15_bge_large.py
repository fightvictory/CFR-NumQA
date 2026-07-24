#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03-R16: 用更强 dense 编码器 bge-large-zh 重跑检索消融（值级覆盖，Table 2 口径）。
证明混合检索增益是方法性质，不是 bge-small mini 编码器太弱的产物。训练机 GPU 跑。
"""
import json, sys, os, collections
import jieba
from rank_bm25 import BM25Okapi
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, ".")
from attribute_errors import ctx_values, gold_values, in_ctx

MODEL = "BAAI/bge-large-zh-v1.5"
CACHE = "data/r15_emb_large.npy"

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def tok(s): return [w for w in jieba.lcut(s) if w.strip() and len(w.strip())>1]

dev = "cuda" if torch.cuda.is_available() else "cpu"
_tk = AutoTokenizer.from_pretrained(MODEL)
_md = AutoModel.from_pretrained(MODEL).to(dev).eval()
def encode(texts, bs=128):
    single=isinstance(texts,str); texts=[texts] if single else texts
    out=[]
    for i in range(0,len(texts),bs):
        enc=_tk(texts[i:i+bs],padding=True,truncation=True,max_length=256,return_tensors="pt").to(dev)
        with torch.no_grad(): v=_md(**enc).last_hidden_state[:,0]
        out.append(torch.nn.functional.normalize(v,dim=-1).cpu().numpy())
        if i % (bs*40)==0: print(f"  enc {i}/{len(texts)}", file=sys.stderr)
    a=np.vstack(out); return a[0] if single else a

units = load("data/corpus/structural.jsonl")
qa = load("data/qa_seed.jsonl")
bm25 = BM25Okapi([tok(u["text"]) for u in units])
if os.path.exists(CACHE):
    emb = np.load(CACHE); print("loaded cached emb", file=sys.stderr)
else:
    print("encoding corpus with bge-large...", file=sys.stderr)
    emb = encode([u["text"] for u in units]); np.save(CACHE, emb)

def rrf(rls,k=60):
    sc={}
    for rl in rls:
        for r,i in enumerate(rl): sc[i]=sc.get(i,0)+1.0/(k+r+1)
    return sorted(sc,key=lambda i:-sc[i])
def rec_of(q,idxs): return {"type":q["type"],"meta":q.get("meta",{}),"gold":q.get("answer",""),
                            "retrieved":[{"text":units[i]["text"]} for i in idxs]}
def cov(q,idxs):
    r=rec_of(q,idxs); g=gold_values(r)
    return all(in_ctx(v,ctx_values(r)) for v in g) if g else None

byt=collections.defaultdict(lambda:[0,0,0]); tot=[0,0,0]
for q in qa:
    qv=encode(q["question"]); ds=emb@qv; bs=np.array(bm25.get_scores(tok(q["question"])))
    d5=list(np.argsort(-ds)[:5]); h5=rrf([list(np.argsort(-ds)[:200]),list(np.argsort(-bs)[:200])])[:5]
    dc,hc=cov(q,d5),cov(q,h5)
    if dc is None: continue
    byt[q["type"]][0]+=dc; byt[q["type"]][1]+=hc; byt[q["type"]][2]+=1
    tot[0]+=dc; tot[1]+=hc; tot[2]+=1
print(f"\n{'类型':<14}{'dense':>8}{'hybrid':>8}{'delta':>7}")
for t,v in sorted(byt.items()):
    print(f"{t:<14}{v[0]/v[2]:>7.1%}{v[1]/v[2]:>7.1%}{(v[1]-v[0])/v[2]*100:>+6.1f}")
print(f"{'ALL':<14}{tot[0]/tot[2]:>7.1%}{tot[1]/tot[2]:>7.1%}{(tot[1]-tot[0])/tot[2]*100:>+6.1f}  (bge-small local: 58.6/66.8 +8.2)")
