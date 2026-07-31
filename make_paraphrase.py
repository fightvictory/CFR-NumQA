#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把一部分问句改写成自然表述，用于检验成绩是否依赖模板匹配。

为什么需要
----------
数据集的 1,016 道问句由模板程序化生成，措辞与表头用词高度重合。外部审稿指出，
这可能让成绩虚高——系统学到的也许是模板与表头的字面对应，而非真正的问题理解。
附录的词汇重叠三分位分层只能部分回应：三个分位全是模板句，分层的是重合度而非
「模板 vs 自然语言」。

做法
----
抽 N 条现有条目，只改写问句表述，gold 答案与 evidence 一字不动，重跑后与同一批
条目的原始表述作对比。成绩若不掉，说明结果不是靠模板匹配撑起来的；若掉，掉多少
就是模板红利的大小——两种结果都值得报告。

改写者用商业模型而非本地生成器：主结果跑在本地 7B 上，用另一家的模型改写，避免
「自己出题自己答」。改写后逐条做程序化校验（公司名、年份必须保留，gold 数值不得
泄漏进问句），不合格的丢弃重来。

用法：
    export $(grep -v '^#' ~/.config/deepseek.env | xargs)
    python make_paraphrase.py -n 200
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import requests

# 改写者可切换。用哪家要记进数据并写进论文——「谁改写的」影响这个实验的可信度。
# 默认 glm-4-flash：它不是论文里被测的生成器（那是 GLM-5.2），且为免费档。
PROVIDERS = {
    # 默认 Kimi：Moonshot 不在论文评测的生成器名单里（那是 GLM-5.2 / GPT-5.5 /
    # DeepSeek / MiniMax），改写者与被测者完全无交集，循环性最小。
    # kimi-k3 是推理模型：推理 token 计入 max_tokens，给小了 content 会是空的
    # （回一个「OK」都要 131 个 completion token）。
    "kimi":     dict(api="https://api.moonshot.cn/v1/chat/completions",
                     model="kimi-k3", key_env="MOONSHOT_API_KEY", max_tokens=3072,
                     temperature=1),   # k3 只接受 temperature=1，给别的值直接 400
    "glm":      dict(api="https://open.bigmodel.cn/api/paas/v4/chat/completions",
                     model="glm-4-flash", key_env="GLM_API_KEY"),
    "glm-air":  dict(api="https://open.bigmodel.cn/api/paas/v4/chat/completions",
                     model="glm-4.5-air", key_env="GLM_API_KEY"),
    "deepseek": dict(api="https://api.deepseek.com/v1/chat/completions",
                     model="deepseek-v4-pro", key_env="DEEPSEEK_API_KEY"),
}
API = MODEL = None
MAX_TOK = 256
TEMP = 0.8

SYS = """你是中文财务分析师。把给定的财报问句改写成你自己会问的自然表述。

必须保留：公司名称、会计年度、所问的财务指标（用词不变）、问题类型（问某年数值／
问同比变动／问两家公司比较）。

要改到位——**换句式，不是删几个字**。目标是让句子看不出模板痕迹：
  原：平安银行2023年度的营业收入是多少？
  好：帮我查一下平安银行2023年报里的营业收入
  好：平安银行2023年营业收入这项，年报上写的是多少
  好：想确认平安银行在2023年实现了多少营业收入
  差：平安银行2023年的营业收入是多少（只删了「度」，等于没改）

要求：
- 只输出改写后的问句，不要解释，不要加引号
- 不要给出或暗示答案
- **指标名必须原样保留，一个字都不能改**：「归母净利润」不可写成「归属于母公司的净利润」，
  「营业收入」不可写成「营业总收入」。展开或替换指标名会改变它与报表行标签的字面距离，
  使这项检验失效。只改指标名以外的部分。
- **不要替换成有方向预设的说法**：原句问「变动了百分之多少」是中性的，不可改成
  「增长了多少」「上升了几个百分点」——指标可能是下跌的，这样问就带了误导。
  中性的说法有：变动／变化／增减／涨跌幅。
- 允许口语化、允许调整语序、允许换成祈使或陈述语气"""


def call(question, key, retries=3):
    for i in range(retries):
        try:
            r = requests.post(
                API, timeout=60,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={"model": MODEL, "temperature": TEMP, "max_tokens": MAX_TOK,
                      "messages": [{"role": "system", "content": SYS},
                                   {"role": "user", "content": question}]})
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip()
            # 模型有时一次给出多个候选（每行一条）。取第一行即可——整块文本会让
            # 后面的校验全部通过，然后把四句话当成一道题送进管线。
            txt = next((l for l in txt.splitlines() if l.strip()), "")
            return re.sub(r'^["「『]|["」』]$', "", txt).strip()
        except Exception as e:
            if i == retries - 1:
                print(f"    调用失败: {e}", file=sys.stderr)
                return None
            time.sleep(2 * (i + 1))
    return None


# 计算词面重合度时要剔除的语法虚词。它们在问句和报表行标签里都常见，却与检索
# 难易无关：实测 35 条被误杀的改写，全部只是因为把「公司2021年指标」写成
# 「公司的指标」而多命中一个「的」字，重合度就从 0.29 抬到 0.36。
STOP = set("的了着是在和与及之其为而或等有被把就都也很更最") | set(" \t\u3000")
# 空白也要排除：PDF 抽取出的行标签里常夹着空格（如「归属于上市公司股 东的净利润」），
# 改写句里的空格会与之「命中」，凭空抬高重合度——和「的」是同一类伪影。


def _content(s):
    return set(s) - STOP


def ok(qa, new):
    """程序化校验：实体必须保住，答案不得泄漏，且确实改写过。"""
    if not new or len(new) < 6:
        return False, "空或过短"
    if new == qa["question"]:
        return False, "与原句相同"
    # 字符级 Jaccard 太高说明只是删了几个字，起不到打破模板词面重合的作用
    a, b = set(new), set(qa["question"])
    if len(a & b) / len(a | b) > 0.85:
        return False, f"改动太小（Jaccard {len(a & b) / len(a | b):.2f}）"
    m = qa.get("meta", {})
    comp = m.get("company") or ""
    comps = m.get("companies") or ([comp] if comp else [])
    for c in comps:                                   # 公司名（去掉排版空格再比）
        if c and c.replace(" ", "") not in new.replace(" ", ""):
            return False, f"丢了公司名 {c}"
    years = m.get("years") or ([m["year"]] if m.get("year") else [])
    for y in years:
        if str(y) not in new:
            return False, f"丢了年份 {y}"
    ind = m.get("indicator") or ""
    # 指标名必须**原样**出现。放松成「首字符命中」会放过「归母净利润 -> 归属于母公司的
    # 净利润」这类展开，而展开恰好把问句拉近报表行标签，让检索变容易——实测 10 条里
    # 有 4 条因此把与行标签的重合度从 0.51 抬到 0.59，全部升高、无一降低。
    if ind and ind not in new:
        return False, f"指标名被改写（须原样保留 {ind}）"
    # 与 gold 行标签的重合度不得升高，否则是把题改简单了而非改自然
    rls = [e.get("row_label", "") for e in qa.get("evidence", []) if e.get("row_label")]
    if rls:
        def ov(q):
            return max((len(_content(q) & _content(r)) / max(1, len(_content(r))))
                       for r in rls if r)
        if ov(new) > ov(qa["question"]) + 0.02:
            return False, f"与行标签重合度升高 {ov(qa['question']):.2f}->{ov(new):.2f}"

    # 方向性预设：原句中性而改写预设了「涨」，在下跌的指标上就是误导。实测弱模型
    # 有 4/7 的同比题犯这个错，而其余校验全都放行——它不改实体、不改重合度。
    if qa.get("type") == "yoy_compare":
        up = re.search(r"增长|上升|增加|提高|增幅", new)
        balanced = re.search(r"下降|下跌|减少|降低|跌", new)   # 「增长或下降」是中性的
        neutral_src = not re.search(r"增长|上升|增加|提高|增幅", qa["question"])
        if up and not balanced and neutral_src:
            return False, "改写预设了上涨方向（原句中性）"

    gold_digits = re.sub(r"[^\d]", "", str(qa.get("answer", "")))
    if len(gold_digits) >= 5 and gold_digits[:5] in re.sub(r"[^\d]", "", new):
        return False, "问句里泄漏了答案"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--qa", default="data/qa_seed.jsonl")
    ap.add_argument("-o", "--out", default="data/qa_seed_para.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--provider", default="kimi", choices=sorted(PROVIDERS))
    ap.add_argument("--resume", action="store_true",
                    help="跳过输出文件里已有的条目，只补缺的")
    args = ap.parse_args()

    global API, MODEL, MAX_TOK, TEMP
    prov = PROVIDERS[args.provider]
    API, MODEL = prov["api"], prov["model"]
    MAX_TOK = prov.get("max_tokens", 256)
    TEMP = prov.get("temperature", 0.8)
    key = os.environ.get(prov["key_env"])
    if not key:
        sys.exit(f"请先 export {prov['key_env']}")
    print(f"改写者: {MODEL}")

    qas = [json.loads(l) for l in open(args.qa, encoding="utf-8")]
    by = defaultdict(list)
    for q in qas:
        by[q["type"]].append(q)
    rng = random.Random(args.seed)
    # 按总体题型比例分层，凑够 n
    share = {t: len(v) / len(qas) for t, v in by.items()}
    quota = {t: round(args.n * s) for t, s in share.items()}
    quota[max(quota, key=quota.get)] += args.n - sum(quota.values())
    sample = []
    for t, k in quota.items():
        sample += rng.sample(by[t], k)
    rng.shuffle(sample)
    print(f"抽样 {len(sample)} 条：" +
          "  ".join(f"{t} {k}" for t, k in sorted(quota.items())))

    # 断点续跑：已合格的不重复调用 API（改了校验后只需补被误杀的那些）
    out, bad = [], []
    done = {}
    if args.resume and os.path.exists(args.out):
        for l in open(args.out, encoding="utf-8"):
            r = json.loads(l)
            done[r["id"]] = r
        out = list(done.values())
        print(f"续跑：已有 {len(done)} 条，仅处理其余 {len(sample) - len(done)} 条")

    for i, qa in enumerate(sample, 1):
        if qa["id"] in done:
            continue
        new = call(qa["question"], key)
        good, why = ok(qa, new)
        if not good:                       # 换个温度重试一次
            new2 = call(qa["question"], key)
            good, why = ok(qa, new2)
            new = new2 if good else new
        if good:
            q2 = dict(qa)
            q2["question"] = new
            q2["meta"] = dict(qa["meta"], paraphrased=True,
                              paraphrased_by=MODEL,
                              original_question=qa["question"])
            out.append(q2)
        else:
            bad.append((qa["id"], why, new))
        if i % 25 == 0:
            print(f"  {i}/{len(sample)}  合格 {len(out)}  丢弃 {len(bad)}")

    with open(args.out, "w", encoding="utf-8") as fh:
        for q in out:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\n写出 {len(out)} 条 -> {args.out}")
    if bad:
        import collections as _c
        # 按原因分类：丢弃若集中在某一类题上，留下的样本就有选择偏差，
        # 「改写后成绩不掉」这个结论也就不可信了。
        tal = _c.Counter(re.sub(r"[\d.]+", "", w).strip() for _, w, _ in bad)
        print(f"丢弃 {len(bad)}/{len(sample)} 条（{len(bad)/len(sample):.0%}），原因分布：")
        for w, k in tal.most_common():
            print(f"    ×{k}  {w}")
        bt = _c.Counter(next((q["type"] for q in sample if q["id"] == i), "?")
                        for i, _, _ in bad)
        st = _c.Counter(q["type"] for q in sample)
        print("  丢弃的题型占比（对比抽样占比）：")
        for t in sorted(st):
            print(f"    {t:<14} 丢 {bt.get(t,0)}/{st[t]} = {bt.get(t,0)/st[t]:.0%}")
        for i, why, n in bad[:3]:
            print(f"    例 {i}  {why}  -> {str(n)[:52]}")
    print("\n改写样例：")
    for q in out[:3]:
        print(f"  原: {q['meta']['original_question']}")
        print(f"  改: {q['question']}\n")


if __name__ == "__main__":
    main()
