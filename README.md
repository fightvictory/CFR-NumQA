# CFR-NumQA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21436517.svg)](https://doi.org/10.5281/zenodo.21436517)

**Chinese Financial Report Numerical QA** — a dataset and reproducible pipeline for
studying and suppressing *numerical hallucination* in retrieval-augmented question
answering over Chinese A-share annual reports.

Companion code and data for the paper *"Mitigating Numerical Hallucination in
Retrieval-Augmented Question Answering over Chinese Financial Reports via
Structure-Aware Retrieval and Lightweight Verification."*

Everything here runs end-to-end on a **single consumer GPU** (developed on an
RTX 4060 Ti 16 GB); only the 32B-generator experiment used one rented RTX 4090.

---

## What is in this repository

| Path | Contents |
|---|---|
| `data/qa_seed.jsonl` | **The dataset**: 1,016 QA pairs (540 extraction / 386 year-over-year / 90 cross-company), each with evidence provenance and re-verifiable gold answers |
| `data/parsed/` | Structure-aware parses of 103 annual reports (text blocks + tables + linearized triples). Lets you rebuild the retrieval corpora without re-downloading any PDF |
| `results/` | Raw model outputs for every experiment in the paper, so all reported numbers can be re-verified **without a GPU** |
| `*.py` | The full pipeline: download → parse → dataset construction → retrieval → generation → verification → evaluation |

**Not included:** the 103 source PDFs (≈860 MB). They are public disclosures on
[cninfo.com.cn](http://www.cninfo.com.cn) and can be re-downloaded with
`cninfo_downloader.py` (see below). The trained verifier LoRA adapter (165 MB) is
attached to the GitHub Release rather than tracked in git.

## Dataset at a glance

Each QA pair carries the evidence needed to audit it:

```json
{
  "id": "seed_0001",
  "type": "extraction",
  "question": "平安银行2023年度的营业收入是多少？",
  "answer": "164,699百万元",
  "evidence": [{"source": "000001_平安银行_2023年年度报告.pdf",
                "page": 15, "table_id": "p15_t0",
                "caption": "2.1 关键指标（货币单位：人民币百万元）",
                "row_label": "营业收入"}],
  "meta": {"indicator": "营业收入", "year": "2023",
           "company": "平安银行", "value": 164699.0, "unit": "百万元"}
}
```

Year-over-year growth rates are computed programmatically from their two operands,
so every gold answer is machine-re-verifiable. A stratified sample of 100 pairs was
audited against the source PDFs with an independent extraction engine and then
reviewed by the author; all 100 passed value-level verification.

## Quick start

```bash
pip install -r requirements.txt

# 1. Rebuild the retrieval corpora from the parsed reports (no PDFs needed)
python build_corpus.py data/parsed/ -o data/corpus/

# 2. Verify any paper number without a GPU, e.g. the full pipeline
python eval_answers.py results/answers_v3_full.jsonl
python attribute_errors.py results/answers_v3_full.jsonl
python eval_bootstrap.py results/answers_v3_full.jsonl        # 95% bootstrap CIs
python eval_bootstrap.py --diff results/answers_v3_full.jsonl \
                                results/answers_bl_crag.jsonl # paired significance

# 3. Re-check any of the three component ablations (all arms ship, no GPU needed)
python eval_bootstrap.py --diff results/answers_abl_with.jsonl \
                                results/answers_abl_nounit.jsonl   # caption unit
python eval_bootstrap.py --diff results/answers_abl_with.jsonl \
                                results/answers_abl_nodoc.jsonl    # document context
python eval_bootstrap.py --diff results/answers_abl_with.jsonl \
                                results/answers_abl_noquota.jsonl  # per-subquery quota
```

To regenerate the ablation arms rather than re-score them, build the control
corpus with `--no-unit-attach` and run the pipeline over each corpus in turn
(this part needs a GPU):

```bash
# caption-unit re-attachment
python build_corpus.py data/parsed/ -o data/corpus_nounit/ --no-unit-attach
python run_e2e.py data/corpus_nounit/structural.jsonl data/qa_seed.jsonl \
    --calc --sub-quota --hybrid --filter-meta -o results/answers_abl_nounit.jsonl

# document-context prefix
python build_corpus.py data/parsed/ -o data/corpus_nodoc/ --no-doc-context
python run_e2e.py data/corpus_nodoc/structural.jsonl data/qa_seed.jsonl \
    --calc --sub-quota --hybrid --filter-meta -o results/answers_abl_nodoc.jsonl

# per-subquery quota needs no rebuild -- just drop the flag
python run_e2e.py data/corpus/structural.jsonl data/qa_seed.jsonl \
    --calc --hybrid --filter-meta -o results/answers_abl_noquota.jsonl
```

All four arms in `results/` were produced in one environment, and
`answers_abl_with.jsonl` — the shared baseline — reproduces
`answers_v3_full.jsonl` accuracy exactly, so every arm is directly
comparable. Add `--limit N` to `run_e2e.py` to smoke-test the path on the
first `N` questions before committing to a full run.

To reproduce generation you need a GPU:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
# full pipeline: unit-aware index + hybrid retrieval + metadata filter + tool contract
python run_e2e.py data/corpus/structural.jsonl data/qa_seed.jsonl \
  --calc --sub-quota --hybrid --filter-meta -o answers.jsonl
python eval_answers.py answers.jsonl
```

To rebuild the dataset from scratch (requires network access to cninfo):

```bash
python cninfo_downloader.py --stocks 000001,600519,300750 --years 2022,2023,2024,2025
python report_parser.py data/raw_pdfs/ -o data/parsed/ --max-pages 60
python build_qa_seed.py data/parsed/ -o data/qa_seed.jsonl
```

## Pipeline components

| Script | Role |
|---|---|
| `cninfo_downloader.py` | Fetch annual-report PDFs from cninfo (with retry/resume) |
| `report_parser.py` | Structure-aware parsing; emits text blocks, tables, and linearized triples |
| `build_corpus.py` | Builds the structural corpus (with **caption-unit re-attachment**) and the naive-chunk control corpus. `--no-unit-attach` and `--no-doc-context` build the ablation controls that drop caption-unit re-attachment or the document-context prefix, one at a time |
| `build_qa_seed.py` | Harvests facts and generates the three question types with provenance |
| `run_e2e.py` | Retrieval + generation. Flags: `--hybrid` (BM25⊕dense RRF), `--filter-meta` (rule-parsed company/year filter), `--sub-quota` (per-entity evidence quota), `--calc` (tool contract for arithmetic and cross-unit comparison), `--auto-decompose` (non-oracle query decomposition), `--limit N` (first `N` questions only, for smoke tests) |
| `run_baselines.py` | Closed-book, Self-RAG-lite, CRAG-lite baselines |
| `run_api_baseline.py` | Frontier commercial model baselines via API (`--provider glm\|openai\|minimax\|deepseek`; `--stratify` for typed pilots) |
| `build_verifier_data.py` | Constructs verifier supervision programmatically from observed error modes; splits **by company** |
| `train_verifier.py` | QLoRA fine-tuning of the verifier (4-bit NF4 + LoRA r16, <2 h on one consumer GPU) |
| `eval_verifier.py` | Verifier judgment metrics + end-to-end gating simulation |
| `run_gates_v2.sh`, `run_gates_glm.sh`, `run_gates_gpt.sh`, `run_gates_ds.sh` | Reproduce all gating runs for the V2 verifier and the commercial generators |
| `make_synth_negs.py` | Perturb commercial models' correct answers with the eight error modes (transfer test at scale) |
| `r15_valuelevel.py`, `r15_bge_large.py` | Vocabulary-overlap stratification of the retrieval gain and the bge-large robustness check |
| `eval_answers.py` | Accuracy / abstention / hallucination / grounded-error metrics |
| `eval_context.py` | Evidence coverage (primary retrieval metric) |
| `attribute_errors.py` | 14-way error attribution taxonomy |
| `eval_bootstrap.py` | 95% bootstrap CIs and paired significance tests |
| `analyze_cmp.py` | Cross-company badcase analysis |
| `make_audit.py`, `preaudit.py` | Human-audit sampling sheet and independent-engine pre-verification |

## Headline results

Accuracy on 1,016 questions with a frozen Qwen2.5-7B-Instruct generator:

| System | Acc. | Abstain | Hallucination |
|---|---|---|---|
| Naive-chunk RAG + tool | 30.3% | 30.1% | 3.6% |
| Self-RAG-lite | 50.5% | 27.4% | 6.2% |
| CRAG-lite | 54.8% | 18.0% | 9.2% |
| GLM-5.2, same naive pipeline | 43.7% | 46.9% | 0.6% |
| GPT-5.5, same naive pipeline | 49.5% | 45.0% | 1.2% |
| **This work (full pipeline)** | **64.5%** | 15.1% | **3.6%** |
| GLM-5.2 on our contexts | 66.2% | 28.8% | 0.5% |
| GPT-5.5 on our contexts | 68.4% | 26.3% | 0.6% |
| **+ verifier gate** | **97.5%** answered accuracy | — | **0** |

**The pipeline matters several-fold more than the generator.** Two frontier
commercial reasoning models from different vendors gain +22.5 and +18.9 points
from swapping the retrieval pipeline, but only +1.8 and +3.9 points from swapping
our 7B open-weight generator for either of them at a fixed pipeline. (We do not
quote a precise ratio: the generator delta's confidence interval approaches zero,
so any quotient is unstable.) A per-type reading matters here — a frontier model's
flat closed-book-vs-naive accuracy hides a significant +7.6–13.1 pp retrieval gain
on extraction and year-over-year questions, cancelled in aggregate by
world-knowledge leakage on comparison questions.

Gated answered accuracy stays at 97.5–98.4% across 7B/14B/32B generators and on
both commercial generators (`results/gate_glm_v1.jsonl`, `results/gate_gpt_v1.jsonl`:
every passed answer correct and fully evidence-backed, zero hallucinations). To
test error detection at scale rather than on the handful of natural errors these
high-abstention models make, we perturbed their correct answers with the eight
error modes and gated the results: the verifier rejects 96% of synthetic errors on
both (`results/synthgate_*.jsonl`).

The M3 and DeepSeek-V4-Pro runs from an earlier draft are retained under
`results/answers_{m3,ds}_*.jsonl` and gave consistent conclusions.

### Verifier operating points

The gate buys precision with coverage. `build_verifier_data.py` originally decided
groundedness by checking that every gold *value* appears in the context — but a
comparison question's gold answer is a company name, not a number, so the check
returned an empty list and **every comparison question's correct answer entered
training as a negative example** (89 negatives, 0 positives). The verifier learned
to reject that question type wholesale. Deciding groundedness by evidence
*provenance* instead repairs it:

| | V1 (original labels) | V2 (repaired labels) |
|---|---|---|
| Coverage (7B / 14B / 32B) | 53.6% / 55.4% / 55.8% | 62.5% / 63.4% / 63.4% |
| Precision | 97.5% / 98.4% / 98.4% | 95.0% / 97.2% / 96.5% |
| Comparison questions passed | 0% | 77–82% |
| True false-block rate | 20.4–25.8% | 12.4–16.4% |

Both hold zero hallucinations and 100% evidence coverage among passed answers.
Both adapters are attached to the GitHub Release. Unzip either into `models/`
to get the layout the scripts expect:

```bash
unzip verifier_lora_v1.0.0.zip -d models/   # -> models/verifier_lora     (V1)
unzip verifier_lora_v2.0.0.zip -d models/   # -> models/verifier_lora_v2  (V2)
```

V1 is the default path in `eval_verifier.py` and `run_gates_v2.sh` uses V2, but
`run_gates_{glm,gpt,ds}.sh` gate with both versions in turn — keep both
unzipped to reproduce the V1/V2 comparison above. Rebuild either with
`build_verifier_data.py` + `train_verifier.py`. See the paper's Section 5.6.

The defect is invisible in aggregate judgment accuracy (V1 scores 97.3% while
refusing an entire question type) and surfaces only in a per-question-type
breakdown. Any groundedness gate built from programmatic supervision should be
reported with such a breakdown.

> **Note (2026-07-22).** `eval_answers.py::extract_number` previously took the
> first number in a prediction, which mis-read the leading year in narrative
> answers such as "…公司2023年度净利润为11,582,226,085.00元". Short-form outputs
> were unaffected, but verbose ones (notably the commercial API baseline) were
> systematically under-scored. The table above reflects the corrected scorer;
> raw model outputs in `results/` are unchanged and can be re-scored.

## Data provenance and use

The source documents are annual reports publicly disclosed on cninfo.com.cn, the
platform designated by the China Securities Regulatory Commission. This repository
redistributes **derived research artifacts** (structured parses, QA annotations,
model outputs), not the original PDF documents; use `cninfo_downloader.py` to
obtain those from the original source. Please respect the terms of the source
platform.

- **Code**: MIT License (see `LICENSE`)
- **Derived data and annotations** (`data/qa_seed.jsonl`, `data/parsed/`, `results/`):
  CC BY 4.0 (see `LICENSE-DATA`)

## Citation

If you use the dataset or code, please cite both the paper and the archived dataset:

```bibtex
@dataset{cfr-numqa-data,
  author    = {Wang, Jikui and Zhang, Guijie and Lu, Yang},
  title     = {{CFR-NumQA}: Chinese Financial Report Numerical QA},
  publisher = {Zenodo},
  version   = {v1.0.0},
  year      = {2026},
  doi       = {10.5281/zenodo.21436517}
}

@article{cfr-numqa,
  author  = {Wang, Jikui and Zhang, Guijie and Lu, Yang},
  title   = {Mitigating Numerical Hallucination in Retrieval-Augmented Question
             Answering over Chinese Financial Reports via Structure-Aware
             Retrieval and Lightweight Verification},
  year    = {2026},
  note    = {Under review}
}
```
