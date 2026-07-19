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
```

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
| `build_corpus.py` | Builds the structural corpus (with **caption-unit re-attachment**) and the naive-chunk control corpus |
| `build_qa_seed.py` | Harvests facts and generates the three question types with provenance |
| `run_e2e.py` | Retrieval + generation. Flags: `--hybrid` (BM25⊕dense RRF), `--filter-meta` (rule-parsed company/year filter), `--sub-quota` (per-entity evidence quota), `--calc` (tool contract for arithmetic and cross-unit comparison), `--auto-decompose` (non-oracle query decomposition) |
| `run_baselines.py` | Closed-book, Self-RAG-lite, CRAG-lite baselines |
| `run_api_baseline.py` | Frontier commercial model comparison via API |
| `build_verifier_data.py` | Constructs verifier supervision programmatically from observed error modes; splits **by company** |
| `train_verifier.py` | QLoRA fine-tuning of the verifier (4-bit NF4 + LoRA r16, <2 h on one consumer GPU) |
| `eval_verifier.py` | Verifier judgment metrics + end-to-end gating simulation |
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
| Naive-chunk RAG + tool | 28.1% | 30.1% | 3.6% |
| Self-RAG-lite | 49.1% | 27.4% | 6.2% |
| CRAG-lite | 53.3% | 18.0% | 9.2% |
| Frontier commercial model, same naive pipeline | 37.0% | 40.1% | 2.3% |
| **This work (full pipeline)** | **63.3%** | 15.1% | **3.7%** |
| **+ verifier gate** | **97.5%** answered accuracy | — | **0** |

Gated answered accuracy stays at ≈98.4% for 7B, 14B and 32B generators: the
verifier, not generator scale, sets the ceiling of system reliability.

## Data provenance and use

The source documents are annual reports publicly disclosed on cninfo.com.cn, the
platform designated by the China Securities Regulatory Commission. This repository
redistributes **derived research artifacts** (structured parses, QA annotations,
model outputs), not the original PDF documents; use `cninfo_downloader.py` to
obtain those from the original source. Please respect the terms of the source
platform.

- **Code**: MIT License (see `LICENSE`)
- **Derived data and annotations** (`data/qa_seed.jsonl`, `data/parsed/`, `results/`):
  CC BY 4.0, for research use

## Citation

If you use the dataset or code, please cite both the paper and the archived dataset:

```bibtex
@dataset{cfr-numqa-data,
  author    = {Wang, Jikui},
  title     = {{CFR-NumQA}: Chinese Financial Report Numerical QA},
  publisher = {Zenodo},
  version   = {v1.0.0},
  year      = {2026},
  doi       = {10.5281/zenodo.21436517}
}

@article{cfr-numqa,
  author  = {Wang, Jikui},
  title   = {Mitigating Numerical Hallucination in Retrieval-Augmented Question
             Answering over Chinese Financial Reports via Structure-Aware
             Retrieval and Lightweight Verification},
  year    = {2026},
  note    = {Under review}
}
```
