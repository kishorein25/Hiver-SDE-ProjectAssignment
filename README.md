# Hiver SDE Intern Assignment — AI Support Agent for @Uber_Support

> End-to-end AI support agent built from **55,182 real Uber customer-support tweets**:
> classifies intent, drafts replies grounded in how Uber actually resolves issues, and
> decides auto-handle vs. escalate-to-human. **100% local and offline, no API keys,
> fully self-contained and reproducible.**

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=1369423046)

---

## What it does

For any inbound customer message, the agent makes three decisions in one pass:

1. **Intent** — what the complaint is about (routing queue),
2. **Reply** — a grounded draft using Uber's *actual* historical resolution,
3. **Escalation** — auto-handle, or hand to a human agent **with a stated reason**.

Data: Kaggle [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
(~2.8M tweets) → extracted 55,182 `Uber_Support` customer→agent pairs → 160 hand-labeled
golden-set examples for evaluation.

---

## The full pipeline (how it works)

```
                     ┌──────────────────────────────────────┐
                     │   INBOUND CUSTOMER MESSAGE            │
                     │   "I was charged twice for the same   │
                     │    trip"                              │
                     └──────────────────┬───────────────────┘
                                        │
                     ┌──────────────────▼───────────────────┐
                     │  1 · UNDERSTAND                       │
                     │     embed text → 1024-dim vector       │
                     │     mxbai-embed-large  (local)        │
                     └──────────────────┬───────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
    ┌─────────▼──────────┐   ┌──────────▼──────────┐   ┌──────────▼───────────┐
    │ 2 · CLASSIFY INTENT│   │ 3 · DRAFT REPLY     │   │ 4 · DECIDE ESCALATION│
    │                    │   │                     │   │                      │
    │ nearest of 10      │   │ nearest of 2,000    │   │ 12 word-boundary     │
    │ intent centroids   │   │ past customer msgs  │   │ safety/fraud/legal   │
    │ (25 real anchors)  │   │ → return the real   │   │ rule patterns        │
    │ + keyword fallback │   │   Uber reply        │   │                      │
    └─────────┬──────────┘   └──────────┬──────────┘   └──────────┬───────────┘
              │                         │                         │
              ▼                         ▼                         ▼
      ┌─────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
      │ trip_fare_billing│  │ "We're sorry to hear│   │ (true) Possible fraud │
      │                  │  │  this… contact us   │   │ / unauthorized charges│
      │                  │  │  via…"              │   │                      │
      └─────────────────┘   └─────────────────────┘   └──────────────────────┘
```

### Stage-by-stage

| # | Stage | How it works | Why it's designed this way |
|---|-------|--------------|----------------------------|
| 1 | **Understand** | A sentence-embedding model converts the message into a 1024-dim vector | Text → numbers so similar meanings become similar vectors |
| 2 | **Classify intent** | Nearest of 10 intent centroids (each built from 25 real examples) + keyword fallback | Few-shot, trained outside the golden set, matches its 55k-pair origin |
| 3 | **Draft reply** | RAG lookup — most-similar historical customer message → verbatim real Uber reply | **Zero hallucination**; replies read like Uber because they *are* Uber |
| 4 | **Decide escalation** | 12 word-boundary pattern rules → auto-handle or hand to human with a reason | Predictable, testable, recall-first (never miss a safety/fraud/legal case) |

The **10 intents** (derived from the data): `trip_fare_billing`, `driver_behavior`,
`trip_cancellation`, `app_technical`, `account_access`, `uber_eats_order`, `lost_item`,
`safety_concern`, `general_inquiry`, `service_feedback`.

> **Design principle:** no LLM runs in the live path. Embeddings do the understanding,
> retrieval does the answering, and rules do the escalation — fast, free, offline, explainable.

---

## Results (`python run_eval.py` · 160-example golden set)

| System | Intent acc. | Escalation P/R/F1 | Reply grounding (≥0.7) |
|--------|------------:|------------------:|----------------------:|
| **Our agent** | **47.5%** | **0.86 / 1.00 / 0.92** | **0.814 (94%)** |
| Simple baseline (keywords) | 48.1% | 0.00 / 0.00 / 0.00 | 0.814 (94%) |
| Trivial baseline (majority) | 21.2% | 0.00 / 0.00 / 0.00 | 0.686 (45%) |

- **Intent:** ~2.2× the trivial baseline; matches keyword accuracy while handling far more real-world phrasing.
- **Escalation:** 0 missed escalations (recall 1.00) on all 12 true escalation cases.
- **Replies:** 94% within 0.7 cosine distance of a *real* Uber reply for the same issue type.

**Read this honestly:** these numbers are strong against the baselines but modest in absolute
terms, and each metric has caveats. Every limit is laid out — see
[REPORT.md → "What is misleading about my headline number?"](REPORT.md).

---

## Quick start

> **Browser (zero setup):** click the Codespaces badge above — Python, Ollama, and both models
> are provisioned automatically. Then run `python interact.py` immediately.

**Locally** (Python 3.10 + [Ollama](https://ollama.com)):

```bash
pip install -r requirements.txt
ollama serve                     # keep this running in another terminal
ollama pull mxbai-embed-large    # embeddings
ollama pull llama3.2:1b          # LLM-as-judge

# ONE command runs the whole pipeline → reproduces the results above:
python run.py --all

# or the parts individually:
python run_eval.py               # headline metrics → results/evaluation_summary.json
python interact.py               # chat with the agent using your own messages
```

All generated data (`data/`, `results/`) is **already committed**, so a fresh clone reproduces
the headline results immediately — no Kaggle download or rebuild needed. Regenerating from
source is optional and documented in [WALKTHROUGH.md](WALKTHROUGH.md).

---

## Repository layout

```
.
├── run.py                  one entry point:  python run.py --all
├── interact.py             live CLI demo:    python interact.py
├── run_eval.py             eval vs 2 baselines → results/
├── build_pairs.py          Kaggle CSV → 55,182 Uber pairs
├── build_rag_index.py      pairs → RAG embedding index
├── label_golden_hand.py    160 hand-labeled golden set
├── analyze_results.py      reply grounding + LLM-as-judge
├── failure_analysis.py     escalation audit + intent confusions
├── calibration.py          judge↔human agreement on 14 samples
├── src/                    agent modules (intent · reply · escalation · evaluation)
├── golden_set/             hand-labeled answer key (committed)
├── data/                   pairs + RAG index + CSVs (committed)
├── results/                evaluation outputs (committed)
├── .devcontainer/          one-click Codespaces environment
├── REPORT.md               full report — framing · baselines · failures · next steps
├── DECISION_LOG.md         15 design decisions with reasoning
└── WALKTHROUGH.md          step-by-step walkthrough of the whole pipeline
```

---

## Deliverables checklist

| Assignment deliverable | Where |
|------------------------|-------|
| Runnable repo; README reproduces results < 15 min | Quick start above |
| Golden eval set (150–250) + sampling note | `golden_set/uber_golden.json` (160) — see `label_golden_hand.py` |
| Eval harness: metrics + LLM-as-judge + human agreement | `run_eval.py`, `analyze_results.py`, `failure_analysis.py`, `calibration.py` |
| Report: framing · 2 baselines · top-5 failures · misleading number · next week | [`REPORT.md`](REPORT.md) |
| Decision log (10–15 decisions) | [`DECISION_LOG.md`](DECISION_LOG.md) |

---

## Attributions

- **Dataset:** Kaggle — *Customer Support on Twitter* (`thoughtvector/customer-support-on-twitter`); operated on a documented subsample (Uber_Support).
- **Models (local, via Ollama):** `mxbai-embed-large` (Apache-2.0) for all embeddings; `llama3.2:1b` (Meta) as the LLM-as-judge only.
- **Libraries:** numpy · scikit-learn · pandas · requests. Standard ML patterns (centroid few-shot classification, RAG retrieval); no external code base was copied.