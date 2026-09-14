# Hiver SDE Intern Take-Home Assignment

**AI support agent for a brand**, built on the *Customer Support on Twitter* dataset
(Kaggle [`thoughtvector/customer-support-on-twitter`](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)).
The agent classifies a customer message into an intent, drafts a grounded reply, and
escalates to a human agent with a stated reason when needed.

- **Full report (max 6 pages):** [`REPORT.md`](REPORT.md)
- **Decision log (10–15 decisions):** [`DECISION_LOG.md`](DECISION_LOG.md)
- **Golden set (160 hand-labeled examples):** [`golden_set/uber_golden.json`](golden_set/uber_golden.json)

## Brand choice

`Uber_Support` — the 3rd largest brand in the dataset (56,270 agent tweets) with the
highest share of **directed conversation pairs** (55,182 clean customer→agent pairs),
which makes it ideal for grounded reply generation and evaluation.

## Reproduce the pipeline (< 15 min)

Everything runs on a local Ollama server (`http://localhost:11434`) with two models:
`mxbai-embed-large` (embeddings) and `llama3.2:1b` (optional LLM judge). No cloud API needed.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Point build_pairs.py at your Kaggle CSV (default D:\dataset\twcs\twcs.csv, or use TWCS_CSV env var):
#    set TWCS_CSV=C:\path\to\twcs.csv   (from Kaggle: thoughtvector/customer-support-on-twitter)

# 3. Run everything from the CLI (extract -> RAG index -> evaluate -> analyze)
python run.py --all

# ...or step by step:
python build_pairs.py                # extract Uber customer->agent pairs
python build_rag_index.py 2000       # build the RAG embedding index
python run_eval.py                   # full evaluation vs baselines
python analyze_results.py            # reply grounding + LLM-judge on 30 samples
python failure_analysis.py           # contrastive judge, escalation audit, confusions

# 4. (Golden set is committed) verify it loads
python -c "import json; g=json.load(open('golden_set/uber_golden.json',encoding='utf-8')); print(len(g),'golden examples')"
```

Manual scripts (the golden set at `golden_set/` is committed; `data/` and `results/` are
regenerated locally and gitignored): `build_pairs.py`, `build_rag_index.py`,
`label_golden_hand.py` (builds the golden set), `extract_golden_draft.py`.

## Results (golden set, 160 hand-labeled examples)

| System        | Intent acc | Escalation P/R/F1 | Reply grounding (emb sim vs historical) |
|---------------|-----------:|------------------:|----------------------------------------:|
| **Our agent** | **47.5%**  | **0.86 / 1.00 / 0.92** | **0.814** (94% ≥ 0.7) |
| Simple (keyword) | 48.1%   | 0.00 / 0.00 / 0.00 | 0.814 (94% ≥ 0.7) |
| Trivial (majority) | 21.2%  | 0.00 / 0.00 / 0.00 | 0.686 (45% ≥ 0.7) |

- **Intent:** the agent (centroid classifier + keyword fallback) is **2.2×** the trivial
  majority baseline and matches the pure-keyword baseline, while handling far more surface
  variation via embeddings.
- **Escalation:** 0 false negatives; the 2 false positives are defensible (a "hacked"
  complaint and a months-long account lockout).
- **Reply quality:** measured *objectively* by embedding similarity to the real Uber reply
  for the same-style issue (no LLM subjectivity). 94% of replies are within 0.7 cosine
  distance of an actual historical Uber reply; mean 0.814.
- **LLM-as-judge:** shipped in the harness (`run_eval.py` → contrastive judge) and explicitly
  calibrated against a human reviewer (14 samples). The tiny local 1B judge only agrees with a
  human ~50% on strict A/B — at chance — so it is reported transparently, not used as the gate.
  With a stronger API model the harness supports gating on the judge directly.

## Repository layout

```
build_pairs.py          Extract Uber customer->agent pairs from the raw CSV
build_rag_index.py      Build resumable RAG embedding index (mxbai-embed-large)
label_golden_hand.py    Build the 160-example hand-labeled golden set
run_eval.py             Full evaluation: our agent, simple, trivial baselines
analyze_results.py      Objective reply grounding + LLM-judge scoring on 30 samples
failure_analysis.py     Contrastive LLM judge, escalation error audit, intent confusions
src/                    Extracted modules (agent pipeline, intent, reply, escalation)
golden_set/             uber_golden.json (160 examples)
data/                   LARGE inputs - gitignored (generate locally)
results/                Evaluation outputs - gitignored (generate locally)
```