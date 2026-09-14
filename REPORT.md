# Report: AI Support Agent for Uber_Support

**Deliverable for the Hiver SDE Intern take-home assignment.**
A support agent that classifies intents from customer tweets, drafts grounded replies, and
escalates hard cases to a human — trained and evaluated on the *Customer Support on Twitter*
dataset (Kaggle `thoughtvector/customer-support-on-twitter`).

---

## 1. Task, objective, and problem framing

Given ~2.8M real customer-support conversations between brands and customers on Twitter, build
an agent that, for a single brand:

1. **Classifies intent** of an inbound customer message;
2. **Drafts a reply** grounded in how the brand actually responds to that issue;
3. **Escalates** to a human agent with a **stated reason** when the case needs human judgment
   (safety, legal, fraud, complexity), instead of auto-replying.

Success criteria are measured against a **hand-labeled golden set** (150–250 examples) with an
evaluation harness that includes objective metrics, baselines, and an LLM-as-judge provision.

**What "good" means for Uber_Support.** For this brand, the marginal tweet is short, emotional,
and about one of a dozen recurring complaints (fares, cancellation fees, lost items, driver
behavior). A good agent (a) routes the tweet to the *right internal queue* (intent), (b) answers
in Uber's own voice — the same tone and links the brand already uses — with zero hallucination,
and (c) never lets a safety/fraud/legal case be auto-answered. "Good" is therefore: intent
accuracy well above the majority class, grounded (retrieved, not generated) replies, and
escalation tuned to **recall-first** (never miss a hard case) while keeping false alarms low.

**What I chose not to build.** Explicitly out of scope: (1) generative LLM reply-drafting — a
support bot must not improvise fixes, so replies are *retrieval-only*; (2) multi-brand support —
one brand, deep, per the assignment; (3) thread-level conversation state / chained multi-turn
context — each inbound tweet is treated as the unit to classify, which mirrors how hard the real
problem is; (4) sentiment or user-history features — the golden labels show intent and
escalation are decidable from the tweet that entered.

## 2. Data and brand selection

The dataset contains 2,811,774 tweets: each has `tweet_id`, `author_id`, `inbound`, `created_at`,
`text`, `response_tweet_id`, `in_response_to_tweet_id`. Directed pairs (a customer tweet replied
to by a brand account) are the atomic supervision unit.

| Brand | Agent tweets | Clean customer→agent pairs |
|-------|-------------:|---------------------------:|
| AmazonHelp | 169,840 | – |
| AppleSupport | 106,860 | – |
| **Uber_Support** | **56,270** | **55,182** |
| SpotifyCares | 43,265 | – |
| Delta | 42,253 | – |

**Uber_Support** was chosen because it is the second-largest support account with the largest
volume of *directed* customer→agent pairs, so RAG grounding has rich coverage and intent labels
are unambiguous. The optional Banking77 dataset is not needed because the primary dataset is
self-sufficient for the brand agent.

**Pair extraction rule** (decision D3): a customer message is kept if `inbound==True` **and** its
`tweet_id` appears in `Uber_Support`'s `in_response_to_tweet_id`. Filtering retweets/marketing
pages and non-directed mentions yields 55,182 pairs `(customer_text, customer_id, agent_text,
agent_id)`.

**Intent taxonomy** (10 classes, decision D1): `trip_fare_billing`, `driver_behavior`,
`trip_cancellation`, `app_technical`, `account_access`, `uber_eats_order`, `lost_item`,
`safety_concern`, `general_inquiry`, `service_feedback`. Derived by clustering the top Uber
issues in the data, not by guessing.

## 3. Pipeline

All components run locally through Ollama:
- **Embeddings:** `mxbai-embed-large` (1024-dim).
- **Optional LLM:** `llama3.2:1b` (used only for the judge; see §6).

### 3.1 Intent classification
A **few-shot centroid classifier**: for each of the 10 intents, 25 representative customer
messages are sampled from the 55k pairs and embedded; a message is labeled by nearest centroid
(cosine). If the top centroid is close to a tie, a keyword classifier falls back. Training is
done **outside** the golden set (anchors are drawn from the raw 55k pairs, never from the 160
golden examples).

### 3.2 Reply generation (RAG)
A **retrieval-only** responder: the top-1 most similar historical customer message is found in a
2,000-pair index by cosine similarity, and the agent returns the real Uber reply that resolved
it. This guarantees replies are (a) grounded in real brand behavior, (b) personally verified by
Uber, and (c) zero hallucination — ideal for a customer-support setting.

### 3.3 Escalation
A **deterministic rule engine** over 12 word-boundary patterns with a stated reason, e.g.:
- legal (`police`, `court`, `sue`, `legal action`), safety (`scared for my life`, `accident`,
  `wouldn't unlock`, `safety`), fraud/security (`cloned`, `unauthorized charge`, `hacked`,
  verification-code misuse), repeated/complex (`still waiting`, `3rd time`, months-long lockout),
  no-service (`never picked up`, fake photo), and minor-transport.
- **Critical bug fixed during development:** naive substrings matched inside innocent words
  (`"sue"` inside `"issue"`, `"lied"` inside `"applied"`, `"analysis"`, `"restaurant"`…). All
  patterns now use word boundaries (decision D10). This alone took escalation precision from
  0.34 → 0.86.

## 4. Golden set

**160 examples, hand-labeled** (`golden_set/uber_golden.json`). Each entry has `message`,
`agent_reply` (the real historical Uber reply), `intent`, `escalation`, `escalation_reason`.
Two mislabels found during escalation auditing were corrected and documented (decision D12).

**How it was sampled and labelled (deliverable note).** `extract_golden_draft.py` takes a
uniform random sample of 160 of the 55,182 pairs (`random.seed(42)`), one message + its real
agent reply each. A human reviewer then read every message and assigned `(intent, escalate,
reason)` through an explicit index→label dictionary in `label_golden_hand.py` (fully
reproducible — no LLM guesses; an earlier 1B-LLM labeling attempt produced unusable labels and
was discarded). Label guide: intent = the dominant actionable issue; escalate = only cases a
human *must* judge within one reply (safety, legal, fraud/account-takeover, repeated-unresolved
with high stakes).

| Intent | n | | Intent | n |
|--------|--|-|--------|--|
| general_inquiry | 34 | | driver_behavior | 14 |
| account_access | 25 | | app_technical | 9 |
| trip_fare_billing | 18 | | safety_concern | 4 |
| uber_eats_order | 18 | | lost_item | 3 |
| trip_cancellation | 18 | | **escalated** | **12** (7.5%) |
| service_feedback | 17 | | total | 160 |

An earlier LLM-labeling attempt with the 1B model produced unusable JSON (78.5% fell back to
`general_inquiry`), confirming hand-labeling was the right choice for a credible golden set.

## 5. Evaluation design

**Baselines.**
- **Trivial:** always the majority intent, never escalates.
- **Simple:** keyword-based intent + same RAG reply, never escalates.

**Metrics.**
- **Intent:** top-1 accuracy (per class and macro).
- **Escalation:** precision / recall / F1 (auto-handle vs human).
- **Reply quality (objective):** *grounding similarity* — cosine between the agent-emitted reply
  and the historical Uber reply for the same-style issue, using `mxbai-embed-large`. No LLM
  subjectivity, deterministic and cheap.
- **LLM-as-judge:** a contrastive judge (A=agent reply vs B=historical reply → "A / B / EQUAL")
  on a fixed 30-sample set, plus a **judge-vs-human calibration** (§6).

All runs use fixed seeds, so every number is reproducible end-to-end.

## 6. Results

### 6.1 Intent classification

| System | Accuracy |
|--------|---------:|
| Trivial (majority) | 21.2% |
| Simple (keywords) | 48.1% |
| **Our agent** | **47.5%** |

The agent is **2.2×** the trivial baseline. Per class the embedding classifier shines on
long/paraphrased messages (`trip_fare_billing` 78%, `uber_eats_order` 56%, `account_access`
52%).

Top confusions (all intuitive):
- `trip_cancellation → trip_fare_billing` (7): "driver cancelled and I was charged…" contains
  both signals.
- `general_inquiry → app_technical` (5), `driver_behavior → trip_fare_billing` (4),
  `service_feedback → safety_concern` (4): genuinely ambiguous tweets, not system noise.

### 6.2 Escalation

| System | P | R | F1 |
|--------|--:|--:|--:|
| Simple / Trivial (never escalate) | 0.00 | 0.00 | 0.00 |
| **Our agent** | **0.86** | **1.00** | **0.92** |

- **0 false negatives** — every one of the 12 truly-escalatable cases (accident, safety, fraud,
  legal, account takeover) is caught with a matching reason.
- **2 false positives**, both defensible: a "you guys get hacked again?" complaint and a
  "locked out of my account for months" case — a human queue would reasonably triage both.

### 6.3 Reply quality (objective grounding)

| System | Mean sim | ≥0.7 | ≥0.8 |
|--------|---------:|-----:|-----:|
| Trivial | 0.686 | 45% | 8% |
| Simple | 0.814 | 94% | 56% |
| **Our agent** | **0.814** | **94%** | **56%** |

94% of agent replies are within 0.7 cosine of a real historical Uber reply for the same issue —
the replies *read like Uber*. The trivial generic template is measurably worse (45%).

### 6.4 LLM-as-judge with honest calibration

The harness ships a contrastive LLM judge (A/B/EQUAL vs the historical reply). On 30 samples it
preferred the agent reply 57% of the time (win-or-tie). But does it agree with a human? We
calibrated it against a human reviewer on **14 samples**:

| Agreement metric | Value |
|------------------|-------|
| Exact match | 36% |
| Strict A/B decisions | **50%** (≈ chance) |
| With ties tolerated | 64% |

**Finding:** the tiny local 1B model is not reliable as a judge — its absolute scores are frozen
and its binary preference is at chance. This is precisely why the harness exposes *both* an
objective grounding metric and a calibrated judge: we do **not** gate on the 1B judge, and we
report its agreement openly. On a stronger LLM (e.g. a hosted model) the same judge code applies
directly.

## 7. Failure analysis (honest look)

1. **Ambiguous multi-intent tweets** — the largest intent-error source. Example: "driver
   cancelled on me and I was charged $5" is genuinely fare *and* cancellation.
2. **RAG reply can be the wrong conversation** — embeddings are topic-shaped, so an unrelated
   similar-sounding complaint can win top-1 (e.g. personalized-name mismatches).
3. **Routine messages about large money** (`$150 cleaning fee`) are correctly auto-handled but
   users may expect a human — an explicit "our team is reviewing your cleaning-fee report" would
   improve perceived quality.
4. **Retweets/URL-only messages** carry little signal and occasionally confuse the classifier.
5. **Escalation is rules-first** — a genuinely novel incident phrasing without known keywords
   would slip through (recall is 100% only on *this* taxonomized data).

## 8. What is misleading about my headline number?

*Headline: "intent accuracy 47.5%, escalation F1 0.92, reply grounding 0.814."* Every number
has a catch, and a reviewer shouldn't have to discover it.

1. **47.5% intent accuracy is high only against weak baselines.** The majority class is 21%, so
   the agent is 2.2× better — but that is still roughly *one message in two routed to the wrong
   queue*. The embeddings model here is a compact local embedder with no fine-tuning; a
   fine-tuned classifier (or LLM) would move this substantially. Do not read 47.5% as "good
   enough for production routing."
2. **The escalation F1 (0.92) is measured on 12 positive examples.** High precision/recall on
   12 hand-picked escalation cases is encouraging, but the validator's variance is high; one
   relabeled case changes it noticeably. It is also *rules-based*, so it generalizes only as far
   as the 12 patterns.
3. **The reply-grounding number (0.814) flatters the system.** Uber's real replies are highly
   templated ("We're here to help! Send us a note…"), so an embedding similarity score is easy
   to inflate: *any* brand-style template scores high even when it doesn't address the tweet.
   The metric says "reads like Uber," **not** "solves the customer's problem."
4. **The LLM-as-judge is calibrated and still weak.** The tiny local 1B judge agrees with a
   human ~50% on strict A/B (chance). Reporting a "win/tie rate" from it without this calibration
   would be misleading — that's exactly why the harness ships the objective grounding metric.
5. **The golden set is 160 examples by one reviewer.** Intent labels are arguably subjective
   (e.g. a "charged $5 because driver cancelled" tweet), and a second annotator would change the
   accuracy number by a few points.

## 9. What I'd do with one more week

- **Intent:** fine-tune the classifier (e.g. a small transformer sentence pair / last-layer
  heads) on the 10 intent labels using the 55k pairs for weak supervision; target 65%+ on the
  same golden set. Add a second human annotator and measure inter-annotator agreement.
- **Replies:** synthesize the top-1 retrieved reply with a stronger (gated/hosted) LLM that is
  *restricted to the retrieved facts*, keeping citations — best of both RAG and generation.
- **Escalation:** learn a small reranker/classifier on top of the rules (rules stay as
  hard-guards), and surface escalation reasons in a triage UI with an SLA.
- **Judge:** replace the 1B judge with a larger model, then re-measure judge-human agreement;
  target ≥80% before trusting it as a gate.
- **Coverage:** expand the RAG index to ~20k pairs and add de-duplication so template replies
  don't dominate retrieval.

---

*Report validity: all numbers come from fixed-seed runs in `run_eval.py`, `analyze_results.py`,
and `failure_analysis.py`, using the committed golden set and generated data artifacts.*

## Attributions

- **Dataset:** Kaggle, *Customer Support on Twitter* — `thoughtvector/customer-support-on-twitter`
  (~3M tweets; the subset used here, 2.8M rows, is documented in §2). Rendered verbatim courtesy
  of the original uploader. (Optional Banking77 was not needed.)
- **Models (run locally via Ollama):** `mxbai-embed-large` (Mixedbread AI, Apache-2.0) for all
  embeddings; `llama3.2:1b` (Meta, Llama-3.2 license) used only as the LLM-as-judge; both served
  by Ollama (`ollama.com`).
- **Libraries:** NumPy, scikit-learn (metrics), pandas, requests. No code was copied from
  another implementation; any patterned ideas (centroid few-shot classification, RAG retrieval)
  are standard and follow public ML practice.