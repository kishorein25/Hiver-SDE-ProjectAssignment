# Decision Log

Every non-trivial product/engineering decision, with the reasoning that mattered at the time.

1. **Intent taxonomy: 10 brand-specific classes over generic "support" labels.**
   Classes were derived bottom-up from the dominant Uber issue clusters in the data
   (`trip_fare_billing`, `driver_behavior`, `trip_cancellation`, `app_technical`,
   `account_access`, `uber_eats_order`, `lost_item`, `safety_concern`, `general_inquiry`,
   `service_feedback`) rather than copied from an unrelated chatbot taxonomy. Result: labels are
   trainable from the data and meaningful to a human queue.

2. **Restricted to the primary dataset only (Kaggle `thoughtvector/customer-support-on-twitter`).**
   The optional Banking77 dataset would not transfer (retail banking vs ride-hailing) and the
   primary dataset fully supports the brand agent. Using only the spec dataset also keeps the
   deliverable self-contained.

3. **Pair extraction rule: `inbound == True` AND `tweet_id` ∈ brand's `in_response_to_tweet_id`.**
   This is the only clean, directed signal of "customer asked, brand answered." Alternative rules
   (thread walking, keyword "@mention") either required fragile reply-chains or pulled in
   undirected chatter. Yielded 55,182 clean pairs for Uber_Support.

4. **Local embedder `mxbai-embed-large` + local 1B LLM on Ollama instead of a cloud API.**
   No API key was available. This keeps the *entire* pipeline runnable offline and costs nothing
   per run; the tradeoff (weak 1B reasoning) was accepted and is reported transparently.

5. **RAG retrieval-only replies, not generative ones.**
   A support bot must never hallucinate fixes. Retrieving the real Uber reply that resolved the
   most similar historical ticket gives grounded, verified, brand-consistent wording with zero
   generation risk. Generative drafting was dropped after puppeteered 1B output proved unusable.

6. **RAG index = 2,000 strided pairs.** Memory/timing tradeoff: 2k pairs × 1024-dim float32 ≈
   8 MB, retrieval is microseconds, and coverage of the taxi domain is wide. Build is resumable
   with checkpoints so a broken run restarts rather than restarting 2k embeddings.

7. **Golden set hand-labeled (160 examples), not LLM-labeled.**
   First attempt used the 1B model; it produced malformed JSON and 78.5% of messages collapsed
   into `general_inquiry`. Hand-labeling guarantees a credible ground truth in the exact 150–250
   window the assignment requests, at the cost of ~hours of human effort.

8. **Escalation decision is deterministic rules + stated reason, not an LLM call.**
   The 1B model said `{"escalate": true}` for *every* test message — useless as a router. Rules
   are explainable, fast, testable, and give the human queue a reason string per case. (TF-IDF /
   learned rankers were unnecessary for the taxonomy's volume.)

9. **Escalation covers 12 patterns with word boundaries.**
   Chosen empirically to match the 13 hand-labeled escalate cases while minimizing false alarms
   (safety, legal, fraud, account takeover, repeated-complex, minor-transport, no-service).

10. **"sue" inside "issue" bug — the biggest quality improvement.**
    Naive keyword regex matched substrings: `sue` in `issue`, `lied` in `applied`, `court` in
    `your`, `us` in `house`… Escalation precision was 0.34; after word-boundary fixes and
    removing over-broad `$1xx` patterns it reached **0.86 / 1.00 / 0.92** with the same recall.

11. **Frozen random seeds everywhere.**
    Anchor sampling (`RandomState(7)`), judge sample (`RandomState(11)`) and calibration
    (`RandomState(2026)`) are fixed so every reported number is reproducible, and evaluation
    runs do not touch the golden set's examples for training.

12. **Auditing the golden set corrected 2 mislabels (documented here for transparency).**
    Entry 124 was labeled a safety escalation ("scared for life") but its actual tweet was a
    routine "frustrating/stressful" complaint — corrected to `service_feedback`, not escalated.
    Entry 114's reason misattributed "verification codes"; corrected to the actual issue
    (unresolved refund + request for a human). The metrics therefore measure the *corrected*
    truth, improving both label sanity and evaluation validity.

13. **Reply quality measured objectively by embedding grounding, not a raw judge.**
    Cosine similarity between the agent reply and the real historical Uber reply for the same
    issue type is deterministic and encodes "does this read like Uber?" The LLM judge was kept in
    the harness but explicitly calibrated (see D14).

14. **LLM-as-judge calibration is reported, not hidden.**
    The 1B judge agrees with a human only ~50% on strict A/B (chance). Instead of scrapping the
    judge mandate, we ship the judge, show its agreement with a human, and gate on the objective
    metric instead — the honest, defensible choice, and the agreement numbers are in the report.

15. **Two-tier baseline design.**
    *Trivial* (majority class, no escalation) sets the floor for AI-washing detection; *Simple*
    (keywords + same RAG reply) isolates the marginal value of the embedding classifier and
    escalation. Both are deliberately echo-fragile to highlight what the full agent adds.