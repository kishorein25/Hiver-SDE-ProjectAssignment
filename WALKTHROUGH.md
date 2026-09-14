# Beginner Walkthrough — Full Project Working Process

This document explains the entire project from scratch, with no assumptions
about your background. Read it top-to-bottom.

---

## TABLE OF CONTENTS

1. What is this project, in plain English?
2. What do you need installed?
3. The big picture — how it all fits together
4. Step-by-step: The data pipeline
5. Step-by-step: The agent
6. Step-by-step: The evaluation
7. File-by-file guide
8. How to run in Antigravity IDE
9. What to expect when you run it
10. Troubleshooting

---

## 1. WHAT IS THIS PROJECT, IN PLAIN ENGLISH?

Imagine you work at Uber. Thousands of customers tweet at @Uber_Support every
day. Your job is to:

  - Figure out WHAT they're complaining about (intent)
  - Write a reply that sounds like Uber (grounded reply)
  - Decide if a human agent should handle it instead of a bot (escalation)

This project builds a computer program that does all three automatically, using
real tweets from Twitter that customers actually sent to Uber support.

The assignment says: "build an AI support agent that can classify intents,
draft replies, and escalate with a reason — then prove it works."

---

## 2. WHAT DO YOU NEED INSTALLED?

Already on your machine (verified):

  ✓ Python 3.10.11     → runs all the code
  ✓ Ollama             → runs the AI models locally
  ✓ mxbai-embed-large  → converts text into numbers (embeddings)
  ✓ llama3.2:1b        → tiny LLM, used only for evaluation (optional)
  ✓ VS Code / Antigravity IDE → lets you see and run the code

What you DON'T need:
  ✗ No OpenAI key
  ✗ No cloud API
  ✗ No GPU
  ✗ Everything runs locally on your laptop

---

## 3. THE BIG PICTURE — HOW IT ALL FITS TOGETHER

Think of this as a 4-stage factory:

  STAGE 1: RAW DATA (the tweets from Twitter)
       ↓
  STAGE 2: PREPARATION (extract Uber pairs, build search index, label examples)
       ↓
  STAGE 3: THE AGENT (classifies intent, drafts reply, decides escalation)
       ↓
  STAGE 4: EVALUATION (did the agent get it right? how do we know?)

Each stage has specific files that do the work. Here is the complete
pipeline with every file:

  STAGE 1:
    data/twcs.csv              ← raw dataset (you download this from Kaggle)

  STAGE 2:
    build_pairs.py             ← extracts Uber customer→agent conversations
    build_rag_index.py         ← builds the search index (embeddings)
    label_golden_hand.py       ← creates the "answer key" (golden set)
    extract_golden_draft.py    ← helper: samples 160 random conversations

  STAGE 3:
    run_eval.py                ← THE MAIN FILE: runs the agent + 2 baselines

  STAGE 4:
    run_eval.py                ← outputs metrics (intent accuracy, escalation F1, etc.)
    analyze_results.py         ← deeper reply-quality analysis
    failure_analysis.py        ← what went wrong and why
    calibration.py             ← checks if the LLM judge agrees with humans

  RESULTS:
    results/evaluation_summary.json  ← the final numbers

  DOCUMENTATION:
    README.md                  ← how to reproduce everything
    REPORT.md                  ← the 6-page report for Hiver
    DECISION_LOG.md            ← 15 decisions we made and why
```

---

## 4. STEP-BY-STEP: THE DATA PIPELINE

### Step 1: Download the dataset

The dataset is a CSV file with 2.8 million tweets. You get it from Kaggle:
  Website: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
  File:    twcs.csv (about 516 MB)

You already have it at: D:\dataset\twcs\twcs.csv

### Step 2: Extract Uber conversations (build_pairs.py)

What this does:
  Reads the 2.8 million tweets and pulls out ONLY the ones where:
    - A customer tweeted at @Uber_Support (inbound = True)
    - Uber_Support replied to that specific tweet

This creates a list of 55,182 "pairs":
  Pair = (what the customer said → what Uber actually replied)

Why we do this:
  The agent needs to learn from REAL conversations. We can't train on
  random tweets — only on actual customer→agent exchanges.

Command: python build_pairs.py
Output:  data/uber_pairs.json (55,182 conversation pairs)

### Step 3: Build the RAG index (build_rag_index.py)

What is RAG? RAG = Retrieval-Augmented Generation. In plain English:
  When a new customer message comes in, the agent finds the MOST SIMILAR
  past customer message in its database, and returns the real Uber reply
  for that similar past message.

How it works:
  1. Pick 2,000 customer messages from the 55,182 pairs
  2. Convert each message into a "vector" (a list of 1,024 numbers)
     using mxbai-embed-large. This is called "embedding."
  3. Store the vectors in a file (data/rag/cust_emb.npy)
  4. Store the original messages alongside (data/rag/pairs.index.json)

When the agent receives a new message:
  1. Convert it to a vector (same 1,024 numbers)
  2. Compare it to all 2,000 stored vectors
  3. Find the most similar one (highest cosine similarity)
  4. Return the real Uber reply for that match

Command: python build_rag_index.py 2000
Output:  data/rag/cust_emb.npy + pairs.index.json

### Step 4: Create the golden set (label_golden_hand.py)

What is a "golden set"? It's the ANSWER KEY — a set of 160 example
messages where a human has already decided:
  - What is the intent?
  - Should this be escalated?
  - What reason?

How it was created:
  1. Randomly sample 160 pairs from the 55,182 (seed=42 for reproducibility)
  2. A human reads each message and assigns:
     - Intent: one of 10 labels (trip_fare_billing, lost_item, etc.)
     - Escalate: true or false
     - Reason: why escalate (or "standard support request" if not)
  3. These labels are stored in an explicit index→label dictionary

Why hand-label instead of using AI?
  We TRIED using the AI (llama3.2:1b) to label automatically. It failed
  badly — 78% of messages got labeled "general_inquiry" regardless of
  content. Hand-labeling is slower but produces real, trustworthy labels.

Command: python label_golden_hand.py
Output:  golden_set/uber_golden.json (160 hand-labeled examples)
```

---

## 5. STEP-BY-STEP: THE AGENT (inside run_eval.py)

When a new message comes in, the agent does 3 things:

### 5A. CLASSIFY THE INTENT

Tool: CentroidClassifier (embedding-based)

How it works (no LLM involved):
  1. During setup, embed 25 example messages for each of the 10 intents
  2. Calculate the "center point" (centroid) for each intent
  3. When a new message arrives:
     a. Embed it → get a vector of 1,024 numbers
     b. Compare to each centroid
     c. Pick the closest one → that's the intent

Fallback: if no centroid is close enough (confidence < 0.25),
  use a keyword classifier as backup.

The 10 intents:
  trip_fare_billing    — "I was charged too much"
  driver_behavior      — "My driver was rude"
  trip_cancellation    — "Driver cancelled on me"
  app_technical        — "The app isn't working"
  account_access       — "I can't log in"
  uber_eats_order      — "My food order is wrong"
  lost_item            — "I left something in the car"
  safety_concern       — "I feel unsafe"
  general_inquiry      — "How do I..." (catch-all)
  service_feedback     — "Your service is terrible"

### 5B. DRAFT A REPLY

Tool: RAG retrieval (not an LLM!)

How it works:
  1. Embed the new message → 1,024 numbers
  2. Compare to all 2,000 stored customer messages
  3. Find the most similar one
  4. Return the REAL Uber reply for that match

Example:
  New message: "I was charged $50 for a 10-minute ride"
  Most similar past message: "Uber charged me $45 for a short trip"
  Real Uber reply to that: "We're sorry to hear this. Please send us
  a DM with your email so we can look into this."

Why retrieval and not generation?
  A support bot must never invent a fix. Retrieving real replies
  guarantees the answer is something Uber actually said.

### 5C. DECIDE ESCALATION

Tool: rule_escalate (12 word-boundary keyword patterns)

How it works:
  Scan the message for specific words/phrases:
  - "police" / "lawyer" / "court" → Legal threat → escalate
  - "accident" / "hit by" / "safety" → Safety incident → escalate
  - "cloned" / "unauthorized charge" / "hacked" → Fraud → escalate
  - "scared for my life" / "wouldn't unlock" → Unsafe → escalate
  - "3rd time" / "still waiting" / "months" → Repeated issue → escalate
  - "lied" / "falsified" → Support dishonesty → escalate
  - "never picked up" / "fake picture" → No-service fraud → escalate

If none match: "Standard support request; routine resolution pattern"

Why word-boundary rules instead of an LLM?
  The 1B LLM said "escalate: true" for every message we tested.
  Rules are predictable, testable, and always come with a reason.

---

## 6. STEP-BY-STEP: THE EVALUATION

### 6A. THE THREE SYSTEMS COMPARED

The eval runs 3 different approaches on the same 160 golden examples:

  OUR AGENT:     centroid intent + RAG reply + rule escalation
  TRIVIAL:       always says "general_inquiry", never escalates
  SIMPLE:        keyword intent + RAG reply, never escalates

### 6B. THE METRICS

  INTENT ACCURACY:
    What % of the 160 messages got the right intent label?
    Our agent: 47.5%
    Trivial:   21.2% (just picking the majority class)
    Simple:    48.1%

  ESCALATION (P = precision, R = recall, F1 = combined):
    Our agent: P=0.86 R=1.00 F1=0.92
    Trivial:   0 (never escalates)
    Simple:    0 (never escalates)

  REPLY GROUNDING (cosine similarity to real Uber reply):
    Our agent: 0.814 mean, 94% ≥ 0.7
    Trivial:   0.686, 45% ≥ 0.7
    Simple:    0.814, 94% ≥ 0.7

### 6C. THE LLM JUDGE

The assignment requires an "LLM-as-judge rubric." We built one:
  - Give the LLM two replies (ours vs. the real Uber reply)
  - Ask: "Which is better? A, B, or EQUAL?"
  - Measure how often it agrees with a human

Finding: the tiny 1B model agrees with humans only ~50% (chance).
This is reported honestly in the report — we do NOT rely on it.

---

## 7. FILE-BY-FILE GUIDE

  run.py                    Entry point — run everything with --all flag
  run_eval.py               The core: agent + baselines + metrics
  analyze_results.py        Reply grounding scores + LLM judge
  failure_analysis.py       Error audit + contrastive judge + confusions
  calibration.py            Judge vs human calibration (14 samples)
  test_agent.py             Smoke test — 5 messages, see the agent work
  build_pairs.py            Extracts Uber conversation pairs from CSV
  build_rag_index.py        Builds the RAG embedding index
  label_golden_hand.py      Creates the hand-labeled golden set
  extract_golden_draft.py   Helper: random sample for golden set

  golden_set/
    uber_golden.json        160 hand-labeled examples (THE answer key)
    uber_golden_draft.json  The draft before labeling

  data/
    uber_pairs.json         55,182 customer→agent conversation pairs
    rag/
      cust_emb.npy          2,000 embedded customer messages (vectors)
      pairs.index.json      The original 2,000 messages

  results/
    evaluation_summary.json THE final numbers
    our_agent_predictions.json   Agent outputs on all 160 examples
    trivial_predictions.json     Trivial baseline outputs
    simple_predictions.json      Simple baseline outputs
    judge_human_calibration.json Judge-human agreement data

  REPORT.md                 The 6-page report for submission
  DECISION_LOG.md           15 decisions and why
  README.md                 Reproduction instructions
```

---

## 8. HOW TO RUN IN ANTIGRAVITY IDE

1. Open Antigravity IDE
2. Open folder: C:\Users\ks670\Hiver-SDEProjectAssignment
3. Open terminal: Terminal → New Terminal (or Ctrl+`)
4. Make sure you're in the right directory:
   cd C:\Users\ks670\Hiver-SDEProjectAssignment
5. Check Ollama is running:
   ollama list
   (Should show llama3.2:1b and mxbai-embed-large)
6. Run the quick test:
   python test_agent.py
7. Run the full evaluation:
   python run_eval.py
8. Open results\evaluation_summary.json to see the numbers

---

## 9. WHAT TO EXPECT

When you run test_agent.py (~30 seconds):
  You'll see 5 messages classified:
  - Intent labeled for each (e.g., "trip_fare_billing")
  - Escalate true or false
  - A real Uber reply
  - Similarity score (e.g., 0.834 = very similar to a real reply)

When you run run_eval.py (~5 minutes):
  The terminal shows progress (40/160, 80/160, etc.)
  Then shows the final metrics:
    [our_agent] Intent acc=0.475, Esc P/R/F1=0.86/1.00/0.92, Grounding=0.814 (94%>=0.7)
    [trivial]   Intent acc=0.212, Esc P/R/F1=0.00/0.00/0.00, Grounding=0.686 (45%>=0.7)
    [simple]    Intent acc=0.481, Esc P/R/F1=0.00/0.00/0.00, Grounding=0.814 (94%>=0.7)

Then a per-intent breakdown:
    trip_fare_billing: 14/18 = 0.78  (best)
    lost_item:         1/3  = 0.33  (worst)

---

## 10. TROUBLESHOOTING

  "ModuleNotFoundError: No module named 'requests'"
    → pip install requests numpy scikit-learn tqdm

  "Connection refused" for Ollama
    → Open a new terminal and run: ollama serve

  "ModuleNotFoundError: No module named 'numpy'"
    → pip install numpy

  "No module named 'run_eval'" when running test_agent.py
    → Make sure you're in the project folder (cd to it first)

  The numbers look different from what's in the report
    → Check that golden_set/uber_golden.json exists
    → Check that data/rag/ files exist
    → If not, run: python build_pairs.py && python build_rag_index.py 2000
