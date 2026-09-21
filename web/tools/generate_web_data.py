"""web\tools\generate_web_data.py - bake committed repo data into web/js/data.js.

Runnable any time; reads the committed JSON assets in this repo and emits a
single static JS file so web/index.html works offline with zero setup:

    python web\\tools\\generate_web_data.py

The dashboard then consumes window.WEB_DATA for every view EXCEPT Live Chat,
which talks to the real agent over the server API (web_server.py).
"""

import json
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from run_eval import keyword_intent  # noqa: E402

INTENT_LABELS = {
    "trip_fare_billing": "Trip / Fare / Billing",
    "driver_behavior": "Driver Behavior",
    "trip_cancellation": "Trip / Cancellation",
    "app_technical": "App / Technical",
    "account_access": "Account / Access",
    "uber_eats_order": "Uber Eats Order",
    "lost_item": "Lost Item",
    "safety_concern": "Safety Concern",
    "general_inquiry": "General Inquiry",
    "service_feedback": "Service Feedback",
    "out_of_scope": "Out of Scope (Non-Uber)",
}

INTENTS = list(INTENT_LABELS.keys())[:10]

FILE_CATALOG = [
    ("README.md", 0, "Single source of truth: how to run, how to use, report, decisions, FAQ."),
    ("run_eval.py", 0, "Core evaluation: agent + baselines on the 160 golden examples."),
    ("interact.py", 0, "CLI chat - type any message, get an instant grounded reply."),
    ("web_server.py", 0, "Local dashboard server: serves web/ + POST /api/chat (real agent)."),
    ("build_pairs.py", 0, "Extracts Uber_Support customer->agent pairs from twcs.csv."),
    ("build_rag_index.py", 0, "Builds the RAG embedding index (data/rag/)."),
    ("label_golden_hand.py", 0, "Hand-labels the 160-example golden set."),
    ("data/uber_pairs.json", 0, "55,182 Uber customer->agent conversation pairs."),
    ("data/rag/pairs.index.json", 0, "2,000-pair RAG search index (text half)."),
    ("data/rag/cust_emb.npy", 0, "2,000 x 1024-d embedding matrix (vector half)."),
    ("golden_set/uber_golden.json", 0, "160 hand-labeled examples - the answer key."),
    ("results/evaluation_summary.json", 0, "Final metrics for all three systems."),
]


def _human_size(n):
    if n < 1024:
        return "%d B" % n
    if n < 1024 * 1024:
        return "%.1f KB" % (n / 1024)
    return "%.1f MB" % (n / (1024 * 1024))


def load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return json.load(f)


def per_intent_accuracy(golden, preds):
    by_true = defaultdict(list)
    for i, g in enumerate(golden):
        by_true[g["intent"]].append(i)
    rows = []
    for intent, idxs in sorted(by_true.items()):
        correct = sum(1 for i in idxs if preds[i].get("intent") == intent)
        rows.append({
            "intent": intent,
            "correct": correct,
            "total": len(idxs),
            "accuracy": correct / len(idxs),
        })
    rows.sort(key=lambda r: -r["accuracy"])
    return rows


def main():
    golden = load("golden_set/uber_golden.json")
    summary = load("results/evaluation_summary.json")

    predictions = {}
    for model in ("our_agent", "simple", "trivial"):
        try:
            predictions[model] = load("results/%s_predictions.json" % model)
        except FileNotFoundError:
            predictions[model] = []

    # Per-system metrics come straight from the committed evaluation summary.
    metrics = {}
    for model in ("our_agent", "simple", "trivial"):
        if model in summary:
            m = summary[model]
            metrics[model] = {
                "intent_accuracy": m["intent_accuracy"],
                "escalation": m["escalation"],
                "reply_grounding": m.get("reply_grounding_sim", {}),
                "avg_retrieval_sim": m.get("reply_avg_retrieval_sim", 0.0),
            }
        else:
            metrics[model] = None

    per_intent = {
        model: per_intent_accuracy(golden, preds)
        for model, preds in predictions.items()
        if preds
    }

    # Sample RAG pairs across intents for the chat grounding demo.
    rng = random.Random(7)
    rag_pairs = load("data/rag/pairs.index.json")
    sampled = defaultdict(list)
    for p in rag_pairs:
        sampled[keyword_intent(p["customer_text"])].append({
            "customer_text": p["customer_text"][:200],
            "agent_text": p["agent_text"][:200],
        })
    pairs_sample = []
    for intent in INTENTS:
        bucket = sampled.get(intent, [])
        pick = bucket if len(bucket) <= 40 else rng.sample(bucket, 40)
        pairs_sample.extend(pick)

    # File browser entries with live sizes from disk.
    files = []
    for path, _sz, note in FILE_CATALOG:
        full = os.path.join(ROOT, path)
        size = os.path.getsize(full) if os.path.exists(full) else -1
        files.append({
            "path": path,
            "size": _human_size(size) if size >= 0 else "n/a",
            "note": note,
        })

    data = {
        "intents": INTENTS,
        "intent_labels": {k: INTENT_LABELS[k] for k in INTENTS},
        "metrics": metrics,
        "per_intent": per_intent,
        "golden": [
            {
                "message": g["message"],
                "agent_reply": g["agent_reply"],
                "intent": g["intent"],
                "escalation": bool(g["escalation"]),
                "escalation_reason": g.get("escalation_reason", ""),
            }
            for g in golden
        ],
        "predictions": predictions,
        "pairs_sample": pairs_sample,
        "files": files,
        "sample_questions": [
            "I was charged twice for the same trip",
            "My driver was rude and honked at me the whole ride",
            "I left my wallet in the uber, how can I get it back?",
            "Someone is trying to hack my account right now",
            "My uber eats order arrived missing 2 items",
            "I got a $150 cleaning fee but there was no mess",
            "My driver was so rude and yelled at me",
        ],
    }

    out_dir = os.path.join(ROOT, "web", "js")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "data.js")
    with open(out, "w", encoding="utf-8") as f:
        f.write("/* Generated by web/tools/generate_web_data.py - do not edit by hand. */\n")
        f.write("window.WEB_DATA = ")
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    print("Wrote %s (%d KB, %d golden, %d sampled RAG pairs)" % (
        out, os.path.getsize(out) // 1024, len(data["golden"]), len(data["pairs_sample"])))


if __name__ == "__main__":
    main()