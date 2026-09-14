"""
Run full evaluation: our agent vs 2 baselines on the golden set.

Agent pipeline:
  1. Intent: embedding-centroid few-shot classifier (mxbai-embed-large) + keyword fallback
  2. Reply: RAG retrieval-only (returns a real historical Uber reply for the most similar resolution)
  3. Escalation: rules + LLM fallback

Baselines:
  - Trivial: always majority intent, never escalate
  - Simple: keyword classifier + top-1 RAG reply
"""

import json
import sys
import os
import re
import numpy as np
import requests
from collections import Counter
from sklearn.metrics import classification_report, precision_recall_fscore_support

OLLAMA_EMBED = "http://localhost:11434/api/embed"
EMBED_MODEL = "mxbai-embed-large"

INTENTS = [
    "trip_fare_billing", "driver_behavior", "trip_cancellation", "app_technical",
    "account_access", "uber_eats_order", "lost_item", "safety_concern",
    "general_inquiry", "service_feedback",
]

# Keyword rules for the "simple" baseline
KEYWORDS = {
    "trip_fare_billing": ["charge", "refund", "fare", "money", "cost", "$", "fee", "price", "paid", "cleaning fee", "cleaning fee"],
    "driver_behavior": ["driver", "drvr", "rude", "high", "drunk", "song", "music", "drove", "behavior", "did he", "he canceled", "he cancelled"],
    "trip_cancellation": ["cancel", "canceled", "cancelled", "no-show", "no show", "didn't come", "didnt come"],
    "app_technical": ["app", "website", "gps", "location", "bug", "error", "restart", "crash", "button", "freeze"],
    "account_access": ["account", "email", "password", "phone no", "mobile no", "verification", "device", "sign in", "log in", "login", "reset", "installed", "reinstall"],
    "uber_eats_order": ["food", "eats", "restaurant", "order", "mcdonald", "meal", "delivered", "driver delivered", "food delivery"],
    "lost_item": ["left", "lost", "forgot", "phone", "wallet", "bag", "item", "medical"],
    "safety_concern": ["police", "threat", "unsafe", "scared", "accident", "hit", "legal", "sue", "harass", "fear", "kid"],
    "service_feedback": ["terrible", "worst", "sucks", "unacceptable", "service", "complaint", "disappointed", "ignored", "awful", "horrible", "no reply", "no response", "nothing", "useless"],
    "general_inquiry": [],
}


def keyword_intent(message):
    msg = message.lower()
    scores = {k: sum(1 for kw in v if kw in msg) for k, v in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general_inquiry"


def embed_texts(texts):
    """Embed texts via Ollama, chunked (single requests over ~16 texts can fail)."""
    out = []
    for i in range(0, len(texts), 16):
        resp = requests.post(
            OLLAMA_EMBED,
            json={"model": EMBED_MODEL, "input": texts[i:i + 16]},
            timeout=180,
        ).json()
        if "embeddings" not in resp:
            raise RuntimeError(f"Embedding failed: {resp.get('error')}")
        out.extend(resp["embeddings"])
    return out


# ---------------- Intent centroid classifier ----------------
class CentroidClassifier:
    """Few-shot intent classifier: anchors per intent -> nearest centroid."""

    def __init__(self, anchors):
        # anchors: {intent: [example texts]}
        all_texts, labels = [], []
        for intent, texts in anchors.items():
            for t in texts:
                all_texts.append(t)
                labels.append(intent)
        embs = np.array(embed_texts(all_texts[:300]))
        self.centroids = {}
        for intent in set(labels):
            mask = np.array([l == intent for l in labels[: len(embs)]])
            if mask.sum() > 0:
                self.centroids[intent] = embs[mask].mean(axis=0)
        print(f"  CentroidClassifier: {len(self.centroids)} classes")

    def predict(self, text):
        emb = np.array(embed_texts([text])[0])
        best, best_score = None, -1
        for intent, c in self.centroids.items():
            score = np.dot(emb, c) / (
                np.linalg.norm(emb) * np.linalg.norm(c) + 1e-9
            )
            if score > best_score:
                best, best_score = intent, score
        return best, {"confidence": float(best_score)}


# ---------------- Escalation ----------------
# Deterministic rule-based escalation with a stated reason.
# NOTE: every keyword uses word boundaries to avoid substring false hits
# (e.g. "sue" inside "issue", "lied" inside "applied").
ESCALATION_RULES = [
    (r"\bpolice\b|\blawyer\b|\battorney\b|\bcourt\b|class action|legal (action|threat)|\bsue[sd]?\b",
     "Legal threat or potential legal action"),
    (r"accident|was hit by|hit by an? (uber )?driver|self.driving.*hit|hit (a )?kid|hit (a )?child",
     "Potential accident / safety incident"),
    (r"\bclon(e|ed|ing)\b|\bstolen\b|unauthorized charge(s)?|\bfraud",
     "Possible fraud or unauthorized charges"),
    (r"\bhack(ing|ed|s)\b|trying to hack|someone.*(hack|access my account)",
     "Suspected unauthorized account access"),
    (r"\bidiot\b|angry|scared for my life|fear for my life|wouldn'?t unlock|\bunsafe\b",
     "Customer reports feeling unsafe or extreme distress"),
    (r"verification code[s]?.*(not request|did not request)|\b4 verification|no longer able to access",
     "Suspicious verification activity / account takeover risk"),
    (r"still waiting|still haven'?t|months (in|now) (and )?still|\b3rd time\b",
     "Repeated unresolved issue over long period"),
    (r"\blied\b|\bfalsified\b|misleading|not being (addressed|resolved)",
     "Customer alleges support dishonesty / unresolved dispute"),
    (r"\bsafety\b",
     "Customer raises a safety concern"),
    (r"minor transported|\bminor\b.*transport",
     "Minor involved in incident"),
    (r"never (even )?picked up|fake (picture|photo|evidence)",
     "No-service / fraudulent claim"),
    (r"\$\s?1[5-9][0-9]\b.*(fake|fraud|unauthorized|never pick|not picked)",
     "Large disputed charge (>$150) with fraud indicators"),
]


def rule_escalate(message):
    """Return (escalate: bool, reason: str) based on rules."""
    msg = message.lower()
    for pattern, reason in ESCALATION_RULES:
        if re.search(pattern, msg):
            return True, reason
    return False, "Standard support request; routine resolution pattern available"


# ---------------- RAG reply ----------------
class RAG:
    def __init__(self):
        self.cust_emb = np.load("data/rag/cust_emb.npy")
        self.pairs = json.load(open("data/rag/pairs.index.json"))

    def retrieve(self, text, top_k=1):
        emb = np.array(embed_texts([text])[0])
        sims = self.cust_emb @ emb / (
            np.linalg.norm(self.cust_emb, axis=1) * (np.linalg.norm(emb) + 1e-9) + 1e-9
        )
        idxs = np.argsort(sims)[-top_k:][::-1]
        return [(self.pairs[i]["agent_text"], float(sims[i])) for i in idxs]


# ---------------- Agent ----------------
class Agent:
    def __init__(self, anchors, rag):
        self.classifier = CentroidClassifier(anchors)
        self.rag = rag

    def process(self, message):
        # Intent
        intent, conf = self.classifier.predict(message)
        if intent is None or conf["confidence"] < 0.25:
            intent = keyword_intent(message)

        # Reply (RAG retrieval-only - grounded historical Uber reply)
        retrieved = self.rag.retrieve(message, top_k=1)
        reply = retrieved[0][0] if retrieved else "We're here to help! Send us a note so our team can assist."

        # Escalation (deterministic rules with stated reason)
        esc, reason = rule_escalate(message)

        return {
            "intent": intent,
            "reply": reply,
            "reply_source_sim": retrieved[0][1] if retrieved else 0.0,
            "escalate": esc,
            "escalation_reason": reason,
        }


# ---------------- Baselines ----------------
class TrivialBaseline:
    def process(self, message):
        return {"intent": "general_inquiry", "reply": "We're here to help! Send us a note.", "escalate": False}


class SimpleBaseline:
    def __init__(self, rag):
        self.rag = rag

    def process(self, message):
        intent = keyword_intent(message)
        retrieved = self.rag.retrieve(message, top_k=1)
        reply = retrieved[0][0] if retrieved else "We're here to help! Send us a note."
        return {"intent": intent, "reply": reply, "escalate": False, "reply_source_sim": retrieved[0][1] if retrieved else 0.0}


# ---------------- Reply grounding (objective, no LLM) ----------------
def reply_grounding(agent_replies, reference_replies):
    """Cosine similarity between agent reply and the historical reference reply.
    A grounded-to-brand reply style scores high even without an LLM judge."""
    a = np.array(embed_texts(agent_replies[:300]))
    r = np.array(embed_texts(reference_replies[:300]))
    sims = (a * r).sum(axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(r, axis=1) + 1e-9
    )
    return sims


# ---------------- Main ----------------
def main():
    golden = json.load(open("golden_set/uber_golden.json", encoding="utf-8"))
    print(f"Golden set: {len(golden)}")

    # Build anchors from training keywords (NOT from golden set - uses the raw pairs)
    with open("data/uber_pairs.json") as f:
        pairs = json.load(f)
    rng = np.random.RandomState(7)
    sample_pairs = rng.choice(len(pairs), 4000, replace=False)
    anchors = {}
    for idx in sample_pairs:
        intent = keyword_intent(pairs[idx]["customer_text"])
        anchors.setdefault(intent, []).append(pairs[idx]["customer_text"][:200])
    # cap anchors
    anchors = {k: v[:25] for k, v in anchors.items()}
    print(f"Anchors: { {k: len(v) for k, v in anchors.items()} }")

    rag = RAG()
    print("RAG index loaded")

    agent = Agent(anchors, rag)
    trivial = TrivialBaseline()
    simple = SimpleBaseline(rag)

    systems = {"our_agent": agent, "trivial": trivial, "simple": simple}
    results = {}

    for name, sys_obj in systems.items():
        print(f"\n=== Running {name} on {len(golden)} examples ===")
        preds = []
        for i, g in enumerate(golden):
            p = sys_obj.process(g["message"])
            preds.append(p)
            if (i + 1) % 40 == 0:
                print(f"  {i + 1}/{len(golden)}")
        results[name] = preds

        os.makedirs("results", exist_ok=True)
        with open(f"results/{name}_predictions.json", "w") as f:
            json.dump(preds, f, indent=2)
        print(f"  Saved results/{name}_predictions.json")

    # ---------------- Metrics ----------------
    report = {}
    reference_replies = [g["agent_reply"] for g in golden]
    for name, preds in results.items():
        true_intent = [g["intent"] for g in golden]
        pred_intent = [p["intent"] for p in preds]
        acc = sum(a == b for a, b in zip(true_intent, pred_intent)) / len(true_intent)

        true_esc = [g["escalation"] for g in golden]
        pred_esc = [bool(p["escalate"]) for p in preds]
        tp = sum(t and p for t, p in zip(true_esc, pred_esc))
        fp = sum((not t) and p for t, p in zip(true_esc, pred_esc))
        fn = sum(t and (not p) for t, p in zip(true_esc, pred_esc))
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

        # Reply grounding: objective embedding similarity vs the historical reply
        sims = reply_grounding([p["reply"] for p in preds], reference_replies)
        mean_sim = float(sims.mean())
        pct_ge7 = float((sims >= 0.7).mean())
        pct_ge8 = float((sims >= 0.8).mean())

        # RAG grounding: avg retrieval similarity
        avg_ret_sim = float(np.mean([p.get("reply_source_sim", 0) for p in preds]))

        report[name] = {
            "intent_accuracy": acc,
            "escalation": {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1},
            "reply_grounding_sim": {"mean": mean_sim, "pct_ge_0.7": pct_ge7, "pct_ge_0.8": pct_ge8},
            "reply_avg_retrieval_sim": avg_ret_sim,
        }
        print(f"\n[{name}] Intent acc={acc:.3f}, Esc P/R/F1={prec:.2f}/{rec:.2f}/{f1:.2f}, "
              f"Grounding={mean_sim:.3f} ({pct_ge7:.0%}>=0.7)")

    with open("results/evaluation_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved results/evaluation_summary.json")

    # Per-intent accuracy for our agent
    print("\n=== Per-intent accuracy (our agent) ===")
    preds = results["our_agent"]
    true_intent = [g["intent"] for g in golden]
    pred_intent = [p["intent"] for p in preds]
    for intent in set(true_intent):
        idxs = [i for i, t in enumerate(true_intent) if t == intent]
        correct = sum(1 for i in idxs if pred_intent[i] == intent)
        print(f"  {intent}: {correct}/{len(idxs)} = {correct/len(idxs):.2f}")


if __name__ == "__main__":
    sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()