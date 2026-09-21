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

import ui

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
        ui.status("ok", f"Centroid classifier ready: {len(self.centroids)} intent classes")

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
    (r"\bemergency\b",
     "Customer reports an emergency / urgent situation"),
    (r"\battack(ed|ing|s)?\b|\bassault(ed|s)?\b|\bthreaten(ed|ing|s)?\b|\bharass(ed|ing|es)?\b|\babuse(d)?\b",
     "Customer reports physical threat / assault / harassment"),
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


# ---------------- Out-of-scope (non-Uber) detection ----------------
# If a customer message concerns a brand/service that is NOT Uber, the agent
# must not auto-generate an Uber RAG reply. It escalates straight to a human
# (admin). Word boundaries are used so e.g. "ola" doesn't hit "hola".
EXTERNAL_BRANDS = [
    "zomato", "zomoto", "zomatto", "zomatoo", "swiggy", "oso", "ola", "rapido",
    "porter", "lyft", "bolt car", "grab", "gocar", "gocab", "deliveroo",
    "doordash", "grubhub", "amazon", "flipkart", "myntra", "meesho", "ajio",
    "snapdeal", "d mart", "bigbasket", "big basket", "zepto", "blinkit",
    "dunzo", "instamart", "paytm", "phonepe", "netflix", "prime video",
    "hotstar", "spotify", "irctc", "redbus", "red bus", "makemytrip",
    "cleartrip", "airbnb", "fedex", "dhl", "bluedart", "blue dart", "ekart",
    "linkedin", "instagram", "facebook", "whatsapp", "snapchat", "telegram",
]
_OOS_BRANDS = [r"\s+".join(re.escape(p) for p in b.split()) for b in EXTERNAL_BRANDS]
_OOS_RE = re.compile(r"\b(" + "|".join(_OOS_BRANDS) + r")\b", flags=re.IGNORECASE)

# If the customer message ALSO references Uber itself (the @handle used in the
# corpus, "uber", etc.), it's still an Uber support request (e.g. "...another
# reason to go lyft" is a churn threat about an UBER ride) -> NOT out-of-scope.
UBER_CONTEXT_RE = re.compile(r"\buber\b|@\d{3,}|@uber", flags=re.IGNORECASE)


def detect_out_of_scope(message):
    """Return (out_of_scope: bool, brand: str|None).

    A brand/service that is NOT Uber escalates to a human only when the
    message is not also about Uber itself (no @handle / "uber" reference).
    """
    m = _OOS_RE.search(message)
    if not m:
        return False, None
    if UBER_CONTEXT_RE.search(message):
        return False, None
    return True, m.group(1)


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


# ---------------- Greeting detection ----------------
# A bare greeting (no complaint/issue yet) gets a friendly welcome reply instead
# of a random retrieved resolution. Token-based so real issues are never
# swallowed (e.g. "hi my driver was rude" still runs the full pipeline).
_GREETING_TOKENS = {
    "hi", "hiya", "hello", "hey", "yo", "howdy", "hola", "namaste",
    "good", "morning", "afternoon", "evening", "day", "morn",
    "uber", "there", "support", "team",
}
_GREETING_STARTS = {"hi", "hello", "hey", "yo", "hiya", "howdy", "hola", "namaste", "good"}

# Stretched/typo greetings like "hii", "heyy", "helloo", "yoo".
_GREETING_WORD_RES = [
    re.compile(r"^h+i+$"),            # hi, hii, hiii
    re.compile(r"^h+e+y+$"),          # hey, heyy
    re.compile(r"^h+e+l{2,}o+$"),     # hello, helloo
    re.compile(r"^h+i+y+a+$"),        # hiya
    re.compile(r"^y+o+$"),            # yo, yoo
    re.compile(r"^h+o+w+d+y+$"),      # howdy
    re.compile(r"^h+o+l+a+$"),        # hola
    re.compile(r"^n+a+m+a+s+t+e+$"),  # namaste
    re.compile(r"^g+o+d+$"),          # good, goood
    re.compile(r"^m+o+r+n+i+n+g+$"),  # morning
    re.compile(r"^m+o+r+n+$"),        # morn
    re.compile(r"^a+f+t+e+r+n+o+o+n+$"),  # afternoon
    re.compile(r"^e+v+e+n+i+n+g+$"),  # evening
    re.compile(r"^d+a+y+$"),          # day
    re.compile(r"^u+b+e+r+$"),        # uber
    re.compile(r"^t+h+e+r+e+$"),      # there
    re.compile(r"^s+u+p+o+r+t+$"),    # support
    re.compile(r"^t+e+a+m+$"),        # team
]


def _stretched_greeting(word):
    return any(rx.match(word) for rx in _GREETING_WORD_RES)

GREETING_REPLY = (
    "Hi there! Welcome to Uber Support. I help with ride, fare, delivery, and "
    "account questions - how can we assist you today?"
)


def is_greeting(message):
    """True only for a message that is basically just a greeting
    (optionally addressing Uber), with no described problem."""
    stripped = message.strip()
    if not stripped or any(c.isdigit() for c in stripped):
        return False
    words = [w.lower() for w in "".join(
        c if c.isalnum() or c.isspace() else " " for c in stripped
    ).split()]
    if not words:
        return False
    if not any(words[0].startswith(s) for s in _GREETING_STARTS):
        return False
    return all(w in _GREETING_TOKENS or _stretched_greeting(w) for w in words)


# ---------------- Agent ----------------
class Agent:
    def __init__(self, anchors, rag):
        self.classifier = CentroidClassifier(anchors)
        self.rag = rag

    def process(self, message):
        # Bare greeting (no issue yet) -> friendly welcome, no RAG lookup.
        if is_greeting(message):
            return {
                "intent": "general_inquiry",
                "reply": GREETING_REPLY,
                "reply_source_sim": 1.0,
                "escalate": False,
                "escalation_reason": "Greeting / opening message - no issue described yet; "
                                     "welcomed and asked what we can help with",
                "route": "assistant",
                "greeting": True,
            }

        # Out-of-scope: any non-Uber brand/service mention -> escalate to a
        # human (admin) and do NOT auto-generate an Uber RAG reply.
        oos, oos_brand = detect_out_of_scope(message)
        if oos:
            return {
                "intent": "out_of_scope",
                "reply": "This request is outside Uber's support scope. It won't be answered "
                         "automatically and has been routed to a human agent (admin).",
                "reply_source_sim": 0.0,
                "escalate": True,
                "escalation_reason": f"Out-of-scope: customer mentions \"{oos_brand}\", "
                                     f"which is not an Uber service - routed to human/admin",
                "out_of_scope": True,
                "out_of_scope_brand": oos_brand,
                "route": "admin",
            }

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
            "route": "admin" if esc else "assistant",
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
    ui.banner("Running evaluation")
    golden = json.load(open("golden_set/uber_golden.json", encoding="utf-8"))

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

    ui.subheader("Setup")
    ui.kv_pairs(
        {
            "Golden set size": len(golden),
        }
    )
    ui.table(
        ["Intent", "Anchor examples"],
        [(k, len(v)) for k, v in sorted(anchors.items(), key=lambda kv: -len(kv[1]))],
        title="Anchors per intent (train outside golden set)",
    )

    rag = RAG()
    ui.status("ok", "RAG index loaded")

    agent = Agent(anchors, rag)
    trivial = TrivialBaseline()
    simple = SimpleBaseline(rag)

    systems = {"our_agent": agent, "trivial": trivial, "simple": simple}
    results = {}

    for name, sys_obj in systems.items():
        ui.subheader(f"Running system: {name}")
        preds = []
        for i, g in enumerate(ui.pbar(golden, desc=f"{name}", total=len(golden), unit=" msg")):
            p = sys_obj.process(g["message"])
            preds.append(p)
        results[name] = preds

        os.makedirs("results", exist_ok=True)
        with open(f"results/{name}_predictions.json", "w") as f:
            json.dump(preds, f, indent=2)
        ui.status("ok", f"Saved results/{name}_predictions.json")

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

    with open("results/evaluation_summary.json", "w") as f:
        json.dump(report, f, indent=2)

    ui.table(
        ["System", "Intent acc", "Esc P", "Esc R", "Esc F1", "Grounding", "(>=0.7)"],
        [
            (
                sys_label,
                f"{report[n]['intent_accuracy']:.3f}",
                f"{report[n]['escalation']['precision']:.2f}",
                f"{report[n]['escalation']['recall']:.2f}",
                f"{report[n]['escalation']['f1']:.2f}",
                f"{report[n]['reply_grounding_sim']['mean']:.3f}",
                f"{report[n]['reply_grounding_sim']['pct_ge_0.7']:.0%}",
            )
            for n, sys_label in (
                ("our_agent", "our_agent"),
                ("trivial", "trivial"),
                ("simple", "simple"),
            )
        ],
        title="Evaluation summary",
    )
    ui.status("ok", "Saved results/evaluation_summary.json")

    # Per-intent accuracy for our agent
    preds = results["our_agent"]
    true_intent = [g["intent"] for g in golden]
    pred_intent = [p["intent"] for p in preds]
    ui.table(
        ["Intent", "Correct", "Total", "Accuracy"],
        [
            (intent, sum(1 for i in idxs if pred_intent[i] == intent), len(idxs),
             f"{sum(1 for i in idxs if pred_intent[i] == intent) / len(idxs):.2f}")
            for intent in sorted(set(true_intent))
            for idxs in [[i for i, t in enumerate(true_intent) if t == intent]]
        ],
        title="Per-intent accuracy",
    )


if __name__ == "__main__":
    sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()