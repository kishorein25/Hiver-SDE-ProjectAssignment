"""
Evaluation harness for the AI Support Agent.

Includes:
- Trivial baseline (majority class)
- Simple baseline (TF-IDF + Logistic Regression)
- Our agent
- LLM-as-judge for reply quality
- Golden set builder
"""

import json
import os
import sys
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data_loader import load_twitter_support
from src.intent_classifier import IntentClassifier
from src.reply_generator import ReplyGenerator
from src.escalation import EscalationRouter
from src.agent import AISupportAgent


# =============================================
# GOLDEN SET BUILDER
# =============================================

def build_golden_set(df: pd.DataFrame, n_samples: int = 200) -> list[dict]:
    """
    Build golden evaluation set by sampling customer messages.

    Labels intent and escalation manually (or via LLM + human review).
    Returns list of dicts with: message, intent, escalation, expected_reply.
    """
    # Sample random customer messages
    customer_msgs = df[df["inbound"] == True]
    samples = customer_msgs.sample(min(n_samples, len(customer_msgs)), random_state=42)

    golden_set = []
    for _, row in samples.iterrows():
        golden_set.append({
            "message": row["text"],
            "intent": None,  # To be labeled
            "escalation": None,  # To be labeled
            "expected_reply": None,  # To be labeled
        })

    return golden_set


def label_golden_set_with_llm(golden_set: list[dict], batch_size: int = 10) -> list[dict]:
    """
    Auto-label golden set using LLM as initial labeler.
    Human should review and correct.
    """
    import requests

    labeled = []
    for i in range(0, len(golden_set), batch_size):
        batch = golden_set[i : i + batch_size]
        print(f"  Labeling batch {i // batch_size + 1}/{(len(golden_set) + batch_size - 1) // batch_size}")

        for item in batch:
            prompt = f"""Label this customer support message. Respond with ONLY valid JSON.

Message: "{item['message'][:200]}"

{{"intent": "<one of: order_status, refund_request, product_complaint, shipping_issue, technical_support, account_help, billing_question, cancellation_request, feedback_positive, general_inquiry>", "escalate": true/false, "escalation_reason": "brief reason"}}"""

            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.2:1b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1},
                    },
                    timeout=60,
                )
                result = response.json()["response"].strip()
                if "{" in result and "}" in result:
                    json_str = result[result.index("{") : result.rindex("}") + 1]
                    parsed = json.loads(json_str)
                    item["intent"] = parsed.get("intent", "general_inquiry")
                    item["escalation"] = parsed.get("escalate", False)
                    item["escalation_reason"] = parsed.get("escalation_reason", "")
            except Exception as e:
                item["intent"] = "general_inquiry"
                item["escalation"] = False

            labeled.append(item)

    return labeled


def save_golden_set(golden_set: list[dict], filepath: str = "golden_set/golden_set.json"):
    """Save golden set to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(golden_set, f, indent=2)
    print(f"  Golden set saved: {filepath} ({len(golden_set)} examples)")


def load_golden_set(filepath: str = "golden_set/golden_set.json") -> list[dict]:
    """Load golden set from disk."""
    with open(filepath, "r") as f:
        return json.load(f)


# =============================================
# BASELINES
# =============================================

class TrivialBaseline:
    """Trivial baseline: always predict majority class."""

    def __init__(self, majority_intent: str = "general_inquiry"):
        self.majority_intent = majority_intent

    def predict(self, message: str) -> dict:
        return {
            "intent": self.majority_intent,
            "confidence": 0.5,
            "method": "trivial",
            "reply": "Thank you for contacting us. We will get back to you.",
            "escalate": False,
        }


class SimpleBaseline:
    """
    Simple baseline: TF-IDF + Logistic Regression.
    Trained on historical data, predicts intent.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.is_trained = False

    def train(self, messages: list[str], intents: list[str]):
        """Train on labeled data."""
        X = self.vectorizer.fit_transform(messages)
        self.classifier.fit(X, intents)
        self.is_trained = True
        print(f"  Trained on {len(messages)} examples, {len(set(intents))} classes")

    def predict(self, message: str) -> dict:
        if not self.is_trained:
            return {"intent": "general_inquiry", "confidence": 0.3, "method": "untrained"}

        X = self.vectorizer.transform([message])
        intent = self.classifier.predict(X)[0]
        probs = self.classifier.predict_proba(X)[0]
        confidence = float(max(probs))

        return {
            "intent": intent,
            "confidence": confidence,
            "method": "tfidf_logreg",
            "reply": "Thank you for contacting us. We will get back to you.",
            "escalate": False,
        }


# =============================================
# LLM-AS-JUDGE
# =============================================

def llm_judge(prompt: str, response: str, criteria: str = "helpfulness, accuracy, tone") -> dict:
    """
    Use LLM to judge reply quality.

    Returns dict with score (1-5) and reasoning.
    """
    import requests

    judge_prompt = f"""You are a customer support quality judge.

Customer message: "{prompt}"
Agent reply: "{response}"

Rate this reply on: {criteria}

Score 1-5 (1=terrible, 5=excellent) and give brief reasoning.

Respond with ONLY valid JSON:
{{"score": <1-5>, "reasoning": "brief explanation"}}"""

    try:
        result = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": judge_prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=60,
        )
        output = result.json()["response"].strip()
        if "{" in output and "}" in output:
            json_str = output[output.index("{") : output.rindex("}") + 1]
            parsed = json.loads(json_str)
            return {
                "score": int(parsed.get("score", 3)),
                "reasoning": parsed.get("reasoning", "No reasoning provided"),
            }
    except Exception as e:
        print(f"  Judge failed: {e}")

    return {"score": 3, "reasoning": "Judge unavailable"}


# =============================================
# EVALUATION
# =============================================

def evaluate_intent_classification(golden_set: list[dict], predictions: list[dict]) -> dict:
    """Evaluate intent classification accuracy."""
    true_intents = [g["intent"] for g in golden_set if g["intent"]]
    pred_intents = [p.get("intent", "general_inquiry") for p in predictions]

    # Align lengths
    min_len = min(len(true_intents), len(pred_intents))
    true_intents = true_intents[:min_len]
    pred_intents = pred_intents[:min_len]

    accuracy = sum(t == p for t, p in zip(true_intents, pred_intents)) / len(true_intents)

    report = classification_report(
        true_intents, pred_intents, output_dict=True, zero_division=0
    )

    return {
        "accuracy": accuracy,
        "report": report,
        "n_samples": min_len,
    }


def evaluate_replies(golden_set: list[dict], predictions: list[dict], sample_size: int = 30) -> dict:
    """Evaluate reply quality using LLM-as-judge."""
    sampled_indices = random.sample(
        range(min(len(golden_set), len(predictions))),
        min(sample_size, min(len(golden_set), len(predictions))),
    )

    scores = []
    for idx in sampled_indices:
        golden = golden_set[idx]
        pred = predictions[idx]
        if golden.get("message") and pred.get("reply"):
            judge_result = llm_judge(golden["message"], pred["reply"])
            scores.append(judge_result)

    avg_score = sum(s["score"] for s in scores) / len(scores) if scores else 0

    return {
        "average_score": avg_score,
        "n_judged": len(scores),
        "scores": scores,
    }


def evaluate_escalation(golden_set: list[dict], predictions: list[dict]) -> dict:
    """Evaluate escalation decisions."""
    true_escalations = [g.get("escalation", False) for g in golden_set]
    pred_escalations = [p.get("escalate", False) for p in predictions]

    min_len = min(len(true_escalations), len(pred_escalations))
    true_escalations = true_escalations[:min_len]
    pred_escalations = pred_escalations[:min_len]

    tp = sum(t and p for t, p in zip(true_escalations, pred_escalations))
    fp = sum(not t and p for t, p in zip(true_escalations, pred_escalations))
    fn = sum(t and not p for t, p in zip(true_escalations, pred_escalations))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def run_full_evaluation(
    brand: str,
    agent_results: list[dict],
    golden_set: list[dict],
) -> dict:
    """Run complete evaluation and save results."""
    print("\n" + "=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)

    # Intent evaluation
    print("\nIntent Classification:")
    intent_eval = evaluate_intent_classification(golden_set, agent_results)
    print(f"  Accuracy: {intent_eval['accuracy']:.3f}")

    # Reply quality
    print("\nReply Quality (LLM-as-Judge):")
    reply_eval = evaluate_replies(golden_set, agent_results)
    print(f"  Average Score: {reply_eval['average_score']:.2f}/5")
    print(f"  Samples Judged: {reply_eval['n_judged']}")

    # Escalation
    print("\nEscalation Routing:")
    escalation_eval = evaluate_escalation(golden_set, agent_results)
    print(f"  Precision: {escalation_eval['precision']:.3f}")
    print(f"  Recall: {escalation_eval['recall']:.3f}")
    print(f"  F1: {escalation_eval['f1']:.3f}")

    results = {
        "brand": brand,
        "intent_classification": intent_eval,
        "reply_quality": reply_eval,
        "escalation": escalation_eval,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to results/evaluation_results.json")

    return results


if __name__ == "__main__":
    print("Evaluation module loaded. Use run_full_evaluation() to evaluate.")
