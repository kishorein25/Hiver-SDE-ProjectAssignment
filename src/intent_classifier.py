"""
Intent classifier for the AI support agent.

Approaches:
1. Trivial baseline: majority class / keyword matching
2. Simple baseline: TF-IDF + Logistic Regression
3. Main: LLM zero-shot classification via Ollama
"""

import json
import requests
from typing import Optional


OLLAMA_URL = "http://localhost:11434/api/generate"


# Uber-specific intents derived from data analysis
UBER_INTENTS = [
    "trip_fare_billing",
    "driver_behavior",
    "trip_cancellation",
    "app_technical",
    "account_access",
    "uber_eats_order",
    "lost_item",
    "safety_concern",
    "general_inquiry",
    "service_feedback",
]

# Predefined intents based on common support patterns
DEFAULT_INTENTS = UBER_INTENTS


def classify_intent_llm(message: str, intents: list[str] = None) -> dict:
    """
    Classify intent using Ollama LLM (zero-shot).

    Returns dict with 'intent' and 'confidence' keys.
    """
    if intents is None:
        intents = DEFAULT_INTENTS

    intent_list = "\n".join(f"- {i}" for i in intents)

    prompt = f"""Classify this customer support message into exactly one intent.

Available intents:
{intent_list}

Customer message: "{message}"

Respond with ONLY valid JSON: {{"intent": "<intent_name>", "confidence": <0.0-1.0>}}
No other text."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=60,
        )
        result = response.json()["response"].strip()

        # Parse JSON from response
        if "{" in result and "}" in result:
            json_str = result[result.index("{") : result.rindex("}") + 1]
            parsed = json.loads(json_str)
            return {
                "intent": parsed.get("intent", "general_inquiry"),
                "confidence": float(parsed.get("confidence", 0.5)),
                "method": "llm",
            }
    except Exception as e:
        print(f"  LLM classification failed: {e}")

    return {"intent": "general_inquiry", "confidence": 0.3, "method": "fallback"}


class IntentClassifier:
    """Wraps multiple classification approaches."""

    def __init__(self, method: str = "llm"):
        """
        Args:
            method: 'llm', 'trivial', 'keyword', or 'tfidf'
        """
        self.method = method
        self.keyword_rules = {
            "trip_fare_billing": ["charge", "charged", "fare", "refund", "money back", "price", "cost", "bill", "payment", "card"],
            "driver_behavior": ["driver", "driver did", "driver was", "driver is", "rude", "high", "drunk", "song", "music", "behavior"],
            "trip_cancellation": ["cancel", "cancelled", "canceled", "no show", "canceled the trip"],
            "app_technical": ["app", "website", "not working", "bug", "error", "crash", "gps", "location", "restart", "online"],
            "account_access": ["account", "login", "log in", "sign in", "password", "email", "hacked", "verify"],
            "uber_eats_order": ["eats", "food", "restaurant", "order", "delivery", "mcdonalds", "meal", "driver delivered"],
            "lost_item": ["left", "lost", "forgot", "phone", "wallet", "medical files", "bag", "item"],
            "safety_concern": ["safety", "unsafe", "scared", "police", "threatened", "harass", "attack", "fear"],
            "general_inquiry": ["how", "what", "when", "where", "why", "question", "gift card", "curious", "info", "request a ride"],
            "service_feedback": ["terrible", "sucks", "worst", "great", "best", "awesome", "happy", "angry", "service", "unacceptable", "disappointed"],
        }

    def classify(self, message: str, intents: list[str] = None) -> dict:
        if self.method == "llm":
            return classify_intent_llm(message, intents)
        elif self.method == "keyword":
            return self._classify_keyword(message)
        else:
            return self._classify_keyword(message)

    def _classify_keyword(self, message: str) -> dict:
        message_lower = message.lower()
        scores = {}
        for intent, keywords in self.keyword_rules.items():
            score = sum(1 for kw in keywords if kw in message_lower)
            scores[intent] = score

        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return {"intent": "general_inquiry", "confidence": 0.3, "method": "keyword"}
        return {"intent": best, "confidence": min(scores[best] / 3.0, 1.0), "method": "keyword"}
