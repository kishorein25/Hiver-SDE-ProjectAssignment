"""
Escalation router: decides auto-handle vs human escalation.

Uses rules + LLM reasoning.
"""

import json
import requests
from typing import Optional


OLLAMA_URL = "http://localhost:11434/api/generate"


# Rule-based escalation signals
ESCALATION_KEYWORDS = [
    "lawyer", "legal", "sue", "threat", "attorney",
    "unacceptable", "worst", "terrible", "horrible",
    "police", "report", "scam", "discrimination",
    "racist", "harass", "safety", "unsafe", "high",
    "drunk", "threatened", "charged me", "stolen",
]

HIGH_VALUE_KEYWORDS = [
    "cancel my account", "delete my data", "gdpr", "privacy",
    "class action", "media", "press", "medical files",
]


class EscalationRouter:
    """Decides whether to auto-handle or escalate to human."""

    def __init__(self):
        pass

    def should_escalate(self, message: str, intent: str, reply: str) -> dict:
        """
        Decide escalation. Returns dict with:
        - escalate: bool
        - reason: str
        - confidence: float
        """
        # Rule-based checks first
        msg_lower = message.lower()

        # High priority: threats, legal
        for kw in ESCALATION_KEYWORDS:
            if kw in msg_lower:
                return {
                    "escalate": True,
                    "reason": f"Contains escalation keyword: '{kw}' - potential legal/safety issue",
                    "confidence": 0.9,
                    "method": "rule",
                }

        # High value: account deletion, data privacy
        for kw in HIGH_VALUE_KEYWORDS:
            if kw in msg_lower:
                return {
                    "escalate": True,
                    "reason": f"High-value request: '{kw}' - requires human approval",
                    "confidence": 0.85,
                    "method": "rule",
                }

        # Intents that typically need human
        human_intents = ["safety_concern", "driver_behavior", "lost_item"]
        if intent in human_intents:
            return {
                "escalate": True,
                "reason": f"Intent '{intent}' typically requires human intervention",
                "confidence": 0.7,
                "method": "intent_rule",
            }

        # Use LLM for nuanced cases
        return self._llm_escalation_check(message, intent, reply)

    def _llm_escalation_check(self, message: str, intent: str, reply: str) -> dict:
        """Use LLM to decide escalation for ambiguous cases."""
        prompt = f"""You are an escalation router for a customer support system.

Customer message: "{message}"
Detected intent: {intent}
Draft reply: "{reply[:200]}"

Decide: Should this be escalated to a human agent?
Consider:
- Customer frustration level (anger, profanity, multiple messages)
- Issue complexity (multi-step, account-specific)
- Financial impact (large amounts, repeated billing)
- Legal/regulatory risk
- If the AI reply would be insufficient

Respond with ONLY valid JSON:
{{"escalate": true/false, "reason": "brief reason", "confidence": 0.0-1.0}}"""

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

            if "{" in result and "}" in result:
                json_str = result[result.index("{") : result.rindex("}") + 1]
                parsed = json.loads(json_str)
                return {
                    "escalate": bool(parsed.get("escalate", False)),
                    "reason": parsed.get("reason", "LLM decision"),
                    "confidence": float(parsed.get("confidence", 0.5)),
                    "method": "llm",
                }
        except Exception as e:
            print(f"  LLM escalation check failed: {e}")

        # Default: auto-handle
        return {
            "escalate": False,
            "reason": "Standard support request, auto-handled",
            "confidence": 0.5,
            "method": "fallback",
        }
