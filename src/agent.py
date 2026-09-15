"""
Main AI Support Agent - orchestrates intent classification,
reply generation, and escalation routing.
"""

import pandas as pd
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import ui
from src.intent_classifier import IntentClassifier
from src.reply_generator import ReplyGenerator
from src.escalation import EscalationRouter


class AISupportAgent:
    """
    End-to-end AI support agent pipeline.

    Pipeline:
    1. Classify intent
    2. Generate grounded reply (RAG)
    3. Route escalation (auto-handle vs human)
    """

    def __init__(self, method: str = "llm"):
        ui.status("info", "Initializing AI Support Agent...")
        self.classifier = IntentClassifier(method=method)
        self.reply_generator = None
        self.escalation_router = EscalationRouter()
        self.method = method
        ui.status("ok", "Agent ready")

    def load_historical_data(self, df: pd.DataFrame, response_col: str = None):
        """
        Load historical support data for RAG.

        Expects DataFrame with customer messages and brand responses.
        If no response column, uses the data as-is for retrieval context.
        """
        historical = []

        if response_col and response_col in df.columns:
            # We have explicit response column
            customer_msgs = df[df["inbound"] == True]
            for _, row in customer_msgs.iterrows():
                # Find response to this message
                if "response_tweet_id" in df.columns:
                    resp_id = row.get("response_tweet_id")
                    if resp_id and resp_id in df["tweet_id"].values:
                        resp_row = df[df["tweet_id"] == resp_id].iloc[0]
                        historical.append({
                            "text": row["text"],
                            "response": resp_row["text"],
                        })
        else:
            # Use brand responses as historical data
            brand_responses = df[df["inbound"] == False]
            customer_messages = df[df["inbound"] == True]

            for _, resp in brand_responses.head(1000).iterrows():
                # Find the message this is responding to
                if "in_response_to_tweet_id" in resp.index:
                    in_reply_to = resp["in_response_to_tweet_id"]
                    matching = customer_messages[customer_messages["tweet_id"] == in_reply_to]
                    if not matching.empty:
                        historical.append({
                            "text": matching.iloc[0]["text"],
                            "response": resp["text"],
                        })

        if not historical:
            # Fallback: just use brand responses
            brand_responses = df[df["inbound"] == False]
            for _, resp in brand_responses.head(500).iterrows():
                historical.append({
                    "text": resp.get("text", ""),
                    "response": resp["text"],
                })

        ui.status("ok", f"Loaded {len(historical)} historical response pairs")
        self.reply_generator = ReplyGenerator(historical_responses=historical)

    def process(self, customer_message: str) -> dict:
        """
        Process a single customer message through the full pipeline.

        Returns dict with:
        - message: original message
        - intent: classified intent + confidence
        - reply: generated reply + sources
        - escalation: escalate decision + reason
        """
        # Step 1: Classify intent
        intent_result = self.classifier.classify(customer_message)

        # Step 2: Generate reply
        if self.reply_generator:
            reply_result = self.reply_generator.generate(
                customer_message, intent=intent_result["intent"]
            )
        else:
            reply_result = {
                "reply": "Thank you for reaching out. We'll get back to you shortly.",
                "sources": [],
                "method": "no_data",
            }

        # Step 3: Escalation routing
        escalation_result = self.escalation_router.should_escalate(
            message=customer_message,
            intent=intent_result["intent"],
            reply=reply_result["reply"],
        )

        return {
            "message": customer_message,
            "intent": intent_result,
            "reply": reply_result,
            "escalation": escalation_result,
        }

    def process_batch(self, messages: list[str]) -> list[dict]:
        """Process multiple messages."""
        results = []
        for i, msg in enumerate(ui.pbar(messages, desc="Processing", total=len(messages), unit=" msg")):
            result = self.process(msg)
            results.append(result)
        return results


def run_pipeline(brand: str = "Amazon", method: str = "llm"):
    """
    Full pipeline: load data -> build agent -> process sample messages.
    """
    from src.data_loader import load_twitter_support

    ui.banner("HIVER SDE ASSIGNMENT - AI Support Agent Pipeline")

    # Load brand data
    ui.subheader(f"1. Loading data for brand: {brand}")
    df = load_twitter_support(brand=brand, split="training", max_rows=50000)

    if df.empty:
        ui.status("warn", f"No data found for brand '{brand}'. Trying 'Amazon'...")
        df = load_twitter_support(brand="Amazon", split="training", max_rows=50000)

    # Initialize agent
    ui.subheader(f"2. Initializing agent (method={method})")
    agent = AISupportAgent(method=method)

    # Load historical data
    ui.subheader("3. Building historical response index")
    agent.load_historical_data(df)

    # Process sample messages
    ui.subheader("4. Processing sample messages")
    sample_messages = df[df["inbound"] == True]["text"].head(20).tolist()

    results = agent.process_batch(sample_messages)

    # Print results
    ui.banner("SAMPLE RESULTS")
    ui.kv_pairs({"Processed": len(results)})
    for i, r in enumerate(results[:5], 1):
        print(f"\n  --- Message {i} ---")
        ui.kv_pairs(
            {
                "Customer": r["message"][:120],
                "Intent": f"{r['intent']['intent']} (conf: {r['intent']['confidence']:.2f})",
                "Reply": r["reply"]["reply"][:120],
                "Escalate": f"{r['escalation']['escalate']} - {r['escalation']['reason'][:80]}",
            }
        )

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/sample_output.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    ui.status("ok", "Saved results/sample_output.json")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", default="Amazon")
    parser.add_argument("--method", default="llm", choices=["llm", "keyword"])
    args = parser.parse_args()
    run_pipeline(brand=args.brand, method=args.method)
