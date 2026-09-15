"""
Reply generator using RAG approach.

1. Embed incoming message using Ollama embedding model (mxbai-embed-large)
2. Retrieve top-k similar historical support responses
3. Generate grounded reply using Ollama LLM (llama3.2)
"""

import requests
import numpy as np
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import ui


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"
GENERATE_MODEL = "llama3.2:1b"


def get_embedding(text: str) -> list[float]:
    """Get embedding vector from Ollama."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    result = response.json()
    return result["embeddings"][0]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    result = response.json()
    return result["embeddings"]


class ReplyGenerator:
    """RAG-based reply generator."""

    def __init__(self, historical_responses: list[dict] = None):
        """
        Args:
            historical_responses: List of dicts with 'text', 'response' keys
        """
        self.historical_responses = historical_responses or []
        self.response_embeddings = None
        self._build_index()

    def _build_index(self):
        """Build embedding index for historical responses."""
        if not self.historical_responses:
            return

        texts = [r["response"] for r in self.historical_responses]
        ui.status("info", f"Embedding {len(texts)} historical responses...")
        self.response_embeddings = np.array(get_embeddings_batch(texts))
        ui.status("ok", f"Index built: {self.response_embeddings.shape}")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve top-k most similar historical responses."""
        if not self.historical_responses or self.response_embeddings is None:
            return []

        query_emb = np.array(get_embedding(query))

        # Cosine similarity
        similarities = np.dot(self.response_embeddings, query_emb) / (
            np.linalg.norm(self.response_embeddings, axis=1)
            * np.linalg.norm(query_emb)
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            results.append(
                {
                    **self.historical_responses[idx],
                    "similarity": float(similarities[idx]),
                }
            )
        return results

    def generate(self, customer_message: str, intent: str = None) -> dict:
        """
        Generate a reply grounded in historical data.

        Returns dict with 'reply', 'sources', 'reasoning'.
        """
        # Step 1: Retrieve similar examples
        similar = self.retrieve(customer_message, top_k=3)

        # Step 2: Build context from retrieved examples
        context_parts = []
        for i, s in enumerate(similar, 1):
            context_parts.append(
                f"Example {i} (similarity: {s['similarity']:.2f}):\n"
                f"  Customer: {s['text'][:150]}\n"
                f"  Brand reply: {s['response'][:150]}"
            )

        context = "\n\n".join(context_parts) if context_parts else "No similar historical examples found."

        prompt = f"""You are a customer support agent. Draft a reply to this customer message.

Ground your response in how the brand has historically replied to similar issues.

Historical examples:
{context}

Customer message: "{customer_message}"
{f'Identified intent: {intent}' if intent else ''}

Rules:
- Be helpful, professional, and empathetic
- Reference the brand's typical response pattern
- Keep reply under 280 characters (Twitter)
- If you don't have enough info, say so honestly

Reply:"""

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": GENERATE_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.4},
                },
                timeout=90,
            )
            reply_text = response.json()["response"].strip()

            return {
                "reply": reply_text,
                "sources": [
                    {"text": s["text"][:100], "response": s["response"][:100], "similarity": s["similarity"]}
                    for s in similar
                ],
                "method": "rag",
            }
        except Exception as e:
            print(f"  LLM generation failed: {e}")
            if similar:
                return {
                    "reply": similar[0]["response"],
                    "sources": [{"text": s["text"][:100]} for s in similar[:1]],
                    "method": "retrieval_only",
                }
            return {"reply": "Thank you for reaching out. We'll look into this and get back to you shortly.", "sources": [], "method": "fallback"}
