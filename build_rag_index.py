"""
Build RAG index: embed historical customer->response pairs using mxbai-embed-large.
Resumable: stores progress to disk, can be re-run to continue.

Usage: python build_rag_index.py [N_PAIRS]
"""

import json
import sys
import os
import io
import numpy as np
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "mxbai-embed-large"
N_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def embed_batch(texts, batch_size=8):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "input": batch},
            timeout=180,
        ).json()
        out.extend(resp["embeddings"])
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return out


def main():
    os.makedirs("data/rag", exist_ok=True)

    with open("data/uber_pairs.json") as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} pairs")

    idxs = range(0, len(pairs), max(1, len(pairs) // N_PAIRS))[:N_PAIRS]
    sample = [pairs[i] for i in idxs]
    print(f"Sampled {len(sample)} pairs for RAG index")

    # Checkpointed embedding
    cust_path = "data/rag/cust_emb.npy"
    idx_path = "data/rag/pairs.index.json"

    if os.path.exists(cust_path):
        cust_emb = np.load(cust_path).tolist()
        with open(idx_path) as f:
            saved = json.load(f)
        print(f"Resume: {len(cust_emb)} already embedded")
        if len(saved) == len(sample) and len(cust_emb) == len(sample):
            print("Index complete, skipping.")
            return
    else:
        saved = []
        cust_emb = []

    # Continue from checkpoint (sample may differ; for simplicity require same count
    # by embedding only the missing ones positionally)
    start = len(cust_emb)
    remaining = sample[start:]
    if remaining:
        print(f"Embedding {len(remaining)} customer messages...")
        new_emb = embed_batch([p["customer_text"][:300] for p in remaining])
        cust_emb.extend(new_emb)
        np.save(cust_path, np.array(cust_emb))
        with open(idx_path, "w") as f:
            json.dump(sample, f)
        print(f"Checkpoint saved: {len(cust_emb)}/{len(sample)}")

    print(f"\nIndex status: {len(cust_emb)}/{len(sample)}")


if __name__ == "__main__":
    main()