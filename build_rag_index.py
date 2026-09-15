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

import ui

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "mxbai-embed-large"
N_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def embed_batch(texts, batch_size=8, desc="Embedding"):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "input": batch},
            timeout=180,
        ).json()
        out.extend(resp["embeddings"])
        if (i // batch_size + 1) % 5 == 0 or i + batch_size >= len(texts):
            print(f"  {desc}: {min(i + batch_size, len(texts))}/{len(texts)} embedded")
    return out


def main():
    os.makedirs("data/rag", exist_ok=True)

    ui.banner("Building RAG index")
    with open("data/uber_pairs.json") as f:
        pairs = json.load(f)

    idxs = range(0, len(pairs), max(1, len(pairs) // N_PAIRS))[:N_PAIRS]
    sample = [pairs[i] for i in idxs]

    ui.subheader("Index configuration")
    ui.kv_pairs(
        {
            "Loaded pairs": f"{len(pairs):,}",
            "Sampled for index": len(sample),
            "Embedding model": EMBED_MODEL,
        }
    )

    # Checkpointed embedding
    cust_path = "data/rag/cust_emb.npy"
    idx_path = "data/rag/pairs.index.json"

    if os.path.exists(cust_path):
        cust_emb = np.load(cust_path).tolist()
        with open(idx_path) as f:
            saved = json.load(f)
        ui.status("info", f"Resume: {len(cust_emb)} already embedded")
        if len(saved) == len(sample) and len(cust_emb) == len(sample):
            ui.status("ok", f"Index complete ({len(cust_emb)}/{len(sample)}), skipping")
            return
    else:
        saved = []
        cust_emb = []

    # Continue from checkpoint (sample may differ; for simplicity require same count
    # by embedding only the missing ones positionally)
    start = len(cust_emb)
    remaining = sample[start:]
    if remaining:
        print()
        new_emb = embed_batch(
            [p["customer_text"][:300] for p in remaining],
            desc="Embedding customer messages",
        )
        cust_emb.extend(new_emb)
        np.save(cust_path, np.array(cust_emb))
        with open(idx_path, "w") as f:
            json.dump(sample, f)
        ui.status("ok", f"Checkpoint saved: {len(cust_emb)}/{len(sample)}")

    ui.subheader("Index status")
    ui.status("ok" if len(cust_emb) == len(sample) else "info",
              f"{len(cust_emb)}/{len(sample)} customer messages embedded")


if __name__ == "__main__":
    main()