import json, sys, io
import numpy as np

import ui

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from run_eval import Agent, RAG, SimpleBaseline, TrivialBaseline, keyword_intent, embed_texts

# Build small anchors
with open("data/uber_pairs.json") as f:
    pairs = json.load(f)
rng = np.random.RandomState(7)
sample_pairs = rng.choice(len(pairs), 800, replace=False)
anchors = {}
for idx in sample_pairs:
    intent = keyword_intent(pairs[idx]["customer_text"])
    anchors.setdefault(intent, []).append(pairs[idx]["customer_text"][:150])
anchors = {k: v[:10] for k, v in anchors.items()}
ui.subheader("Test agent setup")
ui.table(["Intent", "Anchors"], [(k, len(v)) for k, v in anchors.items()])

rag = RAG()
agent = Agent(anchors, rag)

test_msgs = [
    "I got charged three times for my ride last night, can someone look into it?",
    "@Uber_Support my driver cancelled on me and I was charged $5.00",
    "I lost my wallet in the uber, how do I get it back?",
    "Hi I can't log into my account, reset password not working",
    "@Uber_Support my food order was wrong, 3 items missing",
]

ui.subheader("Agent output on test messages")
for n, m in enumerate(test_msgs, 1):
    p = agent.process(m)
    print(f"\n  --- message {n} ---")
    ui.kv_pairs(
        {
            "Message": m[:80],
            "Intent": p["intent"],
            "Escalate": p["escalate"],
            "Reply": p["reply"][:150],
            "Retrieval sim": f"{p['reply_source_sim']:.3f}",
        }
    )