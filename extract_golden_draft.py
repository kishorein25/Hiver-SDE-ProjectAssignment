import json, random, sys, io, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
random.seed(42)

with open("data/uber_pairs.json") as f:
    pairs = json.load(f)

# Load existing golden set to preserve hand adjustments if any
os.makedirs("golden_set", exist_ok=True)
existing = {}
if os.path.exists("golden_set/uber_golden_labeled.json"):
    with open("golden_set/uber_golden_labeled.json") as f:
        for item in json.load(f):
            existing[item["message"][:80]] = item

golden_pairs = random.sample(pairs, 160)
out = []
for p in golden_pairs:
    key = p["customer_text"][:80]
    if key in existing:
        out.append(existing[key])
    else:
        out.append({
            "message": p["customer_text"],
            "agent_reply": p["agent_text"],
            "intent": None,
            "escalation": None,
        })

with open("golden_set/uber_golden_draft.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

n_labeled = sum(1 for x in out if x["intent"])
print(f"Total: {len(out)}, already labeled: {n_labeled}")

# Print all unlabeled for hand-labeling
for i, item in enumerate(out):
    if not item["intent"]:
        print(f"[{i}] {item['message'][:250].replace(chr(10), ' ')}")