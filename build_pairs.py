import pandas as pd
import sys
import io
import json
import os
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
random.seed(42)

CSV_PATH = os.environ.get(
    "TWCS_CSV",
    r"D:\dataset\twcs\twcs.csv",
)

print("Loading only needed columns from CSV...")
df = pd.read_csv(
    CSV_PATH,
    usecols=["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"],
)
print(f"Loaded: {len(df)} rows from {CSV_PATH}")

uber_out = df[df["author_id"] == "Uber_Support"].copy()
print(f"Uber agent tweets: {len(uber_out)}")

cust_ids = set(uber_out["in_response_to_tweet_id"].dropna().astype(int))
cust_msgs = df[(df["tweet_id"].isin(cust_ids)) & (df["inbound"] == True)].copy()
print(f"Matched customer messages (inbound=True): {len(cust_msgs)}")

uber_reply_map = {}
for _, row in uber_out.iterrows():
    val = row["in_response_to_tweet_id"]
    if pd.notna(val):
        uber_reply_map[int(val)] = int(row["tweet_id"])

pairs = []
seen = set()
for _, cust in cust_msgs.iterrows():
    cust_id = int(cust["tweet_id"])
    if cust_id not in uber_reply_map or cust_id in seen:
        continue
    seen.add(cust_id)
    reply_id = int(uber_reply_map[cust_id])
    reply_rows = df[df["tweet_id"] == reply_id]
    if not reply_rows.empty:
        reply = reply_rows.iloc[0]["text"]
        if isinstance(reply, str) and isinstance(cust["text"], str):
            pairs.append(
                {
                    "customer_text": cust["text"],
                    "customer_id": cust_id,
                    "agent_text": reply,
                    "agent_id": reply_id,
                }
            )

print(f"Total customer-agent pairs: {len(pairs)}")

with open("data/uber_pairs.json", "w") as f:
    json.dump(pairs, f)
print("Saved to data/uber_pairs.json")

samples = random.sample(pairs, min(25, len(pairs)))
print("\n=== SAMPLE PAIRS ===")
for i, p in enumerate(samples, 1):
    cust = p["customer_text"][:150].replace("\n", " ")
    agent = p["agent_text"][:150].replace("\n", " ")
    print(f"\n{i}. Customer: {cust}")
    print(f"   Agent:    {agent}")