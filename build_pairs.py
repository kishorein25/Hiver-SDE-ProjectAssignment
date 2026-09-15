import pandas as pd
import sys
import io
import json
import os
import random

import ui

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
random.seed(42)

CSV_PATH = os.environ.get(
    "TWCS_CSV",
    r"D:\dataset\twcs\twcs.csv",
)

ui.banner("Extracting Uber pairs")
ui.status("info", "Loading only needed columns from CSV...")
df = pd.read_csv(
    CSV_PATH,
    usecols=["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"],
)
ui.kv_pairs(
    {
        "Rows loaded": f"{len(df):,}",
        "Source": CSV_PATH,
    }
)

uber_out = df[df["author_id"] == "Uber_Support"].copy()

cust_ids = set(uber_out["in_response_to_tweet_id"].dropna().astype(int))
cust_msgs = df[(df["tweet_id"].isin(cust_ids)) & (df["inbound"] == True)].copy()

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

ui.subheader("Extraction summary")
ui.kv_pairs(
    {
        "Uber agent tweets": f"{len(uber_out):,}",
        "Matched customer messages (inbound=True)": f"{len(cust_msgs):,}",
        "Total customer-agent pairs": f"{len(pairs):,}",
    }
)

with open("data/uber_pairs.json", "w") as f:
    json.dump(pairs, f)
ui.status("ok", "Saved to data/uber_pairs.json")

samples = random.sample(pairs, min(25, len(pairs)))
ui.subheader("Sample pairs")
for i, p in enumerate(samples, 1):
    cust = p["customer_text"][:150].replace("\n", " ")
    agent = p["agent_text"][:150].replace("\n", " ")
    print()
    ui.status("info", f"Pair #{i}")
    print(f"    CUSTOMER : {cust}")
    print(f"    AGENT    : {agent}")