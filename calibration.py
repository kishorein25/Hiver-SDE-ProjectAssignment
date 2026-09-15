"""
Judge-vs-human calibration.
1. Pick 14 random samples (fixed seed), print them + run contrastive judge.
2. Reviewer assigns human verdicts (A/B/EQUAL) -> stored in results/judge_human_calibration.json.
3. Agreement report.
"""
import sys, io, json, re
import numpy as np
import requests

import ui

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

golden = json.load(open("golden_set/uber_golden.json", encoding="utf-8"))
preds = json.load(open("results/our_agent_predictions.json"))


def contrast(cust, a, b):
    prompt = f"""Which reply is BETTER for this customer support message?

Customer: "{cust[:150]}"

Reply A: "{a[:180]}"
Reply B: "{b[:180]}"

Reply ONLY: A, B, or EQUAL."""
    try:
        raw = requests.post("http://localhost:11434/api/generate",
            json={"model": "llama3.2:1b", "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0}}, timeout=60).json().get("response", "").strip()
        up = raw.upper()
        if "EQUAL" in up or up.startswith("E"):
            return "EQUAL"
        m = re.search(r"REPLY\s*([AB])\b", up)
        if m:
            return m.group(1)
        a_probe = re.sub(r"@\w+", "", a).strip().lower()[:30]
        b_probe = re.sub(r"@\w+", "", b).strip().lower()[:30]
        if a_probe and a_probe in raw.lower():
            return "A"
        if b_probe and b_probe in raw.lower():
            return "B"
        if re.match(r"^[AB]\b", up):
            return up[0]
        return "N/A"
    except Exception:
        return "N/A"


rng = np.random.RandomState(2026)
idx = rng.choice(len(golden), 14, replace=False)

ui.banner("Judge-vs-human calibration (14 samples)")
samples = []
for n, i in enumerate(idx):
    v = contrast(golden[i]["message"], preds[i]["reply"], golden[i]["agent_reply"])
    samples.append({"idx": int(i), "judge": v})
    print(f"\n  --- sample {n+1}/14 | idx {i} | judge={v} ---")
    print(f"  CUSTOMER : {golden[i]['message'][:200]}")
    print(f"  A (AGENT):   {preds[i]['reply'][:220]}")
    print(f"  B (HISTORY): {golden[i]['agent_reply'][:220]}")

with open("results/judge_human_sample.json", "w") as f:
    json.dump({"samples": samples}, f, indent=2)
ui.status("ok", "Saved results/judge_human_sample.json")