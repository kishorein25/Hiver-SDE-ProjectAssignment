"""
Final thorough evaluation for the report:
1. Contrastive LLM-judge on 30 samples (win/tie/lose vs historical reply)
2. Escalation false positives + why
3. Intent confusion matrix (top confusions)
"""

import sys, io, json, re
import numpy as np
import requests

import ui

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def contrast(cust, a, b):
    prompt = f"""Which reply is BETTER for this customer support message?

Customer: "{cust[:150]}"

Reply A: "{a[:180]}"
Reply B: "{b[:180]}"

Reply ONLY: A, B, or EQUAL."""
    try:
        out = requests.post("http://localhost:11434/api/generate",
            json={"model": "llama3.2:1b", "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0}}, timeout=60).json().get("response", "")
        raw = out.strip()
        up = raw.upper()
        if "EQUAL" in up or up.startswith("E"):
            return "EQUAL"
        m = re.search(r"REPLY\s*([AB])\b", up)
        if m:
            return m.group(1)
        if re.search(r"\bREPLY A\b", up) and not re.search(r"\bREPLY B\b", up):
            return "A"
        if re.search(r"\bREPLY B\b", up) and not re.search(r"\bREPLY A\b", up):
            return "B"
        a_probe = re.sub(r"@\w+", "", a).strip().lower()[:30]
        b_probe = re.sub(r"@\w+", "", b).strip().lower()[:30]
        if a_probe and a_probe in raw.lower():
            return "A"
        if b_probe and b_probe in raw.lower():
            return "B"
        if raw[:2].upper() == "A " or raw.startswith("Reply A"):
            return "A"
        if raw[:2].upper() == "B " or raw.startswith("Reply B"):
            return "B"
        if re.match(r"^[AB]\b", up):
            return up[0]
        print(f"   [unparsed raw]: {raw[:50]!r}")
        return "N/A"
    except Exception:
        return "N/A"


def main():
    golden = json.load(open("golden_set/uber_golden.json", encoding="utf-8"))
    preds = json.load(open("results/our_agent_predictions.json"))
    ui.banner("Failure analysis")

    # ---- Contrastive judge ----
    rng = np.random.RandomState(11)
    idx = rng.choice(len(golden), 30, replace=False)

    ui.subheader("Contrastive LLM judge (agent vs historical Uber reply)")
    wins = ties = losses = 0
    details = []
    for n, i in enumerate(ui.pbar(idx, desc="Judging", total=len(idx), unit=" sample")):
        c = contrast(golden[i]["message"], preds[i]["reply"], golden[i]["agent_reply"])
        if c == "A":
            wins += 1
        elif c == "EQUAL":
            ties += 1
        elif c == "B":
            losses += 1
        details.append({"idx": int(i), "verdict": c})
        print(f"    [{n+1:2d}] idx {i:3d}  verdict={c}")

    total = wins + ties + losses
    ui.table(
        ["Metric", "Count", "Pct"],
        [
            ("Win (agent > historical)", wins, f"{wins/total:.1%}"),
            ("Tie (equal)", ties, f"{ties/total:.1%}"),
            ("Loss (agent < historical)", losses, f"{losses/total:.1%}"),
            ("Win-or-tie rate", wins + ties, f"{(wins+ties)/total:.1%}"),
        ],
    )
    with open("results/judge_contrast_30.json", "w") as f:
        json.dump({"n": total, "wins": wins, "ties": ties, "losses": losses,
                   "win_or_tie": (wins+ties)/total, "details": details}, f, indent=2)

    # ---- Escalation failure analysis ----
    ui.subheader("Escalation false positives (agent said escalate, golden says no)")
    fp = [i for i, (p, g) in enumerate(zip(preds, golden)) if p["escalate"] and not g["escalation"]]
    ui.status("info", f"False positives: {len(fp)}")
    for i in fp[:8]:
        print(f"    [{i}] {golden[i]['message'][:95]}")
        print(f"        reason: {preds[i]['escalation_reason']}")

    fn = [i for i, (p, g) in enumerate(zip(preds, golden)) if not p["escalate"] and g["escalation"]]
    ui.status("info", f"False negatives: {len(fn)}")
    for i in fn[:8]:
        print(f"    [{i}] reason wanted but missed: {golden[i]['escalation_reason'][:80]}")
        print(f"        msg: {golden[i]['message'][:95]}")

    # ---- Intent confusion ----
    ui.subheader("Intent confusions (top)")
    conf = {}
    for p, g in zip(preds, golden):
        if p["intent"] != g["intent"]:
            key = f"{g['intent']} -> {p['intent']}"
            conf[key] = conf.get(key, 0) + 1
    ui.table(
        ["Confusion", "Count"],
        [(k, v) for k, v in sorted(conf.items(), key=lambda x: -x[1])[:12]],
    )


if __name__ == "__main__":
    main()