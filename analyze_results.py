"""
Compute objective grounding metric + improved LLM-judge scores.
Saves judge scores; human calibration done separately by the reviewer.
"""

import sys, io, json, re
import numpy as np
import requests

import ui

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OLLAMA_EMBED = "http://localhost:11434/api/embed"
EMBED_MODEL = "mxbai-embed-large"


def embed(texts):
    out = []
    for i in range(0, len(texts), 8):
        resp = requests.post(OLLAMA_EMBED,
            json={"model": EMBED_MODEL, "input": texts[i:i+8]}, timeout=180).json()
        out.extend(resp["embeddings"])
    return np.array(out)


def judge_reply(customer, reply, reference):
    prompt = f"""You are an expert customer-support quality judge.
Compare the AGENT reply to how the brand actually replied historically (REFERENCE).
Score 1-5 where:
1 = unhelpful/off-topic/offensive
2 = somewhat relevant but poor
3 = acceptable, on-topic
4 = good, natural, appropriately grounded
5 = excellent, reads like the real brand support

Customer: "{customer[:180]}"
AGENT: "{reply[:250]}"
REFERENCE: "{reference[:250]}"

Tone matching and groundedness matter. Output ONLY a number 1-5."""
    try:
        resp = requests.post("http://localhost:11434/api/generate",
            json={"model": "llama3.2:1b", "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0}}, timeout=60).json()
        out = resp.get("response", "").strip()
        m = re.search(r"[1-5]", out)
        return int(m.group(0)) if m else 3
    except Exception:
        return 3


def main():
    preds = json.load(open("results/our_agent_predictions.json"))
    golden = json.load(open("golden_set/uber_golden.json", encoding="utf-8"))
    ui.banner("Analyzing results")
    ui.kv_pairs({"Predictions": len(preds), "Golden examples": len(golden)})

    ui.subheader("Objective grounding (embedding similarity)")
    a_emb = embed([p["reply"] for p in preds])
    r_emb = embed([g["agent_reply"] for g in golden])
    sims = (a_emb * r_emb).sum(axis=1) / (np.linalg.norm(a_emb, axis=1) * np.linalg.norm(r_emb, axis=1) + 1e-9)
    for p, s in zip(preds, sims):
        p["reply_similarity"] = float(s)
    with open("results/our_agent_predictions.json", "w") as f:
        json.dump(preds, f, indent=2)
    ui.table(
        ["Metric", "Value"],
        [
            ("Mean similarity", f"{sims.mean():.3f}"),
            (">=0.8", f"{(sims>=0.8).mean():.1%}"),
            (">=0.7", f"{(sims>=0.7).mean():.1%}"),
        ],
    )

    rng = np.random.RandomState(11)
    idx = rng.choice(len(golden), 30, replace=False)
    judge_scores = []
    ui.subheader("LLM judge (30 samples)")
    for n, i in enumerate(ui.pbar(idx, desc="Judging", total=len(idx), unit=" sample")):
        s = judge_reply(golden[i]["message"], preds[i]["reply"], golden[i]["agent_reply"])
        judge_scores.append({"idx": int(i), "judge": int(s),
                             "similarity": float(sims[i]),
                             "customer": golden[i]["message"][:100],
                             "agent": preds[i]["reply"][:120],
                             "reference": golden[i]["agent_reply"][:120]})
        print(f"    [{n+1:2d}] idx {i:3d}  score={s}  sim={sims[i]:.2f}")
    avg = np.mean([j["judge"] for j in judge_scores])
    ui.table(
        ["Metric", "Value"],
        [
            ("Judge average", f"{avg:.2f}/5"),
            ("Accept (>=4)", f"{sum(1 for j in judge_scores if j['judge']>=4)/len(judge_scores):.1%}"),
            ("Accept (>=3)", f"{sum(1 for j in judge_scores if j['judge']>=3)/len(judge_scores):.1%}"),
        ],
    )

    # Sanity check: judge the TRUE reference reply against itself
    ui.subheader("Sanity check (AGENT = true reference, should be high)")
    sanity = []
    for n, i in enumerate(ui.pbar(idx[:10], desc="Sanity", total=10, unit=" sample")):
        ref = golden[i]["agent_reply"]
        s = judge_reply(golden[i]["message"], ref, ref)
        sanity.append(s)
        print(f"    [{n+1:2d}] idx {i:3d}  sanity={s}")
    print()
    ui.kv_pairs(
        {
            "Sanity average": f"{np.mean(sanity):.2f}/5",
            "Note": "Low => judge is miscalibrated",
        }
    )

    with open("results/judge_scores_30.json", "w", encoding="utf-8") as f:
        json.dump({"judge_scores": judge_scores, "sanity_avg": float(np.mean(sanity))}, f, indent=2, ensure_ascii=False)
    ui.status("ok", "Saved results/judge_scores_30.json")


if __name__ == "__main__":
    main()