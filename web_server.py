"""
web_server.py - the CLI agent, at the web page level.

Serves the static dashboard (web/index.html) AND POST /api/chat, which runs the
SAME agent pipeline as interact.py / run_eval.py (centroid intent + RAG reply +
rule escalation), so you can talk to the agent directly in the dashboard.

Run:
    python web_server.py            # http://localhost:8000
    python web_server.py --port 8080
"""

import argparse
import json
import os
import sys
import threading

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

from run_eval import Agent, RAG, OLLAMA_EMBED, keyword_intent  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")

app = Flask(__name__, static_folder=None)

_state = {"agent": None, "status": "starting"}


def build_agent():
    """Lazy-build the real agent: anchors (trained outside the golden set)
    + local embedding RAG index. First build takes ~30-60s (embeddings)."""
    with open(os.path.join(ROOT, "data/uber_pairs.json"), encoding="utf-8") as f:
        pairs = json.load(f)
    rng = np.random.RandomState(7)
    sample_pairs = rng.choice(len(pairs), 4000, replace=False)
    anchors = {}
    for idx in sample_pairs:
        intent = keyword_intent(pairs[idx]["customer_text"])
        anchors.setdefault(intent, []).append(pairs[idx]["customer_text"][:200])
    anchors = {k: v[:25] for k, v in anchors.items()}
    return Agent(anchors, RAG())


def preload_agent():
    try:
        _state["agent"] = build_agent()
        _state["status"] = "ready"
    except Exception as exc:  # Ollama down, missing assets, ...
        _state["status"] = "error: %s" % exc


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory(WEB_DIR, path)


@app.get("/api/status")
def api_status():
    return jsonify({"status": _state["status"], "ready": _state["agent"] is not None})


@app.post("/api/chat")
def api_chat():
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400

    agent = _state["agent"]
    if agent is None:
        return jsonify({
            "error": "Agent is still warming up (first load embeds the RAG index, "
                     "~30-60s). Check /api/status and try again in a moment."
        }), 503

    try:
        return jsonify(agent.process(message))
    except Exception as exc:
        return jsonify({"error": "%s: %s" % (type(exc).__name__, exc)}), 503


def main():
    parser = argparse.ArgumentParser(description="Uber_Support dashboard + live agent API")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (use 0.0.0.0 to let other machines on your "
                             "network / public tunnels reach the dashboard)")
    parser.add_argument("--no-preload", action="store_true",
                        help="build the agent on first chat instead of at startup")
    args = parser.parse_args()

    if not args.no_preload:
        print("Pre-loading agent (embeddings, ~30-60s starting in background)...")
        threading.Thread(target=preload_agent, daemon=True).start()

    print("Dashboard: http://%s:%d  (Ctrl+C to stop)" % (args.host, args.port))
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()