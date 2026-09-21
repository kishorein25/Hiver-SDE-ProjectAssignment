"""
start.py - one-command local setup for ANY user on their OWN machine.

What it does, in order:
  1. Installs missing Python dependencies (from requirements.txt)
  2. Checks Ollama is running (tells you how to start it if not)
  3. Pulls the two models if absent (one-time ~2 GB)
  4. Starts the web dashboard on http://localhost:8000 (local-only)

Usage:
    python start.py               # checks + launch dashboard
    python start.py --interact    # checks + launch terminal chat instead

Works on Windows, macOS and Linux. Wait for "Agent ready" before chatting.
"""

import os
import subprocess
import sys

REQUIRED = ["flask", "numpy", "pandas", "sklearn", "requests", "tqdm"]


def _step(name):
    print("\n=== %s ===" % name)


def ensure_deps():
    _step("Python dependencies")
    import importlib

    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    if missing:
        print("Installing missing packages (%s)..." % ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[OK] Dependencies ready")
    else:
        print("[OK] All dependencies already installed")


def ensure_ollama():
    _step("Ollama")
    import requests

    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
        print("[OK] Ollama is running")
    except Exception:
        print("[FAIL] Ollama is not running.")
        print("       Start it first by opening a new terminal and running:  ollama serve")
        print("       (if not installed: https://ollama.com)")
        sys.exit(1)


def ensure_models():
    import requests

    tags = requests.get("http://localhost:11434/api/tags", timeout=5).json()
    present = {m["name"] for m in tags.get("models", [])}
    for model in ("mxbai-embed-large", "llama3.2:1b"):
        if model in present:
            print("[OK] %s present" % model)
        else:
            print("Pulling %s (one-time, ~1 GB)..." % model)
            subprocess.check_call(["ollama", "pull", model])
            print("[OK] %s pulled" % model)


def main():
    mode = "--interact" in sys.argv
    ensure_deps()
    ensure_ollama()
    ensure_models()

    _step("Launching")
    if mode:
        subprocess.check_call([sys.executable, "interact.py"])
    else:
        cmd = [sys.executable, "web_server.py", "--host", "127.0.0.1", "--port", "8000"]
        print("Dashboard starting at http://localhost:8000 (local-only, Ctrl+C to stop)")
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()