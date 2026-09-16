#!/usr/bin/env bash
set -e

echo "[setup] Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "[setup] Starting Ollama in the background..."
nohup ollama serve > /tmp/ollama.log 2>&1 &
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "[setup] Pulling model: mxbai-embed-large (embeddings)..."
ollama pull mxbai-embed-large

echo "[setup] Pulling model: llama3.2:1b (LLM-as-judge)..."
ollama pull llama3.2:1b

echo ""
echo "[setup] Done. Try it:  python interact.py"