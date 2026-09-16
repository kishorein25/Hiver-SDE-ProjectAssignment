#!/usr/bin/env bash
set -e

# Make sure Ollama is serving every time the (re)started container comes up.
if ! curl -fsS http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
  nohup ollama serve > /tmp/ollama.log 2>&1 &
  sleep 2
fi
echo "Ollama ready."