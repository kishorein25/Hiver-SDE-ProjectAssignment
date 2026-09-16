FROM mcr.microsoft.com/devcontainers/python:3.10

# Install Ollama (local LLM runtime used by the agent: embeddings + LLM-as-judge)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*