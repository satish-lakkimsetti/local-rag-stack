#!/bin/bash
set -e

echo "[ollama] Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

echo "[ollama] Waiting for server to accept connections..."
until ollama list > /dev/null 2>&1; do
    sleep 2
done
echo "[ollama] Server ready."

LLM="${LLM_MODEL:-granite4.1:3b}"
EMBED="${EMBED_MODEL:-nomic-embed-text}"

echo "[ollama] Pulling LLM model: $LLM"
ollama pull "$LLM"

echo "[ollama] Pulling embedding model: $EMBED"
ollama pull "$EMBED"

echo "[ollama] All models ready."
touch /tmp/ollama_models_ready

wait $OLLAMA_PID
