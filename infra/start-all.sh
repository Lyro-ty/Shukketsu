#!/usr/bin/env bash
# Start all Shukketsu services.
#
# Usage:
#   bash infra/start-all.sh
#
# Prerequisites:
#   - Ollama running (auto-starts with container)
#   - Models pulled: ollama pull llama3.3:70b qwen3:4b nomic-embed-text
#   - Router model created: ollama create qwen3-router -f infra/Modelfile.qwen3-router
#   - Langfuse stack: docker compose -f infra/docker-compose.langfuse.yml up -d

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "=== Starting Shukketsu services ==="

# 1. Ensure Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[1/3] Starting Ollama..."
    ollama serve &
    sleep 3
else
    echo "[1/3] Ollama already running"
fi

# 2. Verify required models
echo "[2/3] Checking models..."
MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null)
for model in "llama3.3:70b" "qwen3-router" "nomic-embed-text"; do
    if echo "$MODELS" | grep -q "$model"; then
        echo "  $model: OK"
    else
        echo "  $model: MISSING — run 'ollama pull $model'"
        exit 1
    fi
done

# 3. Start Langfuse (idempotent)
echo "[3/3] Starting Langfuse stack..."
docker compose -f "${PROJECT_DIR}/infra/docker-compose.langfuse.yml" up -d 2>&1 | tail -3

echo ""
echo "=== Starting Shukketsu web app on :9000 ==="
exec python3 -m uvicorn code.shukketsu.web.app:app --host 0.0.0.0 --port 9000
