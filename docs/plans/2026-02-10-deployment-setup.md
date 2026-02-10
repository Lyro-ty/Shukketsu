# Shukketsu Phase 1 Deployment Setup

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Get the complete Phase 1 system running end-to-end — chat UI accessible from the local network, backed by vLLM + Ollama model servers, SQLite database, and Langfuse observability.

**Architecture:** Three processes inside the Workbench container (vLLM on :8000, Ollama on :11434, Uvicorn on :9000), Langfuse stack via Docker Compose on host ports (:3000), all fronted by the Workbench Traefik proxy on :10000. Local network access requires re-binding the proxy from 127.0.0.1 to 0.0.0.0.

**Current State (verified):**
- Container: `project-shukettsu` running, Traefik proxy on `127.0.0.1:10000`
- Ollama: binary at `/usr/local/bin/ollama`, running on :11434, **no models pulled**
- vLLM: **not installed** — pip dry-run confirms v0.15.1 ARM64 wheels available (pulls PyTorch 2.9.1)
- sqlite-vec: v0.1.6 installed and **working correctly** on ARM64 (KNN tested)
- Langfuse Python SDK: v3.14.1 installed, compose file exists, **stack not started**
- Database: **no db file** yet (`/project/data/shukketsu.db` doesn't exist)
- Web app: **not registered** in `.project/spec.yaml`, port 9000 not exposed
- Host IP: `192.168.1.142`, DGX Spark GB10 GPU, 128GB unified memory
- Docker: accessible from inside container (docker compose v5.0.1)

---

## Task 1: Install vLLM and Dependencies

**Why:** vLLM serves the Llama 3.3 70B AWQ model for all reasoning/tool-calling. It requires PyTorch + CUDA which are not currently installed.

**Step 1: Install vLLM via pip**

```bash
pip install --break-system-packages vllm
```

This pulls ~2GB of dependencies including PyTorch 2.9.1, numpy, transformers, and CUDA bindings. Takes 5-10 minutes on fast network.

**Step 2: Verify installation**

```bash
python3 -c "import vllm; print(f'vLLM {vllm.__version__}')"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected:
```
vLLM 0.15.1
PyTorch 2.9.1, CUDA: True, Device: NVIDIA GB10
```

**Step 3: Add vllm to requirements.txt**

Add `vllm>=0.15.0` to `requirements.txt` so future Workbench rebuilds include it.

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add vllm to requirements"
```

---

## Task 2: Download Llama 3.3 70B AWQ Model

**Why:** The AWQ INT4 quantized model (~35GB) is needed for all substantive reasoning. The config expects it served as `llama-3.3-70b-instruct-awq` on port 8000.

**Step 1: Download the model via huggingface-hub**

The model ID should match what vLLM can serve. Common choices:
- `hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4`
- `casperhansen/llama-3.3-70b-instruct-awq`

```bash
# Use huggingface-cli (installed with vLLM's transformers dependency)
huggingface-cli download hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4 \
  --local-dir /project/models/llama-3.3-70b-instruct-awq
```

This is ~35GB. Takes 15-30 minutes depending on network speed.

> **Note:** If you don't have a Hugging Face token set up and the model requires one, run `huggingface-cli login` first. Llama 3.3 requires accepting Meta's license on the HF model page.

**Step 2: Verify download**

```bash
ls -la /project/models/llama-3.3-70b-instruct-awq/
# Should contain: config.json, tokenizer.json, *.safetensors files
```

---

## Task 3: Create vLLM Startup Script

**Why:** vLLM needs to be launched with the right flags for AWQ quantization, DGX Spark memory, and OpenAI-compatible API.

**Step 1: Create `infra/start-vllm.sh`**

```bash
#!/usr/bin/env bash
# Start vLLM serving Llama 3.3 70B AWQ on port 8000.
# The DGX Spark GB10 has ~128GB unified memory (CPU+GPU shared).

set -euo pipefail

MODEL_PATH="/project/models/llama-3.3-70b-instruct-awq"
SERVED_NAME="llama-3.3-70b-instruct-awq"
PORT=8000

echo "Starting vLLM server..."
echo "  Model: ${MODEL_PATH}"
echo "  Served as: ${SERVED_NAME}"
echo "  Port: ${PORT}"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --quantization awq \
  --dtype float16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code
```

```bash
chmod +x infra/start-vllm.sh
```

> **Tuning notes:**
> - `--max-model-len 8192` limits context to save memory. Increase if memory allows.
> - `--gpu-memory-utilization 0.85` leaves headroom for Ollama models. Adjust if OOM.
> - If vLLM detects the GB10 as unsupported, add `--enforce-eager` to disable CUDA graphs.

**Step 2: Test vLLM starts**

```bash
bash infra/start-vllm.sh &
# Wait for "Uvicorn running on http://0.0.0.0:8000"
# Then test:
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

Expected: JSON listing `llama-3.3-70b-instruct-awq` as available model.

**Step 3: Test a completion**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b-instruct-awq",
    "messages": [{"role": "user", "content": "Say hello in one word"}],
    "max_tokens": 10
  }' | python3 -m json.tool
```

Expected: A valid chat completion response.

**Step 4: Commit**

```bash
git add infra/start-vllm.sh
git commit -m "infra: add vLLM startup script for Llama 3.3 70B AWQ"
```

---

## Task 4: Pull Ollama Models

**Why:** Ollama serves Qwen3 4B (query routing) and nomic-embed-text (embeddings). Ollama is already running on :11434 but has no models.

**Step 1: Pull the router model**

```bash
ollama pull qwen3:4b
```

~2.5GB download. Takes a few minutes.

**Step 2: Pull the embedding model**

```bash
ollama pull nomic-embed-text
```

~275MB download. Quick.

**Step 3: Verify models**

```bash
ollama list
```

Expected:
```
NAME                  ID           SIZE     MODIFIED
nomic-embed-text      ...          274 MB   ...
qwen3:4b              ...          2.6 GB   ...
```

**Step 4: Test router model**

```bash
curl -s http://localhost:11434/api/chat \
  -d '{"model": "qwen3:4b", "messages": [{"role": "user", "content": "Say hello"}], "stream": false}' \
  | python3 -m json.tool
```

**Step 5: Test embedding model**

```bash
curl -s http://localhost:11434/api/embed \
  -d '{"model": "nomic-embed-text", "input": "test embedding"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Dimensions: {len(d[\"embeddings\"][0])}')"
```

Expected: `Dimensions: 768`

---

## Task 5: Start Langfuse Observability Stack

**Why:** Langfuse provides trace visualization for all LLM calls, agent loops, and tool invocations.

**Step 1: Start the Langfuse stack**

```bash
docker compose -f /project/infra/docker-compose.langfuse.yml up -d
```

This starts 6 containers: postgres, clickhouse, redis, minio, langfuse-worker, langfuse-web. Takes 1-2 minutes for all health checks to pass.

**Step 2: Verify all containers are healthy**

```bash
docker compose -f /project/infra/docker-compose.langfuse.yml ps
```

Expected: All services `Up (healthy)`.

**Step 3: Wait for Langfuse web to be ready**

```bash
# May take 30-60 seconds after containers start
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000
```

Expected: `200` (or `302` redirect to login).

---

## Task 6: Configure Langfuse API Keys

**Why:** The app needs valid Langfuse API keys. The current `variables.env` has placeholders (`local-pk`, `local-sk`).

**Step 1: Create Langfuse account**

1. Open `http://localhost:3000` in a browser (or via Workbench proxy)
2. Click "Sign Up" — create an account (any email/password, it's local)
3. Create a new project called "Shukketsu"

**Step 2: Generate API keys**

1. In the Langfuse UI: Settings > API Keys > Create API Key
2. Copy the **Public Key** and **Secret Key**

**Step 3: Update `variables.env`**

Replace the placeholder values:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx   # <- paste your public key
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx   # <- paste your secret key
```

**Step 4: Restart the Workbench container**

Environment variables from `variables.env` are only loaded on container start. You need to restart the project in the NVIDIA AI Workbench desktop app (or CLI):

```
# From the Workbench CLI (if available), or use the desktop UI:
# Project > Stop, then Project > Start
```

> **Alternative (no restart):** Export the keys in your current shell session:
> ```bash
> export LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
> export LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx
> ```
> This only persists until the shell exits.

---

## Task 7: Initialize the Database

**Why:** The SQLite database doesn't exist yet. The app creates it on first connection via `init_db()`, but we should verify it works before starting the web app.

**Step 1: Ensure the data directory exists**

```bash
mkdir -p /project/data/scratch/cache
```

**Step 2: Initialize the database**

```bash
python3 -c "
from code.shukketsu.db.connection import get_connection, init_db
conn = get_connection()
init_db(conn)
conn.close()
print('Database initialized successfully')
"
```

**Step 3: Verify schema**

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('/project/data/shukketsu.db')
tables = conn.execute(\"SELECT name FROM sqlite_main.sqlite_master WHERE type='table' ORDER BY name\").fetchall()
print('Tables:', [t[0] for t in tables])
conn.close()
"
```

Expected tables: `articles`, `chunks`, `chunks_fts`, `chunks_fts_config`, `chunks_fts_content`, `chunks_fts_data`, `chunks_fts_docsize`, `chunks_fts_idx`, `chunks_vec`, `schema_version`, `sources`

---

## Task 8: Register Web App in Workbench

**Why:** The web app needs to be registered in `.project/spec.yaml` so the Workbench proxy routes traffic to it, and port 9000 is exposed from the container.

**Step 1: Add the app definition to `spec.yaml`**

Add a new entry under `environment.base.apps` (after the tensorboard entry):

```yaml
        - name: shukketsu
          type: custom
          class: webapp
          start_command: python3 -m uvicorn code.shukketsu.web.app:app --host 0.0.0.0 --port 9000
          health_check_command: '[ \$(curl -o /dev/null -s -w ''%{http_code}'' http://localhost:9000/health) == ''200'' ]'
          stop_command: pkill -f "uvicorn.*shukketsu"
          user_msg: ""
          logfile_path: ""
          timeout_seconds: 60
          icon_url: ""
          webapp_options:
            autolaunch: true
            port: "9000"
            proxy:
              trim_prefix: false
            url: http://localhost:9000
```

**Step 2: Rebuild/restart the Workbench project**

The spec.yaml changes require a project rebuild for the port to be exposed and the proxy route to be configured. In the Workbench desktop app:

1. Stop the project
2. Rebuild (this re-reads spec.yaml for port/app changes)
3. Start the project

After rebuild, verify port 9000 is exposed:

```bash
docker inspect project-shukettsu --format '{{json .Config.ExposedPorts}}' | python3 -m json.tool
```

Expected: should include `"9000/tcp": {}`.

**Step 3: Commit**

```bash
git add .project/spec.yaml
git commit -m "infra: register Shukketsu web app in Workbench spec"
```

---

## Task 9: Enable Local Network Access

**Why:** By default, the Workbench proxy (Traefik) binds to `127.0.0.1:10000`, making it accessible only from the DGX Spark itself. To access from other machines on the LAN (`192.168.1.x`), the binding must change to `0.0.0.0:10000`.

There are two approaches. Choose one:

### Option A: Modify Workbench Proxy Binding (preferred)

The Traefik proxy binds to `127.0.0.1:10000`. This is controlled by the Workbench runtime, not by a user-editable config file. Check if there's a Workbench setting:

1. In the Workbench desktop app: Settings > look for "Remote Access" or "Network Binding"
2. Or check the Workbench CLI: `nvwb config` or similar

If the Workbench supports remote access configuration, enable it. The app would then be accessible at:
```
http://192.168.1.142:10000/projects/Shukettsu/applications/shukketsu/
```

### Option B: Direct Port Forwarding (fallback)

If you can't change the Workbench proxy binding, run the web app directly on a port bound to all interfaces. This bypasses the Workbench proxy:

**Step 1: Start uvicorn manually on all interfaces**

```bash
python3 -m uvicorn code.shukketsu.web.app:app --host 0.0.0.0 --port 9000 &
```

**Step 2: Verify from another machine**

From another device on your LAN:
```
http://192.168.1.142:9000/
```

> **Note:** This only works if the Docker container publishes port 9000 to the host. If port 9000 isn't in the container's port bindings (which is why Task 8 adds it to spec.yaml), you may need to use `docker run` flags or `socat` as a port forwarder on the host.

### Option C: SSH Tunnel (quick hack)

From your laptop/desktop:
```bash
ssh -L 9000:localhost:9000 lyro@192.168.1.142
```

Then open `http://localhost:9000` on your laptop. No container port changes needed.

---

## Task 10: Start Everything and Verify End-to-End

**Why:** All services need to be running simultaneously for the full system to work.

**Step 1: Pre-flight checklist**

Verify all services are running:

```bash
echo "=== Ollama ==="
curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; models=json.load(sys.stdin)['models']; print([m['name'] for m in models])"

echo "=== vLLM ==="
curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; models=json.load(sys.stdin)['data']; print([m['id'] for m in models])"

echo "=== Langfuse ==="
curl -s -o /dev/null -w "HTTP %{http_code}" http://localhost:3000

echo ""
echo "=== Database ==="
python3 -c "
from code.shukketsu.db.connection import get_connection
conn = get_connection()
print(f'DB OK: {conn.execute(\"SELECT COUNT(*) FROM schema_version\").fetchone()[0]} schema versions')
conn.close()
"
```

Expected:
```
=== Ollama ===
['nomic-embed-text:latest', 'qwen3:4b']
=== vLLM ===
['llama-3.3-70b-instruct-awq']
=== Langfuse ===
HTTP 200
=== Database ===
DB OK: 1 schema versions
```

**Step 2: Start the web app**

```bash
python3 -m uvicorn code.shukketsu.web.app:app --host 0.0.0.0 --port 9000
```

Or if registered in Workbench, start it via the Workbench UI.

**Step 3: Test the health endpoint**

```bash
curl -s http://localhost:9000/health
```

Expected: `{"status":"ok"}` (or similar).

**Step 4: Test the chat page loads**

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:9000/chat
```

Expected: `200`.

**Step 5: Test WebSocket chat end-to-end**

```bash
# Quick test with websocat if available, otherwise use the browser
python3 -c "
import asyncio, json, websockets

async def test():
    async with websockets.connect('ws://localhost:9000/ws/chat') as ws:
        await ws.send(json.dumps({'content': 'What is a rogue in WoW TBC?'}))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get('type') == 'stream':
                print(msg.get('content', ''), end='', flush=True)
            elif msg.get('type') == 'done':
                print('\n--- DONE ---')
                break
            elif msg.get('type') == 'error':
                print(f'ERROR: {msg}')
                break

asyncio.run(test())
" 2>/dev/null || echo "Install websockets: pip install --break-system-packages websockets"
```

**Step 6: Verify Langfuse traces**

1. Open Langfuse UI at `http://localhost:3000`
2. Navigate to Traces
3. You should see a trace for the chat message you just sent, showing:
   - Root span: chat handler
   - Child spans: router classification, agent loop (if complex), tool calls

---

## Task 11: Create Unified Startup Script

**Why:** After verifying everything works individually, create a single script to start all services.

**Step 1: Create `infra/start-all.sh`**

```bash
#!/usr/bin/env bash
# Start all Shukketsu services.
# Usage: bash infra/start-all.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "=== Starting Shukketsu services ==="

# 1. Ensure Ollama is running (should auto-start, but just in case)
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[1/4] Starting Ollama..."
    ollama serve &
    sleep 3
else
    echo "[1/4] Ollama already running"
fi

# 2. Start Langfuse (idempotent — docker compose up skips already-running containers)
echo "[2/4] Starting Langfuse stack..."
docker compose -f "${PROJECT_DIR}/infra/docker-compose.langfuse.yml" up -d

# 3. Start vLLM (in background)
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "[3/4] vLLM already running"
else
    echo "[3/4] Starting vLLM (this takes 1-2 minutes to load the model)..."
    bash "${SCRIPT_DIR}/start-vllm.sh" &
    VLLM_PID=$!
    echo "  vLLM PID: ${VLLM_PID}"
fi

# 4. Wait for vLLM to be ready, then start web app
echo "[4/4] Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "  vLLM ready after ${i}s"
        break
    fi
    sleep 1
done

echo ""
echo "=== Starting Shukketsu web app ==="
exec python3 -m uvicorn code.shukketsu.web.app:app --host 0.0.0.0 --port 9000
```

```bash
chmod +x infra/start-all.sh
```

**Step 2: Commit**

```bash
git add infra/start-all.sh infra/start-vllm.sh
git commit -m "infra: add startup scripts for all services"
```

---

## Troubleshooting

### vLLM won't start on GB10
- Add `--enforce-eager` to disable CUDA graphs (some new GPU architectures need this)
- Reduce `--gpu-memory-utilization` to `0.70` if OOM
- Reduce `--max-model-len` to `4096` to save memory
- Check `nvidia-smi` for memory usage

### Ollama OOM alongside vLLM
- The GB10 shares 128GB between CPU and GPU. If vLLM takes 85%, Ollama may not fit
- Reduce vLLM's `--gpu-memory-utilization` to `0.60-0.70`
- Ollama's smaller models (4B + embed) need ~3-4GB total

### WebSocket connection fails
- Check that uvicorn is running with `--host 0.0.0.0` (not `127.0.0.1`)
- If behind the Workbench proxy, WebSocket upgrade headers must be preserved (Traefik does this by default)

### Langfuse traces not appearing
- Verify API keys match between `variables.env` and Langfuse UI
- Check `LANGFUSE_TRACING_ENABLED=true` in env
- Langfuse flushes traces asynchronously — may take a few seconds to appear

### Can't access from LAN
- Verify firewall isn't blocking: `sudo ufw status` (if available)
- Verify the DGX Spark's IP: `hostname -I` → should show `192.168.1.142`
- If using the Workbench proxy, it must bind to `0.0.0.0`, not `127.0.0.1`

### sqlite-vec failures
- The v0.1.6 ARM64 .so is currently **working** (verified with KNN test)
- If it breaks after a pip upgrade, recompile from source per MEMORY.md notes
