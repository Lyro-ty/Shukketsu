"""Centralized configuration for Shukketsu.

Reads from environment variables (set in variables.env for Workbench).
"""

import os
from pathlib import Path

# Model serving
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model names
REASONING_MODEL = "llama-3.3-70b-instruct-awq"
ROUTER_MODEL = "qwen3:4b"
EMBEDDING_MODEL = "nomic-embed-text"

# API credentials
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")
WCL_CLIENT_ID = os.getenv("WCL_CLIENT_ID", "")
WCL_CLIENT_SECRET = os.getenv("WCL_CLIENT_SECRET", "")
BLIZZARD_CLIENT_ID = os.getenv("BLIZZARD_CLIENT_ID", "")
BLIZZARD_CLIENT_SECRET = os.getenv("BLIZZARD_CLIENT_SECRET", "")

# Langfuse
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "local-pk")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "local-sk")

# Paths
DB_PATH = Path(os.getenv("SHUKKETSU_DB_PATH", "/project/data/shukketsu.db"))
WIKI_PATH = Path(os.getenv("SHUKKETSU_WIKI_PATH", "/project/knowledge/"))
CACHE_PATH = Path(os.getenv("SHUKKETSU_CACHE_PATH", "/project/data/scratch/cache/"))
BACKUP_PATH = Path(os.getenv("SHUKKETSU_BACKUP_PATH", "/project/data/backups/"))

# Agent defaults
MAX_AGENT_ITERATIONS = 15
MAX_TOTAL_TOKENS = 100_000
DEFAULT_CONTEXT_BUDGET = 50_000
REFLECTION_THRESHOLD = 0.7
