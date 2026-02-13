"""Centralized configuration for Shukketsu.

Reads from environment variables (set in variables.env for Workbench).
"""

import os
from pathlib import Path

# Model serving — all models via Ollama (llama.cpp has native Blackwell/GB10 support)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model names (all served by Ollama)
REASONING_MODEL = os.getenv("REASONING_MODEL", "llama3.3:70b")
ROUTER_MODEL = "qwen3-router"  # Custom Modelfile: qwen3:4b with thinking disabled
FAST_MODEL = os.getenv("FAST_MODEL", "qwen2.5:7b")  # 7B model for moderate queries
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
LANGFUSE_TRACING_ENABLED = os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "true"
LANGFUSE_SAMPLE_RATE = float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0"))

# Paths
DB_PATH = Path(os.getenv("SHUKKETSU_DB_PATH", "/project/data/shukketsu.db"))
WIKI_PATH = Path(os.getenv("SHUKKETSU_WIKI_PATH", "/project/knowledge/"))
CACHE_PATH = Path(os.getenv("SHUKKETSU_CACHE_PATH", "/project/data/scratch/cache/"))
BACKUP_PATH = Path(os.getenv("SHUKKETSU_BACKUP_PATH", "/project/data/backups/"))

# Agent defaults
MAX_TOTAL_TOKENS = 100_000
DEFAULT_CONTEXT_BUDGET = 50_000
REFLECTION_THRESHOLD = 0.7

# Phase 1 agent limits
AGENT_MAX_ITERATIONS = 5
RAG_SEARCH_TOP_K = 5
RAG_SEARCH_FETCH_K = 20  # Candidates per source before RRF fusion
AGENT_GRACEFUL_FAILURE = "I wasn't able to find a complete answer. Please try rephrasing your question."
COMPACTION_THRESHOLD_TOKENS = int(os.getenv("COMPACTION_THRESHOLD_TOKENS", "16000"))
OBSERVATION_MAX_CHARS = int(os.getenv("OBSERVATION_MAX_CHARS", "2000"))

# Chat defaults
SYSTEM_PROMPT = (
    "You are Shukketsu, a research assistant specializing in "
    "World of Warcraft: The Burning Crusade Rogue class. "
    "Answer questions accurately and concisely."
)
CHAT_MAX_HISTORY_PAIRS = 20
CHAT_MAX_MESSAGE_LENGTH = 10_000  # Max characters per user message
CHAT_TEMPERATURE = 0.7
CHAT_MAX_TOKENS = 2048
LLM_TIMEOUT_SECONDS = 300.0  # 70B structured output can take 2+ min on GB10

# Structured output defaults
STRUCTURED_TEMPERATURE = 0.1
STRUCTURED_MAX_TOKENS = 4096
STRUCTURED_MAX_RETRIES = 3

# Chunking
CHUNK_MAX_TOKENS = 400
CHUNK_MIN_TOKENS = 50
CHUNK_OVERLAP_TOKENS = 50

# Embedding
EMBEDDING_DIMENSIONS = 768
EMBEDDING_BATCH_SIZE = 64

# Scraping
BRAVE_SEARCH_MAX_RESULTS = int(os.getenv("BRAVE_SEARCH_MAX_RESULTS", "5"))
SCRAPING_USER_AGENT = "Shukketsu/0.1 (research bot)"
SCRAPING_DEFAULT_TIMEOUT = 15.0
SCRAPING_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB
ROBOTS_CACHE_TTL_HOURS = 24

# Circuit breaker defaults
CB_REASONING_FAILURE_THRESHOLD = 3
CB_REASONING_RECOVERY_TIMEOUT = 30.0
CB_OLLAMA_ROUTER_FAILURE_THRESHOLD = 5
CB_OLLAMA_ROUTER_RECOVERY_TIMEOUT = 60.0
CB_OLLAMA_EMBED_FAILURE_THRESHOLD = 5
CB_OLLAMA_EMBED_RECOVERY_TIMEOUT = 60.0
CB_BRAVE_FAILURE_THRESHOLD = 5
CB_BRAVE_RECOVERY_TIMEOUT = 120.0
# Cross-encoder reranker
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CB_RERANKER_FAILURE_THRESHOLD = int(os.getenv("CB_RERANKER_FAILURE_THRESHOLD", "3"))
CB_RERANKER_RECOVERY_TIMEOUT = float(os.getenv("CB_RERANKER_RECOVERY_TIMEOUT", "30"))

# Retry defaults
RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 30.0

# Loop detection
LOOP_MAX_CONSECUTIVE_SAME = 3
LOOP_MAX_TOTAL_REPEATS = 3

# Phase 2 agent limits
RESEARCHER_MAX_ITERATIONS = 6  # Balanced: enough depth without excessive latency on GB10
RESEARCHER_MAX_TOKENS = 150_000  # Research produces more context
REFLECTION_ENABLED = os.getenv("REFLECTION_ENABLED", "true").lower() == "true"
REFLECTION_TEMPERATURE = float(os.getenv("REFLECTION_TEMPERATURE", "0.1"))
ORCHESTRATOR_MAX_SUBTASKS = 6  # Prevent over-decomposition
ORCHESTRATOR_MAX_DEPTH = 1  # No recursive orchestration
ORCHESTRATOR_TIMEOUT_SECONDS = 120  # Total wall-clock budget

# Graph search
GRAPH_SEARCH_DEFAULT_TOP_K = int(os.getenv("GRAPH_SEARCH_DEFAULT_TOP_K", "20"))

# Reranker
RERANKER_FETCH_MULTIPLIER = int(os.getenv("RERANKER_FETCH_MULTIPLIER", "3"))
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))

# Freshness checking
FRESHNESS_CHECK_INTERVAL_HOURS = int(os.getenv("FRESHNESS_CHECK_INTERVAL_HOURS", "6"))
FRESHNESS_HTTP_TIMEOUT = int(os.getenv("FRESHNESS_HTTP_TIMEOUT", "10"))

# Domain → check_interval_hours mapping for new sources during ingest
DOMAIN_CHECK_INTERVALS: dict[str, int] = {
    "wowhead.com": 720,  # 30 days — guides update slowly
    "icy-veins.com": 720,  # 30 days
    "shadowpanther.net": 2160,  # 90 days — TBC content is static
    "silentshadows.net": 2160,  # 90 days
    "tbcdb.com": 2160,  # 90 days — database, patch-locked
    "warcraftlogs.com": 24,  # 1 day — rankings change constantly
}
DEFAULT_CHECK_INTERVAL_HOURS = int(os.getenv("DEFAULT_CHECK_INTERVAL_HOURS", "168"))


def get_check_interval(url: str) -> int:
    """Extract domain from URL, return check_interval_hours from mapping."""
    from urllib.parse import urlparse

    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return DOMAIN_CHECK_INTERVALS.get(domain, DEFAULT_CHECK_INTERVAL_HOURS)


# Batch ingest
BATCH_CONCURRENCY = int(os.getenv("BATCH_CONCURRENCY", "3"))
MANIFEST_PATH = Path(os.getenv("MANIFEST_PATH", "/project/data/sources/manifest.yaml"))

# WCL API
WCL_TOKEN_URL = "https://www.warcraftlogs.com/oauth/token"
WCL_CLASSIC_ENDPOINT = "https://classic.warcraftlogs.com/api/v2/client"
WCL_FRESH_ENDPOINT = "https://fresh.warcraftlogs.com/api/v2/client"
WCL_RATE_LIMIT_BUDGET = 3600
WCL_RATE_LIMIT_BUFFER = int(os.getenv("WCL_RATE_LIMIT_BUFFER", "600"))
WCL_RATE_CHECK_INTERVAL = int(os.getenv("WCL_RATE_CHECK_INTERVAL", "10"))
WCL_QUERY_TIMEOUT = float(os.getenv("WCL_QUERY_TIMEOUT", "30.0"))
WCL_TRACKED_CHARACTERS: list[dict[str, str | int]] = [
    {"wcl_id": 104956434, "name": "Lyroo", "server": "nightslayer", "region": "us", "endpoint": "fresh"},
]
CB_WCL_FAILURE_THRESHOLD = int(os.getenv("CB_WCL_FAILURE_THRESHOLD", "3"))
CB_WCL_RECOVERY_TIMEOUT = float(os.getenv("CB_WCL_RECOVERY_TIMEOUT", "60"))

# Backup
BACKUP_KEEP_COUNT = int(os.getenv("BACKUP_KEEP_COUNT", "7"))

# Memory
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
MEMORY_RECALL_TOP_K = int(os.getenv("MEMORY_RECALL_TOP_K", "5"))
MEMORY_STRATEGY_TOP_K = int(os.getenv("MEMORY_STRATEGY_TOP_K", "3"))
MEMORY_RECENCY_HALF_LIFE_DAYS = float(os.getenv("MEMORY_RECENCY_HALF_LIFE_DAYS", "30.0"))
MEMORY_COMPOSITE_SIM_WEIGHT = 0.6
MEMORY_COMPOSITE_RECENCY_WEIGHT = 0.2
MEMORY_COMPOSITE_QUALITY_WEIGHT = 0.2
MEMORY_MAX_CONTEXT_CHARS = 2000

# Simulation engine
SIM_DEFAULT_ITERATIONS = int(os.getenv("SIM_DEFAULT_ITERATIONS", "10000"))
SIM_DEFAULT_FIGHT_LENGTH = int(os.getenv("SIM_DEFAULT_FIGHT_LENGTH", "300"))
SIM_STAT_WEIGHT_DELTA = int(os.getenv("SIM_STAT_WEIGHT_DELTA", "80"))
SIM_CACHE_ENABLED = os.getenv("SIM_CACHE_ENABLED", "true").lower() == "true"
ANALYST_MAX_ITERATIONS = int(os.getenv("ANALYST_MAX_ITERATIONS", "5"))

# Role-specific system prompts
from code.shukketsu.llm.prompts.editor import (  # noqa: E402
    EDITOR_SYSTEM_PROMPT as EDITOR_SYSTEM_PROMPT,
)
from code.shukketsu.llm.prompts.researcher import (  # noqa: E402
    RESEARCHER_SYSTEM_PROMPT as RESEARCHER_SYSTEM_PROMPT,
)
from code.shukketsu.llm.prompts.writer import (  # noqa: E402
    WRITER_SYSTEM_PROMPT as WRITER_SYSTEM_PROMPT,
)

# Editor
EDITOR_CONFIDENCE_THRESHOLD = float(os.getenv("EDITOR_CONFIDENCE_THRESHOLD", "0.6"))

from code.shukketsu.llm.prompts.analyst import (  # noqa: E402
    ANALYST_SYSTEM_PROMPT as ANALYST_SYSTEM_PROMPT,
)
from code.shukketsu.llm.prompts.orchestrator import (  # noqa: E402
    ORCHESTRATOR_SYSTEM_PROMPT as ORCHESTRATOR_SYSTEM_PROMPT,
)
