# Step 5: Ingest Pipeline — Design

> Feed text into the system: chunk it, embed it, store it in SQLite.
> Then retrieve it via the agent's rag_search tool.

## Architecture

```
Text + metadata (url, title, source_type)
     │
     ▼
 chunker.py ─── recursive split (headers > paragraphs > sentences > words)
     │            ~400 tokens per chunk, 50 token overlap, merge small chunks
     ▼
 embedder.py ── batch embed via Ollama /v1/embeddings (nomic-embed-text)
     │            search_document: prefix for storage, search_query: for retrieval
     ▼
 pipeline.py ── orchestrate: dedup → chunk → embed → store (single transaction)
                 insert source + chunks + vectors atomically
```

### Integration Points

- `pipeline.py` takes `sqlite3.Connection` (from `db/connection.py`)
- Embedder uses `AsyncOpenAI` pointed at Ollama `localhost:11434/v1`
- Chunks go into existing `sources`, `chunks`, `chunks_vec` tables
- FTS5 auto-populated by existing triggers in `schema.sql`
- No new dependencies (uses `openai` + `sqlite3` already in project)

## Module 1: Chunker (`ingest/chunker.py`)

Pure function, no I/O, no async.

### Splitting Strategy

Recursive character text splitting with semantic separators. Research shows
this achieves 88-89.5% recall at 400 tokens (Chroma benchmarks), only ~3%
below semantic chunking but without the complexity of embedding every sentence.

**Separators** (tried in priority order, recurse on next if chunk too large):
1. `\n## ` — Markdown H2 headers
2. `\n### ` — H3 headers
3. `\n\n` — paragraph breaks
4. `\n` — line breaks
5. `. ` — sentence endings
6. ` ` — word boundaries (last resort)

### Parameters

- `CHUNK_MAX_TOKENS = 400` (~1600 chars) — benchmark sweet spot
- `CHUNK_MIN_TOKENS = 50` (~200 chars) — merge threshold
- `CHUNK_OVERLAP_TOKENS = 50` (~200 chars) — 12.5% overlap (within 10-20% range)
- Token estimation: `len(text) // 4` (no tokenizer dependency)

### Output Type

```python
@dataclass(frozen=True)
class Chunk:
    content: str
    chunk_index: int      # order within document
    char_count: int
    token_estimate: int   # len(content) // 4
```

### Behavior

- After splitting, scan for chunks below `min_tokens` — merge with next
  neighbor (or previous if last)
- After merging, add overlap: prepend last N chars of previous chunk to
  current chunk
- Empty input returns empty list

## Module 2: Embedder (`ingest/embedder.py`)

Thin async wrapper around Ollama's OpenAI-compatible `/v1/embeddings`.

### Interface

```python
class Embedder:
    """Batch text embedder using nomic-embed-text via Ollama."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one API call.

        Adds 'search_document: ' prefix (required by nomic-embed-text).
        Returns list of 768-dim vectors, one per input text.
        """

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single search query.

        Adds 'search_query: ' prefix (required by nomic-embed-text).
        """
```

### Details

- Uses `AsyncOpenAI(base_url=OLLAMA_BASE_URL + "/v1")` — same library as
  `llm/clients.py`
- `embed_texts()` sends all texts in single batch call via
  `client.embeddings.create(input=[...], model="nomic-embed-text")`
- Raises `EmbeddingError` on connection failure
- `embed_query()` matches the `EmbedFn` type alias in `tools/knowledge/search.py`

## Module 3: Pipeline (`ingest/pipeline.py`)

Orchestrates the full flow.

### Interface

```python
class IngestPipeline:
    def __init__(self, conn: sqlite3.Connection, embedder: Embedder) -> None: ...

    async def ingest(
        self,
        text: str,
        url: str,
        title: str,
        source_type: str = "guide",
    ) -> IngestResult: ...
```

```python
@dataclass(frozen=True)
class IngestResult:
    source_id: int
    chunk_count: int
    already_existed: bool
```

### Flow

1. Compute `content_hash = sha256(text)`
2. Check if `sources` has this hash for this URL → return early (`already_existed=True`)
3. If URL exists but hash changed → delete old chunks, update source row
4. If new URL → insert source row
5. `chunker.chunk_text(text)` → list of `Chunk`
6. `embedder.embed_texts([c.content for c in chunks])` → list of vectors
7. Single transaction:
   - Insert chunks into `chunks` table (FTS5 trigger handles `chunks_fts`)
   - Insert vectors into `chunks_vec` (matching rowids)
   - Update `sources.chunk_count`
8. Return `IngestResult`

### Deduplication

- Same URL + same content hash → skip (return `already_existed=True`)
- Same URL + different hash → re-ingest (delete old chunks, insert new)
- New URL → fresh insert

### Error Handling

- Embedding failure before transaction → no partial data, clean state
- DB error during transaction → automatic rollback
- Empty text → return `IngestResult(source_id=..., chunk_count=0, already_existed=False)`

## Config Changes

New constants in `config.py`:

```python
# Chunking
CHUNK_MAX_TOKENS = 400
CHUNK_MIN_TOKENS = 50
CHUNK_OVERLAP_TOKENS = 50

# Embedding
EMBEDDING_DIMENSIONS = 768
EMBEDDING_BATCH_SIZE = 64
```

## New Error Class

In `resilience/errors.py`:

```python
class EmbeddingError(ShukketsuError):
    """Raised when the embedding model fails."""
    def __init__(self, message: str):
        super().__init__(message, FailureMode.EMBEDDING_ERROR)
```

## Testing

### Unit Tests (~17 tests, no Ollama needed)

**test_chunker.py** (~9 tests):
- Splits on H2 headers when text exceeds max_tokens
- Splits on paragraphs when no headers
- Splits on sentences as fallback
- Respects max_tokens limit (no chunk exceeds it)
- Merges chunks below min_tokens with neighbor
- Adds overlap from previous chunk
- Handles empty input (returns empty list)
- Handles single-sentence input (returns one chunk)
- Preserves chunk ordering (chunk_index sequential)

**test_embedder.py** (~4 tests):
- Adds `search_document: ` prefix to texts in `embed_texts()`
- Adds `search_query: ` prefix in `embed_query()`
- Raises `EmbeddingError` on connection failure
- Batch call sends all texts in single request (mock verify)

**test_pipeline.py** (~6 tests):
- Full ingest stores source, chunks, and vectors in DB
- Content hash dedup skips re-ingest of identical content
- Changed content re-ingests (deletes old, inserts new)
- Returns correct `IngestResult` fields
- Empty text returns zero chunks
- Vectors have correct dimensionality (768)

### Integration Tests (needs Ollama + nomic-embed-text)

- Embedder returns 768-dim vector for real text
- Full pipeline: ingest text → rag_search finds it

## References

- [Best Chunking Strategies for RAG 2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)
- [Ollama Embeddings Docs](https://docs.ollama.com/capabilities/embeddings)
- [Ollama OpenAI Compatibility](https://docs.ollama.com/api/openai-compatibility)
