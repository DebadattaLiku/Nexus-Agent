# NexusAI — MCP-Native Enterprise Multi-Agent AI Platform

Built incrementally, phase by phase. This README covers **Phases 1–5**.

## Phase 1 — Document MCP Server

Goal: prove that an MCP server and client work end-to-end, using the
simplest possible tool set — no embeddings, no RAG, no agents yet.

The server exposes three tools over local `.txt` files in `data/documents/`
(a fourth, `semantic_search`, is added in Phase 3 below):

- `list_documents()` — metadata for every document
- `get_document(filename)` — full text of one document
- `search_documents(query, case_sensitive=False)` — line-level substring search

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the client demo (in-process — recommended for development)

```bash
PYTHONPATH=src python -m nexusai.client.mcp_client
```

This connects to the server directly in the same process (no subprocess,
no stdio plumbing) and calls all three tools, printing results.

### Run the client demo over a real stdio subprocess

This mirrors how an external MCP host (e.g. Claude Desktop) would talk to
the server — spawning it as a child process and speaking MCP over stdin/stdout:

```bash
PYTHONPATH=src python -m nexusai.client.mcp_client --stdio
```

### Run the server standalone

```bash
PYTHONPATH=src python -m nexusai.mcp_servers.document_server
```

It will sit waiting for an MCP client to connect over stdio. Use Ctrl+C to stop.

### Run tests

```bash
PYTHONPATH=src pytest tests/ -v
```

## Project structure (Phase 1–3)

```
nexus-ai/
├── src/
│   └── nexusai/
│       ├── __init__.py
│       ├── mcp_servers/
│       │   ├── __init__.py
│       │   └── document_server.py   # Phase 1 tools + Phase 3 semantic_search
│       ├── client/
│       │   ├── __init__.py
│       │   └── mcp_client.py
│       ├── llm/                    # Phase 2
│       │   ├── __init__.py
│       │   └── provider.py
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── tool_agent.py       # Phase 2 — linear tool-calling loop
│       │   └── langgraph_agent.py  # Phase 4 — LangGraph graph orchestration
│       └── rag/                    # Phase 3
│           ├── __init__.py
│           ├── config.py            # RAGConfig — every RAG knob, one place
│           ├── ingestion.py         # documents on disk -> RawDocument
│           ├── chunking.py          # RawDocument -> overlapping Chunks
│           ├── embeddings.py        # EmbeddingProvider: fastembed | hashing
│           ├── vector_store.py      # FAISS IndexFlatIP + metadata, save/load
│           └── pipeline.py          # RAGPipeline: orchestration + caching
├── data/
│   ├── documents/          # sample .txt files the server reads
│   └── index/               # Phase 3 — persisted FAISS index + metadata (generated)
├── tests/
│   ├── __init__.py
│   ├── test_document_server.py
│   ├── test_tool_agent.py           # Phase 2
│   ├── test_groq_provider.py        # Phase 2
│   ├── test_groq_integration_e2e.py # Phase 2
│   ├── test_rag_ingestion.py        # Phase 3
│   ├── test_rag_chunking.py         # Phase 3
│   ├── test_rag_embeddings.py       # Phase 3
│   ├── test_rag_vector_store.py     # Phase 3
│   ├── test_rag_pipeline.py         # Phase 3
│   ├── test_semantic_search_tool.py # Phase 3
│   └── test_langgraph_agent.py      # Phase 4
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Notes on the MCP SDK version

This project pins `mcp==2.0.0`, the current stable major version of the
official Python SDK. v2 renamed the high-level server class from
`FastMCP` to `MCPServer` (now imported from `mcp.server.mcpserver`) and
switched all field names to snake_case. If you're following older MCP
tutorials that use `from mcp.server.fastmcp import FastMCP`, that's the
v1 API — this code intentionally uses the newer one.

## Phase 2 — LLM + MCP Tool Calling

Goal: let an LLM decide, on its own, whether it needs a document tool to
answer a question — and if so, call it through the existing MCP client/server,
never directly.

### Architecture

```text
User
 ↓
LLM (Groq by default, native tool/function calling; Anthropic optional)
 ↓
LLM decides whether a tool is needed
 ↓
ToolAgent validates the request (known tool? valid arguments?)
 ↓
MCP Client  →  Document MCP Server  →  MCP Tool
 ↓
Tool Result
 ↓
LLM
 ↓
another tool call, or a Final Answer
```

- `src/nexusai/llm/provider.py` — `LLMProvider` interface with two concrete
  implementations, both speaking the same Anthropic-shaped content-block
  convention internally so `tool_agent.py` never has to know which is active:
  - `GroqProvider` (default) — Groq's free, no-credit-card developer tier.
    Translates to/from Groq's OpenAI-compatible wire format, and recovers
    gracefully from Llama's "pythonic" `<function=NAME{...}>` tool-call
    syntax if a model emits it instead of a structured `tool_calls` entry
    (seen both as an HTTP 400 `failed_generation` body and as plain text on
    an HTTP 200 response). Recovery is generic over any tool name/arguments
    — nothing is hard-coded to a specific tool.
  - `AnthropicProvider` — the Anthropic Messages API, kept available via
    `LLM_PROVIDER=anthropic`.
- `src/nexusai/agent/tool_agent.py` — `ToolAgent`: discovers MCP tools live
  via `list_tools()`, hands their schemas to the LLM, executes any tool call
  the LLM requests through the Phase 1 MCP client, feeds each result back
  (preserving tool_call IDs), and loops (up to `MAX_TOOL_ROUNDS = 5`) until
  the LLM gives a final answer. Rejects unknown tool names and malformed/
  missing arguments before ever calling `call_tool()`, handles multiple
  tool calls in a single LLM turn, and catches LLM/API failures so a bad
  request never crashes the CLI.

The agent never imports `document_server.py` — all document access goes
through the MCP client, exactly like Phase 1.

### Setup

Same as Phase 1, plus an API key:

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GROQ_API_KEY (or switch to Anthropic)
```

### Environment variables

- `LLM_PROVIDER` — `groq` (default) or `anthropic`.
- `GROQ_API_KEY` — required to run the agent with the default provider. Get a
  free key at https://console.groq.com/keys.
- `GROQ_MODEL` — optional, defaults to `llama-3.3-70b-versatile`.
- `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` — only used if `LLM_PROVIDER=anthropic`.

None of these are required to run the test suite, which uses a mocked/fake
LLM provider throughout.

### Run the CLI

```bash
PYTHONPATH=src python -m nexusai.agent.tool_agent
```

```text
NexusAI Phase 2
You: What documents are available?
You: What do the documents say about RAG?
```

Type `exit` or Ctrl+C to quit.

### Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

- `tests/test_tool_agent.py` — the provider-agnostic `ToolAgent` loop against
  a scripted `FakeLLM`: tool discovery, single and sequential/multi-step tool
  calls, multiple tool calls in one LLM turn, unknown tools, malformed
  arguments, `MAX_TOOL_ROUNDS` loop termination, and LLM/API errors handled
  without crashing.
- `tests/test_groq_provider.py` — `GroqProvider` in isolation (mocked
  `groq.Groq` client, no network/API key): structured tool calls, multiple
  tool calls in one response, both shapes of the pythonic `<function=...>`
  recovery, malformed JSON arguments, and request-shape sanity checks.
- `tests/test_groq_integration_e2e.py` — the real `ToolAgent` + real
  `GroqProvider` + real MCP document server wired together (only the Groq
  HTTP client is mocked), reproducing the two exact questions from the
  original bug report end-to-end.

## Phase 3 — Real Semantic RAG

Goal: replace substring search with a real retrieval pipeline —
ingestion → chunking → embeddings → FAISS vector index → semantic MCP tool
→ grounded LLM answer — while keeping everything free/local and keeping
Phase 1/2 untouched.

### Architecture

```text
documents (data/documents/*.txt)
 ↓
ingestion        (nexusai/rag/ingestion.py)
 ↓
chunking         (nexusai/rag/chunking.py)   — configurable size/overlap
 ↓
embeddings       (nexusai/rag/embeddings.py) — local/free, pluggable provider
 ↓
FAISS index      (nexusai/rag/vector_store.py) — IndexFlatIP, save/load
 ↓
RAGPipeline      (nexusai/rag/pipeline.py)   — orchestration + on-disk cache
 ↓
semantic_search MCP tool (mcp_servers/document_server.py)
 ↓
MCP Client → LLM → grounded, source-attributed answer
```

The agent still never imports `nexusai.rag` or `document_server.py`
directly — it only ever talks to whatever tools `list_tools()` discovers
over MCP, exactly as in Phase 1/2.

### `search_documents` vs `semantic_search`

Both tools are kept, on purpose, because they solve different problems:

- **`search_documents(query, case_sensitive=False)`** — exact/lexical,
  line-level substring match. Best when the caller knows a specific word,
  identifier, or short phrase that must literally appear. Zero setup cost,
  trivially explainable results.
- **`semantic_search(query, top_k=4)`** — conceptual/meaning-based match
  over embedded chunks via FAISS. Best for natural-language questions where
  the right document doesn't necessarily use the query's exact words.
  Returns each chunk's source filename, chunk id, text, and similarity
  score.

The system prompt (`agent/tool_agent.py`) tells the LLM to prefer
`semantic_search` for document questions, fall back to `search_documents`
for exact lookups, ground answers only in retrieved content, cite sources
when useful, and say plainly when the retrieved chunks don't have enough
information.

### Embedding provider

`RAGConfig.embedding_provider` selects the implementation
(`nexusai/rag/embeddings.py`):

- **`fastembed`** (default) — real, local, free embeddings via the
  [`fastembed`](https://github.com/qdrant/fastembed) package, which runs
  Hugging Face sentence-embedding models (default:
  `BAAI/bge-small-en-v1.5`, 384-dim) locally through ONNX Runtime. No API
  key, no per-call cost. Model weights download from Hugging Face once and
  are cached under `.cache/fastembed/`; every call after that is fully
  offline.
- **`hashing`** — a deterministic, dependency-free bag-of-words
  feature-hashing embedding (same idea as scikit-learn's
  `HashingVectorizer`). Same text always produces the same vector, no
  network or model file is ever touched. This is what the test suite uses,
  and it's also a reasonable degraded-mode fallback for fully offline
  deployments.

Both providers return unit-normalized vectors, so FAISS inner-product
search is exactly cosine-similarity search for either one — swapping
providers never requires touching `vector_store.py`.

> **Why `fastembed` instead of `sentence-transformers` directly?**
> `sentence-transformers` requires `torch`, whose PyPI wheel is several GB
> — impractical for this environment's disk budget. `fastembed` runs the
> same class of Hugging Face sentence-embedding models (including
> `BAAI/bge-small-en-v1.5`) through ONNX Runtime instead, giving the same
> "real local/free HF model" result with a much lighter dependency
> footprint. Swap `embedding_provider`/`embedding_model` in `RAGConfig` if
> you'd rather use `sentence-transformers` in an environment where the
> torch install cost is acceptable — the `EmbeddingProvider` interface
> doesn't change.

### FAISS strategy

`vector_store.py` uses `faiss.IndexFlatIP` (exact inner-product search) —
at this data scale (a handful of documents / a few hundred chunks), exact
search is fast enough and needs no approximate-index tuning
(`IndexIVFFlat`, `HNSW`, etc.), which keeps the implementation simple and
fully deterministic. Metadata (filename, chunk id, chunk text) is kept in a
parallel list, indexed by FAISS vector position, and persisted as JSON
alongside the FAISS index file so it survives a save/load cycle.

### Caching (no repeated embedding on every start)

`RAGPipeline.load_or_build()` computes a SHA-256 fingerprint of the
chunking config, embedding provider/model/dim, and every ingested
document's filename + content. It reuses a saved FAISS index from
`data/index/` when the fingerprint matches; otherwise it rebuilds and
persists a fresh one. Restarting the MCP server does **not** re-embed
anything unless the documents or RAG config actually changed.

### Configuration

All knobs live in `RAGConfig` (`nexusai/rag/config.py`), overridable via
environment variables:

| Variable                | Default                   | Meaning                              |
|--------------------------|---------------------------|---------------------------------------|
| `RAG_CHUNK_SIZE`          | `200`                      | words per chunk                        |
| `RAG_CHUNK_OVERLAP`       | `40`                       | words shared between consecutive chunks|
| `RAG_EMBEDDING_PROVIDER`  | `fastembed`                | `fastembed` \| `hashing`               |
| `RAG_EMBEDDING_MODEL`     | `BAAI/bge-small-en-v1.5`   | HF model name (fastembed only)         |
| `RAG_EMBEDDING_DIM`       | `384`                      | embedding vector dimension             |
| `RAG_TOP_K`               | `4`                        | default number of chunks to retrieve   |

Index location defaults to `data/index/` and isn't currently
env-overridden (see "Limitations" below).

### Backward compatibility

`list_documents()` and `get_document()` are unchanged. `search_documents()`
is unchanged and kept as a separate tool (see above) rather than replaced.

### Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

New in Phase 3 — all deterministic, all offline (the `hashing` embedding
provider is used throughout, so no test downloads a model or needs
network/API keys):

- `tests/test_rag_ingestion.py` — loading, sorting, extension filtering,
  missing-directory handling.
- `tests/test_rag_chunking.py` — word-count splitting, overlap windows,
  short-document single-chunk case, empty text, invalid
  size/overlap combinations, stable chunk IDs.
- `tests/test_rag_embeddings.py` — shape, determinism, unit-normalization,
  empty input, similar-vs-unrelated-text similarity ordering, provider
  factory wiring.
- `tests/test_rag_vector_store.py` — add/search ranking, metadata
  preservation, empty-store search, top_k clamping, mismatched-vector
  rejection, save/load round-trip, missing-file load errors.
- `tests/test_rag_pipeline.py` — end-to-end build + retrieve, empty-corpus
  handling, empty-query/invalid-top_k rejection, top_k capping, on-disk
  cache reuse (`build()` proven not to be called again), cache invalidation
  when documents or chunking config change, fingerprint stability.
- `tests/test_semantic_search_tool.py` — tool discoverability, relevant
  results (against the real `data/documents/*.txt` files), top_k
  respected, empty-query/invalid-top_k rejected over MCP, and a
  groundedness check that every returned chunk's text is an actual
  substring of a real document (never fabricated).

`tests/test_document_server.py` and `tests/test_tool_agent.py` (Phase 1/2)
had their tool-set assertions updated to include `semantic_search` — no
other Phase 1/2 test logic changed.

## Phase 4 — LangGraph-Based Agent Orchestration

Goal: replace the linear Python tool-calling loop (`agent/tool_agent.py`,
Phase 2) with an explicit, stateful [LangGraph](https://github.com/langchain-ai/langgraph)
graph, while keeping MCP as the only tool-execution boundary and leaving
Phases 1–3 untouched. `tool_agent.py` is kept as-is (Phase 2 is still a
valid, working agent) — `langgraph_agent.py` is a new, parallel entry
point, not a replacement in place.

### Architecture

```text
User
 ↓
LangGraph Agent           (agent/langgraph_agent.py)
 ↓
agent node  — one LLM call (Groq, via llm/provider.py)
 ↓
 ├─ no tool calls → final_answer set → END
 └─ tool call(s) requested
     ↓
    tools node — executes every call through the MCP client only
     ↓
    MCP Client → Document MCP Server → RAG / document functionality
     ↓
    tool result(s) appended to conversation
     ↓
    back to agent node (repeat, up to MAX_TOOL_ROUNDS)
```

Dependency direction is unchanged and enforced by import hygiene:
`langgraph_agent.py` imports `nexusai.client.mcp_client` (the client
boundary) and small reusable helpers from `agent/tool_agent.py`
(`SYSTEM_PROMPT`, `MAX_TOOL_ROUNDS`, `_validate_arguments`, and
`ToolAgent`'s static schema/result-flattening helpers). It never imports
`nexusai.mcp_servers.document_server` or `nexusai.rag` directly — those
stay reachable only through the MCP client, exactly as in Phase 2.

### State

`AgentState` (a `TypedDict` in `langgraph_agent.py`) carries:

| Field                 | Purpose                                                        |
|------------------------|------------------------------------------------------------------|
| `messages`              | Full conversation history (same content-block shape `LLMProvider` uses) |
| `pending_tool_calls`    | Tool calls just requested by the LLM, not yet executed          |
| `round_count`           | LLM round trips completed so far — drives the max-step cutoff   |
| `final_answer`          | Set once the graph has an answer; its presence routes to `END`  |

### Nodes and edges

```text
START → agent ──tools──→ tools
          │                │
          └────end────→ END │
          ▲                 │
          └─────────────────┘
```

- **`agent`** — calls the LLM with the current messages + the MCP tool
  schemas discovered at the start of the run. On an LLM error, or on
  hitting `MAX_TOOL_ROUNDS`, it sets a graceful `final_answer` itself
  rather than raising — so error handling is normal graph state, not an
  exception path.
- **`tools`** — executes every pending tool call through the MCP client
  (`known_tools` gates against unknown names; `_validate_arguments` checks
  required arguments; MCP/transport failures are caught per-call), then
  clears `pending_tool_calls` and routes back to `agent`.
- **Routing** (`_route_after_agent`) is a single rule: `final_answer is not
  None` → `END`, otherwise → `tools`. One rule covers the normal-finish,
  LLM-error, and max-round-cutoff cases identically.

### Run the CLI

Same setup/env vars as Phase 2 (`GROQ_API_KEY`, etc.) — Phase 4 doesn't add
any new credential requirement:

```bash
PYTHONPATH=src python -m nexusai.agent.langgraph_agent
```

```text
NexusAI Phase 4 (LangGraph)
You: What does the RAG document say about retrieval?
You: List the documents, then show me the LangGraph one.
```

Type `exit` or Ctrl+C to quit.

### MCP integration

`build_graph(llm, client, known_tools, tool_schemas)` takes an
already-connected MCP client and the tools discovered from it — the graph
itself has no idea whether that client is the real in-process/stdio MCP
client or a test double, which is what makes the tool-execution node
testable without a live server. `LangGraphAgent.answer()` is the
production entry point: it opens `connect_in_process()`, discovers tools,
builds the graph, and runs it — one connection per question, mirroring
`ToolAgent.answer()`'s lifecycle exactly.

### Files created/modified

- **Created:** `src/nexusai/agent/langgraph_agent.py`,
  `tests/test_langgraph_agent.py`.
- **Modified:** `requirements.txt` (added `langgraph`), `README.md` (this
  section). `agent/tool_agent.py` and every Phase 1–3 file are untouched.

### Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

`tests/test_langgraph_agent.py` — all deterministic, offline, no Groq/API
key/network required (a fake `LLMProvider` throughout, plus a fake MCP
client for the error-injection cases):

- Graph construction/compilation and initial-state shape.
- Direct final answer with no tool call.
- A single `search_documents` tool call round-tripped through the real
  in-process Document MCP Server.
- A `semantic_search` tool call against a real (hashing-embedded) RAG
  pipeline.
- Multi-step `list_documents()` → `get_document()` sequential tool calls.
- Routing (`agent → tools`, `agent → END`) as isolated unit tests of
  `_route_after_agent`.
- Unknown-tool rejection and malformed-argument rejection, via a fake MCP
  client that asserts the real `call_tool` is never reached.
- Simulated MCP transport failure and simulated LLM failure, both handled
  as a graceful `final_answer` rather than a raised exception.
- Max-round termination (`MAX_TOOL_ROUNDS`), against a fake LLM that always
  requests a tool call.
- An empty tool result (`search_documents` with no matches) not crashing
  the graph.

All existing Phase 1–3 tests are unchanged and still run in the same
suite.

### Known limitations

- No checkpointer/persistence is configured — each `answer()` call builds
  a fresh graph and a fresh MCP connection; there is no cross-turn memory
  (this is explicitly out of scope for Phase 4, see below).
- `round_count` bounds LLM round trips the same way `MAX_TOOL_ROUNDS` did
  in Phase 2, but because every round in the graph can carry multiple
  parallel tool calls, the *tool-call* count is only loosely bounded by
  it — a single LLM turn requesting many parallel tool calls still costs
  one round.
- The graph is single-agent (one `agent` node); multi-agent graphs,
  sub-graphs, and dynamic tool discovery mid-run are Phase 5+ concerns.

### What Phase 5 should implement

Per the Phase 4 prompt's "do not build yet" list: multi-agent systems, a
SQL MCP server (PostgreSQL-backed), conversation memory,
human-in-the-loop approval steps, containerization (Docker), a frontend,
an evaluation framework, advanced/hybrid retrieval (reranking, BM25 +
semantic), and autonomous planning beyond the Phase 4 graph.

## Phase 5 — SQL MCP + Multi-Tool Agent

Goal: add a second MCP server providing safe, read-only access to a local
SQLite database, and extend the Phase 4 LangGraph agent to dynamically
choose between the Document MCP server and the new SQL MCP server per
question — without hard-coding any question to a specific server. Phases
1–4 are otherwise untouched.

### Architecture

```text
User
 ↓
LangGraph Agent            (agent/langgraph_agent.py)
 ↓
Groq LLM                   (llm/provider.py)
 ↓
Dynamic MCP tool selection (LLM picks a tool by name/schema)
 ├── Document MCP  →  RAG / document tools      (mcp_servers/document_server.py)
 │
 └── SQL MCP       →  SQLite database            (mcp_servers/sql_server.py)
 ↓
Tool result
 ↓
LangGraph → Groq → final answer (or another tool call, same or different server)
```

Dependency direction is unchanged and still enforced by import hygiene:
`langgraph_agent.py` never imports `sql_server.py`, `document_server.py`,
`nexusai.rag`, or `nexusai.db` directly. It only imports
`nexusai.client.mcp_client` (`connect_multi_in_process()`), which is the
client-side piece that actually knows which server owns which tool name.

```text
LangGraph Agent → MCP Client → MCP Server → Domain implementation
                                             (nexusai.rag / nexusai.db)
```

### Database

`nexusai/db/sql_database.py` builds a small, fully deterministic (never
randomly generated) SQLite database with two tables:

- **`companies`** — `company_id`, `ticker`, `name`, `sector` (5 rows: AAPL,
  MSFT, JPM, XOM, TSLA across four sectors).
- **`prices`** — `price_id`, `ticker`, `trade_date`, `close_price`,
  `volume` (10 trading days per company, 50 rows total). Closing prices
  follow a fixed linear formula per ticker, so derived figures like
  "highest return over the period" are known constants (TSLA: +18%, the
  highest; JPM: −1.8%, the only decliner) — useful for both demo queries
  and deterministic tests.

The database file (`data/nexus.db`, gitignored) is built lazily on first
use, the same pattern `document_server.py` uses for the RAG pipeline
(`get_pipeline()`/`set_pipeline()`); `sql_server.py` uses
`get_db_path()`/`set_db_path()` for the same purpose, so tests run against
an isolated `tmp_path` database rather than the real one.

### SQL MCP Server

`mcp_servers/sql_server.py` exposes exactly one tool:

```text
query_database(sql: str) -> QueryResult(columns, rows, row_count, truncated)
```

- Accepts a single read-only SQL statement.
- Returns structured rows (as a list of `{column: value}` dicts), capped
  at 500 rows (`truncated=True` if more were available) so an unbounded
  `SELECT` can't flood the LLM's context.
- Invalid SQL (syntax errors) and queries against nonexistent
  tables/columns are both caught (`sqlite3.Error`) and turned into a
  normal MCP tool error via `raise ValueError(...)` — the same convention
  `document_server.get_document()` uses for a missing file — rather than
  an unhandled exception.

### Security

Every query passes through **two independent** read-only layers before a
single row comes back:

1. **`nexusai/db/query_guard.py`** — an allow-list validator run *before*
   the query ever reaches SQLite:
   - Exactly one statement (a single trailing `;` is fine; anything after
     it — e.g. `SELECT 1; DROP TABLE companies;` — is rejected outright).
   - Must start with `SELECT` or `WITH` (a read-only CTE).
   - Rejects `INSERT`, `UPDATE`, `DELETE`, `REPLACE`, `DROP`, `ALTER`,
     `CREATE`, `TRUNCATE`, `ATTACH`, `DETACH`, `PRAGMA`, `VACUUM`,
     `REINDEX`, `GRANT`, `REVOKE`, and transaction-control keywords
     (`BEGIN`/`COMMIT`/`ROLLBACK`/...) if they appear anywhere in the
     statement — checked on a comment-stripped copy, so a keyword hidden
     inside `/* ... */` or `-- ...` can't slip past a naive substring
     check that only looked outside comments.
2. **`nexusai/db/sql_database.get_read_only_connection()`** — the SQLite
   connection itself is opened with the `mode=ro` URI flag and
   `PRAGMA query_only = ON`, so even a query that somehow passed layer 1
   still cannot write at the SQLite-engine level. (Verified directly: an
   `INSERT` attempted straight against this connection raises
   `sqlite3.OperationalError: attempt to write a readonly database`.)

No arbitrary filesystem access is exposed — the tool only ever opens the
one configured database path, never a path derived from the query.

### Multi-MCP tool discovery & routing

`nexusai/client/mcp_client.py` adds:

- `connect_sql_in_process()` / `connect_over_stdio_sql()` — same two
  connection styles Phase 1 already has for the Document server, now for
  the SQL server too.
- **`MultiMCPClient`** — aggregates `list_tools()`/`call_tool()` across
  several already-connected MCP clients. `list_tools()` merges the tool
  lists from every server and records which client each tool name came
  from; `call_tool(name, args)` looks up that mapping and forwards the
  call to the right server. (If two servers ever exposed the same tool
  name, the first discovered wins and a warning is logged — a defensive
  fallback, not a supported configuration; tool names are unique across
  this project's two servers.)
- **`connect_multi_in_process()`** — opens both servers via
  `AsyncExitStack` and yields one `MultiMCPClient` wrapping both. This is
  what `LangGraphAgent.answer()` now uses instead of Phase 4's
  single-server `connect_in_process()`.

Tool *selection* is never hard-coded to a question: `LangGraphAgent`
hands the LLM the full combined schema list from both servers (tool
names + descriptions + input schemas), and the LLM picks by that alone.
`agent/langgraph_agent.py`'s system prompt (kept local to that module,
not shared with Phase 2's `tool_agent.py`, whose prompt stays
document-only) describes both tool families and tells the model to
choose based on what the question needs — document content vs.
structured/aggregate data — and to call tools from both when a question
needs both.

### Run the CLI

Same setup/env vars as Phase 2/4 (`GROQ_API_KEY`, etc.):

```bash
PYTHONPATH=src python -m nexusai.agent.langgraph_agent
```

```text
NexusAI Phase 5 (LangGraph + Document MCP + SQL MCP)
You: Which company had the highest return?
You: What does the RAG document say about retrieval?
You: According to the documents, what is RAG, and which stock in the database had the highest return?
```

### Files created/modified

**Created:**
- `src/nexusai/db/__init__.py`
- `src/nexusai/db/sql_database.py` — schema, deterministic seed data, database creation, read-only connection helper.
- `src/nexusai/db/query_guard.py` — read-only SQL allow-list validator.
- `src/nexusai/mcp_servers/sql_server.py` — the SQL MCP server (`query_database` tool).
- `tests/test_sql_database.py`, `tests/test_query_guard.py`,
  `tests/test_sql_server.py`, `tests/test_multi_mcp_client.py`,
  `tests/test_langgraph_multi_mcp.py`

**Modified:**
- `src/nexusai/client/mcp_client.py` — added SQL server connectors,
  `MultiMCPClient`, `connect_multi_in_process()`.
- `src/nexusai/agent/langgraph_agent.py` — production `answer()` now uses
  `connect_multi_in_process()`; new local, multi-tool-aware
  `SYSTEM_PROMPT`; updated module docstring/CLI banner.
- `requirements.txt`, `.gitignore` (`data/nexus.db`), `README.md` (this section).
- `agent/tool_agent.py` and all Phase 1–3 files are untouched.

### Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

New in Phase 5 — all deterministic, offline, no Groq/API key/network
required (SQL tests run against an isolated `tmp_path` database via
`sql_server.set_db_path()`; LangGraph multi-MCP tests use a scripted fake
LLM against the real, in-process Document + SQL servers):

- `tests/test_sql_database.py` — database creation (and idempotency),
  schema (tables, columns, the `prices.ticker` → `companies.ticker`
  relationship), seed data row counts, the deterministic highest-return
  ticker, a missing-file error from `get_read_only_connection()`, and a
  direct proof that the read-only connection rejects a raw `INSERT`.
- `tests/test_query_guard.py` — valid `SELECT`/`WITH` statements accepted;
  `INSERT`/`UPDATE`/`DELETE`/`DROP`/`ALTER`/`CREATE`/`ATTACH`/`PRAGMA`
  all rejected; multiple statements rejected; a forbidden keyword hidden
  inside a comment still rejected; empty/whitespace-only input rejected.
- `tests/test_sql_server.py` — tool discovery (`query_database` only,
  description mentions both table names), a valid `SELECT` returning
  structured rows, the "highest return" example query end-to-end against
  the real seed data (asserts `TSLA`), invalid SQL syntax, a missing
  table, a missing column, and every forbidden statement type rejected
  over MCP — with each rejection followed by a `SELECT COUNT(*)` proving
  the data was actually untouched.
- `tests/test_multi_mcp_client.py` — discovery merges tool names from
  both real servers; a document-tool call and a SQL-tool call each route
  to the correct server; sequential calls across both servers in one
  session; `call_tool()` before `list_tools()` fails as "unknown tool"
  rather than crashing; plus isolated `MultiMCPClient` unit tests for the
  duplicate-tool-name and unknown-tool-name edge cases with fake clients.
- `tests/test_langgraph_multi_mcp.py` — the production `LangGraphAgent`
  discovers tools from both servers in one call; routes a database
  question to `query_database`; still routes a document question to
  `search_documents` with the SQL server also available; a mixed
  question drives two sequential tool calls across both servers before
  the final answer; a forbidden SQL statement surfaces as a graceful
  tool error (`is_error=True`) rather than a crash.

All existing Phase 1–4 tests are unchanged and still run in the same
suite; `tests/test_langgraph_agent.py`'s tests of `LangGraphAgent.answer()`
now implicitly exercise the multi-server connection too (the SQL tool is
simply never selected by their scripted document-only responses).

### Known limitations

- **Not executed against the real `mcp`/`langgraph` packages in the
  environment this code was written in.** That sandbox has no network
  access and neither package (nor `pytest`) could be installed, so the
  new code could not be run through `pytest` there. What *could* be
  verified directly, with nothing but the Python standard library, was
  the database/validator logic itself — `sql_database.py` and
  `query_guard.py` were exercised standalone (database creation, schema,
  the deterministic highest-return figure, every forbidden statement
  type rejected, and a raw `INSERT` against the read-only connection
  actually raising `sqlite3.OperationalError`) and all behaved as
  documented above. The MCP-server and LangGraph-layer code
  (`sql_server.py`, the `mcp_client.py` additions, the
  `langgraph_agent.py` changes, and their tests) closely follows the
  exact patterns already proven working in Phases 1–4's own code and
  tests (dataclass return types, `raise ValueError(...)` for tool errors,
  `Client`/`list_tools()`/`call_tool()` usage, `FakeLLM` scripting) but
  has only been checked for syntax validity (`ast.parse`), not run end to
  end. Please run `PYTHONPATH=src pytest tests/ -v` after
  `pip install -r requirements.txt` before relying on this in the final
  local testing cycle mentioned in the prompt.
- `query_database`'s row cap (500) is a fixed constant, not configurable
  via environment variable (unlike the RAG knobs in `RAGConfig`).
- The query guard is a keyword allow-list, not a real SQL parser — it's
  intentionally conservative (e.g. it also blocks `BEGIN`/`COMMIT`/etc.,
  not just the keywords the spec listed) but, like any regex-based
  approach, is not a substitute for a proper SQL AST-based validator if
  the database ever stops being a small trusted local demo file.
- No connection pooling — `query_database()` opens and closes a fresh
  read-only SQLite connection per call, which is fine at this scale but
  would need revisiting under real concurrent load.
- Still SQLite only, per the Phase 5 prompt's explicit scope.

### What Phase 6 should implement

Per the Phase 5 prompt's "do not build yet" list: PostgreSQL (replacing
or complementing SQLite), Redis, external databases beyond the local
SQLite file, cross-turn conversation memory, human-in-the-loop approval
steps, a frontend, containerization (Docker), an evaluation framework,
multi-agent specialization (as opposed to one agent with multiple tool
families), and advanced SQL planning (e.g. query rewriting/optimization,
schema-aware few-shot prompting) beyond the single `query_database` tool
built here.
