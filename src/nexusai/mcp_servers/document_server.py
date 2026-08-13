"""
Document MCP Server (Phase 1 + Phase 2 substring search + Phase 3 semantic RAG)
=================================================================================

Exposes four tools over MCP for working with local text documents stored
under ``data/documents/``:

- ``list_documents()``     -> metadata for every document
- ``get_document()``       -> full text of one document, by filename
- ``search_documents()``   -> naive line-level substring search (Phase 1)
- ``semantic_search()``    -> real embedding + FAISS retrieval (Phase 3)

Why both ``search_documents`` and ``semantic_search`` exist
-------------------------------------------------------------
They solve different problems, so Phase 3 keeps both rather than replacing
one with the other:

- ``search_documents`` is exact/lexical: best when the caller knows a
  specific word, identifier, or short phrase that must literally appear
  (e.g. a config key, a function name, an exact term). It's cheap, has zero
  setup cost, and its results are trivially explainable (line + line
  number).
- ``semantic_search`` is conceptual: best for natural-language questions
  where the right document doesn't necessarily contain the query's exact
  words (e.g. "how does the agent avoid infinite tool-call loops?"). It
  requires the RAG index (embeddings + FAISS) described in
  ``nexusai/rag/``.

The agent (``agent/tool_agent.py``) decides which tool fits a given
question — this server just exposes both truthfully.

RAG retrieval itself lives entirely in ``nexusai/rag/`` (ingestion,
chunking, embeddings, vector store). This module only wires that pipeline
up as an MCP tool; the agent never imports ``nexusai.rag`` directly, only
this MCP surface — architecture stays:

    User -> LLM -> MCP Client -> Document MCP Server -> RAG retrieval
          -> result -> LLM -> answer

Run directly with:  python -m nexusai.mcp_servers.document_server
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from nexusai.rag.config import get_default_config
from nexusai.rag.pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Configuration & logging
# ---------------------------------------------------------------------------

# All documents live in <repo_root>/data/documents/. We resolve this relative
# to the repo root (three levels up from this file: mcp_servers -> nexusai ->
# src -> repo root) so the server works regardless of the current working
# directory it's launched from.
REPO_ROOT = Path(__file__).resolve().parents[3]
DOCUMENTS_DIR = REPO_ROOT / "data" / "documents"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nexusai.document_server")

mcp = MCPServer(
    name="nexusai-document-server",
    version="0.2.0",
    instructions=(
        "Provides read-only access to local text documents: list them, "
        "fetch one by filename, search across all of them by exact keyword, "
        "or run a semantic (meaning-based) search over embedded chunks."
    ),
)

# ---------------------------------------------------------------------------
# RAG pipeline (lazy singleton)
# ---------------------------------------------------------------------------
#
# Building the index the first time requires ingesting + chunking +
# embedding every document; after that, RAGPipeline.load_or_build() persists
# it to disk (nexusai/rag/pipeline.py) so restarting this server does not
# re-embed anything unless the documents or RAG config actually changed.
#
# `_pipeline` is created lazily (on first semantic_search() call) rather
# than at import time, so simply importing this module — e.g. to discover
# tools, or to exercise list_documents()/get_document()/search_documents()
# in a test — never requires an embedding model or FAISS index to exist.
# Tests that need a specific pipeline (e.g. a deterministic offline
# embedding provider) can call `set_pipeline()` to inject one directly.

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Return the process-wide RAG pipeline, building/loading it on first use."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline.load_or_build(get_default_config())
    return _pipeline


def set_pipeline(pipeline: RAGPipeline | None) -> None:
    """Inject (or clear, with None) the RAG pipeline. Used by tests."""
    global _pipeline
    _pipeline = pipeline


# ---------------------------------------------------------------------------
# Data models (define the structured output schema for each tool)
# ---------------------------------------------------------------------------


@dataclass
class DocumentMeta:
    """Lightweight metadata about a document, used by list_documents()."""

    filename: str
    size_bytes: int
    num_lines: int


@dataclass
class DocumentContent:
    """Full content of a single document, used by get_document()."""

    filename: str
    content: str


@dataclass
class SearchMatch:
    """One matching line from search_documents()."""

    filename: str
    line_number: int
    line: str


@dataclass
class SemanticMatch:
    """One retrieved chunk from semantic_search(), with its similarity score."""

    filename: str
    chunk_id: str
    text: str
    score: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_safe_path(filename: str) -> Path:
    """
    Resolve `filename` inside DOCUMENTS_DIR only, rejecting any attempt to
    escape the directory (e.g. via '../'). Raises ValueError if unsafe or
    if the file does not exist.
    """
    candidate = (DOCUMENTS_DIR / filename).resolve()
    if DOCUMENTS_DIR.resolve() not in candidate.parents and candidate != DOCUMENTS_DIR.resolve():
        raise ValueError(f"Refusing to access path outside documents directory: {filename!r}")
    if not candidate.exists():
        raise ValueError(f"Document not found: {filename!r}")
    if not candidate.is_file():
        raise ValueError(f"Not a file: {filename!r}")
    return candidate


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_documents() -> list[DocumentMeta]:
    """List every document available in the documents directory, with basic metadata."""
    logger.info("list_documents called")

    if not DOCUMENTS_DIR.exists():
        logger.warning("Documents directory does not exist: %s", DOCUMENTS_DIR)
        return []

    results: list[DocumentMeta] = []
    for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read %s: %s", path.name, exc)
            continue
        results.append(
            DocumentMeta(
                filename=path.name,
                size_bytes=path.stat().st_size,
                num_lines=text.count("\n") + 1,
            )
        )

    logger.info("list_documents returning %d document(s)", len(results))
    return results


@mcp.tool()
def get_document(filename: str) -> DocumentContent:
    """
    Fetch the full text content of one document.

    Call list_documents() first if you are not certain of the exact
    filename — do not guess it. Filenames are case-sensitive and must match
    exactly what list_documents() returns.

    Args:
        filename: Exact name of the document file, e.g. "mcp_overview.txt",
                  as returned by list_documents(). Must live directly inside
                  the documents directory.
    """
    logger.info("get_document called with filename=%r", filename)

    try:
        path = _resolve_safe_path(filename)
    except ValueError as exc:
        logger.warning("get_document rejected: %s", exc)
        raise

    content = path.read_text(encoding="utf-8")
    return DocumentContent(filename=path.name, content=content)


@mcp.tool()
def search_documents(query: str, case_sensitive: bool = False) -> list[SearchMatch]:
    """
    Search all documents for lines containing `query` (simple substring match,
    no embeddings/ranking yet — that comes in a later phase).

    Args:
        query: Text to search for.
        case_sensitive: Whether the match should respect letter case.
    """
    logger.info("search_documents called with query=%r case_sensitive=%s", query, case_sensitive)

    if not query.strip():
        raise ValueError("query must not be empty")

    if not DOCUMENTS_DIR.exists():
        logger.warning("Documents directory does not exist: %s", DOCUMENTS_DIR)
        return []

    needle = query if case_sensitive else query.lower()
    matches: list[SearchMatch] = []

    for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.warning("Could not read %s: %s", path.name, exc)
            continue

        for i, line in enumerate(lines, start=1):
            haystack = line if case_sensitive else line.lower()
            if needle in haystack:
                matches.append(SearchMatch(filename=path.name, line_number=i, line=line.strip()))

    logger.info("search_documents found %d match(es)", len(matches))
    return matches


@mcp.tool()
def semantic_search(query: str, top_k: int = 4) -> list[SemanticMatch]:
    """
    Semantic (meaning-based) search over the document library.

    Use this for natural-language / conceptual questions, where the answer
    may not use the same words as the question. Returns the `top_k` most
    relevant chunks, each with its source filename and a similarity score
    (higher is more relevant), so answers can be grounded and attributed.

    For an exact word/phrase/identifier lookup, prefer search_documents()
    instead — it is cheaper and its matches are exact.

    Args:
        query: Natural-language question or topic to search for.
        top_k: Maximum number of chunks to return (must be a positive
               integer; capped internally at a sane maximum).
    """
    logger.info("semantic_search called with query=%r top_k=%r", query, top_k)

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    pipeline = get_pipeline()
    results = pipeline.retrieve(query, top_k=top_k)

    matches = [
        SemanticMatch(filename=r.filename, chunk_id=r.chunk_id, text=r.text, score=r.score)
        for r in results
    ]
    logger.info("semantic_search returning %d chunk(s)", len(matches))
    return matches


# ---------------------------------------------------------------------------
# Entry point — runs the server over stdio, the standard transport for a
# locally-spawned MCP server talked to by a client subprocess.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Document MCP Server (documents dir: %s)", DOCUMENTS_DIR)
    mcp.run()
