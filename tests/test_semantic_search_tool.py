"""
Tests for the semantic_search() MCP tool, exercised through an in-process
MCP client exactly like test_document_server.py.

The pipeline is injected via `document_server.set_pipeline()` using the
deterministic, offline "hashing" embedding provider over the real sample
documents in data/documents/ — so these tests never download a model or
touch the network, but still exercise the real ingestion -> chunking ->
embedding -> FAISS -> MCP path end to end.
"""

from __future__ import annotations

import pytest
from mcp import Client

from nexusai.mcp_servers import document_server
from nexusai.mcp_servers.document_server import DOCUMENTS_DIR, mcp as document_server_mcp
from nexusai.rag.config import RAGConfig
from nexusai.rag.pipeline import RAGPipeline


@pytest.fixture
def rag_pipeline(tmp_path):
    """Build a real (hashing-embedded) pipeline over the actual sample docs."""
    config = RAGConfig(
        documents_dir=DOCUMENTS_DIR,
        index_dir=tmp_path / "index",
        chunk_size=60,
        chunk_overlap=10,
        embedding_provider="hashing",
        embedding_dim=64,
        top_k=4,
    )
    pipeline = RAGPipeline.build(config)
    document_server.set_pipeline(pipeline)
    yield pipeline
    document_server.set_pipeline(None)  # don't leak state into other tests


@pytest.fixture
async def client(rag_pipeline):
    async with Client(document_server_mcp) as c:
        yield c


@pytest.mark.anyio
async def test_semantic_search_is_discoverable(client):
    tools = await client.list_tools()
    names = {t.name for t in tools.tools}
    assert "semantic_search" in names


@pytest.mark.anyio
async def test_semantic_search_finds_relevant_document(client):
    result = await client.call_tool("semantic_search", {"query": "retrieval augmented generation and embeddings", "top_k": 3})
    matches = result.structured_content["result"]

    assert len(matches) > 0
    assert any(m["filename"] == "rag_basics.txt" for m in matches)
    for m in matches:
        assert set(m.keys()) >= {"filename", "chunk_id", "text", "score"}


@pytest.mark.anyio
async def test_semantic_search_respects_top_k(client):
    result = await client.call_tool("semantic_search", {"query": "documents", "top_k": 1})
    matches = result.structured_content["result"]
    assert len(matches) <= 1


@pytest.mark.anyio
async def test_semantic_search_rejects_empty_query(client):
    result = await client.call_tool("semantic_search", {"query": "   ", "top_k": 3})
    assert result.is_error is True


@pytest.mark.anyio
async def test_semantic_search_rejects_non_positive_top_k(client):
    result = await client.call_tool("semantic_search", {"query": "RAG", "top_k": 0})
    assert result.is_error is True


@pytest.mark.anyio
async def test_semantic_search_results_are_grounded_in_real_text(client):
    """Every returned chunk's text must actually be a substring of some real
    document on disk — i.e. retrieval never fabricates content."""
    result = await client.call_tool("semantic_search", {"query": "MCP tools and servers", "top_k": 4})
    matches = result.structured_content["result"]

    # Chunking joins words with single spaces, so compare on normalized
    # whitespace rather than requiring an exact (including-newlines) match.
    all_text = " ".join(
        " ".join(p.read_text(encoding="utf-8").split()) for p in DOCUMENTS_DIR.glob("*.txt")
    )
    for m in matches:
        assert m["text"] in all_text


@pytest.fixture
def anyio_backend():
    return "asyncio"
