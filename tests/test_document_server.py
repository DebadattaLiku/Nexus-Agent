"""
Tests for the Phase 1 Document MCP Server, run through an in-process MCP
client (no subprocess needed).
"""

from __future__ import annotations

import pytest
from mcp import Client

from nexusai.mcp_servers.document_server import mcp as document_server


@pytest.fixture
async def client():
    async with Client(document_server) as c:
        yield c


@pytest.mark.anyio
async def test_list_tools_exposes_all_four(client):
    tools = await client.list_tools()
    names = {t.name for t in tools.tools}
    assert names == {"list_documents", "get_document", "search_documents", "semantic_search"}


@pytest.mark.anyio
async def test_list_documents_returns_known_files(client):
    result = await client.call_tool("list_documents", {})
    filenames = {doc["filename"] for doc in result.structured_content["result"]}
    assert "mcp_overview.txt" in filenames
    assert "rag_basics.txt" in filenames
    assert "langgraph_notes.txt" in filenames


@pytest.mark.anyio
async def test_get_document_returns_content(client):
    result = await client.call_tool("get_document", {"filename": "mcp_overview.txt"})
    assert "Model Context Protocol" in result.structured_content["content"]


@pytest.mark.anyio
async def test_get_document_missing_file_is_graceful(client):
    result = await client.call_tool("get_document", {"filename": "nope.txt"})
    assert result.is_error is True
    assert "not found" in result.content[0].text.lower()


@pytest.mark.anyio
async def test_get_document_rejects_path_traversal(client):
    result = await client.call_tool("get_document", {"filename": "../../etc/passwd"})
    assert result.is_error is True


@pytest.mark.anyio
async def test_search_documents_finds_expected_match(client):
    result = await client.call_tool("search_documents", {"query": "LangGraph"})
    matches = result.structured_content["result"]
    assert any(m["filename"] == "langgraph_notes.txt" for m in matches)


@pytest.mark.anyio
async def test_search_documents_case_insensitive_by_default(client):
    result = await client.call_tool("search_documents", {"query": "langgraph"})
    matches = result.structured_content["result"]
    assert len(matches) > 0


@pytest.mark.anyio
async def test_search_documents_rejects_empty_query(client):
    result = await client.call_tool("search_documents", {"query": "   "})
    assert result.is_error is True


@pytest.fixture
def anyio_backend():
    return "asyncio"
