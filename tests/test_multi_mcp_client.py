"""
Tests for nexusai/client/mcp_client.py's Phase 5 multi-server aggregation:
`MultiMCPClient` and `connect_multi_in_process()`.

Uses the real, in-process Document MCP Server and SQL MCP Server (the SQL
server pointed at an isolated tmp_path database) -- no subprocess, no
network, no LLM involved. A couple of tests also use a minimal fake client
pair to exercise the duplicate-tool-name and unknown-tool-name edge cases
without needing to construct a real naming collision between the two real
servers.
"""

from __future__ import annotations

import pytest

from nexusai.client.mcp_client import MultiMCPClient, connect_multi_in_process
from nexusai.mcp_servers import sql_server


@pytest.fixture
def isolated_db(tmp_path):
    sql_server.set_db_path(tmp_path / "test_nexus.db")
    yield
    sql_server.set_db_path(None)


@pytest.mark.anyio
async def test_discovers_tools_from_both_servers(isolated_db):
    async with connect_multi_in_process() as client:
        discovered = await client.list_tools()
        names = {t.name for t in discovered.tools}

    assert {"list_documents", "get_document", "search_documents", "semantic_search"} <= names
    assert "query_database" in names


@pytest.mark.anyio
async def test_routes_document_call_to_document_server(isolated_db):
    async with connect_multi_in_process() as client:
        await client.list_tools()
        result = await client.call_tool("list_documents", {})

    assert result.is_error is False
    filenames = {doc["filename"] for doc in result.structured_content["result"]}
    assert "mcp_overview.txt" in filenames


@pytest.mark.anyio
async def test_routes_sql_call_to_sql_server(isolated_db):
    async with connect_multi_in_process() as client:
        await client.list_tools()
        result = await client.call_tool(
            "query_database", {"sql": "SELECT COUNT(*) AS n FROM companies"}
        )

    assert result.is_error is False
    assert result.structured_content["rows"][0]["n"] == 5


@pytest.mark.anyio
async def test_sequential_calls_across_both_servers(isolated_db):
    """A single conversation can call a document tool and then a SQL tool
    (or vice versa) against the one aggregated client -- exactly what a
    mixed document+database question needs."""
    async with connect_multi_in_process() as client:
        await client.list_tools()

        doc_result = await client.call_tool("search_documents", {"query": "RAG"})
        sql_result = await client.call_tool(
            "query_database", {"sql": "SELECT ticker FROM companies WHERE sector = 'Energy'"}
        )

    assert doc_result.is_error is False
    assert sql_result.is_error is False
    assert sql_result.structured_content["rows"][0]["ticker"] == "XOM"


@pytest.mark.anyio
async def test_call_tool_before_discovery_raises_unknown_tool(isolated_db):
    """call_tool() before list_tools() has ever populated the routing table
    must fail gracefully (unknown tool), not crash with an AttributeError."""
    async with connect_multi_in_process() as client:
        with pytest.raises(ValueError, match="Unknown tool"):
            await client.call_tool("query_database", {"sql": "SELECT 1"})


# ---------------------------------------------------------------------------
# MultiMCPClient in isolation, with minimal fake clients
# ---------------------------------------------------------------------------


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"fake {name}"
        self.input_schema = {"type": "object", "properties": {}}


class _FakeToolsResult:
    def __init__(self, tools):
        self.tools = tools


class _FakeClient:
    def __init__(self, tool_names, label):
        self._tools = [_FakeTool(n) for n in tool_names]
        self.label = label
        self.called_with: list[tuple[str, dict]] = []

    async def list_tools(self):
        return _FakeToolsResult(self._tools)

    async def call_tool(self, name, arguments):
        self.called_with.append((name, arguments))
        return f"result from {self.label}"


@pytest.mark.anyio
async def test_duplicate_tool_name_keeps_first_discovered():
    first = _FakeClient(["shared_tool"], "first")
    second = _FakeClient(["shared_tool"], "second")
    multi = MultiMCPClient({"first": first, "second": second})

    await multi.list_tools()
    result = await multi.call_tool("shared_tool", {})

    assert result == "result from first"
    assert first.called_with == [("shared_tool", {})]
    assert second.called_with == []


@pytest.mark.anyio
async def test_unknown_tool_name_raises_value_error():
    client_a = _FakeClient(["tool_a"], "a")
    multi = MultiMCPClient({"a": client_a})

    await multi.list_tools()
    with pytest.raises(ValueError, match="Unknown tool"):
        await multi.call_tool("does_not_exist", {})


@pytest.fixture
def anyio_backend():
    return "asyncio"
