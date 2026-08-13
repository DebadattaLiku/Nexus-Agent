"""
MCP Client (Phase 1, extended in Phase 5 for multi-server discovery)
=======================================================================

A small async client that connects to MCP servers, discovers their tools,
and calls them.

Two ways to connect to any single server:

1. In-process (default, used by run_demo() / tests):
   Import the server object directly and hand it to `Client`. The SDK then
   talks to it over an in-memory transport — no subprocess, no stdio
   plumbing. This is the recommended way to test an MCP server in Python.

2. Real stdio subprocess (what a separate host process would do):
   See `connect_over_stdio()` / `connect_over_stdio_sql()` below, which
   spawn the server module as a child process and speak MCP to it over
   stdin/stdout, exactly as an external MCP host (e.g. Claude Desktop)
   would.

Phase 5 adds a second server (the SQL MCP Server) and `MultiMCPClient` /
`connect_multi_in_process()`, which aggregate tool discovery and routing
across both servers behind the same `list_tools()` / `call_tool()`
interface a single `Client` exposes — so agent code (`langgraph_agent.py`)
never needs to know how many servers, or which one, a given tool call
actually reaches. This is the client-side half of "dynamic MCP tool
selection": the LLM picks a tool by name/schema, and `MultiMCPClient`
is what actually knows which server that name belongs to.

Run directly with:  python -m nexusai.client.mcp_client
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nexusai.mcp_client")

REPO_ROOT = Path(__file__).resolve().parents[3]


@asynccontextmanager
async def connect_in_process():
    """Connect to the Document MCP Server in-process (no subprocess)."""
    from nexusai.mcp_servers.document_server import mcp as document_server

    async with Client(document_server) as client:
        yield client


@asynccontextmanager
async def connect_over_stdio():
    """Connect to the Document MCP Server as a real child process over stdio."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "nexusai.mcp_servers.document_server"],
        cwd=str(REPO_ROOT / "src"),
    )
    async with Client(stdio_client(params)) as client:
        yield client


@asynccontextmanager
async def connect_sql_in_process():
    """Connect to the SQL MCP Server in-process (no subprocess)."""
    from nexusai.mcp_servers.sql_server import mcp as sql_server

    async with Client(sql_server) as client:
        yield client


@asynccontextmanager
async def connect_over_stdio_sql():
    """Connect to the SQL MCP Server as a real child process over stdio."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "nexusai.mcp_servers.sql_server"],
        cwd=str(REPO_ROOT / "src"),
    )
    async with Client(stdio_client(params)) as client:
        yield client


# ---------------------------------------------------------------------------
# Multi-server aggregation (Phase 5)
# ---------------------------------------------------------------------------


class MultiMCPClient:
    """
    Aggregates tool discovery and call routing across several already-
    connected MCP clients, presenting the same `list_tools()` / `call_tool()`
    interface as a single `mcp.Client` so callers (e.g. the LangGraph agent)
    don't need to know or care how many underlying servers exist.

    `list_tools()` merges the tool lists from every underlying client and
    records which client each tool name came from; `call_tool()` looks up
    that mapping and forwards the call to the right server. If two servers
    ever expose the same tool name, the first one discovered (in
    `clients` iteration order) wins and a warning is logged — tool names
    are expected to be unique across servers in this project, so this is a
    defensive fallback, not a supported configuration.
    """

    def __init__(self, clients: dict[str, Any]) -> None:
        self._clients = clients
        self._tool_owner: dict[str, Any] = {}

    async def list_tools(self) -> SimpleNamespace:
        all_tools: list[Any] = []
        owner: dict[str, Any] = {}

        for label, client in self._clients.items():
            discovered = await client.list_tools()
            for tool in discovered.tools:
                if tool.name in owner:
                    logger.warning(
                        "Duplicate tool name %r discovered from multiple MCP "
                        "servers; keeping the first one found, ignoring %s",
                        tool.name,
                        label,
                    )
                    continue
                owner[tool.name] = client
                all_tools.append(tool)

        self._tool_owner = owner
        return SimpleNamespace(tools=all_tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        client = self._tool_owner.get(name)
        if client is None:
            raise ValueError(
                f"Unknown tool (not discovered from any connected MCP server): {name!r}"
            )
        return await client.call_tool(name, arguments)


@asynccontextmanager
async def connect_multi_in_process():
    """
    Connect to both the Document MCP Server and the SQL MCP Server
    in-process, exposing them as a single aggregated `MultiMCPClient`.

    This is the connection the LangGraph agent uses in production: it lets
    the agent discover tools from both servers with one `list_tools()`
    call and dynamically choose between them per question, without ever
    importing either server's implementation directly.
    """
    async with AsyncExitStack() as stack:
        document_client = await stack.enter_async_context(connect_in_process())
        sql_client = await stack.enter_async_context(connect_sql_in_process())
        yield MultiMCPClient({"document": document_client, "sql": sql_client})


async def run_demo(use_stdio: bool = False) -> None:
    """Discover and exercise every tool on the Document MCP Server."""
    connector = connect_over_stdio if use_stdio else connect_in_process
    logger.info("Connecting to Document MCP Server (stdio=%s)...", use_stdio)

    async with connector() as client:
        # 1. Discover tools
        tools = await client.list_tools()
        print("\n=== Available tools ===")
        for tool in tools.tools:
            print(f"- {tool.name}: {tool.description}")

        # 2. list_documents()
        print("\n=== list_documents() ===")
        result = await client.call_tool("list_documents", {})
        for doc in result.structured_content["result"]:
            print(f"- {doc['filename']} ({doc['size_bytes']} bytes, {doc['num_lines']} lines)")

        # 3. get_document() on the first document found
        docs = result.structured_content["result"]
        if docs:
            filename = docs[0]["filename"]
            print(f"\n=== get_document(filename={filename!r}) ===")
            result = await client.call_tool("get_document", {"filename": filename})
            # A single-object tool result is returned flattened (not wrapped
            # in {"result": ...}) — only list/collection results are wrapped.
            content = result.structured_content["content"]
            preview = content[:200] + ("..." if len(content) > 200 else "")
            print(preview)

        # 4. get_document() on a missing file, to show graceful error handling
        print("\n=== get_document(filename='does_not_exist.txt') ===")
        result = await client.call_tool("get_document", {"filename": "does_not_exist.txt"})
        if result.is_error:
            print(f"Handled expected error: {result.content[0].text}")

        # 5. search_documents()
        query = "RAG"
        print(f"\n=== search_documents(query={query!r}) ===")
        result = await client.call_tool("search_documents", {"query": query})
        for match in result.structured_content["result"]:
            print(f"- {match['filename']}:{match['line_number']}: {match['line']}")


if __name__ == "__main__":
    use_stdio = "--stdio" in sys.argv
    asyncio.run(run_demo(use_stdio=use_stdio))
