"""
Tests for the Phase 5 SQL MCP Server, run through an in-process MCP client
(no subprocess needed) -- mirrors tests/test_document_server.py.

Each test runs against an isolated tmp_path database (via set_db_path()),
never the real data/nexus.db file, so tests never interfere with each
other or with manual runs of the server.
"""

from __future__ import annotations

import pytest
from mcp import Client

from nexusai.mcp_servers import sql_server
from nexusai.mcp_servers.sql_server import mcp as sql_mcp_server


@pytest.fixture
def isolated_db(tmp_path):
    """Point the SQL server at a fresh, isolated database for this test."""
    sql_server.set_db_path(tmp_path / "test_nexus.db")
    yield
    sql_server.set_db_path(None)


@pytest.fixture
async def client(isolated_db):
    async with Client(sql_mcp_server) as c:
        yield c


@pytest.mark.anyio
async def test_list_tools_exposes_query_database(client):
    tools = await client.list_tools()
    names = {t.name for t in tools.tools}
    assert names == {"query_database"}


@pytest.mark.anyio
async def test_tool_description_mentions_schema(client):
    tools = await client.list_tools()
    tool = tools.tools[0]
    assert "companies" in tool.description
    assert "prices" in tool.description


@pytest.mark.anyio
async def test_valid_select_returns_structured_rows(client):
    result = await client.call_tool(
        "query_database", {"sql": "SELECT ticker, sector FROM companies ORDER BY ticker"}
    )
    assert result.is_error is False
    payload = result.structured_content
    assert payload["row_count"] == 5
    tickers = [row["ticker"] for row in payload["rows"]]
    assert tickers == sorted(tickers)
    assert "AAPL" in tickers


@pytest.mark.anyio
async def test_highest_return_query(client):
    """The example query from the Phase 5 spec: 'which company had the
    highest return' -- proven against the real, deterministic seed data."""
    sql = """
        WITH bounds AS (
            SELECT ticker, MIN(trade_date) AS d0, MAX(trade_date) AS d1
            FROM prices GROUP BY ticker
        )
        SELECT p0.ticker,
               (p1.close_price - p0.close_price) / p0.close_price * 100.0 AS return_pct
        FROM bounds b
        JOIN prices p0 ON p0.ticker = b.ticker AND p0.trade_date = b.d0
        JOIN prices p1 ON p1.ticker = b.ticker AND p1.trade_date = b.d1
        ORDER BY return_pct DESC
        LIMIT 1
    """
    result = await client.call_tool("query_database", {"sql": sql})
    assert result.is_error is False
    rows = result.structured_content["rows"]
    assert rows[0]["ticker"] == "TSLA"


@pytest.mark.anyio
async def test_invalid_sql_syntax_is_graceful(client):
    result = await client.call_tool("query_database", {"sql": "SELEKT * FROM companies"})
    assert result.is_error is True


@pytest.mark.anyio
async def test_missing_table_is_graceful(client):
    result = await client.call_tool("query_database", {"sql": "SELECT * FROM not_a_real_table"})
    assert result.is_error is True
    assert "no such table" in result.content[0].text.lower()


@pytest.mark.anyio
async def test_missing_column_is_graceful(client):
    result = await client.call_tool(
        "query_database", {"sql": "SELECT not_a_real_column FROM companies"}
    )
    assert result.is_error is True


@pytest.mark.anyio
async def test_empty_sql_is_rejected(client):
    result = await client.call_tool("query_database", {"sql": "   "})
    assert result.is_error is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO companies (ticker, name, sector) VALUES ('X', 'X', 'X')",
        "UPDATE companies SET name = 'X' WHERE ticker = 'AAPL'",
        "DELETE FROM companies",
        "DROP TABLE companies",
        "ALTER TABLE companies ADD COLUMN x TEXT",
        "CREATE TABLE evil (id INTEGER)",
        "ATTACH DATABASE 'other.db' AS other",
        "PRAGMA query_only = OFF",
        "SELECT 1; DROP TABLE companies;",
    ],
)
async def test_forbidden_statements_are_rejected(client, sql):
    result = await client.call_tool("query_database", {"sql": sql})
    assert result.is_error is True

    # And prove the rejection actually protected the data -- companies must
    # still have exactly its original 5 rows.
    check = await client.call_tool("query_database", {"sql": "SELECT COUNT(*) AS n FROM companies"})
    assert check.structured_content["rows"][0]["n"] == 5


@pytest.fixture
def anyio_backend():
    return "asyncio"
