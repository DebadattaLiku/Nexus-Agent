"""
SQL MCP Server (Phase 5)
===========================

Exposes one tool over MCP for safe, read-only access to a local SQLite
finance/markets demo database (``companies`` + ``prices``, see
``nexusai/db/sql_database.py``):

- ``query_database(sql)`` -> run a read-only SELECT/CTE query, return
  structured rows.

Security
--------

Every query passes through two independent read-only layers before a
single row comes back:

1. ``nexusai.db.query_guard.validate_readonly_sql`` -- an allow-list check
   (single statement, must start with SELECT/WITH, no INSERT/UPDATE/
   DELETE/DROP/ALTER/CREATE/ATTACH/PRAGMA/etc. keywords anywhere in it).
2. ``nexusai.db.sql_database.get_read_only_connection`` -- the SQLite
   connection itself is opened with the ``mode=ro`` URI flag and
   ``PRAGMA query_only = ON``, so even a query that somehow passed layer 1
   still cannot write at the database-engine level.

Invalid SQL (syntax errors) and queries against tables/columns that don't
exist are both caught and turned into a normal MCP tool error (via
``raise ValueError(...)``, the same convention ``document_server.py``
uses for ``get_document`` on a missing file) rather than an unhandled
exception.

This module only wires ``nexusai.db`` up as an MCP tool; the agent never
imports ``nexusai.db`` directly, only this MCP surface -- architecture
stays:

    User -> LLM -> MCP Client -> SQL MCP Server -> nexusai.db (SQLite)

Run directly with:  python -m nexusai.mcp_servers.sql_server
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from nexusai.db.query_guard import QueryValidationError, validate_readonly_sql
from nexusai.db.sql_database import DEFAULT_DB_PATH, ensure_database, get_read_only_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nexusai.sql_server")

mcp = MCPServer(
    name="nexusai-sql-server",
    version="0.1.0",
    instructions=(
        "Provides safe, read-only SQL access to a local finance/markets "
        "database with two tables: companies(ticker, name, sector) and "
        "prices(ticker, trade_date, close_price, volume). Use this for "
        "structured/aggregate questions about companies, prices, sectors, "
        "or returns -- not for document/text content, which lives in the "
        "Document MCP server instead. Only read-only SELECT/WITH queries "
        "are accepted; any write, DDL, or administrative statement is "
        "rejected."
    ),
)

# Cap the number of rows a single query can return, so a broad/unbounded
# SELECT can't flood the LLM's context window. Generous relative to this
# demo dataset's actual size (50 price rows total) but still a real bound.
_MAX_ROWS = 500

# ---------------------------------------------------------------------------
# Database path (lazy singleton, mirrors document_server.get_pipeline())
# ---------------------------------------------------------------------------
#
# The database is built on first use rather than at import time, so simply
# importing this module (e.g. to discover tools) never touches the
# filesystem. Tests can point the server at an isolated tmp_path database
# via set_db_path(), the same pattern document_server.py uses for
# set_pipeline().

_db_path: Path = DEFAULT_DB_PATH


def get_db_path() -> Path:
    """Return the SQLite database path this server is configured to use,
    building the database at that path first if it doesn't exist yet."""
    ensure_database(_db_path)
    return _db_path


def set_db_path(path: Path | str | None) -> None:
    """Point the server at a different database file (or, with None,
    reset to the default). Used by tests to run against an isolated,
    disposable database."""
    global _db_path
    _db_path = Path(path) if path is not None else DEFAULT_DB_PATH


# ---------------------------------------------------------------------------
# Data model (defines the structured output schema for query_database())
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Structured result of one query_database() call."""

    columns: list[str]
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@mcp.tool()
def query_database(sql: str) -> QueryResult:
    """
    Run a read-only SQL query against the local finance database and
    return structured rows.

    Only SELECT (or WITH ... SELECT) statements are accepted. Any write,
    schema-modifying, or administrative statement (INSERT, UPDATE, DELETE,
    DROP, ALTER, CREATE, ATTACH, PRAGMA, etc.) is rejected before it ever
    reaches the database.

    Schema:
      - companies(company_id, ticker, name, sector)
      - prices(price_id, ticker, trade_date, close_price, volume)
        -- prices.ticker references companies.ticker

    Args:
        sql: A single read-only SQL SELECT/WITH statement, e.g.
             "SELECT ticker, close_price FROM prices WHERE trade_date = '2024-01-15'".
    """
    logger.info("query_database called with sql=%r", sql)

    try:
        validate_readonly_sql(sql)
    except QueryValidationError as exc:
        logger.warning("query_database rejected by validator: %s", exc)
        raise ValueError(f"Query rejected: {exc}") from exc

    db_path = get_db_path()

    try:
        conn = get_read_only_connection(db_path)
    except FileNotFoundError as exc:
        logger.error("SQL database missing: %s", exc)
        raise ValueError(f"Database unavailable: {exc}") from exc

    try:
        try:
            cursor = conn.execute(sql)
            fetched = cursor.fetchmany(_MAX_ROWS + 1)
        except sqlite3.Error as exc:
            logger.warning("query_database execution failed: %s", exc)
            raise ValueError(f"Invalid SQL query: {exc}") from exc

        columns = [description[0] for description in (cursor.description or [])]
        truncated = len(fetched) > _MAX_ROWS
        rows = [dict(row) for row in fetched[:_MAX_ROWS]]

        result = QueryResult(columns=columns, rows=rows, row_count=len(rows), truncated=truncated)
        logger.info("query_database returning %d row(s) (truncated=%s)", result.row_count, truncated)
        return result
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entry point — runs the server over stdio, the standard transport for a
# locally-spawned MCP server talked to by a client subprocess.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting SQL MCP Server (database: %s)", get_db_path())
    mcp.run()
