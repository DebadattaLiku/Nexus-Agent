"""
SQL Database (Phase 5)
========================

Builds and connects to a small, local, fully deterministic SQLite database
with a realistic finance/markets shape, used by the SQL MCP Server
(``mcp_servers/sql_server.py``) for read-only structured queries.

Schema
------

- ``companies`` — one row per ticker (ticker, name, sector).
- ``prices``    — one row per (ticker, trade_date) daily close price + volume.

Both tables are small enough to demonstrate meaningful aggregate queries
(e.g. "which company had the highest return over the period?") while
staying easy to reason about and fully deterministic — the seed data below
is a fixed literal, never randomly generated, so query results (and tests
that assert on them) never change between runs.

This module only creates/reads the database file; it has no MCP awareness
and no dependency on ``mcp_servers`` or the agent layer, so it can be
tested directly with nothing but the standard library (``sqlite3``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "nexus.db"

SCHEMA_SQL = """
CREATE TABLE companies (
    company_id  INTEGER PRIMARY KEY,
    ticker      TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    sector      TEXT NOT NULL
);

CREATE TABLE prices (
    price_id    INTEGER PRIMARY KEY,
    ticker      TEXT NOT NULL REFERENCES companies(ticker),
    trade_date  TEXT NOT NULL,
    close_price REAL NOT NULL,
    volume      INTEGER NOT NULL,
    UNIQUE(ticker, trade_date)
);
"""

# ---------------------------------------------------------------------------
# Deterministic seed data
# ---------------------------------------------------------------------------
#
# Five companies across four sectors, ten trading days each. Closing prices
# follow a fixed linear formula per ticker (not random) so every derived
# figure -- e.g. "highest return over the period" -- is a known constant,
# which is what makes this data suitable for deterministic tests as well as
# demo queries. TSLA is deliberately the highest-return ticker (+18%) and
# JPM the only decliner (-1.8%), to give the demo dataset a clear answer.

COMPANIES: list[tuple[str, str, str]] = [
    # ticker, name, sector
    ("AAPL", "Apple Inc.", "Technology"),
    ("MSFT", "Microsoft Corporation", "Technology"),
    ("JPM", "JPMorgan Chase & Co.", "Financials"),
    ("XOM", "Exxon Mobil Corporation", "Energy"),
    ("TSLA", "Tesla, Inc.", "Consumer Discretionary"),
]

# Ten business-day trading calendar (Jan 2 - Jan 15, 2024, weekdays only).
TRADE_DATES: list[str] = [
    "2024-01-02",
    "2024-01-03",
    "2024-01-04",
    "2024-01-05",
    "2024-01-08",
    "2024-01-09",
    "2024-01-10",
    "2024-01-11",
    "2024-01-12",
    "2024-01-15",
]

# (ticker, start_price, per_day_change, start_volume, per_day_volume_delta)
_PRICE_SERIES: list[tuple[str, float, float, int, int]] = [
    ("AAPL", 180.00, 1.00, 55_000_000, 250_000),
    ("MSFT", 300.00, 0.50, 28_000_000, 100_000),
    ("JPM", 150.00, -0.30, 12_000_000, 50_000),
    ("XOM", 100.00, 0.20, 18_000_000, 75_000),
    ("TSLA", 200.00, 4.00, 95_000_000, 500_000),
]


def _build_prices() -> list[tuple[str, str, float, int]]:
    """Generate the deterministic (ticker, trade_date, close_price, volume) rows."""
    rows: list[tuple[str, str, float, int]] = []
    for ticker, start_price, delta, start_volume, volume_delta in _PRICE_SERIES:
        for day_index, trade_date in enumerate(TRADE_DATES):
            close_price = round(start_price + delta * day_index, 2)
            volume = start_volume + volume_delta * day_index
            rows.append((ticker, trade_date, close_price, volume))
    return rows


PRICES: list[tuple[str, str, float, int]] = _build_prices()


# ---------------------------------------------------------------------------
# Database creation
# ---------------------------------------------------------------------------


def create_database(path: Path | str = DEFAULT_DB_PATH) -> Path:
    """
    Create (or deterministically rebuild) the SQLite database at `path`
    with the schema and seed data above.

    Always drops and recreates both tables first, so calling this twice
    yields byte-for-byte the same data -- there is no accumulation and no
    randomness. Safe to call from tests with a `tmp_path`-based path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # A plain read-write connection is used only here, for building the
    # database. Every other consumer (the SQL MCP tool, callers of
    # get_read_only_connection()) opens the file read-only -- see below.
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript("DROP TABLE IF EXISTS prices; DROP TABLE IF EXISTS companies;")
        conn.executescript(SCHEMA_SQL)
        conn.executemany(
            "INSERT INTO companies (ticker, name, sector) VALUES (?, ?, ?)",
            COMPANIES,
        )
        conn.executemany(
            "INSERT INTO prices (ticker, trade_date, close_price, volume) VALUES (?, ?, ?, ?)",
            PRICES,
        )
        conn.commit()
    finally:
        conn.close()

    return path


def ensure_database(path: Path | str = DEFAULT_DB_PATH) -> Path:
    """Build the database at `path` only if it doesn't already exist."""
    path = Path(path)
    if not path.exists():
        create_database(path)
    return path


def get_read_only_connection(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Open `path` as a read-only SQLite connection.

    Uses SQLite's URI ``mode=ro`` so the connection itself cannot write to
    the file at the OS/driver level, in addition to the SQL-level
    allow-list enforced by ``query_guard.validate_readonly_sql`` before any
    query reaches this connection. ``query_only = ON`` is set as a second,
    independent read-only guarantee at the SQLite-engine level.

    Raises FileNotFoundError with a clear message if the database file
    does not exist yet (callers should call ensure_database()/
    create_database() first).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Database file not found: {path}. Call ensure_database() or "
            "create_database() first."
        )

    uri = f"file:{path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only = ON;")
    conn.row_factory = sqlite3.Row
    return conn
