"""
Tests for the Phase 5 finance demo database (nexusai/db/sql_database.py).

Pure sqlite3/stdlib -- no MCP client/server involved, so these exercise
database creation, schema, and seed data in isolation from the MCP layer.
"""

from __future__ import annotations

import sqlite3

import pytest

from nexusai.db.sql_database import (
    COMPANIES,
    PRICES,
    TRADE_DATES,
    create_database,
    ensure_database,
    get_read_only_connection,
)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_nexus.db"


# ---------------------------------------------------------------------------
# 1. Database creation
# ---------------------------------------------------------------------------


def test_create_database_creates_file(db_path):
    assert not db_path.exists()
    create_database(db_path)
    assert db_path.exists()


def test_create_database_is_idempotent(db_path):
    """Calling create_database() twice must yield byte-for-byte identical data."""
    create_database(db_path)
    conn = get_read_only_connection(db_path)
    first = conn.execute("SELECT * FROM prices ORDER BY ticker, trade_date").fetchall()
    conn.close()

    create_database(db_path)
    conn = get_read_only_connection(db_path)
    second = conn.execute("SELECT * FROM prices ORDER BY ticker, trade_date").fetchall()
    conn.close()

    assert [tuple(r) for r in first] == [tuple(r) for r in second]


def test_ensure_database_builds_only_if_missing(db_path):
    assert not db_path.exists()
    ensure_database(db_path)
    assert db_path.exists()
    mtime_after_first = db_path.stat().st_mtime_ns

    ensure_database(db_path)  # should not rebuild
    assert db_path.stat().st_mtime_ns == mtime_after_first


# ---------------------------------------------------------------------------
# 2. Schema
# ---------------------------------------------------------------------------


def test_schema_has_expected_tables(db_path):
    create_database(db_path)
    conn = get_read_only_connection(db_path)
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    conn.close()
    assert {"companies", "prices"} <= tables


def test_companies_schema_columns(db_path):
    create_database(db_path)
    conn = get_read_only_connection(db_path)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(companies)")}
    conn.close()
    assert columns == {"company_id", "ticker", "name", "sector"}


def test_prices_schema_columns(db_path):
    create_database(db_path)
    conn = get_read_only_connection(db_path)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(prices)")}
    conn.close()
    assert columns == {"price_id", "ticker", "trade_date", "close_price", "volume"}


def test_prices_ticker_references_known_company(db_path):
    create_database(db_path)
    conn = get_read_only_connection(db_path)
    tickers = {row["ticker"] for row in conn.execute("SELECT DISTINCT ticker FROM companies")}
    price_tickers = {row["ticker"] for row in conn.execute("SELECT DISTINCT ticker FROM prices")}
    conn.close()
    assert price_tickers <= tickers


# ---------------------------------------------------------------------------
# 3. Sample data
# ---------------------------------------------------------------------------


def test_seed_data_row_counts(db_path):
    create_database(db_path)
    conn = get_read_only_connection(db_path)
    n_companies = conn.execute("SELECT COUNT(*) AS n FROM companies").fetchone()["n"]
    n_prices = conn.execute("SELECT COUNT(*) AS n FROM prices").fetchone()["n"]
    conn.close()
    assert n_companies == len(COMPANIES)
    assert n_prices == len(PRICES)
    assert n_prices == len(COMPANIES) * len(TRADE_DATES)


def test_seed_data_is_deterministic_highest_return():
    """The seed data is a fixed formula, not random -- TSLA is always the
    highest-return ticker over the seeded period, by construction."""
    by_ticker: dict[str, list[float]] = {}
    for ticker, _trade_date, close_price, _volume in PRICES:
        by_ticker.setdefault(ticker, []).append(close_price)

    returns = {
        ticker: (prices[-1] - prices[0]) / prices[0] for ticker, prices in by_ticker.items()
    }
    assert max(returns, key=returns.get) == "TSLA"


def test_get_read_only_connection_missing_file_raises(db_path):
    with pytest.raises(FileNotFoundError):
        get_read_only_connection(db_path)


def test_read_only_connection_rejects_writes(db_path):
    create_database(db_path)
    conn = get_read_only_connection(db_path)
    with pytest.raises(sqlite3.Error):
        conn.execute("INSERT INTO companies (ticker, name, sector) VALUES ('X', 'X', 'X')")
    conn.close()
