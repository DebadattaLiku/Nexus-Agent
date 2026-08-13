"""
Tests for nexusai/db/query_guard.py -- the allow-list SQL validator that
sits in front of every query_database() call.
"""

from __future__ import annotations

import pytest

from nexusai.db.query_guard import QueryValidationError, validate_readonly_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM companies",
        "select ticker from companies where sector = 'Technology'",
        "SELECT * FROM companies;",  # single trailing semicolon is fine
        "WITH t AS (SELECT * FROM companies) SELECT * FROM t",
        "  SELECT 1  ",  # leading/trailing whitespace
        "-- a comment\nSELECT * FROM companies",
    ],
)
def test_valid_select_and_with_statements_pass(sql):
    assert validate_readonly_sql(sql) == sql


@pytest.mark.parametrize(
    "sql,expected_keyword",
    [
        ("INSERT INTO companies (ticker) VALUES ('X')", "insert"),
        ("UPDATE companies SET name = 'X'", "update"),
        ("DELETE FROM companies", "delete"),
        ("DROP TABLE companies", "drop"),
        ("ALTER TABLE companies ADD COLUMN x TEXT", "alter"),
        ("CREATE TABLE evil (id INTEGER)", "create"),
    ],
)
def test_forbidden_statement_types_rejected(sql, expected_keyword):
    with pytest.raises(QueryValidationError):
        validate_readonly_sql(sql)


def test_attach_rejected():
    with pytest.raises(QueryValidationError):
        validate_readonly_sql("ATTACH DATABASE 'other.db' AS other")


def test_pragma_rejected():
    with pytest.raises(QueryValidationError):
        validate_readonly_sql("PRAGMA table_info(companies)")


def test_multiple_statements_rejected():
    with pytest.raises(QueryValidationError):
        validate_readonly_sql("SELECT 1; DROP TABLE companies;")


def test_forbidden_keyword_hidden_in_comment_still_rejected():
    """Even if a forbidden keyword only appears inside a comment, reject it --
    comments are stripped before the keyword scan, on the assumption that a
    model emitting one is not being purely decorative."""
    with pytest.raises(QueryValidationError):
        validate_readonly_sql("SELECT * FROM companies /* DROP TABLE companies */")


def test_empty_query_rejected():
    with pytest.raises(QueryValidationError):
        validate_readonly_sql("")


def test_whitespace_only_query_rejected():
    with pytest.raises(QueryValidationError):
        validate_readonly_sql("   \n\t  ")


def test_non_select_start_rejected():
    with pytest.raises(QueryValidationError):
        validate_readonly_sql("EXPLAIN SELECT * FROM companies")
