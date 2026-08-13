"""
Read-Only SQL Query Guard (Phase 5)
=====================================

A small, dependency-free allow-list validator that stands between whatever
SQL an LLM decides to send and the actual SQLite connection.

This is deliberately the *first* of two independent layers of defense (the
second is opening the database itself with SQLite's ``mode=ro`` URI plus
``PRAGMA query_only = ON``, in ``sql_database.get_read_only_connection``).
Either layer alone should already stop a write; together they mean a
single bug in one layer is not enough to allow one through.

Rules enforced here:

- Exactly one statement. A trailing single ``;`` is allowed; anything after
  it (a second statement) is rejected outright -- this is what stops
  ``SELECT 1; DROP TABLE companies;``-style smuggling.
- The statement must start with ``SELECT`` or ``WITH`` (a read-only CTE
  that itself must resolve to a ``SELECT``).
- None of the mutating/DDL/administrative keywords below may appear
  anywhere in the statement, checked as whole words on the raw SQL, including comments (so ``/* DROP */`` or ``-- DELETE`` can't be used to hide
  intent from a human reviewer while still being rejected by this check).
"""

from __future__ import annotations

import re

# Statement-start keywords that are allowed to open a query.
_ALLOWED_START_KEYWORDS = {"select", "with"}

# Anywhere these appear (as a whole word), the query is rejected. This
# covers DML, DDL, and administrative statements, plus a few SQLite-specific
# escape hatches (ATTACH lets a query reach a second, non-read-only database
# file; PRAGMA can flip settings like query_only itself back off).
_FORBIDDEN_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "replace",
    "drop",
    "alter",
    "create",
    "truncate",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "reindex",
    "grant",
    "revoke",
    "begin",
    "commit",
    "rollback",
    "savepoint",
    "release",
}

_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class QueryValidationError(ValueError):
    """Raised when a SQL string fails the read-only allow-list check."""


def _strip_comments(sql: str) -> str:
    sql = _BLOCK_COMMENT_RE.sub(" ", sql)
    sql = _LINE_COMMENT_RE.sub(" ", sql)
    return sql


def validate_readonly_sql(sql: str) -> str:
    """
    Validate that `sql` is a single, read-only SELECT/CTE statement.

    Returns the original (unmodified) SQL string on success. Raises
    QueryValidationError with a human-readable reason on any violation --
    callers (the MCP tool) are expected to surface that message directly
    to the caller/LLM rather than executing anything.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise QueryValidationError("sql must be a non-empty string")

    cleaned = _strip_comments(sql).strip()
    if not cleaned:
        raise QueryValidationError("sql must not be empty or only comments")

    # Exactly one statement: at most one trailing semicolon, and nothing
    # but whitespace after it.
    body = cleaned
    if ";" in cleaned:
        first_semicolon = cleaned.index(";")
        after = cleaned[first_semicolon + 1 :].strip()
        if after:
            raise QueryValidationError(
                "only a single SQL statement is allowed (found content after ';')"
            )
        body = cleaned[:first_semicolon]

    words = _WORD_RE.findall(body)
    if not words:
        raise QueryValidationError("sql does not contain a recognizable statement")

    first_word = words[0].lower()
    if first_word not in _ALLOWED_START_KEYWORDS:
        raise QueryValidationError(
            f"only read-only SELECT/WITH statements are allowed (got a statement "
            f"starting with {words[0]!r})"
        )

    # Check forbidden keywords in the RAW SQL before stripping comments.
    # This prevents forbidden statements from being hidden inside comments.
    raw_words = _WORD_RE.findall(sql)
    raw_lowered_words = {w.lower() for w in raw_words}
    forbidden_found = sorted(raw_lowered_words & _FORBIDDEN_KEYWORDS)
    if forbidden_found:
        raise QueryValidationError(
            f"query contains forbidden keyword(s): {', '.join(forbidden_found)} "
            "-- only read-only SELECT queries are permitted"
        )

    return sql
