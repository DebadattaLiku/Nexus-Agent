"""
Tests for the Phase 5 LangGraph agent's *production* entry point
(`LangGraphAgent.answer()`), now wired to `connect_multi_in_process()` so a
single run can dynamically pick tools from the Document MCP server, the
SQL MCP server, or both.

Mirrors the real-server tests in tests/test_langgraph_agent.py (a scripted
FakeLLM against the real, in-process MCP servers) rather than the
fake-MCP-client error-injection tests -- those still apply unchanged to
`build_graph()` directly and are not duplicated here.
"""

from __future__ import annotations

import pytest

from nexusai.agent.langgraph_agent import LangGraphAgent
from nexusai.llm.provider import LLMResponse, ToolCall
from nexusai.mcp_servers import sql_server


@pytest.fixture
def isolated_sql_db(tmp_path):
    sql_server.set_db_path(tmp_path / "test_nexus.db")
    yield
    sql_server.set_db_path(None)


class FakeLLM:
    """Replays a scripted list of LLMResponse objects, one per chat() call."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, system=None):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        if not self._responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self._responses.pop(0)


def _tool_call_response(name: str, arguments: dict, call_id: str = "call_1") -> LLMResponse:
    call = ToolCall(id=call_id, name=name, arguments=arguments)
    return LLMResponse(
        text=None,
        tool_calls=[call],
        stop_reason="tool_use",
        raw_content=[{"type": "tool_use", "id": call_id, "name": name, "input": arguments}],
    )


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        text=text, tool_calls=[], stop_reason="end_turn", raw_content=[{"type": "text", "text": text}]
    )


# ---------------------------------------------------------------------------
# Tool discovery includes both servers
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_agent_discovers_tools_from_both_servers(isolated_sql_db):
    fake = FakeLLM([_text_response("hi")])
    agent = LangGraphAgent(llm=fake)

    await agent.answer("hello")

    tool_names = {t["name"] for t in fake.calls[0]["tools"]}
    assert {"list_documents", "search_documents", "semantic_search"} <= tool_names
    assert "query_database" in tool_names


# ---------------------------------------------------------------------------
# Routing to SQL MCP for a database question
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_routes_database_question_to_sql_mcp(isolated_sql_db):
    fake = FakeLLM(
        [
            _tool_call_response(
                "query_database",
                {"sql": "SELECT ticker FROM companies WHERE sector = 'Energy'"},
            ),
            _text_response("The energy-sector company in the database is XOM."),
        ]
    )
    agent = LangGraphAgent(llm=fake)

    answer = await agent.answer("Which company in the database is in the energy sector?")

    assert answer == "The energy-sector company in the database is XOM."
    tool_result = fake.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is False
    assert "XOM" in tool_result["content"]


# ---------------------------------------------------------------------------
# Routing to Document MCP still works when SQL MCP is also available
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_routes_document_question_to_document_mcp(isolated_sql_db):
    fake = FakeLLM(
        [
            _tool_call_response("search_documents", {"query": "RAG"}),
            _text_response("The documents describe RAG as retrieval-augmented generation."),
        ]
    )
    agent = LangGraphAgent(llm=fake)

    answer = await agent.answer("What do the documents say about RAG?")

    assert answer == "The documents describe RAG as retrieval-augmented generation."
    tool_result = fake.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is False


# ---------------------------------------------------------------------------
# Mixed question: sequential calls across both servers before a final answer
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_mixed_question_calls_both_servers_before_final_answer(isolated_sql_db):
    fake = FakeLLM(
        [
            _tool_call_response("search_documents", {"query": "RAG"}),
            _tool_call_response(
                "query_database",
                {
                    "sql": (
                        "SELECT ticker, (MAX(close_price) - MIN(close_price)) AS spread "
                        "FROM prices GROUP BY ticker ORDER BY spread DESC LIMIT 1"
                    )
                },
            ),
            _text_response(
                "According to the documents, RAG stands for retrieval-augmented "
                "generation. In the database, TSLA had the largest price spread."
            ),
        ]
    )
    agent = LangGraphAgent(llm=fake)

    answer = await agent.answer(
        "According to the documents, what is RAG, and which stock in the "
        "database moved the most?"
    )

    assert "retrieval-augmented" in answer
    assert "TSLA" in answer
    assert len(fake.calls) == 3

    doc_tool_result = fake.calls[1]["messages"][-1]["content"][0]
    sql_tool_result = fake.calls[2]["messages"][-1]["content"][0]
    assert doc_tool_result["is_error"] is False
    assert sql_tool_result["is_error"] is False
    assert "TSLA" in sql_tool_result["content"]


# ---------------------------------------------------------------------------
# A rejected/forbidden SQL query surfaces as a normal tool error, not a crash
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_forbidden_sql_query_handled_gracefully(isolated_sql_db):
    fake = FakeLLM(
        [
            _tool_call_response("query_database", {"sql": "DROP TABLE companies"}),
            _text_response("I can only run read-only queries against that database."),
        ]
    )
    agent = LangGraphAgent(llm=fake)

    answer = await agent.answer("Delete the companies table.")

    assert answer == "I can only run read-only queries against that database."
    tool_result = fake.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True


@pytest.fixture
def anyio_backend():
    return "asyncio"
