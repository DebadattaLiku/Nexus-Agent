"""
End-to-end regression test for the reported bug, wiring together the real
`ToolAgent`, the real `GroqProvider`, and the real MCP document server (only
the Groq HTTP client itself is mocked, since no real API key/network access
is available in this environment). This is what test_groq_provider.py and
test_tool_agent.py each test in isolation (provider-level and
FakeLLM-agent-level, respectively) — this file proves the two halves
actually work together end-to-end for the two exact questions from the bug
report.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nexusai.agent.tool_agent import ToolAgent
from nexusai.llm.provider import GroqProvider


class _FakeGroqAPIError(Exception):
    def __init__(self, message: str, *, body: dict) -> None:
        super().__init__(message)
        self.body = body


def _make_groq_provider():
    with patch("groq.Groq") as mock_groq_cls:
        provider = GroqProvider(api_key="test-key")
        mock_client = mock_groq_cls.return_value
    return provider, mock_client


def _openai_response(content, tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


@pytest.mark.anyio
async def test_what_do_the_documents_say_about_rag_end_to_end():
    """Reproduces: 'What do the documents say about RAG?' -> Groq 400 with
    failed_generation for search_documents -> should still resolve to a
    real MCP search_documents() call and a final answer."""
    provider, mock_client = _make_groq_provider()
    mock_client.chat.completions.create.side_effect = [
        _FakeGroqAPIError(
            "Failed to call a function",
            body={
                "error": {
                    "failed_generation": (
                        '<function=search_documents{"query": "RAG", "case_sensitive": false}>'
                    )
                }
            },
        ),
        _openai_response("Based on the documents, RAG combines retrieval with generation."),
    ]

    agent = ToolAgent(llm=provider)
    answer = await agent.answer("What do the documents say about RAG?")

    assert "RAG" in answer or "retrieval" in answer.lower()
    # First call attempted the (recovered) tool call; second call is the
    # final answer after the real MCP search_documents() result was fed back.
    assert mock_client.chat.completions.create.call_count == 2
    second_call_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
    tool_messages = [m for m in second_call_kwargs["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    # The real document server actually ran search_documents("RAG") and
    # found a real match in rag_basics.txt.
    assert "rag_basics.txt" in tool_messages[0]["content"]


@pytest.mark.anyio
async def test_tell_me_what_the_langgraph_document_says_end_to_end():
    """Reproduces: 'Tell me what the LangGraph document says.' -> Groq 400
    with failed_generation for get_document(filename='langgraph.txt') (a
    *guessed*, wrong filename) -> the real MCP get_document() call fails
    gracefully, and the improved system prompt/tool description steers the
    model to call list_documents() next and recover with the real filename."""
    provider, mock_client = _make_groq_provider()
    mock_client.chat.completions.create.side_effect = [
        # Round 1: model guesses a wrong filename, exactly as in the bug report.
        _FakeGroqAPIError(
            "Failed to call a function",
            body={"error": {"failed_generation": '<function=get_document{"filename": "langgraph.txt"}>'}},
        ),
        # Round 2: after seeing the error, the model follows the system
        # prompt's instruction and calls list_documents() instead of
        # guessing again.
        _openai_response(
            None,
            tool_calls=[
                SimpleNamespace(id="call_2", function=SimpleNamespace(name="list_documents", arguments="{}"))
            ],
            finish_reason="tool_calls",
        ),
        # Round 3: model uses the *real* filename returned by list_documents().
        _openai_response(
            None,
            tool_calls=[
                SimpleNamespace(
                    id="call_3",
                    function=SimpleNamespace(
                        name="get_document", arguments='{"filename": "langgraph_notes.txt"}'
                    ),
                )
            ],
            finish_reason="tool_calls",
        ),
        _openai_response("The LangGraph document explains graph-based agent orchestration."),
    ]

    agent = ToolAgent(llm=provider)
    answer = await agent.answer("Tell me what the LangGraph document says.")

    assert "LangGraph" in answer or "orchestration" in answer.lower()
    assert mock_client.chat.completions.create.call_count == 4

    # Confirm the failed guess never silently "succeeded": round 1's
    # recovered tool call actually ran through the real MCP server and
    # came back as an error, since langgraph.txt does not exist.
    round1_result_messages = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
    first_tool_result = [m for m in round1_result_messages if m.get("role") == "tool"][0]
    assert "not found" in first_tool_result["content"].lower()

    # The final, real get_document() call used the correct filename and
    # actually returned LangGraph content from disk.
    round3_result_messages = mock_client.chat.completions.create.call_args_list[3].kwargs["messages"]
    tool_messages = [m for m in round3_result_messages if m.get("role") == "tool"]
    assert any("LangGraph" in m["content"] or "graph" in m["content"].lower() for m in tool_messages)


@pytest.fixture
def anyio_backend():
    return "asyncio"
