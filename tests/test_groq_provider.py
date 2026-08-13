"""
Regression tests for GroqProvider's real-world tool-calling failure mode.

Background: the mocked FakeLLM-based tests in test_tool_agent.py never
exercise GroqProvider's actual translation to/from Groq's wire format, so
they couldn't catch this. Against the *real* Groq API, certain models
(e.g. llama-3.3-70b-versatile) sometimes emit Llama's "pythonic" built-in
tool-call syntax --

    <function=search_documents{"query": "RAG", "case_sensitive": false}>

-- instead of a structured `tool_calls` entry. This shows up two ways:

1. Groq's own parser also rejects it and the API call raises an HTTP 400
   ("Failed to call a function"), with the offending text available in
   `error.failed_generation` on the exception body.
2. Groq accepts the request (HTTP 200) but the model put the syntax in
   plain `message.content` instead of `message.tool_calls`.

These tests mock the `groq.Groq` client (no real network access / API key
needed) and drive GroqProvider.chat() through both failure shapes, plus the
pre-existing malformed-JSON-arguments path, to confirm recovery is generic
across tool names and never hard-codes "search_documents" or "get_document"
specifically.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nexusai.llm.provider import GroqProvider


class _FakeGroqAPIError(Exception):
    """Stands in for groq.APIStatusError / BadRequestError. GroqProvider only
    ever inspects `.body`, so a plain Exception subclass is sufficient and
    keeps these tests independent of the groq SDK's exact exception
    hierarchy."""

    def __init__(self, message: str, *, body: dict) -> None:
        super().__init__(message)
        self.body = body


def _make_provider() -> tuple[GroqProvider, MagicMock]:
    """Build a GroqProvider whose `_client` is a MagicMock, so no real
    GROQ_API_KEY or network access is needed."""
    with patch("groq.Groq") as mock_groq_cls:
        provider = GroqProvider(api_key="test-key", model="llama-3.3-70b-versatile")
        mock_client = mock_groq_cls.return_value
    return provider, mock_client


def _tool_schema(name: str, required: list[str] | None = None) -> dict:
    return {
        "name": name,
        "description": f"{name} tool",
        "input_schema": {"type": "object", "properties": {}, "required": required or []},
    }


# ---------------------------------------------------------------------------
# 1. HTTP 400 "Failed to call a function" recovery (the reported bug)
# ---------------------------------------------------------------------------


def test_recovers_search_documents_call_from_failed_generation_400():
    """Exact repro of the reported failure for 'What do the documents say
    about RAG?'."""
    provider, mock_client = _make_provider()
    mock_client.chat.completions.create.side_effect = _FakeGroqAPIError(
        "Failed to call a function",
        body={
            "error": {
                "message": "Failed to call a function. Please adjust your prompt.",
                "failed_generation": '<function=search_documents{"query": "RAG", "case_sensitive": false}>',
            }
        },
    )

    response = provider.chat(
        [{"role": "user", "content": "What do the documents say about RAG?"}],
        tools=[_tool_schema("search_documents"), _tool_schema("get_document"), _tool_schema("list_documents")],
    )

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "search_documents"
    assert call.arguments == {"query": "RAG", "case_sensitive": False}
    assert response.stop_reason == "tool_use"
    # raw_content must be replayable back into the conversation history.
    assert response.raw_content[0]["type"] == "tool_use"
    assert response.raw_content[0]["name"] == "search_documents"


def test_recovers_get_document_call_from_failed_generation_400():
    """Exact repro of the reported failure for 'Tell me what the LangGraph
    document says.' — the model is trusted to have supplied *some*
    filename; ensuring it used the real one is the system-prompt/tool-
    description job (see test_tool_agent.py), not the parser's."""
    provider, mock_client = _make_provider()
    mock_client.chat.completions.create.side_effect = _FakeGroqAPIError(
        "Failed to call a function",
        body={
            "error": {
                "failed_generation": '<function=get_document{"filename": "langgraph_notes.txt"}>',
            }
        },
    )

    response = provider.chat(
        [{"role": "user", "content": "Tell me what the LangGraph document says."}],
        tools=[_tool_schema("get_document", required=["filename"])],
    )

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "get_document"
    assert call.arguments == {"filename": "langgraph_notes.txt"}


def test_recovery_is_generic_not_hardcoded_to_known_tool_names():
    """Recovery must work for any MCP tool, not just search_documents /
    get_document — the requirement is generic tool-call handling."""
    provider, mock_client = _make_provider()
    mock_client.chat.completions.create.side_effect = _FakeGroqAPIError(
        "Failed to call a function",
        body={"error": {"failed_generation": '<function=some_future_tool{"x": 1, "y": [1, 2, 3]}>'}},
    )

    response = provider.chat(
        [{"role": "user", "content": "do the thing"}],
        tools=[_tool_schema("some_future_tool")],
    )

    assert response.tool_calls[0].name == "some_future_tool"
    assert response.tool_calls[0].arguments == {"x": 1, "y": [1, 2, 3]}


def test_unrecoverable_api_error_is_reraised():
    """An API error with no `failed_generation` (e.g. a rate limit or auth
    error) must NOT be swallowed — only this one specific, recoverable
    shape gets special handling."""
    provider, mock_client = _make_provider()
    original_exc = _FakeGroqAPIError("Rate limit exceeded", body={"error": {"message": "rate limited"}})
    mock_client.chat.completions.create.side_effect = original_exc

    with pytest.raises(_FakeGroqAPIError) as exc_info:
        provider.chat([{"role": "user", "content": "hi"}], tools=[_tool_schema("list_documents")])

    assert exc_info.value is original_exc


def test_failed_generation_with_unparseable_content_is_reraised():
    """If failed_generation doesn't actually contain a parseable
    <function=...> call, don't fabricate a tool call — re-raise instead."""
    provider, mock_client = _make_provider()
    original_exc = _FakeGroqAPIError(
        "Failed to call a function",
        body={"error": {"failed_generation": "I think I should search for something but I'm not sure how."}},
    )
    mock_client.chat.completions.create.side_effect = original_exc

    with pytest.raises(_FakeGroqAPIError):
        provider.chat([{"role": "user", "content": "hi"}], tools=[_tool_schema("search_documents")])


# ---------------------------------------------------------------------------
# 2. HTTP 200 response, but the pythonic call leaked into plain text content
# ---------------------------------------------------------------------------


def _openai_response(*, content: str | None, tool_calls=None, finish_reason: str = "stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def test_recovers_pythonic_call_leaked_into_plain_text_response():
    provider, mock_client = _make_provider()
    mock_client.chat.completions.create.return_value = _openai_response(
        content='<function=list_documents{}>',
        tool_calls=None,
    )

    response = provider.chat(
        [{"role": "user", "content": "What documents are available?"}],
        tools=[_tool_schema("list_documents")],
    )

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "list_documents"
    assert response.tool_calls[0].arguments == {}


def test_normal_text_response_without_function_syntax_is_untouched():
    provider, mock_client = _make_provider()
    mock_client.chat.completions.create.return_value = _openai_response(
        content="Paris is the capital of France.",
        tool_calls=None,
    )

    response = provider.chat([{"role": "user", "content": "capital of France?"}])

    assert response.tool_calls == []
    assert response.text == "Paris is the capital of France."


# ---------------------------------------------------------------------------
# 3. Malformed tool-call arguments in the normal (structured) path
# ---------------------------------------------------------------------------


def test_malformed_json_arguments_become_empty_dict_not_a_crash():
    provider, mock_client = _make_provider()
    tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="get_document", arguments="{not valid json"),
    )
    mock_client.chat.completions.create.return_value = _openai_response(
        content=None, tool_calls=[tc], finish_reason="tool_calls"
    )

    response = provider.chat(
        [{"role": "user", "content": "get the doc"}],
        tools=[_tool_schema("get_document", required=["filename"])],
    )

    assert response.tool_calls[0].name == "get_document"
    assert response.tool_calls[0].arguments == {}


# ---------------------------------------------------------------------------
# 4. Request shape sanity check
# ---------------------------------------------------------------------------


def test_tool_choice_auto_is_sent_when_tools_are_provided():
    provider, mock_client = _make_provider()
    mock_client.chat.completions.create.return_value = _openai_response(content="ok", tool_calls=None)

    provider.chat([{"role": "user", "content": "hi"}], tools=[_tool_schema("list_documents")])

    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"][0]["function"]["name"] == "list_documents"


def test_multiple_structured_tool_calls_in_one_response_are_all_parsed():
    """Groq (like OpenAI) can return several entries in message.tool_calls
    for a single turn — e.g. the model asking for list_documents() and
    search_documents() together. Every entry must be parsed, with its
    tool_call id preserved, so ToolAgent can execute and reply to each."""
    provider, mock_client = _make_provider()
    tc1 = SimpleNamespace(id="call_a", function=SimpleNamespace(name="list_documents", arguments="{}"))
    tc2 = SimpleNamespace(
        id="call_b",
        function=SimpleNamespace(name="search_documents", arguments='{"query": "RAG"}'),
    )
    mock_client.chat.completions.create.return_value = _openai_response(
        content=None, tool_calls=[tc1, tc2], finish_reason="tool_calls"
    )

    response = provider.chat(
        [{"role": "user", "content": "list and search"}],
        tools=[_tool_schema("list_documents"), _tool_schema("search_documents")],
    )

    assert len(response.tool_calls) == 2
    assert response.tool_calls[0].id == "call_a"
    assert response.tool_calls[0].name == "list_documents"
    assert response.tool_calls[1].id == "call_b"
    assert response.tool_calls[1].arguments == {"query": "RAG"}


def test_tool_choice_omitted_when_no_tools():
    provider, mock_client = _make_provider()
    mock_client.chat.completions.create.return_value = _openai_response(content="ok", tool_calls=None)

    provider.chat([{"role": "user", "content": "hi"}])

    _, kwargs = mock_client.chat.completions.create.call_args
    assert "tool_choice" not in kwargs
    assert "tools" not in kwargs
