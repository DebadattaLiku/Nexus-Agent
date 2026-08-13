"""
Tests for the Phase 6 minimal FastAPI layer (nexusai.api.app).

These tests exercise only the HTTP boundary -- request validation,
dependency wiring, and error translation. The real `LangGraphAgent` is
replaced with a fake via `app.dependency_overrides[get_agent]`, so no real
LLM/Groq API key, network call, or MCP server is ever required, and no
agent/tool/RAG/SQL logic is reimplemented or re-tested here (that is
already covered by tests/test_langgraph_agent.py and friends).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nexusai.api.app import app, get_agent


class FakeAgent:
    """Stands in for `LangGraphAgent`: records the question it was asked and
    returns a scripted answer, or raises a scripted exception."""

    def __init__(self, answer: str | None = None, error: Exception | None = None) -> None:
        self._answer = answer
        self._error = error
        self.questions: list[str] = []

    async def answer(self, question: str) -> str:
        self.questions.append(question)
        if self._error is not None:
            raise self._error
        return self._answer or ""


@pytest.fixture(autouse=True)
def _clear_overrides():
    """Ensure no dependency override leaks between tests."""
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_query_returns_agent_answer_and_metadata(client: TestClient) -> None:
    fake = FakeAgent(answer="RAG stands for Retrieval-Augmented Generation.")
    app.dependency_overrides[get_agent] = lambda: fake

    response = client.post("/query", json={"question": "What does the RAG document say?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "RAG stands for Retrieval-Augmented Generation."
    assert body["metadata"] == {"question": "What does the RAG document say?"}


def test_query_passes_the_exact_question_to_the_agent(client: TestClient) -> None:
    fake = FakeAgent(answer="ok")
    app.dependency_overrides[get_agent] = lambda: fake

    client.post("/query", json={"question": "Which sector has the highest average return?"})

    assert fake.questions == ["Which sector has the highest average return?"]


def test_query_reuses_the_injected_agent_without_reimplementing_logic(client: TestClient) -> None:
    """The API must delegate to the agent rather than compute an answer
    itself -- proven by swapping in a fake that returns a fixed sentinel
    unrelated to the question asked."""
    fake = FakeAgent(answer="__SENTINEL_FROM_FAKE_AGENT__")
    app.dependency_overrides[get_agent] = lambda: fake

    response = client.post("/query", json={"question": "anything at all"})

    assert response.json()["answer"] == "__SENTINEL_FROM_FAKE_AGENT__"


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_query_missing_question_field_is_rejected(client: TestClient) -> None:
    app.dependency_overrides[get_agent] = lambda: FakeAgent(answer="unused")

    response = client.post("/query", json={})

    assert response.status_code == 422


def test_query_empty_question_string_is_rejected(client: TestClient) -> None:
    app.dependency_overrides[get_agent] = lambda: FakeAgent(answer="unused")

    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422


def test_query_whitespace_only_question_is_rejected(client: TestClient) -> None:
    app.dependency_overrides[get_agent] = lambda: FakeAgent(answer="unused")

    response = client.post("/query", json={"question": "   "})

    assert response.status_code == 422


def test_query_wrong_type_for_question_is_rejected(client: TestClient) -> None:
    app.dependency_overrides[get_agent] = lambda: FakeAgent(answer="unused")

    response = client.post("/query", json={"question": 12345})

    assert response.status_code == 422


def test_query_malformed_json_body_is_rejected(client: TestClient) -> None:
    app.dependency_overrides[get_agent] = lambda: FakeAgent(answer="unused")

    response = client.post(
        "/query",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Agent error handling
# ---------------------------------------------------------------------------


def test_query_agent_exception_returns_502(client: TestClient) -> None:
    fake = FakeAgent(error=RuntimeError("simulated LLM failure"))
    app.dependency_overrides[get_agent] = lambda: fake

    response = client.post("/query", json={"question": "What is NexusAI?"})

    assert response.status_code == 502
    assert "detail" in response.json()


def test_query_agent_exception_does_not_leak_internal_error_text(client: TestClient) -> None:
    """The HTTP error body must not echo the raw exception message, so
    internal details (which could include things like partial stack traces
    or config hints) never reach the client."""
    secret_looking_message = "GROQ_API_KEY=sk-super-secret-value invalid"
    fake = FakeAgent(error=RuntimeError(secret_looking_message))
    app.dependency_overrides[get_agent] = lambda: fake

    response = client.post("/query", json={"question": "What is NexusAI?"})

    assert response.status_code == 502
    assert secret_looking_message not in response.text


def test_query_agent_missing_api_key_error_returns_502_not_500(client: TestClient) -> None:
    """Mirrors the real failure mode of `get_default_provider()` (a
    `RuntimeError` when no provider API key is configured) to confirm it is
    handled as an upstream/agent failure, not an unhandled server crash."""
    fake = FakeAgent(error=RuntimeError("No LLM provider API key configured"))
    app.dependency_overrides[get_agent] = lambda: fake

    response = client.post("/query", json={"question": "hello"})

    assert response.status_code == 502


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


def test_get_agent_returns_a_langgraph_agent_by_default() -> None:
    """Without any override, the real dependency constructs a real
    `LangGraphAgent` -- but constructing it must not require an API key or
    perform any network/MCP activity (mirrors `LangGraphAgent.__init__`)."""
    from nexusai.agent.langgraph_agent import LangGraphAgent
    from nexusai.api.app import get_agent

    agent = get_agent()

    assert isinstance(agent, LangGraphAgent)


def test_get_agent_returns_the_same_cached_instance() -> None:
    from nexusai.api.app import get_agent

    assert get_agent() is get_agent()
