"""
Minimal FastAPI API Layer (Phase 6)
=====================================

    HTTP client
     |
     v
    POST /query
     |
     v
    nexusai.agent.langgraph_agent.LangGraphAgent  (reused as-is)
     |
     v
    MCP Client -> MCP Server(s) -> RAG / SQL

This module is intentionally a thin HTTP wrapper. It does not implement or
duplicate any agent, tool, RAG, or SQL logic -- it only:

- validates the incoming request,
- hands the question to the *existing* `LangGraphAgent` (imported, not
  reimplemented), and
- translates the agent's result (or failure) into an HTTP response.

No API key is required to import this module or to start the app -- the
underlying `LangGraphAgent` only needs an LLM provider (and therefore a key)
once a request actually asks it to answer a question, which is why tests
can exercise this module end-to-end with a fake agent and no GROQ_API_KEY.

Run directly with:  PYTHONPATH=src uvicorn nexusai.api.app:app --reload
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from nexusai.agent.langgraph_agent import LangGraphAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nexusai.api")

app = FastAPI(
    title="NexusAI API",
    description="Minimal HTTP layer over the existing NexusAI LangGraph agent.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Body for POST /query. `question` must be a non-blank string."""

    question: str = Field(..., min_length=1, description="The user's question.")

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        # min_length=1 already rejects "", but not whitespace-only strings
        # like "   " -- reject those too rather than passing blank input on
        # to the agent.
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class QueryResponse(BaseModel):
    """Body for a successful POST /query response."""

    answer: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Agent access (dependency-injected so tests can substitute a fake agent
# without touching the real LLM provider, MCP servers, RAG, or SQL code)
# ---------------------------------------------------------------------------

_agent: LangGraphAgent | None = None


def get_agent() -> LangGraphAgent:
    """
    Return the shared `LangGraphAgent` instance, constructing it lazily on
    first use. Building a `LangGraphAgent()` does not itself require an LLM
    API key (mirrors `LangGraphAgent.__init__`'s own lazy-provider
    convention) -- a key is only needed once `agent.answer(...)` actually
    runs.

    Overridden in tests via `app.dependency_overrides[get_agent]`.
    """
    global _agent
    if _agent is None:
        _agent = LangGraphAgent()
    return _agent


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, agent: LangGraphAgent = Depends(get_agent)) -> QueryResponse:
    """Answer one question by delegating to the existing LangGraph agent."""
    try:
        answer = await agent.answer(request.question)
    except Exception as exc:  # agent/LLM/MCP-level failure -- never leak internals
        logger.error("Agent failed to answer question: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="The agent could not answer the question. Please try again.",
        ) from exc

    return QueryResponse(answer=answer, metadata={"question": request.question})
