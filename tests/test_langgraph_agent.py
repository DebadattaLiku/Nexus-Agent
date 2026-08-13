"""
Tests for the Phase 4 LangGraph agent, using a fake LLM provider (and, for
the lower-level error-path tests, a fake MCP client too) so no real API key,
network call, or live MCP server is ever required.

Two testing strategies are used, matching the two public entry points:

- `LangGraphAgent(llm=fake)` for end-to-end tests that exercise the real,
  in-process Document MCP Server (mirrors tests/test_tool_agent.py) --
  these prove the graph actually reaches real tools through the MCP
  boundary.
- `build_graph(llm, client, known_tools, tool_schemas)` with a fake MCP
  client for the error-injection tests (MCP failure, unknown tool,
  malformed arguments) where the real server can't easily be made to fail
  the way we need.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nexusai.agent.langgraph_agent import (
    LangGraphAgent,
    _initial_state,
    _route_after_agent,
    build_graph,
)
from nexusai.agent.tool_agent import MAX_TOOL_ROUNDS
from nexusai.llm.provider import LLMResponse, ToolCall
from nexusai.mcp_servers import document_server
from nexusai.mcp_servers.document_server import DOCUMENTS_DIR
from nexusai.rag.config import RAGConfig
from nexusai.rag.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


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


class LoopingFakeLLM:
    """Always requests the same tool call -- used to drive the max-round
    termination path without needing a long scripted response list."""

    def __init__(self, tool_name: str, arguments: dict) -> None:
        self._tool_name = tool_name
        self._arguments = arguments
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, system=None):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        return _tool_call_response(self._tool_name, self._arguments)


class FailingLLM:
    def chat(self, messages, tools=None, system=None):
        raise RuntimeError("simulated network failure")


def _tool_call_response(name: str, arguments: dict, call_id: str = "call_1") -> LLMResponse:
    call = ToolCall(id=call_id, name=name, arguments=arguments)
    return LLMResponse(
        text=None,
        tool_calls=[call],
        stop_reason="tool_use",
        raw_content=[{"type": "tool_use", "id": call_id, "name": name, "input": arguments}],
    )


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn", raw_content=[{"type": "text", "text": text}])


class FakeTool:
    """Minimal stand-in for an mcp.types.Tool -- only the attributes
    _mcp_tools_to_schemas()/_validate_arguments() actually read."""

    def __init__(self, name: str, input_schema: dict | None = None, description: str = "") -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema or {"type": "object", "properties": {}}


class FakeToolsResult:
    def __init__(self, tools: list[FakeTool]) -> None:
        self.tools = tools


class FakeCallResult:
    def __init__(self, text: str, is_error: bool = False) -> None:
        self.is_error = is_error
        self.structured_content = None
        self.content = [SimpleNamespace(text=text)]


class FakeMCPClient:
    """Minimal stand-in for the real MCP `Client` -- only list_tools()/
    call_tool() are used by the graph."""

    def __init__(self, tools: list[FakeTool], call_tool_fn) -> None:
        self._tools = tools
        self._call_tool_fn = call_tool_fn
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self):
        return FakeToolsResult(self._tools)

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return await self._call_tool_fn(name, arguments)


def _known_tools(tools: list[FakeTool]) -> dict[str, Any]:
    return {t.name: t for t in tools}


def _schemas(tools: list[FakeTool]) -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools]


# ---------------------------------------------------------------------------
# 1. Graph initialization / 2. initial state
# ---------------------------------------------------------------------------


def test_initial_state_shape():
    state = _initial_state("What does the RAG document say about retrieval?")
    assert state["messages"] == [
        {"role": "user", "content": "What does the RAG document say about retrieval?"}
    ]
    assert state["pending_tool_calls"] == []
    assert state["round_count"] == 0
    assert state["final_answer"] is None


@pytest.mark.anyio
async def test_graph_builds_and_compiles_and_runs():
    """Building the graph with no tool calls involved should just work end
    to end -- proves nodes/edges/compilation are wired correctly."""
    fake = FakeLLM([_text_response("hi there")])
    tool = FakeTool("noop")
    client = FakeMCPClient([tool], call_tool_fn=None)

    graph = build_graph(fake, client, _known_tools([tool]), _schemas([tool]))
    assert graph is not None
    assert hasattr(graph, "ainvoke")

    final_state = await graph.ainvoke(_initial_state("hello"))
    assert final_state["final_answer"] == "hi there"


# ---------------------------------------------------------------------------
# 8/9. Routing (pure function tests)
# ---------------------------------------------------------------------------


def test_routing_agent_to_tools_when_no_final_answer():
    state = _initial_state("q")
    state["pending_tool_calls"] = [ToolCall(id="1", name="list_documents", arguments={})]
    state["final_answer"] = None
    assert _route_after_agent(state) == "tools"


def test_routing_agent_to_end_when_final_answer_set():
    state = _initial_state("q")
    state["final_answer"] = "done"
    assert _route_after_agent(state) == "end"


def test_routing_agent_to_end_on_empty_string_final_answer():
    """An empty-but-not-None final answer (a legitimately blank LLM reply)
    must still terminate the graph, not loop forever."""
    state = _initial_state("q")
    state["final_answer"] = ""
    assert _route_after_agent(state) == "end"


# ---------------------------------------------------------------------------
# 3. Direct final answer without tools
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_direct_final_answer_without_tools():
    fake = FakeLLM([_text_response("Paris is the capital of France.")])
    agent = LangGraphAgent(llm=fake)

    answer = await agent.answer("What is the capital of France?")

    assert answer == "Paris is the capital of France."
    assert len(fake.calls) == 1


# ---------------------------------------------------------------------------
# 4. Single MCP tool call / 7. tool result returned to the LLM
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_single_mcp_tool_call_search_documents():
    fake = FakeLLM(
        [
            _tool_call_response("search_documents", {"query": "RAG"}),
            _text_response("The documents mention RAG."),
        ]
    )
    agent = LangGraphAgent(llm=fake)

    answer = await agent.answer("What do the documents say about RAG?")

    assert answer == "The documents mention RAG."
    # The tool result must have been sent back to the LLM before it gave
    # the final answer.
    second_call_messages = fake.calls[1]["messages"]
    tool_result_msg = second_call_messages[-1]
    assert tool_result_msg["role"] == "user"
    block = tool_result_msg["content"][0]
    assert block["type"] == "tool_result"
    assert block["is_error"] is False
    assert "rag_basics.txt" in block["content"] or "RAG" in block["content"]


# ---------------------------------------------------------------------------
# 5. semantic_search tool call
# ---------------------------------------------------------------------------


@pytest.fixture
def rag_pipeline(tmp_path):
    """Deterministic offline (hashing) RAG pipeline over the real sample
    docs, injected into the document server -- mirrors
    tests/test_semantic_search_tool.py."""
    config = RAGConfig(
        documents_dir=DOCUMENTS_DIR,
        index_dir=tmp_path / "index",
        chunk_size=60,
        chunk_overlap=10,
        embedding_provider="hashing",
        embedding_dim=64,
        top_k=4,
    )
    pipeline = RAGPipeline.build(config)
    document_server.set_pipeline(pipeline)
    yield pipeline
    document_server.set_pipeline(None)


@pytest.mark.anyio
async def test_semantic_search_tool_call(rag_pipeline):
    fake = FakeLLM(
        [
            _tool_call_response("semantic_search", {"query": "retrieval and embeddings", "top_k": 3}),
            _text_response("Retrieval combines embeddings with generation."),
        ]
    )
    agent = LangGraphAgent(llm=fake)

    answer = await agent.answer("What does the RAG document say about retrieval?")

    assert answer == "Retrieval combines embeddings with generation."
    tool_result = fake.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is False
    assert "rag_basics.txt" in tool_result["content"] or "score" in tool_result["content"]


# ---------------------------------------------------------------------------
# 6. Multiple sequential tool calls
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_multiple_sequential_tool_calls_list_then_get_document():
    fake = FakeLLM(
        [
            _tool_call_response("list_documents", {}),
            _tool_call_response("get_document", {"filename": "langgraph_notes.txt"}),
            _text_response("The LangGraph document explains graph-based orchestration."),
        ]
    )
    agent = LangGraphAgent(llm=fake)

    answer = await agent.answer("Tell me what the LangGraph document says.")

    assert answer == "The LangGraph document explains graph-based orchestration."
    assert len(fake.calls) == 3

    round3_last_message = fake.calls[2]["messages"][-1]
    tool_result_content = round3_last_message["content"][0]["content"]
    assert "langgraph_notes.txt" in tool_result_content or "LangGraph" in tool_result_content


# ---------------------------------------------------------------------------
# 10. Unknown tool handling
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_unknown_tool_is_rejected_without_crashing():
    known_tool = FakeTool("list_documents")

    async def call_tool_fn(name, arguments):
        raise AssertionError("should never reach the MCP client for an unknown tool")

    client = FakeMCPClient([known_tool], call_tool_fn)
    fake = FakeLLM(
        [
            _tool_call_response("delete_everything", {}),
            _text_response("I can't do that."),
        ]
    )

    graph = build_graph(fake, client, _known_tools([known_tool]), _schemas([known_tool]))
    final_state = await graph.ainvoke(_initial_state("Delete everything"))

    assert final_state["final_answer"] == "I can't do that."
    block = fake.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    assert "unknown tool" in block["content"].lower()


# ---------------------------------------------------------------------------
# 11. Malformed arguments
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_malformed_arguments_are_handled_gracefully():
    tool = FakeTool("get_document", input_schema={"type": "object", "required": ["filename"]})

    async def call_tool_fn(name, arguments):
        raise AssertionError("should never reach the MCP client with missing required args")

    client = FakeMCPClient([tool], call_tool_fn)
    fake = FakeLLM(
        [
            _tool_call_response("get_document", {}),
            _text_response("I need a filename."),
        ]
    )

    graph = build_graph(fake, client, _known_tools([tool]), _schemas([tool]))
    final_state = await graph.ainvoke(_initial_state("Get the document"))

    assert final_state["final_answer"] == "I need a filename."
    block = fake.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    assert "filename" in block["content"].lower()


# ---------------------------------------------------------------------------
# 12. MCP failure
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_mcp_transport_failure_is_handled_gracefully():
    tool = FakeTool("list_documents")

    async def call_tool_fn(name, arguments):
        raise ConnectionError("simulated MCP transport failure")

    client = FakeMCPClient([tool], call_tool_fn)
    fake = FakeLLM(
        [
            _tool_call_response("list_documents", {}),
            _text_response("Something went wrong fetching documents."),
        ]
    )

    graph = build_graph(fake, client, _known_tools([tool]), _schemas([tool]))
    final_state = await graph.ainvoke(_initial_state("List documents"))

    assert final_state["final_answer"] == "Something went wrong fetching documents."
    block = fake.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    assert "error calling tool" in block["content"].lower()


# ---------------------------------------------------------------------------
# 13. LLM failure
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_llm_failure_is_handled_gracefully():
    tool = FakeTool("list_documents")
    client = FakeMCPClient([tool], call_tool_fn=None)

    graph = build_graph(FailingLLM(), client, _known_tools([tool]), _schemas([tool]))
    final_state = await graph.ainvoke(_initial_state("hello"))

    answer = final_state["final_answer"]
    assert answer is not None
    assert "failed" in answer.lower() or "sorry" in answer.lower()


@pytest.mark.anyio
async def test_llm_failure_via_public_agent_returns_message_not_exception():
    agent = LangGraphAgent(llm=FailingLLM())
    answer = await agent.answer("hello")
    assert "failed" in answer.lower() or "sorry" in answer.lower()


# ---------------------------------------------------------------------------
# 14. Maximum-step termination
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_graph_terminates_after_max_rounds():
    tool = FakeTool("list_documents")

    async def call_tool_fn(name, arguments):
        return FakeCallResult("[]")

    client = FakeMCPClient([tool], call_tool_fn)
    looping_llm = LoopingFakeLLM("list_documents", {})

    graph = build_graph(looping_llm, client, _known_tools([tool]), _schemas([tool]))
    final_state = await graph.ainvoke(_initial_state("Keep looping forever"), config={"recursion_limit": 200})

    assert final_state["final_answer"] == "I couldn't finish within the allowed number of tool calls."
    assert len(looping_llm.calls) == MAX_TOOL_ROUNDS


# ---------------------------------------------------------------------------
# Empty tool results
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_empty_tool_result_does_not_crash_the_graph():
    fake = FakeLLM(
        [
            _tool_call_response("search_documents", {"query": "zzzznonexistentzzzz"}),
            _text_response("I couldn't find anything about that."),
        ]
    )
    agent = LangGraphAgent(llm=fake)

    answer = await agent.answer("What do the documents say about zzzznonexistentzzzz?")

    assert answer == "I couldn't find anything about that."
    block = fake.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is False
    assert block["content"].strip() != ""


@pytest.fixture
def anyio_backend():
    return "asyncio"
