"""
Tests for the Phase 2 tool-calling agent, using a fake LLM provider so no
real API key or network call is ever required.
"""

from __future__ import annotations

import pytest

from nexusai.agent.tool_agent import ToolAgent
from nexusai.llm.provider import LLMResponse, ToolCall


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


def _tool_call_response(name: str, arguments: dict) -> LLMResponse:
    call = ToolCall(id="call_1", name=name, arguments=arguments)
    return LLMResponse(
        text=None,
        tool_calls=[call],
        stop_reason="tool_use",
        raw_content=[{"type": "tool_use", "id": "call_1", "name": name, "input": arguments}],
    )


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn", raw_content=[{"type": "text", "text": text}])


@pytest.mark.anyio
async def test_llm_receives_mcp_tool_definitions():
    fake = FakeLLM([_text_response("no tool needed")])
    agent = ToolAgent(llm=fake)

    await agent.answer("hello")

    tool_names = {t["name"] for t in fake.calls[0]["tools"]}
    assert tool_names == {"list_documents", "get_document", "search_documents", "semantic_search"}


@pytest.mark.anyio
async def test_tool_call_is_detected_and_correct_tool_executed_and_returns_final_answer():
    fake = FakeLLM(
        [
            _tool_call_response("search_documents", {"query": "RAG"}),
            _text_response("The documents mention RAG."),
        ]
    )
    agent = ToolAgent(llm=fake)

    answer = await agent.answer("What do the documents say about RAG?")

    assert answer == "The documents mention RAG."
    # Second chat() call must include the tool result sent back to the LLM.
    second_call_messages = fake.calls[1]["messages"]
    tool_result_msg = second_call_messages[-1]
    assert tool_result_msg["role"] == "user"
    block = tool_result_msg["content"][0]
    assert block["type"] == "tool_result"
    assert block["is_error"] is False
    assert "rag_basics.txt" in block["content"] or "RAG" in block["content"]


@pytest.mark.anyio
async def test_list_documents_tool_call_returns_known_files():
    fake = FakeLLM(
        [
            _tool_call_response("list_documents", {}),
            _text_response("Here are the documents."),
        ]
    )
    agent = ToolAgent(llm=fake)

    answer = await agent.answer("What documents are available?")

    assert answer == "Here are the documents."
    tool_result_content = fake.calls[1]["messages"][-1]["content"][0]["content"]
    assert "mcp_overview.txt" in tool_result_content


@pytest.mark.anyio
async def test_unknown_tool_is_rejected_without_crashing():
    fake = FakeLLM(
        [
            _tool_call_response("delete_everything", {}),
            _text_response("I can't do that."),
        ]
    )
    agent = ToolAgent(llm=fake)

    answer = await agent.answer("Delete everything")

    assert answer == "I can't do that."
    block = fake.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    assert "unknown tool" in block["content"].lower()


@pytest.mark.anyio
async def test_malformed_arguments_are_handled_gracefully():
    # get_document requires a "filename" argument; omit it.
    fake = FakeLLM(
        [
            _tool_call_response("get_document", {}),
            _text_response("I need a filename."),
        ]
    )
    agent = ToolAgent(llm=fake)

    answer = await agent.answer("Get the document")

    assert answer == "I need a filename."
    block = fake.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True
    assert "filename" in block["content"].lower()


@pytest.mark.anyio
async def test_no_tool_needed_returns_direct_answer():
    fake = FakeLLM([_text_response("Paris is the capital of France.")])
    agent = ToolAgent(llm=fake)

    answer = await agent.answer("What is the capital of France?")

    assert answer == "Paris is the capital of France."
    assert len(fake.calls) == 1


@pytest.mark.anyio
async def test_sequential_multi_step_tool_calls_list_then_get_document():
    """Generic multi-round flow: list_documents() to discover the real
    filename, then get_document() with it, then a final answer — the same
    shape as the real-world 'langgraph_notes.txt' workflow, driven purely
    through the provider-agnostic ToolAgent loop (no Groq-specific code
    involved)."""
    fake = FakeLLM(
        [
            _tool_call_response("list_documents", {}),
            _tool_call_response("get_document", {"filename": "langgraph_notes.txt"}),
            _text_response("The LangGraph document explains graph-based orchestration."),
        ]
    )
    agent = ToolAgent(llm=fake)

    answer = await agent.answer("Tell me what the LangGraph document says.")

    assert answer == "The LangGraph document explains graph-based orchestration."
    assert len(fake.calls) == 3

    # Round 2's outgoing messages must contain round 1's tool result.
    round2_messages = fake.calls[1]["messages"]
    assert any(
        isinstance(m["content"], list)
        and m["content"]
        and m["content"][0].get("type") == "tool_result"
        for m in round2_messages
    )

    # Round 3's tool result (from the real MCP get_document call) must
    # contain the actual file content.
    round3_last_message = fake.calls[2]["messages"][-1]
    tool_result_content = round3_last_message["content"][0]["content"]
    assert "langgraph_notes.txt" in tool_result_content or "LangGraph" in tool_result_content


@pytest.mark.anyio
async def test_multiple_tool_calls_in_a_single_llm_response_are_all_executed():
    """Some models request several tool calls in one turn (parallel tool
    calls). Every one of them must be validated, executed through MCP, and
    have its result returned with the matching tool_use_id."""
    calls = [
        ToolCall(id="call_a", name="list_documents", arguments={}),
        ToolCall(id="call_b", name="search_documents", arguments={"query": "RAG"}),
    ]
    parallel_response = LLMResponse(
        text=None,
        tool_calls=calls,
        stop_reason="tool_use",
        raw_content=[
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in calls
        ],
    )
    fake = FakeLLM([parallel_response, _text_response("Done.")])
    agent = ToolAgent(llm=fake)

    answer = await agent.answer("List the documents and search for RAG.")

    assert answer == "Done."
    tool_results = fake.calls[1]["messages"][-1]["content"]
    assert len(tool_results) == 2
    ids = {r["tool_use_id"] for r in tool_results}
    assert ids == {"call_a", "call_b"}
    assert all(r["is_error"] is False for r in tool_results)


@pytest.mark.anyio
async def test_tool_call_loop_terminates_after_max_rounds():
    """A model that keeps requesting tools forever must not hang the agent
    — it should stop after MAX_TOOL_ROUNDS and return a clear message."""
    from nexusai.agent import tool_agent as tool_agent_module

    always_calls_tool = [
        _tool_call_response("list_documents", {}) for _ in range(tool_agent_module.MAX_TOOL_ROUNDS)
    ]
    fake = FakeLLM(always_calls_tool)
    agent = ToolAgent(llm=fake)

    answer = await agent.answer("Keep looping forever")

    assert "couldn't finish" in answer.lower()
    assert len(fake.calls) == tool_agent_module.MAX_TOOL_ROUNDS


@pytest.mark.anyio
async def test_llm_error_is_handled_gracefully_without_crashing():
    """If the LLM call itself raises (network error, auth error, etc.),
    the agent must return a friendly message instead of propagating the
    exception to the caller."""

    class FailingLLM:
        def chat(self, messages, tools=None, system=None):
            raise RuntimeError("simulated network failure")

    agent = ToolAgent(llm=FailingLLM())

    answer = await agent.answer("hello")

    assert "failed" in answer.lower() or "sorry" in answer.lower()


@pytest.fixture
def anyio_backend():
    return "asyncio"
