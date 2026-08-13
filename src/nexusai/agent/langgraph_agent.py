"""
LangGraph Agent Orchestration (Phase 4, extended in Phase 5 for multi-MCP)
=============================================================================

Replaces the linear Python loop in `tool_agent.py` (Phase 2) with an
explicit, stateful LangGraph graph, while preserving MCP as the *only*
tool execution boundary:

    User
     |
     v
    LangGraph Agent
     |
     v
    Reason/Decide  (agent node -- calls the LLM)
     |
     v
    Dynamic MCP tool selection (tools node -- calls nexusai.client.mcp_client only)
     |
     +--> Document MCP  (RAG / document tools)
     |
     +--> SQL MCP       (read-only SQLite queries)
     |
     v
    Tool Result
     |
     v
    Reason/Decide
     |
     v
    Another MCP Tool (same or different server)  OR  Final Answer

Dependency direction (unchanged from Phase 1-4, enforced by import
hygiene in this file):

    LangGraph Agent -> MCP Client -> MCP Server -> RAG / document / SQL functionality

This module imports `nexusai.client.mcp_client` (the MCP *client*
boundary) and reuses small, already-public helpers from
`nexusai.agent.tool_agent` (round limit, and
tool-schema/result-flattening/argument-validation helpers). It never
imports `nexusai.mcp_servers.document_server`, `nexusai.mcp_servers.sql_server`,
`nexusai.rag`, or `nexusai.db` directly -- those stay reachable only
through the MCP client (as of Phase 5, `connect_multi_in_process()`,
which discovers tools from *both* servers behind one aggregated client).

Which MCP server actually handles a given tool call is never decided
here or hard-coded to a specific question -- the LLM picks a tool by
name/description from the combined schema list `answer()` hands it, and
`MultiMCPClient` (in `nexusai.client.mcp_client`) is what routes that
tool name to the server that actually owns it.

Run directly with:  python -m nexusai.agent.langgraph_agent
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from nexusai.agent.tool_agent import MAX_TOOL_ROUNDS, ToolAgent, _validate_arguments
from nexusai.client.mcp_client import connect_multi_in_process
from nexusai.llm.provider import LLMProvider, ToolCall, get_default_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nexusai.langgraph_agent")

# System prompt for the multi-MCP agent. Kept local to this module (rather
# than importing tool_agent.SYSTEM_PROMPT) because Phase 2's ToolAgent only
# ever talks to the Document MCP server and its prompt should keep
# reflecting exactly that; this agent talks to two servers and needs to
# describe both, and to tell the LLM to choose between them by tool
# description/schema rather than by any hard-coded rule.
SYSTEM_PROMPT = (
    "You are NexusAI, an assistant with access to two families of tools, "
    "discovered dynamically -- use whichever fits the question, and both "
    "together when a question needs both:\n\n"
    "- Document tools (list_documents, get_document, search_documents, "
    "semantic_search) -- for questions about the content of the local text "
    "document library.\n"
    "- Database tools (query_database) -- for structured/aggregate "
    "questions about companies, stock prices, sectors, or returns, backed "
    "by a small local finance database (tables: companies(ticker, name, "
    "sector), prices(ticker, trade_date, close_price, volume)).\n\n"
    "When the user asks about stock return, interpret return as the "
    "percentage price change between the earliest and latest available "
    "close_price for each ticker: "
    "((latest_close - earliest_close) / earliest_close) * 100. "
    "Do not interpret return as the maximum close_price.\n\n"
    "Pick the tool based on what the question is actually asking for, not "
    "on its phrasing -- a question about document content needs a document "
    "tool even if it mentions a company name, and a question about prices, "
    "returns, or sectors needs query_database even if it doesn't say "
    "'database'. If a question needs both (e.g. 'what do the documents say "
    "about X, and which stock had the highest return'), call tools from "
    "both, in any order, before giving a final answer.\n\n"
    "You do not know the exact filenames in the document library or the "
    "exact database schema details in advance beyond what the tool "
    "descriptions tell you -- never guess a filename or a table/column "
    "name; call list_documents() or write query_database() SQL that "
    "matches the schema described in its tool description.\n\n"
    "For document questions, prefer semantic_search(query, top_k) over "
    "search_documents() unless you need an exact word/phrase match. For "
    "database questions, query_database only accepts a single read-only "
    "SELECT (or WITH ... SELECT) statement -- if a query is rejected, "
    "rewrite it as a read-only SELECT rather than retrying the same "
    "statement.\n\n"
    "When answering from retrieved content:\n"
    "- Ground your answer only in what a tool actually returned -- never "
    "invent or assume document facts or database values that were not "
    "actually returned.\n"
    "- Mention which document(s) (by filename) or which query result an "
    "answer came from when it's useful for the user to know the source.\n"
    "- If a tool's results don't contain enough information to answer "
    "confidently, say so plainly instead of guessing."
    
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """
    LangGraph state threaded through every node.

    - `messages`: full conversation history, in the same provider-agnostic
      content-block shape used by `LLMProvider` (`nexusai.llm.provider`) --
      the same convention Phase 2's `ToolAgent` uses, so both agents share
      one message format and either can be swapped in behind the same MCP
      boundary.
    - `pending_tool_calls`: tool calls the agent node just requested, not
      yet executed. Only ever non-empty between the `agent` node and the
      `tools` node within a single graph step; `tools` always clears it.
    - `round_count`: number of LLM round trips completed so far. Used for
      the max-step termination condition.
    - `final_answer`: set (to a string, possibly empty) once the graph has
      an answer ready -- whether from a normal LLM text reply, an LLM/MCP
      error, or hitting the round limit. Its presence is what routes the
      graph to END.
    """

    messages: list[dict[str, Any]]
    pending_tool_calls: list[ToolCall]
    round_count: int
    final_answer: str | None


def _initial_state(question: str) -> AgentState:
    return {
        "messages": [{"role": "user", "content": question}],
        "pending_tool_calls": [],
        "round_count": 0,
        "final_answer": None,
    }


# ---------------------------------------------------------------------------
# Tool execution (through the MCP client only -- mirrors
# ToolAgent._execute_tool_call in tool_agent.py so both agents validate and
# fail the same way; kept local rather than imported since it is not part
# of tool_agent's public/reusable surface).
# ---------------------------------------------------------------------------


async def _execute_tool_call(
    client: Any,
    known_tools: dict[str, Any],
    call: ToolCall,
) -> dict[str, Any]:
    """Validate + execute one tool call through the MCP client only."""
    tool = known_tools.get(call.name)

    if tool is None:
        logger.warning("Rejected unknown tool: %r", call.name)
        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": f"Error: unknown tool {call.name!r}",
            "is_error": True,
        }

    error = _validate_arguments(tool.input_schema, call.arguments)
    if error:
        logger.warning("Rejected malformed arguments for %s: %s", call.name, error)
        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": f"Error: {error}",
            "is_error": True,
        }

    try:
        result = await client.call_tool(call.name, call.arguments)
    except Exception as exc:  # MCP transport/server-level failure
        logger.warning("MCP call_tool failed for %s: %s", call.name, exc)
        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": f"Error calling tool: {exc}",
            "is_error": True,
        }

    content = ToolAgent._tool_result_to_text(result)
    if not content.strip():
        # Empty tool results (e.g. a search that matched nothing) must not
        # be silently dropped -- an empty message body would look like a
        # malformed turn to some providers. Make the emptiness explicit so
        # the LLM can reason about it ("no matches") instead of stalling.
        content = "(tool returned no results)"

    return {
        "type": "tool_result",
        "tool_use_id": call.id,
        "content": content,
        "is_error": result.is_error,
    }


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def _make_agent_node(llm: LLMProvider, tool_schemas: list[dict[str, Any]]):
    """
    Build the `agent` node: one LLM call that either produces a final
    answer or requests one or more tool calls. Closes over `llm` and the
    (fixed, per-run) `tool_schemas` discovered from MCP, so the node
    signature stays the plain `(state) -> state` shape LangGraph expects.
    """

    def agent_node(state: AgentState) -> AgentState:
        round_count = state.get("round_count", 0)

        if round_count >= MAX_TOOL_ROUNDS:
            logger.warning("Max tool-call rounds (%d) reached; terminating.", MAX_TOOL_ROUNDS)
            return {
                **state,
                "pending_tool_calls": [],
                "final_answer": "I couldn't finish within the allowed number of tool calls.",
            }

        try:
            response = llm.chat(state["messages"], tools=tool_schemas, system=SYSTEM_PROMPT)
        except Exception as exc:  # LLM/API-level failure
            logger.error("LLM call failed: %s", exc)
            return {
                **state,
                "pending_tool_calls": [],
                "final_answer": f"Sorry, the LLM request failed: {exc}",
            }

        new_messages = state["messages"] + [{"role": "assistant", "content": response.raw_content}]

        if not response.tool_calls:
            return {
                **state,
                "messages": new_messages,
                "pending_tool_calls": [],
                "final_answer": response.text or "",
            }

        return {
            **state,
            "messages": new_messages,
            "pending_tool_calls": list(response.tool_calls),
            "round_count": round_count + 1,
            "final_answer": None,
        }

    return agent_node


def _make_tools_node(client: Any, known_tools: dict[str, Any]):
    """Build the `tools` node: executes every pending tool call through the
    MCP client and feeds the results back into the conversation."""

    async def tools_node(state: AgentState) -> AgentState:
        pending = state.get("pending_tool_calls", [])
        if not pending:
            # Defensive only -- the agent node never routes here without
            # pending calls, but a no-op is safer than a crash.
            return {**state, "pending_tool_calls": []}

        tool_results = [await _execute_tool_call(client, known_tools, call) for call in pending]
        new_messages = state["messages"] + [{"role": "user", "content": tool_results}]

        return {**state, "messages": new_messages, "pending_tool_calls": []}

    return tools_node


def _route_after_agent(state: AgentState) -> Literal["tools", "end"]:
    """agent -> tools if a tool call is pending, else agent -> END."""
    return "end" if state.get("final_answer") is not None else "tools"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(
    llm: LLMProvider,
    client: Any,
    known_tools: dict[str, Any],
    tool_schemas: list[dict[str, Any]],
):
    """
    Build and compile the Phase 4 graph:

        START -> agent --tools--> tools -> agent
                    \\--end--> END

    Exposed as a standalone function (rather than only inside
    `LangGraphAgent`) so tests can construct a graph directly against a
    fake LLM and/or a fake MCP client, without going through
    `connect_in_process()`.
    """
    graph = StateGraph(AgentState)
    graph.add_node("agent", _make_agent_node(llm, tool_schemas))
    graph.add_node("tools", _make_tools_node(client, known_tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ---------------------------------------------------------------------------
# Public agent
# ---------------------------------------------------------------------------


class LangGraphAgent:
    """Runs the LangGraph tool-calling graph for a single user question."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        # Lazily constructed if not injected, so building an agent never
        # requires an API key until answer() runs (same convention as
        # ToolAgent in tool_agent.py).
        self._llm = llm

    def _get_llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = get_default_provider()
        return self._llm

    async def answer(self, question: str) -> str:
        """Answer one question by running it through the compiled graph."""
        llm = self._get_llm()

        async with connect_multi_in_process() as client:
            discovered = await client.list_tools()
            known_tools = {tool.name: tool for tool in discovered.tools}
            tool_schemas = ToolAgent._mcp_tools_to_schemas(discovered.tools)

            graph = build_graph(llm, client, known_tools, tool_schemas)
            final_state = await graph.ainvoke(_initial_state(question))

            return final_state.get("final_answer") or ""


async def _run_cli() -> None:
    print("NexusAI Phase 5 (LangGraph + Document MCP + SQL MCP)")
    agent = LangGraphAgent()
    loop = asyncio.get_event_loop()

    while True:
        try:
            question = await loop.run_in_executor(None, input, "You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        question = question.strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            answer = await agent.answer(question)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        print(f"NexusAI: {answer}\n")


def main() -> None:
    asyncio.run(_run_cli())


if __name__ == "__main__":
    main()
