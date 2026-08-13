"""
Tool-Calling Agent (Phase 2)
==============================

    User -> LLM -> (decides if a tool is needed) -> MCP Client
          -> Document MCP Server -> MCP Tool -> Tool Result
          -> LLM -> Final Answer

The agent talks to documents *only* through the MCP client. It never
imports `document_server.py` directly, and it never executes anything the
LLM asks for unless that exact tool name was discovered from the live MCP
`list_tools()` call.

Run directly with:  python -m nexusai.agent.tool_agent
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from nexusai.client.mcp_client import connect_in_process
from nexusai.llm.provider import LLMProvider, get_default_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nexusai.tool_agent")

SYSTEM_PROMPT = (
    "You are NexusAI, an assistant with access to a small document library "
    "through tools. Use a tool when you need document content to answer "
    "accurately; answer directly when you already know the answer.\n\n"
    "You do not know the exact filenames in the library in advance — never "
    "guess or invent one. If you need a specific document but are not "
    "certain of its exact filename, call list_documents() first to see "
    "what is actually available, then use the exact filename it returns.\n\n"
    "For document questions, prefer semantic_search(query, top_k) — it "
    "retrieves the chunks most relevant to the meaning of the question, "
    "even if they don't share its exact wording. Use search_documents() "
    "instead only when you need an exact word, phrase, or identifier match. "
    "When answering from retrieved chunks:\n"
    "- Ground your answer only in the retrieved content — never invent or "
    "assume document facts that were not actually returned by a tool.\n"
    "- Mention which document(s) (by filename) the answer came from when "
    "it's useful for the user to know the source.\n"
    "- If the retrieved chunks don't contain enough information to answer "
    "confidently, say so plainly instead of guessing."
)

# Hard cap on LLM <-> tool round trips per question, so a confused LLM can't
# loop forever.
MAX_TOOL_ROUNDS = 5


def _validate_arguments(schema: dict[str, Any] | None, arguments: Any) -> str | None:
    """
    Minimal JSON-Schema-shaped validation: arguments must be an object, and
    every field the schema marks `required` must be present. Returns an
    error message, or None if the arguments look valid.
    """
    if not isinstance(arguments, dict):
        return "tool arguments must be a JSON object"

    if not schema:
        return None

    required = schema.get("required", [])
    missing = [name for name in required if name not in arguments]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"

    return None


class ToolAgent:
    """Runs the LLM <-> MCP tool-calling loop for a single user question."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        # Lazily constructed if not injected, so building an agent (and
        # discovering tools) never requires an API key until chat() runs.
        self._llm = llm

    def _get_llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = get_default_provider()
        return self._llm

    @staticmethod
    def _mcp_tools_to_schemas(tools: list[Any]) -> list[dict[str, Any]]:
        """
        Convert discovered MCP tools into the provider-agnostic
        (Anthropic-shaped) tool-schema dicts used internally; each
        LLMProvider translates these to its own wire format.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]

    @staticmethod
    def _tool_result_to_text(result: Any) -> str:
        """Flatten an MCP CallToolResult into text suitable for the LLM."""
        if result.structured_content is not None:
            return json.dumps(result.structured_content, default=str)
        return "\n".join(getattr(block, "text", "") for block in result.content)

    async def _execute_tool_call(
        self,
        client: Any,
        known_tools: dict[str, Any],
        call: Any,
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

        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": self._tool_result_to_text(result),
            "is_error": result.is_error,
        }

    async def answer(self, question: str) -> str:
        """Answer one question, calling MCP tools through the LLM as needed."""
        llm = self._get_llm()

        async with connect_in_process() as client:
            discovered = await client.list_tools()
            known_tools = {tool.name: tool for tool in discovered.tools}
            tool_schemas = self._mcp_tools_to_schemas(discovered.tools)

            messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

            for _ in range(MAX_TOOL_ROUNDS):
                try:
                    response = llm.chat(messages, tools=tool_schemas, system=SYSTEM_PROMPT)
                except Exception as exc:  # LLM/API-level failure
                    logger.error("LLM call failed: %s", exc)
                    return f"Sorry, the LLM request failed: {exc}"

                if not response.tool_calls:
                    return response.text or ""

                messages.append({"role": "assistant", "content": response.raw_content})

                tool_results = [
                    await self._execute_tool_call(client, known_tools, call)
                    for call in response.tool_calls
                ]
                messages.append({"role": "user", "content": tool_results})

            return "I couldn't finish within the allowed number of tool calls."


async def _run_cli() -> None:
    print("NexusAI Phase 2")
    agent = ToolAgent()
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
