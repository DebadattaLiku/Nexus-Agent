"""
LLM Provider (Phase 2)
========================

Thin abstraction around an LLM's chat/tool-calling API so the underlying
provider can be swapped later without touching agent code.

Two concrete providers exist:

- `GroqProvider` (default) — backed by Groq's OpenAI-compatible chat
  completions API. Groq has a genuinely free, no-credit-card developer tier
  (rate-limited, not trial-limited), which is why it's the default for local
  development. See `.env.example` for setup.
- `AnthropicProvider` — backed by the Anthropic Messages API, kept available
  for anyone who wants to opt back into it (e.g. in production).

Both providers speak the *same* internal message convention (Anthropic's
content-block shape: `{"type": "text", ...}` / `{"type": "tool_use", ...}` /
`{"type": "tool_result", ...}`) so `tool_agent.py` never needs to know which
provider is active. `GroqProvider` translates that shape to/from the
OpenAI-style wire format Groq expects, internally.

Which provider is used is controlled by the `LLM_PROVIDER` environment
variable (`groq` or `anthropic`, defaults to `groq`) via `get_default_provider()`.
Credentials are read from the environment (see `.env.example`) and are never
logged.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("nexusai.llm.provider")

DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Matches Llama's "pythonic" built-in tool-call syntax, e.g.
#   <function=search_documents{"query": "RAG", "case_sensitive": false}>
# Some Groq-hosted models occasionally emit this instead of a proper
# structured `tool_calls` entry. It shows up in two places:
#   1. As plain text inside a normal (HTTP 200) response.
#   2. Inside the `failed_generation` field of a 400 "Failed to call a
#      function" error, when Groq's own parser also fails to accept it.
# This regex only locates the `<function=NAME` opening tag; the JSON
# argument object that follows is extracted separately via brace-matching,
# since arguments may themselves contain nested `{}`.
_FUNCTION_TAG_RE = re.compile(r"<function=([A-Za-z0-9_\-]+)")


def _parse_pythonic_tool_calls(text: str) -> list["ToolCall"]:
    """
    Recover tool calls from Llama's "pythonic" `<function=NAME{...}>` syntax.

    Generic over any tool name and any JSON-serializable arguments (no
    special-casing of specific tool names) — this is what lets it work for
    every tool discovered from the MCP server, not just the ones seen during
    development.
    """
    calls: list[ToolCall] = []
    for match in _FUNCTION_TAG_RE.finditer(text):
        name = match.group(1)
        start = match.end()

        # Groq/Llama may emit either:
        # <function=name{"key": "value"}>
        # or:
        # <function=name={"key": "value"}>
        if start < len(text) and text[start] == "=":
            start += 1

        if start >= len(text) or text[start] != "{":
            continue

        depth = 0
        end = None
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue  # unterminated JSON object; can't safely recover

        try:
            arguments = json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
        if not isinstance(arguments, dict):
            continue

        calls.append(ToolCall(id=f"pythonic_{uuid.uuid4().hex[:8]}", name=name, arguments=arguments))
    return calls


@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized response from a provider's chat call."""

    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    # Provider-native assistant content blocks, needed to replay the
    # assistant turn back into the conversation history verbatim.
    raw_content: Any = None


class LLMProvider(Protocol):
    """Interface every LLM provider must implement."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Send a conversation (+ optional tool schemas) and get one response."""
        ...


class AnthropicProvider:
    """LLMProvider backed by the Anthropic Messages API."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        import anthropic  # imported lazily so unit tests never need this installed key

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model or DEFAULT_ANTHROPIC_MODEL
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_content: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                raw_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input or {}))
                raw_content.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                )

        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            raw_content=raw_content,
        )


class GroqProvider:
    """
    LLMProvider backed by Groq's OpenAI-compatible chat completions API
    (https://api.groq.com/openai/v1). Groq's free developer tier requires no
    credit card and supports native tool/function calling on
    `llama-3.3-70b-versatile` (the default model here).

    `chat()` accepts and returns messages in the same Anthropic content-block
    shape used throughout this codebase, translating to/from the OpenAI-style
    wire format internally, so callers (and tests) never need to care which
    provider is configured.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        import groq  # imported lazily so unit tests never need this installed key

        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in "
                "with a free key from https://console.groq.com/keys."
            )
        self._client = groq.Groq(api_key=api_key)
        self.model = model or DEFAULT_GROQ_MODEL
        self.max_tokens = max_tokens

    # -- message/tool translation: our content-block shape <-> OpenAI shape --

    @staticmethod
    def _tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _messages_to_openai(
        messages: list[dict[str, Any]], system: str | None
    ) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []
        if system:
            openai_messages.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                # Plain-text user/assistant turn (e.g. the initial question).
                openai_messages.append({"role": role, "content": content})
                continue

            if role == "assistant":
                text_parts = [b["text"] for b in content if b.get("type") == "text"]
                tool_calls = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {
                            "name": b["name"],
                            "arguments": json.dumps(b.get("input") or {}),
                        },
                    }
                    for b in content
                    if b.get("type") == "tool_use"
                ]
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                openai_messages.append(assistant_msg)
                continue

            # role == "user" carrying a list of tool_result blocks: each one
            # becomes its own OpenAI-style "tool" message.
            for block in content:
                if block.get("type") == "tool_result":
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block.get("content") or "",
                        }
                    )

        return openai_messages

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._messages_to_openai(messages, system),
        }
        if tools:
            kwargs["tools"] = self._tools_to_openai(tools)
            # Explicit (rather than relying on Groq's default) so tool-call
            # routing is deterministic across models/model updates.
            kwargs["tool_choice"] = "auto"

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # a real Groq SDK error (e.g. BadRequestError)
            recovered = self._recover_from_failed_generation(exc)
            if recovered is not None:
                return recovered
            raise

        message = response.choices[0].message

        raw_content: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            arguments = self._parse_tool_call_arguments(tc.function.arguments)
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))
            raw_content.append(
                {"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": arguments}
            )

        if not tool_calls and message.content:
            # Defense in depth: some models emit the pythonic
            # `<function=...>` syntax as plain assistant text on an
            # otherwise-successful (HTTP 200) response, rather than in the
            # structured `tool_calls` field. Recover it the same way we
            # recover from a 400 `failed_generation` payload.
            recovered_calls = _parse_pythonic_tool_calls(message.content)
            if recovered_calls:
                logger.warning(
                    "Groq returned a pythonic tool call as plain text instead "
                    "of structured tool_calls; recovered %d call(s).",
                    len(recovered_calls),
                )
                tool_calls = recovered_calls
                raw_content = [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in tool_calls
                ]

        if not tool_calls and message.content:
            raw_content.append({"type": "text", "text": message.content})

        return LLMResponse(
            text=None if tool_calls else message.content,
            tool_calls=tool_calls,
            stop_reason=response.choices[0].finish_reason,
            raw_content=raw_content,
        )

    @staticmethod
    def _parse_tool_call_arguments(raw_arguments: str | None) -> dict[str, Any]:
        """Best-effort parse of a tool call's argument string. Never raises —
        malformed/empty JSON becomes `{}`, which downstream argument
        validation (in `ToolAgent`) will then reject as missing required
        fields, rather than crashing the request."""
        if not raw_arguments:
            return {}
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            logger.warning("Could not parse tool call arguments as JSON: %r", raw_arguments)
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _extract_failed_generation(exc: Exception) -> str | None:
            """Extract Groq's failed_generation from the exception body or message."""
            body = getattr(exc, "body", None)
    
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict):
                    failed_generation = error.get("failed_generation")
                    if isinstance(failed_generation, str):
                        return failed_generation
    
            # Some Groq SDK/error versions expose the payload only through
            # the exception string rather than exc.body.
            message = str(exc)
            marker = "'failed_generation': "
            if marker in message:
                remainder = message.split(marker, 1)[1]
                if remainder.startswith("'"):
                    remainder = remainder[1:]
                    end = remainder.find("'}")
                    if end != -1:
                        return remainder[:end]
    
            return None    
    


    def _recover_from_failed_generation(self, exc: Exception) -> LLMResponse | None:
        """
        Groq can reject a request with HTTP 400 ("Failed to call a
        function") when the model's own tool-call generation doesn't match
        the strict format Groq's parser expects. The response body still
        contains what the model *tried* to generate, in
        `error.failed_generation` — usually Llama's pythonic
        `<function=NAME{...}>` syntax. Rather than surfacing this as a hard
        failure, recover the tool call the model clearly intended, so the
        MCP tool still gets called and the user still gets an answer.

        Returns None (meaning "not recoverable, re-raise the original
        error") for any exception that isn't this specific shape, or whose
        `failed_generation` text doesn't contain a parseable tool call.
        """
        failed_generation = self._extract_failed_generation(exc)
        if not failed_generation:
            return None

        tool_calls = _parse_pythonic_tool_calls(failed_generation)
        if not tool_calls:
            return None

        logger.warning(
            "Groq rejected native tool-call generation (%s); recovered %d "
            "call(s) from failed_generation instead of failing the request.",
            exc,
            len(tool_calls),
        )
        raw_content = [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
            for c in tool_calls
        ]
        return LLMResponse(text=None, tool_calls=tool_calls, stop_reason="tool_use", raw_content=raw_content)


def get_default_provider() -> LLMProvider:
    """
    Build the LLM provider selected by the `LLM_PROVIDER` env var
    (`groq` | `anthropic`, defaults to `groq`). This is the single place
    that decides which concrete provider backs the agent, so switching
    providers never requires touching `tool_agent.py`.
    """
    name = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
    if name == "groq":
        return GroqProvider()
    if name == "anthropic":
        return AnthropicProvider()
    raise RuntimeError(
        f"Unknown LLM_PROVIDER: {name!r}. Supported values are 'groq' or 'anthropic'."
    )
