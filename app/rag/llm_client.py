import json
import logging
from typing import AsyncGenerator

import httpx
from opentelemetry import trace
from openinference.semconv.trace import MessageAttributes, SpanAttributes

from app.config import settings

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class LLMClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        logger.info("LLM: OpenAI %s", settings.openai_model)

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 512,
    ) -> tuple[dict, str]:
        """Call chat completions and return (message_dict, finish_reason).

        The message_dict is the raw OpenAI message object — it may contain
        `content` (str | None) and/or `tool_calls` (list | None) depending
        on `finish_reason` ("stop" vs "tool_calls").
        """
        body: dict = {
            "model": settings.openai_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.9,
        }
        if tools:
            body["tools"] = tools

        with tracer.start_as_current_span("llm.complete") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, settings.openai_model)
            span.set_attribute(
                SpanAttributes.LLM_INVOCATION_PARAMETERS,
                json.dumps({"temperature": 0.9, "max_tokens": max_tokens}),
            )

            # Record input messages
            for i, msg in enumerate(messages):
                prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{i}"
                span.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", msg["role"])
                content = msg.get("content") or ""
                if content:
                    span.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}", content)

            response = await self._client.post("/v1/chat/completions", json=body)
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            out = choice["message"]

            # Record output message
            span.set_attribute(
                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}",
                "assistant",
            )
            if out.get("content"):
                span.set_attribute(
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}",
                    out["content"],
                )

            # Record token usage
            usage = data.get("usage", {})
            if usage:
                span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage.get("prompt_tokens", 0))
                span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, usage.get("completion_tokens", 0))
                span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, usage.get("total_tokens", 0))

            return out, choice["finish_reason"]

    async def stream_complete(
        self,
        messages: list[dict],
        max_tokens: int = 512,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completions, yielding text chunks as they arrive."""
        body = {
            "model": settings.openai_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.9,
            "stream": True,
        }
        async with self._client.stream("POST", "/v1/chat/completions", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    content = chunk["choices"][0]["delta"].get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def generate(self, messages: list[dict], max_tokens: int = 512) -> str:
        """Convenience wrapper — returns content string only (no tool support)."""
        msg, _ = await self.complete(messages, max_tokens=max_tokens)
        return (msg.get("content") or "").strip()

    async def close(self) -> None:
        await self._client.aclose()
