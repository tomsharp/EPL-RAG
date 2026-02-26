import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


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

        response = await self._client.post("/v1/chat/completions", json=body)
        response.raise_for_status()
        choice = response.json()["choices"][0]
        return choice["message"], choice["finish_reason"]

    async def generate(self, messages: list[dict], max_tokens: int = 512) -> str:
        """Convenience wrapper — returns content string only (no tool support)."""
        msg, _ = await self.complete(messages, max_tokens=max_tokens)
        return (msg.get("content") or "").strip()

    async def close(self) -> None:
        await self._client.aclose()
