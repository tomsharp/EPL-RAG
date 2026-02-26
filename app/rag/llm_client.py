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

    async def generate(self, messages: list[dict], max_tokens: int = 512) -> str:
        response = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": settings.openai_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.9,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    async def close(self) -> None:
        await self._client.aclose()
