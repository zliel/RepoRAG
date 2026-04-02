from __future__ import annotations

import logging
from typing import Any

import httpx

from reporag.llm.backends.base import LLMBackend

logger = logging.getLogger(__name__)


class OpenAICompatBackend(LLMBackend):
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {"model": model, "input": texts}
        r = self._client.post("/v1/embeddings", json=payload)
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("data", [])
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
            )
        return [list(map(float, e.get("embedding", []))) for e in embeddings]

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        temperature: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        r = self._client.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No choices in chat response")
        return choices[0].get("message", {}).get("content", "")

    def close(self) -> None:
        self._client.close()
