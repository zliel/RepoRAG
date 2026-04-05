from __future__ import annotations

import json
import logging
from collections.abc import Iterator
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

    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        with self._client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed SSE data: %s", data_str)
                        continue
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content

    def close(self) -> None:
        self._client.close()
