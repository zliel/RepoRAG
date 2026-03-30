from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE = "http://127.0.0.1:11434"


def _base_url(override: str | None) -> str:
    if override:
        return override.rstrip("/")
    return os.environ.get("OLLAMA_HOST", DEFAULT_BASE).rstrip("/")


class OllamaClient:
    """HTTP client for Ollama embed and chat APIs."""

    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = _base_url(base_url)
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {"model": model, "input": texts}
        try:
            r = self._client.post("/api/embed", json=payload)
            r.raise_for_status()
            data = r.json()
            embs = data.get("embeddings")
            if isinstance(embs, list) and len(embs) == len(texts):
                return [list(map(float, e)) for e in embs]
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
            logger.debug("/api/embed not available, trying /api/embeddings")
        except httpx.HTTPError:
            raise

        # Legacy single-prompt API (one request per text)
        out: list[list[float]] = []
        for t in texts:
            r = self._client.post("/api/embeddings", json={"model": model, "prompt": t})
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding")
            if not isinstance(emb, list):
                raise ValueError("Unexpected embeddings response shape")
            out.append(list(map(float, emb)))
        return out

    def chat(self, model: str, messages: list[dict[str, str]], stream: bool = False) -> str:
        r = self._client.post(
            "/api/chat",
            json={"model": model, "messages": messages, "stream": stream},
        )
        r.raise_for_status()
        data = r.json()
        msg = data.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise ValueError("Unexpected chat response: missing message.content")
        return content
