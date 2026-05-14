from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import httpx

from reporag.constants import DEFAULT_NUM_CTX, DEFAULT_TEMPERATURE
from reporag.llm.backends.base import LLMBackend
from reporag.llm.retry import with_retry

logger = logging.getLogger(__name__)

DEFAULT_BASE = "http://127.0.0.1:11434"


def _base_url(override: str | None) -> str:
    if override:
        return override.rstrip("/")
    import os

    return os.environ.get("OLLAMA_HOST", DEFAULT_BASE).rstrip("/")


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        self.base_url = _base_url(base_url)
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        if not texts:
            return []

        payload: dict[str, Any] = {"model": model, "input": texts}
        try:
            r = with_retry(
                lambda: self._client.post("/api/embed", json=payload),
                max_retries=self.max_retries,
                backoff_factor=self.backoff_factor,
            )
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

        # Legacy per-text fallback
        out: list[list[float]] = []
        for t in texts:
            r = with_retry(
                lambda t=t: self._client.post(
                    "/api/embeddings", json={"model": model, "prompt": t}
                ),
                max_retries=self.max_retries,
                backoff_factor=self.backoff_factor,
            )
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding")
            if not isinstance(emb, list):
                raise ValueError("Unexpected embeddings response shape")
            out.append(list(map(float, emb)))
        return out

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        temperature: float | None = None,
    ) -> str:
        r = with_retry(
            lambda: self._client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                    "num_ctx": DEFAULT_NUM_CTX,
                    "options": {
                        "temperature": (
                            temperature
                            if temperature is not None
                            else DEFAULT_TEMPERATURE
                        )
                    },
                },
            ),
            max_retries=self.max_retries,
            backoff_factor=self.backoff_factor,
        )
        r.raise_for_status()
        data = r.json()
        msg = data.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise ValueError("Unexpected chat response: missing message.content")
        return content

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
            "num_ctx": DEFAULT_NUM_CTX,
            "options": {
                "temperature": (
                    temperature if temperature is not None else DEFAULT_TEMPERATURE
                )
            },
        }

        # Retry connection establishment, then stream tokens
        def _open_stream() -> tuple[httpx.StreamContextManager, httpx.Response]:
            cm = self._client.stream("POST", "/api/chat", json=payload)
            # Entering the context manager actually sends the request.
            response = cm.__enter__()
            if response.is_error:
                cm.__exit__(None, None, None)
                response.raise_for_status()
            return cm, response

        cm, response = with_retry(
            _open_stream,
            max_retries=self.max_retries,
            backoff_factor=self.backoff_factor,
        )
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed JSON line: %s", line)
                        continue
                    msg = data.get("message") or {}
                    content = msg.get("content", "")
                    if content:
                        yield content
        finally:
            cm.__exit__(None, None, None)

    def close(self) -> None:
        self._client.close()
