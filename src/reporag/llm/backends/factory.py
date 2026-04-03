from __future__ import annotations

import logging
from typing import Literal

from reporag.llm.backends.base import LLMBackend
from reporag.llm.backends.ollama import OllamaBackend
from reporag.llm.backends.openai_compat import OpenAICompatBackend

logger = logging.getLogger(__name__)

BackendType = Literal["ollama", "vllm", "llamacpp", "lmstudio", "lmstudio-local"]


def create_backend(
    backend: BackendType,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMBackend:
    match backend:
        case "ollama":
            return OllamaBackend(base_url=base_url)
        case "vllm" | "llamacpp" | "lmstudio" | "lmstudio-local":
            if not base_url:
                msg = f"base_url required for {backend} backend"
                raise ValueError(msg)
            return OpenAICompatBackend(base_url=base_url, api_key=api_key)
        case _:
            msg = f"Unknown backend: {backend}"
            raise ValueError(msg)
