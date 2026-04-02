from __future__ import annotations

from reporag.llm.backends.base import LLMBackend
from reporag.llm.backends.factory import BackendType, create_backend
from reporag.llm.backends.ollama import OllamaBackend
from reporag.llm.backends.openai_compat import OpenAICompatBackend

__all__ = [
    "LLMBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
    "BackendType",
    "create_backend",
]
