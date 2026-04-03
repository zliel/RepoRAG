from reporag.llm.backends import (
    BackendType,
    LLMBackend,
    OllamaBackend,
    OpenAICompatBackend,
    create_backend,
)

OllamaClient = OllamaBackend

__all__ = [
    "LLMBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
    "BackendType",
    "create_backend",
    "OllamaClient",
]
