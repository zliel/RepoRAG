from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMBackend(ABC):
    @abstractmethod
    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        temperature: float | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> Iterator[str]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
