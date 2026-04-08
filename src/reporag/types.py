from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """A code unit extracted from a source file (function, class, or text block)."""

    path: str
    symbol_name: str
    start_line: int
    end_line: int
    text: str
    language: str
