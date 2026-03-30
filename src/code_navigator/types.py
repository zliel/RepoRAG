from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """A code unit extracted from a Python file (function or class)."""

    path: str
    symbol_name: str
    start_line: int
    end_line: int
    text: str
