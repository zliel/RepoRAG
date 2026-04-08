from __future__ import annotations

import logging

from reporag.parsing.base import ParserProtocol
from reporag.parsing.registry import set_fallback_parser
from reporag.types import Chunk

logger = logging.getLogger(__name__)

_MAX_CHUNK_SIZE = 5000


class TextChunker(ParserProtocol):
    """Parser for plain text files (markdown, configs, scripts, etc.)."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return ()

    @property
    def language_name(self) -> str:
        return "plaintext"

    def extract_chunks(self, relative_path: str, source_bytes: bytes) -> list[Chunk]:
        """Chunk plain text into line-based segments."""
        try:
            text = source_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning("Failed to decode %s: %s", relative_path, e)
            return []

        chunks: list[Chunk] = []
        lines = text.split("\n")
        chunk_lines: list[str] = []
        start_line = 1

        for i, line in enumerate(lines, start=1):
            chunk_lines.append(line)
            if len("\n".join(chunk_lines)) >= _MAX_CHUNK_SIZE or i == len(lines):
                if chunk_lines:
                    chunks.append(
                        Chunk(
                            path=relative_path,
                            symbol_name=f"lines {start_line}-{i}",
                            start_line=start_line,
                            end_line=i,
                            text="\n".join(chunk_lines),
                            language="plaintext",
                        )
                    )
                    chunk_lines = []
                    start_line = i + 1

        if chunk_lines:
            chunks.append(
                Chunk(
                    path=relative_path,
                    symbol_name=f"lines {start_line}-{len(lines)}",
                    start_line=start_line,
                    end_line=len(lines),
                    text="\n".join(chunk_lines),
                    language="plaintext",
                )
            )

        return chunks


set_fallback_parser(TextChunker())
