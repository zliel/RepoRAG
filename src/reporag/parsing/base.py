from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from reporag.types import Chunk


class ParserProtocol(ABC):
    """Abstract base class for language-specific parsers."""

    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        """Return tuple of file extensions this parser handles (e.g., '.py', '.js')."""

    @property
    @abstractmethod
    def language_name(self) -> str:
        """Return the language identifier (e.g., 'python', 'javascript', 'plaintext')."""

    @abstractmethod
    def extract_chunks(self, relative_path: str, source_bytes: bytes) -> list[Chunk]:
        """Parse source bytes and return list of chunks (functions, classes, etc.)."""

    def extract_chunks_from_file(self, path: Path, root: Path) -> list[Chunk] | None:
        """Read and parse a file. Returns None if the file cannot be read or parsed critically."""
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
            source_bytes = path.read_bytes()
        except OSError as e:
            import logging

            logging.getLogger(__name__).warning("Cannot read %s: %s", path, e)
            return None
        try:
            return self.extract_chunks(rel, source_bytes)
        except (UnicodeDecodeError, OSError) as e:
            import logging

            logging.getLogger(__name__).warning("Failed to parse %s: %s", path, e)
            return None
