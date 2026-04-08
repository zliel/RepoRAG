from __future__ import annotations

import logging
from pathlib import Path

from reporag.parsing.registry import (
    get_all_extensions,
    get_fallback_parser,
    get_parser_for_extension,
)

logger = logging.getLogger(__name__)

_IGNORED_DIRS = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
}


def walk_supported_files(root: Path) -> list[Path]:
    """
    Recursively list all files that can be parsed (based on registered extensions),
    sorted by relative path. Skips symlinks and common ignored directories.
    """
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    extensions = set(get_all_extensions())
    if not extensions:
        logger.warning("No parsers registered; walker will return no files")

    paths: list[Path] = []
    for p in root.rglob("*"):
        try:
            if not p.is_file() or p.is_symlink():
                continue
            if any(part.startswith(".") or part in _IGNORED_DIRS for part in p.parts):
                continue
            ext = p.suffix.lower()
            if ext in extensions:
                paths.append(p.resolve())
        except OSError as e:
            logger.warning("Skipping path %s: %s", p, e)

    paths.sort(key=lambda x: x.relative_to(root).as_posix())
    return paths


def parse_file(path: Path, root: Path) -> list:
    """
    Parse a file using the appropriate parser based on extension.
    Returns list of Chunks, or empty list if parsing fails.
    """
    ext = path.suffix.lower()
    parser = get_parser_for_extension(ext)

    if parser is None:
        fallback = get_fallback_parser()
        if fallback is not None:
            parser = fallback
        else:
            logger.debug("No parser for %s (no fallback configured)", ext)
            return []

    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
        source_bytes = path.read_bytes()
        return parser.extract_chunks(rel, source_bytes)
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Failed to parse %s: %s", path, e)
        return []


def get_supported_extensions() -> list[str]:
    """Return sorted list of all supported file extensions."""
    return sorted(get_all_extensions())


def get_supported_extensions_display() -> str:
    """Return human-readable string of supported extensions."""
    exts = get_supported_extensions()
    return ", ".join(exts) if exts else "none"
