from __future__ import annotations

import fnmatch
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

# Separate entries that contain glob characters (need fnmatch) from simple names
_IGNORED_DIRS_GLOB = {d for d in _IGNORED_DIRS if set(d) & {"*", "?", "[", "]"}}
_IGNORED_DIRS_EXACT = _IGNORED_DIRS - _IGNORED_DIRS_GLOB


def _load_gitignore(root: Path) -> list[str]:
    """Load .gitignore patterns from *root*/.gitignore (if it exists).

    Returns a list of raw pattern lines (comments and blanks are skipped).
    """
    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists():
        return []
    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _matches_any_pattern(path_str: str, patterns: list[str]) -> bool:
    """Check whether *path_str* matches any of the given glob patterns.

    Handles simple ``.gitignore``-style conventions:
    * A leading ``/`` is stripped (the path is always relative so anchoring
      to root is the default behaviour).
    * A trailing ``/`` is stripped (the caller matches files, not directories).
    * Patterns that do **not** contain a ``/`` are also matched against every
      single path component (to support both ``*.pyc`` and ``__pycache__``
      semantics).
    """
    for pattern in patterns:
        # Strip .gitignore prefixes/suffixes that are irrelevant for file matching
        if pattern.startswith("/"):
            pattern = pattern[1:]
        pattern = pattern.rstrip("/")

        # Full relative-path match
        if fnmatch.fnmatch(path_str, pattern):
            return True

        # Pattern without a directory separator: also match individual components
        # so that e.g. "*.egg-info" matches "foo.egg-info/bar.py" and
        # "__pycache__" matches "src/__pycache__/cache.py".
        if "/" not in pattern:
            for part in Path(path_str).parts:
                if fnmatch.fnmatch(part, pattern):
                    return True

    return False


def walk_supported_files(
    root: Path,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    """
    Recursively list all files that can be parsed (based on registered extensions),
    sorted by relative path. Skips symlinks and common ignored directories.

    Exclusion logic (applied in order):

    1. Any path component starting with ``.`` is skipped (hidden files/directories).
    2. Simple (non-glob) entries in ``_IGNORED_DIRS`` are checked via exact
       component matching (existing behaviour).
    3. Glob entries in ``_IGNORED_DIRS`` (e.g. ``*.egg-info``) are checked via
       :func:`fnmatch.fnmatch` against every path component.
    4. Patterns from ``.gitignore`` (in *root*) are loaded and applied.
    5. *exclude_patterns* are appended and applied.
    6. Negation patterns (lines starting with ``!``) override any exclusion.

    Args:
        root: Project root directory to walk.
        exclude_patterns:
            Additional glob patterns to exclude
            (e.g. ``["tests/", "*.pyc"]``).
    """
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    extensions = set(get_all_extensions())
    if not extensions:
        logger.warning("No parsers registered; walker will return no files")

    gitignore_patterns = _load_gitignore(root)
    user_patterns = list(exclude_patterns or [])

    # Separate negation (!prefix) from regular exclusion patterns
    negations = [p[1:] for p in (*gitignore_patterns, *user_patterns) if p.startswith("!")]
    exclusions = [p for p in (*gitignore_patterns, *user_patterns) if not p.startswith("!")]

    paths: list[Path] = []
    for p in root.rglob("*"):
        try:
            if not p.is_file() or p.is_symlink():
                continue

            parts = p.parts

            # Skip hidden files/directories (names starting with ".")
            if any(part.startswith(".") for part in parts):
                continue

            # Skip exact-match ignored directories (existing behaviour)
            if any(part in _IGNORED_DIRS_EXACT for part in parts):
                continue

            # Skip glob-pattern ignored directories (fixes the *.egg-info bug)
            if any(fnmatch.fnmatch(part, g) for g in _IGNORED_DIRS_GLOB for part in parts):
                continue

            # Gitignore + user exclusion patterns (with negation support)
            rel_path = p.relative_to(root).as_posix()
            if _matches_any_pattern(rel_path, exclusions):
                if not _matches_any_pattern(rel_path, negations):
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
