"""Code graph extraction and retrieval utilities."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# =============================================================================
# Import Extraction - Language-specific patterns
# =============================================================================

# Python: from x import y | import x
_PYTHON_IMPORT_RE = re.compile(
    r"^(?:from\s+([\w.]+)\s+import|(?:import\s+([\w.]+)))",
    re.MULTILINE,
)

# JavaScript/TypeScript: import x from 'y' | require('y') | export from 'y'
_JS_IMPORT_RE = re.compile(
    r"(?:import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]|"
    r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)|"
    r"export\s+\{[^}]*\bfrom\s+['\"]([^'\"]+)['\"])",
    re.MULTILINE,
)

# Rust: use x::y | extern crate x | mod x
_RUST_IMPORT_RE = re.compile(
    r"^(?:use\s+([\w:]+)|mod\s+([\w_]+))$",
    re.MULTILINE,
)

# Go: import "x"
_GO_IMPORT_RE = re.compile(r'"([^"]+)"', re.MULTILINE)


# =============================================================================
# Call Extraction - Function/method calls
# =============================================================================

# Python class/function calls (PascalCase identifiers)
_PYTHON_CALL_RE = re.compile(r"\b([A-Z][\w]*)\s*\(")

# JavaScript/TypeScript function calls (camelCase identifiers)
_JS_CALL_RE = re.compile(r"\b([a-z][A-Za-z0-9]*)\s*\(")


def extract_imports_from_source(source_code: str, language: str) -> list[str]:
    """
    Extract import/module references from source code.

    Args:
        source_code: The raw source code (full file content).
        language: Programming language (python, javascript, typescript, rust, go).

    Returns:
        List of import/module reference strings.
    """
    imports: set[str] = set()

    if language == "python":
        for match in _PYTHON_IMPORT_RE.finditer(source_code):
            for group in match.groups():
                if group:
                    # Handle "from foo.bar import baz" -> "foo.bar"
                    imports.add(group)

    elif language in ("javascript", "typescript"):
        for match in _JS_IMPORT_RE.finditer(source_code):
            for group in match.groups():
                if group:
                    imports.add(group)

    elif language == "rust":
        for match in _RUST_IMPORT_RE.finditer(source_code):
            for group in match.groups():
                if group and "prelude" not in group.lower():
                    imports.add(group)

    elif language == "go":
        # Go uses a specific import block
        import_block_match = re.search(r"import\s*\((.*?)\)", source_code, re.DOTALL)
        if import_block_match:
            inner = import_block_match.group(1)
            for line in inner.splitlines():
                m = _GO_IMPORT_RE.search(line)
                if m:
                    imports.add(m.group(1))

    return list(imports)


def extract_calls_from_chunk(text: str, language: str) -> list[str]:
    """
    Extract function/method calls from code chunk text.

    Args:
        text: The code chunk text (extracted function/class).
        language: Programming language.

    Returns:
        List of called identifier names.
    """
    calls: set[str] = set()

    if language == "python":
        for match in _PYTHON_CALL_RE.finditer(text):
            calls.add(match.group(1))

    elif language in ("javascript", "typescript"):
        for match in _JS_CALL_RE.finditer(text):
            calls.add(match.group(1))

    return list(calls)


def build_chunk_symbol_map(
    chunk_rows: list[tuple],
) -> dict[str, int]:
    """
    Build a mapping from symbol names to chunk IDs.

    Args:
        chunk_rows: List of (id, path, symbol, text, source_text, language) tuples.

    Returns:
        Dict mapping symbol name -> chunk ID.
    """
    return {str(row[2]): int(row[0]) for row in chunk_rows if row[2]}


def get_callers(
    symbol: str,
    chunk_rows: list[tuple],
) -> list[str]:
    """
    Find chunks that call the given symbol.

    Args:
        symbol: The symbol name being called.
        chunk_rows: List of (id, path, symbol, text, source_text, language) tuples.

    Returns:
        List of chunk IDs that make calls to this symbol.
    """
    callers: list[str] = []
    for row in chunk_rows:
        _, _, _, text, _, language = row
        if text and symbol:
            calls = extract_calls_from_chunk(text, language)
            if symbol in calls:
                callers.append(str(row[0]))
    return callers


def get_callees(
    chunk_rows: list[tuple],
) -> dict[str, list[str]]:
    """
    Build mapping from chunk ID to symbols it calls.

    Args:
        chunk_rows: List of (id, path, symbol, text, source_text, language) tuples.

    Returns:
        Dict mapping chunk ID -> list of called symbol names.
    """
    callees: dict[str, list[str]] = {}
    for row in chunk_rows:
        chunk_id = str(row[0])
        _, _, _, text, _, language = row
        if text:
            callees[chunk_id] = extract_calls_from_chunk(text, language)
        else:
            callees[chunk_id] = []
    return callees
