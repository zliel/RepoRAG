from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reporag.retrieval.search import RetrievedChunk

# First ```mermaid ... ``` block (case-insensitive language tag).
_MERMAID_FENCE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Matches patterns like "CITATION id=3" or "CITATION id=3 (symbol_name)" or '"CITATION id=3"'.
_CITATION_MARKER = re.compile(r'"?CITATION\s+id=(\d+)(?:\s*\([^)]*\))?"?', re.IGNORECASE)

# Supported diagram headers (first meaningful line of body).
_MERMAID_HEADER = re.compile(
    r"^\s*(flowchart|sequenceDiagram|classDiagram|graph|stateDiagram|erDiagram|mindmap|timeline)\b",
    re.MULTILINE | re.IGNORECASE,
)


def extract_mermaid_fence(text: str) -> tuple[str | None, str]:
    """
    Extract the first fenced mermaid block.
    Returns (inner_body_or_none, preamble_text_before_fence).
    If no fence, returns (None, full stripped text).
    """
    m = _MERMAID_FENCE.search(text)
    if not m:
        return None, text.strip()
    start = m.start()
    preamble = text[:start].strip()
    body = m.group(1).strip("\n")
    return body, preamble


def validate_mermaid_light(body: str) -> bool:
    """True if body looks like a known Mermaid diagram type (heuristic)."""
    if not body.strip():
        return False
    return _MERMAID_HEADER.search(body) is not None


def normalize_diagram_markdown(preamble: str, mermaid_body: str) -> str:
    """Build a single Markdown document with optional intro and one mermaid fence."""
    parts: list[str] = []
    if preamble:
        parts.append(preamble.strip())
        parts.append("")
    parts.append("```mermaid")
    parts.append(mermaid_body.strip("\n"))
    parts.append("```")
    parts.append("")
    return "\n".join(parts)


def resolve_citation_markers(
    text: str,
    chunks: list[RetrievedChunk] | None,
) -> str:
    """Replace leftover 'CITATION id=N (…)' markers with human-readable names.
    Wraps the result in double quotes for Mermaid compatibility.

    Falls back to a no-op if *chunks* is None or the id is out of range.
    """
    if chunks is None:
        return text

    def _replace(m: re.Match) -> str:
        idx = int(m.group(1)) - 1  # CITATION ids are 1-based
        if 0 <= idx < len(chunks):
            c = chunks[idx]
            label = f"{c.symbol} ({c.path} lines {c.start_line}-{c.end_line})"
            return f'"{label}"'
        return m.group(0)  # leave as-is if out of range

    return _CITATION_MARKER.sub(_replace, text)


def _clean_preamble(preamble: str) -> str:
    """Strip explanatory text, keeping only title and brief description.

    Heuristic: If preamble has multiple paragraphs or looks like analysis,
    keep only the first line (likely a title). If it looks like a reasonable
    title+description (1-3 short lines), keep as-is.
    """
    if not preamble:
        return ""

    lines = preamble.split("\n")
    non_empty = [line.strip() for line in lines if line.strip()]

    if not non_empty:
        return ""

    if len(non_empty) <= 3 and all(len(line) < 200 for line in non_empty):
        return preamble

    return non_empty[0] if non_empty else ""


def format_model_diagram_response(
    raw: str,
    chunks: list[RetrievedChunk] | None = None,
) -> tuple[str, bool, bool]:
    """
    Turn model output into Markdown for stdout/file.
    Returns (markdown, had_fenced_mermaid, shape_ok).

    If *chunks* is provided, any leftover ``CITATION id=N`` markers in
    the legend are resolved to the actual symbol names.
    """
    raw = resolve_citation_markers(raw, chunks)

    body, preamble = extract_mermaid_fence(raw)
    if body is None:
        return raw.strip() + "\n", False, False

    clean_preamble = _clean_preamble(preamble)
    shape_ok = validate_mermaid_light(body)
    return normalize_diagram_markdown(clean_preamble, body), True, shape_ok
