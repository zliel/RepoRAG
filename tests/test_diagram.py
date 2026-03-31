from __future__ import annotations

from reporag.llm.diagram import (
    extract_mermaid_fence,
    format_model_diagram_response,
    normalize_diagram_markdown,
    validate_mermaid_light,
)
from reporag.llm.prompts import DIAGRAM_SYSTEM, build_rag_user_content


def test_extract_mermaid_fence_ok() -> None:
    raw = """Legend: A = id 1

```mermaid
flowchart LR
  A --> B
```
"""
    body, pre = extract_mermaid_fence(raw)
    assert body is not None
    assert "flowchart LR" in body
    assert "Legend" in pre


def test_extract_mermaid_fence_missing() -> None:
    body, pre = extract_mermaid_fence("just prose")
    assert body is None
    assert pre == "just prose"


def test_validate_mermaid_light() -> None:
    assert validate_mermaid_light("flowchart TD\n  A-->B")
    assert validate_mermaid_light("sequenceDiagram\n  A->>B: hi")
    assert not validate_mermaid_light("")
    assert not validate_mermaid_light("not mermaid")


def test_normalize_diagram_markdown() -> None:
    md = normalize_diagram_markdown("Legend here", "flowchart TD\n  A-->B")
    assert md.startswith("Legend here")
    assert "```mermaid" in md
    assert "flowchart TD" in md
    assert md.strip().endswith("```")


def test_diagram_system_mentions_mermaid() -> None:
    assert "mermaid" in DIAGRAM_SYSTEM.lower()
    assert "CITATION" in DIAGRAM_SYSTEM


def test_build_rag_user_content() -> None:
    s = build_rag_user_content("q?", "ctx")
    assert "q?" in s
    assert "ctx" in s


def test_format_model_diagram_response_fenced() -> None:
    raw = """L1

```mermaid
flowchart LR
  X --> Y
```
"""
    md, had_fence, shape_ok = format_model_diagram_response(raw)
    assert had_fence
    assert shape_ok
    assert "```mermaid" in md
    assert "flowchart LR" in md


def test_format_model_diagram_response_raw_fallback() -> None:
    md, had_fence, shape_ok = format_model_diagram_response("no diagram here")
    assert not had_fence
    assert not shape_ok
    assert md == "no diagram here\n"
