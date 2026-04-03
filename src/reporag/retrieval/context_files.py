from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reporag.llm.backends import LLMBackend


@dataclass(frozen=True, slots=True)
class ContextSection:
    source_path: str
    heading: str | None
    text: str
    score: float = 0.0


SUPPORTED_EXTENSIONS = {".md", ".txt", ".text", ".markdown"}


def chunk_context_file(path: Path) -> list[ContextSection]:
    """Split a single context file into sections by markdown headings."""
    content = path.read_text(encoding="utf-8")
    sections: list[ContextSection] = []

    heading_pattern = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(content))

    if not matches:
        return [ContextSection(source_path=str(path), heading=None, text=content.strip())]

    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_text = content[start:end].strip()
        if section_text:
            sections.append(
                ContextSection(source_path=str(path), heading=heading, text=section_text)
            )

    return sections


def chunk_context_directory(path: Path) -> list[ContextSection]:
    """Recursively chunk all supported files in a directory."""
    sections: list[ContextSection] = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            sections.extend(chunk_context_file(p))
    return sections


def chunk_context_path(path: Path) -> list[ContextSection]:
    """Chunk a file or directory into context sections."""
    if path.is_file():
        return chunk_context_file(path)
    return chunk_context_directory(path)


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def retrieve_context_sections(
    client: LLMBackend,
    query: str,
    sections: list[ContextSection],
    embed_model: str,
    k: int,
) -> list[ContextSection]:
    """Embed query and sections, return top-k sections by cosine similarity."""
    if not sections:
        return []

    texts = [s.text for s in sections]
    query_vecs = client.embed([query], embed_model)
    section_vecs = client.embed(texts, embed_model)

    q = np.array(query_vecs[0], dtype=np.float32).reshape(1, -1)
    qn = _l2_normalize_rows(q)

    emb_matrix = np.array(section_vecs, dtype=np.float32)
    en = _l2_normalize_rows(emb_matrix)

    scores = (en @ qn.T).reshape(-1)
    k = min(k, len(scores))
    top_idx = np.argpartition(-scores, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    results: list[ContextSection] = []
    for i in top_idx:
        s = sections[i]
        results.append(
            ContextSection(
                source_path=s.source_path,
                heading=s.heading,
                text=s.text,
                score=float(scores[i]),
            )
        )
    return results
