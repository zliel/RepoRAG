from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: int
    path: str
    symbol: str
    start_line: int
    end_line: int
    text: str
    score: float


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def top_k_similar(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    meta: list[dict[str, str | int]],
    k: int,
) -> list[RetrievedChunk]:
    """
    Cosine similarity via normalized dot product. query_vec shape (dim,), embeddings (n, dim).
    """
    if embeddings.size == 0 or not meta:
        return []
    k = min(k, len(meta))
    q = query_vec.astype(np.float32, copy=False).reshape(1, -1)
    qn = _l2_normalize_rows(q)
    en = _l2_normalize_rows(embeddings)
    scores = (en @ qn.T).reshape(-1)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    out: list[RetrievedChunk] = []
    for i in idx:
        m = meta[int(i)]
        out.append(
            RetrievedChunk(
                chunk_id=int(m["id"]),
                path=str(m["path"]),
                symbol=str(m["symbol"]),
                start_line=int(m["start_line"]),
                end_line=int(m["end_line"]),
                text=str(m["text"]),
                score=float(scores[int(i)]),
            )
        )
    return out
