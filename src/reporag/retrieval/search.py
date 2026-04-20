from __future__ import annotations

from collections import defaultdict
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
    language: str
    score: float


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


# RRF constant - standard value from literature
RRF_K = 60


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    """Compute Reciprocal Rank Fusion score for a given rank position."""
    return 1.0 / (rank + k)


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
                language=str(m.get("language", "python")),
                score=float(scores[int(i)]),
            )
        )
    return out


def hybrid_search(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    meta: list[dict[str, str | int]],
    fts_results: list[dict[str, str | int]],
    k: int,
) -> list[RetrievedChunk]:
    """
    Combine vector search and FTS5 search results using Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = sum(1 / (rank(d) + k)) for each retrieval list d containing the document
    where k = 60 (standard constant).

    Args:
        query_vec: Query embedding vector
        embeddings: All chunk embeddings matrix
        meta: Metadata list aligned with embeddings
        fts_results: List of FTS5 search results (dict with id, path, symbol, etc.)
        k: Number of results to return

    Returns:
        Combined and re-ranked list of RetrievedChunk
    """
    if embeddings.size == 0 or not meta:
        return []

    # Get vector search results (ranked by similarity)
    vector_hits = top_k_similar(query_vec, embeddings, meta, k * 2)  # Get extra for RRF

    # Build rank maps
    # Vector ranks: chunk_id -> position (0-indexed)
    vector_ranks: dict[int, int] = {}
    for rank, hit in enumerate(vector_hits):
        vector_ranks[hit.chunk_id] = rank

    # FTS5 ranks: chunk_id -> position (0-indexed)
    fts_ranks: dict[int, int] = {}
    for rank, result in enumerate(fts_results):
        chunk_id = int(result["id"])
        fts_ranks[chunk_id] = rank

    # Collect all chunk IDs from both sources
    all_chunk_ids = set(vector_ranks.keys()) | set(fts_ranks.keys())

    if not all_chunk_ids:
        return []

    # Compute RRF scores for all chunks
    rrf_scores: dict[int, float] = {}
    for chunk_id in all_chunk_ids:
        score = 0.0
        if chunk_id in vector_ranks:
            score += _rrf_score(vector_ranks[chunk_id])
        if chunk_id in fts_ranks:
            score += _rrf_score(fts_ranks[chunk_id])
        rrf_scores[chunk_id] = score

    # Sort by RRF score and take top k
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    top_ids = sorted_ids[:k]

    # Build result chunks - need to look up metadata from vector meta list
    # Create a lookup by chunk_id
    meta_by_id: dict[int, dict[str, str | int]] = {}
    for m in meta:
        meta_by_id[int(m["id"])] = m

    out: list[RetrievedChunk] = []
    for chunk_id in top_ids:
        m = meta_by_id.get(chunk_id)
        if m:
            out.append(
                RetrievedChunk(
                    chunk_id=int(m["id"]),
                    path=str(m["path"]),
                    symbol=str(m["symbol"]),
                    start_line=int(m["start_line"]),
                    end_line=int(m["end_line"]),
                    text=str(m["text"]),
                    language=str(m.get("language", "python")),
                    score=rrf_scores[chunk_id],
                )
            )
    return out
