from __future__ import annotations

import numpy as np
import pytest

from reporag.retrieval.search import _rrf_score, hybrid_search, top_k_similar


def test_top_k_similar_ordering() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    meta = [
        {"id": 1, "path": "a.py", "symbol": "a", "start_line": 1, "end_line": 1, "text": "a"},
        {"id": 2, "path": "b.py", "symbol": "b", "start_line": 1, "end_line": 1, "text": "b"},
        {"id": 3, "path": "c.py", "symbol": "c", "start_line": 1, "end_line": 1, "text": "c"},
    ]
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    hits = top_k_similar(q, embeddings, meta, k=2)
    assert len(hits) == 2
    assert hits[0].path in ("a.py", "c.py")
    assert hits[0].score >= hits[1].score


def test_top_k_empty() -> None:
    q = np.array([1.0], dtype=np.float32)
    hits = top_k_similar(q, np.zeros((0, 1), dtype=np.float32), [], k=5)
    assert hits == []


def test_rrf_score() -> None:
    """Test RRF score decreases with rank."""
    assert _rrf_score(0) > _rrf_score(1)
    assert _rrf_score(1) > _rrf_score(2)
    # With k=60, typical values
    # score(0) = 1/60 = 0.01666...
    assert _rrf_score(0) == pytest.approx(1.0 / 60)
    assert _rrf_score(1) == pytest.approx(1.0 / 61)


def test_hybrid_search_combines_results() -> None:
    """Test that hybrid search combines vector and FTS5 results."""
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # ID 1: close to query
            [0.0, 1.0, 0.0],  # ID 2
            [0.5, 0.5, 0.0],  # ID 3
        ],
        dtype=np.float32,
    )
    meta = [
        {
            "id": 1,
            "path": "a.py",
            "symbol": "a",
            "start_line": 1,
            "end_line": 1,
            "text": "test func a",
            "language": "python",
        },
        {
            "id": 2,
            "path": "b.py",
            "symbol": "b",
            "start_line": 1,
            "end_line": 1,
            "text": "test func b",
            "language": "python",
        },
        {
            "id": 3,
            "path": "c.py",
            "symbol": "c",
            "start_line": 1,
            "end_line": 1,
            "text": "test func c",
            "language": "python",
        },
    ]
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # FTS5 returns only ID 2 (matching "func b")
    fts_results = [
        {
            "id": 2,
            "path": "b.py",
            "symbol": "b",
            "start_line": 1,
            "end_line": 1,
            "text": "test func b",
            "language": "python",
        },
    ]

    # Hybrid should combine vector results (1, 3) with FTS5 result (2)
    hits = hybrid_search(q, embeddings, meta, fts_results, k=3)

    assert len(hits) == 3
    # All three should be present
    paths = [h.path for h in hits]
    assert "a.py" in paths
    assert "b.py" in paths
    assert "c.py" in paths
    # b.py gets a boost from appearing in both lists
    b_hit = next(h for h in hits if h.path == "b.py")
    assert b_hit.score > 0  # Has RRF score from both list appearances


def test_hybrid_search_empty_fts() -> None:
    """Test hybrid search when FTS5 returns no results."""
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    meta = [
        {
            "id": 1,
            "path": "a.py",
            "symbol": "a",
            "start_line": 1,
            "end_line": 1,
            "text": "test",
            "language": "python",
        },
        {
            "id": 2,
            "path": "b.py",
            "symbol": "b",
            "start_line": 1,
            "end_line": 1,
            "text": "test",
            "language": "python",
        },
    ]
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Empty FTS5 results falls back to vector-only
    hits = hybrid_search(q, embeddings, meta, [], k=2)

    assert len(hits) == 2
    assert hits[0].path in ("a.py", "b.py")
