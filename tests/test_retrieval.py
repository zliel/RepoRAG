from __future__ import annotations

import numpy as np

from reporag.retrieval.search import top_k_similar


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
