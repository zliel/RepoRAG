from __future__ import annotations

from pathlib import Path

import pytest

from code_navigator.indexing.store import open_index
from code_navigator.types import Chunk


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "t.sqlite"


def test_chunk_index_roundtrip(db_path: Path) -> None:
    idx = open_index(db_path)
    try:
        idx.clear()
        ch = Chunk(
            path="a.py",
            symbol_name="f",
            start_line=1,
            end_line=3,
            text="def f():\n  pass\n",
        )
        vec = [1.0, 0.0, 0.0]
        idx.insert_chunk(ch, vec)
        idx.set_meta("embed_dim", "3")
        idx.set_meta("embed_model", "test")
        assert idx.chunk_count() == 1
        mat, meta = idx.load_embeddings_matrix()
        assert mat.shape == (1, 3)
        assert meta[0]["path"] == "a.py"
        assert meta[0]["symbol"] == "f"
    finally:
        idx.close()
