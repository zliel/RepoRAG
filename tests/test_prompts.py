from __future__ import annotations

from reporag.llm.prompts import build_context_block
from reporag.retrieval.search import RetrievedChunk


def test_build_context_block() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id=1,
            path="pkg/mod.py",
            symbol="Foo",
            start_line=10,
            end_line=12,
            text="class Foo:\n    pass\n",
            score=0.9,
        )
    ]
    block = build_context_block(chunks)
    assert "CITATION id=1" in block
    assert "pkg/mod.py" in block
    assert "lines=10-12" in block
    assert "class Foo" in block
