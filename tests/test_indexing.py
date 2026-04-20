from __future__ import annotations

from pathlib import Path

import pytest

from reporag.indexing.store import open_index
from reporag.types import Chunk


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
            language="python",
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
        assert meta[0]["language"] == "python"
    finally:
        idx.close()


def test_file_metadata_upsert_and_get(db_path: Path) -> None:
    idx = open_index(db_path)
    try:
        idx.clear_file_metadata()
        idx.upsert_file_mtime("a.py", 1000.0, 2000.0)
        idx.upsert_file_mtime("b.py", 1500.0, 2500.0)

        mtimes = idx.get_all_file_mtimes()
        assert len(mtimes) == 2
        assert mtimes["a.py"] == (1000.0, 2000.0)
        assert mtimes["b.py"] == (1500.0, 2500.0)

        idx.upsert_file_mtime("a.py", 1100.0, 2100.0)
        mtimes = idx.get_all_file_mtimes()
        assert len(mtimes) == 2
        assert mtimes["a.py"] == (1100.0, 2100.0)
    finally:
        idx.close()


def test_delete_chunks_by_paths(db_path: Path) -> None:
    idx = open_index(db_path)
    try:
        idx.clear()
        ch1 = Chunk(
            path="a.py",
            symbol_name="f",
            start_line=1,
            end_line=3,
            text="def f(): pass",
            language="python",
        )
        ch2 = Chunk(
            path="b.py",
            symbol_name="g",
            start_line=1,
            end_line=3,
            text="def g(): pass",
            language="python",
        )
        ch3 = Chunk(
            path="a.py",
            symbol_name="h",
            start_line=5,
            end_line=7,
            text="def h(): pass",
            language="python",
        )
        vec = [1.0, 0.0, 0.0]
        idx.insert_chunk(ch1, vec)
        idx.insert_chunk(ch2, vec)
        idx.insert_chunk(ch3, vec)
        assert idx.chunk_count() == 3

        idx.delete_chunks_by_paths(["a.py"])
        assert idx.chunk_count() == 1
        mat, meta = idx.load_embeddings_matrix()
        assert meta[0]["path"] == "b.py"

        idx.delete_chunks_by_paths([])
        assert idx.chunk_count() == 1

        idx.delete_chunks_by_paths(["b.py", "nonexistent.py"])
        assert idx.chunk_count() == 0
    finally:
        idx.close()


def test_clear_removes_file_metadata(db_path: Path) -> None:
    idx = open_index(db_path)
    try:
        idx.clear()
        idx.upsert_file_mtime("a.py", 1000.0, 2000.0)
        assert len(idx.get_all_file_mtimes()) == 1

        idx.clear()
        assert len(idx.get_all_file_mtimes()) == 0
    finally:
        idx.close()


def test_file_metadata_persists_across_connections(db_path: Path) -> None:
    idx1 = open_index(db_path)
    try:
        idx1.clear()
        idx1.upsert_file_mtime("a.py", 1000.0, 2000.0)
        idx1.upsert_file_mtime("b.py", 1500.0, 2500.0)
    finally:
        idx1.close()

    idx2 = open_index(db_path)
    try:
        mtimes = idx2.get_all_file_mtimes()
        assert len(mtimes) == 2
        assert "a.py" in mtimes
        assert "b.py" in mtimes
    finally:
        idx2.close()


def test_delete_file_metadata_by_paths(db_path: Path) -> None:
    idx = open_index(db_path)
    try:
        idx.clear()
        idx.upsert_file_mtime("a.py", 1000.0, 2000.0)
        idx.upsert_file_mtime("b.py", 1500.0, 2500.0)
        idx.upsert_file_mtime("c.py", 2000.0, 3000.0)
        assert len(idx.get_all_file_mtimes()) == 3

        idx.delete_file_metadata_by_paths(["a.py", "b.py"])
        mtimes = idx.get_all_file_mtimes()
        assert len(mtimes) == 1
        assert "c.py" in mtimes

        idx.delete_file_metadata_by_paths([])
        assert len(idx.get_all_file_mtimes()) == 1
    finally:
        idx.close()


def test_fuzzy_hash_basic(tmp_path: Path) -> None:
    from reporag.indexing.store import _fuzzy_hash

    # Test basic normalization - collapse multiple whitespaces, strip
    assert _fuzzy_hash("def foo():\n    pass") == _fuzzy_hash("def foo():\n  pass")
    # Should strip and lowercase
    assert _fuzzy_hash("DEF FOO():\n    PASS") == "def foo(): pass"


def test_fuzzy_hash_removes_comments(tmp_path: Path) -> None:
    from reporag.indexing.store import _fuzzy_hash

    # Should remove comments
    text_with_comment = "def foo():\n    x = 1  # comment"
    text_without = "def foo():\n    x = 1"
    assert _fuzzy_hash(text_with_comment) == _fuzzy_hash(text_without)


def test_fuzzy_hash_removes_docstrings(tmp_path: Path) -> None:
    from reporag.indexing.store import _fuzzy_hash

    # Should remove triple-quoted docstrings
    text_with_docstring = 'def foo():\n    """docstring"""\n    pass'
    text_without = "def foo():\n    pass"
    assert _fuzzy_hash(text_with_docstring) == _fuzzy_hash(text_without)


def test_fuzzy_hash_case_insensitive(tmp_path: Path) -> None:
    from reporag.indexing.store import _fuzzy_hash

    assert _fuzzy_hash("DEF FOO():\n    PASS") == _fuzzy_hash("def foo():\n    pass")


def test_find_canonical_chunk_exact_match(db_path: Path) -> None:
    idx = open_index(db_path)
    try:
        idx.clear()
        vec = [1.0, 0.0, 0.0]

        # Insert first chunk (becomes canonical)
        ch1 = Chunk(
            path="a.py",
            symbol_name="foo",
            start_line=1,
            end_line=3,
            text="def foo():\n    pass",
            language="python",
        )
        idx.insert_chunk(ch1, vec)

        # Insert duplicate (different path, same text)
        ch2 = Chunk(
            path="b.py",
            symbol_name="foo",
            start_line=1,
            end_line=3,
            text="def foo():\n    pass",
            language="python",
        )
        idx.insert_chunk(ch2, vec)

        # Should only have 2 rows but the second should reference the first as canonical
        assert idx.chunk_count() == 2
        mat, meta = idx.load_embeddings_matrix()

        # First chunk is canonical (canonical_id = "")
        assert meta[0]["canonical_id"] == ""
        # Second chunk references first as canonical
        assert meta[1]["canonical_id"] == "1"
    finally:
        idx.close()


def test_insert_canonical_and_alias_paths(db_path: Path) -> None:
    idx = open_index(db_path)
    try:
        idx.clear()
        vec = [1.0, 0.0, 0.0]

        # Insert first chunk
        ch1 = Chunk(
            path="a.py",
            symbol_name="foo",
            start_line=1,
            end_line=3,
            text="def foo():\n    pass",
            language="python",
        )
        idx.insert_chunk(ch1, vec)

        # Insert duplicate in different path
        ch2 = Chunk(
            path="b.py",
            symbol_name="foo",
            start_line=1,
            end_line=3,
            text="def foo():\n    pass",
            language="python",
        )
        idx.insert_chunk(ch2, vec)

        # Canonical should have both paths in aliases
        mat, meta = idx.load_embeddings_matrix()
        canonical_aliases = meta[0]["aliases"]
        assert "a.py" in canonical_aliases
        assert "b.py" in canonical_aliases

        # Alias chunk should have empty aliases
        assert meta[1]["aliases"] == []
    finally:
        idx.close()


def test_insert_unique_chunks_no_deduplication(db_path: Path) -> None:
    idx = open_index(db_path)
    try:
        idx.clear()
        vec = [1.0, 0.0, 0.0]

        # Insert different chunks (not duplicates)
        ch1 = Chunk(
            path="a.py",
            symbol_name="foo",
            start_line=1,
            end_line=3,
            text="def foo():\n    pass",
            language="python",
        )
        ch2 = Chunk(
            path="b.py",
            symbol_name="bar",
            start_line=1,
            end_line=3,
            text="def bar():\n    pass",
            language="python",
        )
        idx.insert_chunk(ch1, vec)
        idx.insert_chunk(ch2, vec)

        # Should have 2 rows, both canonicals
        assert idx.chunk_count() == 2
        mat, meta = idx.load_embeddings_matrix()
        assert meta[0]["canonical_id"] == ""
        assert meta[1]["canonical_id"] == ""
    finally:
        idx.close()
