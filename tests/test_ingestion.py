from __future__ import annotations

from pathlib import Path

import pytest

from reporag.ingestion.walker import parse_file, walk_supported_files


def test_walk_supported_files_sorted(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y=2\n", encoding="utf-8")
    paths = walk_supported_files(tmp_path)
    rels = [p.relative_to(tmp_path).as_posix() for p in paths]
    assert rels == ["a.py", "sub/b.py"]


def test_walk_supported_files_not_a_dir(tmp_path: Path) -> None:
    f = tmp_path / "nope.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        walk_supported_files(f)


def test_parse_file(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text("def f():\n    pass\n", encoding="utf-8")
    chunks = parse_file(p, tmp_path)
    assert len(chunks) >= 1
    assert chunks[0].language == "python"
    assert chunks[0].symbol_name == "f"


def test_parse_file_text_fallback(tmp_path: Path) -> None:
    p = tmp_path / "readme.md"
    p.write_text("# Hello\n\nThis is a markdown file.\n", encoding="utf-8")
    chunks = parse_file(p, tmp_path)
    assert len(chunks) >= 1
    assert chunks[0].language == "plaintext"


def test_walk_supported_files_respects_ignored_dirs(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("ignore me", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.py").write_text("ignore me too", encoding="utf-8")
    paths = walk_supported_files(tmp_path)
    rels = [p.relative_to(tmp_path).as_posix() for p in paths]
    assert "a.py" in rels
    assert ".git/config" not in rels
    assert "node_modules/pkg.py" not in rels
