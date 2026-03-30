from __future__ import annotations

from pathlib import Path

import pytest

from code_navigator.ingestion.walker import read_py_file, walk_py_files


def test_walk_py_files_sorted(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y=2\n", encoding="utf-8")
    paths = walk_py_files(tmp_path)
    rels = [p.relative_to(tmp_path).as_posix() for p in paths]
    assert rels == ["a.py", "sub/b.py"]


def test_walk_py_files_not_a_dir(tmp_path: Path) -> None:
    f = tmp_path / "nope.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        walk_py_files(f)


def test_read_py_file(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text("# hi\n", encoding="utf-8")
    rel, text = read_py_file(p, tmp_path)
    assert rel == "m.py"
    assert text == "# hi\n"
