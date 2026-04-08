from __future__ import annotations

from pathlib import Path

from reporag.parsing.python_chunks import extract_chunks, extract_chunks_from_file

FIXTURES = Path(__file__).parent / "fixtures"


def test_extract_chunks_sample_ok() -> None:
    path = FIXTURES / "sample_ok.py"
    source_bytes = path.read_bytes()
    chunks = extract_chunks("sample_ok.py", source_bytes)
    names = {(c.symbol_name, c.start_line) for c in chunks}
    assert ("AuthService", 4) in names
    assert ("verify", 7) in names
    assert ("inner", 8) in names
    assert ("login", 14) in names


def test_extract_chunks_includes_language() -> None:
    path = FIXTURES / "sample_ok.py"
    source_bytes = path.read_bytes()
    chunks = extract_chunks("sample_ok.py", source_bytes)
    for ch in chunks:
        assert ch.language == "python"


def test_extract_chunks_from_file(tmp_path: Path) -> None:
    src = FIXTURES / "sample_ok.py"
    dst = tmp_path / "sample_ok.py"
    dst.write_bytes(src.read_bytes())
    chunks = extract_chunks_from_file(dst, tmp_path)
    assert chunks is not None
    assert len(chunks) >= 4
    for ch in chunks:
        assert ch.language == "python"


def test_broken_file_still_returns_chunks_or_none() -> None:
    path = FIXTURES / "broken.py"
    source_bytes = path.read_bytes()
    chunks = extract_chunks("broken.py", source_bytes)
    assert isinstance(chunks, list)
