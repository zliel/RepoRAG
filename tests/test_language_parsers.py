from __future__ import annotations

from pathlib import Path

from reporag.parsing import go_chunks, javascript_chunks, rust_chunks, typescript_chunks

FIXTURES = Path(__file__).parent / "fixtures"


def test_javascript_parser_extracts_functions_and_classes() -> None:
    path = FIXTURES / "sample.js"
    source_bytes = path.read_bytes()
    chunks = javascript_chunks.JavaScriptParser().extract_chunks("sample.js", source_bytes)
    symbols = {c.symbol_name for c in chunks}
    assert "DataService" in symbols
    assert "fetchData" in symbols
    assert "processData" in symbols
    assert "initialize" in symbols
    for ch in chunks:
        assert ch.language == "javascript"


def test_typescript_parser_extracts_functions_classes_interfaces() -> None:
    path = FIXTURES / "sample.ts"
    source_bytes = path.read_bytes()
    chunks = typescript_chunks.TypeScriptParser().extract_chunks("sample.ts", source_bytes)
    symbols = {c.symbol_name for c in chunks}
    assert "UserService" in symbols
    assert "fetchUser" in symbols
    assert "User" in symbols
    for ch in chunks:
        assert ch.language == "typescript"


def test_go_parser_extracts_functions_and_types() -> None:
    path = FIXTURES / "sample.go"
    source_bytes = path.read_bytes()
    chunks = go_chunks.GoParser().extract_chunks("sample.go", source_bytes)
    symbols = {c.symbol_name for c in chunks}
    assert "User" in symbols
    assert "UserService" in symbols
    assert "NewUserService" in symbols
    assert "AddUser" in symbols
    assert "GetUser" in symbols
    assert "main" in symbols
    for ch in chunks:
        assert ch.language == "go"


def test_rust_parser_extracts_functions_structs_impls() -> None:
    path = FIXTURES / "sample.rs"
    source_bytes = path.read_bytes()
    chunks = rust_chunks.RustParser().extract_chunks("sample.rs", source_bytes)
    symbols = {c.symbol_name for c in chunks}
    assert "User" in symbols
    assert "UserService" in symbols
    assert "new" in symbols
    assert "add_user" in symbols
    assert "get_user" in symbols
    assert "for_each" in symbols
    assert "main" in symbols
    for ch in chunks:
        assert ch.language == "rust"
