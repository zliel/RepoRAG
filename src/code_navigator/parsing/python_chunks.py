from __future__ import annotations

import logging
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from code_navigator.types import Chunk

logger = logging.getLogger(__name__)

_PY_LANGUAGE = Language(tspython.language())
_PARSER = Parser(_PY_LANGUAGE)


def _symbol_name(node: Node) -> str:
    name_node = node.child_by_field_name("name")
    if name_node is None or name_node.type != "identifier":
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8", errors="replace")
        logger.warning("No identifier for %s node; using <anonymous>", node.type)
        return "<anonymous>"
    return name_node.text.decode("utf-8", errors="replace")


def _iter_def_nodes(node: Node) -> list[Node]:
    found: list[Node] = []

    def visit(n: Node) -> None:
        if n.type in ("function_definition", "class_definition"):
            found.append(n)
        for c in n.children:
            visit(c)

    visit(node)
    return found


def extract_chunks(relative_path: str, source_bytes: bytes) -> list[Chunk]:
    """
    Parse Python source and return one Chunk per function_definition or class_definition
    (including nested). Lines are 1-based inclusive.
    """
    tree = _PARSER.parse(source_bytes)
    root = tree.root_node
    if root.has_error:
        logger.warning("Parse tree has errors for %s; chunks may be incomplete", relative_path)

    chunks: list[Chunk] = []
    for node in _iter_def_nodes(root):
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        chunks.append(
            Chunk(
                path=relative_path,
                symbol_name=_symbol_name(node),
                start_line=start_line,
                end_line=end_line,
                text=text,
            )
        )
    return chunks


def extract_chunks_from_file(path: Path, root: Path) -> list[Chunk] | None:
    """
    Read and parse a file. Returns None if the file cannot be read or parsed critically.
    """
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
        source_bytes = path.read_bytes()
    except OSError as e:
        logger.warning("Cannot read %s: %s", path, e)
        return None
    try:
        return extract_chunks(rel, source_bytes)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", path, e)
        return None
