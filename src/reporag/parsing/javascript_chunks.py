from __future__ import annotations

import logging

import tree_sitter_javascript as tsjs
from tree_sitter import Language, Node, Parser

from reporag.parsing.base import ParserProtocol
from reporag.parsing.registry import register
from reporag.types import Chunk

logger = logging.getLogger(__name__)

_JS_LANGUAGE = Language(tsjs.language())
_JS_PARSER = Parser(_JS_LANGUAGE)

_FUNCTIONS = {
    "function_declaration",
    "function_expression",
    "method_definition",
}
_CLASSES = {
    "class_declaration",
    "class_expression",
}


def _symbol_name(node: Node, source_bytes: bytes) -> str:
    name_node = node.child_by_field_name("name")
    if name_node is not None and name_node.type == "identifier":
        return name_node.text.decode("utf-8", errors="replace")
    if node.type == "function_declaration":
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8", errors="replace")
    if node.type == "method_definition":
        for child in node.children:
            if child.type in ("property_identifier", "identifier"):
                return child.text.decode("utf-8", errors="replace")
    logger.warning("No identifier for %s node; using <anonymous>", node.type)
    return "<anonymous>"


def _iter_def_nodes(node: Node) -> list[Node]:
    found: list[Node] = []

    def visit(n: Node) -> None:
        if n.type in _FUNCTIONS or n.type in _CLASSES:
            found.append(n)
        for c in n.children:
            visit(c)

    visit(node)
    return found


@register(".js", ".jsx")
class JavaScriptParser(ParserProtocol):
    """Parser for JavaScript/JSX source files using tree-sitter."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".js", ".jsx", ".mjs", ".cjs")

    @property
    def language_name(self) -> str:
        return "javascript"

    def extract_chunks(self, relative_path: str, source_bytes: bytes) -> list[Chunk]:
        """Parse JavaScript source and return one Chunk per function or class definition."""
        tree = _JS_PARSER.parse(source_bytes)
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
                    symbol_name=_symbol_name(node, source_bytes),
                    start_line=start_line,
                    end_line=end_line,
                    text=text,
                    language="javascript",
                )
            )
        return chunks
