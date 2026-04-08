from __future__ import annotations

import logging

import tree_sitter_go as tsgo
from tree_sitter import Language, Node, Parser

from reporag.parsing.base import ParserProtocol
from reporag.parsing.registry import register
from reporag.types import Chunk

logger = logging.getLogger(__name__)

_GO_LANGUAGE = Language(tsgo.language())
_GO_PARSER = Parser(_GO_LANGUAGE)

_FUNCTIONS = {
    "function_declaration",
    "method_declaration",
}
_TYPES = {
    "type_declaration",
}


def _find_identifier(node: Node, source_bytes: bytes) -> str | None:
    """Find the identifier for a declaration node, searching at appropriate depth."""
    if node.type == "type_declaration":
        for child in node.children:
            if child.type == "type_spec":
                for grandchild in child.children:
                    if grandchild.type == "type_identifier":
                        return grandchild.text.decode("utf-8", errors="replace")
    if node.type == "function_declaration":
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8", errors="replace")
    if node.type == "method_declaration":
        for child in node.children:
            if child.type == "field_identifier":
                return child.text.decode("utf-8", errors="replace")
    return None


def _symbol_name(node: Node, source_bytes: bytes) -> str:
    result = _find_identifier(node, source_bytes)
    if result:
        return result
    logger.warning("No identifier for %s node; using <anonymous>", node.type)
    return "<anonymous>"


def _iter_def_nodes(node: Node) -> list[Node]:
    found: list[Node] = []

    def visit(n: Node) -> None:
        if n.type in _FUNCTIONS or n.type in _TYPES:
            found.append(n)
        for c in n.children:
            visit(c)

    visit(node)
    return found


@register(".go")
class GoParser(ParserProtocol):
    """Parser for Go source files using tree-sitter."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".go",)

    @property
    def language_name(self) -> str:
        return "go"

    def extract_chunks(self, relative_path: str, source_bytes: bytes) -> list[Chunk]:
        """Parse Go source and return one Chunk per function or type declaration."""
        tree = _GO_PARSER.parse(source_bytes)
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
                    language="go",
                )
            )
        return chunks
