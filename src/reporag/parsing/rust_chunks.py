from __future__ import annotations

import logging

import tree_sitter_rust as tsrust
from tree_sitter import Language, Node, Parser

from reporag.parsing.base import ParserProtocol
from reporag.parsing.registry import register
from reporag.types import Chunk

logger = logging.getLogger(__name__)

_RUST_LANGUAGE = Language(tsrust.language())
_RUST_PARSER = Parser(_RUST_LANGUAGE)

_FUNCTIONS = {
    "function_item",
    "method_definition",
}
_TYPES = {
    "struct_item",
    "enum_item",
    "trait_item",
    "impl_item",
    "type_alias_item",
}


def _find_identifier(node: Node, source_bytes: bytes) -> str | None:
    """Find the identifier for a declaration node, searching at appropriate depth."""
    if node.type == "impl_item":
        for child in node.children:
            if child.type == "type_identifier":
                return child.text.decode("utf-8", errors="replace")
    if node.type in ("struct_item", "enum_item", "trait_item", "function_item", "type_alias_item"):
        for child in node.children:
            if child.type in ("identifier", "type_identifier"):
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


@register(".rs")
class RustParser(ParserProtocol):
    """Parser for Rust source files using tree-sitter."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".rs",)

    @property
    def language_name(self) -> str:
        return "rust"

    def extract_chunks(self, relative_path: str, source_bytes: bytes) -> list[Chunk]:
        """Parse Rust source: return one Chunk per function, struct, enum, trait, or impl."""
        tree = _RUST_PARSER.parse(source_bytes)
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
                    language="rust",
                )
            )
        return chunks
