from __future__ import annotations

import logging

import tree_sitter_typescript as tsts
from tree_sitter import Language, Node, Parser

from reporag.parsing.base import ParserProtocol
from reporag.parsing.registry import register
from reporag.types import Chunk

logger = logging.getLogger(__name__)

_TS_LANGUAGE = Language(tsts.language_typescript())
_TSX_LANGUAGE = Language(tsts.language_tsx())

_FUNCTIONS = {
    "function_declaration",
    "function_expression",
    "method_definition",
    "method_signature",
}
_CLASSES = {
    "class_declaration",
    "class_expression",
}
_TYPES = {
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
}


def _symbol_name(node: Node, source_bytes: bytes) -> str:
    name_node = node.child_by_field_name("name")
    if name_node is not None and name_node.type in ("identifier", "type_identifier"):
        return name_node.text.decode("utf-8", errors="replace")
    for child in node.children:
        if child.type in ("property_identifier", "identifier", "type_identifier"):
            return child.text.decode("utf-8", errors="replace")
    logger.warning("No identifier for %s node; using <anonymous>", node.type)
    return "<anonymous>"


def _iter_def_nodes(node: Node) -> list[Node]:
    found: list[Node] = []

    def visit(n: Node) -> None:
        if n.type in _FUNCTIONS or n.type in _CLASSES or n.type in _TYPES:
            found.append(n)
        for c in n.children:
            visit(c)

    visit(node)
    return found


@register(".ts", ".tsx")
class TypeScriptParser(ParserProtocol):
    """Parser for TypeScript/TSX source files using tree-sitter."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".ts", ".tsx")

    @property
    def language_name(self) -> str:
        return "typescript"

    def extract_chunks(self, relative_path: str, source_bytes: bytes) -> list[Chunk]:
        """Parse TypeScript/TSX source and return one Chunk per function, class, or type definition."""
        is_tsx = relative_path.lower().endswith(".tsx")
        parser = Parser(_TSX_LANGUAGE if is_tsx else _TS_LANGUAGE)
        tree = parser.parse(source_bytes)
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
                    language="typescript",
                )
            )
        return chunks
