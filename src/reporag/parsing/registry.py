from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reporag.parsing.base import ParserProtocol

PARSER_REGISTRY: dict[str, ParserProtocol] = {}

_TEXT_PARSER: ParserProtocol | None = None


def register(*extensions: str) -> type[ParserProtocol]:
    """Decorator to register a parser for specific file extensions.

    Usage:
        @register(".py")
        class PythonParser(ParserProtocol):
            ...
    """

    def decorator(cls: type[ParserProtocol]) -> type[ParserProtocol]:
        instance = cls()
        for ext in extensions:
            PARSER_REGISTRY[ext.lower()] = instance
        return cls

    return decorator


def get_parser_for_extension(extension: str) -> ParserProtocol | None:
    """Get the registered parser for a file extension."""
    return PARSER_REGISTRY.get(extension.lower())


def get_fallback_parser() -> ParserProtocol | None:
    """Get the plain text fallback parser."""
    global _TEXT_PARSER
    return _TEXT_PARSER


def set_fallback_parser(parser: ParserProtocol) -> None:
    """Set the plain text fallback parser."""
    global _TEXT_PARSER
    _TEXT_PARSER = parser


def get_all_extensions() -> list[str]:
    """Get list of all registered file extensions."""
    return list(PARSER_REGISTRY.keys())


def get_supported_languages() -> list[str]:
    """Get list of all supported language names."""
    languages = set()
    for parser in PARSER_REGISTRY.values():
        languages.add(parser.language_name)
    return sorted(languages)
