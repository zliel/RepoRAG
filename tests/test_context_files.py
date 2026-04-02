from __future__ import annotations

from pathlib import Path

import pytest

from reporag.retrieval.context_files import (
    ContextSection,
    chunk_context_directory,
    chunk_context_file,
    chunk_context_path,
)


class TestChunkContextFile:
    def test_chunk_by_headings(self, tmp_path: Path) -> None:
        content = """# Introduction

This is the intro.

## Getting Started

Start here.

## Advanced Topics

Advanced content here.
"""
        file_path = tmp_path / "test.md"
        file_path.write_text(content)

        sections = chunk_context_file(file_path)

        assert len(sections) == 3
        assert sections[0].heading == "Introduction"
        assert "intro" in sections[0].text.lower()
        assert sections[1].heading == "Getting Started"
        assert sections[2].heading == "Advanced Topics"

    def test_single_file_no_headings(self, tmp_path: Path) -> None:
        content = "This is a plain text file without any markdown headings."
        file_path = tmp_path / "test.txt"
        file_path.write_text(content)

        sections = chunk_context_file(file_path)

        assert len(sections) == 1
        assert sections[0].heading is None
        assert sections[0].text == content
        assert sections[0].source_path == str(file_path)

    def test_heading_with_content(self, tmp_path: Path) -> None:
        content = """# Title

Some content here.

## Subtitle

More content.
"""
        file_path = tmp_path / "test.md"
        file_path.write_text(content)

        sections = chunk_context_file(file_path)

        assert len(sections) == 2
        assert sections[0].heading == "Title"
        assert sections[1].heading == "Subtitle"
        assert "# Title" in sections[0].text
        assert "# Subtitle" in sections[1].text

    def test_only_heading(self, tmp_path: Path) -> None:
        content = "# Only Heading"
        file_path = tmp_path / "test.md"
        file_path.write_text(content)

        sections = chunk_context_file(file_path)

        assert len(sections) == 1
        assert sections[0].heading == "Only Heading"

    def test_h6_headings(self, tmp_path: Path) -> None:
        content = """# H1
###### H6
## H2
"""
        file_path = tmp_path / "test.md"
        file_path.write_text(content)

        sections = chunk_context_file(file_path)

        assert len(sections) == 3
        assert sections[0].heading == "H1"
        assert sections[1].heading == "H6"
        assert sections[2].heading == "H2"


class TestChunkContextDirectory:
    def test_recursive_chunking(self, tmp_path: Path) -> None:
        subdir = tmp_path / "docs"
        subdir.mkdir()

        (tmp_path / "readme.md").write_text("# Readme\nReadme content.")
        (subdir / "guide.md").write_text("# Guide\nGuide content.")

        sections = chunk_context_directory(tmp_path)

        assert len(sections) == 2
        paths = [s.source_path for s in sections]
        assert any("readme.md" in p for p in paths)
        assert any("guide.md" in p for p in paths)

    def test_ignores_unsupported_files(self, tmp_path: Path) -> None:
        (tmp_path / "readme.md").write_text("# Readme\nContent.")
        (tmp_path / "script.py").write_text("print('hello')")
        (tmp_path / "data.json").write_text('{"key": "value"}')

        sections = chunk_context_directory(tmp_path)

        assert len(sections) == 1
        assert "readme.md" in sections[0].source_path

    def test_empty_directory(self, tmp_path: Path) -> None:
        sections = chunk_context_directory(tmp_path)
        assert sections == []


class TestChunkContextPath:
    def test_file_path(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.md"
        file_path.write_text("# Title\nContent.")

        sections = chunk_context_path(file_path)

        assert len(sections) == 1
        assert sections[0].source_path == str(file_path)

    def test_directory_path(self, tmp_path: Path) -> None:
        subdir = tmp_path / "docs"
        subdir.mkdir()
        (subdir / "test.md").write_text("# Doc\nContent.")

        sections = chunk_context_path(tmp_path)

        assert len(sections) == 1
        assert "docs" in sections[0].source_path and "test.md" in sections[0].source_path


class TestContextSection:
    def test_context_section_frozen(self) -> None:
        section = ContextSection(
            source_path="/path/to/file.md",
            heading="Test",
            text="Some text",
            score=0.95,
        )
        assert section.source_path == "/path/to/file.md"
        assert section.heading == "Test"
        assert section.text == "Some text"
        assert section.score == 0.95

    def test_context_section_immutable(self) -> None:
        section = ContextSection(
            source_path="/path/to/file.md",
            heading="Test",
            text="Some text",
        )
        with pytest.raises(AttributeError):
            section.score = 0.5
