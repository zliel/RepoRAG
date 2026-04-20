from __future__ import annotations

import json
import logging
import re
import sqlite3
import struct
from pathlib import Path
from typing import Any

import numpy as np

from reporag.types import Chunk

logger = logging.getLogger(__name__)

# Regex patterns for fuzzy text normalization
_WHITESPACE_PATTERN = re.compile(r"\s+")
_SINGLE_LINE_COMMENT_PATTERN = re.compile(r"#.*$", re.MULTILINE)
_DOCSTRING_QUOTE_PATTERN = re.compile(r'(""".*?"""|\'\'\'.*?\'\'\')', re.DOTALL)
_DOCSTRING_HASH_PATTERN = re.compile(r'""".*?"""|\'\'\'.*?\'\'\'', re.DOTALL)


def _fuzzy_hash(text: str) -> str:
    """
    Normalize text for fuzzy deduplication.

    Removes:
    - All whitespace sequences (collapses to single spaces)
    - Single-line Python comments (# ...)
    - Triple-quoted docstrings (double and single quotes)
    - Empty/parens-only bodies like: pass, ..., )

    Returns lowercase stripped string for comparison.
    """
    if not text:
        return ""

    # Step 1: Remove docstring quote characters first
    normalized = _DOCSTRING_QUOTE_PATTERN.sub("", text)

    # Step 2: Remove single-line comments
    normalized = _SINGLE_LINE_COMMENT_PATTERN.sub("", normalized)

    # Step 3: Collapse all whitespace to single space
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized)

    # Step 4: Strip and lowercase
    normalized = normalized.strip().lower()

    return normalized


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    symbol TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'python',
    embedding BLOB NOT NULL,
    canonical_id TEXT NOT NULL DEFAULT '',
    aliases TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS file_metadata (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    indexed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS idx_chunks_canonical_id ON chunks(canonical_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(path, symbol, text);
"""


def _float32_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_float32(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.frombuffer(blob, dtype=np.float32, count=n).copy()


# RRF constant for hybrid search
RRF_K = 60


def _rrf_score(rank: int) -> float:
    """Compute Reciprocal Rank Fusion score for a given rank position."""
    return 1.0 / (rank + RRF_K)


class ChunkIndex:
    """SQLite-backed chunk + embedding storage."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._migrate_language_column()
        self._migrate_canonical_id_column()
        self._migrate_aliases_column()
        self._conn.commit()

    def _migrate_language_column(self) -> None:
        columns = self._conn.execute("PRAGMA table_info(chunks)").fetchall()
        column_names = {col[1] for col in columns}
        if "language" not in column_names:
            self._conn.execute(
                "ALTER TABLE chunks ADD COLUMN language TEXT NOT NULL DEFAULT 'python'"
            )
            logger = logging.getLogger(__name__)
            logger.info("Migrated chunks table: added 'language' column")

    def _migrate_canonical_id_column(self) -> None:
        columns = self._conn.execute("PRAGMA table_info(chunks)").fetchall()
        column_names = {col[1] for col in columns}
        if "canonical_id" not in column_names:
            self._conn.execute(
                "ALTER TABLE chunks ADD COLUMN canonical_id TEXT NOT NULL DEFAULT ''"
            )
            logger = logging.getLogger(__name__)
            logger.info("Migrated chunks table: added 'canonical_id' column")

    def _migrate_aliases_column(self) -> None:
        columns = self._conn.execute("PRAGMA table_info(chunks)").fetchall()
        column_names = {col[1] for col in columns}
        if "aliases" not in column_names:
            self._conn.execute("ALTER TABLE chunks ADD COLUMN aliases TEXT NOT NULL DEFAULT '[]'")
            logger = logging.getLogger(__name__)
            logger.info("Migrated chunks table: added 'aliases' column")

    def find_canonical_chunk(self, text: str) -> str | None:
        """
        Find an existing canonical chunk that matches the given text using fuzzy hashing.

        A chunk is considered a match if its fuzzy hash equals the fuzzy hash of the
        provided text.

        Args:
            text: The text content to match against existing chunks.

        Returns:
            The canonical_id of the matching canonical chunk, or None if no match found.
        """
        target_hash = _fuzzy_hash(text)
        if not target_hash:
            return None

        # Get all canonical chunks (where canonical_id is empty - they're the canonical)
        rows = self._conn.execute("SELECT id, text FROM chunks WHERE canonical_id = ''").fetchall()

        for row in rows:
            row_id, row_text = row
            if _fuzzy_hash(row_text) == target_hash:
                return str(row_id)

        return None

    def close(self) -> None:
        self._conn.close()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM chunks_fts")
        self._conn.execute("DELETE FROM chunks")
        self._conn.execute("DELETE FROM meta")
        self._conn.execute("DELETE FROM file_metadata")
        self._conn.commit()

    def upsert_file_mtime(self, path: str, mtime: float, indexed_at: float) -> None:
        sql = """
            INSERT INTO file_metadata(path, mtime, indexed_at) VALUES(?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET mtime = excluded.mtime, indexed_at = excluded.indexed_at
        """
        self._conn.execute(sql, (path, mtime, indexed_at))
        self._conn.commit()

    def get_all_file_mtimes(self) -> dict[str, tuple[float, float]]:
        rows = self._conn.execute("SELECT path, mtime, indexed_at FROM file_metadata").fetchall()
        return {row[0]: (row[1], row[2]) for row in rows}

    def delete_chunks_by_paths(self, paths: list[str]) -> None:
        if not paths:
            return
        # First delete matching FTS5 entries by path
        placeholders = ",".join("?" * len(paths))
        self._conn.execute(f"DELETE FROM chunks_fts WHERE path IN ({placeholders})", paths)
        self._conn.execute(f"DELETE FROM chunks WHERE path IN ({placeholders})", paths)

    def delete_file_metadata_by_paths(self, paths: list[str]) -> None:
        if not paths:
            return
        placeholders = ",".join("?" * len(paths))
        self._conn.execute(f"DELETE FROM file_metadata WHERE path IN ({placeholders})", paths)
        self._conn.commit()

    def clear_file_metadata(self) -> None:
        self._conn.execute("DELETE FROM file_metadata")
        self._conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        sql = (
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )
        self._conn.execute(sql, (key, value))
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def insert_chunk(
        self,
        chunk: Chunk,
        embedding: list[float],
    ) -> int:
        """
        Insert a chunk into the index with fuzzy deduplication.

        If a matching canonical chunk exists (fuzzy hash match), inserts this
        chunk as an alias referencing the canonical. Otherwise, inserts as a new
        canonical chunk.
        """
        # Check for existing canonical using fuzzy matching
        canonical_id = self.find_canonical_chunk(chunk.text)

        blob = _float32_blob(embedding)

        if canonical_id:
            # Found duplicate - insert as alias referencing canonical
            # Add current path to canonical's aliases list
            self._add_alias_to_canonical(canonical_id, chunk.path)

            cur = self._conn.execute(
                """
                INSERT INTO chunks(path, symbol, start_line, end_line, text, language, embedding, canonical_id, aliases)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.path,
                    chunk.symbol_name,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.text,
                    chunk.language,
                    blob,
                    canonical_id,
                    json.dumps([]),  # Aliases empty for alias chunks
                ),
            )
        else:
            # No duplicate found - insert as new canonical
            cur = self._conn.execute(
                """
                INSERT INTO chunks(path, symbol, start_line, end_line, text, language, embedding, canonical_id, aliases)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.path,
                    chunk.symbol_name,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.text,
                    chunk.language,
                    blob,
                    "",
                    json.dumps([chunk.path]),  # Canonical's aliases include its own path
                ),
            )

        chunk_id = int(cur.lastrowid)
        # Also insert into FTS5 for keyword search
        self._conn.execute(
            "INSERT INTO chunks_fts(path, symbol, text) VALUES (?, ?, ?)",
            (chunk.path, chunk.symbol_name, chunk.text),
        )
        self._conn.commit()
        return chunk_id

    def _add_alias_to_canonical(self, canonical_id: str, new_alias_path: str) -> None:
        """
        Add a new alias path to a canonical chunk's aliases list.

        Args:
            canonical_id: The database id of the canonical chunk.
            new_alias_path: The file path to add as an alias.
        """
        # Get current aliases
        row = self._conn.execute(
            "SELECT aliases FROM chunks WHERE id = ?",
            (canonical_id,),
        ).fetchone()

        if not row:
            return

        aliases: list[str] = json.loads(row[0]) if row[0] else []

        # Add new alias if not already present
        if new_alias_path not in aliases:
            aliases.append(new_alias_path)
            self._conn.execute(
                "UPDATE chunks SET aliases = ? WHERE id = ?",
                (json.dumps(aliases), canonical_id),
            )

    def chunk_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def load_embeddings_matrix(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """
        Load all rows: returns (matrix float32 [n, dim], metadata list aligned with rows).
        Each metadata dict: id, path, symbol, start_line, end_line, text, language, canonical_id, aliases.
        """
        rows = self._conn.execute(
            "SELECT id, path, symbol, start_line, end_line, text, language, embedding, canonical_id, aliases "
            "FROM chunks ORDER BY id"
        ).fetchall()
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), []

        embeddings: list[np.ndarray] = []
        meta: list[dict[str, Any]] = []
        dim: int | None = None
        for row in rows:
            rid, path, symbol, sl, el, text, language, emb_blob, canonical_id, aliases_json = row
            v = _blob_to_float32(emb_blob)
            if dim is None:
                dim = int(v.shape[0])
            elif int(v.shape[0]) != dim:
                raise ValueError(f"Inconsistent embedding dim: expected {dim}, got {v.shape[0]}")
            embeddings.append(v)
            meta.append(
                {
                    "id": int(rid),
                    "path": str(path),
                    "symbol": str(symbol),
                    "start_line": int(sl),
                    "end_line": int(el),
                    "text": str(text),
                    "language": str(language),
                    "canonical_id": str(canonical_id),
                    "aliases": json.loads(aliases_json) if aliases_json else [],
                }
            )
        mat = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
        return mat, meta

    def search_fts(self, query: str, k: int) -> list[dict[str, str | int]]:
        """
        Search using FTS5 full-text search. Returns up to k results with chunk_id and rank.

        Uses BM25-style ranking from FTS5. The query is passed as-is to FTS5 MATCH.
        """
        if not query:
            return []

        fts_query = query.strip()

        try:
            # Use FTS5 MATCH - SQLite handles ranking internally
            # Join with chunks to get the actual chunk ID, use DISTINCT to avoid duplicates
            rows = self._conn.execute(
                """
                SELECT DISTINCT c.id, c.path, c.symbol, c.start_line, c.end_line, c.text, c.language
                FROM chunks_fts fts
                JOIN chunks c ON c.path = fts.path AND c.symbol = fts.symbol
                WHERE chunks_fts MATCH ?
                LIMIT ?
                """,
                (fts_query, k),
            ).fetchall()
        except sqlite3.OperationalError as e:
            # If query syntax error, fall back to empty results
            logger.warning("FTS5 query error: %s", e)
            return []

        results: list[dict[str, str | int]] = []
        for row in rows:
            results.append(
                {
                    "id": int(row[0]),
                    "path": str(row[1]),
                    "symbol": str(row[2]),
                    "start_line": int(row[3]),
                    "end_line": int(row[4]),
                    "text": str(row[5]),
                    "language": str(row[6]),
                    "rank": 0.0,  # Rank will be determined by position in list for RRF
                }
            )
        return results


def open_index(db_path: Path) -> ChunkIndex:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return ChunkIndex(db_path.resolve())
