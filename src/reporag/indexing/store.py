from __future__ import annotations

import logging
import sqlite3
import struct
from pathlib import Path

import numpy as np

from reporag.types import Chunk

logger = logging.getLogger(__name__)

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
    embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS file_metadata (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    indexed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
"""


def _float32_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_float32(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.frombuffer(blob, dtype=np.float32, count=n).copy()


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
        self._migrate_dedup_columns()
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

    def _migrate_dedup_columns(self) -> None:
        """Add canonical_id and aliases columns for chunk deduplication."""
        columns = self._conn.execute("PRAGMA table_info(chunks)").fetchall()
        column_names = {col[1] for col in columns}
        if "canonical_id" not in column_names:
            self._conn.execute(
                "ALTER TABLE chunks ADD COLUMN canonical_id INTEGER REFERENCES chunks(id)"
            )
            logger.info("Migrated chunks table: added 'canonical_id' column")
        if "aliases" not in column_names:
            self._conn.execute("ALTER TABLE chunks ADD COLUMN aliases TEXT DEFAULT ''")
            logger.info("Migrated chunks table: added 'aliases' column")

    def close(self) -> None:
        self._conn.close()

    def clear(self) -> None:
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
        placeholders = ",".join("?" * len(paths))
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
        blob = _float32_blob(embedding)
        cur = self._conn.execute(
            """
            INSERT INTO chunks(path, symbol, start_line, end_line, text, language, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.path,
                chunk.symbol_name,
                chunk.start_line,
                chunk.end_line,
                chunk.text,
                chunk.language,
                blob,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def chunk_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def load_embeddings_matrix(self) -> tuple[np.ndarray, list[dict[str, str | int]]]:
        """
        Load all rows: returns (matrix float32 [n, dim], metadata list aligned with rows).
        Each metadata dict: id, path, symbol, start_line, end_line, text, language, canonical_id, aliases.
        """
        rows = self._conn.execute(
            "SELECT id, path, symbol, start_line, end_line, text, language, canonical_id, aliases, embedding "
            "FROM chunks ORDER BY id"
        ).fetchall()
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), []

        embeddings: list[np.ndarray] = []
        meta: list[dict[str, str | int]] = []
        dim: int | None = None
        for rid, path, symbol, sl, el, text, language, canonical_id, aliases, emb_blob in rows:
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
                    "canonical_id": int(canonical_id) if canonical_id is not None else None,
                    "aliases": str(aliases) if aliases else "",
                }
            )
        mat = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
        return mat, meta


def open_index(db_path: Path) -> ChunkIndex:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return ChunkIndex(db_path.resolve())
