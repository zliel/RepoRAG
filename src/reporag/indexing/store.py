from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np

from reporag.types import Chunk

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
    embedding BLOB NOT NULL
);
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
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM chunks")
        self._conn.execute("DELETE FROM meta")
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
            INSERT INTO chunks(path, symbol, start_line, end_line, text, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.path,
                chunk.symbol_name,
                chunk.start_line,
                chunk.end_line,
                chunk.text,
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
        Each metadata dict: id, path, symbol, start_line, end_line, text.
        """
        rows = self._conn.execute(
            "SELECT id, path, symbol, start_line, end_line, text, embedding FROM chunks ORDER BY id"
        ).fetchall()
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), []

        embeddings: list[np.ndarray] = []
        meta: list[dict[str, str | int]] = []
        dim: int | None = None
        for rid, path, symbol, sl, el, text, emb_blob in rows:
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
                }
            )
        mat = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
        return mat, meta


def open_index(db_path: Path) -> ChunkIndex:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return ChunkIndex(db_path.resolve())
