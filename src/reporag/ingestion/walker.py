from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def walk_py_files(root: Path) -> list[Path]:
    """
    Recursively list all .py files under root, sorted by relative path.
    Skips symlinks to avoid cycles.
    """
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    paths: list[Path] = []
    for p in root.rglob("*.py"):
        try:
            if p.is_file() and not p.is_symlink():
                paths.append(p.resolve())
        except OSError as e:
            logger.warning("Skipping path %s: %s", p, e)

    paths.sort(key=lambda x: x.relative_to(root).as_posix())
    return paths


def read_py_file(path: Path, root: Path) -> tuple[str, str]:
    """
    Read UTF-8 text from path. Returns (relative_posix_path, text).
    """
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    return rel, text
