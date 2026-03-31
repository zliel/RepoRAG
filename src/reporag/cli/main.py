from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import typer

from reporag.indexing.store import open_index
from reporag.ingestion.walker import read_py_file, walk_py_files
from reporag.llm.diagram import format_model_diagram_response
from reporag.llm.ollama_client import OllamaClient
from reporag.llm.prompts import (
    ANSWER_SYSTEM,
    DIAGRAM_SYSTEM,
    REWRITE_SYSTEM,
    build_context_block,
    build_rag_user_content,
)
from reporag.parsing.python_chunks import extract_chunks
from reporag.retrieval.search import RetrievedChunk, top_k_similar
from reporag.types import Chunk

logger = logging.getLogger(__name__)

app = typer.Typer(help="Local-first semantic Python codebase navigator (Ollama + SQLite).")

DEFAULT_EMBED_MODEL = "nomic-embed-text-v2-moe"
# DEFAULT_CHAT_MODEL = "qwen3-vl:8b-instruct"
DEFAULT_CHAT_MODEL = "deepseek-coder-v2:16b-lite-instruct-q3_K_M"
EMBED_BATCH = 32


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _retrieve_hits(
    client: OllamaClient,
    query: str,
    db: Path,
    k: int,
    embed_model: str,
    chat_model: str,
    no_rewrite: bool,
) -> tuple[list[RetrievedChunk], str]:
    """
    Open index, optional rewrite, embed search string, return top-k chunks and search_query used.
    """
    idx = open_index(db)
    try:
        mat, meta = idx.load_embeddings_matrix()
        stored_model = idx.get_meta("embed_model")
    finally:
        idx.close()
    if mat.size == 0:
        typer.echo("Index is empty. Run `reporag index` first.", err=True)
        raise typer.Exit(1)
    if stored_model and stored_model != embed_model:
        logger.warning("Index built with %s; query uses %s", stored_model, embed_model)

    search_query = query
    if not no_rewrite:
        try:
            search_query = client.chat(
                chat_model,
                [
                    {"role": "system", "content": REWRITE_SYSTEM},
                    {"role": "user", "content": query},
                ],
            ).strip()
            logger.info("Rewritten query: %s", search_query)
        except Exception as e:
            logger.warning("Query rewrite failed, using original: %s", e)
            search_query = query

    vecs = client.embed([search_query], embed_model)
    q = np.array(vecs[0], dtype=np.float32)
    hits = top_k_similar(q, mat, meta, k)
    return hits, search_query


@app.callback()
def _global(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
) -> None:
    _setup_logging(verbose)


@app.command("list")
def cmd_list(
    root: Path = typer.Argument(..., exists=True, file_okay=False, help="Project root."),
) -> None:
    """List all .py files under root."""
    root = root.resolve()
    paths = walk_py_files(root)
    for p in paths:
        typer.echo(p.relative_to(root).as_posix())
    logger.info("Found %d Python files", len(paths))
    typer.echo(f"Total: {len(paths)}", err=True)


@app.command("chunks")
def cmd_chunks(
    root: Path = typer.Argument(..., exists=True, file_okay=False, help="Project root."),
) -> None:
    """Extract function/class chunks and print JSON lines."""
    root = root.resolve()
    total = 0
    for p in walk_py_files(root):
        rel, text = read_py_file(p, root)
        source_bytes = text.encode("utf-8")
        for ch in extract_chunks(rel, source_bytes):
            typer.echo(
                json.dumps(
                    {
                        "path": ch.path,
                        "symbol": ch.symbol_name,
                        "start_line": ch.start_line,
                        "end_line": ch.end_line,
                        "text_len": len(ch.text),
                    },
                    ensure_ascii=False,
                )
            )
            total += 1
    logger.info("Extracted %d chunks", total)
    typer.echo(f"Total chunks: {total}", err=True)


@app.command("index")
def cmd_index(
    root: Path = typer.Argument(..., exists=True, file_okay=False, help="Project root."),
    db: Path = typer.Option(Path("index.sqlite"), "--db", help="SQLite index path."),
    embed_model: str = typer.Option(
        DEFAULT_EMBED_MODEL,
        "--embed-model",
        help="Ollama embedding model.",
    ),
    ollama_base: str | None = typer.Option(
        None,
        "--ollama-base-url",
        help="Ollama base URL (default: OLLAMA_HOST or http://127.0.0.1:11434).",
    ),
) -> None:
    """Build embedding index for all Python chunks."""
    root = root.resolve()
    client = OllamaClient(base_url=ollama_base)
    idx = open_index(db)
    try:
        idx.clear()
        chunks_flat: list[Chunk] = []
        for p in walk_py_files(root):
            rel, text = read_py_file(p, root)
            source_bytes = text.encode("utf-8")
            chunks_flat.extend(extract_chunks(rel, source_bytes))
        if not chunks_flat:
            logger.warning("No chunks found under %s", root)
            idx.set_meta("embed_model", embed_model)
            idx.set_meta("embed_dim", "0")
            return
        logger.info("Embedding %d chunks with model %s", len(chunks_flat), embed_model)
        dim: int | None = None
        for i in range(0, len(chunks_flat), EMBED_BATCH):
            batch = chunks_flat[i : i + EMBED_BATCH]
            texts = [c.text for c in batch]
            vectors = client.embed(texts, embed_model)
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding count mismatch")
            for ch, vec in zip(batch, vectors, strict=True):
                if dim is None:
                    dim = len(vec)
                elif len(vec) != dim:
                    raise ValueError(f"Inconsistent embedding dimension: {len(vec)} vs {dim}")
                idx.insert_chunk(ch, vec)
            done = min(i + EMBED_BATCH, len(chunks_flat))
            logger.info("Indexed %d / %d chunks", done, len(chunks_flat))
        if dim is not None:
            idx.set_meta("embed_dim", str(dim))
        idx.set_meta("embed_model", embed_model)
        logger.info("Done. Rows in DB: %d", idx.chunk_count())
    finally:
        client.close()
        idx.close()


@app.command("search")
def cmd_search(
    query: str = typer.Argument(..., help="Natural language query."),
    db: Path = typer.Option(Path("index.sqlite"), "--db", help="SQLite index path."),
    k: int = typer.Option(8, "-k", "--top-k", help="Number of chunks to retrieve."),
    embed_model: str = typer.Option(
        DEFAULT_EMBED_MODEL,
        "--embed-model",
        help="Ollama embedding model.",
    ),
    ollama_base: str | None = typer.Option(None, "--ollama-base-url"),
) -> None:
    """Retrieve top-k chunks for a query."""
    client = OllamaClient(base_url=ollama_base)
    try:
        idx = open_index(db)
        try:
            mat, meta = idx.load_embeddings_matrix()
            stored_model = idx.get_meta("embed_model")
        finally:
            idx.close()
        if mat.size == 0:
            typer.echo("Index is empty.", err=True)
            raise typer.Exit(1)
        if stored_model and stored_model != embed_model:
            logger.warning("Index built with %s; query uses %s", stored_model, embed_model)
        vecs = client.embed([query], embed_model)
        q = np.array(vecs[0], dtype=np.float32)
        hits = top_k_similar(q, mat, meta, k)
        for h in hits:
            typer.echo(
                json.dumps(
                    {
                        "score": round(h.score, 6),
                        "path": h.path,
                        "symbol": h.symbol,
                        "start_line": h.start_line,
                        "end_line": h.end_line,
                        "preview": h.text[:200] + ("…" if len(h.text) > 200 else ""),
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        client.close()


@app.command("ask")
def cmd_ask(
    query: str = typer.Argument(..., help="Question about the codebase."),
    db: Path = typer.Option(Path("index.sqlite"), "--db", help="SQLite index path."),
    k: int = typer.Option(8, "-k", "--top-k", help="Chunks to pass to the model."),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, "--embed-model"),
    chat_model: str = typer.Option(DEFAULT_CHAT_MODEL, "--chat-model"),
    ollama_base: str | None = typer.Option(None, "--ollama-base-url"),
    no_rewrite: bool = typer.Option(False, "--no-rewrite", help="Skip query rewrite step."),
) -> None:
    """Answer using retrieved context and Ollama chat."""
    client = OllamaClient(base_url=ollama_base)
    try:
        hits, _ = _retrieve_hits(client, query, db, k, embed_model, chat_model, no_rewrite)
        if not hits:
            typer.echo("No chunks retrieved.", err=True)
            raise typer.Exit(1)
        context = build_context_block(hits)
        user_msg = build_rag_user_content(query, context)
        answer = client.chat(
            chat_model,
            [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        typer.echo(answer)
        typer.echo("\n---\nSources:", err=True)
        for h in hits:
            typer.echo(f"  {h.path} lines {h.start_line}-{h.end_line} ({h.symbol})", err=True)
    finally:
        client.close()


@app.command("diagram")
def cmd_diagram(
    query: str = typer.Argument(..., help="What to visualize (flow, dependencies, classes, etc.)."),
    db: Path = typer.Option(Path("index.sqlite"), "--db", help="SQLite index path."),
    k: int = typer.Option(8, "-k", "--top-k", help="Chunks to pass to the model."),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, "--embed-model"),
    chat_model: str = typer.Option(DEFAULT_CHAT_MODEL, "--chat-model"),
    ollama_base: str | None = typer.Option(None, "--ollama-base-url"),
    no_rewrite: bool = typer.Option(False, "--no-rewrite", help="Skip query rewrite step."),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Write Markdown (legend + mermaid) to this file.",
    ),
    png: Path | None = typer.Option(
        None,
        "--png",
        "-p",
        help="Generate PNG image from mermaid diagram.",
    ),
) -> None:
    """Generate a grounded Mermaid diagram from retrieved code context."""
    client = OllamaClient(base_url=ollama_base)
    try:
        hits, _ = _retrieve_hits(client, query, db, k, embed_model, chat_model, no_rewrite)
        if not hits:
            typer.echo("No chunks retrieved.", err=True)
            raise typer.Exit(1)
        context = build_context_block(hits)
        user_msg = build_rag_user_content(query, context)
        raw = client.chat(
            chat_model,
            [
                {"role": "system", "content": DIAGRAM_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        md, had_fence, shape_ok = format_model_diagram_response(raw, chunks=hits)
        if not had_fence:
            logger.warning("No ```mermaid fence in model output; writing raw response")
        elif not shape_ok:
            logger.warning(
                "Mermaid body does not start with a known diagram type; output may not render"
            )
        typer.echo(md, nl=False)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(md, encoding="utf-8")
            logger.info("Wrote %s", out.resolve())

        # Generate PNG if requested
        if png is not None:
            md_path = out if out is not None else png
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(md, encoding="utf-8")

            # Check for mermaid-cli
            import shutil

            mmdc_path = shutil.which("mmdc")
            if mmdc_path is None:
                logger.warning(
                    "mermaid-cli not found; install with `npm install -g @mermaid-js/mermaid-cli` "
                    "or specify with --mermaid-path"
                )
                # Try to find it via env
                mmdc_path = os.environ.get("MMDC_PATH")
            if mmdc_path:
                cmd = [mmdc_path, "-i", str(md_path.resolve()), "-o", str(png.resolve())]
                logger.info("Running: %s", " ".join(cmd))
                subprocess.run(cmd, check=True)
                logger.info("Wrote PNG: %s", png.resolve())
            else:
                raise RuntimeError("Could not locate mmdc")

        typer.echo("\n---\nSources:", err=True)
        for h in hits:
            typer.echo(f"  {h.path} lines {h.start_line}-{h.end_line} ({h.symbol})", err=True)
    finally:
        client.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
