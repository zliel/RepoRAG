from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
import typer

from reporag.config import Config
from reporag.indexing.store import open_index
from reporag.ingestion.walker import read_py_file, walk_py_files
from reporag.llm.backends import BackendType, LLMBackend, create_backend
from reporag.llm.diagram import format_model_diagram_response
from reporag.llm.prompts import (
    ANSWER_SYSTEM,
    DIAGRAM_SYSTEM,
    REWRITE_SYSTEM,
    build_context_block,
    build_rag_user_content,
)
from reporag.parsing.python_chunks import extract_chunks
from reporag.retrieval.context_files import (
    ContextSection,
    chunk_context_path,
    retrieve_context_sections,
)
from reporag.retrieval.search import RetrievedChunk, top_k_similar
from reporag.types import Chunk

logger = logging.getLogger(__name__)

app = typer.Typer(help="Local-first semantic Python codebase navigator (Ollama + SQLite).")


@lru_cache
def get_config() -> Config:
    return Config.from_file()


def get_backend(
    cfg: Config | None = None, backend_override: BackendType | None = None
) -> LLMBackend:
    if cfg is None:
        cfg = get_config()
    backend_type = backend_override or cfg.backend
    return create_backend(backend_type, base_url=cfg.base_url, api_key=cfg.api_key)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def read_context_path(path: Path) -> str:
    """Read context from a file or directory. Returns combined text from all files."""
    if path.is_file():
        return path.read_text(encoding="utf-8")

    supported = {".md", ".txt", ".text", ".markdown"}
    files: list[str] = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in supported:
            rel = p.relative_to(path)
            content = p.read_text(encoding="utf-8")
            files.append(f"# {rel}\n{content}")

    if not files:
        logger.warning("No supported files (.md, .txt) found in %s", path)
        return ""

    return "\n\n---\n\n".join(files)


def _retrieve_hits(
    client: LLMBackend,
    query: str,
    db: Path,
    k: int,
    embed_model: str,
    chat_model: str,
    no_rewrite: bool,
    temperature: float | None = None,
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
                temperature=temperature,
            ).strip()
            logger.info("Rewritten query: %s", search_query)
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("Query rewrite failed, using original: %s", e)
            search_query = query

    vecs = client.embed([search_query], embed_model)
    q = np.array(vecs[0], dtype=np.float32)
    hits = top_k_similar(q, mat, meta, k)
    return hits, search_query


def stream_output(chunks: Iterator[str], silent: bool = False) -> str:
    """Print streaming chunks to stdout (unless silent), return full response."""
    if not silent:
        typer.echo("Generating response...", err=True)
    full: list[str] = []
    for chunk in chunks:
        if not silent:
            print(chunk, end="", flush=True)
        full.append(chunk)
    if not silent:
        print()
    return "".join(full)


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
    db: Path | None = typer.Option(
        None, "--db", help="SQLite index path (default: from config or index.sqlite)."
    ),
    embed_model: str | None = typer.Option(None, "--embed-model", help="Embedding model."),
    backend: BackendType | None = typer.Option(
        None, "--backend", help="LLM backend (ollama, vllm, llamacpp, lmstudio)."
    ),
) -> None:
    """Build embedding index for all Python chunks."""
    cfg = get_config()
    root = root.resolve()
    db = db or Path(cfg.db)
    embed_model = embed_model or cfg.embed_model
    client = get_backend(cfg, backend)
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
        for i in range(0, len(chunks_flat), cfg.embed_batch):
            batch = chunks_flat[i : i + cfg.embed_batch]
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
            done = min(i + cfg.embed_batch, len(chunks_flat))
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
    db: Path | None = typer.Option(
        None, "--db", help="SQLite index path (default: from config or index.sqlite)."
    ),
    k: int = typer.Option(8, "-k", "--top-k", help="Number of chunks to retrieve."),
    embed_model: str | None = typer.Option(None, "--embed-model", help="Embedding model."),
    backend: BackendType | None = typer.Option(None, "--backend", help="LLM backend."),
) -> None:
    """Retrieve top-k chunks for a query."""
    cfg = get_config()
    db = db or Path(cfg.db)
    embed_model = embed_model or cfg.embed_model
    client = get_backend(cfg, backend)
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
    db: Path | None = typer.Option(
        None, "--db", help="SQLite index path (default: from config or index.sqlite)."
    ),
    k: int = typer.Option(8, "-k", "--top-k", help="Chunks to pass to the model."),
    embed_model: str | None = typer.Option(None, "--embed-model"),
    chat_model: str | None = typer.Option(None, "--chat-model"),
    backend: BackendType | None = typer.Option(None, "--backend", help="LLM backend."),
    no_rewrite: bool = typer.Option(False, "--no-rewrite", help="Skip query rewrite step."),
    context_file: Path | None = typer.Option(
        None,
        "--context",
        "-c",
        help="Additional context file or directory to retrieve from and include in prompt.",
    ),
    context_k: int = typer.Option(
        3, "--context-k", help="Context sections to retrieve from -c files."
    ),
    stream: bool | None = typer.Option(
        None, "--stream/--no-stream", help="Stream tokens to stdout (default: auto-detect TTY)."
    ),
) -> None:
    """Answer using retrieved context and LLM chat."""
    if stream is None:
        stream = sys.stdout.isatty()
    cfg = get_config()
    db = db or Path(cfg.db)
    embed_model = embed_model or cfg.embed_model
    chat_model = chat_model or cfg.chat_model
    temperature = cfg.temperature
    client = get_backend(cfg, backend)
    context_sections: list[ContextSection] | None = None
    if context_file is not None:
        sections = chunk_context_path(context_file)
        if sections:
            context_sections = retrieve_context_sections(
                client, query, sections, embed_model, context_k
            )
            logger.info(
                "Retrieved %d context sections from %s", len(context_sections), context_file
            )
    try:
        hits, _ = _retrieve_hits(
            client, query, db, k, embed_model, chat_model, no_rewrite, temperature
        )
        if not hits:
            typer.echo("No chunks retrieved.", err=True)
            raise typer.Exit(1)
        context_block = build_context_block(hits)
        user_msg = build_rag_user_content(query, context_block, context_sections=context_sections)
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        if stream:
            try:
                answer = stream_output(
                    client.stream_chat(chat_model, messages, temperature=temperature)
                )
            except httpx.HTTPError as e:
                typer.secho(f"\nError during streaming: {e}", fg="red", err=True)
                raise typer.Exit(1)
        else:
            answer = client.chat(
                chat_model,
                messages,
                temperature=temperature,
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
    db: Path | None = typer.Option(
        None, "--db", help="SQLite index path (default: from config or index.sqlite)."
    ),
    k: int = typer.Option(8, "-k", "--top-k", help="Chunks to pass to the model."),
    embed_model: str | None = typer.Option(None, "--embed-model"),
    chat_model: str | None = typer.Option(None, "--chat-model"),
    backend: BackendType | None = typer.Option(None, "--backend", help="LLM backend."),
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
    context_path: Path | None = typer.Option(
        None,
        "--context",
        "-c",
        help="Additional context file or directory to retrieve from and include in prompt.",
    ),
    context_k: int = typer.Option(
        3,
        "--context-k",
        help="Context sections to retrieve from -c files.",
    ),
    stream: bool | None = typer.Option(
        None, "--stream/--no-stream", help="Stream tokens to stdout (default: auto-detect TTY)."
    ),
) -> None:
    """Generate a grounded Mermaid diagram from retrieved code context."""
    if stream is None:
        stream = sys.stdout.isatty()
    cfg = get_config()
    db = db or Path(cfg.db)
    embed_model = embed_model or cfg.embed_model
    chat_model = chat_model or cfg.chat_model
    temperature = cfg.temperature
    client = get_backend(cfg, backend)
    context_sections: list[ContextSection] | None = None
    if context_path is not None:
        sections = chunk_context_path(context_path)
        if sections:
            context_sections = retrieve_context_sections(
                client, query, sections, embed_model, context_k
            )
            logger.info(
                "Retrieved %d context sections from %s", len(context_sections), context_path
            )
    try:
        hits, _ = _retrieve_hits(
            client, query, db, k, embed_model, chat_model, no_rewrite, temperature
        )
        if not hits:
            typer.echo("No chunks retrieved.", err=True)
            raise typer.Exit(1)
        context_block = build_context_block(hits)
        user_msg = build_rag_user_content(query, context_block, context_sections=context_sections)
        messages = [
            {"role": "system", "content": DIAGRAM_SYSTEM},
            {"role": "user", "content": f"{user_msg}\nOnly output a mermaid code block"},
        ]
        if stream:
            try:
                typer.echo("Generating response...", err=True)
                raw = stream_output(
                    client.stream_chat(chat_model, messages, temperature=temperature),
                )
            except httpx.HTTPError as e:
                typer.secho(f"\nError during streaming: {e}", fg="red", err=True)
                raise typer.Exit(1)
        else:
            raw = client.chat(
                chat_model,
                messages,
                temperature=temperature,
            )
        md, had_fence, shape_ok = format_model_diagram_response(raw, chunks=hits)
        if not had_fence:
            logger.warning("No ```mermaid fence in model output; writing raw response")
        elif not shape_ok:
            logger.warning(
                "Mermaid body does not start with a known diagram type; output may not render"
            )
        if not stream:
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
