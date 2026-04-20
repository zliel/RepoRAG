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
from tqdm import tqdm

from reporag.config import Config
from reporag.indexing.store import open_index
from reporag.ingestion.walker import (
    get_supported_extensions_display,
    parse_file,
    walk_supported_files,
)
from reporag.llm.backends import BackendType, LLMBackend, create_backend
from reporag.llm.diagram import format_model_diagram_response
from reporag.llm.prompts import (
    ANSWER_SYSTEM,
    DIAGRAM_SYSTEM,
    REWRITE_SYSTEM,
    build_context_block,
    build_rag_user_content,
)
from reporag.retrieval.context_files import (
    ContextSection,
    chunk_context_path,
    retrieve_context_sections,
)
from reporag.retrieval.search import RetrievedChunk, top_k_similar, hybrid_search
from reporag.types import Chunk

logger = logging.getLogger(__name__)

app = typer.Typer(help="Local-first semantic codebase navigator with RAG (Ollama + SQLite).")


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
    no_hybrid: bool = False,
) -> tuple[list[RetrievedChunk], str]:
    """
    Open index, optional rewrite, embed search string, return top-k chunks and search_query used.

    Args:
        no_hybrid: If True, skip FTS5 hybrid and use vector-only search.
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

    if no_hybrid:
        hits = top_k_similar(q, mat, meta, k)
    else:
        # Hybrid search: get FTS5 results and combine with RRF
        idx = open_index(db)
        try:
            fts_results = idx.search_fts(search_query, k * 2)
        finally:
            idx.close()
        hits = hybrid_search(q, mat, meta, fts_results, k)

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
    """List all supported files under root."""
    root = root.resolve()
    paths = walk_supported_files(root)
    for p in paths:
        typer.echo(p.relative_to(root).as_posix())
    ext_display = get_supported_extensions_display()
    logger.info("Found %d supported files (%s)", len(paths), ext_display)
    typer.echo(f"Total: {len(paths)}", err=True)


@app.command("chunks")
def cmd_chunks(
    root: Path = typer.Argument(..., exists=True, file_okay=False, help="Project root."),
) -> None:
    """Extract function/class chunks and print JSON lines."""
    root = root.resolve()
    total = 0
    for p in walk_supported_files(root):
        for ch in parse_file(p, root):
            typer.echo(
                json.dumps(
                    {
                        "path": ch.path,
                        "symbol": ch.symbol_name,
                        "language": ch.language,
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
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force full reindex: clear DB and reindex all files.",
    ),
    graph: bool = typer.Option(
        False,
        "--graph/--no-graph",
        help="Build code graph (imports/calls) for graph-based retrieval.",
    ),
) -> None:
    """Build embedding index for all supported source files (incremental by default)."""
    cfg = get_config()
    root = root.resolve()
    db = db or Path(cfg.db)
    embed_model = embed_model or cfg.embed_model
    client = get_backend(cfg, backend)
    idx = open_index(db)

    httpx_logger = logging.getLogger("httpx")
    original_httpx_level = httpx_logger.level
    httpx_logger.setLevel(logging.WARNING)

    try:
        import time

        current_time = time.time()
        indexed_mtimes: dict[str, tuple[float, float]] = {}
        if not force:
            indexed_mtimes = idx.get_all_file_mtimes()

        tqdm.write("Discovering files...", file=sys.stderr)
        file_paths = walk_supported_files(root)

        to_reindex: list[Path] = []
        to_remove: list[str] = []
        unchanged_count = 0

        for p in file_paths:
            rel = p.relative_to(root).as_posix()
            mtime = p.stat().st_mtime
            if rel in indexed_mtimes:
                indexed_mtime, _ = indexed_mtimes[rel]
                if mtime <= indexed_mtime:
                    unchanged_count += 1
                else:
                    to_reindex.append(p)
            else:
                to_reindex.append(p)

        indexed_paths = {p.relative_to(root).as_posix() for p in file_paths}
        for rel in indexed_mtimes:
            if rel not in indexed_paths:
                to_remove.append(rel)

        total_files = len(file_paths)
        ext_display = get_supported_extensions_display()
        tqdm.write(
            f"Found {total_files} supported files ({ext_display}): "
            f"{unchanged_count} unchanged, "
            f"{len(to_reindex)} to reindex, {len(to_remove)} to remove.",
            file=sys.stderr,
        )

        if force:
            idx.clear()
            to_reindex = list(file_paths)
            to_remove = []
            tqdm.write("Force flag: full reindex.", file=sys.stderr)

        if to_remove:
            idx.delete_chunks_by_paths(to_remove)
            idx.delete_file_metadata_by_paths(to_remove)
            for path in to_remove:
                del indexed_mtimes[path]
            tqdm.write(f"Removed {len(to_remove)} deleted files from index.", file=sys.stderr)

        if not to_reindex:
            tqdm.write("No files to reindex.", file=sys.stderr)
            return

        chunks_flat: list[Chunk] = []
        with tqdm(
            total=len(to_reindex),
            desc="Parsing files",
            unit="file",
            file=sys.stderr,
        ) as pbar:
            for p in to_reindex:
                chunks = parse_file(p, root)
                chunks_flat.extend(chunks)
                rel = p.relative_to(root).as_posix()
                idx.upsert_file_mtime(rel, p.stat().st_mtime, current_time)
                pbar.update(1)

        if not chunks_flat:
            logger.warning("No chunks found in reindexed files")
            return

        lang_counts: dict[str, int] = {}
        for ch in chunks_flat:
            lang_counts[ch.language] = lang_counts.get(ch.language, 0) + 1

        tqdm.write(
            f"Extracted {len(chunks_flat)} chunks from {len(to_reindex)} files "
            f"({', '.join(f'{k}:{v}' for k, v in sorted(lang_counts.items()))}). Embedding...",
            file=sys.stderr,
        )
        dim: int | None = None
        with tqdm(total=len(chunks_flat), desc="Embedding", unit="chunk", file=sys.stderr) as pbar:
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
                pbar.update(len(batch))
        if dim is not None:
            idx.set_meta("embed_dim", str(dim))
        idx.set_meta("embed_model", embed_model)
        idx._conn.commit()

        new_or_updated = len(to_reindex)
        removed = len(to_remove)
        total_chunks = idx.chunk_count()

        # Build code graph if requested
        if graph and total_chunks > 0:
            tqdm.write("Building code graph...", file=sys.stderr)
            # Re-load chunks with source text
            source_map: dict[str, str] = {}
            for p in file_paths:
                try:
                    rel = p.relative_to(root).as_posix()
                    source_map[rel] = p.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning("Failed to read %s: %s", p, e)

            # Update chunks with source text
            for rel, source in source_map.items():
                idx._conn.execute(
                    "UPDATE chunks SET source_text = ? WHERE path = ?",
                    (source, rel),
                )
            idx._conn.commit()

            # Build graph
            edge_count = idx.build_code_graph(base_path=root)
            tqdm.write(f"Built code graph with {edge_count} edges.", file=sys.stderr)

        tqdm.write(
            f"Done. Indexed {new_or_updated} file(s), removed {removed} file(s), "
            f"{unchanged_count} unchanged. Total chunks: {total_chunks}.",
            file=sys.stderr,
        )
    finally:
        httpx_logger.setLevel(original_httpx_level)
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
    no_hybrid: bool = typer.Option(
        False, "--no-hybrid", help="Use vector search only, skip FTS5 hybrid."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Hide duplicate location info."),
    use_graph: bool = typer.Option(
        False,
        "--graph/--no-graph",
        help="Use graph-based retrieval to find related chunks.",
    ),
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
        if no_hybrid:
            hits = top_k_similar(q, mat, meta, k)
        else:
            # Hybrid search: get FTS5 results and combine with RRF
            idx = open_index(db)
            try:
                fts_results = idx.search_fts(query, k * 2)
            finally:
                idx.close()
            hits = hybrid_search(q, mat, meta, fts_results, k)

        # Optionally enhance with graph-based related chunks
        if use_graph:
            from reporag.indexing.store import ChunkIndex

            idx_for_graph = open_index(db)
            try:
                # Add related chunks from graph
                graph_related: list[RetrievedChunk] = []
                for h in hits[:3]:  # Get related for top 3 semantic hits
                    related = idx_for_graph.get_related_chunks(h.chunk_id, max_results=3)
                    for rel in related:
                        # Check if not already in hits
                        if not any(r.chunk_id == rel["id"] for r in hits):
                            graph_related.append(
                                RetrievedChunk(
                                    chunk_id=rel["id"],
                                    path=rel["path"],
                                    symbol=rel["symbol"],
                                    start_line=rel["start_line"],
                                    end_line=rel["end_line"],
                                    text=rel["text"],
                                    language=rel["language"],
                                    score=h.score * 0.9,  # Slightly lower score
                                    canonical_id=None,
                                    aliases=(),
                                )
                            )
                hits = list(hits) + graph_related
            finally:
                idx_for_graph.close()

        for h in hits:
            output: dict[str, object] = {
                "score": round(h.score, 6),
                "path": h.path,
                "symbol": h.symbol,
                "start_line": h.start_line,
                "end_line": h.end_line,
                "preview": h.text[:200] + ("…" if len(h.text) > 200 else ""),
            }
            if not quiet and h.aliases:
                output["also_in"] = list(h.aliases)
            typer.echo(json.dumps(output, ensure_ascii=False))
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
    no_hybrid: bool = typer.Option(
        False, "--no-hybrid", help="Use vector search only, skip FTS5 hybrid."
    ),
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
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Hide duplicate location info."),
    use_graph: bool = typer.Option(
        False,
        "--graph/--no-graph",
        help="Use graph-based retrieval to find related chunks.",
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
            client, query, db, k, embed_model, chat_model, no_rewrite, temperature, no_hybrid
        )
        if not hits:
            typer.echo("No chunks retrieved.", err=True)
            raise typer.Exit(1)

        # Optionally enhance with graph-based related chunks
        if use_graph:
            idx_for_graph = open_index(db)
            try:
                graph_related: list[RetrievedChunk] = []
                for h in hits[:3]:  # Get related for top 3 semantic hits
                    related = idx_for_graph.get_related_chunks(h.chunk_id, max_results=3)
                    for rel in related:
                        # Check if not already in hits
                        if not any(r.chunk_id == rel["id"] for r in hits):
                            graph_related.append(
                                RetrievedChunk(
                                    chunk_id=rel["id"],
                                    path=rel["path"],
                                    symbol=rel["symbol"],
                                    start_line=rel["start_line"],
                                    end_line=rel["end_line"],
                                    text=rel["text"],
                                    language=rel["language"],
                                    score=h.score * 0.9,  # Slightly lower score
                                    canonical_id=None,
                                    aliases=(),
                                )
                            )
                hits = list(hits) + graph_related
            finally:
                idx_for_graph.close()

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
            source_line = f"  {h.path} lines {h.start_line}-{h.end_line} ({h.symbol})"
            if not quiet and h.aliases:
                source_line += f" (also in: {', '.join(h.aliases)})"
            typer.echo(source_line, err=True)
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
    no_hybrid: bool = typer.Option(
        False, "--no-hybrid", help="Use vector search only, skip FTS5 hybrid."
    ),
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
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Hide duplicate location info."),
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
            client, query, db, k, embed_model, chat_model, no_rewrite, temperature, no_hybrid
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
            source_line = f"  {h.path} lines {h.start_line}-{h.end_line} ({h.symbol})"
            if not quiet and h.aliases:
                source_line += f" (also in: {', '.join(h.aliases)})"
            typer.echo(source_line, err=True)
    finally:
        client.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
