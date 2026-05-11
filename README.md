# RepoRAG

Local-first CLI to ask natural-language questions about a codebase using LLMs and SQLite storage.

Supports **Python, JavaScript/JSX, TypeScript/TSX, Go, and Rust** out of the box, with a plain-text fallback for other file types.

## Setup

Requires Python 3.11+.

### Quick start (Ollama)

1. Install and run [Ollama](https://ollama.com/)
2. Pull the default models:
   ```bash
   ollama pull nomic-embed-text-v2-moe
   ollama pull qwen3-vl:8b-instruct
   ```
3. Install reporag:
   ```bash
   pip install -e ".[dev]"
   ```

### Other backends

RepoRAG also supports OpenAI-compatible backends. See [Backends](#backends) and [Configuration](#configuration).

## Commands

All commands support `-v` / `--verbose` for debug logs and `--version` to print the package version.

### Index

```bash
# Build (or incrementally update) the embedding index for a project
reporag index <root> --db ./index.sqlite

# Force full reindex
reporag index <root> --db ./index.sqlite --force

# Build with code graph for graph-based retrieval
reporag index <root> --db ./index.sqlite --graph

# Exclude additional glob patterns
reporag index <root> --exclude "tests/" --exclude "*.pyc"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `./index.sqlite` | SQLite index path |
| `--embed-model` | *(config)* | Embedding model override |
| `--backend` | `ollama` | LLM backend |
| `--force`, `-f` | `false` | Clear DB and reindex all files |
| `--graph/--no-graph` | `false` | Build code graph (imports/calls) |
| `--exclude` | — | Additional exclude glob patterns (repeatable) |

### List

```bash
reporag list <root>
```

Lists all supported source files (`.py`, `.js`, `.jsx`, `.ts`, `.tsx`, `.go`, `.rs`) under *root*, relative paths.

### Chunks

```bash
reporag chunks <root>
```

Extracts function/class chunks from supported files and prints chunk metadata as JSON lines.

### Search

```bash
# Top-k semantic search
reporag search "<query>" --db ./index.sqlite -k 8

# Vector-only search (skip FTS5 hybrid)
reporag search "<query>" --db ./index.sqlite --no-hybrid

# Graph-enhanced search
reporag search "<query>" --db ./index.sqlite --graph
```

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `./index.sqlite` | SQLite index path |
| `-k`, `--top-k` | `8` | Number of chunks to retrieve |
| `--embed-model` | *(config)* | Embedding model override |
| `--backend` | `ollama` | LLM backend |
| `--no-hybrid` | `false` | Use vector search only (skip FTS5) |
| `--quiet`, `-q` | `false` | Hide duplicate/alias info |
| `--graph/--no-graph` | `false` | Graph-based related-chunk expansion |
| `--graph-k` | `3` | Top results to expand via graph |

### Ask

```bash
# Ask a question about the codebase
reporag ask "<query>" --db ./index.sqlite

# With extra context from a file or directory
reporag ask "<query>" --db ./index.sqlite --context ./docs/

# Skip query rewrite and disable streaming
reporag ask "<query>" --db ./index.sqlite --no-rewrite --no-stream
```

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `./index.sqlite` | SQLite index path |
| `-k`, `--top-k` | `8` | Chunks to pass to the model |
| `--embed-model` | *(config)* | Embedding model override |
| `--chat-model` | *(config)* | Chat model override |
| `--backend` | `ollama` | LLM backend |
| `--no-rewrite` | `false` | Skip query rewrite step |
| `--no-hybrid` | `false` | Vector search only (skip FTS5) |
| `--context`, `-c` | — | Context file/directory to include |
| `--context-k` | `3` | Context sections to retrieve |
| `--stream/--no-stream` | *(auto)* | Stream tokens (auto: TTY) |
| `--quiet`, `-q` | `false` | Hide duplicate/alias info |
| `--graph/--no-graph` | `false` | Graph-based related-chunk expansion |
| `--graph-k` | `3` | Top results to expand via graph |

### Diagram

```bash
# Generate a Mermaid diagram to stdout
reporag diagram "<query>" --db ./index.sqlite

# Write Markdown to file and render PNG
reporag diagram "<query>" --db ./index.sqlite -o flow.md -p flow.png
```

Output is Markdown containing a ` ```mermaid … ``` ` block, normalized from the model response. Requires `--png`/`-p` and [Mermaid CLI](https://github.com/mermaid-js/mermaid-cli) (`npm install -g @mermaid-js/mermaid-cli`) for PNG/SVG rendering.

Supports all flags from [`ask`](#ask) (except `--graph`/`--graph-k`), plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--out`, `-o` | — | Write Markdown to this file |
| `--png`, `-p` | — | Generate PNG via Mermaid CLI |

### Stats

```bash
reporag stats --db ./index.sqlite
```

Shows index statistics: total chunks, files, embed model/dimension, languages, graph edges, and creation timestamp.

## Backends

| Backend | Configuration |
|---------|--------------|
| `ollama` (default) | `base_url` or defaults to `http://127.0.0.1:11434` |
| `vllm` | `base_url` required |
| `llamacpp` | `base_url` required |
| `lmstudio` | `base_url` required |
| `lmstudio-local` | `base_url` required |

Override with `--backend` on applicable commands, or set `backend` in [configuration](#configuration).

## Configuration

RepoRAG reads from the first config file found:

1. `.reporag.toml` (project root)
2. `~/.config/reporag/config.toml`
3. `~/.reporag.toml`

If none exist, a default config is created at `~/.config/reporag/config.toml`.

Example config:

```toml
[reporag]
backend = "ollama"
# base_url = "http://127.0.0.1:11434"
embed_model = "nomic-embed-text-v2-moe"
chat_model = "qwen3-vl:8b-instruct"
temperature = 0.2
db = "index.sqlite"
embed_batch = 32
# api_key = ""
# timeout = 60
# exclude_patterns = ["tests/", "venv/"]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `backend` | `"ollama"` | LLM backend type |
| `base_url` | — | API base URL (required for non-Ollama backends) |
| `api_key` | — | API key for remote backends |
| `embed_model` | `"nomic-embed-text-v2-moe"` | Embedding model name |
| `chat_model` | `"qwen3-vl:8b-instruct"` | Chat model name |
| `temperature` | `0.2` | LLM temperature (0.0–1.0) |
| `db` | `"index.sqlite"` | Default index path |
| `embed_batch` | `32` | Embedding batch size |
| `timeout` | — | HTTP timeout (seconds) |
| `exclude_patterns` | — | Glob patterns to exclude from indexing |

## Supported languages

| Language | Extensions | Parser |
|----------|-----------|--------|
| Python | `.py` | tree-sitter-python |
| JavaScript | `.js`, `.jsx` | tree-sitter-javascript |
| TypeScript | `.ts`, `.tsx` | tree-sitter-typescript |
| Go | `.go` | tree-sitter-go |
| Rust | `.rs` | tree-sitter-rust |
| Plain text (fallback) | any other file | line-based chunking |
