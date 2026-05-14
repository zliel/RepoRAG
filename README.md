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

# Retry settings for transient HTTP errors (timeouts, 5xx)
max_retries = 3
backoff_factor = 2.0

# Cross-encoder reranking (opt-in; adds latency but improves relevance)
[reporag.rerank]
enabled = false
top_k = 20
final_k = 8
method = "llm"
# model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
| `max_retries` | `3` | Maximum retry attempts for transient HTTP errors |
| `backoff_factor` | `2.0` | Exponential backoff multiplier (1s → 2s → 4s …) |

#### Rerank options

The `[reporag.rerank]` section configures relevance reranking. When enabled, the top *top_k* first-stage retrieval results are re-scored using either the chat LLM or a local cross-encoder model, and only the best *final_k* are kept for the final answer.

| Option | Default | Description |
|--------|---------|-------------|
| `enabled` | `false` | Enable reranking (opt-in) |
| `top_k` | `20` | Number of first-stage results to rerank |
| `final_k` | `8` | Number of results to keep after reranking |
| `method` | `"llm"` | Reranking method: `"llm"` or `"cross-encoder"` |
| `model` | `""` | Cross-encoder model name (e.g. `"cross-encoder/ms-marco-MiniLM-L-6-v2"`) |

## Reranking

Reranking re-scores the top first-stage retrieval results to improve relevance before the LLM generates an answer. It is **opt-in** (disabled by default) since it adds latency.

### LLM-based reranking (default)

When `method = "llm"`, each batch of passages is sent to your configured chat model with a scoring prompt. No extra dependencies required — works with any backend (Ollama, OpenAI-compatible, etc.).

```toml
[reporag.rerank]
enabled = true
method = "llm"
top_k = 20
final_k = 8
```

### Cross-encoder reranking (faster, requires PyTorch)

When `method = "cross-encoder"`, RepoRAG uses a local cross-encoder model from [sentence-transformers](https://www.sbert.net/) to score passages. This is significantly faster than LLM-based scoring (batched prediction, no token generation) but requires PyTorch.

**Setup:**

```bash
pip install sentence-transformers
```

**Usage:**

```toml
[reporag.rerank]
enabled = true
method = "cross-encoder"
model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_k = 20
final_k = 8
```

Good cross-encoder models for code/document relevance:

| Model | Notes |
|-------|-------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast, good general-purpose (default recommendation) |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | More accurate, slightly slower |
| `BAAI/bge-reranker-v2-m3` | Multilingual, strong zero-shot performance |
| `jinaai/jina-reranker-v2-base-multilingual` | Good for mixed-language codebases |

Reranking applies to both the `search` and `ask` commands.

## Supported languages

| Language | Extensions | Parser |
|----------|-----------|--------|
| Python | `.py` | tree-sitter-python |
| JavaScript | `.js`, `.jsx` | tree-sitter-javascript |
| TypeScript | `.ts`, `.tsx` | tree-sitter-typescript |
| Go | `.go` | tree-sitter-go |
| Rust | `.rs` | tree-sitter-rust |
| Plain text (fallback) | any other file | line-based chunking |
