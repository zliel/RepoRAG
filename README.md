# RepoRAG (MVP)

Local-first CLI to ask natural-language questions about a Python codebase using **Ollama** (embeddings + chat) and **SQLite** storage.

## Setup

Requires Python 3.11+, [Ollama](https://ollama.com/) running locally, and embedding/chat models pulled (defaults: `nomic-embed-text-v2-moe`, `qwen3-vl:8b-instruct` — run `ollama pull` for each).

```bash
pip install -e ".[dev]"
```

Optional: set `OLLAMA_HOST` (default `http://127.0.0.1:11434`).

## Commands

- `reporag list <root>` — list `.py` files
- `reporag chunks <root>` — print chunk metadata (JSON lines)
- `reporag index <root> --db ./index.sqlite` — build index
- `reporag search "<query>" --db ./index.sqlite -k 8` — top-k retrieval
- `reporag ask "<query>" --db ./index.sqlite` — answer with citations
- `reporag diagram "<query>" --db ./index.sqlite` — grounded **Mermaid** diagram (Markdown to stdout)
- `reporag diagram "…" --db ./toy_index.sqlite -o flow.md` — write the same Markdown to a file (open in VS Code / Cursor with a Mermaid preview extension)

Use `-v` / `--verbose` for debug logs. Override models with `--embed-model` and `--chat-model`.

### Diagrams

Output is Markdown containing a ` ```mermaid ` … ` ``` ` block, normalized from the model response. If the model omits a fence, the raw text is printed and a warning is logged. A light check warns when the diagram type is not one of: `flowchart`, `sequenceDiagram`, `classDiagram`, `graph`, `stateDiagram`, `erDiagram`, `mindmap`, `timeline`.

**Optional PNG/SVG:** install [Mermaid CLI](https://github.com/mermaid-js/mermaid-cli) (`npm install -g @mermaid-js/mermaid-cli`), save with `--out diagram.md`, then run `mmdc -i diagram.md -o diagram.png` (or `.svg`). Not wired into `reporag` yet.

## Integration tests

Set `OLLAMA_INTEGRATION=1` to run tests that call a live Ollama instance.
