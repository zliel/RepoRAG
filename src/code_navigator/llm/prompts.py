from __future__ import annotations

from code_navigator.retrieval.search import RetrievedChunk

REWRITE_SYSTEM = (
    "You write search queries for use in a RAG retrieval system. Output only the search query text. It should always reference 'this system'"
    # "You rewrite user questions into short search queries for retrieving code. "
    # "Output only the search query text, no quotes or explanation."
)

ANSWER_SYSTEM = """You are a careful assistant answering questions about a codebase.

Rules:
- Use ONLY the provided code excerpts. Do not invent files, symbols, or behavior not shown.
- Every factual claim about the code must include a citation in the form: path (lines START-END).
- If the excerpts are insufficient, say so clearly and answer only what is supported.
- Do not cite paths or line ranges that are not present in the provided CITATION headers."""

DIAGRAM_SYSTEM = (
    "You draw diagrams of Python code structure and flow based ONLY on "
    "retrieved excerpts.\n\n"
    "Rules:\n"
    "- Represent ONLY relationships, calls, or structure justified by the "
    "provided [CITATION ...] blocks.\n"
    "- Do NOT add modules, classes, functions, or edges not supported by "
    "those citations.\n"
    "- If excerpts are insufficient, say so in one short sentence, then "
    "output a minimal Mermaid diagram (e.g. one node 'Insufficient context' "
    "or only cited symbols, no guessed links).\n"
    "In the fenced mermaid block, you should always define a diagram type. "
    "- Prefer simple Mermaid: flowchart TD or LR, sequenceDiagram, or "
    "classDiagram. Avoid exotic diagram types.\n\n"
    "Output format (REQUIRED):\n"
    "1) A Title"
    "2) A brief description of the flow as a numbered list. "
    "3) Exactly one fenced block:\n\n"
    "```mermaid\n"
    "...valid mermaid source...\n"
    "```\n\n"
    "CRITICAL: For node labels in Mermaid, you MUST wrap the label text in "
    "double quotes.\n"
    "## Example: A[\"login_handler (app.py lines 10-20)\"] --> B[\"SessionManager (auth.py)\"]\n"
    "Use stable node IDs (e.g. A, B, C) with descriptive quoted labels ALWAYS in the same format as in the example provided. "
    "Nodes in the diagram should not made be without a quoted label unless explicitly asked for. "
    "NEVER output the word CITATION in the legend or anywhere in your response."
)


def build_rag_user_content(query: str, context: str) -> str:
    """Shared user message body for ask/diagram (question + retrieved code)."""
    return (
        f"User question:\n{query}\n\n"
        f"Retrieved code (cite only these paths and line ranges):\n\n{context}"
    )


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        header = (
            f"[CITATION id={i} path={c.path} lines={c.start_line}-{c.end_line} symbol={c.symbol}]"
        )
        parts.append(header + "\n" + c.text)
    return "\n\n---\n\n".join(parts)
