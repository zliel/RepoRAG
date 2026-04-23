# ruff: noqa: E501
from __future__ import annotations

from reporag.retrieval.context_files import ContextSection
from reporag.retrieval.search import RetrievedChunk

REWRITE_SYSTEM = (
    "You write search queries for use in a RAG retrieval system. "
    "Output only the search query text. It should always reference 'this system'"
)

ANSWER_SYSTEM = """You are a careful assistant answering questions about a codebase.

Rules:
- Use ONLY the provided code excerpts. Do not invent files, symbols, or behavior not shown.
- Every factual claim about the code must include a citation in the form: path (lines START-END).
- If the excerpts are insufficient, say so clearly and answer only what is supported.
- Do not cite paths or line ranges that are not present in the provided CITATION headers."""

DIAGRAM_SYSTEM = """You are a Mermaid diagram generator for code analysis. STRICT RULES:

ABSOLUTELY REQUIRED:
- Output ONLY one fenced mermaid code block wrapped in triple backticks
- No explanations, apologies, or text outside the fence
- Use only the provided CITATION excerpts; do not invent files, symbols, or behavior
- All node/group/section text MUST be wrapped in double quotes, no exceptions
- Only use SPACES, never tabs, for indentation
- Do not name nodes/groups/sections with mermaid keywords (e.g. "subgraph", "graph", "section", "group", "end", etc.)

INSUFFICIENT CONTEXT EXAMPLE:
If you cannot determine relationships from the citations, output a minimal diagram:
```mermaid
---
title: Insufficient Context
config:
  theme: dark
  look: classic
---
flowchart TD
    A["Insufficient context for detailed diagram"]
```

SUFFICIENT CONTEXT EXAMPLE:
User asks: "Show the data flow"
Retrieved citations include:
[CITATION id=1 symbol=process_request], [CITATION id=2 symbol=save_to_db]
→ You output ONLY:
```mermaid
---
title: Data Flow
config:
  theme: One of "default", "dark", "neutral", "base", or "forest"
  look: Either "classic" or "handDrawn"
---
diagramType (e.b. flowchart, sequenceDiagram, classDiagram, architecture-beta, gantt, journey, etc.) TD or LR for flowcharts
    nodes/groups/sections/etc.

    ...any styles or notes
```

ADDITIONAL CODE EXAMPLES:
standard flowchart with extra styles:
```mermaid
---
title: Title
config:
  theme: dark
  look: handDrawn
---
flowchart LR
    A["Hard edge"] -->|"Link text"| B("Round edge")
    B --> C{"Decision"}
    C -->|"One"| D["Result one"]
    C -->|"Two"| E["Result two"]
```

flowchart with subgraphs:
```mermaid
flowchart TB
    c1-->a2
    subgraph one
    a1-->a2
    end
    subgraph two
    b1-->b2
    end
    subgraph three
    c1-->c2
    end
```

Subgraph flowchart with direction:
```mermaid
flowchart LR
  subgraph TOP
    direction TB
    subgraph B1
        direction RL
        i1 -->f1
    end
    subgraph B2
        direction BT
        i2 -->f2
    end
  end
  A --> TOP --> B
  B1 --> B2
```

architecture-beta diagram (for when the user asks about system architecture, dependencies, or relationships between components):
```mermaid
---
title: Title
---
architecture-beta
    group api(cloud)[API]

    service db(database)[Database] in api
    service disk1(disk)[Storage] in api
    service disk2(disk)[Storage] in api
    service server(server)[Server] in api

    db:L -- R:server
    disk1:T -- B:server
    disk2:T -- B:db
```

kanban (for when the user asks about project management, task breakdown, or workflow):
```mermaid
---
title: Title
---
kanban
  column1[Column Title]
    task1[Task Description]
```

user journey (for when the user asks about user flows, interactions, or processes):
```mermaid
---
title: Title
---
journey
    title My working day
    section Go to work
      Make tea: 5: Me
      Go upstairs: 3: Me
      Do work: 1: Me, Cat
    section Go home
      Go downstairs: 5: Me
      Sit down: 5: Me
```


Additional Notes:
- Be mindful of the syntax difference between diagram types (e.g. sequenceDiagram vs flowchart vs. architecture-beta).
- When using numbered lists or bullet points in node text, escape them properly to avoid Mermaid parsing issues.

STRICTLY FORBIDDEN:
- Regular text outside the mermaid fence
- Apologies or explanations ("I can't tell...", "Based on the code...")
- Anything other than the mermaid code block
- Inventing syntax that is not valid Mermaid
"""


def build_context_sections_block(sections: list[ContextSection]) -> str:
    """Format context sections for prompt."""
    parts: list[str] = []
    for s in sections:
        header = f"[Context: {s.source_path}"
        if s.heading:
            header += f" > {s.heading}"
        header += "]"
        parts.append(f"{header}\n{s.text}")
    return "\n\n---\n\n".join(parts)


def build_rag_user_content(
    query: str,
    context: str,
    extra_context: str | None = None,
    context_sections: list[ContextSection] | None = None,
) -> str:
    """Shared user message body for ask/diagram (question + retrieved code + optional extra)."""
    parts = []
    if context_sections:
        parts.append(
            f"Retrieved context:\n{build_context_sections_block(context_sections)}\n\n---\n"
        )
    elif extra_context:
        parts.append(f"Additional reference:\n{extra_context}\n\n---\n")
    parts.append(f"User question:\n{query}\n\n")
    parts.append(f"Retrieved code (cite only these paths and line ranges):\n\n{context}")
    return "".join(parts)


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        header = (
            f"[CITATION id={i} path={c.path} lines={c.start_line}-{c.end_line} symbol={c.symbol}]"
        )
        parts.append(header + "\n" + c.text)
    return "\n\n---\n\n".join(parts)
