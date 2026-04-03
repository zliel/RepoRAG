from __future__ import annotations

import httpx
import respx

from reporag.llm import OllamaClient
from reporag.llm.diagram import format_model_diagram_response


@respx.mock
def test_embed_api_embed_endpoint() -> None:
    respx.post("http://test/api/embed").mock(
        return_value=httpx.Response(
            200,
            json={"embeddings": [[0.0, 1.0], [1.0, 0.0]]},
        )
    )
    c = OllamaClient(base_url="http://test")
    try:
        out = c.embed(["a", "b"], "m")
        assert len(out) == 2
        assert out[0] == [0.0, 1.0]
    finally:
        c.close()


@respx.mock
def test_embed_fallback_legacy() -> None:
    respx.post("http://test/api/embed").mock(return_value=httpx.Response(404))
    respx.post("http://test/api/embeddings").mock(
        side_effect=[
            httpx.Response(200, json={"embedding": [0.0, 0.0, 1.0]}),
            httpx.Response(200, json={"embedding": [0.0, 1.0, 0.0]}),
        ]
    )
    c = OllamaClient(base_url="http://test")
    try:
        out = c.embed(["x", "y"], "m")
        assert len(out) == 2
        assert len(out[0]) == 3
    finally:
        c.close()


@respx.mock
def test_chat() -> None:
    respx.post("http://test/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "hello"}},
        )
    )
    c = OllamaClient(base_url="http://test")
    try:
        text = c.chat("m", [{"role": "user", "content": "hi"}])
        assert text == "hello"
    finally:
        c.close()


@respx.mock
def test_chat_mermaid_response_parses() -> None:
    content = """Legend: A = citation 1

```mermaid
flowchart TD
  A --> B
```
"""
    respx.post("http://test/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": content}},
        )
    )
    c = OllamaClient(base_url="http://test")
    try:
        text = c.chat("m", [{"role": "user", "content": "draw"}])
        md, had_fence, shape_ok = format_model_diagram_response(text)
        assert had_fence and shape_ok
        assert "flowchart TD" in md
    finally:
        c.close()
