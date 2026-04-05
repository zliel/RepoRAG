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


@respx.mock
def test_stream_chat() -> None:
    """Test streaming response yields chunks correctly."""
    ndjson_content = (
        b'{"message": {"content": "Hello "}}\n'
        b'{"message": {"content": "world!"}}\n'
        b'{"message": {"content": ""}, "done": true}'
    )
    respx.post("http://test/api/chat").mock(
        return_value=httpx.Response(200, content=ndjson_content)
    )
    c = OllamaClient(base_url="http://test")
    try:
        chunks = list(c.stream_chat("m", [{"role": "user", "content": "hi"}]))
        assert "Hello " in chunks
        assert "world!" in chunks
        assert all(isinstance(c, str) for c in chunks)
    finally:
        c.close()


@respx.mock
def test_stream_chat_skips_empty_content() -> None:
    """Test that empty content strings are skipped in streaming."""
    ndjson_responses = [
        b'{"message": {"content": "Hello"}}',
        b'{"message": {"content": ""}}',
        b'{"message": {"content": "world"}}',
        b'{"message": {"content": ""}, "done": true}',
    ]

    respx.post("http://test/api/chat").mock(
        return_value=httpx.Response(200, content=b"\n".join(ndjson_responses))
    )
    c = OllamaClient(base_url="http://test")
    try:
        chunks = list(c.stream_chat("m", [{"role": "user", "content": "hi"}]))
        assert chunks == ["Hello", "world"]
    finally:
        c.close()


@respx.mock
def test_stream_chat_handles_done_field() -> None:
    """Test that streaming continues past done=False until done=True."""
    ndjson_responses = [
        b'{"message": {"content": "Part1"}, "done": false}',
        b'{"message": {"content": "Part2"}, "done": false}',
        b'{"message": {"content": "Part3"}, "done": true}',
    ]
    respx.post("http://test/api/chat").mock(
        return_value=httpx.Response(200, content=b"\n".join(ndjson_responses))
    )
    c = OllamaClient(base_url="http://test")
    try:
        chunks = list(c.stream_chat("m", [{"role": "user", "content": "hi"}]))
        assert chunks == ["Part1", "Part2", "Part3"]
    finally:
        c.close()
