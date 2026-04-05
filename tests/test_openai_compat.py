from __future__ import annotations

import httpx
import respx

from reporag.llm import OpenAICompatBackend


@respx.mock
def test_stream_chat() -> None:
    """Test streaming response yields chunks correctly."""
    sse_content = (
        'data: {"choices":[{"delta":{"content":"Hello "}}]}\n'
        'data: {"choices":[{"delta":{"content":"world!"}}]}\n'
        "data: [DONE]\n"
    )
    respx.post("http://test/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_content.encode())
    )
    c = OpenAICompatBackend(base_url="http://test")
    try:
        chunks = list(c.stream_chat("m", [{"role": "user", "content": "hi"}]))
        assert "Hello " in chunks
        assert "world!" in chunks
        assert all(isinstance(c, str) for c in chunks)
    finally:
        c.close()


@respx.mock
def test_stream_chat_handles_done() -> None:
    """Test that streaming stops on [DONE] sentinel."""
    sse_content = (
        'data: {"choices":[{"delta":{"content":"Part1"}}]}\n'
        'data: {"choices":[{"delta":{"content":"Part2"}}]}\n'
        "data: [DONE]\n"
        'data: {"choices":[{"delta":{"content":"Should not appear"}}]}\n'
    )
    respx.post("http://test/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_content.encode())
    )
    c = OpenAICompatBackend(base_url="http://test")
    try:
        chunks = list(c.stream_chat("m", [{"role": "user", "content": "hi"}]))
        assert chunks == ["Part1", "Part2"]
    finally:
        c.close()


@respx.mock
def test_stream_chat_skips_empty_content() -> None:
    """Test that empty content strings are skipped in streaming."""
    sse_content = (
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
        'data: {"choices":[{"delta":{"content":""}}]}\n'
        'data: {"choices":[{"delta":{"content":"world"}}]}\n'
        "data: [DONE]\n"
    )
    respx.post("http://test/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_content.encode())
    )
    c = OpenAICompatBackend(base_url="http://test")
    try:
        chunks = list(c.stream_chat("m", [{"role": "user", "content": "hi"}]))
        assert chunks == ["Hello", "world"]
    finally:
        c.close()


@respx.mock
def test_stream_chat_with_api_key() -> None:
    """Test streaming with API key authentication."""
    sse_content = 'data: {"choices":[{"delta":{"content":"Hello"}}]}\ndata: [DONE]\n'
    route = respx.post("http://test/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_content.encode())
    )
    c = OpenAICompatBackend(base_url="http://test", api_key="test-key")
    try:
        chunks = list(c.stream_chat("m", [{"role": "user", "content": "hi"}]))
        assert chunks == ["Hello"]
        assert "Authorization" in route.calls[0].request.headers
        assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"
    finally:
        c.close()
