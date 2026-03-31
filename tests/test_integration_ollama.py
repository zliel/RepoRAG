from __future__ import annotations

import os

import pytest

from reporag.llm.ollama_client import OllamaClient


@pytest.mark.skipif(
    os.environ.get("OLLAMA_INTEGRATION") != "1",
    reason="Set OLLAMA_INTEGRATION=1 to run against local Ollama",
)
def test_live_embed_minimal() -> None:
    model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")
    c = OllamaClient()
    try:
        out = c.embed(["test phrase"], model)
        assert len(out) == 1
        assert len(out[0]) > 0
    finally:
        c.close()
