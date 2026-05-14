"""Tests for the cross-encoder / LLM reranking module."""

from __future__ import annotations

from unittest.mock import MagicMock

from reporag.retrieval.reranking import _parse_scores, _truncate, rerank_chunks
from reporag.retrieval.search import RetrievedChunk

# ── helpers ──────────────────────────────────────────────────────────────


def _make_chunk(
    chunk_id: int,
    path: str = "a.py",
    symbol: str = "foo",
    text: str = "def foo():\n    pass",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        path=path,
        symbol=symbol,
        start_line=1,
        end_line=3,
        text=text,
        language="python",
        score=0.5,
    )


# ── _truncate ────────────────────────────────────────────────────────────


class TestTruncate:
    def test_short_text(self) -> None:
        assert _truncate("hello") == "hello"

    def test_exact_length(self) -> None:
        text = "a" * 800
        assert _truncate(text) == text

    def test_truncates_with_ellipsis(self) -> None:
        text = "a" * 1000
        result = _truncate(text, max_chars=10)
        assert result == "aaaaaaaaaa…"
        assert result.endswith("…")


# ── _parse_scores ────────────────────────────────────────────────────────


class TestParseScores:
    def test_one_per_line(self) -> None:
        response = "7\n3\n9"
        assert _parse_scores(response, 3) == [7.0, 3.0, 9.0]

    def test_trailing_commas(self) -> None:
        response = "7,\n3,\n9,"
        assert _parse_scores(response, 3) == [7.0, 3.0, 9.0]

    def test_json_array(self) -> None:
        response = '[7, 3, 9]'
        assert _parse_scores(response, 3) == [7.0, 3.0, 9.0]

    def test_bare_numbers(self) -> None:
        response = "Scores: 7, 3, 9"
        assert _parse_scores(response, 3) == [7.0, 3.0, 9.0]

    def test_fewer_numbers_than_expected(self) -> None:
        response = "7\n3"
        scores = _parse_scores(response, 3)
        assert scores == [5.0, 5.0, 5.0]  # fallback uniform

    def test_empty_response(self) -> None:
        scores = _parse_scores("", 2)
        assert scores == [5.0, 5.0]


# ── rerank_chunks ────────────────────────────────────────────────────────


class TestRerankChunks:
    def test_empty_chunks(self) -> None:
        client = MagicMock()
        result = rerank_chunks(
            client=client,
            query="test",
            chunks=[],
            chat_model="test-model",
        )
        assert result == []

    def test_fewer_chunks_than_top_k(self) -> None:
        """When there are fewer chunks than top_k, all are reranked."""
        chunks = [_make_chunk(1), _make_chunk(2)]
        client = MagicMock()
        client.chat.return_value = "9\n1"  # chunk 1 gets 9, chunk 2 gets 1

        result = rerank_chunks(
            client=client,
            query="test query",
            chunks=chunks,
            chat_model="test-model",
            top_k=10,
            final_k=2,
            method="llm",
        )
        assert len(result) == 2
        # chunk 1 (score 9) should be first
        assert result[0].chunk_id == 1

    def test_rerank_reorders_by_score(self) -> None:
        chunks = [_make_chunk(1), _make_chunk(2), _make_chunk(3)]
        client = MagicMock()
        # Scores: chunk 3 highest, chunk 1 lowest
        client.chat.return_value = "1\n10\n5"

        result = rerank_chunks(
            client=client,
            query="test",
            chunks=chunks,
            chat_model="test-model",
            top_k=3,
            final_k=3,
            method="llm",
        )
        assert [c.chunk_id for c in result] == [2, 3, 1]

    def test_rerank_keeps_remaining_beyond_top_k(self) -> None:
        """Chunks beyond top_k are appended at original order after reranked ones."""
        chunks = [
            _make_chunk(1, text="alpha"),
            _make_chunk(2, text="beta"),
            _make_chunk(3, text="gamma"),
            _make_chunk(4, text="delta"),
        ]
        client = MagicMock()
        # Only top 2 get reranked; scores: chunk2 > chunk1
        client.chat.return_value = "5\n8"

        result = rerank_chunks(
            client=client,
            query="test",
            chunks=chunks,
            chat_model="test-model",
            top_k=2,
            final_k=2,
            method="llm",
        )
        assert len(result) == 2 + 2  # final_k + remaining beyond top_k
        # After reranking, chunk 2 (score 8) is first, then chunk 1 (score 5)
        # Then chunks 3, 4 are appended in original order
        assert [c.chunk_id for c in result] == [2, 1, 3, 4]

    def test_final_k_trims_reranked_results(self) -> None:
        chunks = [_make_chunk(1), _make_chunk(2), _make_chunk(3)]
        client = MagicMock()
        client.chat.return_value = "1\n10\n5"

        result = rerank_chunks(
            client=client,
            query="test",
            chunks=chunks,
            chat_model="test-model",
            top_k=3,
            final_k=1,
            method="llm",
        )
        # Only the top reranked chunk (chunk 2 with score 10) should remain
        assert len(result) == 1
        assert result[0].chunk_id == 2

    def test_rerank_handles_chat_failure(self) -> None:
        """When the LLM call fails, uniform mid scores are used."""
        chunks = [_make_chunk(1), _make_chunk(2)]
        client = MagicMock()
        client.chat.side_effect = RuntimeError("LLM down")

        # Should not crash; falls back to original order with uniform scores
        result = rerank_chunks(
            client=client,
            query="test",
            chunks=chunks,
            chat_model="test-model",
            top_k=2,
            final_k=2,
            method="llm",
        )
        # With uniform scores, order should be stable
        assert len(result) == 2
        assert result[0].chunk_id == 1
        assert result[1].chunk_id == 2
