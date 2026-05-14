"""Cross-encoder reranking for improving retrieval relevance.

Two modes (configurable via ``rerank.method``):

* ``"llm"`` (default) — uses the configured chat model to score relevance.
  No extra dependencies; works with any backend.
* ``"cross-encoder"`` — uses a local cross-encoder model via
  ``sentence-transformers``.  Faster but requires PyTorch.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from reporag.llm.backends import LLMBackend
from reporag.llm.prompts import RERANK_SYSTEM
from reporag.retrieval.search import RetrievedChunk

logger = logging.getLogger(__name__)

_MAX_CHARS_PER_PASSAGE = 800
_BATCH_SIZE = 5


# ══════════════════════════════════════════════════════════════════════
# LLM-based reranker
# ══════════════════════════════════════════════════════════════════════


def _truncate(text: str, max_chars: int = _MAX_CHARS_PER_PASSAGE) -> str:
    """Truncate *text* at *max_chars*, appending ``…`` if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def _parse_scores(response: str, expected: int) -> list[float]:
    """Extract ``expected`` scores from a model response.

    Tries, in order:
    1. One score per line (with optional trailing comma).
    2. JSON array ``[s1, s2, …]``.
    3. Any bare numbers found in the response.
    4. Fallback: uniform ``5.0`` scores.
    """
    # Strategy 1: one score per line
    scores: list[float] = []
    for line in response.strip().splitlines():
        line = line.strip().rstrip(",")
        try:
            scores.append(float(line))
        except ValueError:
            pass
        if len(scores) == expected:
            return scores

    # Strategy 2: JSON array
    scores = []
    response_stripped = response.strip()
    if response_stripped.startswith("[") and response_stripped.endswith("]"):
        import json

        try:
            parsed = json.loads(response_stripped)
            if isinstance(parsed, list) and len(parsed) == expected:
                return [float(v) for v in parsed]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 3: scan for bare numbers
    scores = []
    for token in re.findall(r"\b(\d+(?:\.\d+)?)\b", response):
        try:
            scores.append(float(token))
        except ValueError:
            pass
        if len(scores) == expected:
            return scores

    # Fallback: uniform middling score
    logger.warning(
        "Could not parse %d scores from LLM response; using uniform 5.0.  "
        "Response snippet: %s …",
        expected,
        response[:200],
    )
    return [5.0] * expected


def _rerank_with_llm(
    client: LLMBackend,
    query: str,
    chunks: Sequence[RetrievedChunk],
    chat_model: str,
    temperature: float | None = None,
) -> list[RetrievedChunk]:
    """Score chunks by asking the chat model to rate relevance in batches."""
    if not chunks:
        return list(chunks)

    # Prepare truncated passage strings
    passages = [
        f"[{c.path}:{c.start_line}-{c.end_line} ({c.symbol})]\n{_truncate(c.text)}"
        for c in chunks
    ]

    all_scores: list[float] = []

    for batch_start in range(0, len(passages), _BATCH_SIZE):
        batch_passages = passages[batch_start : batch_start + _BATCH_SIZE]
        passage_block = "\n\n---\n\n".join(
            f"Passage {j + 1}:\n{p}" for j, p in enumerate(batch_passages)
        )
        user_msg = f"Query: {query}\n\n{passage_block}"

        try:
            response = client.chat(
                chat_model,
                [
                    {"role": "system", "content": RERANK_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature if temperature is not None else 0.0,
            )
        except Exception as e:
            logger.warning("Reranking batch failed: %s. Using uniform scores.", e)
            response = ""

        scores = _parse_scores(response, len(batch_passages))
        all_scores.extend(scores)

    # Reorder chunks by score descending
    scored = list(zip(all_scores, chunks))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


# ══════════════════════════════════════════════════════════════════════
# Cross-encoder reranker (optional)
# ══════════════════════════════════════════════════════════════════════


def _rerank_with_cross_encoder(
    query: str,
    chunks: Sequence[RetrievedChunk],
    model_name: str,
) -> list[RetrievedChunk]:
    """Score chunks with a local cross-encoder model (requires sentence-transformers)."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.error(
            "sentence-transformers is not installed. "
            "Install with: pip install sentence-transformers"
        )
        return list(chunks)

    try:
        model = CrossEncoder(model_name)
    except Exception as e:
        logger.error("Failed to load cross-encoder model '%s': %s", model_name, e)
        return list(chunks)

    pairs = [(query, c.text) for c in chunks]
    try:
        scores = model.predict(pairs)  # type: ignore[union-attr]
    except Exception as e:
        logger.error("Cross-encoder scoring failed: %s", e)
        return list(chunks)

    scored = list(zip(scores, chunks))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════


def rerank_chunks(
    client: LLMBackend,
    query: str,
    chunks: Sequence[RetrievedChunk],
    chat_model: str,
    temperature: float | None = None,
    top_k: int = 20,
    final_k: int = 8,
    method: str = "llm",
    cross_encoder_model: str = "",
) -> list[RetrievedChunk]:
    """Re-rank *chunks* by relevance to *query*.

    Args:
        client: The LLM backend client.
        query: The original user query.
        chunks: First-stage retrieval results.
        chat_model: Chat model name (used when method="llm").
        temperature: LLM temperature for scoring.
        top_k: How many of the top first-stage results to rerank.
        final_k: How many results to keep after reranking.
        method: ``"llm"`` or ``"cross-encoder"``.
        cross_encoder_model: Model name for cross-encoder (ignored for llm).

    Returns:
        Re-ranked and trimmed list of chunks.
    """
    if not chunks:
        return list(chunks)

    # Take the top-k from the original ranking for reranking
    candidates = list(chunks[:top_k])

    if method == "cross-encoder":
        reranked = _rerank_with_cross_encoder(query, candidates, cross_encoder_model)
    else:
        reranked = _rerank_with_llm(
            client, query, candidates, chat_model, temperature=temperature
        )

    # Trim to final_k and re-append any chunks beyond top_k (at original order)
    final = reranked[:final_k]
    remaining = list(chunks[top_k:])
    return final + remaining
