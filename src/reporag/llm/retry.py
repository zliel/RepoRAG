"""Exponential backoff retry for LLM backend HTTP calls."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.NetworkError,
    httpx.PoolTimeout,
)


def is_retryable(exc: Exception) -> bool:
    """Return *True* if *exc* represents a transient error worth retrying."""
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    return False


def with_retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
) -> T:
    """Call *fn* with exponential-backoff retry on retryable errors.

    The delay sequence is ``backoff_factor^0``, ``backoff_factor^1``, …  (i.e.
    1 s, 2 s, 4 s …  when *backoff_factor* is 2.0), capped at *max_delay*.

    Non-retryable exceptions (4xx client errors, assertion errors, …) are
    re-raised immediately.
    """
    last_exc: Exception | None = None
    delay = 1.0

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not is_retryable(exc):
                raise
            if attempt >= max_retries:
                logger.error(
                    "All %d retries exhausted for %s: %s",
                    max_retries,
                    getattr(fn, "__name__", str(fn)),
                    exc,
                )
                break
            logger.warning(
                "Retryable error (attempt %d/%d): %s.  Waiting %.1f s …",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    # Only reachable when all retries exhausted and last_exc is set.
    assert last_exc is not None
    raise last_exc
