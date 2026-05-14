"""Tests for exponential-backoff retry logic."""

from __future__ import annotations

import httpx
import pytest

from reporag.llm.retry import is_retryable, with_retry


class TestIsRetryable:
    def test_timeout_is_retryable(self) -> None:
        assert is_retryable(httpx.TimeoutException("timed out"))

    def test_connect_error_is_retryable(self) -> None:
        assert is_retryable(httpx.ConnectError("connection refused"))

    def test_remote_protocol_error_is_retryable(self) -> None:
        assert is_retryable(httpx.RemoteProtocolError("remote hung up"))

    def test_network_error_is_retryable(self) -> None:
        assert is_retryable(httpx.NetworkError("network unreachable"))

    def test_500_is_retryable(self) -> None:
        request = httpx.Request("GET", "http://example.com")
        response = httpx.Response(503, request=request)
        exc = httpx.HTTPStatusError("503 Server Error", request=request, response=response)
        assert is_retryable(exc)

    def test_400_is_not_retryable(self) -> None:
        request = httpx.Request("GET", "http://example.com")
        response = httpx.Response(404, request=request)
        exc = httpx.HTTPStatusError("404 Not Found", request=request, response=response)
        assert not is_retryable(exc)

    def test_401_is_not_retryable(self) -> None:
        request = httpx.Request("GET", "http://example.com")
        response = httpx.Response(401, request=request)
        exc = httpx.HTTPStatusError("401 Unauthorized", request=request, response=response)
        assert not is_retryable(exc)

    def test_value_error_is_not_retryable(self) -> None:
        assert not is_retryable(ValueError("bad data"))


class TestWithRetry:
    def test_success_first_attempt(self) -> None:
        """Function succeeds immediately — no retries needed."""
        call_count = 0

        def ok() -> str:
            nonlocal call_count
            call_count += 1
            return "done"

        result = with_retry(ok, max_retries=3)
        assert result == "done"
        assert call_count == 1

    def test_retry_on_timeout_then_succeed(self) -> None:
        """Fail twice with timeout, succeed on third attempt."""
        call_count = 0

        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.TimeoutException("timeout", request=None)  # type: ignore[arg-type]
            return "ok"

        result = with_retry(flaky, max_retries=3, backoff_factor=0.01)
        assert result == "ok"
        assert call_count == 3

    def test_all_retries_exhausted(self) -> None:
        """Always raises — should exhaust all retries and raise last error."""
        call_count = 0

        def always_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise httpx.TimeoutException("always timeout", request=None)  # type: ignore[arg-type]

        with pytest.raises(httpx.TimeoutException):
            with_retry(always_fail, max_retries=2, backoff_factor=0.01)
        assert call_count == 3  # initial + 2 retries

    def test_non_retryable_raises_immediately(self) -> None:
        """4xx error raises immediately without retrying."""
        request = httpx.Request("GET", "http://example.com")
        response = httpx.Response(404, request=request)

        def bad_request() -> str:
            raise httpx.HTTPStatusError("404", request=request, response=response)

        with pytest.raises(httpx.HTTPStatusError):
            with_retry(bad_request, max_retries=3)

    def test_unknown_exception_raises_immediately(self) -> None:
        """Non-HTTP exceptions are not retried."""

        def crash() -> str:
            raise ValueError("unexpected")

        with pytest.raises(ValueError):
            with_retry(crash, max_retries=3)
