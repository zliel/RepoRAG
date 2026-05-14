"""Tests for embedding model validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
import typer

from reporag.cli.main import validate_embed_model


class TestValidateEmbedModel:
    def test_valid_model_succeeds(self) -> None:
        """A properly responding model should not raise."""
        client = MagicMock()
        client.embed.return_value = [[0.1, 0.2, 0.3]]

        # Should not raise
        validate_embed_model(client, "test-model")

        client.embed.assert_called_once_with(["test"], "test-model")

    def test_http_error_raises_exit(self) -> None:
        """If the backend is unreachable, should exit with error."""
        client = MagicMock()
        client.embed.side_effect = httpx.ConnectError("connection refused")

        with pytest.raises(typer.Exit):
            validate_embed_model(client, "bad-model")

    def test_empty_result_raises_exit(self) -> None:
        """Empty embedding results should trigger an error."""
        client = MagicMock()
        client.embed.return_value = [[]]

        with pytest.raises(typer.Exit):
            validate_embed_model(client, "empty-model")

    def test_malformed_result_raises_exit(self) -> None:
        """Non-list results should trigger an error."""
        client = MagicMock()
        client.embed.return_value = [None]

        with pytest.raises(typer.Exit):
            validate_embed_model(client, "bad-model")

    def test_http_status_error_404_raises_exit(self) -> None:
        """A 404 on the embed endpoint should exit with a clear message."""
        client = MagicMock()
        client.embed.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )

        with pytest.raises(typer.Exit):
            validate_embed_model(client, "missing-model")
