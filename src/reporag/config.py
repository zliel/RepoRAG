from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from reporag.constants import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_BATCH,
    DEFAULT_EMBED_MODEL,
    DEFAULT_TEMPERATURE,
)
from reporag.llm.backends import BackendType

logger = logging.getLogger(__name__)


@dataclass
class RerankConfig:
    enabled: bool = False
    top_k: int = 20
    final_k: int = 8
    method: str = "llm"  # "llm" or "cross-encoder"
    model: str = ""


@dataclass
class Config:
    backend: BackendType = "ollama"
    embed_model: str = DEFAULT_EMBED_MODEL
    chat_model: str = DEFAULT_CHAT_MODEL
    db: str = "index.sqlite"
    base_url: str | None = None
    api_key: str | None = None
    embed_batch: int = DEFAULT_EMBED_BATCH
    temperature: float = DEFAULT_TEMPERATURE
    timeout: float | None = None
    exclude_patterns: list[str] | None = None
    max_retries: int = 3
    backoff_factor: float = 2.0
    rerank: RerankConfig = field(default_factory=RerankConfig)

    @property
    def ollama_base(self) -> str | None:
        if self.backend == "ollama":
            return self.base_url
        return None

    DEFAULT_CONFIG = f"""\
# RepoRAG Configuration
# https://github.com/your-repo/reporag

[reporag]
# LLM backend: ollama, vllm, llamacpp, lmstudio, lmstudio-local
backend = "ollama"

# Ollama base URL (leave empty for default http://127.0.0.1:11434)
# base_url = "http://127.0.0.1:11434"

# Embedding model for vector search
embed_model = "{DEFAULT_EMBED_MODEL}"

# Chat model for RAG and generation
chat_model = "{DEFAULT_CHAT_MODEL}"

# Temperature for chat model (0.0-1.0, lower = more deterministic)
temperature = {DEFAULT_TEMPERATURE}

# Database path for the embedding index
db = "index.sqlite"

# Batch size for embedding generation
embed_batch = {DEFAULT_EMBED_BATCH}

# API key (for OpenAI-compatible backends like vllm, llamacpp, lmstudio)
# api_key = ""

# Additional glob exclusion patterns (comma-separated in TOML array)
# exclude_patterns = ["tests/", "venv/", "*.pyc"]

# HTTP request retry settings
max_retries = 3
backoff_factor = 2.0

# Cross-encoder reranking (opt-in; adds latency but improves relevance)
[reporag.rerank]
enabled = false
top_k = 20
final_k = 8
method = "llm"
# model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
"""

    @classmethod
    def from_file(cls, path: Path | None = None) -> Config:
        config = cls()
        if path is not None:
            config._load(path)
            return config

        found = None
        for candidate in _config_locations():
            if candidate.exists():
                config._load(candidate)
                found = candidate
                break

        if found is None:
            default_path = Path.home() / ".config" / "reporag" / "config.toml"
            default_path.parent.mkdir(parents=True, exist_ok=True)
            if not default_path.exists():
                logger.info("Creating default config at %s", default_path)
                default_path.write_text(cls.DEFAULT_CONFIG, encoding="utf-8")

        return config

    def _load(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("Could not read config %s: %s", path, e)
            return

        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            logger.warning("Could not parse config %s: %s", path, e)
            return

        reporag = data.get("reporag", {})
        if backend := reporag.get("backend"):
            self.backend = backend  # type: ignore[assignment]
        if embed_model := reporag.get("embed_model"):
            self.embed_model = embed_model
        if chat_model := reporag.get("chat_model"):
            self.chat_model = chat_model
        if db := reporag.get("db"):
            self.db = db
        if base_url := reporag.get("base_url"):
            self.base_url = base_url
        if api_key := reporag.get("api_key"):
            self.api_key = api_key
        if embed_batch := reporag.get("embed_batch"):
            self.embed_batch = embed_batch
        if temperature := reporag.get("temperature"):
            self.temperature = temperature
        if exclude_patterns := reporag.get("exclude_patterns"):
            self.exclude_patterns = exclude_patterns
        if timeout := reporag.get("timeout"):
            self.timeout = timeout
        if max_retries := reporag.get("max_retries"):
            self.max_retries = max_retries
        if backoff_factor := reporag.get("backoff_factor"):
            self.backoff_factor = backoff_factor

        # Load rerank config from nested section
        if rerank_raw := reporag.get("rerank"):
            if isinstance(rerank_raw, dict):
                if "enabled" in rerank_raw:
                    self.rerank.enabled = bool(rerank_raw["enabled"])
                if "top_k" in rerank_raw:
                    self.rerank.top_k = int(rerank_raw["top_k"])
                if "final_k" in rerank_raw:
                    self.rerank.final_k = int(rerank_raw["final_k"])
                if "method" in rerank_raw:
                    self.rerank.method = str(rerank_raw["method"])
                if "model" in rerank_raw:
                    self.rerank.model = str(rerank_raw["model"])


def _config_locations() -> list[Path]:
    locations = []
    if cwd := os.getcwd():
        locations.append(Path(cwd) / ".reporag.toml")
    if home := Path.home():
        locations.append(home / ".config" / "reporag" / "config.toml")
        locations.append(home / ".reporag.toml")
    if xdg_config := os.environ.get("XDG_CONFIG_HOME"):
        locations.append(Path(xdg_config) / "reporag" / "config.toml")
    return locations
