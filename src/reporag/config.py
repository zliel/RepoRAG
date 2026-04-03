from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from reporag.llm.backends import BackendType


@dataclass
class Config:
    backend: BackendType = "ollama"
    embed_model: str = "nomic-embed-text-v2-moe"
    chat_model: str = "deepseek-coder-v2:16b-lite-instruct-q3_K_M"
    db: str = "index.sqlite"
    base_url: str | None = None
    api_key: str | None = None
    embed_batch: int = 32
    temperature: float = 0.2

    @property
    def ollama_base(self) -> str | None:
        if self.backend == "ollama":
            return self.base_url
        return None

    DEFAULT_CONFIG = """\
# RepoRAG Configuration
# https://github.com/your-repo/reporag

[reporag]
# LLM backend: ollama, vllm, llamacpp, lmstudio, lmstudio-local
backend = "ollama"

# Ollama base URL (leave empty for default http://127.0.0.1:11434)
# base_url = "http://127.0.0.1:11434"

# Embedding model for vector search
embed_model = "nomic-embed-text-v2-moe"

# Chat model for RAG and generation
chat_model = "deepseek-coder-v2:16b-lite-instruct-q3_K_M"

# Temperature for chat model (0.0-1.0, lower = more deterministic)
temperature = 0.2

# Database path for the embedding index
db = "index.sqlite"

# Batch size for embedding generation
embed_batch = 32

# API key (for OpenAI-compatible backends like vllm, llamacpp, lmstudio)
# api_key = ""
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
                import logging

                logging.getLogger(__name__).info("Creating default config at %s", default_path)
                default_path.write_text(cls.DEFAULT_CONFIG, encoding="utf-8")

        return config

    def _load(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            import logging

            logging.getLogger(__name__).warning("Could not read config %s: %s", path, e)
            return

        try:
            data = tomllib.loads(text)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning("Could not parse config %s: %s", path, e)
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
