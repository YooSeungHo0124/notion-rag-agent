"""Application settings: secrets from .env, tunables from config/settings.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLMConfig(BaseModel):
    timeout_s: int = 180
    max_retries: int = 2


class ChunkingConfig(BaseModel):
    max_chars: int = 1200
    overlap_chars: int = 150
    min_chars: int = 200  # merge smaller chunks into a neighbour (0 = off)


class EmbeddingConfig(BaseModel):
    dense_model: str = "Qwen/Qwen3-Embedding-4B"
    sparse_model: str = "BAAI/bge-m3"
    device: str = "cuda"
    batch_size: int = 16


class RetrievalConfig(BaseModel):
    collection: str = "notion_chunks"
    candidate_k: int = 50
    final_k: int = 5
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # LoRA adapter from training/rerank_ft.py (path relative to the project, "" = base model)
    reranker_adapter: str = ""
    min_rerank_score: float = 0.2


class FileConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    notion_token: str = ""
    notion_root_ids: str = ""
    # Demo ERP/CRM (Odoo). Empty url -> ERP tools disabled.
    odoo_url: str = ""
    odoo_db: str = ""
    odoo_user: str = ""
    odoo_password: str = ""
    company_docs_dir: str = "scenario/docs"

    llm_provider: Literal["claude_cli", "anthropic_api"] = "claude_cli"
    llm_model: str = "sonnet"
    anthropic_api_key: str = ""

    qdrant_url: str = ""
    qdrant_path: Path = PROJECT_ROOT / "data" / "qdrant"
    state_db: Path = PROJECT_ROOT / "data" / "state.db"

    file: FileConfig = FileConfig()

    @property
    def root_ids(self) -> list[str]:
        return [s.strip() for s in self.notion_root_ids.split(",") if s.strip()]


@lru_cache
def get_settings(config_path: Path | None = None) -> Settings:
    path = config_path or PROJECT_ROOT / "config" / "settings.yaml"
    file_cfg = FileConfig()
    if path.exists():
        file_cfg = FileConfig.model_validate(yaml.safe_load(path.read_text()) or {})
    return Settings(file=file_cfg)
