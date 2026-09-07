"""Application settings for Local RAG Service.

All runtime configuration is read from environment variables and, when present,
from the project-level ``.env`` file. Import and use the singleton ``settings``
instead of importing individual constants.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Validated runtime configuration for the whole application."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # FastAPI / application
    app_title: str = "Local RAG Service"
    log_level: str = "INFO"
    frontend_dir: Path = PROJECT_ROOT / "frontend"
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    # Ollama / LLM
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_temperature: float = 0.0
    ollama_status_timeout_seconds: float = Field(default=5.0, gt=0)

    # Embeddings
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_device: str = "cpu"
    normalize_embeddings: bool = True

    # Chroma / storage
    chroma_persist_path: Path = PROJECT_ROOT / "chroma_db"
    chroma_collection_name: str = "local_rag"
    pdf_folder: Path = PROJECT_ROOT / "data" / "pdfs"
    index_schema_version: str = "1"

    # RAG
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k_results: int = Field(default=4, gt=0)
    relevance_score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


settings = Settings()
