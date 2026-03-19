from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
secrets_path = project_root / ".streamlit" / "secrets.toml"


def _load_secrets() -> dict[str, str]:
    if not secrets_path.exists():
        return {}

    with secrets_path.open("rb") as secrets_file:
        raw_secrets = tomllib.load(secrets_file)

    return {key: str(value) for key, value in raw_secrets.items()}


def _read_setting(name: str, default: str, secrets: dict[str, str]) -> str:
    return os.getenv(name, secrets.get(name, default))


@dataclass(frozen=True)
class Settings:
    app_title: str
    app_subtitle: str
    chat_model: str
    embedding_model: str
    embedding_dimensions: int
    openrouter_base_url: str
    openrouter_api_key: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    quiz_questions_per_round: int
    max_quiz_rounds: int
    max_chat_tokens: int
    allowed_file_types: tuple[str, ...]
    max_file_size_mb: int
    database_url: str
    pinecone_api_key: str
    pinecone_index_name: str
    langfuse_secret_key: str
    langfuse_public_key: str
    langfuse_base_url: str

    @classmethod
    def load(cls) -> Settings:
        secrets = _load_secrets()
        return cls(
            app_title="Study Pal",
            app_subtitle="Ask questions about your uploaded notes and get cited answers.",
            chat_model=_read_setting("OPENROUTER_CHAT_MODEL", "openai/gpt-4.1-mini", secrets),
            embedding_model=_read_setting("OPENROUTER_EMBEDDING_MODEL", "text-embedding-3-small", secrets),
            embedding_dimensions=int(_read_setting("EMBEDDING_DIMENSIONS", "512", secrets)),
            openrouter_base_url=_read_setting("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1", secrets),
            openrouter_api_key=_read_setting("OPENROUTER_API_KEY", "", secrets),
            chunk_size=int(_read_setting("STUDYPAL_CHUNK_SIZE", "900", secrets)),
            chunk_overlap=int(_read_setting("STUDYPAL_CHUNK_OVERLAP", "150", secrets)),
            top_k=int(_read_setting("STUDYPAL_TOP_K", "4", secrets)),
            quiz_questions_per_round=int(_read_setting("STUDYPAL_QUIZ_QUESTIONS", "3", secrets)),
            max_quiz_rounds=int(_read_setting("STUDYPAL_MAX_QUIZ_ROUNDS", "3", secrets)),
            max_chat_tokens=int(_read_setting("STUDYPAL_MAX_CHAT_TOKENS", "600", secrets)),
            allowed_file_types=("pdf", "txt", "md"),
            max_file_size_mb=int(_read_setting("STUDYPAL_MAX_FILE_SIZE_MB", "150", secrets)),
            database_url=_read_setting("DATABASE_URL", "", secrets),
            pinecone_api_key=_read_setting("PINECONE_API_KEY", "", secrets),
            pinecone_index_name=_read_setting("PINECONE_INDEX_NAME", "study-pal", secrets),
            langfuse_secret_key=_read_setting("LANGFUSE_SECRET_KEY", "", secrets),
            langfuse_public_key=_read_setting("LANGFUSE_PUBLIC_KEY", "", secrets),
            langfuse_base_url=_read_setting("LANGFUSE_BASE_URL", "https://cloud.langfuse.com", secrets),
        )


settings = Settings.load()
