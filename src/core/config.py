"""config.py: Load configuration from environment variables and secrets."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
secrets_path = project_root / ".streamlit" / "secrets.toml"
dotenv_path = project_root / ".env"


def _load_secrets() -> dict[str, str]:
    """Load secrets from the Streamlit secrets file if it exists.

    Returns:
        dict[str, str]: Secrets loaded from disk.
    """
    if not secrets_path.exists():
        return {}

    try:
        with secrets_path.open("rb") as secrets_file:
            raw_secrets = tomllib.load(secrets_file)
    except (OSError, tomllib.TOMLDecodeError):
        return {}

    return {key: str(value) for key, value in raw_secrets.items()}


def _load_dotenv() -> dict[str, str]:
    """Load key/value pairs from a local .env file when present.

    Returns:
        dict[str, str]: Parsed .env values.
    """
    if not dotenv_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        cleaned_value = value.strip()
        if (
            (cleaned_value.startswith('"') and cleaned_value.endswith('"'))
            or (cleaned_value.startswith("'") and cleaned_value.endswith("'"))
        ):
            cleaned_value = cleaned_value[1:-1]
        if key:
            values[key] = cleaned_value
    return values


def _read_setting(name: str, default: str, dotenv: dict[str, str], secrets: dict[str, str]) -> str:
    """Read a setting from env vars with secrets fallback.

    Args:
        name (str): Environment variable name.
        default (str): Default value if nothing is set.
        dotenv (dict[str, str]): Values loaded from .env.
        secrets (dict[str, str]): Secrets map loaded from disk.

    Returns:
        str: Resolved setting value.
    """
    return os.getenv(name, dotenv.get(name, secrets.get(name, default)))


def _read_bool_setting(name: str, default: bool, dotenv: dict[str, str], secrets: dict[str, str]) -> bool:
    """Read a boolean setting from env vars with secrets fallback.

    Args:
        name (str): Environment variable name.
        default (bool): Default value if nothing is set.
        secrets (dict[str, str]): Secrets map loaded from disk.

    Returns:
        bool: Parsed boolean setting.
    """
    raw_value = _read_setting(name, "true" if default else "false", dotenv, secrets)
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Configuration values for Study Pal."""

    app_title: str
    app_subtitle: str
    chat_model: str
    embedding_model: str
    embedding_dimensions: int
    openrouter_base_url: str
    openrouter_api_key: str
    openrouter_key_encryption_secret: str
    openrouter_allow_global_fallback: bool
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
    pinecone_host: str
    pinecone_index_name: str
    supabase_url: str
    supabase_public_key: str
    supabase_redirect_url: str
    langfuse_secret_key: str
    langfuse_public_key: str
    langfuse_base_url: str
    langfuse_prompt_answer: str
    langfuse_prompt_structured_answer: str
    langfuse_prompt_document_metadata: str
    langfuse_prompt_quiz: str
    langfuse_prompt_reteach: str
    langfuse_prompt_study_plan: str
    langfuse_prompt_follow_up: str
    langfuse_prompt_version: str
    rerank_model: str
    rerank_candidates: int

    @classmethod
    def load(cls) -> "Settings":
        """Load all configuration values from secrets and environment variables.

        Returns:
            Settings: Loaded configuration object.
        """
        secrets = _load_secrets()
        dotenv = _load_dotenv()
        return cls(
            app_title="Study Pal",
            app_subtitle="Ask questions about your uploaded notes and get cited answers.",
            chat_model=_read_setting("OPENROUTER_CHAT_MODEL", "openai/gpt-4.1-mini", dotenv, secrets),
            embedding_model=_read_setting("OPENROUTER_EMBEDDING_MODEL", "text-embedding-3-small", dotenv, secrets),
            embedding_dimensions=int(_read_setting("EMBEDDING_DIMENSIONS", "512", dotenv, secrets)),
            openrouter_base_url=_read_setting("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1", dotenv, secrets),
            openrouter_api_key=_read_setting("OPENROUTER_API_KEY", "", dotenv, secrets),
            openrouter_key_encryption_secret=_read_setting("OPENROUTER_KEY_ENCRYPTION_SECRET", "", dotenv, secrets),
            openrouter_allow_global_fallback=_read_bool_setting(
                "OPENROUTER_ALLOW_GLOBAL_FALLBACK",
                False,
                dotenv,
                secrets,
            ),
            chunk_size=int(_read_setting("STUDYPAL_CHUNK_SIZE", "900", dotenv, secrets)),
            chunk_overlap=int(_read_setting("STUDYPAL_CHUNK_OVERLAP", "150", dotenv, secrets)),
            top_k=int(_read_setting("STUDYPAL_TOP_K", "4", dotenv, secrets)),
            quiz_questions_per_round=int(_read_setting("STUDYPAL_QUIZ_QUESTIONS", "3", dotenv, secrets)),
            max_quiz_rounds=int(_read_setting("STUDYPAL_MAX_QUIZ_ROUNDS", "3", dotenv, secrets)),
            max_chat_tokens=int(_read_setting("STUDYPAL_MAX_CHAT_TOKENS", "600", dotenv, secrets)),
            allowed_file_types=("pdf", "txt", "md"),
            max_file_size_mb=int(_read_setting("STUDYPAL_MAX_FILE_SIZE_MB", "150", dotenv, secrets)),
            database_url=_read_setting("DATABASE_URL", "", dotenv, secrets),
            pinecone_api_key=_read_setting("PINECONE_API_KEY", "", dotenv, secrets),
            pinecone_host=_read_setting("PINECONE_HOST", "", dotenv, secrets),
            pinecone_index_name=_read_setting("PINECONE_INDEX_NAME", "study-pal", dotenv, secrets),
            supabase_url=_read_setting("SUPABASE_URL", "", dotenv, secrets),
            supabase_public_key=_read_setting("SUPABASE_PUBLIC_KEY", "", dotenv, secrets),
            supabase_redirect_url=_read_setting("SUPABASE_REDIRECT_URL", "", dotenv, secrets),
            langfuse_secret_key=_read_setting("LANGFUSE_SECRET_KEY", "", dotenv, secrets),
            langfuse_public_key=_read_setting("LANGFUSE_PUBLIC_KEY", "", dotenv, secrets),
            langfuse_base_url=_read_setting("LANGFUSE_BASE_URL", "https://cloud.langfuse.com", dotenv, secrets),
            langfuse_prompt_answer=_read_setting("LANGFUSE_PROMPT_ANSWER", "study_pal_answer", dotenv, secrets),
            langfuse_prompt_structured_answer=_read_setting(
                "LANGFUSE_PROMPT_STRUCTURED_ANSWER",
                "study_pal_structured_answer",
                dotenv,
                secrets,
            ),
            langfuse_prompt_document_metadata=_read_setting(
                "LANGFUSE_PROMPT_DOCUMENT_METADATA",
                "study_pal_document_metadata",
                dotenv,
                secrets,
            ),
            langfuse_prompt_quiz=_read_setting("LANGFUSE_PROMPT_QUIZ", "study_pal_quiz", dotenv, secrets),
            langfuse_prompt_reteach=_read_setting("LANGFUSE_PROMPT_RETEACH", "study_pal_reteach", dotenv, secrets),
            langfuse_prompt_study_plan=_read_setting(
                "LANGFUSE_PROMPT_STUDY_PLAN",
                "study_pal_study_plan",
                dotenv,
                secrets,
            ),
            langfuse_prompt_follow_up=_read_setting(
                "LANGFUSE_PROMPT_FOLLOW_UP",
                "study_pal_follow_up",
                dotenv,
                secrets,
            ),
            langfuse_prompt_version=_read_setting("LANGFUSE_PROMPT_VERSION", "", dotenv, secrets),
            rerank_model=_read_setting("OPENROUTER_RERANK_MODEL", "", dotenv, secrets),
            rerank_candidates=int(_read_setting("OPENROUTER_RERANK_CANDIDATES", "12", dotenv, secrets)),
        )


settings = Settings.load()
