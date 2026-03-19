from __future__ import annotations

import os

import openlit
from langfuse import get_client

from src.config import settings


def initialize_observability() -> bool:
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return False

    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_base_url

    try:
        langfuse = get_client()
        if not langfuse.auth_check():
            return False
        openlit.init(disable_batch=True)
    except Exception:
        return False
    return True
