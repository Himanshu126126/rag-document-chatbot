"""Central configuration. Values can be overridden via environment variables."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_secret(name: str, default: str = "") -> str:
    """Read from env vars first, then Streamlit secrets (used on Streamlit Cloud)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return str(st.secrets.get(name, default))
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = _get_secret("LLM_MODEL", "openai/gpt-oss-20b")
    groq_api_key: str = _get_secret("GROQ_API_KEY", "")
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    fetch_k: int = 12


settings = Settings()
