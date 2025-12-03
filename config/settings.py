import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv


@dataclass
class Settings:
    ollama_host: str
    ollama_model: str
    ollama_embed_model: str
    gemini_api_key: Optional[str]
    gemini_model: str
    gemini_embed_model: str
    vector_store_path: Path
    ingest_data_dir: Path
    log_path: Path


def load_settings() -> Settings:
    load_dotenv()
    root = Path(__file__).resolve().parent.parent

    def secret_or_env(key: str, default: Optional[str] = None) -> Optional[str]:
        # Streamlit secrets take precedence if present, otherwise fall back to env/default.
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
        return os.getenv(key, default)

    return Settings(
        ollama_host=secret_or_env("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=secret_or_env("OLLAMA_MODEL", "llama3"),
        ollama_embed_model=secret_or_env("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        gemini_api_key=secret_or_env("GEMINI_API_KEY"),
        gemini_model=secret_or_env("GEMINI_MODEL", "gemini-1.5-flash"),
        gemini_embed_model=secret_or_env("GEMINI_EMBED_MODEL", "text-embedding-004"),
        vector_store_path=Path(secret_or_env("VECTOR_STORE_PATH", root / "store/index.faiss")),
        ingest_data_dir=Path(secret_or_env("INGEST_DATA_DIR", root / "data/uploads")),
        log_path=Path(secret_or_env("LOG_PATH", root / "logs/app.log")),
    )


settings = load_settings()
