import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    return Settings(
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:1b"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        gemini_embed_model=os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004"),
        vector_store_path=Path(os.getenv("VECTOR_STORE_PATH", root / "store/index.faiss")),
        ingest_data_dir=Path(os.getenv("INGEST_DATA_DIR", root / "data/uploads")),
        log_path=Path(os.getenv("LOG_PATH", root / "logs/app.log")),
    )


settings = load_settings()
