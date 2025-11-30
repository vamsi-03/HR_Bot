import logging
from dataclasses import dataclass
from typing import List, Optional

import google.generativeai as genai
import numpy as np
import ollama
from ollama import Client as OllamaClient

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ProviderStatus:
    name: str
    available: bool
    detail: str = ""


class EmbeddingProvider:
    def embed(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover - interface
        raise NotImplementedError

    def available(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class OllamaEmbeddings(EmbeddingProvider):
    def __init__(self) -> None:
        self.client = OllamaClient(host=settings.ollama_host)
        self.model = settings.ollama_embed_model

    def embed(self, texts: List[str]) -> List[List[float]]:
        logger.info("Embedding with Ollama model %s", self.model)
        vectors: List[List[float]] = []
        for text in texts:
            resp = self.client.embeddings(model=self.model, prompt=text)
            vectors.append(resp["embedding"])
        return vectors

    def available(self) -> bool:
        try:
            _ = self.client.list()
            return True
        except Exception:  # pylint: disable=broad-except
            return False

    def status(self) -> ProviderStatus:
        try:
            _ = self.client.list()
            return ProviderStatus("ollama-embed", True, "reachable")
        except Exception as exc:  # pylint: disable=broad-except
            return ProviderStatus("ollama-embed", False, str(exc))


class GeminiEmbeddings(EmbeddingProvider):
    def __init__(self) -> None:
        self.api_key: Optional[str] = settings.gemini_api_key
        self.model = settings.gemini_embed_model
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise RuntimeError("Gemini API key not configured")
        logger.info("Embedding with Gemini model %s", self.model)
        vectors: List[List[float]] = []
        for text in texts:
            result = genai.embed_content(model=self.model, content=text)
            vectors.append(result["embedding"])
        return vectors

    def available(self) -> bool:
        return bool(self.api_key)

    def status(self) -> ProviderStatus:
        if not self.api_key:
            return ProviderStatus("gemini-embed", False, "API key missing")
        return ProviderStatus("gemini-embed", True, "configured")


class EmbeddingRouter:
    def __init__(self) -> None:
        self.providers: List[EmbeddingProvider] = [OllamaEmbeddings(), GeminiEmbeddings()]

    def provider_statuses(self) -> List[ProviderStatus]:
        statuses: List[ProviderStatus] = []
        for provider in self.providers:
            if hasattr(provider, "status"):
                statuses.append(provider.status())  # type: ignore
            else:  # pragma: no cover - fallback
                statuses.append(ProviderStatus(provider.__class__.__name__, provider.available()))
        return statuses

    def embed(self, texts: List[str]) -> np.ndarray:
        last_error: Optional[str] = None
        for provider in self.providers:
            if not provider.available():
                continue
            try:
                vectors = provider.embed(texts)
                return np.array(vectors, dtype="float32")
            except Exception as exc:  # pylint: disable=broad-except
                last_error = str(exc)
                logger.exception("Embedding provider failed")
        raise RuntimeError(f"No embedding providers available. Last error: {last_error}")
