import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from llm.embeddings import EmbeddingRouter

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.meta_path = index_path.with_suffix(".meta.json")
        self.embedder = EmbeddingRouter()
        self.metadata: List[Dict] = []
        self.index: faiss.IndexFlatIP | None = None
        self._load()
        # Best-effort refresh in case cache persisted an old embedder without statuses
        if not hasattr(self.embedder, "provider_statuses"):
            self.embedder = EmbeddingRouter()

    def _load(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            logger.info("Loading vector store from %s", self.index_path)
            self.index = faiss.read_index(str(self.index_path))
            with self.meta_path.open() as f:
                self.metadata = json.load(f)
        else:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.index = None
            self.metadata = []

    def _save(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("w") as f:
            json.dump(self.metadata, f)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        return vectors / norms

    def add_texts(self, texts: List[str], metadatas: List[Dict]) -> int:
        vectors = self.embedder.embed(texts)
        vectors = self._normalize(vectors)
        if self.index is None:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.metadata.extend(metadatas)
        self._save()
        return len(texts)

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Dict, float]]:
        if self.index is None or not self.metadata:
            return []
        query_vec = self.embedder.embed([query])
        query_vec = self._normalize(query_vec)
        scores, idxs = self.index.search(query_vec, top_k)
        hits: List[Tuple[Dict, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx >= len(self.metadata) or score <= 0:
                continue
            hits.append((self.metadata[idx], float(score)))
        return hits
