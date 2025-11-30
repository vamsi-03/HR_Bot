from typing import Optional

import streamlit as st

from config.settings import settings
from llm.client import LLMRouter
from rag.vector_store import VectorStore

_store: Optional[VectorStore] = None
_llm: Optional[LLMRouter] = None


def _build_store() -> VectorStore:
    return VectorStore(settings.vector_store_path)


def _build_llm() -> LLMRouter:
    return LLMRouter()


if hasattr(st, "cache_resource"):
    @st.cache_resource
    def get_store() -> VectorStore:
        return _build_store()

    @st.cache_resource
    def get_llm() -> LLMRouter:
        return _build_llm()
else:  # pragma: no cover - fallback for non-Streamlit usage
    def get_store() -> VectorStore:
        global _store
        if _store is None:
            _store = _build_store()
        return _store

    def get_llm() -> LLMRouter:
        global _llm
        if _llm is None:
            _llm = _build_llm()
        return _llm
