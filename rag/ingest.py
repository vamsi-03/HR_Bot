import logging
import uuid
from pathlib import Path
from typing import Dict, List

import docx
from pypdf import PdfReader

from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".pdf", ".docx"}


def extract_text(file_path: Path) -> List[str]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(file_path)
    if suffix == ".docx":
        return _extract_docx(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _extract_pdf(file_path: Path) -> List[str]:
    reader = PdfReader(str(file_path))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if not text.strip():
            continue
        # Prefer paragraph-level splits to keep chunks tighter
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if paragraphs:
            pages.extend(paragraphs)
        else:
            pages.append(text.strip())
    return pages


def _extract_docx(file_path: Path) -> List[str]:
    document = docx.Document(str(file_path))
    paragraphs = [para.text for para in document.paragraphs if para.text.strip()]
    return paragraphs


def chunk_text(text: str, max_words: int = 220, overlap: int = 30) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    step = max(max_words - overlap, 1)
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += step
    return chunks


def ingest_file(file_path: Path, store: VectorStore) -> int:
    if file_path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file: {file_path}")
    logger.info("Ingesting %s", file_path.name)
    parts = extract_text(file_path)
    doc_id = str(uuid.uuid4())
    total_chunks = 0
    for idx, part in enumerate(parts):
        chunks = chunk_text(part)
        metadatas: List[Dict] = []
        chunk_texts: List[str] = []
        for c_idx, chunk in enumerate(chunks):
            chunk_texts.append(chunk)
            metadatas.append(
                {
                    "doc_id": doc_id,
                    "source": file_path.name,
                    "page": idx + 1,
                    "chunk_id": f"{idx + 1}-{c_idx + 1}",
                    "text": chunk,
                }
            )
        store.add_texts(chunk_texts, metadatas)
        total_chunks += len(chunks)
    logger.info("Ingested %d chunks from %s", total_chunks, file_path.name)
    return total_chunks
