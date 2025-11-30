<<<<<<< HEAD
# HR_Bot
A Streamlit-based HR assistant that answers employee policy questions using retrieval-augmented generation (RAG). Upload PDFs/DOCX (handbooks, PTO policies, onboarding docs), and the app ingests, chunks, embeds (Ollama or Gemini), and stores them in a local FAISS index.
=======
# HR RAG Bot (Streamlit)

Local HR assistant using RAG with Ollama (primary) and Gemini (fallback). Supports PDF/DOCX ingestion, grounded answers with citations, and a simple eval harness.

## Quick start
1. Create `.env` from `.env.example` and set any Gemini keys or custom models.
2. Install deps: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
4. In the **Ingest Documents** page, upload PDF/DOCX files. The FAISS store persists to `store/index.faiss`.
5. Use the chat to ask HR questions. If no supporting context is found, the bot returns “No information found.”

## Project layout
- `app.py` — main chat UI with citations and grounding guardrail.
- `pages/ingest.py` — upload and ingest documents.
- `pages/history.py` — view/clear chat history and store info.
- `rag/` — ingestion, chunking, vector store, retrieval.
- `llm/` — LLM and embedding routers with Ollama/Gemini fallback.
- `services/resources.py` — shared cached instances for Streamlit pages.
- `tests/eval.py` — tiny eval harness reading `tests/sample_eval.csv`.

## Notes
- Defaults to Ollama for generation/embeddings. Gemini is used if configured or as fallback.
- Answers are strictly grounded; no retrieval results -> direct “No information found.”
- Logging writes to `logs/app.log`.
- If embeddings fail during ingestion, ensure Ollama is running and the embedding model is available (e.g., `ollama pull nomic-embed-text`), or set a Gemini API key to use that embedder instead.
>>>>>>> e5856f8 (Initial Commit)
