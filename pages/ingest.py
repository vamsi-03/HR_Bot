import streamlit as st

from config.settings import settings
from rag.ingest import SUPPORTED_EXTS, ingest_file
from services.resources import get_store

store = get_store()

st.set_page_config(page_title="Ingest Documents")
st.title("Ingest HR Documents")
st.caption("Upload PDF or DOCX files to make them searchable. Files are stored locally.")

st.subheader("Embedding providers")
get_statuses = getattr(store.embedder, "provider_statuses", None)
statuses = get_statuses() if callable(get_statuses) else []
if statuses:
    for status in statuses:
        icon = "✅" if status.available else "⚠️"
        st.write(f"{icon} {status.name}: {status.detail or 'ready'}")
else:
    st.write("⚠️ Provider status unavailable (restart may be needed after code updates).")

uploaded_files = st.file_uploader(
    "Select files", type=[ext.replace(".", "") for ext in SUPPORTED_EXTS], accept_multiple_files=True
)

if uploaded_files and st.button("Ingest now"):
    settings.ingest_data_dir.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    try:
        for uploaded in uploaded_files:
            file_path = settings.ingest_data_dir / uploaded.name
            file_path.write_bytes(uploaded.getvalue())
            chunks = ingest_file(file_path, store)
            total_chunks += chunks
            st.success(f"Ingested {uploaded.name} ({chunks} chunks)")
        st.info(f"Total chunks added: {total_chunks}")
    except Exception as exc:  # pylint: disable=broad-except
        st.error(
            "Failed to ingest. Ensure an embedding provider is running. "
            "If using Ollama, start the service and pull the embedding model "
            f"({settings.ollama_embed_model}). Error: {exc}"
        )
elif not uploaded_files:
    st.info("Upload one or more PDF/DOCX files to start ingestion.")
