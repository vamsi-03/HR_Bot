import streamlit as st

from config.settings import settings
from rag.ingest import SUPPORTED_EXTS, ingest_file
from services.resources import get_store

store = get_store()

st.set_page_config(page_title="Ingest Documents")
st.title("Ingest HR Documents")
st.caption("Upload PDF or DOCX files to make them searchable. Files are stored locally.")

st.markdown(
    """
<style>
.block-container {max-width: 1180px; padding-top: 2rem;}
body {background: #0b1220; color: #e2e8f0;}
[data-testid="stSidebarNav"] ul {
    background: #f8fafc;
    border-radius: 12px;
    padding: 0.5rem;
    box-shadow: inset 0 1px 0 rgba(0,0,0,0.02);
}
[data-testid="stSidebarNav"] ul li a {
    border-radius: 10px;
    padding: 0.55rem 0.75rem;
    color: #0f172a;
    font-weight: 600;
    transition: background 0.2s ease, color 0.2s ease;
}
[data-testid="stSidebarNav"] ul li a:hover {
    background: #e2e8f0;
}
[data-testid="stSidebarNav"] ul li a[aria-current="page"] {
    background: linear-gradient(120deg, rgba(34,211,238,0.22), rgba(124,58,237,0.18));
    color: #0f172a;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}
.upload-card {
    padding: 1rem 1.25rem;
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 12px;
    background: #0f172a;
        box-shadow: 0 12px 32px rgba(0,0,0,0.3);
    }
    .pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(14,165,233,0.15);
        color: #7dd3fc;
        margin: 0.2rem 0.35rem 0 0;
        font-size: 0.9rem;
    }
    .header-card {
        display: flex;
        align-items: center;
        gap: 12px;
        background: linear-gradient(120deg, rgba(34,211,238,0.12), rgba(124,58,237,0.12));
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="header-card">
      <div style="width:40px;height:40px;border-radius:12px;background:rgba(34,211,238,0.2);display:flex;align-items:center;justify-content:center;font-size:20px;">📂</div>
      <div>
        <div style="font-weight:700;font-size:18px;">Document Ingestion</div>
        <div style="color:#cbd5e1;font-size:14px;">Add HR PDFs/DOCX to keep answers grounded.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("#### Upload")
with st.container():
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

st.subheader("Ingested files")
if store.metadata:
    sources = sorted({m.get("source", "") for m in store.metadata if m.get("source")})
    for name in sources:
        st.markdown(f"<span class='pill'>{name}</span>", unsafe_allow_html=True)
else:
    st.info("No files ingested yet.")
