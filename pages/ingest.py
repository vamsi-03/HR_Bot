import streamlit as st
from pathlib import Path

from config.settings import settings
from rag.ingest import SUPPORTED_EXTS, ingest_file
from services.resources import get_store

st.set_page_config(page_title="Ingest Documents")

# Per-session reset: clear uploads and index on a new session
if "session_reset_done" not in st.session_state:
    if settings.ingest_data_dir.exists():
        for f in settings.ingest_data_dir.glob("*"):
            f.unlink()
    index_file = Path(settings.vector_store_path)
    meta_file = index_file.with_suffix(".meta.json")
    for f in [index_file, meta_file]:
        if f.exists():
            f.unlink()
    # drop cached resources so a fresh VectorStore is created
    if hasattr(st, "cache_resource"):
        st.cache_resource.clear()
    st.session_state["session_reset_done"] = True

store = get_store()

st.title("Ingest HR Documents")
st.caption("Upload PDF or DOCX files to make them searchable. Files are stored locally.")

st.markdown(
    """
<style>
.block-container {max-width: 1180px; padding-top: 2rem;}
body {background: #0b1220; color: #e2e8f0;}
[data-testid="stSidebar"] {
    background: #f7f9fc;
    width: 260px;
    min-width: 260px;
}
[data-testid="stSidebarNav"] { color: #0f172a; }
[data-testid="stSidebarNav"] ul {
    background: #eef2f7;
    border-radius: 12px;
    padding: 0.6rem;
    box-shadow: inset 0 1px 0 rgba(0,0,0,0.04);
}
[data-testid="stSidebarNav"] ul li { margin-bottom: 0.2rem; }
[data-testid="stSidebarNav"] ul li a {
    border-radius: 12px;
    padding: 0.7rem 0.85rem;
    color: #0f172a;
    font-weight: 700;
    font-size: 15px;
    transition: background 0.2s ease, color 0.2s ease;
}
[data-testid="stSidebarNav"] ul li a:hover { background: #dce7f5; }
[data-testid="stSidebarNav"] ul li a[aria-current="page"] {
    background: linear-gradient(120deg, rgba(34,211,238,0.22), rgba(124,58,237,0.18));
    color: #0b1220;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
@media (prefers-color-scheme: dark) {
    [data-testid="stSidebar"] { background: #0d1525; }
    [data-testid="stSidebarNav"] { color: #e2e8f0; }
    [data-testid="stSidebarNav"] ul { background: #111b2d; box-shadow: inset 0 1px 0 rgba(255,255,255,0.04); }
    [data-testid="stSidebarNav"] ul li a { color: #e2e8f0; }
    [data-testid="stSidebarNav"] ul li a:hover { background: rgba(148,163,184,0.2); }
    [data-testid="stSidebarNav"] ul li a[aria-current="page"] {
        background: linear-gradient(120deg, rgba(34,211,238,0.3), rgba(124,58,237,0.28));
        color: #0b1220;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
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

if "session_uploads" not in st.session_state:
    st.session_state["session_uploads"] = []

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
            st.session_state["session_uploads"].append(uploaded.name)
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
if st.button("Reload index"):
    if hasattr(store, "reload"):
        store.reload()
    else:  # fallback if cached instance lacks reload (after code update)
        from rag.vector_store import VectorStore
        store = VectorStore(settings.vector_store_path)
    st.rerun()

if store.metadata:
    sources = sorted({m.get("source", "") for m in store.metadata if m.get("source")})
    for name in sources:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"<span class='pill'>{name}</span>", unsafe_allow_html=True)
        with col2:
            if st.button("Remove", key=f"rm_{name}"):
                if hasattr(store, "remove_source"):
                    removed = store.remove_source(name)
                else:
                    # fallback to fresh instance if cached store lacks method
                    from rag.vector_store import VectorStore
                    fresh = VectorStore(settings.vector_store_path)
                    removed = fresh.remove_source(name)
                    # swap the cached instance reference
                    store = fresh
                file_on_disk = settings.ingest_data_dir / name
                if file_on_disk.exists():
                    file_on_disk.unlink()
                if name in st.session_state.get("session_uploads", []):
                    st.session_state["session_uploads"].remove(name)
                if removed:
                    st.success(f"Removed {name} and its chunks.")
                else:
                    st.info(f"No chunks removed for {name}.")
                st.rerun()
else:
    st.info("No files ingested yet.")

st.subheader("Session uploads")
if st.session_state["session_uploads"]:
    st.write(", ".join(st.session_state["session_uploads"]))
else:
    st.info("No uploads this session.")

if st.button("Clear session uploads and index"):
    if settings.ingest_data_dir.exists():
        for f in settings.ingest_data_dir.glob("*"):
            f.unlink()
    index_file = Path(settings.vector_store_path)
    meta_file = index_file.with_suffix(".meta.json")
    for f in [index_file, meta_file]:
        if f.exists():
            f.unlink()
    from rag.vector_store import VectorStore
    store = VectorStore(settings.vector_store_path)
    st.session_state["session_uploads"] = []
    st.success("Cleared uploads and index for this session.")
    st.rerun()
