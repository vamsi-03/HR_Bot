import streamlit as st

from config.settings import settings
from services.resources import get_store

store = get_store()

st.set_page_config(page_title="History")
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Session & Store")

st.subheader("Chat history")
if "history" not in st.session_state or not st.session_state["history"]:
    st.info("No chat history yet.")
else:
    for turn in st.session_state["history"]:
        st.markdown(f"- **Q:** {turn['question']}")
        st.markdown(f"  **A:** {turn['answer']}")
    if st.button("Clear history"):
        st.session_state["history"] = []
        st.success("History cleared")

# st.subheader("Vector store info")
# st.write(f"Indexed chunks: {len(store.metadata)}")
# st.write(f"Store path: {settings.vector_store_path}")
# st.write(f"Uploads path: {settings.ingest_data_dir}")
