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
