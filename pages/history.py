import streamlit as st

from config.settings import settings
from services.resources import get_store

store = get_store()

st.set_page_config(page_title="History")
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

st.subheader("Vector store info")
st.write(f"Indexed chunks: {len(store.metadata)}")
st.write(f"Store path: {settings.vector_store_path}")
st.write(f"Uploads path: {settings.ingest_data_dir}")
