import logging
from typing import Dict, Iterator, List

import streamlit as st

from config.logging_config import setup_logging
from config.settings import settings
from rag.retrieval import answer_question, answer_question_stream
from services.resources import get_llm, get_store

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="HR RAG Bot", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 1.5rem;
        # left: 0;
        width: calc(95% - 25rem);
        margin: 0 1rem;
        z-index: 10;
    }
    section.main > div {padding-bottom: 90px;}
    </style>
    """,
    unsafe_allow_html=True,
)

store = get_store()
llm = get_llm()

if "history" not in st.session_state:
    st.session_state["history"]: List[Dict] = []


def render_sidebar() -> None:
    st.sidebar.header("Status")
    statuses = llm.provider_statuses()
    for status in statuses:
        icon = "✅" if status.available else "⚠️"
        st.sidebar.write(f"{icon} {status.name}: {status.detail or 'ready'}")
    st.sidebar.write(f"Indexed chunks: {len(store.metadata)}")
    st.sidebar.write(f"Vector store: {settings.vector_store_path}")
    st.sidebar.divider()
    st.sidebar.write("Use the Ingestion page to upload PDFs or DOCX files.")


def render_history() -> None:
    for idx, turn in enumerate(st.session_state["history"]):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["citations"]:
                with st.expander("Citations"):
                    for c_idx, cite in enumerate(turn["citations"], start=1):
                        snippet = cite.get("text") or ""
                        st.markdown(
                            f"**Source {c_idx}** — {cite.get('source')} (page {cite.get('page')}) "
                            f"[chunk {cite.get('chunk_id')}], score: {cite.get('score'):.2f}"
                        )
                        st.write(snippet)


st.title("HR Assistant (Grounded RAG)")
st.caption("Grounded answers with strict citations; falls back to 'No information found.'")
render_sidebar()

render_history()

prompt = st.chat_input("Ask an HR question")
if prompt:
    if not store.metadata:
        answer_payload = {"answer": "No information found. Please ingest documents first.", "citations": []}
        citations: List[Dict] = []
        stream = iter([answer_payload["answer"]])
    else:
        stream, citations, grounded = answer_question_stream(prompt, store, llm)
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        chunks: List[str] = []
        def collector() -> Iterator[str]:
            for token in stream:
                chunks.append(token)
                yield token
        st.write_stream(collector())
        if store.metadata and citations:
            with st.expander("Citations"):
                for idx, cite in enumerate(citations, start=1):
                    snippet = cite.get("text") or ""
                    st.markdown(
                        f"**Source {idx}** — {cite.get('source')} (page {cite.get('page')}) "
                        f"[chunk {cite.get('chunk_id')}], score: {cite.get('score'):.2f}"
                    )
                    st.write(snippet)
        answer_text = "".join(chunks).strip()
    st.session_state["history"].append(
        {
            "id": len(st.session_state["history"]),
            "question": prompt,
            "answer": answer_text if answer_text else "No information found.",
            "citations": citations if store.metadata else [],
        }
    )
    st.rerun()
