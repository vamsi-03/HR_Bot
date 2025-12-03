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
    body {background: #0b1220; color: #e2e8f0;}
    .block-container {max-width: 1180px; padding-top: 2rem;}
    h1, h2, h3, h4 {color: #e2e8f0;}
    .hero {
        background: linear-gradient(120deg, rgba(34,211,238,0.12), rgba(124,58,237,0.12));
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    [data-testid="stExpander"] {
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 10px;
    }
    [data-testid="stExpander"] summary { font-weight: 600;}
    [data-testid="stExpander"] .streamlit-expanderContent {
        white-space: pre-wrap;
        word-break: break-word;
        overflow-wrap: anywhere;
        max-height: 320px;
        overflow-y: auto;
    }
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 1.5rem;
        # left: 0;
        width: calc(95% - 25rem);
        margin: 0 1rem;
        z-index: 10;
    }
    section.main > div {padding-bottom: 90px;}
    /* Sidebar navigation styling */
    [data-testid="stSidebarNav"] {
        padding-top: 0.5rem;
    }
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

store = get_store()
llm = get_llm()

if "history" not in st.session_state:
    st.session_state["history"]: List[Dict] = []


# def render_sidebar() -> None:
    # st.sidebar.header("HR Bot")


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
st.markdown(
    """
    <div class="hero">
      <div style="display:flex;align-items:center;gap:12px;">
        <div style="width:42px;height:42px;border-radius:12px;background:rgba(34,211,238,0.2);display:flex;align-items:center;justify-content:center;font-size:22px;">🛡️</div>
        <div>
          <div style="font-size:20px;font-weight:700;">HR Assistant (Grounded RAG)</div>
          <div style="font-size:14px;">Grounded answers with citations; no hallucinations.</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
# render_sidebar()

st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
render_history()
st.markdown('</div>', unsafe_allow_html=True)

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
