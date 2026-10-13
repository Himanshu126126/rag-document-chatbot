"""Streamlit UI for AskMyPDF."""
import logging
import tempfile
from pathlib import Path

import streamlit as st

from src.chain import ask
from src.ingest import build_index

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="AskMyPDF", page_icon=":page_facing_up:", layout="wide")
st.title("AskMyPDF - Chat with your documents")
st.caption("Upload PDFs, ask questions, and get answers with page-level sources.")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("store", None)

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if st.button("Process documents", disabled=not files):
        with st.spinner("Indexing documents..."):
            tmp_dir = Path(tempfile.mkdtemp())
            paths = []
            for f in files:
                target = tmp_dir / f.name
                target.write_bytes(f.getvalue())
                paths.append(target)
            try:
                st.session_state.store = build_index(paths)
                st.session_state.messages = []
                st.success(f"Ready: {len(paths)} document(s) indexed.")
            except Exception as exc:
                st.error(f"Could not process documents: {exc}")
    if st.button("Clear chat"):
        st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask something about your documents...")
if question:
    if st.session_state.store is None:
        st.warning("Please upload and process a PDF first.")
        st.stop()

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                answer, docs = ask(
                    st.session_state.store, question, st.session_state.messages
                )
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.stop()
        st.write(answer)
        with st.expander("Sources"):
            for d in docs:
                page = int(d.metadata.get("page", 0)) + 1
                st.markdown(f"**{d.metadata.get('source')}** - page {page}")
                st.caption(d.page_content[:300] + "...")

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})
