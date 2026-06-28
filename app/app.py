import streamlit as st
from rag import (
    init_weaviate_client,
    get_embeddings,
    get_llm,
    create_vectorstore,
    parse_document,
    add_document,
    answer_question,
)

st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="wide")
st.title("📚 RAG Chat")


# ── Session-state bootstrap ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to services…")
def get_resources():
    client = init_weaviate_client()
    embeddings = get_embeddings()
    llm = get_llm()
    vectorstore = create_vectorstore(client, embeddings)
    return client, embeddings, llm, vectorstore


try:
    _client, _embeddings, _llm, _vectorstore = get_resources()
except Exception as exc:
    st.error(f"Could not connect to backend services: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content, sources?}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # [(human, ai), …] for memory
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = set()


# ── Sidebar – document upload ─────────────────────────────────────────────────

with st.sidebar:
    st.header("Documents")
    st.caption("Upload PDF, TXT, Markdown, or DOCX files.")

    uploaded = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.loaded_docs:
                with st.spinner(f"Processing **{f.name}**…"):
                    try:
                        text = parse_document(f)
                        n = add_document(_vectorstore, text, f.name)
                        st.session_state.loaded_docs.add(f.name)
                        st.success(f"✅ {f.name} — {n} chunks indexed")
                    except Exception as exc:
                        st.error(f"❌ {f.name}: {exc}")

    if st.session_state.loaded_docs:
        st.divider()
        st.subheader("Indexed documents")
        for name in sorted(st.session_state.loaded_docs):
            st.write(f"• {name}")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()


# ── Chat area ─────────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

prompt = st.chat_input("Ask a question about your documents…")

if prompt:
    if not st.session_state.loaded_docs:
        st.warning("Upload at least one document before asking questions.")
        st.stop()

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer, source_docs = answer_question(
                    _vectorstore, _llm, prompt, st.session_state.chat_history
                )
            except Exception as exc:
                st.error(f"Generation error: {exc}")
                st.stop()

        # Deduplicate sources; keep order
        seen, sources = set(), []
        for doc in source_docs:
            src = doc.metadata.get("source", "unknown")
            chunk = doc.metadata.get("chunk_idx", "?")
            label = f"**{src}** — chunk {chunk}"
            if label not in seen:
                seen.add(label)
                sources.append(label)

        st.markdown(answer)
        if sources:
            with st.expander("Sources", expanded=True):
                for src in sources:
                    st.markdown(f"- {src}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    st.session_state.chat_history.append((prompt, answer))
