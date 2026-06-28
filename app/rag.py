import os
import time
import weaviate
import weaviate.classes.config as wc
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
LLM_MODEL = os.getenv("LLM_MODEL", "granite4.1:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
COLLECTION_NAME = "RagDocuments"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.

Context from uploaded documents:
{context}

Answer the question using only the context above. If the context does not contain enough information, say so clearly."""


def init_weaviate_client() -> weaviate.WeaviateClient:
    last_exc = None
    for attempt in range(30):
        try:
            client = weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
            if client.is_ready():
                return client
        except Exception as exc:
            last_exc = exc
        time.sleep(2)
    raise RuntimeError(f"Weaviate not ready after 60 s: {last_exc}")


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)


def get_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST, temperature=0)


def create_vectorstore(client: weaviate.WeaviateClient, embeddings: OllamaEmbeddings) -> WeaviateVectorStore:
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=wc.Configure.Vectorizer.none(),
        properties=[
            wc.Property(name="content", data_type=wc.DataType.TEXT,
                        skip_vectorization=True, vectorize_property_name=False),
            wc.Property(name="source", data_type=wc.DataType.TEXT,
                        skip_vectorization=True, vectorize_property_name=False),
            wc.Property(name="chunk_idx", data_type=wc.DataType.TEXT,
                        skip_vectorization=True, vectorize_property_name=False),
        ],
    )

    return WeaviateVectorStore(
        client=client,
        index_name=COLLECTION_NAME,
        text_key="content",
        embedding=embeddings,
        attributes=["source", "chunk_idx"],
    )


def parse_document(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        import docx
        doc = docx.Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        raw = uploaded_file.read()
        return raw.decode("utf-8", errors="replace")


def add_document(vectorstore: WeaviateVectorStore, text: str, filename: str) -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    if not chunks:
        return 0
    metadatas = [{"source": filename, "chunk_idx": str(i)} for i in range(len(chunks))]
    vectorstore.add_texts(chunks, metadatas=metadatas)
    return len(chunks)


def answer_question(
    vectorstore: WeaviateVectorStore,
    llm: ChatOllama,
    question: str,
    chat_history: list[tuple[str, str]],
) -> tuple[str, list]:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    source_docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    history_messages = []
    for human, ai in chat_history:
        history_messages.append(HumanMessage(content=human))
        history_messages.append(AIMessage(content=ai))

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "history": history_messages,
        "question": question,
    })

    return answer, source_docs
