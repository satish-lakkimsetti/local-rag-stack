# RAG Chat

A fully self-hosted Retrieval-Augmented Generation (RAG) application.  
Upload documents, ask questions, and get answers grounded in your content — no external APIs, no cloud dependencies.

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Docker Desktop (or Docker Engine + Compose plugin) | 24.x / Compose 2.x |

> **First-run note:** Ollama will download `granite4.1:3b` (~2.1 GB) and `nomic-embed-text` (~274 MB) automatically on first `docker compose up`. The UI will not appear until both models are fully downloaded. Subsequent runs skip the download.

---

## Running

```bash
cd local-rag-stack
docker compose up --build
```

Open **http://localhost:8501** in your browser.

The app is ready when all three containers show `healthy` in `docker compose ps`.

### Customising models

Override via environment variables:

```bash
LLM_MODEL=llama3.2:3b EMBED_MODEL=nomic-embed-text docker compose up --build
```

Or create a `.env` file in this directory:

```
LLM_MODEL=granite4.1:3b
EMBED_MODEL=nomic-embed-text
```

---

## Usage

1. **Upload documents** — use the sidebar file uploader (PDF, TXT, Markdown, DOCX).  
   Each file is parsed, chunked, embedded, and indexed in Weaviate immediately.
2. **Ask questions** — type in the chat box.  
   Answers include **source references** showing which document and chunk each piece of information came from.
3. **Follow-up questions** — the chat remembers previous turns, so context-aware follow-ups work naturally.
4. **Clear chat** — use the "Clear chat history" button in the sidebar to start a fresh conversation without re-uploading documents.

---

## Clean up

Stop and remove all containers and networks:

```bash
docker compose down
```

---

## Architecture

```
Browser
  │
  ▼
┌──────────────────────────────────┐
│  Streamlit app  (port 8501)      │
│                                  │
│  • Document parsing              │
│    (pypdf, python-docx, text)    │
│  • Chunking (1000 chars,         │
│    200 overlap)                  │
│  • Conversational memory         │
│  • Source citation               │
└───────────┬──────────────────────┘
            │  embed & store        retrieve top-4 chunks
            ▼                              │
┌───────────────────────┐       ┌──────────▼────────────┐
│  Ollama  (port 11434) │       │  Weaviate (port 8080) │
│                       │◄──────│                       │
│  • nomic-embed-text   │ embed │  • Vector index       │
│  • granite4.1:3b      │ query │  • Document chunks +  │
│    (generation)       │       │    metadata           │
└───────────────────────┘       └───────────────────────┘
```

### Services

| Service | Image | Role |
|---------|-------|------|
| **app** | Python 3.11 + Streamlit | Web UI and RAG pipeline |
| **ollama** | `ollama/ollama:latest` | Serves the LLM and embedding model |
| **weaviate** | `semitechnologies/weaviate:1.27.0` | Vector database for document chunks |

### RAG pipeline (per query)

1. The user's question is embedded with `nomic-embed-text` via Ollama.
2. Weaviate returns the top-4 most similar document chunks.
3. Those chunks plus the full conversation history are sent to `granite4.1:3b` for answer generation.
4. The answer and source references are displayed in the UI.

### Startup ordering

`docker compose` waits for each service's health check before starting dependents:

- **Ollama** becomes healthy only after both models finish downloading (sentinel file `/tmp/ollama_models_ready`).
- **Weaviate** becomes healthy after its `/v1/.well-known/ready` endpoint responds.
- **app** starts only after both are healthy.

### Data lifecycle

Document embeddings live in Weaviate's in-memory store with no volume mounted — everything is cleared on `docker compose down`. The stack starts clean each time.
