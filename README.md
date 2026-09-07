# Local RAG Service

**Local RAG Service** is a small, production-minded local RAG chatbot: ask questions over your own PDFs, fully offline, with **Ollama** as the LLM and **ChromaDB** + **sentence-transformers** for retrieval.

## Prerequisites

- **Python 3.11+** (required — Local RAG Service is not tested on older interpreters)
- **Ollama** installed and running locally
- The configured Ollama model pulled (default: **llama3.2:3b**): `ollama pull llama3.2:3b`

Before creating the venv, confirm the interpreter: `python --version` should report **3.11.x or newer**. On Windows, if `python` still points at an older release, use the **Python 3.11+** launcher instead, for example `py -3.11 -m venv .venv`, so `pip install` pulls pre-built wheels and you avoid a from-source `greenlet` build that requires **Microsoft C++ Build Tools**.

## Setup

1. Open a terminal in the `local-rag-service` directory (the folder that contains `backend/`, `frontend/`, and `requirements.txt`).
2. Create a virtual environment (recommended) and install dependencies:
  ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
  ```
   On Windows, if `python` is not 3.11+, prefer: `py -3.11 -m venv .venv` then activate and `pip install` as above.
3. (Optional) Copy `.env.example` to `.env` if you want to document local overrides for your environment.
4. Start the API (from the `local-rag-service` directory):
  ```bash
   python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
  ```
5. Open the UI in a browser: `http://127.0.0.1:8000/`

## Usage

1. Copy PDFs into `data/pdfs/` (subfolders are fine; ingestion is recursive).
2. Click **Ingest PDFs** in the sidebar (or `POST /ingest`). Ingestion is idempotent: unchanged PDFs are skipped, while changed PDFs replace their previously indexed chunks.
3. Choose **Langgraph** or **Langchain** as the pipeline, then chat.

The vector database is written to `chroma_db/` on disk and reused across restarts. Repeating `/ingest` with unchanged files does not duplicate chunks.

## API


| Method | Path      | Description                                                                                                                                                                                                                                                                                 |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/health` | Liveness probe: `{ "status": "ok" }`.                                                                                                                                                                                                                                                       |
| `GET`  | `/status` | Ollama reachability (`/api/tags`), indexed chunk count, active models, `top_k_results`, and the current relevance-score threshold. |
| `POST` | `/ingest` | Body: `{ "folder_path": "<optional path>" }`. Defaults to configured `data/pdfs`. Idempotently indexes PDFs in Chroma: unchanged files are skipped and changed files replace old chunks. Returns processing, skip, replacement, and collection-size counters.                                                                                            |
| `POST` | `/query`  | Body: `{ "question": "<text>", "use_graph": true }`. Both pipelines retrieve `TOP_K_RESULTS` candidates and keep only chunks whose relevance score is at least `RELEVANCE_SCORE_THRESHOLD`. Returns `answer`, `sources`, `chunks_used`, `retrieval_scores`, and `pipeline`. |


Errors return JSON with a `detail` field (FastAPI default). If nothing has been ingested yet, `/query` responds with **400** and: `No documents found. Please ingest PDFs first via POST /ingest`.

## Architecture

Local RAG Service ships **two interchangeable RAG paths** over the same Chroma vector store:

1. **LangGraph (`use_graph: true`)** — A `StateGraph` with two nodes: **retrieve** and **generate**. Retrieval asks Chroma for the top-k candidates together with relevance scores, filters them by `RELEVANCE_SCORE_THRESHOLD`, and stores accepted scores in graph state. If no chunk passes the threshold, the graph returns a grounded "no answer in documents" response without calling the LLM.
2. **LangChain chain (`use_graph: false`)** — `create_retrieval_chain` + `create_stuff_documents_chain`, using the same shared score-threshold retriever, so both modes follow the same retrieval policy.

The sentence-transformers embedding model is cached once per process and reused by ingestion and query-side Chroma clients. FastAPI runtime objects (`vectorstore`, LangChain chain, and LangGraph graph) live in `app.state` and are obtained in request handlers through a dependency instead of module-level globals. All configuration comes from `backend/config.py` via `pydantic-settings` (environment variables / `.env`).

## Project structure

```
local-rag-service/
├── backend/
│   ├── __init__.py
│   ├── main.py        # FastAPI app, CORS, routes, lifespan
│   ├── ingest.py      # PDF load → split → embed → Chroma
│   ├── retrieval.py   # Shared relevance-score retrieval policy
│   ├── rag_chain.py   # LangChain retrieval + stuff chain
│   ├── graph.py       # LangGraph StateGraph RAG
│   └── config.py      # Paths, models, chunking, top-k, score threshold
├── frontend/
│   └── index.html     # Single-file dark UI
├── data/
│   └── pdfs/          # Drop PDFs here
├── chroma_db/         # Created by Chroma after first ingest
├── requirements.txt
├── .env.example
└── README.md
```

