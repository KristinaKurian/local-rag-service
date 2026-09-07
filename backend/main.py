"""FastAPI server for Local RAG Service — ingest, query, status, health."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import settings
from .graph import build_rag_graph, run_graph
from .ingest import get_vectorstore, ingest_pdfs
from .rag_chain import build_rag_chain, query_chain

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("local_rag_service")


@dataclass(slots=True)
class AppRuntime:
    """Runtime dependencies owned by one FastAPI application instance."""

    vectorstore: Any = None
    rag_chain: Any = None
    rag_graph: Any = None


def _build_runtime() -> AppRuntime:
    """Build a complete pipeline before publishing it to ``app.state``."""
    logger.info("Loading vector store and rebuilding RAG pipelines…")
    vectorstore = get_vectorstore()
    rag_chain = build_rag_chain(vectorstore)
    rag_graph = build_rag_graph(vectorstore)
    logger.info("Pipelines ready.")
    return AppRuntime(
        vectorstore=vectorstore,
        rag_chain=rag_chain,
        rag_graph=rag_graph,
    )


def _reload_pipeline(app: FastAPI) -> AppRuntime:
    """Atomically replace the application's runtime dependencies."""
    runtime = _build_runtime()
    app.state.runtime = runtime
    return runtime


def get_runtime(request: Request) -> AppRuntime:
    """FastAPI dependency that exposes the current application runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        runtime = AppRuntime()
        request.app.state.runtime = runtime
    return runtime


def _store_document_count(runtime: AppRuntime) -> int:
    if runtime.vectorstore is None:
        return 0
    return int(runtime.vectorstore._collection.count())  # noqa: SLF001


def _store_has_documents(runtime: AppRuntime) -> bool:
    if runtime.vectorstore is None:
        return False
    try:
        sample = runtime.vectorstore.get(limit=1)
        ids = sample.get("ids") if isinstance(sample, dict) else None
        return bool(ids)
    except Exception:
        logger.exception("Failed to probe vector store for documents")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = AppRuntime()
    app.state.ingest_lock = asyncio.Lock()
    try:
        await run_in_threadpool(_reload_pipeline, app)
    except FileNotFoundError as exc:
        logger.warning("Vector store not ready at startup: %s", exc)
    except Exception:
        logger.exception("Startup pipeline load failed — ingest will be required")
    yield


app = FastAPI(title=settings.app_title, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


@app.get("/")
def serve_ui() -> FileResponse:
    index = settings.frontend_dir / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index)


class QueryBody(BaseModel):
    question: str = Field(..., min_length=1)
    use_graph: bool = True


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
async def status(runtime: AppRuntime = Depends(get_runtime)) -> dict[str, Any]:
    ollama_connected = False
    try:
        async with httpx.AsyncClient(timeout=settings.ollama_status_timeout_seconds) as client:
            r = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            ollama_connected = r.status_code == 200
    except Exception:
        logger.debug("Ollama not reachable at %s", settings.ollama_base_url, exc_info=True)

    documents_indexed = 0
    if runtime.vectorstore is not None:
        try:
            documents_indexed = await run_in_threadpool(_store_document_count, runtime)
        except Exception:
            logger.exception("Could not read Chroma document count")

    return {
        "ollama_connected": ollama_connected,
        "documents_indexed": documents_indexed,
        "model": settings.ollama_model,
        "embedding_model": settings.embedding_model,
        "top_k_results": settings.top_k_results,
        "relevance_score_threshold": settings.relevance_score_threshold,
    }


@app.post("/ingest")
async def ingest(request: Request) -> dict[str, Any]:
    # The HTTP API intentionally indexes only the operator-configured PDF root.
    # Arbitrary filesystem paths are not accepted from untrusted request data.
    folder = settings.pdf_folder
    logger.info("Ingest requested for configured folder: %s", folder)

    async with request.app.state.ingest_lock:
        try:
            summary = await run_in_threadpool(ingest_pdfs, folder)
        except Exception as exc:
            logger.exception("Ingest failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        try:
            await run_in_threadpool(_reload_pipeline, request.app)
        except FileNotFoundError:
            if summary["chunks_created"] > 0:
                logger.exception("Reload after ingest failed: vector store missing despite new chunks")
                raise HTTPException(
                    status_code=500,
                    detail="Ingest reported new chunks but the vector store could not be loaded.",
                ) from None
            logger.warning("Ingest completed with no indexed chunks; vector store not initialized.")
        except Exception as exc:
            logger.exception("Reload after ingest failed")
            raise HTTPException(
                status_code=500,
                detail=f"Ingest succeeded but failed to reload vector store: {exc}",
            ) from exc

    return {
        "status": "success",
        "files_processed": summary["files_processed"],
        "files_indexed": summary["files_indexed"],
        "files_skipped": summary["files_skipped"],
        "chunks_created": summary["chunks_created"],
        "chunks_removed": summary["chunks_removed"],
        "collection_size": summary["collection_size"],
    }


@app.post("/query")
async def query(
    body: QueryBody,
    runtime: AppRuntime = Depends(get_runtime),
) -> dict[str, Any]:
    has_documents = (
        runtime.vectorstore is not None
        and await run_in_threadpool(_store_has_documents, runtime)
    )
    if not has_documents:
        raise HTTPException(
            status_code=400,
            detail="No documents found. Please ingest PDFs first via POST /ingest",
        )

    use_graph = body.use_graph
    logger.info("Query (graph=%s): %r", use_graph, body.question[:200])

    try:
        if use_graph:
            if runtime.rag_graph is None:
                raise HTTPException(status_code=503, detail="LangGraph pipeline is not initialized.")
            out = await run_in_threadpool(run_graph, runtime.rag_graph, body.question)
            pipeline = "langgraph"
        else:
            if runtime.rag_chain is None:
                raise HTTPException(status_code=503, detail="LangChain pipeline is not initialized.")
            out = await run_in_threadpool(query_chain, runtime.rag_chain, body.question)
            pipeline = "langchain"
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Query failed")
        msg = str(exc).lower()
        if "connection" in msg or "refused" in msg or "connect" in msg:
            raise HTTPException(
                status_code=503,
                detail="Ollama does not appear to be running or reachable. Start Ollama and ensure "
                f"'{settings.ollama_model}' is available "
                f"(e.g. `ollama pull {settings.ollama_model}`).",
            ) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "answer": out["answer"],
        "sources": out["sources"],
        "chunks_used": out["chunks_used"],
        "retrieval_scores": out.get("retrieval_scores", []),
        "pipeline": pipeline,
    }
