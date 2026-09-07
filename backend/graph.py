"""LangGraph StateGraph pipeline: retrieve → generate."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from .config import settings
from .retrieval import search_relevant_documents

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = "I could not find an answer in the provided documents."

SYSTEM_PROMPT = """You are a helpful assistant. Answer the question using ONLY the context provided below. If the answer is not in the context, say 'I could not find an answer in the provided documents.' Always end your response by listing the source filenames you used under a 'Sources:' heading.

Context:
{context}

Question: {question}"""


class RAGState(TypedDict, total=False):
    question: str
    retrieved_docs: list
    retrieval_scores: list[float]
    answer: str
    sources: list[str]


def _format_context(docs: list[Any]) -> str:
    parts: list[str] = []
    for i, doc in enumerate(docs or [], start=1):
        text = getattr(doc, "page_content", "") or ""
        parts.append(f"[{i}] {text}")
    return "\n\n".join(parts)


def _sources_from_docs(docs: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for d in docs or []:
        meta = getattr(d, "metadata", None) or {}
        if isinstance(meta, dict):
            src = meta.get("source")
            if src and src not in seen:
                seen.add(str(src))
                out.append(str(src))
    return out


def build_rag_graph(vectorstore: Chroma) -> Any:
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature,
    )

    def retrieve_node(state: RAGState) -> dict[str, Any]:
        q = state.get("question") or ""
        docs, scores = search_relevant_documents(vectorstore, q)
        logger.debug(
            "retrieve_node: %d relevant docs for question=%r threshold=%.3f",
            len(docs),
            q[:80],
            settings.relevance_score_threshold,
        )
        return {
            "retrieved_docs": docs,
            "retrieval_scores": scores,
        }

    def generate_node(state: RAGState) -> dict[str, Any]:
        q = state.get("question") or ""
        docs = state.get("retrieved_docs") or []

        # Do not ask the LLM to improvise when retrieval found no sufficiently
        # relevant context. This makes the threshold meaningful and keeps the
        # graph grounded in the indexed PDFs.
        if not docs:
            return {
                "answer": NO_CONTEXT_ANSWER,
                "sources": [],
            }

        context = _format_context(docs)
        sources = _sources_from_docs(docs)
        prompt_text = SYSTEM_PROMPT.format(context=context, question=q)
        messages = [HumanMessage(content=prompt_text)]
        try:
            resp = llm.invoke(messages)
        except Exception:
            logger.exception("ChatOllama invoke failed in generate_node")
            raise
        content = getattr(resp, "content", None) or str(resp)
        return {"answer": str(content), "sources": sources}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    compiled = graph.compile()
    logger.info(
        "Compiled LangGraph RAG pipeline (model=%s, top_k=%d, threshold=%.3f)",
        settings.ollama_model,
        settings.top_k_results,
        settings.relevance_score_threshold,
    )
    return compiled


def run_graph(graph: Any, question: str) -> dict[str, Any]:
    final = graph.invoke({"question": question})
    docs = final.get("retrieved_docs") or []
    chunks_used = len(docs) if isinstance(docs, list) else 0
    return {
        "answer": str(final.get("answer") or ""),
        "sources": list(final.get("sources") or []),
        "chunks_used": chunks_used,
        "retrieval_scores": [float(score) for score in final.get("retrieval_scores") or []],
    }
