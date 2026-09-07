"""Shared retrieval helpers with relevance-score filtering."""

from __future__ import annotations

import logging
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .config import settings

logger = logging.getLogger(__name__)


def search_relevant_documents(
    vectorstore: Chroma,
    query: str,
) -> tuple[list[Document], list[float]]:
    """Return only chunks whose normalized relevance score passes the threshold.

    The raw vector search still asks Chroma for ``top_k_results`` candidates. We
    then filter those candidates using ``relevance_score_threshold``. The score
    is copied into each returned Document's metadata so callers can expose it in
    diagnostics without running a second vector search.
    """
    candidates = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=settings.top_k_results,
    )

    documents: list[Document] = []
    scores: list[float] = []

    for document, raw_score in candidates:
        score = float(raw_score)
        if score < settings.relevance_score_threshold:
            continue

        metadata = dict(getattr(document, "metadata", None) or {})
        metadata["relevance_score"] = score
        documents.append(
            Document(
                page_content=document.page_content,
                metadata=metadata,
            )
        )
        scores.append(score)

    logger.debug(
        "Retrieval query=%r candidates=%d accepted=%d threshold=%.3f scores=%s",
        query[:100],
        len(candidates),
        len(documents),
        settings.relevance_score_threshold,
        [round(score, 4) for score in scores],
    )
    return documents, scores


class ScoreThresholdRetriever(BaseRetriever):
    """LangChain-compatible retriever backed by the shared score-filtered search."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Any

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        documents, _ = search_relevant_documents(self.vectorstore, query)
        return documents
