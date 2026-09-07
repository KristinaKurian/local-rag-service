"""PDF ingestion: load, chunk, embed, and persist to ChromaDB.

Ingestion is idempotent per PDF path:
- unchanged PDFs are skipped;
- changed PDFs replace their previously indexed chunks;
- repeated POST /ingest calls do not grow the collection with duplicates.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the embedding model once and reuse it for the process lifetime.

    Loading a sentence-transformers model is relatively expensive. Both ingest
    and query-side Chroma clients use this cached instance instead of loading
    the same model repeatedly whenever the pipeline is rebuilt.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": settings.normalize_embeddings},
    )


def _collect_pdf_paths(pdf_folder: str | Path) -> list[Path]:
    root = Path(pdf_folder).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PDF folder does not exist or is not a directory: {root}")
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def _sha256_file(path: Path) -> str:
    """Return a content hash without loading the entire PDF into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(pdf_path: Path, source_root: Path) -> str:
    """Return a stable, human-readable path relative to the ingestion root."""
    try:
        return pdf_path.resolve().relative_to(source_root).as_posix()
    except ValueError:
        return pdf_path.resolve().as_posix()


def _document_id(pdf_path: Path, source_root: Path) -> str:
    """Create a stable id for one logical PDF location."""
    identity = f"{source_root.as_posix()}::{_source_path(pdf_path, source_root)}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _chunk_id(document_id: str, chunk_index: int) -> str:
    """Create a deterministic id so the same logical chunk can be upserted safely."""
    value = f"{document_id}:{chunk_index}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _collection_get_ids(
    store: Chroma,
    *,
    where: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Read matching Chroma ids and metadata through one compatibility helper."""
    result = store._collection.get(  # noqa: SLF001 — Chroma wrapper has no equivalent metadata API
        where=where,
        include=["metadatas"],
    )
    ids = list(result.get("ids") or [])
    metadatas = [dict(item or {}) for item in (result.get("metadatas") or [])]
    return ids, metadatas


def _find_existing_document(
    store: Chroma,
    *,
    document_id: str,
    source_path: str,
    source_name: str,
) -> tuple[list[str], list[dict[str, Any]], bool]:
    """Find current chunks, including entries created by the pre-idempotent version.

    Returns ``(ids, metadatas, legacy_match)``. The legacy fallback makes the
    first ingest after this upgrade replace old chunks that only had ``source``
    metadata instead of leaving them alongside the new deterministic records.
    """
    ids, metadatas = _collection_get_ids(store, where={"document_id": document_id})
    if ids:
        return ids, metadatas, False

    ids, metadatas = _collection_get_ids(store, where={"source_path": source_path})
    if ids:
        return ids, metadatas, False

    # Backward-compatibility path for legacy records that stored only the basename.
    ids, metadatas = _collection_get_ids(store, where={"source": source_name})
    return ids, metadatas, bool(ids)


def _is_unchanged(
    ids: list[str],
    metadatas: list[dict[str, Any]],
    document_hash: str,
) -> bool:
    """An indexed PDF is unchanged only when every stored chunk has the same hash."""
    if not ids or len(ids) != len(metadatas):
        return False
    return all(meta.get("document_hash") == document_hash for meta in metadatas)


def ingest_pdfs(pdf_folder: str | Path | None = None) -> dict[str, int]:
    """Index PDFs without duplicating chunks on repeated ingestion.

    For every PDF we store a stable ``document_id`` and the current file
    ``document_hash`` in metadata. If the same file is ingested again unchanged,
    it is skipped. If its bytes changed, all previous chunks for that document
    are deleted and the new chunks are inserted with deterministic ids.
    """
    source_root = Path(pdf_folder or settings.pdf_folder).resolve()
    pdf_paths = _collect_pdf_paths(source_root)
    if not pdf_paths:
        logger.warning("No PDF files found under %s", source_root)

    embeddings = get_embeddings()
    persist_path = settings.chroma_persist_path
    persist_path.mkdir(parents=True, exist_ok=True)

    # Chroma get_or_create semantics let us use one path for both a fresh and an
    # already persisted collection.
    store = Chroma(
        persist_directory=str(persist_path),
        embedding_function=embeddings,
        collection_name=settings.chroma_collection_name,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    files_indexed = 0
    files_skipped = 0
    chunks_created = 0
    chunks_removed = 0

    for pdf_path in pdf_paths:
        relative_source = _source_path(pdf_path, source_root)
        document_id = _document_id(pdf_path, source_root)
        document_hash = _sha256_file(pdf_path)

        existing_ids, existing_metadatas, legacy_match = _find_existing_document(
            store,
            document_id=document_id,
            source_path=relative_source,
            source_name=pdf_path.name,
        )

        if not legacy_match and _is_unchanged(existing_ids, existing_metadatas, document_hash):
            files_skipped += 1
            logger.info("Skipping unchanged PDF: %s", relative_source)
            continue

        if existing_ids:
            store._collection.delete(ids=existing_ids)  # noqa: SLF001
            chunks_removed += len(existing_ids)
            logger.info(
                "Removed %d previously indexed chunk(s) for %s%s",
                len(existing_ids),
                relative_source,
                " (legacy records)" if legacy_match else "",
            )

        logger.info("Loading PDF: %s", pdf_path)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for doc in pages:
            meta = dict(doc.metadata) if doc.metadata else {}
            meta["source"] = pdf_path.name
            meta["source_path"] = relative_source
            meta["document_id"] = document_id
            meta["document_hash"] = document_hash
            meta["page"] = int(meta.get("page", 0))
            doc.metadata = meta

        chunks = splitter.split_documents(pages)
        chunk_ids: list[str] = []

        for index, chunk in enumerate(chunks):
            chunk_identifier = _chunk_id(document_id, index)
            meta = dict(chunk.metadata) if chunk.metadata else {}
            meta["chunk_index"] = index
            meta["chunk_id"] = chunk_identifier
            chunk.metadata = meta
            chunk_ids.append(chunk_identifier)

        if chunks:
            # LangChain's Chroma integration uses Chroma upsert under the hood.
            # Explicit deterministic ids also make accidental duplicate writes
            # safe if the same logical chunks are submitted again.
            store.add_documents(chunks, ids=chunk_ids)
            chunks_created += len(chunks)
            logger.info("Indexed %d chunk(s) for %s", len(chunks), relative_source)
        else:
            logger.warning("PDF produced no text chunks: %s", relative_source)

        files_indexed += 1

    collection_size = int(store._collection.count())  # noqa: SLF001

    return {
        "files_processed": len(pdf_paths),
        "files_indexed": files_indexed,
        "files_skipped": files_skipped,
        "chunks_created": chunks_created,
        "chunks_removed": chunks_removed,
        "collection_size": collection_size,
    }


def get_vectorstore() -> Chroma:
    """Load persisted Chroma. Raises a clear error if the store has not been created yet."""
    path = settings.chroma_persist_path
    if not path.exists() or not any(path.iterdir()):
        raise FileNotFoundError(
            f"No vector database found at '{path}'. "
            "Run POST /ingest after placing PDFs in the configured PDF folder."
        )

    embeddings = get_embeddings()
    try:
        return Chroma(
            persist_directory=str(path),
            embedding_function=embeddings,
            collection_name=settings.chroma_collection_name,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not load Chroma vector store from '{path}': {exc}") from exc
