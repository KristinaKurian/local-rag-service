"""PDF ingestion: load, chunk, embed, and persist to ChromaDB.

Ingestion is idempotent per PDF path and indexing configuration:
- unchanged PDFs are skipped only when the stored index fingerprint matches;
- changed PDFs are staged before old chunks are removed;
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
    """Create the embedding model once and reuse it for the process lifetime."""
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


def _index_config_fingerprint() -> str:
    """Fingerprint settings that change the persisted vector index representation.

    The collection name is derived from this value so an embedding model or
    chunking change never mixes incompatible vectors/chunks with an old index.
    """
    payload = "|".join(
        [
            f"schema={settings.index_schema_version}",
            f"embedding_model={settings.embedding_model}",
            f"normalize_embeddings={int(settings.normalize_embeddings)}",
            f"chunk_size={settings.chunk_size}",
            f"chunk_overlap={settings.chunk_overlap}",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _effective_collection_name() -> str:
    """Return a Chroma collection isolated by the current index configuration."""
    suffix = _index_config_fingerprint()[:12]
    base = settings.chroma_collection_name[:48].rstrip("._-") or "rag"
    return f"{base}-{suffix}"


def _document_index_fingerprint(document_hash: str) -> str:
    """Fingerprint both PDF bytes and the configuration used to index them."""
    value = f"{document_hash}:{_index_config_fingerprint()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _chunk_id(document_id: str, index_fingerprint: str, chunk_index: int) -> str:
    """Create a deterministic id for one chunk version."""
    value = f"{document_id}:{index_fingerprint}:{chunk_index}"
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
    """Find current chunks, including entries created by older index versions."""
    ids, metadatas = _collection_get_ids(store, where={"document_id": document_id})
    if ids:
        return ids, metadatas, False

    ids, metadatas = _collection_get_ids(store, where={"source_path": source_path})
    if ids:
        return ids, metadatas, False

    # Backward compatibility for records that stored only the basename.
    ids, metadatas = _collection_get_ids(store, where={"source": source_name})
    return ids, metadatas, bool(ids)


def _is_index_current(
    ids: list[str],
    metadatas: list[dict[str, Any]],
    index_fingerprint: str,
) -> bool:
    """Return True only when every stored chunk belongs to the current index version."""
    if not ids or len(ids) != len(metadatas):
        return False
    return all(meta.get("index_fingerprint") == index_fingerprint for meta in metadatas)


def _rollback_staged_chunks(store: Chroma, staged_ids: list[str]) -> None:
    """Best-effort rollback of chunks created during a failed reindex."""
    if not staged_ids:
        return
    try:
        store._collection.delete(ids=staged_ids)  # noqa: SLF001
    except Exception:
        logger.exception("Failed to roll back %d staged chunk(s)", len(staged_ids))


def ingest_pdfs(pdf_folder: str | Path | None = None) -> dict[str, int]:
    """Index PDFs idempotently while preserving the previous version on failure.

    New chunks are fully prepared and written first. Previous chunks are removed
    only after the new version was stored successfully. If staging or cleanup
    fails, newly staged ids are rolled back and the previous index is retained.
    """
    source_root = Path(pdf_folder or settings.pdf_folder).resolve()
    pdf_paths = _collect_pdf_paths(source_root)
    if not pdf_paths:
        logger.warning("No PDF files found under %s", source_root)

    embeddings = get_embeddings()
    persist_path = settings.chroma_persist_path
    persist_path.mkdir(parents=True, exist_ok=True)

    store = Chroma(
        persist_directory=str(persist_path),
        embedding_function=embeddings,
        collection_name=_effective_collection_name(),
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
        index_fingerprint = _document_index_fingerprint(document_hash)

        existing_ids, existing_metadatas, legacy_match = _find_existing_document(
            store,
            document_id=document_id,
            source_path=relative_source,
            source_name=pdf_path.name,
        )

        if not legacy_match and _is_index_current(
            existing_ids,
            existing_metadatas,
            index_fingerprint,
        ):
            files_skipped += 1
            logger.info("Skipping unchanged PDF/index: %s", relative_source)
            continue

        # Prepare the complete new document version before touching old chunks.
        logger.info("Loading PDF: %s", pdf_path)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for doc in pages:
            meta = dict(doc.metadata) if doc.metadata else {}
            meta["source"] = pdf_path.name
            meta["source_path"] = relative_source
            meta["document_id"] = document_id
            meta["document_hash"] = document_hash
            meta["index_fingerprint"] = index_fingerprint
            meta["index_schema_version"] = settings.index_schema_version
            meta["page"] = int(meta.get("page", 0))
            doc.metadata = meta

        chunks = splitter.split_documents(pages)
        if not chunks:
            raise RuntimeError(
                f"PDF produced no text chunks; previous index was kept unchanged: {relative_source}"
            )

        chunk_ids: list[str] = []
        for index, chunk in enumerate(chunks):
            chunk_identifier = _chunk_id(document_id, index_fingerprint, index)
            meta = dict(chunk.metadata) if chunk.metadata else {}
            meta["chunk_index"] = index
            meta["chunk_id"] = chunk_identifier
            chunk.metadata = meta
            chunk_ids.append(chunk_identifier)

        existing_id_set = set(existing_ids)
        new_id_set = set(chunk_ids)
        staged_ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in existing_id_set]
        old_ids_to_remove = [chunk_id for chunk_id in existing_ids if chunk_id not in new_id_set]

        # Stage the new version first. On failure, remove any newly introduced ids.
        try:
            store.add_documents(chunks, ids=chunk_ids)
        except Exception:
            _rollback_staged_chunks(store, staged_ids)
            logger.exception("Failed to stage new chunks for %s; previous index kept", relative_source)
            raise

        # Commit the swap by removing only the previous version after staging succeeded.
        if old_ids_to_remove:
            try:
                store._collection.delete(ids=old_ids_to_remove)  # noqa: SLF001
            except Exception:
                _rollback_staged_chunks(store, staged_ids)
                logger.exception(
                    "Failed to remove previous chunks for %s; staged version rolled back",
                    relative_source,
                )
                raise

            chunks_removed += len(old_ids_to_remove)
            logger.info(
                "Removed %d previous chunk(s) for %s%s",
                len(old_ids_to_remove),
                relative_source,
                " (legacy records)" if legacy_match else "",
            )

        chunks_created += len(chunks)
        files_indexed += 1
        logger.info("Indexed %d chunk(s) for %s", len(chunks), relative_source)

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
    """Load the Chroma collection for the current indexing configuration."""
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
            collection_name=_effective_collection_name(),
        )
    except Exception as exc:
        raise RuntimeError(f"Could not load Chroma vector store from '{path}': {exc}") from exc
