import hashlib
import os
from collections import defaultdict
from pathlib import Path
from threading import RLock
from typing import Iterable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAIError

from src.config import OPENAI_EMBEDDING_MODEL, VECTOR_STORE_DIR
from src.rag_errors import VectorStoreError


class PersistentVectorStore:
    """Thread-safe, persistent FAISS index with per-source replacement."""

    def __init__(
        self,
        index_dir: Path | str = VECTOR_STORE_DIR,
        embeddings: Embeddings | None = None,
    ) -> None:
        self.index_dir = Path(index_dir).resolve()
        self._embeddings = embeddings
        self._vector_store: FAISS | None = None
        self._loaded = False
        self._lock = RLock()

    def _get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise VectorStoreError(
                    "OPENAI_API_KEY is not configured. Add it to the project .env "
                    "file or the process environment."
                )
            self._embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        return self._embeddings

    def _load_persisted_store(self) -> FAISS | None:
        index_file = self.index_dir / "index.faiss"
        metadata_file = self.index_dir / "index.pkl"
        if not index_file.exists() and not metadata_file.exists():
            return None
        if not index_file.is_file() or not metadata_file.is_file():
            raise VectorStoreError(
                f"FAISS index at '{self.index_dir}' is incomplete. "
                "Restore both index files or remove the incomplete index."
            )

        try:
            return FAISS.load_local(
                str(self.index_dir),
                self._get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        except (OSError, ValueError, RuntimeError) as error:
            raise VectorStoreError(
                f"Could not load the FAISS index at '{self.index_dir}': {error}"
            ) from error

    def _get_store(self) -> FAISS | None:
        if not self._loaded:
            self._vector_store = self._load_persisted_store()
            self._loaded = True
        return self._vector_store

    @staticmethod
    def _indexed_sources(
        vector_store: FAISS,
    ) -> dict[str, tuple[list[str], set[str]]]:
        sources: dict[str, tuple[list[str], set[str]]] = {}
        for document_id in vector_store.index_to_docstore_id.values():
            document = vector_store.docstore.search(document_id)
            if not isinstance(document, Document):
                continue
            source_id = document.metadata.get("source_id")
            if not isinstance(source_id, str):
                continue
            document_ids, hashes = sources.setdefault(source_id, ([], set()))
            document_ids.append(document_id)
            content_hash = document.metadata.get("content_hash")
            if isinstance(content_hash, str):
                hashes.add(content_hash)
        return sources

    @staticmethod
    def _document_ids(documents: list[Document]) -> list[str]:
        ids: list[str] = []
        for chunk_number, document in enumerate(documents):
            source_id = str(document.metadata["source_id"])
            content_hash = str(document.metadata["content_hash"])
            digest = hashlib.sha256(
                f"{source_id}\0{content_hash}\0{chunk_number}".encode("utf-8")
            ).hexdigest()
            ids.append(digest)
        return ids

    def sync_documents(
        self,
        documents: Iterable[Document],
        *,
        active_source_ids: set[str] | None = None,
    ) -> int:
        """Add new sources and replace changed sources without duplicating chunks."""
        grouped: dict[str, list[Document]] = defaultdict(list)
        for document in documents:
            source_id = document.metadata.get("source_id")
            content_hash = document.metadata.get("content_hash")
            if not isinstance(source_id, str) or not source_id:
                raise VectorStoreError("Every document chunk needs a source_id.")
            if not isinstance(content_hash, str) or not content_hash:
                raise VectorStoreError(
                    f"Document chunks for '{source_id}' need a content_hash."
                )
            grouped[source_id].append(document)
        if active_source_ids is None:
            active_source_ids = set(grouped)
        elif not set(grouped).issubset(active_source_ids):
            raise VectorStoreError(
                "active_source_ids must include every source being indexed."
            )
        with self._lock:
            current_store = self._get_store()
            indexed_sources = (
                self._indexed_sources(current_store)
                if current_store is not None
                else {}
            )
            changed_sources: list[tuple[str, list[Document], list[str]]] = []
            for source_id, source_documents in sorted(grouped.items()):
                hashes = {
                    str(document.metadata["content_hash"])
                    for document in source_documents
                }
                if len(hashes) != 1:
                    raise VectorStoreError(
                        f"Document chunks for '{source_id}' have inconsistent hashes."
                    )

                old_ids: list[str] = []
                if source_id in indexed_sources:
                    old_ids, old_hashes = indexed_sources[source_id]
                    if old_hashes == hashes:
                        continue
                changed_sources.append(
                    (source_id, source_documents, old_ids)
                )

            stale_ids = [
                document_id
                for source_id, (document_ids, _) in indexed_sources.items()
                if source_id not in active_source_ids
                for document_id in document_ids
            ]
            if not changed_sources and not stale_ids:
                return 0

            candidate = self._load_persisted_store()
            total_added = 0
            try:
                for _, source_documents, old_ids in changed_sources:
                    if candidate is not None and old_ids:
                        candidate.delete(ids=old_ids)
                    document_ids = self._document_ids(source_documents)
                    if candidate is None:
                        candidate = FAISS.from_documents(
                            source_documents,
                            self._get_embeddings(),
                            ids=document_ids,
                        )
                    else:
                        candidate.add_documents(
                            source_documents,
                            ids=document_ids,
                        )
                    total_added += len(source_documents)

                if candidate is not None and stale_ids:
                    candidate.delete(ids=stale_ids)
                self.index_dir.mkdir(parents=True, exist_ok=True)
                if candidate is not None:
                    candidate.save_local(str(self.index_dir))
            except OpenAIError as error:
                raise VectorStoreError(
                    f"OpenAI embedding request failed: {error}"
                ) from error
            except (OSError, ValueError, RuntimeError) as error:
                raise VectorStoreError(
                    f"Could not update the FAISS index: {error}"
                ) from error

            self._vector_store = candidate
            self._loaded = True
            return total_added

    def similarity_search(self, question: str, *, k: int = 4) -> list[Document]:
        if not question.strip():
            raise ValueError("The search question cannot be empty.")
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        with self._lock:
            vector_store = self._get_store()
            if vector_store is None:
                raise VectorStoreError(
                    "The knowledge base has no indexed documents yet. "
                    "Upload a PDF/TXT file or add one under data/."
                )
            if not vector_store.index_to_docstore_id:
                raise VectorStoreError(
                    "The knowledge base has no indexed documents. "
                    "Upload a PDF/TXT file or add one under data/."
                )
            try:
                return vector_store.similarity_search(question, k=k)
            except OpenAIError as error:
                raise VectorStoreError(
                    f"OpenAI query-embedding request failed: {error}"
                ) from error
            except (ValueError, RuntimeError) as error:
                raise VectorStoreError(
                    f"Could not search the FAISS index: {error}"
                ) from error

    def similarity_search_with_score(
        self,
        question: str,
        *,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        """Return documents with raw FAISS squared-L2 distances (lower is better)."""
        if not question.strip():
            raise ValueError("The search question cannot be empty.")
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        with self._lock:
            vector_store = self._get_store()
            if vector_store is None or not vector_store.index_to_docstore_id:
                raise VectorStoreError(
                    "The knowledge base has no indexed documents. "
                    "Upload a PDF/TXT file or add one under data/."
                )
            try:
                return vector_store.similarity_search_with_score(question, k=k)
            except OpenAIError as error:
                raise VectorStoreError(
                    f"OpenAI query-embedding request failed: {error}"
                ) from error
            except (ValueError, RuntimeError) as error:
                raise VectorStoreError(
                    f"Could not search the FAISS index: {error}"
                ) from error
