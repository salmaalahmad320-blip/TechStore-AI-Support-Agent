import hashlib
import re
from langchain_core.documents import Document

from src.vector_store import PersistentVectorStore


DEFAULT_RETRIEVAL_COUNT = 4
RETRIEVAL_OVERFETCH_FACTOR = 3
DEFAULT_MAX_CITATIONS = 3
# OpenAI embeddings are unit-normalized. For FAISS squared-L2 distance d,
# cosine similarity is 1 - d/2; 1.50 therefore requires cosine >= 0.25.
DEFAULT_MAX_DISTANCE = 1.50


def _normalized_content(document: Document) -> str:
    return re.sub(r"\s+", " ", document.page_content).strip().casefold()


def _source_identity(document: Document) -> str:
    source_id = document.metadata.get("source_id")
    source = source_id or document.metadata.get("source", "Unknown source")
    return str(source).replace("\\", "/").casefold()


def _page_identity(document: Document) -> str:
    page = document.metadata.get("page")
    return "" if page is None else str(page)


def _unique_relevant_documents(
    documents: list[Document],
    *,
    limit: int,
) -> list[Document]:
    """Keep the highest-ranked copy of repeated evidence without reordering results."""
    unique_documents: list[Document] = []
    seen_chunks: set[str] = set()

    for document in documents:
        normalized_content = _normalized_content(document)
        if not normalized_content:
            continue
        content_hash = hashlib.sha256(
            normalized_content.encode("utf-8")
        ).hexdigest()
        if content_hash in seen_chunks:
            continue
        seen_chunks.add(content_hash)
        unique_documents.append(document)
        if len(unique_documents) == limit:
            break

    return unique_documents


def retrieve_information(
    vector_store: PersistentVectorStore,
    question: str,
    *,
    k: int = DEFAULT_RETRIEVAL_COUNT,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> list[Document]:
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if max_distance < 0:
        raise ValueError("max_distance cannot be negative.")

    scored_candidates = vector_store.similarity_search_with_score(
        question,
        k=k * RETRIEVAL_OVERFETCH_FACTOR,
    )
    candidates = [
        document
        for document, distance in scored_candidates
        if distance <= max_distance
    ]
    return _unique_relevant_documents(candidates, limit=k)


def format_source_citations(
    documents: list[Document],
    *,
    max_citations: int = DEFAULT_MAX_CITATIONS,
) -> list[str]:
    return [
        citation
        for citation, _ in select_source_documents(
            documents,
            max_citations=max_citations,
        )
    ]


def select_source_documents(
    documents: list[Document],
    *,
    max_citations: int = DEFAULT_MAX_CITATIONS,
) -> list[tuple[str, Document]]:
    """Select distinct, ranked source evidence and its display citation."""
    if max_citations < 0:
        raise ValueError("max_citations cannot be negative.")
    if max_citations == 0:
        return []

    selected: list[tuple[str, Document]] = []
    seen_citations: set[tuple[str, str]] = set()
    seen_content: dict[str, str] = {}

    for document in documents:
        raw_source = str(document.metadata.get("source", "Unknown source"))
        source = raw_source.replace("\\", "/").rsplit("/", 1)[-1]
        if not source:
            source = "Unknown source"
        page = document.metadata.get("page")
        citation = f"{source}, page {page}" if page is not None else source
        citation_key = (source.casefold(), _page_identity(document))
        if citation_key in seen_citations:
            continue

        normalized_content = _normalized_content(document)
        content_hash = hashlib.sha256(
            normalized_content.encode("utf-8")
        ).hexdigest()
        previous_source = seen_content.get(content_hash)
        if previous_source is not None and previous_source != _source_identity(document):
            continue

        seen_citations.add(citation_key)
        seen_content[content_hash] = _source_identity(document)
        selected.append((citation, document))
        if len(selected) == max_citations:
            break

    return selected
