from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pypdf.errors import PdfReadError

from src.rag_errors import KnowledgeBaseError


def load_pdf(
    file_path: Path | str,
    *,
    source: str | None = None,
    source_id: str | None = None,
    content_hash: str | None = None,
) -> list[Document]:
    path = Path(file_path)
    display_name = source or path.name
    try:
        documents = PyPDFLoader(str(path)).load()
    except (OSError, PdfReadError, ValueError) as error:
        raise KnowledgeBaseError(
            f"Could not read PDF '{display_name}': {error}"
        ) from error

    pages: list[Document] = []
    for document in documents:
        text = document.page_content.strip()
        if not text:
            continue
        metadata = {
            "source": display_name,
            "page": int(document.metadata.get("page", 0)) + 1,
        }
        if source_id is not None:
            metadata["source_id"] = source_id
        if content_hash is not None:
            metadata["content_hash"] = content_hash
        pages.append(Document(page_content=text, metadata=metadata))

    if not pages:
        raise KnowledgeBaseError(
            f"PDF '{display_name}' contains no extractable text. "
            "Scanned PDFs need OCR before they can be indexed."
        )
    return pages
