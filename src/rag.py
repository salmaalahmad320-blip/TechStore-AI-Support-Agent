import hashlib
import os
import tempfile
from pathlib import Path
from threading import RLock
from typing import Iterable, Protocol

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DATA_DIR, PROJECT_ROOT, UPLOAD_DIR
from src.pdf_loader import load_pdf
from src.rag_errors import KnowledgeBaseError
from src.retriever import retrieve_information
from src.vector_store import PersistentVectorStore


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class UploadedFile(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def _source_document(path: Path, data_dir: Path) -> list[Document]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise KnowledgeBaseError(f"Could not read '{path.name}': {error}") from error

    if not content:
        raise KnowledgeBaseError(f"'{path.name}' is empty.")

    content_hash = hashlib.sha256(content).hexdigest()
    try:
        source_id = path.relative_to(data_dir).as_posix()
    except ValueError:
        source_id = path.relative_to(PROJECT_ROOT).as_posix()

    if path.suffix.lower() == ".pdf":
        return load_pdf(
            path,
            source=path.name,
            source_id=source_id,
            content_hash=content_hash,
        )

    try:
        text = content.decode("utf-8-sig").strip()
    except UnicodeDecodeError as error:
        raise KnowledgeBaseError(
            f"'{path.name}' is not a valid UTF-8 text file."
        ) from error

    if not text:
        raise KnowledgeBaseError(f"'{path.name}' contains no readable text.")

    return [
        Document(
            page_content=text,
            metadata={
                "source": path.name,
                "source_id": source_id,
                "content_hash": content_hash,
            },
        )
    ]


def load_knowledge_base(data_dir: Path | str = DATA_DIR) -> list[Document]:
    """Load supported files under data_dir and the bundled root-level sample PDF."""
    root = Path(data_dir).resolve()
    if not root.exists():
        raise KnowledgeBaseError(f"Knowledge-base directory does not exist: {root}")
    if not root.is_dir():
        raise KnowledgeBaseError(f"Knowledge-base path is not a directory: {root}")

    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    sample_pdf = PROJECT_ROOT / "uploaded_company.pdf"
    if root == DATA_DIR.resolve() and sample_pdf.is_file():
        paths.append(sample_pdf)

    documents: list[Document] = []
    for path in paths:
        documents.extend(_source_document(path, root))
    return documents


def knowledge_base_signature(
    data_dir: Path | str = DATA_DIR,
) -> tuple[tuple[str, int, int], ...]:
    """Return a cheap file signature so unchanged sources are not re-parsed."""
    root = Path(data_dir).resolve()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    sample_pdf = PROJECT_ROOT / "uploaded_company.pdf"
    if root == DATA_DIR.resolve() and sample_pdf.is_file():
        paths.append(sample_pdf)

    try:
        return tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for path in paths
        )
    except OSError as error:
        raise KnowledgeBaseError(
            f"Could not inspect knowledge-base files: {error}"
        ) from error


def split_knowledge(
    documents: Iterable[Document],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split source documents while retaining their source and page metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(list(documents))


class RAGService:
    def __init__(
        self,
        vector_store: PersistentVectorStore | None = None,
        data_dir: Path | str = DATA_DIR,
    ) -> None:
        self.vector_store = vector_store or PersistentVectorStore()
        self.data_dir = Path(data_dir).resolve()
        self._source_signature: tuple[tuple[str, int, int], ...] | None = None
        self._sync_lock = RLock()

    def sync_knowledge_base(self) -> int:
        with self._sync_lock:
            signature = knowledge_base_signature(self.data_dir)
            if signature == self._source_signature:
                return 0

            source_documents = load_knowledge_base(self.data_dir)
            chunks = split_knowledge(source_documents)
            active_source_ids = {
                str(document.metadata["source_id"])
                for document in chunks
            }
            indexed_count = self.vector_store.sync_documents(
                chunks,
                active_source_ids=active_source_ids,
            )
            self._source_signature = signature
            return indexed_count

    def add_uploaded_files(
        self,
        uploaded_files: Iterable[UploadedFile],
    ) -> tuple[list[Path], int]:
        saved_paths = save_uploaded_files(uploaded_files, UPLOAD_DIR)
        if saved_paths:
            with self._sync_lock:
                self._source_signature = None
            indexed_count = self.sync_knowledge_base()
            return saved_paths, indexed_count
        return [], 0

    def retrieve(self, question: str) -> list[Document]:
        return retrieve_information(self.vector_store, question)


def save_uploaded_files(
    uploaded_files: Iterable[UploadedFile],
    upload_dir: Path | str = UPLOAD_DIR,
) -> list[Path]:
    """Persist selected PDF/TXT files safely, retaining their original filenames."""
    files = list(uploaded_files)
    if not files:
        return []

    target_dir = Path(upload_dir).resolve()
    filenames: set[str] = set()
    pending: list[tuple[str, bytes]] = []

    for uploaded_file in files:
        filename = Path(uploaded_file.name.replace("\\", "/")).name
        if not filename or filename in {".", ".."}:
            raise KnowledgeBaseError("An uploaded document has an invalid filename.")
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise KnowledgeBaseError(
                f"'{filename}' is not supported. Upload PDF or TXT files."
            )
        if filename.casefold() in filenames:
            raise KnowledgeBaseError(
                f"'{filename}' was selected more than once. Rename one copy and retry."
            )
        filenames.add(filename.casefold())

        content = uploaded_file.getvalue()
        if not content:
            raise KnowledgeBaseError(f"'{filename}' is empty.")
        pending.append((filename, content))

    saved_paths: list[Path] = []
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in pending:
            target_path = target_dir / filename
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=target_dir,
                    prefix=".upload-",
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    temporary_file.write(content)
                os.replace(temporary_path, target_path)
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()
            saved_paths.append(target_path)
    except OSError as error:
        raise KnowledgeBaseError(f"Could not save uploaded documents: {error}") from error

    return saved_paths
