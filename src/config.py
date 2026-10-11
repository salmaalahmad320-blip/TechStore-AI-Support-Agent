import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

_configured_index_dir = Path(os.getenv("FAISS_INDEX_DIR", ".faiss_index"))
VECTOR_STORE_DIR = (
    _configured_index_dir
    if _configured_index_dir.is_absolute()
    else PROJECT_ROOT / _configured_index_dir
)

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)
