class KnowledgeBaseError(RuntimeError):
    """Raised when a knowledge-base source cannot be read or saved."""


class VectorStoreError(RuntimeError):
    """Raised when a FAISS index cannot be loaded, updated, or queried."""


class ChatbotError(RuntimeError):
    """Raised when the answer-generation service fails."""
