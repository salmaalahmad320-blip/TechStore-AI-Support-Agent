import hashlib
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.rag import load_knowledge_base, save_uploaded_files, split_knowledge
from src.retriever import (
    format_source_citations,
    retrieve_information,
    select_source_documents,
)
from src.vector_store import PersistentVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class UploadedFileStub:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


class KeywordEmbeddings(Embeddings):
    KEYWORDS = (
        "warranty",
        "policy",
        "shipping",
        "returns",
        "payment",
        "selected countries",
        "cars",
        "airplanes",
    )

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        normalized = text.casefold()
        values = [float(normalized.count(word)) for word in cls.KEYWORDS]
        magnitude = math.sqrt(sum(value * value for value in values))
        return [value / magnitude for value in values] if magnitude else values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def make_document(
    text: str,
    *,
    source: str,
    source_id: str,
    page: int | None = None,
) -> Document:
    metadata: dict[str, str | int] = {
        "source": source,
        "source_id": source_id,
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    if page is not None:
        metadata["page"] = page
    return Document(page_content=text, metadata=metadata)


class KnowledgeBaseTests(unittest.TestCase):
    def test_loads_multiple_text_sources_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            data_dir = Path(temporary_dir)
            (data_dir / "company.txt").write_text(
                "Returns are accepted within fourteen days.",
                encoding="utf-8",
            )
            (data_dir / "policy.TXT").write_text(
                "The warranty lasts two years.",
                encoding="utf-8",
            )

            documents = load_knowledge_base(data_dir)

        self.assertEqual(
            {"company.txt", "policy.TXT"},
            {document.metadata["source"] for document in documents},
        )
        self.assertTrue(all(document.metadata.get("content_hash") for document in documents))
        self.assertFalse(any("page" in document.metadata for document in documents))

    def test_reads_bundled_pdf_with_one_based_page_number(self) -> None:
        documents = load_knowledge_base(PROJECT_ROOT / "data")
        pdf_pages = [
            document
            for document in documents
            if document.metadata.get("source") == "uploaded_company.pdf"
        ]

        self.assertTrue(pdf_pages)
        self.assertEqual(1, pdf_pages[0].metadata["page"])

    def test_uploads_multiple_documents_without_using_path_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            paths = save_uploaded_files(
                [
                    UploadedFileStub("policies/warranty.txt", b"Warranty terms."),
                    UploadedFileStub("shipping.txt", b"Shipping terms."),
                ],
                temporary_dir,
            )

            self.assertEqual(
                {"warranty.txt", "shipping.txt"},
                {path.name for path in paths},
            )
            self.assertTrue(all(path.parent == Path(temporary_dir) for path in paths))

    def test_splitter_preserves_pdf_page_metadata(self) -> None:
        source = make_document(
            "warranty " * 100,
            source="uploaded_company.pdf",
            source_id="uploads/uploaded_company.pdf",
            page=1,
        )

        chunks = split_knowledge([source], chunk_size=100, chunk_overlap=10)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(
                chunk.metadata["source"] == "uploaded_company.pdf"
                and chunk.metadata["page"] == 1
                for chunk in chunks
            )
        )


class PersistentVectorStoreTests(unittest.TestCase):
    def test_retrieves_the_bundled_pdf_after_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            chunks = split_knowledge(load_knowledge_base(PROJECT_ROOT / "data"))
            store = PersistentVectorStore(
                temporary_dir,
                embeddings=KeywordEmbeddings(),
            )

            store.sync_documents(chunks)
            results = retrieve_information(
                store,
                "What is the warranty policy?"
            )

        self.assertEqual("uploaded_company.pdf", results[0].metadata["source"])
        self.assertEqual(1, results[0].metadata["page"])
        self.assertIn("Warranty:", results[0].page_content)
        self.assertEqual(
            ["uploaded_company.pdf, page 1"],
            format_source_citations(results),
        )

    def test_unrelated_query_has_no_results_from_the_knowledge_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            chunks = split_knowledge(load_knowledge_base(PROJECT_ROOT / "data"))
            store = PersistentVectorStore(
                temporary_dir,
                embeddings=KeywordEmbeddings(),
            )
            store.sync_documents(chunks)

            results = retrieve_information(store, "Do you sell cars?")

        self.assertEqual([], results)

    def test_supported_customer_questions_retrieve_citable_passages(self) -> None:
        expected_terms = {
            "What is the warranty policy?": "warranty",
            "How long does shipping take?": "3-5 business days",
            "What payment methods do you accept?": "PayPal",
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            chunks = split_knowledge(load_knowledge_base(PROJECT_ROOT / "data"))
            store = PersistentVectorStore(
                temporary_dir,
                embeddings=KeywordEmbeddings(),
            )
            store.sync_documents(chunks)

            for question, expected_term in expected_terms.items():
                with self.subTest(question=question):
                    results = retrieve_information(store, question)

                    self.assertTrue(results)
                    self.assertTrue(
                        any(
                            expected_term.casefold() in document.page_content.casefold()
                            for document in results
                        ),
                        f"Expected evidence containing {expected_term!r}",
                    )
                    self.assertTrue(format_source_citations(results))

    def test_cars_and_airplanes_questions_return_no_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            chunks = split_knowledge(load_knowledge_base(PROJECT_ROOT / "data"))
            store = PersistentVectorStore(
                temporary_dir,
                embeddings=KeywordEmbeddings(),
            )
            store.sync_documents(chunks)

            for question in ("Do you sell cars?", "Do you sell airplanes?"):
                with self.subTest(question=question):
                    self.assertEqual([], retrieve_information(store, question))

    def test_retrieves_pdf_source_and_reloads_persisted_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            index_dir = Path(temporary_dir) / "index"
            documents = [
                make_document(
                    "Shipping takes three to five business days.",
                    source="company_knowledge.txt",
                    source_id="company_knowledge.txt",
                ),
                make_document(
                    "The warranty policy covers defects for two years.",
                    source="uploaded_company.pdf",
                    source_id="uploads/uploaded_company.pdf",
                    page=1,
                ),
            ]

            first_store = PersistentVectorStore(
                index_dir,
                embeddings=KeywordEmbeddings(),
            )
            self.assertEqual(2, first_store.sync_documents(documents))
            results = first_store.similarity_search("What is the warranty policy?")
            self.assertEqual("uploaded_company.pdf", results[0].metadata["source"])
            self.assertEqual(["uploaded_company.pdf, page 1"], format_source_citations(results[:1]))

            reloaded_store = PersistentVectorStore(
                index_dir,
                embeddings=KeywordEmbeddings(),
            )
            reloaded_results = reloaded_store.similarity_search(
                "What is the warranty policy?"
            )

            self.assertEqual(
                "uploaded_company.pdf",
                reloaded_results[0].metadata["source"],
            )
            self.assertEqual(0, reloaded_store.sync_documents(documents))

    def test_scored_search_exposes_faiss_distance_for_relevant_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            store = PersistentVectorStore(
                temporary_dir,
                embeddings=KeywordEmbeddings(),
            )
            documents = [
                make_document(
                    "The warranty policy covers defects for two years.",
                    source="uploaded_company.pdf",
                    source_id="uploads/uploaded_company.pdf",
                    page=1,
                ),
                make_document(
                    "Orders ship within five business days.",
                    source="company_knowledge.txt",
                    source_id="company_knowledge.txt",
                ),
            ]
            store.sync_documents(documents)

            results = store.similarity_search_with_score(
                "What is the warranty policy?",
                k=2,
            )

        self.assertEqual("uploaded_company.pdf", results[0][0].metadata["source"])
        self.assertLess(results[0][1], results[1][1])

    def test_changed_source_replaces_old_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            store = PersistentVectorStore(
                temporary_dir,
                embeddings=KeywordEmbeddings(),
            )
            source_id = "uploads/policy.txt"
            original = make_document(
                "Warranty lasts for one year.",
                source="policy.txt",
                source_id=source_id,
            )
            replacement = make_document(
                "Warranty lasts for two years.",
                source="policy.txt",
                source_id=source_id,
            )

            store.sync_documents([original])
            self.assertEqual(1, store.sync_documents([replacement]))
            self.assertEqual(1, len(store._vector_store.index_to_docstore_id))

            result = store.similarity_search("warranty policy")
            self.assertIn("two years", result[0].page_content)

    def test_removed_sources_are_pruned_from_the_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            store = PersistentVectorStore(
                temporary_dir,
                embeddings=KeywordEmbeddings(),
            )
            company_doc = make_document(
                "Shipping is available worldwide.",
                source="company.txt",
                source_id="company.txt",
            )
            upload_doc = make_document(
                "Warranty lasts for two years.",
                source="policy.txt",
                source_id="uploads/policy.txt",
            )
            store.sync_documents(
                [company_doc, upload_doc],
                active_source_ids={"company.txt", "uploads/policy.txt"},
            )

            store.sync_documents(
                [company_doc],
                active_source_ids={"company.txt"},
            )

            self.assertEqual(1, len(store._vector_store.index_to_docstore_id))


class RetrievalCitationTests(unittest.TestCase):
    def test_retrieval_overfetches_and_keeps_best_unique_evidence(self) -> None:
        repeated_txt = make_document(
            "The warranty covers manufacturing defects for two years.",
            source="company_knowledge.txt",
            source_id="company_knowledge.txt",
        )
        repeated_pdf = make_document(
            "  THE WARRANTY covers manufacturing defects for two years. ",
            source="TechStore_Company_Info.pdf",
            source_id="uploads/TechStore_Company_Info.pdf",
            page=1,
        )
        useful_pdf = make_document(
            "The warranty policy applies from the purchase date.",
            source="uploaded_company.pdf",
            source_id="uploads/uploaded_company.pdf",
            page=1,
        )
        vector_store = Mock()
        vector_store.similarity_search_with_score.return_value = [
            (repeated_pdf, 0.20),
            (repeated_txt, 0.25),
            (useful_pdf, 0.55),
        ]

        results = retrieve_information(vector_store, "warranty policy", k=2)

        vector_store.similarity_search_with_score.assert_called_once_with(
            "warranty policy",
            k=6,
        )
        self.assertEqual([repeated_pdf, useful_pdf], results)

    def test_unrelated_results_are_filtered_by_distance_threshold(self) -> None:
        car_document = make_document(
            "The company warranty covers electronic products.",
            source="uploaded_company.pdf",
            source_id="uploaded_company.pdf",
            page=1,
        )
        vector_store = Mock()
        vector_store.similarity_search_with_score.return_value = [
            (car_document, 1.75),
            (car_document, 1.90),
        ]

        results = retrieve_information(vector_store, "Do you sell cars?")

        self.assertEqual([], results)

    def test_relevant_result_at_relaxed_boundary_is_kept(self) -> None:
        document = make_document(
            "Orders are delivered within 3-5 business days.",
            source="company_knowledge.txt",
            source_id="company_knowledge.txt",
        )
        vector_store = Mock()
        vector_store.similarity_search_with_score.return_value = [(document, 1.45)]

        results = retrieve_information(vector_store, "How long does shipping take?")

        self.assertEqual([document], results)

    def test_retrieval_rejects_negative_max_distance(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            retrieve_information(Mock(), "warranty", max_distance=-0.1)

    def test_citations_deduplicate_repeated_content_and_limit_to_top_three(self) -> None:
        documents = [
            make_document(
                "Returns are accepted within fourteen days.",
                source="policies/returns.pdf",
                source_id="policies/returns.pdf",
                page=1,
            ),
            make_document(
                "  RETURNS are accepted within fourteen days. ",
                source="returns-copy.pdf",
                source_id="uploads/returns-copy.pdf",
                page=1,
            ),
            make_document(
                "Warranty coverage lasts two years.",
                source="warranty.pdf",
                source_id="warranty.pdf",
                page=2,
            ),
            make_document(
                "Shipping takes five business days.",
                source="shipping.pdf",
                source_id="shipping.pdf",
                page=3,
            ),
            make_document(
                "Online payments are accepted.",
                source="payments.txt",
                source_id="payments.txt",
            ),
        ]

        citations = format_source_citations(documents)

        self.assertEqual(
            [
                "returns.pdf, page 1",
                "warranty.pdf, page 2",
                "shipping.pdf, page 3",
            ],
            citations,
        )
        self.assertEqual(
            citations,
            [citation for citation, _ in select_source_documents(documents)],
        )
        self.assertEqual(
            ["payments.txt"],
            format_source_citations(documents[-1:], max_citations=3),
        )
        self.assertEqual([], format_source_citations(documents, max_citations=0))

    def test_citation_limit_must_not_be_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            format_source_citations([], max_citations=-1)


if __name__ == "__main__":
    unittest.main()
