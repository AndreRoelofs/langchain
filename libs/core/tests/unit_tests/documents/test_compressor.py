"""Unit tests for BaseDocumentCompressor."""

from collections.abc import Sequence
from typing import override

import pytest

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document


class TestBaseDocumentCompressor:
    """Tests for BaseDocumentCompressor abstract base class."""

    def test_compressor_is_abstract(self) -> None:
        """Test that BaseDocumentCompressor cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseDocumentCompressor()  # type: ignore[abstract]

    def test_concrete_compressor_implementation(self) -> None:
        """Test implementing a concrete compressor."""

        class SimpleCompressor(BaseDocumentCompressor):
            """A simple compressor that filters based on page content length."""

            @override
            def compress_documents(
                self,
                documents: Sequence[Document],
                query: str,
                callbacks: Callbacks | None = None,
            ) -> Sequence[Document]:
                """Keep only documents with content longer than query length."""
                threshold = len(query)
                return [doc for doc in documents if len(doc.page_content) > threshold]

        compressor = SimpleCompressor()
        docs = [
            Document(page_content="short"),
            Document(page_content="this is a longer document"),
            Document(page_content="mid"),
        ]

        result = compressor.compress_documents(docs, query="test query")
        assert len(result) == 1
        assert result[0].page_content == "this is a longer document"

    def test_compressor_with_metadata_preservation(self) -> None:
        """Test that compressor preserves document metadata."""

        class MetadataPreservingCompressor(BaseDocumentCompressor):
            """Compressor that keeps first N documents."""

            @override
            def compress_documents(
                self,
                documents: Sequence[Document],
                query: str,
                callbacks: Callbacks | None = None,
            ) -> Sequence[Document]:
                """Keep first 2 documents."""
                return documents[:2]

        compressor = MetadataPreservingCompressor()
        docs = [
            Document(page_content="doc1", metadata={"id": 1, "source": "a"}),
            Document(page_content="doc2", metadata={"id": 2, "source": "b"}),
            Document(page_content="doc3", metadata={"id": 3, "source": "c"}),
        ]

        result = compressor.compress_documents(docs, query="test")
        assert len(result) == 2
        assert result[0].metadata == {"id": 1, "source": "a"}
        assert result[1].metadata == {"id": 2, "source": "b"}

    def test_compressor_with_query_context(self) -> None:
        """Test that compressor can use query context."""

        class QueryAwareCompressor(BaseDocumentCompressor):
            """Compressor that filters based on query keyword."""

            @override
            def compress_documents(
                self,
                documents: Sequence[Document],
                query: str,
                callbacks: Callbacks | None = None,
            ) -> Sequence[Document]:
                """Keep documents containing query terms."""
                return [
                    doc
                    for doc in documents
                    if query.lower() in doc.page_content.lower()
                ]

        compressor = QueryAwareCompressor()
        docs = [
            Document(page_content="Python is great"),
            Document(page_content="JavaScript is popular"),
            Document(page_content="Python programming"),
        ]

        result = compressor.compress_documents(docs, query="python")
        assert len(result) == 2
        assert all("python" in doc.page_content.lower() for doc in result)

    def test_compressor_returns_empty_sequence(self) -> None:
        """Test compressor can return empty results."""

        class StrictCompressor(BaseDocumentCompressor):
            """Compressor that only keeps exact matches."""

            @override
            def compress_documents(
                self,
                documents: Sequence[Document],
                query: str,
                callbacks: Callbacks | None = None,
            ) -> Sequence[Document]:
                """Keep only exact content matches."""
                return [doc for doc in documents if doc.page_content == query]

        compressor = StrictCompressor()
        docs = [
            Document(page_content="foo"),
            Document(page_content="bar"),
        ]

        result = compressor.compress_documents(docs, query="baz")
        assert len(result) == 0

    def test_compressor_reorders_documents(self) -> None:
        """Test compressor can reorder documents (e.g., for re-ranking)."""

        class ReverseCompressor(BaseDocumentCompressor):
            """Compressor that reverses document order."""

            @override
            def compress_documents(
                self,
                documents: Sequence[Document],
                query: str,
                callbacks: Callbacks | None = None,
            ) -> Sequence[Document]:
                """Reverse document order."""
                return list(reversed(documents))

        compressor = ReverseCompressor()
        docs = [
            Document(page_content="first"),
            Document(page_content="second"),
            Document(page_content="third"),
        ]

        result = compressor.compress_documents(docs, query="")
        assert result[0].page_content == "third"
        assert result[1].page_content == "second"
        assert result[2].page_content == "first"

    async def test_acompress_documents_default_implementation(self) -> None:
        """Test default async implementation delegates to sync method."""

        class SyncCompressor(BaseDocumentCompressor):
            """Compressor with only sync implementation."""

            @override
            def compress_documents(
                self,
                documents: Sequence[Document],
                query: str,
                callbacks: Callbacks | None = None,
            ) -> Sequence[Document]:
                """Keep first document."""
                return documents[:1]

        compressor = SyncCompressor()
        docs = [
            Document(page_content="first"),
            Document(page_content="second"),
        ]

        # Default async should delegate to sync
        result = await compressor.acompress_documents(docs, query="test")
        assert len(result) == 1
        assert result[0].page_content == "first"

    async def test_acompress_documents_custom_implementation(self) -> None:
        """Test custom async implementation."""

        class AsyncCompressor(BaseDocumentCompressor):
            """Compressor with custom async implementation."""

            @override
            def compress_documents(
                self,
                documents: Sequence[Document],
                query: str,
                callbacks: Callbacks | None = None,
            ) -> Sequence[Document]:
                """Sync version - should not be called."""
                msg = "Sync method should not be called"
                raise RuntimeError(msg)

            @override
            async def acompress_documents(
                self,
                documents: Sequence[Document],
                query: str,
                callbacks: Callbacks | None = None,
            ) -> Sequence[Document]:
                """Custom async implementation."""
                return [doc for doc in documents if len(doc.page_content) > 5]

        compressor = AsyncCompressor()
        docs = [
            Document(page_content="hi"),
            Document(page_content="hello world"),
        ]

        result = await compressor.acompress_documents(docs, query="test")
        assert len(result) == 1
        assert result[0].page_content == "hello world"

    def test_compressor_with_callbacks_parameter(self) -> None:
        """Test that compressor accepts callbacks parameter."""

        class CallbackAwareCompressor(BaseDocumentCompressor):
            """Compressor that accepts but doesn't use callbacks."""

            @override
            def compress_documents(
                self,
                documents: Sequence[Document],
                query: str,
                callbacks: Callbacks | None = None,
            ) -> Sequence[Document]:
                """Simple passthrough."""
                # Callbacks can be None or a list of callback handlers
                return documents

        compressor = CallbackAwareCompressor()
        docs = [Document(page_content="test")]

        # Should work with None callbacks
        result = compressor.compress_documents(docs, query="test", callbacks=None)
        assert len(result) == 1

        # Should work without callbacks parameter (defaults to None)
        result = compressor.compress_documents(docs, query="test")
        assert len(result) == 1

    def test_compressor_preserves_document_ids(self) -> None:
        """Test that compressor preserves document IDs."""

        class IdPreservingCompressor(BaseDocumentCompressor):
            """Compressor that keeps documents with IDs."""

            @override
            def compress_documents(
                self,
                documents: Sequence[Document],
                query: str,
                callbacks: Callbacks | None = None,
            ) -> Sequence[Document]:
                """Keep documents that have an ID."""
                return [doc for doc in documents if doc.id is not None]

        compressor = IdPreservingCompressor()
        docs = [
            Document(page_content="doc1", id="1"),
            Document(page_content="doc2"),
            Document(page_content="doc3", id="3"),
        ]

        result = compressor.compress_documents(docs, query="test")
        assert len(result) == 2
        assert result[0].id == "1"
        assert result[1].id == "3"

    def test_compressor_returns_same_type_as_input(self) -> None:
        """Test that compressor returns Sequence[Document]."""

        class TypeCheckCompressor(BaseDocumentCompressor):
            """Compressor for type checking."""

            @override
            def compress_documents(
                self,
                documents: Sequence[Document],
                query: str,
                callbacks: Callbacks | None = None,
            ) -> Sequence[Document]:
                """Return as list."""
                return list(documents)

        compressor = TypeCheckCompressor()
        docs = [Document(page_content="test")]

        result = compressor.compress_documents(docs, query="test")
        assert isinstance(result, Sequence)
        assert all(isinstance(doc, Document) for doc in result)
