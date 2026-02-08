"""Unit tests for BaseDocumentTransformer."""

from abc import ABC
from collections.abc import Sequence
from typing import Any, override

import pytest
from pydantic import BaseModel

from langchain_core.documents import BaseDocumentTransformer, Document


class TestBaseDocumentTransformer:
    """Tests for BaseDocumentTransformer abstract base class."""

    def test_transformer_is_abstract(self) -> None:
        """Test that BaseDocumentTransformer cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseDocumentTransformer()  # type: ignore[abstract]

    def test_concrete_transformer_implementation(self) -> None:
        """Test implementing a concrete transformer."""

        class UpperCaseTransformer(BaseDocumentTransformer):
            """Transform documents to uppercase."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Convert all page content to uppercase."""
                return [
                    Document(
                        page_content=doc.page_content.upper(),
                        metadata=doc.metadata,
                        id=doc.id,
                    )
                    for doc in documents
                ]

        transformer = UpperCaseTransformer()
        docs = [
            Document(page_content="hello world"),
            Document(page_content="foo bar"),
        ]

        result = transformer.transform_documents(docs)
        assert len(result) == 2
        assert result[0].page_content == "HELLO WORLD"
        assert result[1].page_content == "FOO BAR"

    def test_transformer_preserves_metadata(self) -> None:
        """Test that transformer can preserve document metadata."""

        class TruncateTransformer(BaseDocumentTransformer):
            """Truncate document content."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Truncate content to 10 characters."""
                return [
                    Document(
                        page_content=doc.page_content[:10],
                        metadata=doc.metadata,
                        id=doc.id,
                    )
                    for doc in documents
                ]

        transformer = TruncateTransformer()
        docs = [
            Document(
                page_content="This is a long document that will be truncated",
                metadata={"source": "test", "page": 1},
                id="doc1",
            )
        ]

        result = transformer.transform_documents(docs)
        assert result[0].page_content == "This is a "
        assert result[0].metadata == {"source": "test", "page": 1}
        assert result[0].id == "doc1"

    def test_transformer_can_split_documents(self) -> None:
        """Test transformer that splits documents into multiple."""

        class SplitByLineTransformer(BaseDocumentTransformer):
            """Split documents by lines."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Split each document by lines."""
                result = []
                for doc in documents:
                    lines = doc.page_content.split("\n")
                    for i, line in enumerate(lines):
                        if line.strip():
                            result.append(
                                Document(
                                    page_content=line,
                                    metadata={
                                        **doc.metadata,
                                        "line_number": i,
                                        "parent_id": doc.id,
                                    },
                                )
                            )
                return result

        transformer = SplitByLineTransformer()
        docs = [Document(page_content="Line 1\nLine 2\nLine 3", id="parent")]

        result = transformer.transform_documents(docs)
        assert len(result) == 3
        assert result[0].page_content == "Line 1"
        assert result[1].page_content == "Line 2"
        assert result[2].page_content == "Line 3"
        assert result[0].metadata["parent_id"] == "parent"

    def test_transformer_can_filter_documents(self) -> None:
        """Test transformer that filters out documents."""

        class MinLengthTransformer(BaseDocumentTransformer):
            """Filter documents by minimum length."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Keep only documents with content length > 10."""
                return [doc for doc in documents if len(doc.page_content) > 10]

        transformer = MinLengthTransformer()
        docs = [
            Document(page_content="short"),
            Document(page_content="this is long enough"),
            Document(page_content="hi"),
        ]

        result = transformer.transform_documents(docs)
        assert len(result) == 1
        assert result[0].page_content == "this is long enough"

    def test_transformer_with_kwargs(self) -> None:
        """Test transformer that uses keyword arguments."""

        class PrefixTransformer(BaseDocumentTransformer):
            """Add prefix to document content."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Add prefix from kwargs."""
                prefix = kwargs.get("prefix", "")
                return [
                    Document(
                        page_content=f"{prefix}{doc.page_content}",
                        metadata=doc.metadata,
                        id=doc.id,
                    )
                    for doc in documents
                ]

        transformer = PrefixTransformer()
        docs = [Document(page_content="content")]

        result = transformer.transform_documents(docs, prefix="PREFIX: ")
        assert result[0].page_content == "PREFIX: content"

    def test_transformer_returns_empty_sequence(self) -> None:
        """Test transformer can return empty results."""

        class EmptyTransformer(BaseDocumentTransformer):
            """Transformer that filters out all documents."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Return empty list."""
                return []

        transformer = EmptyTransformer()
        docs = [Document(page_content="test1"), Document(page_content="test2")]

        result = transformer.transform_documents(docs)
        assert len(result) == 0

    def test_transformer_modifies_metadata(self) -> None:
        """Test transformer that adds metadata."""

        class MetadataEnricherTransformer(BaseDocumentTransformer):
            """Add metadata to documents."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Add character count to metadata."""
                return [
                    Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "char_count": len(doc.page_content),
                            "word_count": len(doc.page_content.split()),
                        },
                        id=doc.id,
                    )
                    for doc in documents
                ]

        transformer = MetadataEnricherTransformer()
        docs = [Document(page_content="Hello world", metadata={"source": "test"})]

        result = transformer.transform_documents(docs)
        assert result[0].metadata["char_count"] == 11
        assert result[0].metadata["word_count"] == 2
        assert result[0].metadata["source"] == "test"

    async def test_atransform_documents_default_implementation(self) -> None:
        """Test default async implementation delegates to sync method."""

        class SyncTransformer(BaseDocumentTransformer):
            """Transformer with only sync implementation."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Reverse content."""
                return [
                    Document(
                        page_content=doc.page_content[::-1],
                        metadata=doc.metadata,
                        id=doc.id,
                    )
                    for doc in documents
                ]

        transformer = SyncTransformer()
        docs = [Document(page_content="hello")]

        # Default async should delegate to sync
        result = await transformer.atransform_documents(docs)
        assert len(result) == 1
        assert result[0].page_content == "olleh"

    async def test_atransform_documents_custom_implementation(self) -> None:
        """Test custom async implementation."""

        class AsyncTransformer(BaseDocumentTransformer):
            """Transformer with custom async implementation."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Sync version - should not be called."""
                msg = "Sync method should not be called"
                raise RuntimeError(msg)

            @override
            async def atransform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Custom async implementation."""
                return [
                    Document(
                        page_content=f"ASYNC: {doc.page_content}",
                        metadata=doc.metadata,
                        id=doc.id,
                    )
                    for doc in documents
                ]

        transformer = AsyncTransformer()
        docs = [Document(page_content="test")]

        result = await transformer.atransform_documents(docs)
        assert len(result) == 1
        assert result[0].page_content == "ASYNC: test"

    async def test_atransform_documents_with_kwargs(self) -> None:
        """Test async transform with kwargs."""

        class AsyncKwargsTransformer(BaseDocumentTransformer):
            """Transformer that uses kwargs in async."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Sync implementation."""
                suffix = kwargs.get("suffix", "")
                return [
                    Document(page_content=f"{doc.page_content}{suffix}")
                    for doc in documents
                ]

        transformer = AsyncKwargsTransformer()
        docs = [Document(page_content="test")]

        result = await transformer.atransform_documents(docs, suffix="!")
        assert result[0].page_content == "test!"

    def test_transformer_chain(self) -> None:
        """Test chaining multiple transformers."""

        class UpperCaseTransformer(BaseDocumentTransformer):
            """Transform to uppercase."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                return [
                    Document(page_content=doc.page_content.upper()) for doc in documents
                ]

        class PrefixTransformer(BaseDocumentTransformer):
            """Add prefix."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                return [
                    Document(page_content=f">> {doc.page_content}") for doc in documents
                ]

        docs = [Document(page_content="hello")]

        # Apply transformers in sequence
        transformer1 = UpperCaseTransformer()
        transformer2 = PrefixTransformer()

        result = transformer1.transform_documents(docs)
        result = transformer2.transform_documents(result)

        assert result[0].page_content == ">> HELLO"

    def test_transformer_preserves_document_id(self) -> None:
        """Test that transformer preserves document IDs."""

        class IdentityTransformer(BaseDocumentTransformer):
            """Pass through transformer."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Return documents as-is."""
                return list(documents)

        transformer = IdentityTransformer()
        docs = [
            Document(page_content="test1", id="id1"),
            Document(page_content="test2", id="id2"),
        ]

        result = transformer.transform_documents(docs)
        assert result[0].id == "id1"
        assert result[1].id == "id2"

    def test_transformer_returns_same_type_as_input(self) -> None:
        """Test that transformer returns Sequence[Document]."""

        class TypeCheckTransformer(BaseDocumentTransformer):
            """Transformer for type checking."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Return as list."""
                return list(documents)

        transformer = TypeCheckTransformer()
        docs = [Document(page_content="test")]

        result = transformer.transform_documents(docs)
        assert isinstance(result, Sequence)
        assert all(isinstance(doc, Document) for doc in result)

    def test_transformer_is_abc_subclass(self) -> None:
        """Test that BaseDocumentTransformer is an ABC subclass."""
        assert issubclass(BaseDocumentTransformer, ABC)

    def test_transformer_is_not_base_model(self) -> None:
        """Test that BaseDocumentTransformer is NOT a BaseModel subclass."""
        assert not issubclass(BaseDocumentTransformer, BaseModel)

    def test_transformer_with_empty_input(self) -> None:
        """Test transformer with empty document list."""

        class PassthroughTransformer(BaseDocumentTransformer):
            """Transformer that passes through all documents."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Return documents as-is."""
                return list(documents)

        transformer = PassthroughTransformer()
        result = transformer.transform_documents([])
        assert len(result) == 0
        assert isinstance(result, list)

    def test_transformer_with_tuple_input(self) -> None:
        """Test transformer accepts tuple as Sequence input."""

        class PassthroughTransformer(BaseDocumentTransformer):
            """Transformer that passes through all documents."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Return documents as-is."""
                return list(documents)

        transformer = PassthroughTransformer()
        docs = (
            Document(page_content="doc1"),
            Document(page_content="doc2"),
        )
        result = transformer.transform_documents(docs)
        assert len(result) == 2
        assert result[0].page_content == "doc1"

    def test_transformer_with_no_kwargs(self) -> None:
        """Test transformer called without any kwargs."""

        class NoKwargsTransformer(BaseDocumentTransformer):
            """Transformer that checks kwargs are empty."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Verify no kwargs passed."""
                assert len(kwargs) == 0
                return list(documents)

        transformer = NoKwargsTransformer()
        docs = [Document(page_content="test")]
        result = transformer.transform_documents(docs)
        assert len(result) == 1

    def test_transformer_with_multiple_kwargs(self) -> None:
        """Test transformer with multiple keyword arguments."""

        class MultiKwargsTransformer(BaseDocumentTransformer):
            """Transformer that uses multiple kwargs."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Apply prefix and suffix from kwargs."""
                prefix = kwargs.get("prefix", "")
                suffix = kwargs.get("suffix", "")
                return [
                    Document(
                        page_content=f"{prefix}{doc.page_content}{suffix}",
                        metadata=doc.metadata,
                        id=doc.id,
                    )
                    for doc in documents
                ]

        transformer = MultiKwargsTransformer()
        docs = [Document(page_content="content")]
        result = transformer.transform_documents(docs, prefix="[", suffix="]")
        assert result[0].page_content == "[content]"

    async def test_atransform_documents_with_empty_input(self) -> None:
        """Test async transform with empty document list."""

        class PassthroughTransformer(BaseDocumentTransformer):
            """Transformer that passes through all documents."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Return documents as-is."""
                return list(documents)

        transformer = PassthroughTransformer()
        result = await transformer.atransform_documents([])
        assert len(result) == 0

    def test_transformer_single_document(self) -> None:
        """Test transformer with a single document."""

        class DoubleTransformer(BaseDocumentTransformer):
            """Transformer that doubles content."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Double the page content."""
                return [
                    Document(
                        page_content=doc.page_content * 2,
                        metadata=doc.metadata,
                        id=doc.id,
                    )
                    for doc in documents
                ]

        transformer = DoubleTransformer()
        docs = [Document(page_content="ab", id="1")]
        result = transformer.transform_documents(docs)
        assert result[0].page_content == "abab"
        assert result[0].id == "1"

    def test_transformer_can_produce_more_documents(self) -> None:
        """Test transformer that produces more documents than input."""

        class DuplicateTransformer(BaseDocumentTransformer):
            """Transformer that duplicates each document."""

            @override
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                """Duplicate each document."""
                result = []
                for doc in documents:
                    result.append(doc)
                    result.append(
                        Document(
                            page_content=doc.page_content,
                            metadata={**doc.metadata, "copy": True},
                            id=f"{doc.id}-copy" if doc.id else None,
                        )
                    )
                return result

        transformer = DuplicateTransformer()
        docs = [
            Document(page_content="a", id="1"),
            Document(page_content="b", id="2"),
        ]
        result = transformer.transform_documents(docs)
        assert len(result) == 4
        assert result[0].id == "1"
        assert result[1].id == "1-copy"
        assert result[1].metadata["copy"] is True
        assert result[2].id == "2"
        assert result[3].id == "2-copy"
