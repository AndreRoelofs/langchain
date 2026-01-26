"""Test Base Schema of documents."""

from collections.abc import Iterator

import pytest
from typing_extensions import override

from langchain_core.document_loaders.base import BaseBlobParser, BaseLoader
from langchain_core.documents import Document
from langchain_core.documents.base import Blob


def test_base_blob_parser() -> None:
    """Verify that the eager method is hooked up to the lazy method by default."""

    class MyParser(BaseBlobParser):
        """A simple parser that returns a single document."""

        @override
        def lazy_parse(self, blob: Blob) -> Iterator[Document]:
            """Lazy parsing interface."""
            yield Document(
                page_content="foo",
            )

    parser = MyParser()

    assert isinstance(parser.lazy_parse(Blob(data="who?")), Iterator)

    # We're verifying that the eager method is hooked up to the lazy method by default.
    docs = parser.parse(Blob(data="who?"))
    assert len(docs) == 1
    assert docs[0].page_content == "foo"


def test_default_lazy_load() -> None:
    class FakeLoader(BaseLoader):
        @override
        def load(self) -> list[Document]:
            return [
                Document(page_content="foo"),
                Document(page_content="bar"),
            ]

    loader = FakeLoader()
    docs = list(loader.lazy_load())
    assert docs == [Document(page_content="foo"), Document(page_content="bar")]


def test_lazy_load_not_implemented() -> None:
    class FakeLoader(BaseLoader):
        pass

    loader = FakeLoader()
    with pytest.raises(NotImplementedError):
        loader.lazy_load()


async def test_default_aload() -> None:
    class FakeLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            yield from [
                Document(page_content="foo"),
                Document(page_content="bar"),
            ]

    loader = FakeLoader()
    docs = loader.load()
    assert docs == [Document(page_content="foo"), Document(page_content="bar")]
    assert docs == [doc async for doc in loader.alazy_load()]
    assert docs == await loader.aload()


def test_load_and_split_with_default_splitter() -> None:
    """Test load_and_split with default RecursiveCharacterTextSplitter."""

    class FakeLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            yield Document(page_content="a" * 100 + "\n\n" + "b" * 100)
            yield Document(page_content="c" * 100, metadata={"source": "test"})

    loader = FakeLoader()

    # Should work with default splitter if langchain_text_splitters is available
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401

        docs = loader.load_and_split()
        assert len(docs) > 0
        assert all(isinstance(doc, Document) for doc in docs)
        # Verify documents were split
        assert len(docs) >= 2
    except ImportError:
        # If text_splitters not installed, should raise ImportError
        with pytest.raises(ImportError, match="Unable to import from langchain_text_splitters"):
            loader.load_and_split()


def test_load_and_split_with_custom_splitter() -> None:
    """Test load_and_split with custom text splitter."""
    from unittest.mock import MagicMock

    class FakeLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            yield Document(page_content="foo bar baz")
            yield Document(page_content="qux quux")

    loader = FakeLoader()
    mock_splitter = MagicMock()
    mock_splitter.split_documents.return_value = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]

    docs = loader.load_and_split(text_splitter=mock_splitter)

    assert len(docs) == 3
    assert docs[0].page_content == "foo"
    assert docs[1].page_content == "bar"
    assert docs[2].page_content == "baz"

    # Verify the splitter was called with the loaded documents
    mock_splitter.split_documents.assert_called_once()
    call_args = mock_splitter.split_documents.call_args[0][0]
    assert len(call_args) == 2
    assert call_args[0].page_content == "foo bar baz"
    assert call_args[1].page_content == "qux quux"


def test_load_and_split_preserves_metadata() -> None:
    """Test that load_and_split preserves document metadata."""
    from unittest.mock import MagicMock

    class FakeLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            yield Document(
                page_content="long content here",
                metadata={"source": "file.txt", "page": 1}
            )

    loader = FakeLoader()
    mock_splitter = MagicMock()
    mock_splitter.split_documents.return_value = [
        Document(page_content="long", metadata={"source": "file.txt", "page": 1}),
        Document(page_content="content", metadata={"source": "file.txt", "page": 1}),
    ]

    docs = loader.load_and_split(text_splitter=mock_splitter)

    assert all(doc.metadata.get("source") == "file.txt" for doc in docs)
    assert all(doc.metadata.get("page") == 1 for doc in docs)


def test_base_blob_parser_parse_returns_list() -> None:
    """Verify that parse() returns a list, not an iterator."""

    class MyParser(BaseBlobParser):
        @override
        def lazy_parse(self, blob: Blob) -> Iterator[Document]:
            yield Document(page_content="doc1")
            yield Document(page_content="doc2")

    parser = MyParser()
    result = parser.parse(Blob(data="test"))

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].page_content == "doc1"
    assert result[1].page_content == "doc2"


def test_base_blob_parser_with_empty_results() -> None:
    """Test parser that yields no documents."""

    class EmptyParser(BaseBlobParser):
        @override
        def lazy_parse(self, blob: Blob) -> Iterator[Document]:
            return
            yield  # Make it a generator

    parser = EmptyParser()
    docs = parser.parse(Blob(data="empty"))

    assert docs == []


def test_base_blob_parser_with_metadata() -> None:
    """Test parser that creates documents with metadata from blob."""

    class MetadataParser(BaseBlobParser):
        @override
        def lazy_parse(self, blob: Blob) -> Iterator[Document]:
            yield Document(
                page_content=blob.as_string() if blob.data else "",
                metadata={"source": blob.source, "mimetype": blob.mimetype}
            )

    parser = MetadataParser()
    blob = Blob(data="test content", metadata={"source": "file.txt"}, mimetype="text/plain")
    docs = parser.parse(blob)

    assert len(docs) == 1
    assert docs[0].page_content == "test content"
    assert docs[0].metadata["source"] == "file.txt"
    assert docs[0].metadata["mimetype"] == "text/plain"


def test_base_loader_load_calls_lazy_load() -> None:
    """Verify that load() properly consumes the lazy_load iterator."""

    call_count = 0

    class CountingLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            nonlocal call_count
            for i in range(3):
                call_count += 1
                yield Document(page_content=f"doc{i}")

    loader = CountingLoader()
    docs = loader.load()

    assert call_count == 3
    assert len(docs) == 3
    assert [doc.page_content for doc in docs] == ["doc0", "doc1", "doc2"]


async def test_alazy_load_with_multiple_docs() -> None:
    """Test async lazy loading with multiple documents."""

    class MultiDocLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            for i in range(5):
                yield Document(page_content=f"doc{i}", metadata={"index": i})

    loader = MultiDocLoader()
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    assert len(docs) == 5
    for i, doc in enumerate(docs):
        assert doc.page_content == f"doc{i}"
        assert doc.metadata["index"] == i


async def test_aload_empty_loader() -> None:
    """Test async load with loader that yields no documents."""

    class EmptyLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            return
            yield  # Make it a generator

    loader = EmptyLoader()
    docs = await loader.aload()

    assert docs == []


def test_base_loader_with_exception_in_lazy_load() -> None:
    """Test that exceptions in lazy_load are properly propagated."""

    class FailingLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            yield Document(page_content="first")
            raise ValueError("Intentional error")

    loader = FailingLoader()

    with pytest.raises(ValueError, match="Intentional error"):
        loader.load()


async def test_base_loader_with_exception_in_alazy_load() -> None:
    """Test that exceptions in alazy_load are properly propagated."""

    class FailingLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            yield Document(page_content="first")
            raise ValueError("Intentional async error")

    loader = FailingLoader()

    with pytest.raises(ValueError, match="Intentional async error"):
        await loader.aload()
