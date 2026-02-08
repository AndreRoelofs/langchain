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
        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter,  # noqa: F401
        )

        docs = loader.load_and_split()
        assert len(docs) > 0
        assert all(isinstance(doc, Document) for doc in docs)
        # Verify documents were split
        assert len(docs) >= 2
    except ImportError:
        # If text_splitters not installed, should raise ImportError
        with pytest.raises(
            ImportError, match="Unable to import from langchain_text_splitters"
        ):
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
                metadata={"source": "file.txt", "page": 1},
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
                metadata={"source": blob.source, "mimetype": blob.mimetype},
            )

    parser = MetadataParser()
    blob = Blob(
        data="test content", metadata={"source": "file.txt"}, mimetype="text/plain"
    )
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


def test_lazy_load_fallback_returns_iterator() -> None:
    """When subclass overrides load() but not lazy_load(), the fallback returns
    an iterator wrapping the list from load()."""

    class LoadOnlyLoader(BaseLoader):
        @override
        def load(self) -> list[Document]:
            return [
                Document(page_content="a"),
                Document(page_content="b"),
            ]

    loader = LoadOnlyLoader()
    result = loader.lazy_load()

    # Must be a lazy iterator, not a list
    assert hasattr(result, "__next__")
    assert next(result).page_content == "a"
    assert next(result).page_content == "b"
    with pytest.raises(StopIteration):
        next(result)


def test_lazy_load_fallback_does_not_trigger_for_lazy_load_override() -> None:
    """When subclass overrides lazy_load(), the load() fallback path should
    NOT be triggered — lazy_load is called directly."""

    load_called = False

    class LazyOnlyLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            yield Document(page_content="lazy")

        @override
        def load(self) -> list[Document]:
            nonlocal load_called
            load_called = True
            return [Document(page_content="eager")]

    loader = LazyOnlyLoader()
    docs = list(loader.lazy_load())

    assert docs == [Document(page_content="lazy")]
    assert not load_called


def test_load_consumes_lazy_load_when_only_lazy_load_overridden() -> None:
    """When subclass only overrides lazy_load(), load() should consume it."""

    class LazyOnlyLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            yield Document(page_content="x")
            yield Document(page_content="y")

    loader = LazyOnlyLoader()
    docs = loader.load()

    assert isinstance(docs, list)
    assert len(docs) == 2
    assert docs[0].page_content == "x"
    assert docs[1].page_content == "y"


def test_base_blob_parser_is_abstract() -> None:
    """Verify that BaseBlobParser cannot be instantiated without implementing
    lazy_parse."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseBlobParser()  # type: ignore[abstract]


def test_base_blob_parser_lazy_parse_is_abstract() -> None:
    """Verify lazy_parse must be implemented by subclasses."""

    class IncompleteParser(BaseBlobParser):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteParser()  # type: ignore[abstract]


def test_base_loader_is_not_strictly_abstract() -> None:
    """BaseLoader uses ABC but has no abstract methods (noqa: B024),
    so it can technically be instantiated — but lazy_load raises
    NotImplementedError."""

    class BareLoader(BaseLoader):
        pass

    loader = BareLoader()
    with pytest.raises(
        NotImplementedError, match="BareLoader does not implement lazy_load"
    ):
        loader.lazy_load()


def test_load_and_split_import_error_when_no_text_splitters() -> None:
    """Test that load_and_split raises ImportError when langchain_text_splitters
    is not available and no text_splitter is provided."""
    from unittest.mock import patch as mock_patch

    class SimpleLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            yield Document(page_content="hello world")

    loader = SimpleLoader()

    with mock_patch("langchain_core.document_loaders.base._HAS_TEXT_SPLITTERS", False):
        with pytest.raises(ImportError, match="Unable to import"):
            loader.load_and_split()


def test_load_and_split_uses_provided_splitter_even_without_text_splitters() -> None:
    """Even if langchain_text_splitters is unavailable, a provided text_splitter
    should work."""
    from unittest.mock import MagicMock
    from unittest.mock import patch as mock_patch

    class SimpleLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            yield Document(page_content="hello world")

    loader = SimpleLoader()
    mock_splitter = MagicMock()
    mock_splitter.split_documents.return_value = [
        Document(page_content="hello"),
        Document(page_content="world"),
    ]

    with mock_patch("langchain_core.document_loaders.base._HAS_TEXT_SPLITTERS", False):
        result = loader.load_and_split(text_splitter=mock_splitter)

    assert len(result) == 2
    assert result[0].page_content == "hello"
    mock_splitter.split_documents.assert_called_once()


async def test_alazy_load_yields_documents_in_order() -> None:
    """Verify alazy_load preserves document ordering from lazy_load."""

    class OrderedLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            for i in range(10):
                yield Document(page_content=str(i))

    loader = OrderedLoader()
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    assert [doc.page_content for doc in docs] == [str(i) for i in range(10)]


async def test_alazy_load_empty() -> None:
    """Verify alazy_load handles empty iterators correctly."""

    class EmptyLazyLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            return
            yield  # type: ignore[misc]

    loader = EmptyLazyLoader()
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    assert docs == []


def test_base_blob_parser_parse_consumes_all_from_lazy_parse() -> None:
    """Verify parse() eagerly consumes all documents from lazy_parse."""

    class MultiDocParser(BaseBlobParser):
        @override
        def lazy_parse(self, blob: Blob) -> Iterator[Document]:
            for i in range(5):
                yield Document(page_content=f"doc-{i}")

    parser = MultiDocParser()
    docs = parser.parse(Blob(data="test"))

    assert len(docs) == 5
    assert [d.page_content for d in docs] == [f"doc-{i}" for i in range(5)]


def test_base_blob_parser_lazy_parse_is_lazy() -> None:
    """Verify lazy_parse is actually lazy — documents are produced on demand."""

    call_count = 0

    class CountingParser(BaseBlobParser):
        @override
        def lazy_parse(self, blob: Blob) -> Iterator[Document]:
            nonlocal call_count
            for i in range(3):
                call_count += 1
                yield Document(page_content=f"doc-{i}")

    parser = CountingParser()
    gen = parser.lazy_parse(Blob(data="test"))

    assert call_count == 0
    next(gen)
    assert call_count == 1
    next(gen)
    assert call_count == 2


def test_load_returns_list_type() -> None:
    """Verify load() always returns a list, not a generator or iterator."""

    class SimpleLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            yield Document(page_content="a")

    loader = SimpleLoader()
    result = loader.load()

    assert type(result) is list


async def test_aload_returns_list_type() -> None:
    """Verify aload() always returns a list."""

    class SimpleLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            yield Document(page_content="a")

    loader = SimpleLoader()
    result = await loader.aload()

    assert type(result) is list


def test_lazy_load_not_implemented_error_message_includes_class_name() -> None:
    """The NotImplementedError message should include the class name."""

    class MyCustomLoader(BaseLoader):
        pass

    loader = MyCustomLoader()
    with pytest.raises(
        NotImplementedError, match="MyCustomLoader does not implement lazy_load"
    ):
        loader.lazy_load()


def test_load_and_split_calls_load_then_split() -> None:
    """Verify the exact call sequence: load() first, then split_documents()."""
    from unittest.mock import MagicMock, call

    call_order: list[str] = []

    class TrackingLoader(BaseLoader):
        @override
        def lazy_load(self) -> Iterator[Document]:
            call_order.append("lazy_load")
            yield Document(page_content="content")

    loader = TrackingLoader()
    mock_splitter = MagicMock()
    mock_splitter.split_documents.side_effect = lambda docs: (
        call_order.append("split_documents") or [Document(page_content="split")]
    )

    loader.load_and_split(text_splitter=mock_splitter)

    assert call_order == ["lazy_load", "split_documents"]
