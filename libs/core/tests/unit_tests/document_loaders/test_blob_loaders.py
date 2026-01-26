"""Test Blob Loaders."""

from collections.abc import Iterable

import pytest
from typing_extensions import override

from langchain_core.document_loaders.blob_loaders import Blob, BlobLoader, PathLike
from langchain_core.documents.base import Blob as BlobImport


def test_blob_loader_is_abstract() -> None:
    """Verify that BlobLoader cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BlobLoader()  # type: ignore[abstract]


def test_blob_loader_implementation() -> None:
    """Test a concrete implementation of BlobLoader."""

    class SimpleBlobLoader(BlobLoader):
        def __init__(self, data_list: list[str]) -> None:
            self.data_list = data_list

        @override
        def yield_blobs(self) -> Iterable[Blob]:
            for i, data in enumerate(self.data_list):
                yield Blob(data=data, metadata={"source": f"source_{i}.txt"})

    loader = SimpleBlobLoader(["content1", "content2", "content3"])
    blobs = list(loader.yield_blobs())

    assert len(blobs) == 3
    assert blobs[0].as_string() == "content1"
    assert blobs[0].source == "source_0.txt"
    assert blobs[1].as_string() == "content2"
    assert blobs[1].source == "source_1.txt"
    assert blobs[2].as_string() == "content3"
    assert blobs[2].source == "source_2.txt"


def test_blob_loader_with_empty_results() -> None:
    """Test BlobLoader that yields no blobs."""

    class EmptyBlobLoader(BlobLoader):
        @override
        def yield_blobs(self) -> Iterable[Blob]:
            return
            yield  # Make it a generator

    loader = EmptyBlobLoader()
    blobs = list(loader.yield_blobs())

    assert blobs == []


def test_blob_loader_with_metadata() -> None:
    """Test BlobLoader that includes metadata in blobs."""

    class MetadataBlobLoader(BlobLoader):
        @override
        def yield_blobs(self) -> Iterable[Blob]:
            yield Blob(
                data=b"binary data",
                mimetype="application/octet-stream",
                metadata={"source": "file.bin", "size": 11, "created": "2024-01-01"}
            )
            yield Blob(
                data="text data",
                mimetype="text/plain",
                metadata={"source": "file.txt", "size": 9, "created": "2024-01-02"}
            )

    loader = MetadataBlobLoader()
    blobs = list(loader.yield_blobs())

    assert len(blobs) == 2

    # First blob (binary)
    assert blobs[0].data == b"binary data"
    assert blobs[0].source == "file.bin"
    assert blobs[0].mimetype == "application/octet-stream"
    assert blobs[0].metadata["size"] == 11

    # Second blob (text)
    assert blobs[1].as_string() == "text data"
    assert blobs[1].source == "file.txt"
    assert blobs[1].mimetype == "text/plain"
    assert blobs[1].metadata["size"] == 9


def test_blob_loader_is_lazy() -> None:
    """Test that BlobLoader uses lazy iteration."""

    call_count = 0

    class CountingBlobLoader(BlobLoader):
        @override
        def yield_blobs(self) -> Iterable[Blob]:
            nonlocal call_count
            for i in range(5):
                call_count += 1
                yield Blob(data=f"blob{i}")

    loader = CountingBlobLoader()
    iterator = loader.yield_blobs()

    # No blobs should be created yet
    assert call_count == 0

    # Get first blob
    first_blob = next(iterator)
    assert call_count == 1
    assert first_blob.as_string() == "blob0"

    # Get second blob
    second_blob = next(iterator)
    assert call_count == 2
    assert second_blob.as_string() == "blob1"

    # Consume the rest
    remaining = list(iterator)
    assert call_count == 5
    assert len(remaining) == 3


def test_blob_loader_with_different_sources() -> None:
    """Test BlobLoader that loads from multiple sources."""

    class MultiSourceBlobLoader(BlobLoader):
        def __init__(self, sources: list[tuple[str, str]]) -> None:
            self.sources = sources

        @override
        def yield_blobs(self) -> Iterable[Blob]:
            for source, content in self.sources:
                yield Blob(data=content, metadata={"source": source})

    sources = [
        ("memory://data1", "content from memory"),
        ("file:///path/to/file.txt", "content from file"),
        ("https://example.com/data", "content from web"),
    ]

    loader = MultiSourceBlobLoader(sources)
    blobs = list(loader.yield_blobs())

    assert len(blobs) == 3
    assert blobs[0].source == "memory://data1"
    assert blobs[1].source == "file:///path/to/file.txt"
    assert blobs[2].source == "https://example.com/data"


def test_blob_loader_handles_exceptions() -> None:
    """Test that BlobLoader properly propagates exceptions."""

    class FailingBlobLoader(BlobLoader):
        @override
        def yield_blobs(self) -> Iterable[Blob]:
            yield Blob(data="first")
            raise IOError("Failed to load blob")
            yield Blob(data="never reached")  # noqa: B901

    loader = FailingBlobLoader()
    iterator = loader.yield_blobs()

    # First blob should work
    first = next(iterator)
    assert first.as_string() == "first"

    # Second iteration should raise
    with pytest.raises(IOError, match="Failed to load blob"):
        next(iterator)


def test_blob_loader_with_bytes_and_strings() -> None:
    """Test BlobLoader handling both bytes and string data."""

    class MixedDataBlobLoader(BlobLoader):
        @override
        def yield_blobs(self) -> Iterable[Blob]:
            yield Blob(data=b"bytes data", source="binary.dat")
            yield Blob(data="string data", source="text.txt")
            yield Blob(
                data="unicode: café ☕",
                source="unicode.txt",
                mimetype="text/plain; charset=utf-8"
            )

    loader = MixedDataBlobLoader()
    blobs = list(loader.yield_blobs())

    assert len(blobs) == 3
    assert blobs[0].data == b"bytes data"
    assert blobs[1].as_string() == "string data"
    assert blobs[2].as_string() == "unicode: café ☕"


def test_blob_import_compatibility() -> None:
    """Test that Blob is correctly re-exported for backwards compatibility."""
    # Verify that the import from blob_loaders works
    from langchain_core.document_loaders.blob_loaders import Blob as BlobFromBlobLoaders

    # Verify that it's the same as importing from base
    assert BlobFromBlobLoaders is BlobImport

    # Verify we can create instances
    blob = BlobFromBlobLoaders(data="test")
    assert blob.as_string() == "test"


def test_pathlike_import_compatibility() -> None:
    """Test that PathLike is correctly re-exported."""
    # Verify that the import works
    from langchain_core.document_loaders.blob_loaders import PathLike

    # PathLike should be a type alias/protocol
    # Just verify it can be imported without errors
    assert PathLike is not None


def test_blob_loader_with_conditional_loading() -> None:
    """Test BlobLoader with conditional logic."""

    class FilteredBlobLoader(BlobLoader):
        def __init__(self, items: list[tuple[str, str]], filter_fn) -> None:  # type: ignore[no-untyped-def]
            self.items = items
            self.filter_fn = filter_fn

        @override
        def yield_blobs(self) -> Iterable[Blob]:
            for source, data in self.items:
                if self.filter_fn(source):
                    yield Blob(data=data, metadata={"source": source})

    items = [
        ("file1.txt", "content1"),
        ("file2.log", "content2"),
        ("file3.txt", "content3"),
        ("file4.json", "content4"),
    ]

    # Only load .txt files
    loader = FilteredBlobLoader(items, lambda s: s.endswith(".txt"))
    blobs = list(loader.yield_blobs())

    assert len(blobs) == 2
    assert all(blob.source and blob.source.endswith(".txt") for blob in blobs)
    assert blobs[0].as_string() == "content1"
    assert blobs[1].as_string() == "content3"


def test_blob_loader_iterator_multiple_times() -> None:
    """Test that yield_blobs can be called multiple times."""

    class ReusableBlobLoader(BlobLoader):
        def __init__(self, data: list[str]) -> None:
            self.data = data

        @override
        def yield_blobs(self) -> Iterable[Blob]:
            for item in self.data:
                yield Blob(data=item)

    loader = ReusableBlobLoader(["a", "b", "c"])

    # First iteration
    blobs1 = list(loader.yield_blobs())
    assert len(blobs1) == 3
    assert [b.as_string() for b in blobs1] == ["a", "b", "c"]

    # Second iteration should work the same
    blobs2 = list(loader.yield_blobs())
    assert len(blobs2) == 3
    assert [b.as_string() for b in blobs2] == ["a", "b", "c"]


def test_blob_loader_with_large_data() -> None:
    """Test BlobLoader with larger amounts of data (performance check)."""

    class LargeBlobLoader(BlobLoader):
        def __init__(self, count: int) -> None:
            self.count = count

        @override
        def yield_blobs(self) -> Iterable[Blob]:
            for i in range(self.count):
                yield Blob(
                    data=f"Document {i}: {'x' * 100}",
                    source=f"doc_{i}.txt",
                    metadata={"index": i}
                )

    loader = LargeBlobLoader(100)
    blobs = list(loader.yield_blobs())

    assert len(blobs) == 100
    assert all(isinstance(b, Blob) for b in blobs)
    assert blobs[0].metadata["index"] == 0
    assert blobs[99].metadata["index"] == 99
