"""Unit tests for BaseMedia and Blob classes."""

import tempfile
from io import BufferedReader, BytesIO
from pathlib import Path

import pytest

from langchain_core.documents import Document
from langchain_core.documents.base import BaseMedia, Blob


class TestBaseMedia:
    """Tests for BaseMedia class."""

    def test_base_media_defaults(self) -> None:
        """Test BaseMedia initialization with default values."""

        class TestMedia(BaseMedia):
            """Test subclass of BaseMedia."""

            pass

        media = TestMedia()
        assert media.id is None
        assert media.metadata == {}

    def test_base_media_with_id_string(self) -> None:
        """Test BaseMedia with string ID."""

        class TestMedia(BaseMedia):
            """Test subclass of BaseMedia."""

            pass

        media = TestMedia(id="test-id-123")
        assert media.id == "test-id-123"

    def test_base_media_with_id_coercion(self) -> None:
        """Test BaseMedia with numeric ID that gets coerced to string."""

        class TestMedia(BaseMedia):
            """Test subclass of BaseMedia."""

            pass

        media = TestMedia(id=123)
        assert media.id == "123"
        assert isinstance(media.id, str)

    def test_base_media_with_metadata(self) -> None:
        """Test BaseMedia with metadata dictionary."""

        class TestMedia(BaseMedia):
            """Test subclass of BaseMedia."""

            pass

        metadata = {"source": "test", "page": 1, "author": "Test Author"}
        media = TestMedia(metadata=metadata)
        assert media.metadata == metadata


class TestBlobFromData:
    """Tests for Blob.from_data class method."""

    def test_from_data_with_string(self) -> None:
        """Test creating Blob from string data."""
        blob = Blob.from_data("Hello, world!")
        assert blob.data == "Hello, world!"
        assert blob.encoding == "utf-8"
        assert blob.mimetype is None
        assert blob.path is None

    def test_from_data_with_bytes(self) -> None:
        """Test creating Blob from bytes data."""
        data = b"Hello, bytes!"
        blob = Blob.from_data(data)
        assert blob.data == data
        assert blob.encoding == "utf-8"

    def test_from_data_with_mime_type(self) -> None:
        """Test creating Blob with explicit MIME type."""
        blob = Blob.from_data("Hello", mime_type="text/plain")
        assert blob.mimetype == "text/plain"

    def test_from_data_with_custom_encoding(self) -> None:
        """Test creating Blob with custom encoding."""
        blob = Blob.from_data("Hello", encoding="ascii")
        assert blob.encoding == "ascii"

    def test_from_data_with_path(self) -> None:
        """Test creating Blob from data but with path metadata."""
        blob = Blob.from_data("data", path="/fake/path.txt")
        assert blob.data == "data"
        assert blob.path == "/fake/path.txt"

    def test_from_data_with_metadata(self) -> None:
        """Test creating Blob with metadata."""
        metadata = {"author": "test", "version": 1}
        blob = Blob.from_data("data", metadata=metadata)
        assert blob.metadata == metadata


class TestBlobFromPath:
    """Tests for Blob.from_path class method."""

    def test_from_path_basic(self, tmp_path: Path) -> None:
        """Test creating Blob from file path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("File content")

        blob = Blob.from_path(test_file)
        assert blob.path == test_file
        assert blob.data is None  # Data is lazy-loaded
        assert blob.encoding == "utf-8"

    def test_from_path_with_mime_type_guess(self, tmp_path: Path) -> None:
        """Test MIME type guessing from file extension."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Text content")

        blob = Blob.from_path(test_file, guess_type=True)
        assert blob.mimetype == "text/plain"

    def test_from_path_without_mime_type_guess(self, tmp_path: Path) -> None:
        """Test disabling MIME type guessing."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Text content")

        blob = Blob.from_path(test_file, guess_type=False)
        assert blob.mimetype is None

    def test_from_path_with_explicit_mime_type(self, tmp_path: Path) -> None:
        """Test explicit MIME type overrides guessing."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Text content")

        blob = Blob.from_path(test_file, mime_type="application/custom")
        assert blob.mimetype == "application/custom"

    def test_from_path_with_custom_encoding(self, tmp_path: Path) -> None:
        """Test custom encoding for file path Blob."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Text content")

        blob = Blob.from_path(test_file, encoding="ascii")
        assert blob.encoding == "ascii"

    def test_from_path_with_metadata(self, tmp_path: Path) -> None:
        """Test creating Blob from path with metadata."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Text content")

        metadata = {"author": "test"}
        blob = Blob.from_path(test_file, metadata=metadata)
        assert blob.metadata == metadata


class TestBlobAsString:
    """Tests for Blob.as_string method."""

    def test_as_string_from_string_data(self) -> None:
        """Test reading string from string data."""
        blob = Blob.from_data("Hello, world!")
        assert blob.as_string() == "Hello, world!"

    def test_as_string_from_bytes_data(self) -> None:
        """Test reading string from bytes data."""
        blob = Blob.from_data(b"Hello, bytes!")
        assert blob.as_string() == "Hello, bytes!"

    def test_as_string_from_file(self, tmp_path: Path) -> None:
        """Test reading string from file path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("File content", encoding="utf-8")

        blob = Blob.from_path(test_file)
        assert blob.as_string() == "File content"

    def test_as_string_with_custom_encoding(self, tmp_path: Path) -> None:
        """Test reading string with custom encoding."""
        test_file = tmp_path / "test.txt"
        content = "Hello with special chars: é"
        test_file.write_text(content, encoding="utf-8")

        blob = Blob.from_path(test_file, encoding="utf-8")
        assert blob.as_string() == content

    def test_as_string_bytes_with_custom_encoding(self) -> None:
        """Test decoding bytes with custom encoding."""
        data = "Café".encode("utf-8")
        blob = Blob.from_data(data, encoding="utf-8")
        assert blob.as_string() == "Café"


class TestBlobAsBytes:
    """Tests for Blob.as_bytes method."""

    def test_as_bytes_from_bytes_data(self) -> None:
        """Test reading bytes from bytes data."""
        data = b"Binary data"
        blob = Blob.from_data(data)
        assert blob.as_bytes() == data

    def test_as_bytes_from_string_data(self) -> None:
        """Test converting string to bytes."""
        blob = Blob.from_data("Hello")
        assert blob.as_bytes() == b"Hello"

    def test_as_bytes_from_file(self, tmp_path: Path) -> None:
        """Test reading bytes from file path."""
        test_file = tmp_path / "test.bin"
        data = b"Binary file content"
        test_file.write_bytes(data)

        blob = Blob.from_path(test_file)
        assert blob.as_bytes() == data

    def test_as_bytes_with_encoding(self) -> None:
        """Test encoding string to bytes with custom encoding."""
        blob = Blob.from_data("Café", encoding="utf-8")
        assert blob.as_bytes() == "Café".encode("utf-8")


class TestBlobAsBytesIO:
    """Tests for Blob.as_bytes_io context manager."""

    def test_as_bytes_io_from_bytes_data(self) -> None:
        """Test byte stream from bytes data."""
        data = b"Stream data"
        blob = Blob.from_data(data)

        with blob.as_bytes_io() as stream:
            assert isinstance(stream, BytesIO)
            content = stream.read()
            assert content == data

    def test_as_bytes_io_from_file(self, tmp_path: Path) -> None:
        """Test byte stream from file path."""
        test_file = tmp_path / "test.txt"
        data = b"File stream data"
        test_file.write_bytes(data)

        blob = Blob.from_path(test_file)

        with blob.as_bytes_io() as stream:
            assert isinstance(stream, BufferedReader)
            content = stream.read()
            assert content == data

    def test_as_bytes_io_from_string_raises(self) -> None:
        """Test that string data raises NotImplementedError."""
        blob = Blob.from_data("String data")

        with pytest.raises(NotImplementedError, match="Unable to convert blob"):
            with blob.as_bytes_io():
                pass


class TestBlobSource:
    """Tests for Blob.source property."""

    def test_source_from_path(self, tmp_path: Path) -> None:
        """Test source property returns path when available."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        blob = Blob.from_path(test_file)
        assert blob.source == str(test_file)

    def test_source_from_metadata(self) -> None:
        """Test source property from metadata overrides path."""
        blob = Blob.from_data(
            "data", path="/some/path.txt", metadata={"source": "custom_source"}
        )
        assert blob.source == "custom_source"

    def test_source_none_when_no_path(self) -> None:
        """Test source property is None when no path or metadata."""
        blob = Blob.from_data("data")
        assert blob.source is None


class TestBlobValidation:
    """Tests for Blob validation."""

    def test_blob_requires_data_or_path(self) -> None:
        """Test that Blob requires either data or path."""
        with pytest.raises(ValueError, match="Either data or path must be provided"):
            Blob()

    def test_blob_with_only_metadata_raises(self) -> None:
        """Test that Blob with only metadata raises error."""
        with pytest.raises(ValueError, match="Either data or path must be provided"):
            Blob(metadata={"key": "value"})


class TestBlobRepr:
    """Tests for Blob.__repr__ method."""

    def test_repr_with_source(self, tmp_path: Path) -> None:
        """Test __repr__ includes source when available."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        blob = Blob.from_path(test_file)
        repr_str = repr(blob)

        assert "Blob" in repr_str
        assert str(test_file) in repr_str

    def test_repr_without_source(self) -> None:
        """Test __repr__ without source."""
        blob = Blob.from_data("data")
        repr_str = repr(blob)

        assert "Blob" in repr_str
        assert str(id(blob)) in repr_str


class TestBlobImmutability:
    """Tests for Blob immutability (frozen model)."""

    def test_blob_is_frozen(self) -> None:
        """Test that Blob fields cannot be modified after creation."""
        blob = Blob.from_data("test")

        with pytest.raises(Exception):  # Pydantic ValidationError
            blob.data = "new data"  # type: ignore[misc]


class TestBlobIntegrationWithDocument:
    """Integration tests between Blob and Document."""

    def test_blob_and_document_separate_concerns(self) -> None:
        """Test that Blob and Document serve different purposes."""
        # Blob is for raw data
        blob = Blob.from_data("Raw content", mime_type="text/plain")
        assert blob.data == "Raw content"

        # Document is for processed text
        doc = Document(page_content="Processed content", metadata={"source": "test"})
        assert doc.page_content == "Processed content"

        # They both inherit from BaseMedia
        assert isinstance(blob, BaseMedia)
        assert isinstance(doc, BaseMedia)
