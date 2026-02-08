"""Unit tests for BaseMedia and Blob classes."""

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

        media = TestMedia()
        assert media.id is None
        assert media.metadata == {}

    def test_base_media_with_id_string(self) -> None:
        """Test BaseMedia with string ID."""

        class TestMedia(BaseMedia):
            """Test subclass of BaseMedia."""

        media = TestMedia(id="test-id-123")
        assert media.id == "test-id-123"

    def test_base_media_with_id_coercion(self) -> None:
        """Test BaseMedia with numeric ID that gets coerced to string."""

        class TestMedia(BaseMedia):
            """Test subclass of BaseMedia."""

        media = TestMedia(id=123)
        assert media.id == "123"
        assert isinstance(media.id, str)

    def test_base_media_with_metadata(self) -> None:
        """Test BaseMedia with metadata dictionary."""

        class TestMedia(BaseMedia):
            """Test subclass of BaseMedia."""

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
        data = "Café".encode()
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
        assert blob.as_bytes() == "Café".encode()


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

        with (
            pytest.raises(NotImplementedError, match="Unable to convert blob"),
            blob.as_bytes_io(),
        ):
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

        # Blob is frozen so attempting to modify raises ValidationError
        with pytest.raises(ValueError, match="frozen"):
            blob.data = "new data"  # type: ignore[misc]


class TestBaseMediaIdCoercion:
    """Tests for BaseMedia ID coercion edge cases."""

    def test_base_media_id_zero_coercion(self) -> None:
        """Test that numeric 0 is coerced to string '0'."""

        class TestMedia(BaseMedia):
            """Test subclass of BaseMedia."""

        media = TestMedia(id=0)
        assert media.id == "0"
        assert isinstance(media.id, str)

    def test_base_media_id_float_coercion(self) -> None:
        """Test that float ID is coerced to string."""

        class TestMedia(BaseMedia):
            """Test subclass of BaseMedia."""

        media = TestMedia(id=3.14)
        assert media.id == "3.14"
        assert isinstance(media.id, str)

    def test_base_media_id_negative_int_coercion(self) -> None:
        """Test that negative int ID is coerced to string."""

        class TestMedia(BaseMedia):
            """Test subclass of BaseMedia."""

        media = TestMedia(id=-1)
        assert media.id == "-1"
        assert isinstance(media.id, str)

    def test_base_media_id_empty_string(self) -> None:
        """Test that empty string ID is accepted as-is."""

        class TestMedia(BaseMedia):
            """Test subclass of BaseMedia."""

        media = TestMedia(id="")
        assert media.id == ""
        assert isinstance(media.id, str)


class TestBlobWithBothDataAndPath:
    """Tests for Blob when both data and path are provided."""

    def test_blob_with_data_and_path(self, tmp_path: Path) -> None:
        """Test Blob creation with both data and path set."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("file content")

        blob = Blob(data="in-memory content", path=test_file)
        assert blob.data == "in-memory content"
        assert blob.path == test_file

    def test_as_string_prefers_data_over_path(self, tmp_path: Path) -> None:
        """Test as_string returns in-memory data, not file content."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("file content")

        blob = Blob(data="in-memory content", path=test_file)
        # data is a string, so as_string returns it directly
        assert blob.as_string() == "in-memory content"

    def test_as_bytes_prefers_data_over_path(self, tmp_path: Path) -> None:
        """Test as_bytes returns in-memory data, not file content."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("file content")

        blob = Blob(data=b"in-memory bytes", path=test_file)
        assert blob.as_bytes() == b"in-memory bytes"


class TestBlobEmptyData:
    """Tests for Blob with empty data."""

    def test_blob_empty_string_data(self) -> None:
        """Test Blob with empty string data."""
        blob = Blob.from_data("")
        assert blob.as_string() == ""
        assert blob.as_bytes() == b""

    def test_blob_empty_bytes_data(self) -> None:
        """Test Blob with empty bytes data."""
        blob = Blob.from_data(b"")
        assert blob.as_bytes() == b""
        assert blob.as_string() == ""

    def test_blob_empty_bytes_as_bytes_io(self) -> None:
        """Test as_bytes_io with empty bytes returns valid BytesIO."""
        blob = Blob.from_data(b"")
        with blob.as_bytes_io() as stream:
            assert isinstance(stream, BytesIO)
            assert stream.read() == b""


class TestBlobAsStringErrors:
    """Tests for Blob.as_string error paths."""

    def test_as_string_raises_when_no_data_and_no_path(self) -> None:
        """Test as_string raises ValueError when data is None and no path."""
        # Blob with only path=some_string but data=None, and path doesn't exist
        # Use from_path with a nonexistent file to get data=None, path=set
        # But that would raise a file error. Instead, test the ValueError branch
        # by creating a blob where data=None but path is also None.
        # The validator requires either data or path, so we need to bypass it.
        # The actual ValueError path: data=None and path exists but file missing
        # triggers a FileNotFoundError from Path.read_text, not a ValueError.
        # The ValueError in as_string is unreachable if validator passes.
        # So let's test the FileNotFoundError case instead.
        blob = Blob.from_path("/nonexistent/path/to/file.txt")
        with pytest.raises(FileNotFoundError):
            blob.as_string()

    def test_as_bytes_raises_when_file_missing(self) -> None:
        """Test as_bytes raises FileNotFoundError for missing file."""
        blob = Blob.from_path("/nonexistent/path/to/file.bin")
        with pytest.raises(FileNotFoundError):
            blob.as_bytes()


class TestBlobAsBytesIOErrors:
    """Tests for Blob.as_bytes_io error paths."""

    def test_as_bytes_io_raises_not_implemented_for_none_data_no_path(self) -> None:
        """Test as_bytes_io raises NotImplementedError for string data."""
        # String data falls through to the else branch
        blob = Blob.from_data("string data")
        with pytest.raises(NotImplementedError, match="Unable to convert blob"):
            with blob.as_bytes_io():
                pass


class TestBlobSourceEdgeCases:
    """Tests for Blob.source property edge cases."""

    def test_source_metadata_without_source_key(self) -> None:
        """Test source property when metadata exists but has no 'source' key."""
        blob = Blob.from_data("data", metadata={"author": "test"})
        # metadata is non-empty but no 'source' key; path is None
        assert blob.source is None

    def test_source_metadata_source_is_none(self) -> None:
        """Test source property when metadata 'source' is explicitly None."""
        blob = Blob.from_data("data", path="/some/path.txt", metadata={"source": None})
        # metadata has 'source' key so it takes priority, even if None
        assert blob.source is None

    def test_source_from_data_with_path(self) -> None:
        """Test source returns path when from_data with path but no metadata source."""
        blob = Blob.from_data("data", path="/some/path.txt")
        assert blob.source == "/some/path.txt"


class TestBlobFromPathMimeTypes:
    """Tests for MIME type guessing in Blob.from_path."""

    def test_from_path_html_mime_type(self, tmp_path: Path) -> None:
        """Test MIME type guessing for .html file."""
        test_file = tmp_path / "page.html"
        test_file.write_text("<html></html>")

        blob = Blob.from_path(test_file)
        assert blob.mimetype == "text/html"

    def test_from_path_json_mime_type(self, tmp_path: Path) -> None:
        """Test MIME type guessing for .json file."""
        test_file = tmp_path / "data.json"
        test_file.write_text("{}")

        blob = Blob.from_path(test_file)
        assert blob.mimetype == "application/json"

    def test_from_path_unknown_extension_mime_type(self, tmp_path: Path) -> None:
        """Test MIME type guessing for unknown extension returns None."""
        test_file = tmp_path / "data.xyz123"
        test_file.write_text("data")

        blob = Blob.from_path(test_file)
        assert blob.mimetype is None

    def test_from_path_no_extension_mime_type(self, tmp_path: Path) -> None:
        """Test MIME type guessing for file with no extension."""
        test_file = tmp_path / "Makefile"
        test_file.write_text("all:")

        blob = Blob.from_path(test_file)
        # No extension means mimetypes can't guess
        assert blob.mimetype is None


class TestBlobFromPathWithPurePath:
    """Tests for Blob.from_path with different path types."""

    def test_from_path_with_path_object(self, tmp_path: Path) -> None:
        """Test from_path with pathlib.Path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        blob = Blob.from_path(Path(test_file))
        assert blob.as_string() == "content"

    def test_from_path_with_string(self, tmp_path: Path) -> None:
        """Test from_path with string path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        blob = Blob.from_path(str(test_file))
        assert blob.as_string() == "content"


class TestBlobReprEdgeCases:
    """Tests for Blob.__repr__ edge cases."""

    def test_repr_with_metadata_source(self) -> None:
        """Test __repr__ uses metadata source over path."""
        blob = Blob.from_data(
            "data", path="/file.txt", metadata={"source": "custom_source"}
        )
        repr_str = repr(blob)
        assert "custom_source" in repr_str

    def test_repr_format(self) -> None:
        """Test __repr__ format matches 'Blob <id>' pattern."""
        blob = Blob.from_data("data")
        repr_str = repr(blob)
        assert repr_str.startswith("Blob ")


class TestBlobImmutabilityExtended:
    """Extended tests for Blob immutability."""

    def test_blob_path_cannot_be_modified(self, tmp_path: Path) -> None:
        """Test that Blob path cannot be modified after creation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        blob = Blob.from_path(test_file)
        with pytest.raises(ValueError, match="frozen"):
            blob.path = "/new/path"  # type: ignore[misc]

    def test_blob_mimetype_cannot_be_modified(self) -> None:
        """Test that Blob mimetype cannot be modified after creation."""
        blob = Blob.from_data("data", mime_type="text/plain")
        with pytest.raises(ValueError, match="frozen"):
            blob.mimetype = "application/json"  # type: ignore[misc]

    def test_blob_encoding_cannot_be_modified(self) -> None:
        """Test that Blob encoding cannot be modified after creation."""
        blob = Blob.from_data("data")
        with pytest.raises(ValueError, match="frozen"):
            blob.encoding = "ascii"  # type: ignore[misc]


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
