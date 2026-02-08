import pytest

from langchain_core.documents import Document
from langchain_core.documents.base import BaseMedia


def test_init() -> None:
    for doc in [
        Document(page_content="foo"),
        Document(page_content="foo", metadata={"a": 1}),
        Document(page_content="foo", id=None),
        Document(page_content="foo", id="1"),
        Document(page_content="foo", id=1),
    ]:
        assert isinstance(doc, Document)


def test_document_type_field() -> None:
    """Test that Document has correct type field."""
    doc = Document(page_content="test")
    assert doc.type == "Document"


def test_document_metadata_default() -> None:
    """Test Document metadata defaults to empty dict."""
    doc = Document(page_content="test")
    assert doc.metadata == {}
    assert isinstance(doc.metadata, dict)


def test_document_metadata_custom() -> None:
    """Test Document with custom metadata."""
    metadata = {"source": "test.txt", "page": 5, "author": "John"}
    doc = Document(page_content="content", metadata=metadata)
    assert doc.metadata == metadata


def test_document_id_string() -> None:
    """Test Document with string ID."""
    doc = Document(page_content="test", id="doc-123")
    assert doc.id == "doc-123"


def test_document_id_numeric_coercion() -> None:
    """Test Document ID numeric coercion to string."""
    doc = Document(page_content="test", id=42)
    assert doc.id == "42"
    assert isinstance(doc.id, str)


def test_document_id_none() -> None:
    """Test Document with None ID."""
    doc = Document(page_content="test", id=None)
    assert doc.id is None


def test_document_positional_arg() -> None:
    """Test Document accepts page_content as positional argument."""
    doc = Document("test content")
    assert doc.page_content == "test content"


def test_document_keyword_arg() -> None:
    """Test Document accepts page_content as keyword argument."""
    doc = Document(page_content="test content")
    assert doc.page_content == "test content"


def test_document_serialization_flag() -> None:
    """Test Document is marked as serializable."""
    assert Document.is_lc_serializable() is True


def test_document_namespace() -> None:
    """Test Document has correct namespace for serialization."""
    namespace = Document.get_lc_namespace()
    assert namespace == ["langchain", "schema", "document"]


def test_document_equality() -> None:
    """Test Document equality comparison."""
    doc1 = Document(page_content="test", metadata={"a": 1}, id="1")
    doc2 = Document(page_content="test", metadata={"a": 1}, id="1")
    # Documents should be equal if all fields match
    assert doc1.page_content == doc2.page_content
    assert doc1.metadata == doc2.metadata
    assert doc1.id == doc2.id


def test_document_metadata_modification() -> None:
    """Test that document metadata can be modified."""
    doc = Document(page_content="test", metadata={"key": "value"})
    doc.metadata["new_key"] = "new_value"
    assert doc.metadata["new_key"] == "new_value"


def test_document_empty_page_content() -> None:
    """Test Document with empty page content."""
    doc = Document(page_content="")
    assert doc.page_content == ""


def test_document_multiline_content() -> None:
    """Test Document with multiline page content."""
    content = "Line 1\nLine 2\nLine 3"
    doc = Document(page_content=content)
    assert doc.page_content == content
    assert "\n" in doc.page_content


def test_document_special_characters() -> None:
    """Test Document with special characters in content."""
    content = "Special chars: @#$%^&*()[]{}|\\;:',.<>?/~`"
    doc = Document(page_content=content)
    assert doc.page_content == content


def test_document_unicode_content() -> None:
    """Test Document with Unicode characters."""
    content = "Unicode: 你好世界 🌍 Café ñoño"
    doc = Document(page_content=content)
    assert doc.page_content == content


def test_document_nested_metadata() -> None:
    """Test Document with nested metadata structure."""
    metadata = {
        "source": {"type": "web", "url": "https://example.com"},
        "tags": ["python", "ai"],
        "metrics": {"score": 0.95, "count": 100},
    }
    doc = Document(page_content="test", metadata=metadata)
    assert doc.metadata == metadata
    assert doc.metadata["source"]["url"] == "https://example.com"


def test_document_str_without_id() -> None:
    """Test __str__ doesn't include id field."""
    doc = Document(page_content="test", id="123")
    str_repr = str(doc)
    # __str__ should only include page_content and metadata
    assert "page_content" in str_repr
    assert "test" in str_repr
    assert "123" not in str_repr  # id should not appear in __str__


def test_document_repr_includes_all_fields() -> None:
    """Test __repr__ includes metadata but follows pydantic format."""
    doc = Document(page_content="test", metadata={"key": "value"})
    repr_str = repr(doc)
    assert "Document" in repr_str
    assert "page_content" in repr_str
    assert "metadata" in repr_str


def test_document_inherits_from_base_media() -> None:
    """Test Document is a subclass of BaseMedia."""
    doc = Document(page_content="test")
    assert isinstance(doc, BaseMedia)


def test_document_pydantic_equality() -> None:
    """Test pydantic model equality for Documents with same fields."""
    doc1 = Document(page_content="test", metadata={"a": 1}, id="1")
    doc2 = Document(page_content="test", metadata={"a": 1}, id="1")
    assert doc1 == doc2


def test_document_pydantic_inequality_different_content() -> None:
    """Test pydantic model inequality for Documents with different content."""
    doc1 = Document(page_content="test1", metadata={"a": 1})
    doc2 = Document(page_content="test2", metadata={"a": 1})
    assert doc1 != doc2


def test_document_pydantic_inequality_different_metadata() -> None:
    """Test pydantic model inequality for Documents with different metadata."""
    doc1 = Document(page_content="test", metadata={"a": 1})
    doc2 = Document(page_content="test", metadata={"a": 2})
    assert doc1 != doc2


def test_document_pydantic_inequality_different_id() -> None:
    """Test pydantic model inequality for Documents with different id."""
    doc1 = Document(page_content="test", id="1")
    doc2 = Document(page_content="test", id="2")
    assert doc1 != doc2


def test_document_str_with_empty_metadata() -> None:
    """Test __str__ with explicitly empty metadata dict."""
    doc = Document(page_content="test", metadata={})
    str_repr = str(doc)
    # Empty metadata dict is falsy, so __str__ should not include it
    assert str_repr == "page_content='test'"


def test_document_str_with_metadata() -> None:
    """Test __str__ includes metadata when non-empty."""
    doc = Document(page_content="test", metadata={"key": "value"})
    str_repr = str(doc)
    assert "metadata=" in str_repr
    assert "key" in str_repr


def test_document_str_excludes_type_field() -> None:
    """Test __str__ does not include the type field."""
    doc = Document(page_content="test", metadata={"a": 1})
    str_repr = str(doc)
    assert "type=" not in str_repr


def test_document_page_content_is_mutable() -> None:
    """Test that Document page_content can be modified (not frozen)."""
    doc = Document(page_content="original")
    doc.page_content = "modified"
    assert doc.page_content == "modified"


def test_document_id_is_mutable() -> None:
    """Test that Document id can be modified."""
    doc = Document(page_content="test", id="original")
    doc.id = "modified"
    assert doc.id == "modified"


def test_document_model_copy() -> None:
    """Test Document model_copy creates independent copy."""
    doc = Document(page_content="test", metadata={"a": 1}, id="1")
    copy = doc.model_copy()
    assert copy == doc
    assert copy is not doc
    # Modifying copy's page_content should not affect original
    copy.page_content = "modified"
    assert doc.page_content == "test"


def test_document_model_copy_with_update() -> None:
    """Test Document model_copy with update dict."""
    doc = Document(page_content="test", metadata={"a": 1}, id="1")
    copy = doc.model_copy(update={"page_content": "updated", "id": "2"})
    assert copy.page_content == "updated"
    assert copy.id == "2"
    assert copy.metadata == {"a": 1}
    # Original is unchanged
    assert doc.page_content == "test"
    assert doc.id == "1"


def test_document_model_dump() -> None:
    """Test Document serialization to dict via model_dump."""
    doc = Document(page_content="test", metadata={"a": 1}, id="doc-1")
    data = doc.model_dump()
    assert data["page_content"] == "test"
    assert data["metadata"] == {"a": 1}
    assert data["id"] == "doc-1"
    assert data["type"] == "Document"


def test_document_model_dump_without_optional_fields() -> None:
    """Test Document model_dump with default optional fields."""
    doc = Document(page_content="test")
    data = doc.model_dump()
    assert data["page_content"] == "test"
    assert data["metadata"] == {}
    assert data["id"] is None
    assert data["type"] == "Document"


def test_document_construct_from_model_validate() -> None:
    """Test Document construction from dict via model_validate."""
    data = {"page_content": "test", "metadata": {"a": 1}, "id": "1"}
    doc = Document.model_validate(data)
    assert doc.page_content == "test"
    assert doc.metadata == {"a": 1}
    assert doc.id == "1"


def test_document_roundtrip_model_dump_validate() -> None:
    """Test Document survives model_dump -> model_validate roundtrip."""
    original = Document(page_content="test", metadata={"a": 1}, id="1")
    data = original.model_dump()
    restored = Document.model_validate(data)
    assert restored == original


def test_document_requires_page_content() -> None:
    """Test that Document requires page_content argument."""
    with pytest.raises(TypeError):
        Document()  # type: ignore[call-arg]


def test_document_metadata_is_independent_per_instance() -> None:
    """Test that default metadata dicts are independent between instances."""
    doc1 = Document(page_content="test1")
    doc2 = Document(page_content="test2")
    doc1.metadata["key"] = "value"
    assert "key" not in doc2.metadata


def test_document_very_long_content() -> None:
    """Test Document with very long page content."""
    content = "x" * 100_000
    doc = Document(page_content=content)
    assert len(doc.page_content) == 100_000
    assert doc.page_content == content


def test_document_whitespace_content() -> None:
    """Test Document with whitespace-only content."""
    doc = Document(page_content="   \t\n  ")
    assert doc.page_content == "   \t\n  "


def test_document_id_coercion_large_int() -> None:
    """Test Document ID coercion with large integer."""
    doc = Document(page_content="test", id=99999999999999)
    assert doc.id == "99999999999999"
    assert isinstance(doc.id, str)
