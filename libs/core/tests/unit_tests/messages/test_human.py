"""Tests for langchain_core.messages.human module."""

import pytest

from langchain_core.load import dumpd, load
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.system import SystemMessageChunk


class TestHumanMessage:
    """Tests for the HumanMessage class."""

    def test_init_basic(self) -> None:
        """Test basic HumanMessage initialization."""
        msg = HumanMessage(content="Hello, how are you?")
        assert msg.content == "Hello, how are you?"
        assert msg.type == "human"

    def test_init_with_name(self) -> None:
        """Test HumanMessage with name."""
        msg = HumanMessage(content="Hello", name="user1")
        assert msg.name == "user1"

    def test_init_with_id(self) -> None:
        """Test HumanMessage with ID."""
        msg = HumanMessage(content="Hello", id="msg-123")
        assert msg.id == "msg-123"

    def test_init_with_additional_kwargs(self) -> None:
        """Test HumanMessage with additional_kwargs."""
        msg = HumanMessage(
            content="Hello",
            additional_kwargs={"custom": "value"},
        )
        assert msg.additional_kwargs["custom"] == "value"

    def test_init_with_response_metadata(self) -> None:
        """Test HumanMessage with response_metadata."""
        msg = HumanMessage(
            content="Hello",
            response_metadata={"source": "web"},
        )
        assert msg.response_metadata["source"] == "web"

    def test_init_with_list_content(self) -> None:
        """Test HumanMessage with list content."""
        content = [{"type": "text", "text": "Hello"}]
        msg = HumanMessage(content=content)
        assert msg.content == content

    def test_init_with_multimodal_content(self) -> None:
        """Test HumanMessage with multimodal content."""
        content = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        msg = HumanMessage(content=content)
        assert len(msg.content) == 2
        assert msg.content[0]["type"] == "text"
        assert msg.content[1]["type"] == "image_url"

    def test_init_with_content_blocks(self) -> None:
        """Test HumanMessage with content_blocks parameter."""
        blocks = [
            {"type": "text", "text": "Hello"},
            {"type": "image", "url": "https://example.com/img.png"},
        ]
        msg = HumanMessage(content_blocks=blocks)
        assert msg.content == blocks

    def test_type_is_human(self) -> None:
        """Test that HumanMessage type is 'human'."""
        msg = HumanMessage(content="Test")
        assert msg.type == "human"

    def test_serialization_roundtrip(self) -> None:
        """Test HumanMessage serialization and deserialization."""
        msg = HumanMessage(
            content="Hello",
            name="user1",
            id="msg-123",
            additional_kwargs={"custom": "value"},
        )
        dumped = dumpd(msg)
        assert dumped["type"] == "constructor"
        assert dumped["id"] == ["langchain", "schema", "messages", "HumanMessage"]

        loaded = load(dumped)
        assert isinstance(loaded, HumanMessage)
        assert loaded.content == "Hello"
        assert loaded.name == "user1"
        assert loaded.id == "msg-123"
        assert loaded.additional_kwargs["custom"] == "value"

    def test_text_property(self) -> None:
        """Test the .text property."""
        msg = HumanMessage(content="Hello world")
        assert msg.text == "Hello world"

    def test_text_property_list_content(self) -> None:
        """Test .text property with list content."""
        msg = HumanMessage(
            content=[{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}]
        )
        assert msg.text == "Part 1Part 2"

    def test_text_property_multimodal_content(self) -> None:
        """Test .text property with multimodal content (non-text filtered out)."""
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "https://example.com"}},
                {"type": "text", "text": " world"},
            ]
        )
        assert msg.text == "Hello world"

    def test_content_blocks_property(self) -> None:
        """Test the content_blocks property."""
        msg = HumanMessage(content="Hello")
        blocks = msg.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Hello"

    def test_content_blocks_multimodal(self) -> None:
        """Test content_blocks with multimodal content."""
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "What's in this?"},
                {"type": "image", "url": "https://example.com/img.png"},
            ]
        )
        blocks = msg.content_blocks
        assert len(blocks) == 2
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "image"

    def test_pretty_repr(self) -> None:
        """Test pretty_repr output."""
        msg = HumanMessage(content="Hello")
        result = msg.pretty_repr()
        assert "Human Message" in result
        assert "Hello" in result

    def test_pretty_repr_with_name(self) -> None:
        """Test pretty_repr with name."""
        msg = HumanMessage(content="Hello", name="user1")
        result = msg.pretty_repr()
        assert "Name: user1" in result

    def test_empty_content(self) -> None:
        """Test HumanMessage with empty content."""
        msg = HumanMessage(content="")
        assert msg.content == ""
        assert msg.text == ""

    def test_empty_list_content(self) -> None:
        """Test HumanMessage with empty list content."""
        msg = HumanMessage(content=[])
        assert msg.content == []
        assert msg.text == ""


class TestHumanMessageChunk:
    """Tests for the HumanMessageChunk class."""

    def test_init_basic(self) -> None:
        """Test basic HumanMessageChunk initialization."""
        chunk = HumanMessageChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.type == "HumanMessageChunk"

    def test_type_is_human_message_chunk(self) -> None:
        """Test that HumanMessageChunk type is 'HumanMessageChunk'."""
        chunk = HumanMessageChunk(content="Test")
        assert chunk.type == "HumanMessageChunk"

    def test_add_two_chunks(self) -> None:
        """Test adding two HumanMessageChunks."""
        chunk1 = HumanMessageChunk(content="Hello", id="1")
        chunk2 = HumanMessageChunk(content=" world")
        result = chunk1 + chunk2
        assert isinstance(result, HumanMessageChunk)
        assert result.content == "Hello world"
        assert result.id == "1"

    def test_add_with_additional_kwargs(self) -> None:
        """Test adding chunks with additional_kwargs."""
        chunk1 = HumanMessageChunk(
            content="Hello",
            additional_kwargs={"key1": "value1"},
        )
        chunk2 = HumanMessageChunk(
            content=" world",
            additional_kwargs={"key2": "value2"},
        )
        result = chunk1 + chunk2
        assert result.additional_kwargs["key1"] == "value1"
        assert result.additional_kwargs["key2"] == "value2"

    def test_add_with_response_metadata(self) -> None:
        """Test adding chunks with response_metadata."""
        chunk1 = HumanMessageChunk(
            content="Hello",
            response_metadata={"meta1": "data1"},
        )
        chunk2 = HumanMessageChunk(
            content=" world",
            response_metadata={"meta2": "data2"},
        )
        result = chunk1 + chunk2
        assert result.response_metadata["meta1"] == "data1"
        assert result.response_metadata["meta2"] == "data2"

    def test_add_with_list_content(self) -> None:
        """Test adding chunks with list content.

        Note: Without an 'index' key, list items are appended, not merged.
        """
        chunk1 = HumanMessageChunk(content=[{"type": "text", "text": "Hello"}])
        chunk2 = HumanMessageChunk(content=[{"type": "text", "text": " world"}])
        result = chunk1 + chunk2
        assert isinstance(result.content, list)
        # Items without 'index' key are appended, not merged
        assert len(result.content) == 2
        assert result.content[0]["text"] == "Hello"
        assert result.content[1]["text"] == " world"

    def test_add_with_list_content_with_index(self) -> None:
        """Test adding chunks with list content that have matching index keys."""
        chunk1 = HumanMessageChunk(content=[{"type": "text", "text": "Hello", "index": 0}])
        chunk2 = HumanMessageChunk(content=[{"type": "text", "text": " world", "index": 0}])
        result = chunk1 + chunk2
        assert isinstance(result.content, list)
        # Items with same 'index' key are merged
        assert len(result.content) == 1
        assert result.content[0]["text"] == "Hello world"
        assert result.content[0]["index"] == 0

    def test_add_preserves_id(self) -> None:
        """Test that adding chunks preserves the ID from the first chunk."""
        chunk1 = HumanMessageChunk(content="Hello", id="original-id")
        chunk2 = HumanMessageChunk(content=" world", id="other-id")
        result = chunk1 + chunk2
        assert result.id == "original-id"

    def test_add_list_of_chunks(self) -> None:
        """Test adding a list of chunks to a chunk."""
        chunk1 = HumanMessageChunk(content="a", id="1")
        chunk2 = HumanMessageChunk(content="b")
        chunk3 = HumanMessageChunk(content="c")
        result = chunk1 + [chunk2, chunk3]
        assert result.content == "abc"
        assert result.id == "1"

    def test_serialization_roundtrip(self) -> None:
        """Test HumanMessageChunk serialization and deserialization."""
        chunk = HumanMessageChunk(
            content="Hello",
            name="user1",
            id="chunk-123",
        )
        dumped = dumpd(chunk)
        assert dumped["type"] == "constructor"
        assert dumped["id"] == ["langchain", "schema", "messages", "HumanMessageChunk"]

        loaded = load(dumped)
        assert isinstance(loaded, HumanMessageChunk)
        assert loaded.content == "Hello"
        assert loaded.name == "user1"
        assert loaded.id == "chunk-123"

    def test_multiple_chunks_addition(self) -> None:
        """Test adding multiple chunks together."""
        chunk1 = HumanMessageChunk(content="a")
        chunk2 = HumanMessageChunk(content="b")
        chunk3 = HumanMessageChunk(content="c")
        result = chunk1 + chunk2 + chunk3
        assert result.content == "abc"

    def test_empty_content_chunks(self) -> None:
        """Test adding chunks with empty content."""
        chunk1 = HumanMessageChunk(content="Hello")
        chunk2 = HumanMessageChunk(content="")
        result = chunk1 + chunk2
        assert result.content == "Hello"

    def test_add_different_chunk_type(self) -> None:
        """Test adding HumanMessageChunk to another chunk type."""
        chunk1 = HumanMessageChunk(content="Hello", id="1")
        chunk2 = SystemMessageChunk(content=" world")
        # Adding different types falls through to base implementation
        result = chunk1 + chunk2
        assert isinstance(result, HumanMessageChunk)
        assert result.content == "Hello world"

    def test_add_incompatible_type_raises_error(self) -> None:
        """Test that adding incompatible types raises TypeError."""
        chunk = HumanMessageChunk(content="Hello")
        with pytest.raises(TypeError):
            chunk + "not a chunk"

    def test_text_property(self) -> None:
        """Test the .text property on chunk."""
        chunk = HumanMessageChunk(content="Hello world")
        assert chunk.text == "Hello world"

    def test_content_blocks_property(self) -> None:
        """Test the content_blocks property on chunk."""
        chunk = HumanMessageChunk(content="Hello")
        blocks = chunk.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Hello"

    def test_content_blocks_multimodal_chunk(self) -> None:
        """Test content_blocks with multimodal chunk."""
        chunk = HumanMessageChunk(
            content=[
                {"type": "text", "text": "Check this:"},
                {"type": "image", "url": "https://example.com/img.png"},
            ]
        )
        blocks = chunk.content_blocks
        assert len(blocks) == 2
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "image"
