"""Tests for langchain_core.messages.chat module."""

import pytest

from langchain_core.load import dumpd, load
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.base import BaseMessageChunk
from langchain_core.messages.human import HumanMessageChunk


class TestChatMessage:
    """Tests for the ChatMessage class."""

    def test_init_basic(self) -> None:
        """Test basic ChatMessage initialization."""
        msg = ChatMessage(content="Hello", role="user")
        assert msg.content == "Hello"
        assert msg.role == "user"
        assert msg.type == "chat"

    def test_init_with_name(self) -> None:
        """Test ChatMessage with name."""
        msg = ChatMessage(content="Hello", role="assistant", name="bot")
        assert msg.name == "bot"
        assert msg.role == "assistant"

    def test_init_with_id(self) -> None:
        """Test ChatMessage with ID."""
        msg = ChatMessage(content="Hello", role="user", id="msg-123")
        assert msg.id == "msg-123"

    def test_init_with_additional_kwargs(self) -> None:
        """Test ChatMessage with additional_kwargs."""
        msg = ChatMessage(
            content="Hello",
            role="user",
            additional_kwargs={"custom": "value"},
        )
        assert msg.additional_kwargs["custom"] == "value"

    def test_init_with_response_metadata(self) -> None:
        """Test ChatMessage with response_metadata."""
        msg = ChatMessage(
            content="Hello",
            role="system",
            response_metadata={"model": "custom"},
        )
        assert msg.response_metadata["model"] == "custom"

    def test_init_with_list_content(self) -> None:
        """Test ChatMessage with list content."""
        content = [{"type": "text", "text": "Hello"}]
        msg = ChatMessage(content=content, role="user")
        assert msg.content == content

    def test_different_roles(self) -> None:
        """Test ChatMessage with different roles."""
        roles = ["user", "assistant", "system", "admin", "custom_role"]
        for role in roles:
            msg = ChatMessage(content="Test", role=role)
            assert msg.role == role

    def test_type_is_chat(self) -> None:
        """Test that ChatMessage type is 'chat'."""
        msg = ChatMessage(content="Test", role="user")
        assert msg.type == "chat"

    def test_serialization_roundtrip(self) -> None:
        """Test ChatMessage serialization and deserialization."""
        msg = ChatMessage(
            content="Hello",
            role="moderator",
            name="mod1",
            id="chat-123",
            additional_kwargs={"priority": "high"},
        )
        dumped = dumpd(msg)
        assert dumped["type"] == "constructor"
        assert dumped["id"] == ["langchain", "schema", "messages", "ChatMessage"]

        loaded = load(dumped)
        assert isinstance(loaded, ChatMessage)
        assert loaded.content == "Hello"
        assert loaded.role == "moderator"
        assert loaded.name == "mod1"
        assert loaded.id == "chat-123"
        assert loaded.additional_kwargs["priority"] == "high"

    def test_text_property(self) -> None:
        """Test the .text property."""
        msg = ChatMessage(content="Hello world", role="user")
        assert msg.text == "Hello world"

    def test_text_property_list_content(self) -> None:
        """Test .text property with list content."""
        msg = ChatMessage(
            content=[{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}],
            role="user",
        )
        assert msg.text == "Part 1Part 2"

    def test_content_blocks_property(self) -> None:
        """Test the content_blocks property."""
        msg = ChatMessage(content="Hello", role="user")
        blocks = msg.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Hello"

    def test_pretty_repr(self) -> None:
        """Test pretty_repr output."""
        msg = ChatMessage(content="Hello", role="user")
        result = msg.pretty_repr()
        assert "Chat Message" in result
        assert "Hello" in result


class TestChatMessageChunk:
    """Tests for the ChatMessageChunk class."""

    def test_init_basic(self) -> None:
        """Test basic ChatMessageChunk initialization."""
        chunk = ChatMessageChunk(content="Hello", role="user")
        assert chunk.content == "Hello"
        assert chunk.role == "user"
        assert chunk.type == "ChatMessageChunk"

    def test_type_is_chat_message_chunk(self) -> None:
        """Test that ChatMessageChunk type is 'ChatMessageChunk'."""
        chunk = ChatMessageChunk(content="Test", role="user")
        assert chunk.type == "ChatMessageChunk"

    def test_add_same_role_chunks(self) -> None:
        """Test adding ChatMessageChunks with the same role."""
        chunk1 = ChatMessageChunk(content="Hello", role="user", id="1")
        chunk2 = ChatMessageChunk(content=" world", role="user")
        result = chunk1 + chunk2
        assert isinstance(result, ChatMessageChunk)
        assert result.content == "Hello world"
        assert result.role == "user"
        assert result.id == "1"

    def test_add_different_role_chunks_raises_error(self) -> None:
        """Test that adding chunks with different roles raises ValueError."""
        chunk1 = ChatMessageChunk(content="Hello", role="user")
        chunk2 = ChatMessageChunk(content=" world", role="assistant")
        with pytest.raises(ValueError, match="Cannot concatenate.*different roles"):
            chunk1 + chunk2

    def test_add_with_additional_kwargs(self) -> None:
        """Test adding chunks with additional_kwargs."""
        chunk1 = ChatMessageChunk(
            content="Hello",
            role="user",
            additional_kwargs={"key1": "value1"},
        )
        chunk2 = ChatMessageChunk(
            content=" world",
            role="user",
            additional_kwargs={"key2": "value2"},
        )
        result = chunk1 + chunk2
        assert result.additional_kwargs["key1"] == "value1"
        assert result.additional_kwargs["key2"] == "value2"

    def test_add_with_response_metadata(self) -> None:
        """Test adding chunks with response_metadata."""
        chunk1 = ChatMessageChunk(
            content="Hello",
            role="user",
            response_metadata={"meta1": "data1"},
        )
        chunk2 = ChatMessageChunk(
            content=" world",
            role="user",
            response_metadata={"meta2": "data2"},
        )
        result = chunk1 + chunk2
        assert result.response_metadata["meta1"] == "data1"
        assert result.response_metadata["meta2"] == "data2"

    def test_add_with_list_content(self) -> None:
        """Test adding chunks with list content.

        Note: Without an 'index' key, list items are appended, not merged.
        """
        chunk1 = ChatMessageChunk(
            content=[{"type": "text", "text": "Hello"}],
            role="user",
        )
        chunk2 = ChatMessageChunk(
            content=[{"type": "text", "text": " world"}],
            role="user",
        )
        result = chunk1 + chunk2
        assert isinstance(result.content, list)
        # Items without 'index' key are appended, not merged
        assert len(result.content) == 2
        assert result.content[0]["text"] == "Hello"
        assert result.content[1]["text"] == " world"

    def test_add_with_list_content_with_index(self) -> None:
        """Test adding chunks with list content that have matching index keys."""
        chunk1 = ChatMessageChunk(
            content=[{"type": "text", "text": "Hello", "index": 0}],
            role="user",
        )
        chunk2 = ChatMessageChunk(
            content=[{"type": "text", "text": " world", "index": 0}],
            role="user",
        )
        result = chunk1 + chunk2
        assert isinstance(result.content, list)
        # Items with same 'index' key are merged
        assert len(result.content) == 1
        assert result.content[0]["text"] == "Hello world"
        assert result.content[0]["index"] == 0

    def test_add_chat_chunk_to_other_base_chunk(self) -> None:
        """Test adding ChatMessageChunk to another BaseMessageChunk type."""
        chunk1 = ChatMessageChunk(content="Hello", role="user", id="1")
        chunk2 = HumanMessageChunk(content=" world")
        result = chunk1 + chunk2
        # Should return ChatMessageChunk since it's the receiver
        assert isinstance(result, ChatMessageChunk)
        assert result.content == "Hello world"
        assert result.role == "user"
        assert result.id == "1"

    def test_add_preserves_id(self) -> None:
        """Test that adding chunks preserves the ID from the first chunk."""
        chunk1 = ChatMessageChunk(content="Hello", role="user", id="original-id")
        chunk2 = ChatMessageChunk(content=" world", role="user", id="other-id")
        result = chunk1 + chunk2
        assert result.id == "original-id"

    def test_serialization_roundtrip(self) -> None:
        """Test ChatMessageChunk serialization and deserialization."""
        chunk = ChatMessageChunk(
            content="Hello",
            role="moderator",
            id="chunk-123",
        )
        dumped = dumpd(chunk)
        assert dumped["type"] == "constructor"
        assert dumped["id"] == ["langchain", "schema", "messages", "ChatMessageChunk"]

        loaded = load(dumped)
        assert isinstance(loaded, ChatMessageChunk)
        assert loaded.content == "Hello"
        assert loaded.role == "moderator"
        assert loaded.id == "chunk-123"

    def test_multiple_chunks_addition(self) -> None:
        """Test adding multiple chunks together."""
        chunk1 = ChatMessageChunk(content="a", role="user")
        chunk2 = ChatMessageChunk(content="b", role="user")
        chunk3 = ChatMessageChunk(content="c", role="user")
        result = chunk1 + chunk2 + chunk3
        assert result.content == "abc"
        assert result.role == "user"

    def test_empty_content_chunks(self) -> None:
        """Test adding chunks with empty content."""
        chunk1 = ChatMessageChunk(content="Hello", role="user")
        chunk2 = ChatMessageChunk(content="", role="user")
        result = chunk1 + chunk2
        assert result.content == "Hello"

    def test_add_incompatible_type_raises_error(self) -> None:
        """Test that adding incompatible types raises TypeError."""
        chunk = ChatMessageChunk(content="Hello", role="user")
        with pytest.raises(TypeError):
            chunk + "not a chunk"

    def test_text_property(self) -> None:
        """Test the .text property on chunk."""
        chunk = ChatMessageChunk(content="Hello world", role="user")
        assert chunk.text == "Hello world"

    def test_content_blocks_property(self) -> None:
        """Test the content_blocks property on chunk."""
        chunk = ChatMessageChunk(content="Hello", role="user")
        blocks = chunk.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Hello"
