"""Tests for langchain_core.messages.system module."""

import pytest

from langchain_core.load import dumpd, load
from langchain_core.messages.human import HumanMessageChunk
from langchain_core.messages.system import SystemMessage, SystemMessageChunk


class TestSystemMessage:
    """Tests for the SystemMessage class."""

    def test_init_basic(self) -> None:
        """Test basic SystemMessage initialization."""
        msg = SystemMessage(content="You are a helpful assistant.")
        assert msg.content == "You are a helpful assistant."
        assert msg.type == "system"

    def test_init_with_name(self) -> None:
        """Test SystemMessage with name."""
        msg = SystemMessage(content="Instructions", name="system_prompt")
        assert msg.name == "system_prompt"

    def test_init_with_id(self) -> None:
        """Test SystemMessage with ID."""
        msg = SystemMessage(content="Instructions", id="sys-123")
        assert msg.id == "sys-123"

    def test_init_with_additional_kwargs(self) -> None:
        """Test SystemMessage with additional_kwargs."""
        msg = SystemMessage(
            content="Instructions",
            additional_kwargs={"priority": "high"},
        )
        assert msg.additional_kwargs["priority"] == "high"

    def test_init_with_response_metadata(self) -> None:
        """Test SystemMessage with response_metadata."""
        msg = SystemMessage(
            content="Instructions",
            response_metadata={"version": "1.0"},
        )
        assert msg.response_metadata["version"] == "1.0"

    def test_init_with_list_content(self) -> None:
        """Test SystemMessage with list content."""
        content = [{"type": "text", "text": "Instructions"}]
        msg = SystemMessage(content=content)
        assert msg.content == content

    def test_init_with_content_blocks(self) -> None:
        """Test SystemMessage with content_blocks parameter."""
        blocks = [
            {"type": "text", "text": "First instruction"},
            {"type": "text", "text": "Second instruction"},
        ]
        msg = SystemMessage(content_blocks=blocks)
        assert msg.content == blocks

    def test_type_is_system(self) -> None:
        """Test that SystemMessage type is 'system'."""
        msg = SystemMessage(content="Test")
        assert msg.type == "system"

    def test_serialization_roundtrip(self) -> None:
        """Test SystemMessage serialization and deserialization."""
        msg = SystemMessage(
            content="You are a helpful assistant.",
            name="system_prompt",
            id="sys-123",
            additional_kwargs={"version": "1.0"},
        )
        dumped = dumpd(msg)
        assert dumped["type"] == "constructor"
        assert dumped["id"] == ["langchain", "schema", "messages", "SystemMessage"]

        loaded = load(dumped)
        assert isinstance(loaded, SystemMessage)
        assert loaded.content == "You are a helpful assistant."
        assert loaded.name == "system_prompt"
        assert loaded.id == "sys-123"
        assert loaded.additional_kwargs["version"] == "1.0"

    def test_text_property(self) -> None:
        """Test the .text property."""
        msg = SystemMessage(content="Hello world")
        assert msg.text == "Hello world"

    def test_text_property_list_content(self) -> None:
        """Test .text property with list content."""
        msg = SystemMessage(
            content=[
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ]
        )
        assert msg.text == "Part 1Part 2"

    def test_content_blocks_property(self) -> None:
        """Test the content_blocks property."""
        msg = SystemMessage(content="Instructions")
        blocks = msg.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Instructions"

    def test_pretty_repr(self) -> None:
        """Test pretty_repr output."""
        msg = SystemMessage(content="Instructions")
        result = msg.pretty_repr()
        assert "System Message" in result
        assert "Instructions" in result

    def test_pretty_repr_with_name(self) -> None:
        """Test pretty_repr with name."""
        msg = SystemMessage(content="Instructions", name="sys_prompt")
        result = msg.pretty_repr()
        assert "Name: sys_prompt" in result

    def test_empty_content(self) -> None:
        """Test SystemMessage with empty content."""
        msg = SystemMessage(content="")
        assert msg.content == ""
        assert msg.text == ""

    def test_empty_list_content(self) -> None:
        """Test SystemMessage with empty list content."""
        msg = SystemMessage(content=[])
        assert msg.content == []
        assert msg.text == ""

    def test_developer_role_via_additional_kwargs(self) -> None:
        """Test SystemMessage with developer role via additional_kwargs."""
        msg = SystemMessage(
            content="Developer instructions",
            additional_kwargs={"__openai_role__": "developer"},
        )
        assert msg.additional_kwargs["__openai_role__"] == "developer"


class TestSystemMessageChunk:
    """Tests for the SystemMessageChunk class."""

    def test_init_basic(self) -> None:
        """Test basic SystemMessageChunk initialization."""
        chunk = SystemMessageChunk(content="Instructions")
        assert chunk.content == "Instructions"
        assert chunk.type == "SystemMessageChunk"

    def test_type_is_system_message_chunk(self) -> None:
        """Test that SystemMessageChunk type is 'SystemMessageChunk'."""
        chunk = SystemMessageChunk(content="Test")
        assert chunk.type == "SystemMessageChunk"

    def test_add_two_chunks(self) -> None:
        """Test adding two SystemMessageChunks."""
        chunk1 = SystemMessageChunk(content="Hello", id="1")
        chunk2 = SystemMessageChunk(content=" world")
        result = chunk1 + chunk2
        assert isinstance(result, SystemMessageChunk)
        assert result.content == "Hello world"
        assert result.id == "1"

    def test_add_with_additional_kwargs(self) -> None:
        """Test adding chunks with additional_kwargs."""
        chunk1 = SystemMessageChunk(
            content="Hello",
            additional_kwargs={"key1": "value1"},
        )
        chunk2 = SystemMessageChunk(
            content=" world",
            additional_kwargs={"key2": "value2"},
        )
        result = chunk1 + chunk2
        assert result.additional_kwargs["key1"] == "value1"
        assert result.additional_kwargs["key2"] == "value2"

    def test_add_with_response_metadata(self) -> None:
        """Test adding chunks with response_metadata."""
        chunk1 = SystemMessageChunk(
            content="Hello",
            response_metadata={"meta1": "data1"},
        )
        chunk2 = SystemMessageChunk(
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
        chunk1 = SystemMessageChunk(content=[{"type": "text", "text": "Hello"}])
        chunk2 = SystemMessageChunk(content=[{"type": "text", "text": " world"}])
        result = chunk1 + chunk2
        assert isinstance(result.content, list)
        # Items without 'index' key are appended, not merged
        assert len(result.content) == 2
        assert result.content[0]["text"] == "Hello"
        assert result.content[1]["text"] == " world"

    def test_add_with_list_content_with_index(self) -> None:
        """Test adding chunks with list content that have matching index keys."""
        chunk1 = SystemMessageChunk(
            content=[{"type": "text", "text": "Hello", "index": 0}]
        )
        chunk2 = SystemMessageChunk(
            content=[{"type": "text", "text": " world", "index": 0}]
        )
        result = chunk1 + chunk2
        assert isinstance(result.content, list)
        # Items with same 'index' key are merged
        assert len(result.content) == 1
        assert result.content[0]["text"] == "Hello world"
        assert result.content[0]["index"] == 0

    def test_add_preserves_id(self) -> None:
        """Test that adding chunks preserves the ID from the first chunk."""
        chunk1 = SystemMessageChunk(content="Hello", id="original-id")
        chunk2 = SystemMessageChunk(content=" world", id="other-id")
        result = chunk1 + chunk2
        assert result.id == "original-id"

    def test_add_list_of_chunks(self) -> None:
        """Test adding a list of chunks to a chunk."""
        chunk1 = SystemMessageChunk(content="a", id="1")
        chunk2 = SystemMessageChunk(content="b")
        chunk3 = SystemMessageChunk(content="c")
        result = chunk1 + [chunk2, chunk3]
        assert result.content == "abc"
        assert result.id == "1"

    def test_serialization_roundtrip(self) -> None:
        """Test SystemMessageChunk serialization and deserialization."""
        chunk = SystemMessageChunk(
            content="Instructions",
            name="sys_prompt",
            id="chunk-123",
        )
        dumped = dumpd(chunk)
        assert dumped["type"] == "constructor"
        assert dumped["id"] == ["langchain", "schema", "messages", "SystemMessageChunk"]

        loaded = load(dumped)
        assert isinstance(loaded, SystemMessageChunk)
        assert loaded.content == "Instructions"
        assert loaded.name == "sys_prompt"
        assert loaded.id == "chunk-123"

    def test_multiple_chunks_addition(self) -> None:
        """Test adding multiple chunks together."""
        chunk1 = SystemMessageChunk(content="a")
        chunk2 = SystemMessageChunk(content="b")
        chunk3 = SystemMessageChunk(content="c")
        result = chunk1 + chunk2 + chunk3
        assert result.content == "abc"

    def test_empty_content_chunks(self) -> None:
        """Test adding chunks with empty content."""
        chunk1 = SystemMessageChunk(content="Hello")
        chunk2 = SystemMessageChunk(content="")
        result = chunk1 + chunk2
        assert result.content == "Hello"

    def test_add_different_chunk_type(self) -> None:
        """Test adding SystemMessageChunk to another chunk type."""
        chunk1 = SystemMessageChunk(content="Hello", id="1")
        chunk2 = HumanMessageChunk(content=" world")
        # Adding different types falls through to base implementation
        result = chunk1 + chunk2
        assert isinstance(result, SystemMessageChunk)
        assert result.content == "Hello world"

    def test_add_incompatible_type_raises_error(self) -> None:
        """Test that adding incompatible types raises TypeError."""
        chunk = SystemMessageChunk(content="Hello")
        with pytest.raises(TypeError):
            chunk + "not a chunk"

    def test_text_property(self) -> None:
        """Test the .text property on chunk."""
        chunk = SystemMessageChunk(content="Hello world")
        assert chunk.text == "Hello world"

    def test_content_blocks_property(self) -> None:
        """Test the content_blocks property on chunk."""
        chunk = SystemMessageChunk(content="Instructions")
        blocks = chunk.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Instructions"


class TestSystemMessageDeveloperRole:
    """Tests for SystemMessage with developer role (OpenAI specific)."""

    def test_developer_role_preserved_in_serialization(self) -> None:
        """Test that developer role is preserved after serialization."""
        msg = SystemMessage(
            content="Developer instructions",
            additional_kwargs={"__openai_role__": "developer"},
        )
        dumped = dumpd(msg)
        loaded = load(dumped)

        assert isinstance(loaded, SystemMessage)
        assert loaded.additional_kwargs["__openai_role__"] == "developer"

    def test_multiple_system_messages_with_different_roles(self) -> None:
        """Test multiple system messages with different roles."""
        system_msg = SystemMessage(content="System instructions")
        developer_msg = SystemMessage(
            content="Developer instructions",
            additional_kwargs={"__openai_role__": "developer"},
        )

        assert "__openai_role__" not in system_msg.additional_kwargs
        assert developer_msg.additional_kwargs["__openai_role__"] == "developer"


class TestSystemMessageModelDumpSnapshot:
    """Tests for SystemMessage model_dump snapshot verification."""

    def test_model_dump_exact_keys_and_values(self) -> None:
        """Test that model_dump() returns exact expected keys and values."""
        msg = SystemMessage(content="Be helpful", id="sys-001", name="prompt")
        dumped = msg.model_dump()
        assert dumped["content"] == "Be helpful"
        assert dumped["type"] == "system"
        assert dumped["name"] == "prompt"
        assert dumped["id"] == "sys-001"
        assert dumped["additional_kwargs"] == {}
        assert dumped["response_metadata"] == {}

    def test_model_dump_default_values(self) -> None:
        """Test that model_dump() has correct defaults for minimal message."""
        msg = SystemMessage(content="Instructions")
        dumped = msg.model_dump()
        assert dumped["content"] == "Instructions"
        assert dumped["type"] == "system"
        assert dumped["name"] is None
        assert dumped["id"] is None
        assert dumped["additional_kwargs"] == {}
        assert dumped["response_metadata"] == {}


class TestSystemMessageChunkContentBlocksEmpty:
    """Tests for SystemMessageChunk content_blocks with empty content."""

    def test_content_blocks_empty_string(self) -> None:
        """Test content_blocks with empty string content."""
        chunk = SystemMessageChunk(content="")
        blocks = chunk.content_blocks
        assert blocks == []

    def test_content_blocks_empty_list(self) -> None:
        """Test content_blocks with empty list content."""
        chunk = SystemMessageChunk(content=[])
        blocks = chunk.content_blocks
        assert blocks == []


class TestSystemMessageContentBlocksInit:
    """Tests for SystemMessage init with content_blocks parameter."""

    def test_init_with_content_blocks_sets_content(self) -> None:
        """Test that content_blocks parameter sets content equal to blocks."""
        blocks = [
            {"type": "text", "text": "First instruction"},
            {"type": "text", "text": "Second instruction"},
        ]
        msg = SystemMessage(content_blocks=blocks)
        assert msg.content == blocks

    def test_content_blocks_roundtrip(self) -> None:
        """Test that content_blocks init produces correct content_blocks property."""
        blocks = [
            {"type": "text", "text": "Rule 1"},
            {"type": "text", "text": "Rule 2"},
        ]
        msg = SystemMessage(content_blocks=blocks)
        result_blocks = msg.content_blocks
        assert len(result_blocks) == 2
        assert result_blocks[0]["text"] == "Rule 1"
        assert result_blocks[1]["text"] == "Rule 2"


class TestSystemMessageEquality:
    """Tests for SystemMessage equality comparison."""

    def test_same_content_messages_are_equal(self) -> None:
        """Test that two SystemMessages with same content are equal."""
        msg1 = SystemMessage(content="Be helpful")
        msg2 = SystemMessage(content="Be helpful")
        assert msg1 == msg2

    def test_different_content_messages_are_not_equal(self) -> None:
        """Test that two SystemMessages with different content are not equal."""
        msg1 = SystemMessage(content="Be helpful")
        msg2 = SystemMessage(content="Be concise")
        assert msg1 != msg2

    def test_same_content_different_id_are_not_equal(self) -> None:
        """Test that messages with same content but different IDs are not equal."""
        msg1 = SystemMessage(content="Instructions", id="1")
        msg2 = SystemMessage(content="Instructions", id="2")
        assert msg1 != msg2

    def test_same_content_and_metadata_are_equal(self) -> None:
        """Test that messages with same content and metadata are equal."""
        msg1 = SystemMessage(content="Instructions", name="sys", id="sys-1")
        msg2 = SystemMessage(content="Instructions", name="sys", id="sys-1")
        assert msg1 == msg2


class TestSystemMessageSerializableNamespace:
    """Tests for SystemMessage is_lc_serializable and get_lc_namespace."""

    def test_is_lc_serializable(self) -> None:
        """Test that SystemMessage is LangChain serializable."""
        assert SystemMessage.is_lc_serializable() is True

    def test_get_lc_namespace(self) -> None:
        """Test that SystemMessage returns correct LangChain namespace."""
        namespace = SystemMessage.get_lc_namespace()
        assert namespace == ["langchain", "schema", "messages"]

    def test_instance_is_lc_serializable(self) -> None:
        """Test is_lc_serializable on an instance."""
        msg = SystemMessage(content="Instructions")
        assert msg.is_lc_serializable() is True

    def test_instance_get_lc_namespace(self) -> None:
        """Test get_lc_namespace on an instance."""
        msg = SystemMessage(content="Instructions")
        assert msg.get_lc_namespace() == ["langchain", "schema", "messages"]
