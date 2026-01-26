"""Unit tests for ChatGeneration and ChatGenerationChunk classes."""

import pytest

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk
from langchain_core.outputs.chat_generation import merge_chat_generation_chunks


class TestChatGeneration:
    """Test suite for ChatGeneration class."""

    @pytest.mark.parametrize(
        "content",
        [
            "foo",
            ["foo"],
            [{"text": "foo", "type": "text"}],
            [
                {"tool_use": {}, "type": "tool_use"},
                {"text": "foo", "type": "text"},
                "bar",
            ],
        ],
    )
    def test_msg_with_text(self, content: str | list) -> None:
        """Test that text is extracted correctly from various content formats."""
        expected = "foo"
        actual = ChatGeneration(message=AIMessage(content=content)).text
        assert actual == expected

    @pytest.mark.parametrize("content", [[], [{"tool_use": {}, "type": "tool_use"}]])
    def test_msg_no_text(self, content: str | list) -> None:
        """Test that empty text is returned when no text content is present."""
        expected = ""
        actual = ChatGeneration(message=AIMessage(content=content)).text
        assert actual == expected

    def test_creation_with_string_content(self) -> None:
        """Test creating ChatGeneration with string content."""
        msg = AIMessage(content="Hello, world!")
        gen = ChatGeneration(message=msg)
        assert gen.text == "Hello, world!"
        assert gen.message == msg
        assert gen.type == "ChatGeneration"

    def test_creation_with_generation_info(self) -> None:
        """Test creating ChatGeneration with generation_info."""
        msg = AIMessage(content="Test")
        gen_info = {"finish_reason": "stop", "model": "gpt-4"}
        gen = ChatGeneration(message=msg, generation_info=gen_info)
        assert gen.text == "Test"
        assert gen.generation_info == gen_info

    def test_text_extraction_from_dict_block(self) -> None:
        """Test text extraction from dictionary content blocks."""
        content = [{"text": "extracted", "type": "text"}]
        gen = ChatGeneration(message=AIMessage(content=content))
        assert gen.text == "extracted"

    def test_text_extraction_prioritizes_string_in_list(self) -> None:
        """Test that string in list is prioritized over dict blocks."""
        content = ["first_string", {"text": "dict_text", "type": "text"}]
        gen = ChatGeneration(message=AIMessage(content=content))
        assert gen.text == "first_string"

    def test_type_field_is_literal(self) -> None:
        """Test that type field is set correctly."""
        gen = ChatGeneration(message=AIMessage(content="test"))
        assert gen.type == "ChatGeneration"


class TestChatGenerationChunk:
    """Test suite for ChatGenerationChunk class."""

    def test_creation(self) -> None:
        """Test creating a ChatGenerationChunk."""
        msg = AIMessageChunk(content="chunk")
        chunk = ChatGenerationChunk(message=msg)
        assert chunk.text == "chunk"
        assert chunk.message == msg
        assert chunk.type == "ChatGenerationChunk"

    def test_add_two_chunks(self) -> None:
        """Test concatenating two ChatGenerationChunks."""
        chunk1 = ChatGenerationChunk(message=AIMessageChunk(content="Hello, "))
        chunk2 = ChatGenerationChunk(message=AIMessageChunk(content="world!"))
        result = chunk1 + chunk2
        assert isinstance(result, ChatGenerationChunk)
        assert result.text == "Hello, world!"
        assert result.generation_info is None

    def test_add_chunks_with_generation_info(self) -> None:
        """Test concatenating chunks with generation_info."""
        chunk1 = ChatGenerationChunk(
            message=AIMessageChunk(content="Hello"),
            generation_info={"key1": "value1", "shared": "first"},
        )
        chunk2 = ChatGenerationChunk(
            message=AIMessageChunk(content=" world"),
            generation_info={"key2": "value2", "shared": "second"},
        )
        result = chunk1 + chunk2
        assert result.text == "Hello world"
        assert result.generation_info is not None
        assert result.generation_info["key1"] == "value1"
        assert result.generation_info["key2"] == "value2"
        # String values are concatenated in merge_dicts
        assert result.generation_info["shared"] == "firstsecond"

    def test_add_chunk_with_none_generation_info(self) -> None:
        """Test concatenating chunks where one has None generation_info."""
        chunk1 = ChatGenerationChunk(
            message=AIMessageChunk(content="Hello"),
            generation_info={"key": "value"},
        )
        chunk2 = ChatGenerationChunk(
            message=AIMessageChunk(content=" world"),
            generation_info=None,
        )
        result = chunk1 + chunk2
        assert result.text == "Hello world"
        assert result.generation_info == {"key": "value"}

    def test_add_chunks_both_none_generation_info(self) -> None:
        """Test concatenating chunks where both have None generation_info."""
        chunk1 = ChatGenerationChunk(
            message=AIMessageChunk(content="Hello"),
            generation_info=None,
        )
        chunk2 = ChatGenerationChunk(
            message=AIMessageChunk(content=" world"),
            generation_info=None,
        )
        result = chunk1 + chunk2
        assert result.text == "Hello world"
        assert result.generation_info is None

    def test_add_list_of_chunks(self) -> None:
        """Test concatenating a chunk with a list of chunks."""
        chunk1 = ChatGenerationChunk(message=AIMessageChunk(content="A"))
        chunk2 = ChatGenerationChunk(message=AIMessageChunk(content="B"))
        chunk3 = ChatGenerationChunk(message=AIMessageChunk(content="C"))
        result = chunk1 + [chunk2, chunk3]
        assert isinstance(result, ChatGenerationChunk)
        assert result.text == "ABC"

    def test_add_list_of_chunks_with_generation_info(self) -> None:
        """Test concatenating with list of chunks that have generation_info."""
        chunk1 = ChatGenerationChunk(
            message=AIMessageChunk(content="A"),
            generation_info={"key1": "value1"},
        )
        chunk2 = ChatGenerationChunk(
            message=AIMessageChunk(content="B"),
            generation_info={"key2": "value2"},
        )
        chunk3 = ChatGenerationChunk(
            message=AIMessageChunk(content="C"),
            generation_info={"key3": "value3"},
        )
        result = chunk1 + [chunk2, chunk3]
        assert result.text == "ABC"
        assert result.generation_info is not None
        assert result.generation_info["key1"] == "value1"
        assert result.generation_info["key2"] == "value2"
        assert result.generation_info["key3"] == "value3"

    def test_add_empty_list(self) -> None:
        """Test concatenating with an empty list."""
        chunk = ChatGenerationChunk(message=AIMessageChunk(content="test"))
        result = chunk + []
        assert result.text == "test"

    def test_add_with_invalid_type_raises_error(self) -> None:
        """Test that adding non-ChatGenerationChunk raises TypeError."""
        chunk = ChatGenerationChunk(message=AIMessageChunk(content="test"))
        with pytest.raises(TypeError) as exc_info:
            _ = chunk + "invalid"  # type: ignore[operator]
        assert "unsupported operand type(s) for +" in str(exc_info.value)

    def test_add_with_chat_generation_raises_error(self) -> None:
        """Test that adding ChatGeneration (not chunk) raises TypeError."""
        chunk = ChatGenerationChunk(message=AIMessageChunk(content="test"))
        gen = ChatGeneration(message=AIMessage(content="other"))
        with pytest.raises(TypeError) as exc_info:
            _ = chunk + gen  # type: ignore[operator]
        assert "unsupported operand type(s) for +" in str(exc_info.value)

    def test_inherits_from_chat_generation(self) -> None:
        """Test that ChatGenerationChunk inherits from ChatGeneration."""
        chunk = ChatGenerationChunk(message=AIMessageChunk(content="test"))
        assert isinstance(chunk, ChatGeneration)
        assert isinstance(chunk, ChatGenerationChunk)

    def test_type_field_is_literal(self) -> None:
        """Test that type field is set correctly."""
        chunk = ChatGenerationChunk(message=AIMessageChunk(content="test"))
        assert chunk.type == "ChatGenerationChunk"


class TestMergeChatGenerationChunks:
    """Test suite for merge_chat_generation_chunks function."""

    def test_merge_empty_list(self) -> None:
        """Test merging an empty list returns None."""
        result = merge_chat_generation_chunks([])
        assert result is None

    def test_merge_single_chunk(self) -> None:
        """Test merging a single chunk returns the chunk itself."""
        chunk = ChatGenerationChunk(message=AIMessageChunk(content="single"))
        result = merge_chat_generation_chunks([chunk])
        assert result is chunk

    def test_merge_two_chunks(self) -> None:
        """Test merging two chunks."""
        chunk1 = ChatGenerationChunk(message=AIMessageChunk(content="Hello "))
        chunk2 = ChatGenerationChunk(message=AIMessageChunk(content="world"))
        result = merge_chat_generation_chunks([chunk1, chunk2])
        assert result is not None
        assert result.text == "Hello world"

    def test_merge_multiple_chunks(self) -> None:
        """Test merging multiple chunks."""
        chunks = [
            ChatGenerationChunk(message=AIMessageChunk(content="A")),
            ChatGenerationChunk(message=AIMessageChunk(content="B")),
            ChatGenerationChunk(message=AIMessageChunk(content="C")),
            ChatGenerationChunk(message=AIMessageChunk(content="D")),
        ]
        result = merge_chat_generation_chunks(chunks)
        assert result is not None
        assert result.text == "ABCD"

    def test_merge_chunks_with_generation_info(self) -> None:
        """Test merging chunks preserves and merges generation_info."""
        chunks = [
            ChatGenerationChunk(
                message=AIMessageChunk(content="A"),
                generation_info={"key1": "value1"},
            ),
            ChatGenerationChunk(
                message=AIMessageChunk(content="B"),
                generation_info={"key2": "value2"},
            ),
        ]
        result = merge_chat_generation_chunks(chunks)
        assert result is not None
        assert result.generation_info is not None
        assert result.generation_info["key1"] == "value1"
        assert result.generation_info["key2"] == "value2"
