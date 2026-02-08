"""Unit tests for ChatGeneration and ChatGenerationChunk classes."""

import pytest

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, Generation
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

    def test_merge_chunks_all_none_generation_info(self) -> None:
        """Test merging chunks where all have None generation_info."""
        chunks = [
            ChatGenerationChunk(message=AIMessageChunk(content="A")),
            ChatGenerationChunk(message=AIMessageChunk(content="B")),
            ChatGenerationChunk(message=AIMessageChunk(content="C")),
        ]
        result = merge_chat_generation_chunks(chunks)
        assert result is not None
        assert result.text == "ABC"
        assert result.generation_info is None

    def test_merge_chunks_returns_chat_generation_chunk_type(self) -> None:
        """Test that merge returns ChatGenerationChunk type."""
        chunks = [
            ChatGenerationChunk(message=AIMessageChunk(content="A")),
            ChatGenerationChunk(message=AIMessageChunk(content="B")),
        ]
        result = merge_chat_generation_chunks(chunks)
        assert result is not None
        assert type(result) is ChatGenerationChunk


class TestChatGenerationInheritance:
    """Test suite for ChatGeneration inheritance behavior."""

    def test_chat_generation_inherits_from_generation(self) -> None:
        """Test that ChatGeneration is a subclass of Generation."""
        gen = ChatGeneration(message=AIMessage(content="test"))
        assert isinstance(gen, Generation)

    def test_chat_generation_is_lc_serializable(self) -> None:
        """Test that ChatGeneration inherits serializable property."""
        assert ChatGeneration.is_lc_serializable() is True

    def test_chat_generation_get_lc_namespace(self) -> None:
        """Test that ChatGeneration inherits namespace."""
        assert ChatGeneration.get_lc_namespace() == ["langchain", "schema", "output"]

    def test_chat_generation_chunk_inherits_from_generation(self) -> None:
        """Test that ChatGenerationChunk is a subclass of Generation."""
        chunk = ChatGenerationChunk(message=AIMessageChunk(content="test"))
        assert isinstance(chunk, Generation)
        assert isinstance(chunk, ChatGeneration)
        assert isinstance(chunk, ChatGenerationChunk)


class TestChatGenerationTextExtraction:
    """Test suite for ChatGeneration text extraction edge cases."""

    def test_empty_string_content(self) -> None:
        """Test text extraction from empty string content."""
        gen = ChatGeneration(message=AIMessage(content=""))
        assert gen.text == ""

    def test_content_list_with_only_non_text_dicts(self) -> None:
        """Test text extraction with list of dicts that have no 'text' key."""
        content = [
            {"type": "image", "url": "http://example.com/img.png"},
            {"type": "tool_use", "id": "123"},
        ]
        gen = ChatGeneration(message=AIMessage(content=content))
        assert gen.text == ""

    def test_content_list_dict_text_first(self) -> None:
        """Test text extraction picks first dict with 'text' key."""
        content = [
            {"text": "first_text", "type": "text"},
            {"text": "second_text", "type": "text"},
        ]
        gen = ChatGeneration(message=AIMessage(content=content))
        assert gen.text == "first_text"

    def test_content_list_string_before_dict(self) -> None:
        """Test that string block is picked before dict block."""
        content = ["string_block", {"text": "dict_text", "type": "text"}]
        gen = ChatGeneration(message=AIMessage(content=content))
        assert gen.text == "string_block"

    def test_content_list_non_text_dict_then_string(self) -> None:
        """Test that non-text dict is skipped, string is picked."""
        content = [
            {"type": "tool_use", "id": "123"},
            "my_text",
        ]
        gen = ChatGeneration(message=AIMessage(content=content))
        assert gen.text == "my_text"

    def test_content_list_non_text_dict_then_text_dict(self) -> None:
        """Test that non-text dict is skipped, text dict is picked."""
        content = [
            {"type": "tool_use", "id": "123"},
            {"text": "found_text", "type": "text"},
        ]
        gen = ChatGeneration(message=AIMessage(content=content))
        assert gen.text == "found_text"

    def test_text_not_set_directly_is_overridden(self) -> None:
        """Test that explicitly set text is overridden by model_validator."""
        gen = ChatGeneration(
            text="should_be_overridden",
            message=AIMessage(content="from_message"),
        )
        assert gen.text == "from_message"

    def test_with_human_message(self) -> None:
        """Test ChatGeneration with a HumanMessage."""
        gen = ChatGeneration(message=HumanMessage(content="user input"))
        assert gen.text == "user input"

    def test_content_list_empty_string_block(self) -> None:
        """Test that empty string block in list is still selected."""
        content = ["", {"text": "dict_text", "type": "text"}]
        gen = ChatGeneration(message=AIMessage(content=content))
        assert gen.text == ""


class TestChatGenerationSerialization:
    """Test suite for ChatGeneration serialization roundtrips."""

    def test_model_dump_basic(self) -> None:
        """Test model_dump for ChatGeneration."""
        gen = ChatGeneration(message=AIMessage(content="Hello"))
        data = gen.model_dump()
        assert data["text"] == "Hello"
        assert data["type"] == "ChatGeneration"
        assert "message" in data

    def test_model_dump_with_generation_info(self) -> None:
        """Test model_dump includes generation_info."""
        gen = ChatGeneration(
            message=AIMessage(content="test"),
            generation_info={"finish_reason": "stop"},
        )
        data = gen.model_dump()
        assert data["generation_info"] == {"finish_reason": "stop"}

    def test_chat_generation_chunk_model_dump(self) -> None:
        """Test model_dump for ChatGenerationChunk."""
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="chunk"),
            generation_info={"key": "val"},
        )
        data = chunk.model_dump()
        assert data["text"] == "chunk"
        assert data["type"] == "ChatGenerationChunk"
        assert data["generation_info"] == {"key": "val"}


class TestChatGenerationChunkMergingEdgeCases:
    """Test suite for ChatGenerationChunk merging edge cases."""

    def test_add_list_with_mixed_none_generation_info(self) -> None:
        """Test adding list where some chunks have None generation_info."""
        chunk1 = ChatGenerationChunk(
            message=AIMessageChunk(content="A"),
            generation_info={"k1": "v1"},
        )
        chunk_list = [
            ChatGenerationChunk(
                message=AIMessageChunk(content="B"),
                generation_info=None,
            ),
            ChatGenerationChunk(
                message=AIMessageChunk(content="C"),
                generation_info={"k3": "v3"},
            ),
        ]
        result = chunk1 + chunk_list
        assert result.text == "ABC"
        assert result.generation_info is not None
        assert result.generation_info["k1"] == "v1"
        assert result.generation_info["k3"] == "v3"

    def test_add_list_all_none_generation_info(self) -> None:
        """Test adding list where all chunks have None generation_info."""
        chunk1 = ChatGenerationChunk(message=AIMessageChunk(content="A"))
        chunk_list = [
            ChatGenerationChunk(message=AIMessageChunk(content="B")),
            ChatGenerationChunk(message=AIMessageChunk(content="C")),
        ]
        result = chunk1 + chunk_list
        assert result.text == "ABC"
        assert result.generation_info is None

    def test_add_with_none_raises_error(self) -> None:
        """Test that adding None raises TypeError."""
        chunk = ChatGenerationChunk(message=AIMessageChunk(content="test"))
        with pytest.raises(TypeError):
            _ = chunk + None  # type: ignore[operator]

    def test_add_with_int_raises_error(self) -> None:
        """Test that adding int raises TypeError."""
        chunk = ChatGenerationChunk(message=AIMessageChunk(content="test"))
        with pytest.raises(TypeError):
            _ = chunk + 42  # type: ignore[operator]

    def test_add_returns_correct_type(self) -> None:
        """Test that addition returns ChatGenerationChunk type."""
        chunk1 = ChatGenerationChunk(message=AIMessageChunk(content="A"))
        chunk2 = ChatGenerationChunk(message=AIMessageChunk(content="B"))
        result = chunk1 + chunk2
        assert type(result) is ChatGenerationChunk

    def test_add_list_returns_correct_type(self) -> None:
        """Test that list addition returns ChatGenerationChunk type."""
        chunk1 = ChatGenerationChunk(message=AIMessageChunk(content="A"))
        result = chunk1 + [
            ChatGenerationChunk(message=AIMessageChunk(content="B")),
        ]
        assert type(result) is ChatGenerationChunk

    def test_merge_generation_info_with_nested_dicts(self) -> None:
        """Test merging generation_info with nested dicts."""
        chunk1 = ChatGenerationChunk(
            message=AIMessageChunk(content="A"),
            generation_info={"meta": {"key1": "val1"}},
        )
        chunk2 = ChatGenerationChunk(
            message=AIMessageChunk(content="B"),
            generation_info={"meta": {"key2": "val2"}},
        )
        result = chunk1 + chunk2
        assert result.generation_info is not None
        assert result.generation_info["meta"]["key1"] == "val1"
        assert result.generation_info["meta"]["key2"] == "val2"

    def test_merge_generation_info_with_int_values(self) -> None:
        """Test merging generation_info with integer values (addition)."""
        chunk1 = ChatGenerationChunk(
            message=AIMessageChunk(content="A"),
            generation_info={"tokens": 10},
        )
        chunk2 = ChatGenerationChunk(
            message=AIMessageChunk(content="B"),
            generation_info={"tokens": 20},
        )
        result = chunk1 + chunk2
        assert result.generation_info is not None
        assert result.generation_info["tokens"] == 30

    def test_sequential_add_chunks(self) -> None:
        """Test sequential addition of multiple chunks."""
        c1 = ChatGenerationChunk(
            message=AIMessageChunk(content="A"),
            generation_info={"k1": "v1"},
        )
        c2 = ChatGenerationChunk(
            message=AIMessageChunk(content="B"),
            generation_info={"k2": "v2"},
        )
        c3 = ChatGenerationChunk(
            message=AIMessageChunk(content="C"),
            generation_info={"k3": "v3"},
        )
        result = c1 + c2 + c3
        assert result.text == "ABC"
        assert result.generation_info is not None
        assert result.generation_info["k1"] == "v1"
        assert result.generation_info["k2"] == "v2"
        assert result.generation_info["k3"] == "v3"

    def test_add_empty_content_chunks(self) -> None:
        """Test adding chunks with empty content."""
        chunk1 = ChatGenerationChunk(message=AIMessageChunk(content=""))
        chunk2 = ChatGenerationChunk(message=AIMessageChunk(content=""))
        result = chunk1 + chunk2
        assert result.text == ""
