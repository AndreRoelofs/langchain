"""Tests for langchain_core.messages.function module."""

import pytest

from langchain_core.load import dumpd, load
from langchain_core.messages.base import BaseMessageChunk
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk


class TestFunctionMessage:
    """Tests for the FunctionMessage class."""

    def test_init_basic(self) -> None:
        """Test basic FunctionMessage initialization."""
        msg = FunctionMessage(content="Result: 42", name="calculator")
        assert msg.content == "Result: 42"
        assert msg.name == "calculator"
        assert msg.type == "function"

    def test_init_with_id(self) -> None:
        """Test FunctionMessage with ID."""
        msg = FunctionMessage(content="Result", name="func", id="func-123")
        assert msg.id == "func-123"

    def test_init_with_additional_kwargs(self) -> None:
        """Test FunctionMessage with additional_kwargs."""
        msg = FunctionMessage(
            content="Result",
            name="func",
            additional_kwargs={"custom": "value"},
        )
        assert msg.additional_kwargs["custom"] == "value"

    def test_init_with_response_metadata(self) -> None:
        """Test FunctionMessage with response_metadata."""
        msg = FunctionMessage(
            content="Result",
            name="func",
            response_metadata={"model": "gpt-4"},
        )
        assert msg.response_metadata["model"] == "gpt-4"

    def test_init_with_list_content(self) -> None:
        """Test FunctionMessage with list content."""
        content = [{"type": "text", "text": "Result"}]
        msg = FunctionMessage(content=content, name="func")
        assert msg.content == content

    def test_name_is_required(self) -> None:
        """Test that name is required for FunctionMessage."""
        with pytest.raises(Exception):  # Pydantic validation error
            FunctionMessage(content="Result")  # type: ignore[call-arg]

    def test_type_is_function(self) -> None:
        """Test that FunctionMessage type is 'function'."""
        msg = FunctionMessage(content="Result", name="func")
        assert msg.type == "function"

    def test_serialization_roundtrip(self) -> None:
        """Test FunctionMessage serialization and deserialization."""
        msg = FunctionMessage(
            content="Result: 42",
            name="calculator",
            id="func-123",
            additional_kwargs={"status": "success"},
        )
        dumped = dumpd(msg)
        assert dumped["type"] == "constructor"
        assert dumped["id"] == ["langchain", "schema", "messages", "FunctionMessage"]

        loaded = load(dumped)
        assert isinstance(loaded, FunctionMessage)
        assert loaded.content == "Result: 42"
        assert loaded.name == "calculator"
        assert loaded.id == "func-123"
        assert loaded.additional_kwargs["status"] == "success"

    def test_text_property(self) -> None:
        """Test the .text property."""
        msg = FunctionMessage(content="Hello world", name="func")
        assert msg.text == "Hello world"

    def test_text_property_list_content(self) -> None:
        """Test .text property with list content."""
        msg = FunctionMessage(
            content=[
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ],
            name="func",
        )
        assert msg.text == "Part 1Part 2"

    def test_content_blocks_property(self) -> None:
        """Test the content_blocks property."""
        msg = FunctionMessage(content="Result", name="func")
        blocks = msg.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Result"

    def test_pretty_repr(self) -> None:
        """Test pretty_repr output."""
        msg = FunctionMessage(content="Result", name="calculator")
        result = msg.pretty_repr()
        assert "Function Message" in result
        assert "Result" in result
        assert "Name: calculator" in result


class TestFunctionMessageChunk:
    """Tests for the FunctionMessageChunk class."""

    def test_init_basic(self) -> None:
        """Test basic FunctionMessageChunk initialization."""
        chunk = FunctionMessageChunk(content="Result", name="func")
        assert chunk.content == "Result"
        assert chunk.name == "func"
        assert chunk.type == "FunctionMessageChunk"

    def test_type_is_function_message_chunk(self) -> None:
        """Test that FunctionMessageChunk type is 'FunctionMessageChunk'."""
        chunk = FunctionMessageChunk(content="Result", name="func")
        assert chunk.type == "FunctionMessageChunk"

    def test_add_same_name_chunks(self) -> None:
        """Test adding FunctionMessageChunks with the same name."""
        chunk1 = FunctionMessageChunk(content="Hello", name="func", id="1")
        chunk2 = FunctionMessageChunk(content=" world", name="func")
        result = chunk1 + chunk2
        assert isinstance(result, FunctionMessageChunk)
        assert result.content == "Hello world"
        assert result.name == "func"
        assert result.id == "1"

    def test_add_different_name_chunks_raises_error(self) -> None:
        """Test that adding chunks with different names raises ValueError."""
        chunk1 = FunctionMessageChunk(content="Hello", name="func1")
        chunk2 = FunctionMessageChunk(content=" world", name="func2")
        with pytest.raises(ValueError, match="Cannot concatenate.*different names"):
            chunk1 + chunk2

    def test_add_with_additional_kwargs(self) -> None:
        """Test adding chunks with additional_kwargs."""
        chunk1 = FunctionMessageChunk(
            content="Hello",
            name="func",
            additional_kwargs={"key1": "value1"},
        )
        chunk2 = FunctionMessageChunk(
            content=" world",
            name="func",
            additional_kwargs={"key2": "value2"},
        )
        result = chunk1 + chunk2
        assert result.additional_kwargs["key1"] == "value1"
        assert result.additional_kwargs["key2"] == "value2"

    def test_add_with_response_metadata(self) -> None:
        """Test adding chunks with response_metadata."""
        chunk1 = FunctionMessageChunk(
            content="Hello",
            name="func",
            response_metadata={"meta1": "data1"},
        )
        chunk2 = FunctionMessageChunk(
            content=" world",
            name="func",
            response_metadata={"meta2": "data2"},
        )
        result = chunk1 + chunk2
        assert result.response_metadata["meta1"] == "data1"
        assert result.response_metadata["meta2"] == "data2"

    def test_add_with_list_content(self) -> None:
        """Test adding chunks with list content.

        Note: Without an 'index' key, list items are appended, not merged.
        """
        chunk1 = FunctionMessageChunk(
            content=[{"type": "text", "text": "Hello"}],
            name="func",
        )
        chunk2 = FunctionMessageChunk(
            content=[{"type": "text", "text": " world"}],
            name="func",
        )
        result = chunk1 + chunk2
        assert isinstance(result.content, list)
        # Items without 'index' key are appended, not merged
        assert len(result.content) == 2
        assert result.content[0]["text"] == "Hello"
        assert result.content[1]["text"] == " world"

    def test_add_with_list_content_with_index(self) -> None:
        """Test adding chunks with list content that have matching index keys."""
        chunk1 = FunctionMessageChunk(
            content=[{"type": "text", "text": "Hello", "index": 0}],
            name="func",
        )
        chunk2 = FunctionMessageChunk(
            content=[{"type": "text", "text": " world", "index": 0}],
            name="func",
        )
        result = chunk1 + chunk2
        assert isinstance(result.content, list)
        # Items with same 'index' key are merged
        assert len(result.content) == 1
        assert result.content[0]["text"] == "Hello world"
        assert result.content[0]["index"] == 0

    def test_add_preserves_id(self) -> None:
        """Test that adding chunks preserves the ID from the first chunk."""
        chunk1 = FunctionMessageChunk(content="Hello", name="func", id="original-id")
        chunk2 = FunctionMessageChunk(content=" world", name="func", id="other-id")
        result = chunk1 + chunk2
        assert result.id == "original-id"

    def test_serialization_roundtrip(self) -> None:
        """Test FunctionMessageChunk serialization and deserialization."""
        chunk = FunctionMessageChunk(
            content="Result",
            name="calculator",
            id="chunk-123",
        )
        dumped = dumpd(chunk)
        assert dumped["type"] == "constructor"
        assert dumped["id"] == [
            "langchain",
            "schema",
            "messages",
            "FunctionMessageChunk",
        ]

        loaded = load(dumped)
        assert isinstance(loaded, FunctionMessageChunk)
        assert loaded.content == "Result"
        assert loaded.name == "calculator"
        assert loaded.id == "chunk-123"

    def test_multiple_chunks_addition(self) -> None:
        """Test adding multiple chunks together."""
        chunk1 = FunctionMessageChunk(content="a", name="func")
        chunk2 = FunctionMessageChunk(content="b", name="func")
        chunk3 = FunctionMessageChunk(content="c", name="func")
        result = chunk1 + chunk2 + chunk3
        assert result.content == "abc"
        assert result.name == "func"

    def test_empty_content_chunks(self) -> None:
        """Test adding chunks with empty content."""
        chunk1 = FunctionMessageChunk(content="Hello", name="func")
        chunk2 = FunctionMessageChunk(content="", name="func")
        result = chunk1 + chunk2
        assert result.content == "Hello"

    def test_add_incompatible_type_raises_error(self) -> None:
        """Test that adding incompatible types raises TypeError."""
        chunk = FunctionMessageChunk(content="Hello", name="func")
        with pytest.raises(TypeError):
            chunk + "not a chunk"

    def test_text_property(self) -> None:
        """Test the .text property on chunk."""
        chunk = FunctionMessageChunk(content="Hello world", name="func")
        assert chunk.text == "Hello world"

    def test_content_blocks_property(self) -> None:
        """Test the content_blocks property on chunk."""
        chunk = FunctionMessageChunk(content="Result", name="func")
        blocks = chunk.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Result"


class TestFunctionMessageDeprecation:
    """Tests related to FunctionMessage being an older pattern."""

    def test_function_message_vs_tool_message(self) -> None:
        """Test that FunctionMessage lacks tool_call_id unlike ToolMessage."""
        from langchain_core.messages import ToolMessage

        func_msg = FunctionMessage(content="Result", name="func")
        tool_msg = ToolMessage(content="Result", tool_call_id="call-123")

        # FunctionMessage doesn't have tool_call_id
        assert not hasattr(func_msg, "tool_call_id")
        # ToolMessage does have tool_call_id
        assert tool_msg.tool_call_id == "call-123"

    def test_function_message_still_serializable(self) -> None:
        """Test that FunctionMessage is still fully serializable."""
        msg = FunctionMessage(content="test", name="test_func")
        dumped = dumpd(msg)
        loaded = load(dumped)
        assert isinstance(loaded, FunctionMessage)
        assert loaded.content == "test"
        assert loaded.name == "test_func"


class TestFunctionMessageModelDump:
    """Tests for FunctionMessage model_dump snapshot."""

    def test_model_dump_snapshot(self) -> None:
        """Test FunctionMessage model_dump returns exact expected keys and values."""
        msg = FunctionMessage(
            content="Result: 42",
            name="calculator",
            id="func-snap",
            additional_kwargs={"status": "ok"},
            response_metadata={"model": "gpt-4"},
        )
        dumped = msg.model_dump()
        assert dumped == {
            "content": "Result: 42",
            "name": "calculator",
            "id": "func-snap",
            "type": "function",
            "additional_kwargs": {"status": "ok"},
            "response_metadata": {"model": "gpt-4"},
        }


class TestFunctionMessageChunkAddList:
    """Tests for FunctionMessageChunk adding with a list of chunks."""

    def test_add_sequential_chunks(self) -> None:
        """Test adding multiple FunctionMessageChunks sequentially."""
        chunks = [
            FunctionMessageChunk(content="a", name="func", id="id-1"),
            FunctionMessageChunk(content="b", name="func"),
            FunctionMessageChunk(content="c", name="func"),
        ]
        result = chunks[0]
        for chunk in chunks[1:]:
            result = result + chunk
        assert isinstance(result, FunctionMessageChunk)
        assert result.content == "abc"
        assert result.name == "func"
        assert result.id == "id-1"

    def test_add_with_list_of_function_chunks_raises(self) -> None:
        """Test that adding with a list of FunctionMessageChunks raises.

        The list-of-chunks path in BaseMessageChunk.__add__ does not pass the
        'name' kwarg when constructing the result via self.__class__(...),
        which causes a validation error for FunctionMessageChunk.
        """
        from pydantic import ValidationError

        chunk = FunctionMessageChunk(content="a", name="func", id="id-1")
        others = [
            FunctionMessageChunk(content="b", name="func"),
            FunctionMessageChunk(content="c", name="func"),
        ]
        with pytest.raises(ValidationError):
            chunk + others


class TestFunctionMessageEquality:
    """Tests for FunctionMessage equality comparison."""

    def test_equal_messages(self) -> None:
        """Test that two FunctionMessages with same fields are equal."""
        msg1 = FunctionMessage(content="Result", name="func", id="eq-1")
        msg2 = FunctionMessage(content="Result", name="func", id="eq-1")
        assert msg1 == msg2

    def test_different_content_not_equal(self) -> None:
        """Test that FunctionMessages with different content are not equal."""
        msg1 = FunctionMessage(content="Result A", name="func")
        msg2 = FunctionMessage(content="Result B", name="func")
        assert msg1 != msg2

    def test_different_name_not_equal(self) -> None:
        """Test that FunctionMessages with different names are not equal."""
        msg1 = FunctionMessage(content="Result", name="func1")
        msg2 = FunctionMessage(content="Result", name="func2")
        assert msg1 != msg2


class TestFunctionMessageSerializableNamespace:
    """Tests for FunctionMessage is_lc_serializable and get_lc_namespace."""

    def test_is_lc_serializable(self) -> None:
        """Test that FunctionMessage is LangChain serializable."""
        assert FunctionMessage.is_lc_serializable() is True

    def test_get_lc_namespace(self) -> None:
        """Test that FunctionMessage has the expected LangChain namespace."""
        assert FunctionMessage.get_lc_namespace() == [
            "langchain",
            "schema",
            "messages",
        ]

    def test_chunk_is_lc_serializable(self) -> None:
        """Test that FunctionMessageChunk is LangChain serializable."""
        assert FunctionMessageChunk.is_lc_serializable() is True

    def test_chunk_get_lc_namespace(self) -> None:
        """Test that FunctionMessageChunk has the expected LangChain namespace."""
        assert FunctionMessageChunk.get_lc_namespace() == [
            "langchain",
            "schema",
            "messages",
        ]


class TestFunctionMessageChunkContentBlocksEmpty:
    """Tests for FunctionMessageChunk content_blocks with empty content."""

    def test_content_blocks_empty_string(self) -> None:
        """Test content_blocks returns empty list for empty string content."""
        chunk = FunctionMessageChunk(content="", name="func")
        assert chunk.content_blocks == []

    def test_content_blocks_empty_list(self) -> None:
        """Test content_blocks returns empty list for empty list content."""
        chunk = FunctionMessageChunk(content=[], name="func")
        assert chunk.content_blocks == []


class TestFunctionMessageEmptyContent:
    """Tests for FunctionMessage with empty string content."""

    def test_init_empty_string_content(self) -> None:
        """Test FunctionMessage initializes with empty string content."""
        msg = FunctionMessage(content="", name="func")
        assert msg.content == ""
        assert msg.name == "func"
        assert msg.type == "function"

    def test_empty_content_text_property(self) -> None:
        """Test .text property returns empty string for empty content."""
        msg = FunctionMessage(content="", name="func")
        assert msg.text == ""

    def test_empty_content_blocks(self) -> None:
        """Test content_blocks returns empty list for empty string content."""
        msg = FunctionMessage(content="", name="func")
        assert msg.content_blocks == []


class TestFunctionMessageChunkFallthrough:
    """Tests for FunctionMessageChunk falling through to BaseMessageChunk.__add__."""

    def test_add_with_non_function_chunk_raises(self) -> None:
        """Test adding FunctionMessageChunk with a non-FunctionMessageChunk.

        When the other chunk is not a FunctionMessageChunk, the __add__ method
        falls through to BaseMessageChunk.__add__. However, BaseMessageChunk
        constructs the result via self.__class__(...) without passing 'name',
        which causes a ValidationError since FunctionMessageChunk requires it.
        """
        from pydantic import ValidationError

        from langchain_core.messages.human import HumanMessageChunk

        func_chunk = FunctionMessageChunk(content="Hello", name="func", id="func-id")
        human_chunk = HumanMessageChunk(content=" world")
        with pytest.raises(ValidationError):
            func_chunk + human_chunk
