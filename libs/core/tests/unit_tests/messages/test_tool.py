"""Tests for langchain_core.messages.tool module."""

import json
from uuid import UUID

import pytest

from langchain_core.load import dumpd, load
from langchain_core.messages.tool import (
    ToolCall,
    ToolCallChunk,
    ToolMessage,
    ToolMessageChunk,
    ToolOutputMixin,
    default_tool_chunk_parser,
    default_tool_parser,
    invalid_tool_call,
    tool_call,
    tool_call_chunk,
)


class TestToolMessage:
    """Tests for the ToolMessage class."""

    def test_init_basic(self) -> None:
        """Test basic ToolMessage initialization."""
        msg = ToolMessage(content="Result: 42", tool_call_id="call-123")
        assert msg.content == "Result: 42"
        assert msg.tool_call_id == "call-123"
        assert msg.type == "tool"
        assert msg.status == "success"

    def test_init_with_name(self) -> None:
        """Test ToolMessage with name."""
        msg = ToolMessage(content="Result", tool_call_id="call-123", name="calculator")
        assert msg.name == "calculator"

    def test_init_with_id(self) -> None:
        """Test ToolMessage with ID."""
        msg = ToolMessage(content="Result", tool_call_id="call-123", id="msg-123")
        assert msg.id == "msg-123"

    def test_init_with_artifact(self) -> None:
        """Test ToolMessage with artifact."""
        artifact = {"image": "base64_data", "metadata": {"width": 100}}
        msg = ToolMessage(
            content="Image generated", tool_call_id="call-123", artifact=artifact
        )
        assert msg.artifact == artifact

    def test_init_with_status_success(self) -> None:
        """Test ToolMessage with success status."""
        msg = ToolMessage(content="Result", tool_call_id="call-123", status="success")
        assert msg.status == "success"

    def test_init_with_status_error(self) -> None:
        """Test ToolMessage with error status."""
        msg = ToolMessage(
            content="Error: Division by zero", tool_call_id="call-123", status="error"
        )
        assert msg.status == "error"

    def test_init_with_list_content(self) -> None:
        """Test ToolMessage with list content."""
        content = [{"type": "text", "text": "Result"}]
        msg = ToolMessage(content=content, tool_call_id="call-123")
        assert msg.content == content

    def test_init_with_content_blocks(self) -> None:
        """Test ToolMessage with content_blocks parameter."""
        blocks = [
            {"type": "text", "text": "Result: 42"},
            {"type": "text", "text": "Calculation complete"},
        ]
        msg = ToolMessage(content_blocks=blocks, tool_call_id="call-123")
        assert msg.content == blocks

    def test_tool_call_id_is_required(self) -> None:
        """Test that tool_call_id is required."""
        with pytest.raises(Exception):  # Pydantic validation error
            ToolMessage(content="Result")  # type: ignore[call-arg]

    def test_tool_call_id_coerced_to_string(self) -> None:
        """Test that tool_call_id is coerced to string."""
        # UUID type
        msg1 = ToolMessage(
            content="Result", tool_call_id=UUID("12345678-1234-5678-1234-567812345678")
        )
        assert isinstance(msg1.tool_call_id, str)

        # Integer type
        msg2 = ToolMessage(content="Result", tool_call_id=12345)
        assert msg2.tool_call_id == "12345"

        # Float type
        msg3 = ToolMessage(content="Result", tool_call_id=123.45)
        assert msg3.tool_call_id == "123.45"

    def test_type_is_tool(self) -> None:
        """Test that ToolMessage type is 'tool'."""
        msg = ToolMessage(content="Test", tool_call_id="call-123")
        assert msg.type == "tool"

    def test_serialization_roundtrip(self) -> None:
        """Test ToolMessage serialization and deserialization."""
        msg = ToolMessage(
            content="Result: 42",
            tool_call_id="call-123",
            name="calculator",
            id="msg-123",
            artifact={"data": "value"},
            status="success",
        )
        dumped = dumpd(msg)
        assert dumped["type"] == "constructor"
        assert dumped["id"] == ["langchain", "schema", "messages", "ToolMessage"]

        loaded = load(dumped)
        assert isinstance(loaded, ToolMessage)
        assert loaded.content == "Result: 42"
        assert loaded.tool_call_id == "call-123"
        assert loaded.name == "calculator"
        assert loaded.id == "msg-123"
        assert loaded.artifact == {"data": "value"}
        assert loaded.status == "success"

    def test_text_property(self) -> None:
        """Test the .text property."""
        msg = ToolMessage(content="Hello world", tool_call_id="call-123")
        assert msg.text == "Hello world"

    def test_text_property_list_content(self) -> None:
        """Test .text property with list content."""
        msg = ToolMessage(
            content=[
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ],
            tool_call_id="call-123",
        )
        assert msg.text == "Part 1Part 2"

    def test_content_blocks_property(self) -> None:
        """Test the content_blocks property."""
        msg = ToolMessage(content="Result", tool_call_id="call-123")
        blocks = msg.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Result"

    def test_pretty_repr(self) -> None:
        """Test pretty_repr output."""
        msg = ToolMessage(content="Result", tool_call_id="call-123", name="calculator")
        result = msg.pretty_repr()
        assert "Tool Message" in result
        assert "Result" in result

    def test_content_coercion_non_string(self) -> None:
        """Test that non-string/non-list content is coerced to string."""
        msg = ToolMessage(content=42, tool_call_id="call-123")  # type: ignore[arg-type]
        assert msg.content == "42"

    def test_content_coercion_list_with_non_dict(self) -> None:
        """Test that list content with non-dict items is coerced."""
        msg = ToolMessage(content=[123, 456], tool_call_id="call-123")  # type: ignore[list-item]
        assert msg.content == ["123", "456"]

    def test_content_coercion_tuple_to_list(self) -> None:
        """Test that tuple content is converted to list."""
        msg = ToolMessage(content=("a", "b"), tool_call_id="call-123")  # type: ignore[arg-type]
        assert msg.content == ["a", "b"]


class TestToolMessageChunk:
    """Tests for the ToolMessageChunk class."""

    def test_init_basic(self) -> None:
        """Test basic ToolMessageChunk initialization."""
        chunk = ToolMessageChunk(content="Result", tool_call_id="call-123")
        assert chunk.content == "Result"
        assert chunk.tool_call_id == "call-123"
        assert chunk.type == "ToolMessageChunk"

    def test_type_is_tool_message_chunk(self) -> None:
        """Test that ToolMessageChunk type is 'ToolMessageChunk'."""
        chunk = ToolMessageChunk(content="Test", tool_call_id="call-123")
        assert chunk.type == "ToolMessageChunk"

    def test_add_same_tool_call_id_chunks(self) -> None:
        """Test adding ToolMessageChunks with the same tool_call_id."""
        chunk1 = ToolMessageChunk(content="Hello", tool_call_id="call-123", id="1")
        chunk2 = ToolMessageChunk(content=" world", tool_call_id="call-123")
        result = chunk1 + chunk2
        assert isinstance(result, ToolMessageChunk)
        assert result.content == "Hello world"
        assert result.tool_call_id == "call-123"
        assert result.id == "1"

    def test_add_different_tool_call_id_raises_error(self) -> None:
        """Test that adding chunks with different tool_call_ids raises ValueError."""
        chunk1 = ToolMessageChunk(content="Hello", tool_call_id="call-123")
        chunk2 = ToolMessageChunk(content=" world", tool_call_id="call-456")
        with pytest.raises(ValueError, match="Cannot concatenate.*different names"):
            chunk1 + chunk2

    def test_add_with_artifact(self) -> None:
        """Test adding chunks with artifacts."""
        chunk1 = ToolMessageChunk(
            content="Part 1",
            tool_call_id="call-123",
            artifact={"data": "value1"},
        )
        chunk2 = ToolMessageChunk(
            content=" Part 2",
            tool_call_id="call-123",
            artifact={"more": "value2"},
        )
        result = chunk1 + chunk2
        # Artifacts are merged
        assert result.artifact is not None

    def test_add_with_different_status(self) -> None:
        """Test adding chunks with different statuses."""
        chunk1 = ToolMessageChunk(
            content="Part 1",
            tool_call_id="call-123",
            status="success",
        )
        chunk2 = ToolMessageChunk(
            content=" Part 2",
            tool_call_id="call-123",
            status="error",
        )
        result = chunk1 + chunk2
        # Error status takes precedence
        assert result.status == "error"

    def test_add_with_response_metadata(self) -> None:
        """Test adding chunks with response_metadata."""
        chunk1 = ToolMessageChunk(
            content="Hello",
            tool_call_id="call-123",
            response_metadata={"meta1": "data1"},
        )
        chunk2 = ToolMessageChunk(
            content=" world",
            tool_call_id="call-123",
            response_metadata={"meta2": "data2"},
        )
        result = chunk1 + chunk2
        assert result.response_metadata["meta1"] == "data1"
        assert result.response_metadata["meta2"] == "data2"

    def test_add_with_list_content(self) -> None:
        """Test adding chunks with list content.

        Note: Without an 'index' key, list items are appended, not merged.
        """
        chunk1 = ToolMessageChunk(
            content=[{"type": "text", "text": "Hello"}],
            tool_call_id="call-123",
        )
        chunk2 = ToolMessageChunk(
            content=[{"type": "text", "text": " world"}],
            tool_call_id="call-123",
        )
        result = chunk1 + chunk2
        assert isinstance(result.content, list)
        # Items without 'index' key are appended, not merged
        assert len(result.content) == 2
        assert result.content[0]["text"] == "Hello"
        assert result.content[1]["text"] == " world"

    def test_add_with_list_content_with_index(self) -> None:
        """Test adding chunks with list content that have matching index keys."""
        chunk1 = ToolMessageChunk(
            content=[{"type": "text", "text": "Hello", "index": 0}],
            tool_call_id="call-123",
        )
        chunk2 = ToolMessageChunk(
            content=[{"type": "text", "text": " world", "index": 0}],
            tool_call_id="call-123",
        )
        result = chunk1 + chunk2
        assert isinstance(result.content, list)
        # Items with same 'index' key are merged
        assert len(result.content) == 1
        assert result.content[0]["text"] == "Hello world"
        assert result.content[0]["index"] == 0

    def test_serialization_roundtrip(self) -> None:
        """Test ToolMessageChunk serialization and deserialization."""
        chunk = ToolMessageChunk(
            content="Result",
            tool_call_id="call-123",
            id="chunk-123",
        )
        dumped = dumpd(chunk)
        assert dumped["type"] == "constructor"
        assert dumped["id"] == ["langchain", "schema", "messages", "ToolMessageChunk"]

        loaded = load(dumped)
        assert isinstance(loaded, ToolMessageChunk)
        assert loaded.content == "Result"
        assert loaded.tool_call_id == "call-123"
        assert loaded.id == "chunk-123"

    def test_add_incompatible_type_raises_error(self) -> None:
        """Test that adding incompatible types raises TypeError."""
        chunk = ToolMessageChunk(content="Hello", tool_call_id="call-123")
        with pytest.raises(TypeError):
            chunk + "not a chunk"


class TestToolCallFactory:
    """Tests for the tool_call factory function."""

    def test_basic_tool_call(self) -> None:
        """Test creating a basic tool call."""
        tc = tool_call(name="test_tool", args={"param": "value"}, id="call-123")
        assert tc["name"] == "test_tool"
        assert tc["args"] == {"param": "value"}
        assert tc["id"] == "call-123"
        assert tc["type"] == "tool_call"

    def test_tool_call_with_none_id(self) -> None:
        """Test creating a tool call with None id."""
        tc = tool_call(name="test_tool", args={}, id=None)
        assert tc["id"] is None
        assert tc["type"] == "tool_call"

    def test_tool_call_with_empty_args(self) -> None:
        """Test creating a tool call with empty args."""
        tc = tool_call(name="test_tool", args={}, id="call-123")
        assert tc["args"] == {}

    def test_tool_call_with_complex_args(self) -> None:
        """Test creating a tool call with complex args."""
        complex_args = {
            "string": "value",
            "number": 42,
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }
        tc = tool_call(name="test_tool", args=complex_args, id="call-123")
        assert tc["args"] == complex_args


class TestToolCallChunkFactory:
    """Tests for the tool_call_chunk factory function."""

    def test_basic_tool_call_chunk(self) -> None:
        """Test creating a basic tool call chunk."""
        tc = tool_call_chunk(
            name="test_tool", args='{"param": "value"}', id="call-123", index=0
        )
        assert tc["name"] == "test_tool"
        assert tc["args"] == '{"param": "value"}'
        assert tc["id"] == "call-123"
        assert tc["index"] == 0
        assert tc["type"] == "tool_call_chunk"

    def test_tool_call_chunk_with_none_values(self) -> None:
        """Test creating a tool call chunk with None values."""
        tc = tool_call_chunk(name=None, args=None, id=None, index=None)
        assert tc["name"] is None
        assert tc["args"] is None
        assert tc["id"] is None
        assert tc["index"] is None
        assert tc["type"] == "tool_call_chunk"

    def test_tool_call_chunk_defaults(self) -> None:
        """Test tool call chunk with default values."""
        tc = tool_call_chunk()
        assert tc["name"] is None
        assert tc["args"] is None
        assert tc["id"] is None
        assert tc["index"] is None
        assert tc["type"] == "tool_call_chunk"

    def test_tool_call_chunk_partial_args(self) -> None:
        """Test tool call chunk with partial JSON args (for streaming)."""
        tc1 = tool_call_chunk(name="test", args='{"key":', id="123", index=0)
        tc2 = tool_call_chunk(name=None, args='"value"}', id=None, index=0)
        assert tc1["args"] == '{"key":'
        assert tc2["args"] == '"value"}'


class TestInvalidToolCallFactory:
    """Tests for the invalid_tool_call factory function."""

    def test_basic_invalid_tool_call(self) -> None:
        """Test creating a basic invalid tool call."""
        itc = invalid_tool_call(
            name="test_tool",
            args="invalid json",
            id="call-123",
            error="JSON parse error",
        )
        assert itc["name"] == "test_tool"
        assert itc["args"] == "invalid json"
        assert itc["id"] == "call-123"
        assert itc["error"] == "JSON parse error"
        assert itc["type"] == "invalid_tool_call"

    def test_invalid_tool_call_with_none_values(self) -> None:
        """Test creating an invalid tool call with None values."""
        itc = invalid_tool_call(name=None, args=None, id=None, error=None)
        assert itc["name"] is None
        assert itc["args"] is None
        assert itc["id"] is None
        assert itc["error"] is None
        assert itc["type"] == "invalid_tool_call"

    def test_invalid_tool_call_defaults(self) -> None:
        """Test invalid tool call with default values."""
        itc = invalid_tool_call()
        assert itc["name"] is None
        assert itc["args"] is None
        assert itc["id"] is None
        assert itc["error"] is None


class TestDefaultToolParser:
    """Tests for the default_tool_parser function."""

    def test_parse_valid_tool_calls(self) -> None:
        """Test parsing valid tool calls."""
        raw_calls = [
            {
                "id": "call-1",
                "function": {
                    "name": "calculator",
                    "arguments": '{"operation": "add", "a": 1, "b": 2}',
                },
            },
            {
                "id": "call-2",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "weather"}',
                },
            },
        ]
        tool_calls, invalid_calls = default_tool_parser(raw_calls)

        assert len(tool_calls) == 2
        assert len(invalid_calls) == 0

        assert tool_calls[0]["name"] == "calculator"
        assert tool_calls[0]["args"] == {"operation": "add", "a": 1, "b": 2}
        assert tool_calls[0]["id"] == "call-1"

        assert tool_calls[1]["name"] == "search"
        assert tool_calls[1]["args"] == {"query": "weather"}

    def test_parse_invalid_json_args(self) -> None:
        """Test parsing tool calls with invalid JSON arguments."""
        raw_calls = [
            {
                "id": "call-1",
                "function": {
                    "name": "test_tool",
                    "arguments": "not valid json",
                },
            },
        ]
        tool_calls, invalid_calls = default_tool_parser(raw_calls)

        assert len(tool_calls) == 0
        assert len(invalid_calls) == 1

        assert invalid_calls[0]["name"] == "test_tool"
        assert invalid_calls[0]["args"] == "not valid json"
        assert invalid_calls[0]["id"] == "call-1"

    def test_parse_empty_args(self) -> None:
        """Test parsing tool calls with empty arguments."""
        raw_calls = [
            {
                "id": "call-1",
                "function": {
                    "name": "no_args_tool",
                    "arguments": "{}",
                },
            },
        ]
        tool_calls, invalid_calls = default_tool_parser(raw_calls)

        assert len(tool_calls) == 1
        assert tool_calls[0]["args"] == {}

    def test_parse_without_function_key(self) -> None:
        """Test parsing raw calls without function key."""
        raw_calls = [
            {"id": "call-1", "other": "data"},
        ]
        tool_calls, invalid_calls = default_tool_parser(raw_calls)

        assert len(tool_calls) == 0
        assert len(invalid_calls) == 0

    def test_parse_empty_list(self) -> None:
        """Test parsing empty list."""
        tool_calls, invalid_calls = default_tool_parser([])
        assert len(tool_calls) == 0
        assert len(invalid_calls) == 0

    def test_parse_mixed_valid_and_invalid(self) -> None:
        """Test parsing a mix of valid and invalid tool calls."""
        raw_calls = [
            {
                "id": "call-1",
                "function": {
                    "name": "valid_tool",
                    "arguments": '{"key": "value"}',
                },
            },
            {
                "id": "call-2",
                "function": {
                    "name": "invalid_tool",
                    "arguments": "broken json {",
                },
            },
        ]
        tool_calls, invalid_calls = default_tool_parser(raw_calls)

        assert len(tool_calls) == 1
        assert len(invalid_calls) == 1

        assert tool_calls[0]["name"] == "valid_tool"
        assert invalid_calls[0]["name"] == "invalid_tool"


class TestDefaultToolChunkParser:
    """Tests for the default_tool_chunk_parser function."""

    def test_parse_tool_call_chunks(self) -> None:
        """Test parsing tool call chunks."""
        raw_calls = [
            {
                "id": "call-1",
                "index": 0,
                "function": {
                    "name": "test_tool",
                    "arguments": '{"key":',
                },
            },
            {
                "id": "call-1",
                "index": 0,
                "function": {
                    "name": None,
                    "arguments": '"value"}',
                },
            },
        ]
        chunks = default_tool_chunk_parser(raw_calls)

        assert len(chunks) == 2
        assert chunks[0]["name"] == "test_tool"
        assert chunks[0]["args"] == '{"key":'
        assert chunks[0]["index"] == 0

        assert chunks[1]["args"] == '"value"}'

    def test_parse_without_function_key(self) -> None:
        """Test parsing chunks without function key."""
        raw_calls = [
            {"id": "call-1", "index": 0},
        ]
        chunks = default_tool_chunk_parser(raw_calls)

        assert len(chunks) == 1
        assert chunks[0]["name"] is None
        assert chunks[0]["args"] is None
        assert chunks[0]["id"] == "call-1"
        assert chunks[0]["index"] == 0

    def test_parse_empty_list(self) -> None:
        """Test parsing empty list."""
        chunks = default_tool_chunk_parser([])
        assert len(chunks) == 0


class TestToolOutputMixin:
    """Tests for the ToolOutputMixin class."""

    def test_tool_message_is_tool_output_mixin(self) -> None:
        """Test that ToolMessage is an instance of ToolOutputMixin."""
        msg = ToolMessage(content="Result", tool_call_id="call-123")
        assert isinstance(msg, ToolOutputMixin)

    def test_custom_class_with_mixin(self) -> None:
        """Test that custom classes can use ToolOutputMixin."""

        class CustomToolOutput(ToolOutputMixin):
            def __init__(self, result: str) -> None:
                self.result = result

        output = CustomToolOutput(result="42")
        assert isinstance(output, ToolOutputMixin)
        assert output.result == "42"


class TestToolCallTypedDict:
    """Tests for the ToolCall TypedDict structure."""

    def test_tool_call_structure(self) -> None:
        """Test ToolCall TypedDict structure."""
        tc: ToolCall = {
            "name": "test_tool",
            "args": {"param": "value"},
            "id": "call-123",
        }
        assert tc["name"] == "test_tool"
        assert tc["args"]["param"] == "value"
        assert tc["id"] == "call-123"

    def test_tool_call_with_type(self) -> None:
        """Test ToolCall with type field."""
        tc: ToolCall = {
            "name": "test_tool",
            "args": {},
            "id": "call-123",
            "type": "tool_call",
        }
        assert tc["type"] == "tool_call"


class TestToolCallChunkTypedDict:
    """Tests for the ToolCallChunk TypedDict structure."""

    def test_tool_call_chunk_structure(self) -> None:
        """Test ToolCallChunk TypedDict structure."""
        tc: ToolCallChunk = {
            "name": "test_tool",
            "args": '{"key": "value"}',
            "id": "call-123",
            "index": 0,
        }
        assert tc["name"] == "test_tool"
        assert tc["args"] == '{"key": "value"}'
        assert tc["id"] == "call-123"
        assert tc["index"] == 0

    def test_tool_call_chunk_with_type(self) -> None:
        """Test ToolCallChunk with type field."""
        tc: ToolCallChunk = {
            "name": None,
            "args": None,
            "id": None,
            "index": None,
            "type": "tool_call_chunk",
        }
        assert tc["type"] == "tool_call_chunk"


class TestMergeStatus:
    """Tests for the _merge_status helper function."""

    def test_success_plus_success(self) -> None:
        """Test that merging two success statuses returns success."""
        from langchain_core.messages.tool import _merge_status

        result = _merge_status("success", "success")
        assert result == "success"

    def test_error_plus_success(self) -> None:
        """Test that merging error + success returns error."""
        from langchain_core.messages.tool import _merge_status

        result = _merge_status("error", "success")
        assert result == "error"

    def test_success_plus_error(self) -> None:
        """Test that merging success + error returns error."""
        from langchain_core.messages.tool import _merge_status

        result = _merge_status("success", "error")
        assert result == "error"

    def test_error_plus_error(self) -> None:
        """Test that merging two error statuses returns error."""
        from langchain_core.messages.tool import _merge_status

        result = _merge_status("error", "error")
        assert result == "error"


class TestToolMessageContentCoercion:
    """Tests for ToolMessage content coercion edge cases."""

    def test_tuple_content_converted_to_list(self) -> None:
        """Test that tuple content is converted to a list preserving items."""
        msg = ToolMessage(
            content=("first", {"type": "text", "text": "second"}),  # type: ignore[arg-type]
            tool_call_id="call-100",
        )
        assert isinstance(msg.content, list)
        assert msg.content == ["first", {"type": "text", "text": "second"}]

    def test_list_with_non_string_non_dict_items_coerced_to_strings(self) -> None:
        """Test that list items that are not str or dict are coerced to strings."""
        msg = ToolMessage(
            content=[42, True, 3.14, None],  # type: ignore[list-item]
            tool_call_id="call-200",
        )
        assert isinstance(msg.content, list)
        assert msg.content == ["42", "True", "3.14", "None"]

    def test_non_string_non_list_content_coerced_to_string(self) -> None:
        """Test that content that is neither str nor list is coerced to str."""
        msg_int = ToolMessage(content=99, tool_call_id="call-300")  # type: ignore[arg-type]
        assert msg_int.content == "99"

        msg_float = ToolMessage(content=2.718, tool_call_id="call-301")  # type: ignore[arg-type]
        assert msg_float.content == "2.718"

        msg_bool = ToolMessage(content=False, tool_call_id="call-302")  # type: ignore[arg-type]
        assert msg_bool.content == "False"

        msg_dict = ToolMessage(content={"key": "val"}, tool_call_id="call-303")  # type: ignore[arg-type]
        assert msg_dict.content == "{'key': 'val'}"

    def test_empty_string_content(self) -> None:
        """Test ToolMessage with empty string content."""
        msg = ToolMessage(content="", tool_call_id="call-400")
        assert msg.content == ""
        assert msg.tool_call_id == "call-400"
        assert msg.status == "success"
        assert msg.type == "tool"


class TestToolMessageContentBlocks:
    """Tests for ToolMessage with content_blocks parameter."""

    def test_content_blocks_sets_content(self) -> None:
        """Test that content_blocks parameter populates the content field."""
        blocks = [
            {"type": "text", "text": "Hello from blocks"},
            {"type": "text", "text": "Second block"},
        ]
        msg = ToolMessage(content_blocks=blocks, tool_call_id="call-500")
        assert msg.content == blocks
        assert msg.content[0] == {"type": "text", "text": "Hello from blocks"}
        assert msg.content[1] == {"type": "text", "text": "Second block"}

    def test_content_blocks_single_block(self) -> None:
        """Test content_blocks with a single content block."""
        blocks = [{"type": "text", "text": "Only block"}]
        msg = ToolMessage(content_blocks=blocks, tool_call_id="call-501")
        assert len(msg.content) == 1
        assert msg.content[0] == {"type": "text", "text": "Only block"}


class TestToolMessageSerializationWithArtifactAndError:
    """Tests for ToolMessage serialization with artifact and status=error."""

    def test_serialization_roundtrip_with_artifact_and_error_status(self) -> None:
        """Test serialization roundtrip preserves artifact and error status."""
        artifact_data = {"raw_output": "traceback info", "exit_code": 1}
        msg = ToolMessage(
            content="Tool execution failed",
            tool_call_id="call-600",
            name="failing_tool",
            artifact=artifact_data,
            status="error",
            id="msg-600",
        )
        dumped = dumpd(msg)

        assert dumped["type"] == "constructor"
        assert dumped["id"] == ["langchain", "schema", "messages", "ToolMessage"]

        loaded = load(dumped)
        assert isinstance(loaded, ToolMessage)
        assert loaded.content == "Tool execution failed"
        assert loaded.tool_call_id == "call-600"
        assert loaded.name == "failing_tool"
        assert loaded.artifact == artifact_data
        assert loaded.status == "error"
        assert loaded.id == "msg-600"


class TestToolMessageChunkAddExtended:
    """Extended tests for ToolMessageChunk __add__ behavior."""

    def test_both_success_statuses_result_in_success(self) -> None:
        """Test adding two chunks with success status produces success."""
        chunk1 = ToolMessageChunk(
            content="Part A",
            tool_call_id="call-700",
            status="success",
        )
        chunk2 = ToolMessageChunk(
            content=" Part B",
            tool_call_id="call-700",
            status="success",
        )
        result = chunk1 + chunk2
        assert isinstance(result, ToolMessageChunk)
        assert result.status == "success"
        assert result.content == "Part A Part B"

    def test_tool_call_id_preserved_from_first_chunk(self) -> None:
        """Test that adding chunks preserves tool_call_id from the first chunk."""
        chunk1 = ToolMessageChunk(
            content="Hello",
            tool_call_id="call-800",
            id="chunk-first",
        )
        chunk2 = ToolMessageChunk(
            content=" World",
            tool_call_id="call-800",
            id="chunk-second",
        )
        result = chunk1 + chunk2
        assert result.tool_call_id == "call-800"
        assert result.id == "chunk-first"

    def test_list_content_with_index_merging(self) -> None:
        """Test adding chunks with list content using index-based merging."""
        chunk1 = ToolMessageChunk(
            content=[
                {"type": "text", "text": "alpha", "index": 0},
                {"type": "text", "text": "beta", "index": 1},
            ],
            tool_call_id="call-900",
        )
        chunk2 = ToolMessageChunk(
            content=[
                {"type": "text", "text": "-more", "index": 0},
                {"type": "text", "text": "-extra", "index": 1},
            ],
            tool_call_id="call-900",
        )
        result = chunk1 + chunk2
        assert isinstance(result, ToolMessageChunk)
        assert isinstance(result.content, list)
        assert len(result.content) == 2
        assert result.content[0]["text"] == "alpha-more"
        assert result.content[0]["index"] == 0
        assert result.content[1]["text"] == "beta-extra"
        assert result.content[1]["index"] == 1

    def test_fallback_to_base_message_chunk_add_for_different_chunk_types(
        self,
    ) -> None:
        """Test that adding a non-ToolMessageChunk falls back to BaseMessageChunk.__add__.

        The base class __add__ attempts to reconstruct via self.__class__() but
        does not pass tool_call_id, so this raises a KeyError from the
        ToolMessage model validator.
        """
        from langchain_core.messages.human import HumanMessageChunk

        tool_chunk = ToolMessageChunk(
            content="tool output",
            tool_call_id="call-1000",
        )
        human_chunk = HumanMessageChunk(content=" from human")
        # Falls back to BaseMessageChunk.__add__, which calls self.__class__()
        # without tool_call_id, causing a KeyError in coerce_args validator
        with pytest.raises(KeyError, match="tool_call_id"):
            tool_chunk + human_chunk


class TestDefaultToolParserExtended:
    """Extended tests for default_tool_parser edge cases."""

    def test_tool_call_with_no_id_field(self) -> None:
        """Test parsing a tool call that has no id field uses None."""
        raw_calls = [
            {
                "function": {
                    "name": "lookup",
                    "arguments": '{"term": "python"}',
                },
            },
        ]
        tool_calls, invalid_calls = default_tool_parser(raw_calls)
        assert len(tool_calls) == 1
        assert len(invalid_calls) == 0
        assert tool_calls[0]["name"] == "lookup"
        assert tool_calls[0]["args"] == {"term": "python"}
        assert tool_calls[0]["id"] is None

    def test_empty_function_args_string(self) -> None:
        """Test parsing with empty string arguments (valid JSON empty string is not valid)."""
        raw_calls = [
            {
                "id": "call-a",
                "function": {
                    "name": "no_args",
                    "arguments": "",
                },
            },
        ]
        tool_calls, invalid_calls = default_tool_parser(raw_calls)
        # Empty string is not valid JSON, so it should be an invalid tool call
        assert len(tool_calls) == 0
        assert len(invalid_calls) == 1
        assert invalid_calls[0]["name"] == "no_args"
        assert invalid_calls[0]["args"] == ""
        assert invalid_calls[0]["id"] == "call-a"

    def test_null_function_args(self) -> None:
        """Test parsing with null JSON arguments results in empty dict args."""
        raw_calls = [
            {
                "id": "call-b",
                "function": {
                    "name": "null_args_tool",
                    "arguments": "null",
                },
            },
        ]
        tool_calls, invalid_calls = default_tool_parser(raw_calls)
        # json.loads("null") returns None, then `function_args or {}` yields {}
        assert len(tool_calls) == 1
        assert len(invalid_calls) == 0
        assert tool_calls[0]["name"] == "null_args_tool"
        assert tool_calls[0]["args"] == {}
        assert tool_calls[0]["id"] == "call-b"


class TestDefaultToolChunkParserExtended:
    """Extended tests for default_tool_chunk_parser edge cases."""

    def test_tool_calls_with_function_name_none(self) -> None:
        """Test parsing chunks where function.name is None (continuation chunks)."""
        raw_calls = [
            {
                "id": None,
                "index": 0,
                "function": {
                    "name": None,
                    "arguments": '"continued"}',
                },
            },
        ]
        chunks = default_tool_chunk_parser(raw_calls)
        assert len(chunks) == 1
        assert chunks[0]["name"] is None
        assert chunks[0]["args"] == '"continued"}'
        assert chunks[0]["id"] is None
        assert chunks[0]["index"] == 0
        assert chunks[0]["type"] == "tool_call_chunk"


class TestToolMessageFieldDefaults:
    """Tests for ToolMessage field default values and repr settings."""

    def test_additional_kwargs_default_is_empty_dict(self) -> None:
        """Test that additional_kwargs defaults to an empty dict."""
        msg = ToolMessage(content="test", tool_call_id="call-1100")
        assert msg.additional_kwargs == {}
        assert isinstance(msg.additional_kwargs, dict)

    def test_response_metadata_default_is_empty_dict(self) -> None:
        """Test that response_metadata defaults to an empty dict."""
        msg = ToolMessage(content="test", tool_call_id="call-1200")
        assert msg.response_metadata == {}
        assert isinstance(msg.response_metadata, dict)

    def test_additional_kwargs_and_response_metadata_not_in_repr(self) -> None:
        """Test that additional_kwargs and response_metadata with repr=False are excluded from repr."""
        msg = ToolMessage(content="test", tool_call_id="call-1300")
        msg_repr = repr(msg)
        assert "additional_kwargs" not in msg_repr
        assert "response_metadata" not in msg_repr

    def test_additional_kwargs_and_response_metadata_repr_false_with_values(
        self,
    ) -> None:
        """Test that even when populated, repr=False fields are excluded from repr."""
        msg = ToolMessage(
            content="test",
            tool_call_id="call-1301",
            additional_kwargs={"custom": "value"},
            response_metadata={"meta": "data"},
        )
        msg_repr = repr(msg)
        assert "additional_kwargs" not in msg_repr
        assert "response_metadata" not in msg_repr
        # But they are still accessible
        assert msg.additional_kwargs == {"custom": "value"}
        assert msg.response_metadata == {"meta": "data"}


class TestToolMessagePrettyReprExtended:
    """Extended tests for ToolMessage pretty_repr output."""

    def test_pretty_repr_includes_tool_name(self) -> None:
        """Test that pretty_repr includes the tool name when provided."""
        msg = ToolMessage(
            content="42",
            tool_call_id="call-1400",
            name="calculator",
        )
        result = msg.pretty_repr()
        assert "Tool Message" in result
        assert "Name: calculator" in result
        assert "42" in result

    def test_pretty_repr_without_name(self) -> None:
        """Test pretty_repr without a tool name."""
        msg = ToolMessage(content="result data", tool_call_id="call-1500")
        result = msg.pretty_repr()
        assert "Tool Message" in result
        assert "Name:" not in result
        assert "result data" in result

    def test_pretty_repr_with_error_content(self) -> None:
        """Test pretty_repr renders error content correctly."""
        msg = ToolMessage(
            content="Error: division by zero",
            tool_call_id="call-1600",
            name="math_tool",
            status="error",
        )
        result = msg.pretty_repr()
        assert "Tool Message" in result
        assert "Name: math_tool" in result
        assert "Error: division by zero" in result
