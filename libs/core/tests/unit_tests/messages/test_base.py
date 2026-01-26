"""Tests for langchain_core.messages.base module."""

import warnings
from typing import Any

import pytest

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    TextAccessor,
    merge_content,
    message_to_dict,
    messages_to_dict,
    get_msg_title_repr,
)
from langchain_core.messages.human import HumanMessageChunk
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.system import SystemMessageChunk


class TestTextAccessor:
    """Tests for the TextAccessor class."""

    def test_text_accessor_str_behavior(self) -> None:
        """Test that TextAccessor behaves like a string."""
        accessor = TextAccessor("hello world")
        assert str(accessor) == "hello world"
        assert accessor == "hello world"
        assert len(accessor) == 11
        assert accessor.upper() == "HELLO WORLD"
        assert accessor.startswith("hello")

    def test_text_accessor_call_emits_deprecation_warning(self) -> None:
        """Test that calling TextAccessor as a method emits a deprecation warning."""
        accessor = TextAccessor("test")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = accessor()
            assert result == "test"
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()

    def test_text_accessor_empty_string(self) -> None:
        """Test TextAccessor with an empty string."""
        accessor = TextAccessor("")
        assert str(accessor) == ""
        assert accessor == ""
        assert len(accessor) == 0


class TestBaseMessageText:
    """Tests for the .text property on BaseMessage."""

    def test_text_property_string_content(self) -> None:
        """Test .text property with string content."""
        msg = HumanMessage(content="Hello, world!")
        assert msg.text == "Hello, world!"
        assert isinstance(msg.text, TextAccessor)

    def test_text_property_list_content_with_text_blocks(self) -> None:
        """Test .text property with list content containing text blocks."""
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "First part"},
                {"type": "text", "text": " second part"},
            ]
        )
        assert msg.text == "First part second part"

    def test_text_property_list_content_with_mixed_blocks(self) -> None:
        """Test .text property with mixed content blocks."""
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                {"type": "text", "text": " world"},
            ]
        )
        assert msg.text == "Hello world"

    def test_text_property_list_content_with_strings(self) -> None:
        """Test .text property with list content containing plain strings."""
        msg = HumanMessage(content=["Hello", " world"])
        assert msg.text == "Hello world"

    def test_text_property_empty_content(self) -> None:
        """Test .text property with empty content."""
        msg = HumanMessage(content="")
        assert msg.text == ""

    def test_text_property_empty_list_content(self) -> None:
        """Test .text property with empty list content."""
        msg = HumanMessage(content=[])
        assert msg.text == ""

    def test_text_property_no_text_blocks(self) -> None:
        """Test .text property when there are no text blocks."""
        msg = HumanMessage(
            content=[{"type": "image_url", "image_url": {"url": "http://example.com"}}]
        )
        assert msg.text == ""


class TestMergeContent:
    """Tests for the merge_content function."""

    def test_merge_two_strings(self) -> None:
        """Test merging two string contents."""
        result = merge_content("Hello", " world")
        assert result == "Hello world"

    def test_merge_string_and_list(self) -> None:
        """Test merging a string with a list."""
        result = merge_content("Hello", [{"type": "text", "text": " world"}])
        assert result == ["Hello", {"type": "text", "text": " world"}]

    def test_merge_list_and_string(self) -> None:
        """Test merging a list with a string."""
        result = merge_content([{"type": "text", "text": "Hello"}], " world")
        assert result == [{"type": "text", "text": "Hello"}, " world"]

    def test_merge_two_lists(self) -> None:
        """Test merging two list contents.

        Note: Without an 'index' key, list items are appended, not merged.
        """
        result = merge_content(
            [{"type": "text", "text": "Hello"}],
            [{"type": "text", "text": " world"}],
        )
        # Items without 'index' key are appended, not merged
        assert result == [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " world"},
        ]

    def test_merge_two_lists_with_index(self) -> None:
        """Test merging two list contents with matching index keys."""
        result = merge_content(
            [{"type": "text", "text": "Hello", "index": 0}],
            [{"type": "text", "text": " world", "index": 0}],
        )
        # Items with same 'index' key are merged
        assert result == [
            {"type": "text", "text": "Hello world", "index": 0}
        ]

    def test_merge_multiple_strings(self) -> None:
        """Test merging multiple strings."""
        result = merge_content("a", "b", "c", "d")
        assert result == "abcd"

    def test_merge_empty_string_first(self) -> None:
        """Test merging when first content is empty string."""
        result = merge_content("", "Hello")
        assert result == "Hello"

    def test_merge_empty_string_second(self) -> None:
        """Test merging when second content is empty string."""
        result = merge_content("Hello", "")
        assert result == "Hello"

    def test_merge_list_with_empty_string(self) -> None:
        """Test merging a list with an empty string."""
        result = merge_content([{"type": "text", "text": "Hello"}], "")
        assert result == [{"type": "text", "text": "Hello"}]

    def test_merge_with_none_first_content(self) -> None:
        """Test merging when first content is None (treated as empty)."""
        # merge_content treats None as ""
        result = merge_content("", "Hello")
        assert result == "Hello"


class TestMessageToDict:
    """Tests for message_to_dict and messages_to_dict functions."""

    def test_message_to_dict_human_message(self) -> None:
        """Test converting a HumanMessage to dict."""
        msg = HumanMessage(content="Hello", name="user1", id="msg1")
        result = message_to_dict(msg)
        assert result["type"] == "human"
        assert result["data"]["content"] == "Hello"
        assert result["data"]["name"] == "user1"
        assert result["data"]["id"] == "msg1"

    def test_message_to_dict_ai_message(self) -> None:
        """Test converting an AIMessage to dict."""
        msg = AIMessage(content="Hi there", id="ai1")
        result = message_to_dict(msg)
        assert result["type"] == "ai"
        assert result["data"]["content"] == "Hi there"
        assert result["data"]["id"] == "ai1"

    def test_message_to_dict_system_message(self) -> None:
        """Test converting a SystemMessage to dict."""
        msg = SystemMessage(content="You are a helpful assistant")
        result = message_to_dict(msg)
        assert result["type"] == "system"
        assert result["data"]["content"] == "You are a helpful assistant"

    def test_message_to_dict_with_additional_kwargs(self) -> None:
        """Test converting a message with additional_kwargs."""
        msg = AIMessage(
            content="Hello",
            additional_kwargs={"function_call": {"name": "test", "arguments": "{}"}},
        )
        result = message_to_dict(msg)
        assert result["data"]["additional_kwargs"]["function_call"]["name"] == "test"

    def test_messages_to_dict_multiple_messages(self) -> None:
        """Test converting multiple messages to dicts."""
        messages = [
            SystemMessage(content="System"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
        ]
        result = messages_to_dict(messages)
        assert len(result) == 3
        assert result[0]["type"] == "system"
        assert result[1]["type"] == "human"
        assert result[2]["type"] == "ai"

    def test_messages_to_dict_empty_list(self) -> None:
        """Test converting an empty list of messages."""
        result = messages_to_dict([])
        assert result == []


class TestBaseMessageContentBlocks:
    """Tests for the content_blocks property on BaseMessage."""

    def test_content_blocks_string_content(self) -> None:
        """Test content_blocks with string content."""
        msg = HumanMessage(content="Hello")
        blocks = msg.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Hello"

    def test_content_blocks_empty_string(self) -> None:
        """Test content_blocks with empty string content."""
        msg = HumanMessage(content="")
        blocks = msg.content_blocks
        assert blocks == []

    def test_content_blocks_list_with_string(self) -> None:
        """Test content_blocks with list containing a string."""
        msg = HumanMessage(content=["Hello", "world"])
        blocks = msg.content_blocks
        assert len(blocks) == 2
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Hello"
        assert blocks[1]["type"] == "text"
        assert blocks[1]["text"] == "world"

    def test_content_blocks_standard_text_block(self) -> None:
        """Test content_blocks with standard text block."""
        msg = HumanMessage(content=[{"type": "text", "text": "Hello"}])
        blocks = msg.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Hello"

    def test_content_blocks_non_standard_block(self) -> None:
        """Test content_blocks with non-standard block type."""
        msg = HumanMessage(content=[{"type": "custom_type", "data": "value"}])
        blocks = msg.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "non_standard"
        assert blocks[0]["value"]["type"] == "custom_type"

    def test_content_blocks_mixed_content(self) -> None:
        """Test content_blocks with mixed content types."""
        msg = HumanMessage(
            content=[
                "Plain string",
                {"type": "text", "text": "Text block"},
                {"type": "image", "url": "http://example.com/img.png"},
            ]
        )
        blocks = msg.content_blocks
        assert len(blocks) == 3
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "text"
        assert blocks[2]["type"] == "image"


class TestBaseMessageChunkAdd:
    """Tests for BaseMessageChunk.__add__ method."""

    def test_add_human_message_chunks(self) -> None:
        """Test adding two HumanMessageChunks."""
        chunk1 = HumanMessageChunk(content="Hello", id="1")
        chunk2 = HumanMessageChunk(content=" world")
        result = chunk1 + chunk2
        assert isinstance(result, HumanMessageChunk)
        assert result.content == "Hello world"
        assert result.id == "1"

    def test_add_system_message_chunks(self) -> None:
        """Test adding two SystemMessageChunks."""
        chunk1 = SystemMessageChunk(content="You are")
        chunk2 = SystemMessageChunk(content=" helpful")
        result = chunk1 + chunk2
        assert isinstance(result, SystemMessageChunk)
        assert result.content == "You are helpful"

    def test_add_chunks_with_additional_kwargs(self) -> None:
        """Test adding chunks with additional_kwargs."""
        chunk1 = HumanMessageChunk(
            content="Hello", additional_kwargs={"key1": "value1"}
        )
        chunk2 = HumanMessageChunk(
            content=" world", additional_kwargs={"key2": "value2"}
        )
        result = chunk1 + chunk2
        assert result.additional_kwargs["key1"] == "value1"
        assert result.additional_kwargs["key2"] == "value2"

    def test_add_chunks_with_response_metadata(self) -> None:
        """Test adding chunks with response_metadata."""
        chunk1 = HumanMessageChunk(
            content="Hello", response_metadata={"meta1": "data1"}
        )
        chunk2 = HumanMessageChunk(
            content=" world", response_metadata={"meta2": "data2"}
        )
        result = chunk1 + chunk2
        assert result.response_metadata["meta1"] == "data1"
        assert result.response_metadata["meta2"] == "data2"

    def test_add_chunk_list_content(self) -> None:
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

    def test_add_chunk_list_content_with_index(self) -> None:
        """Test adding chunks with list content that have matching index keys."""
        chunk1 = HumanMessageChunk(content=[{"type": "text", "text": "Hello", "index": 0}])
        chunk2 = HumanMessageChunk(content=[{"type": "text", "text": " world", "index": 0}])
        result = chunk1 + chunk2
        assert isinstance(result.content, list)
        # Items with same 'index' key are merged
        assert len(result.content) == 1
        assert result.content[0]["text"] == "Hello world"
        assert result.content[0]["index"] == 0

    def test_add_list_of_chunks(self) -> None:
        """Test adding a list of chunks to a chunk."""
        chunk1 = HumanMessageChunk(content="a", id="1")
        chunk2 = HumanMessageChunk(content="b")
        chunk3 = HumanMessageChunk(content="c")
        result = chunk1 + [chunk2, chunk3]
        assert result.content == "abc"
        assert result.id == "1"

    def test_add_incompatible_type_raises_error(self) -> None:
        """Test that adding incompatible types raises TypeError."""
        chunk = HumanMessageChunk(content="Hello")
        with pytest.raises(TypeError):
            chunk + "not a chunk"

    def test_add_incompatible_object_raises_error(self) -> None:
        """Test that adding non-message objects raises TypeError."""
        chunk = HumanMessageChunk(content="Hello")
        with pytest.raises(TypeError):
            chunk + 123


class TestBaseMessagePrettyRepr:
    """Tests for the pretty_repr method on BaseMessage."""

    def test_pretty_repr_basic(self) -> None:
        """Test basic pretty_repr output."""
        msg = HumanMessage(content="Hello")
        result = msg.pretty_repr()
        assert "Human Message" in result
        assert "Hello" in result

    def test_pretty_repr_with_name(self) -> None:
        """Test pretty_repr with message name."""
        msg = HumanMessage(content="Hello", name="user1")
        result = msg.pretty_repr()
        assert "Name: user1" in result

    def test_pretty_repr_html_mode(self) -> None:
        """Test pretty_repr with html=True."""
        msg = HumanMessage(content="Hello")
        result = msg.pretty_repr(html=True)
        assert "Human Message" in result


class TestGetMsgTitleRepr:
    """Tests for the get_msg_title_repr function."""

    def test_get_msg_title_repr_basic(self) -> None:
        """Test basic title representation."""
        result = get_msg_title_repr("Test Title")
        assert "Test Title" in result
        assert "=" in result

    def test_get_msg_title_repr_bold(self) -> None:
        """Test title representation with bold."""
        result = get_msg_title_repr("Test Title", bold=True)
        assert "Test Title" in result

    def test_get_msg_title_repr_long_title(self) -> None:
        """Test title representation with a long title."""
        result = get_msg_title_repr("A" * 100)
        assert "A" * 100 in result


class TestBaseMessageSerialization:
    """Tests for BaseMessage serialization."""

    def test_is_lc_serializable(self) -> None:
        """Test that BaseMessage subclasses are serializable."""
        assert HumanMessage.is_lc_serializable() is True
        assert AIMessage.is_lc_serializable() is True
        assert SystemMessage.is_lc_serializable() is True

    def test_get_lc_namespace(self) -> None:
        """Test the LangChain namespace for messages."""
        assert HumanMessage.get_lc_namespace() == ["langchain", "schema", "messages"]


class TestBaseMessageInit:
    """Tests for BaseMessage initialization."""

    def test_init_with_content_blocks(self) -> None:
        """Test initializing with content_blocks parameter."""
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": "Hello"},
            {"type": "image", "url": "http://example.com/img.png"},
        ]
        msg = HumanMessage(content_blocks=blocks)
        assert msg.content == blocks

    def test_init_with_string_content(self) -> None:
        """Test initializing with string content."""
        msg = HumanMessage(content="Hello world")
        assert msg.content == "Hello world"

    def test_init_with_list_content(self) -> None:
        """Test initializing with list content."""
        content = [{"type": "text", "text": "Hello"}]
        msg = HumanMessage(content=content)
        assert msg.content == content

    def test_init_with_id(self) -> None:
        """Test initializing with an ID."""
        msg = HumanMessage(content="Hello", id="msg-123")
        assert msg.id == "msg-123"

    def test_init_with_numeric_id_coerced_to_string(self) -> None:
        """Test that numeric IDs are coerced to strings."""
        msg = HumanMessage(content="Hello", id=123)
        assert msg.id == "123"
        assert isinstance(msg.id, str)

    def test_init_with_name(self) -> None:
        """Test initializing with a name."""
        msg = HumanMessage(content="Hello", name="user1")
        assert msg.name == "user1"

    def test_init_with_additional_kwargs(self) -> None:
        """Test initializing with additional_kwargs."""
        msg = HumanMessage(
            content="Hello",
            additional_kwargs={"custom_key": "custom_value"},
        )
        assert msg.additional_kwargs["custom_key"] == "custom_value"

    def test_init_with_response_metadata(self) -> None:
        """Test initializing with response_metadata."""
        msg = AIMessage(
            content="Hello",
            response_metadata={"model": "gpt-4", "tokens": 10},
        )
        assert msg.response_metadata["model"] == "gpt-4"
        assert msg.response_metadata["tokens"] == 10


class TestBaseMessageAdd:
    """Tests for BaseMessage.__add__ which creates ChatPromptTemplate."""

    def test_add_messages_creates_prompt_template(self) -> None:
        """Test that adding messages creates a ChatPromptTemplate."""
        from langchain_core.prompts.chat import ChatPromptTemplate

        msg1 = HumanMessage(content="Hello")
        msg2 = AIMessage(content="Hi there")
        result = msg1 + msg2
        assert isinstance(result, ChatPromptTemplate)

    def test_add_message_to_string(self) -> None:
        """Test adding a message to a string."""
        from langchain_core.prompts.chat import ChatPromptTemplate

        msg = HumanMessage(content="Hello")
        result = msg + "Additional content"
        assert isinstance(result, ChatPromptTemplate)
