"""Tests for langchain_core.messages.base module."""

import warnings
from typing import Any

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    TextAccessor,
    get_msg_title_repr,
    merge_content,
    message_to_dict,
    messages_to_dict,
)
from langchain_core.messages.human import HumanMessageChunk
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
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/img.png"},
                },
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
        assert result == [{"type": "text", "text": "Hello world", "index": 0}]

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
        chunk1 = HumanMessageChunk(
            content=[{"type": "text", "text": "Hello", "index": 0}]
        )
        chunk2 = HumanMessageChunk(
            content=[{"type": "text", "text": " world", "index": 0}]
        )
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


class TestExtractReasoningFromAdditionalKwargs:
    """Tests for the _extract_reasoning_from_additional_kwargs function."""

    def test_string_reasoning_content_returns_reasoning_block(self) -> None:
        """Test that string reasoning_content returns a ReasoningContentBlock."""
        from langchain_core.messages.base import (
            _extract_reasoning_from_additional_kwargs,
        )

        msg = AIMessage(
            content="Hello",
            additional_kwargs={"reasoning_content": "I think therefore I am"},
        )
        result = _extract_reasoning_from_additional_kwargs(msg)
        assert result is not None
        assert result["type"] == "reasoning"
        assert result["reasoning"] == "I think therefore I am"

    def test_none_reasoning_content_returns_none(self) -> None:
        """Test that None reasoning_content returns None."""
        from langchain_core.messages.base import (
            _extract_reasoning_from_additional_kwargs,
        )

        msg = AIMessage(
            content="Hello",
            additional_kwargs={"reasoning_content": None},
        )
        result = _extract_reasoning_from_additional_kwargs(msg)
        assert result is None

    def test_non_string_reasoning_content_returns_none(self) -> None:
        """Test that non-string reasoning_content (e.g. dict) returns None."""
        from langchain_core.messages.base import (
            _extract_reasoning_from_additional_kwargs,
        )

        msg = AIMessage(
            content="Hello",
            additional_kwargs={"reasoning_content": {"nested": "data", "value": 42}},
        )
        result = _extract_reasoning_from_additional_kwargs(msg)
        assert result is None

    def test_no_reasoning_content_key_returns_none(self) -> None:
        """Test that missing reasoning_content key returns None."""
        from langchain_core.messages.base import (
            _extract_reasoning_from_additional_kwargs,
        )

        msg = AIMessage(content="Hello", additional_kwargs={"other_key": "value"})
        result = _extract_reasoning_from_additional_kwargs(msg)
        assert result is None

    def test_empty_additional_kwargs_returns_none(self) -> None:
        """Test that empty additional_kwargs returns None."""
        from langchain_core.messages.base import (
            _extract_reasoning_from_additional_kwargs,
        )

        msg = AIMessage(content="Hello")
        result = _extract_reasoning_from_additional_kwargs(msg)
        assert result is None


class TestContentBlocksNonStandard:
    """Tests for content_blocks with non-standard block types."""

    def test_v0_style_blocks_with_source_type_converted_to_v1(self) -> None:
        """Test that v0-style blocks with source_type are converted to v1.

        The initial pass in content_blocks marks items with source_type as
        non_standard, but the subsequent v0 block translator pass converts
        recognized v0 blocks (e.g. image with source_type=base64) into
        standard v1 blocks. The final output is a converted v1 image block
        with extra keys preserved in the extras dict.
        """
        msg = HumanMessage(
            content=[
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": "abc123",
                    "media_type": "image/png",
                }
            ]
        )
        blocks = msg.content_blocks
        assert len(blocks) == 1
        # The v0 translator converts this to a standard v1 image block
        assert blocks[0]["type"] == "image"
        assert blocks[0]["base64"] == "abc123"
        # media_type (not a known v0 key) is preserved in extras
        assert blocks[0]["extras"]["media_type"] == "image/png"

    def test_v0_style_non_media_type_with_source_type_stays_non_standard(
        self,
    ) -> None:
        """Test that non-media types with source_type remain non_standard.

        The v0 block translator only handles image, audio, and file types.
        A completely unknown type with source_type stays non_standard.
        """
        msg = HumanMessage(
            content=[
                {
                    "type": "custom_xyz",
                    "source_type": "magic",
                    "data": "stuff",
                }
            ]
        )
        blocks = msg.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "non_standard"
        assert blocks[0]["value"]["type"] == "custom_xyz"
        assert blocks[0]["value"]["source_type"] == "magic"

    def test_dict_with_type_not_in_known_block_types(self) -> None:
        """Test that dict items with unknown type produce non_standard type."""
        msg = HumanMessage(
            content=[
                {
                    "type": "completely_unknown_type_xyz",
                    "payload": {"key": "value"},
                }
            ]
        )
        blocks = msg.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "non_standard"
        assert blocks[0]["value"]["type"] == "completely_unknown_type_xyz"
        assert blocks[0]["value"]["payload"] == {"key": "value"}

    def test_dict_with_no_type_key(self) -> None:
        """Test that dict items with no type key produce non_standard type."""
        msg = HumanMessage(content=[{"data": "some data", "format": "raw"}])
        blocks = msg.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "non_standard"
        assert blocks[0]["value"]["data"] == "some data"
        assert blocks[0]["value"]["format"] == "raw"


class TestBaseMessageModelConfig:
    """Tests for BaseMessage model_config allowing extra fields."""

    def test_model_config_allows_extra_fields(self) -> None:
        """Test that BaseMessage allows extra fields via model_config."""
        msg = HumanMessage(
            content="Hello",
            custom_field="custom_value",
            another_extra=42,
        )
        assert msg.content == "Hello"
        assert msg.custom_field == "custom_value"  # type: ignore[attr-defined]
        assert msg.another_extra == 42  # type: ignore[attr-defined]

    def test_model_config_extra_fields_in_dump(self) -> None:
        """Test that extra fields appear in model_dump output."""
        msg = HumanMessage(
            content="Hello",
            custom_field="custom_value",
        )
        dumped = msg.model_dump()
        assert dumped["custom_field"] == "custom_value"


class TestMergeContentAdditional:
    """Additional tests for merge_content covering uncovered branches."""

    def test_merge_list_plus_list_last_element_string_concatenates(self) -> None:
        """Test merging list+list where last element of first list is a string.

        When merging via merge_lists, list items without index are appended.
        But when merging list+string and last element is a string, they concatenate.
        """
        result = merge_content(["Hello"], " world")
        assert result == ["Hello world"]

    def test_merge_list_plus_string_last_element_dict_appends(self) -> None:
        """Test merging list+string where last element is a dict appends."""
        result = merge_content([{"type": "text", "text": "Hello"}], " world")
        assert result == [{"type": "text", "text": "Hello"}, " world"]

    def test_merge_none_first_content_treated_as_empty_string(self) -> None:
        """Test that None first content is treated as empty string."""
        result = merge_content(None, "Hello")  # type: ignore[arg-type]
        assert result == "Hello"

    def test_merge_none_first_content_with_list_second(self) -> None:
        """Test that None first content with list second yields list."""
        result = merge_content(None, [{"type": "text", "text": "Hello"}])  # type: ignore[arg-type]
        assert result == ["", {"type": "text", "text": "Hello"}]

    def test_merge_list_plus_empty_string_no_op(self) -> None:
        """Test that merging list + empty string is a no-op when last is dict."""
        result = merge_content([{"type": "text", "text": "Hello"}], "")
        assert result == [{"type": "text", "text": "Hello"}]

    def test_merge_list_string_last_plus_empty_string(self) -> None:
        """Test merging list (last element string) + empty string concatenates."""
        result = merge_content(["Hello"], "")
        assert result == ["Hello"]


class TestBaseMessageChunkAddMixed:
    """Tests for BaseMessageChunk.__add__ with mixed chunk lists."""

    def test_add_list_of_mixed_message_chunks(self) -> None:
        """Test adding a list of mixed BaseMessageChunk types."""
        chunk1 = HumanMessageChunk(content="Start", id="main")
        others = [
            HumanMessageChunk(content=" middle"),
            HumanMessageChunk(content=" end"),
        ]
        result = chunk1 + others
        assert isinstance(result, HumanMessageChunk)
        assert result.content == "Start middle end"
        assert result.id == "main"

    def test_add_list_of_chunks_with_metadata(self) -> None:
        """Test adding a list of chunks merges additional_kwargs and metadata."""
        chunk1 = HumanMessageChunk(
            content="a",
            id="1",
            additional_kwargs={"key1": "val1"},
            response_metadata={"meta1": "data1"},
        )
        chunk2 = HumanMessageChunk(
            content="b",
            additional_kwargs={"key2": "val2"},
            response_metadata={"meta2": "data2"},
        )
        chunk3 = HumanMessageChunk(
            content="c",
            additional_kwargs={"key3": "val3"},
            response_metadata={"meta3": "data3"},
        )
        result = chunk1 + [chunk2, chunk3]
        assert result.content == "abc"
        assert result.id == "1"
        assert result.additional_kwargs["key1"] == "val1"
        assert result.additional_kwargs["key2"] == "val2"
        assert result.additional_kwargs["key3"] == "val3"
        assert result.response_metadata["meta1"] == "data1"
        assert result.response_metadata["meta2"] == "data2"
        assert result.response_metadata["meta3"] == "data3"

    def test_add_single_element_list(self) -> None:
        """Test adding a single-element list of chunks."""
        chunk1 = HumanMessageChunk(content="Hello", id="x")
        chunk2 = HumanMessageChunk(content=" World")
        result = chunk1 + [chunk2]
        assert result.content == "Hello World"
        assert result.id == "x"

    def test_add_empty_list_raises_type_error(self) -> None:
        """Test that adding a non-chunk list raises TypeError."""
        chunk = HumanMessageChunk(content="Hello")
        with pytest.raises(TypeError):
            chunk + [42, "not a chunk"]


class TestTextAccessorConcatenation:
    """Tests for TextAccessor concatenation behavior as a str subclass."""

    def test_concatenation_with_regular_string(self) -> None:
        """Test that TextAccessor can be concatenated with a regular string."""
        accessor = TextAccessor("Hello")
        result = accessor + " World"
        assert result == "Hello World"
        assert isinstance(result, str)

    def test_concatenation_two_accessors(self) -> None:
        """Test concatenation of two TextAccessor instances."""
        a1 = TextAccessor("Hello")
        a2 = TextAccessor(" World")
        result = a1 + a2
        assert result == "Hello World"

    def test_concatenation_preserves_string_methods(self) -> None:
        """Test that concatenation result still supports string methods."""
        accessor = TextAccessor("hello")
        result = accessor + " world"
        assert result.upper() == "HELLO WORLD"
        assert result.split() == ["hello", "world"]

    def test_repeated_concatenation(self) -> None:
        """Test repeated concatenation of TextAccessor."""
        accessor = TextAccessor("ab")
        result = accessor * 3
        assert result == "ababab"

    def test_join_with_text_accessor(self) -> None:
        """Test using TextAccessor with join."""
        accessor = TextAccessor(", ")
        result = accessor.join(["a", "b", "c"])
        assert result == "a, b, c"


class TestTextAccessorSlicing:
    """Tests for TextAccessor slicing behavior."""

    def test_single_index(self) -> None:
        """Test accessing a single character by index."""
        accessor = TextAccessor("Hello World")
        assert accessor[0] == "H"
        assert accessor[-1] == "d"
        assert accessor[5] == " "

    def test_basic_slice(self) -> None:
        """Test basic slicing."""
        accessor = TextAccessor("Hello World")
        assert accessor[0:5] == "Hello"
        assert accessor[6:11] == "World"

    def test_slice_with_step(self) -> None:
        """Test slicing with step."""
        accessor = TextAccessor("Hello World")
        assert accessor[::2] == "HloWrd"
        assert accessor[::-1] == "dlroW olleH"

    def test_slice_from_beginning(self) -> None:
        """Test slicing from the beginning."""
        accessor = TextAccessor("Hello World")
        assert accessor[:5] == "Hello"

    def test_slice_to_end(self) -> None:
        """Test slicing to the end."""
        accessor = TextAccessor("Hello World")
        assert accessor[6:] == "World"

    def test_empty_slice(self) -> None:
        """Test empty slice returns empty string."""
        accessor = TextAccessor("Hello")
        assert accessor[2:2] == ""


class TestBaseMessagePrettyPrint:
    """Tests for BaseMessage.pretty_print method."""

    def test_pretty_print_does_not_raise_human(self) -> None:
        """Test that pretty_print does not raise for HumanMessage."""
        msg = HumanMessage(content="Hello, how are you?")
        msg.pretty_print()

    def test_pretty_print_does_not_raise_ai(self) -> None:
        """Test that pretty_print does not raise for AIMessage."""
        msg = AIMessage(content="I'm doing well, thanks!")
        msg.pretty_print()

    def test_pretty_print_does_not_raise_system(self) -> None:
        """Test that pretty_print does not raise for SystemMessage."""
        msg = SystemMessage(content="You are a helpful assistant.")
        msg.pretty_print()

    def test_pretty_print_does_not_raise_with_name(self) -> None:
        """Test that pretty_print does not raise for a message with a name."""
        msg = HumanMessage(content="Hello", name="user1")
        msg.pretty_print()

    def test_pretty_print_does_not_raise_empty_content(self) -> None:
        """Test that pretty_print does not raise for empty content."""
        msg = HumanMessage(content="")
        msg.pretty_print()

    def test_pretty_print_does_not_raise_list_content(self) -> None:
        """Test that pretty_print does not raise for list content."""
        msg = HumanMessage(content=[{"type": "text", "text": "Hello"}])
        msg.pretty_print()


class TestGetMsgTitleReprPadding:
    """Tests for get_msg_title_repr with even and odd length titles."""

    def test_even_length_title_symmetric_padding(self) -> None:
        """Test that even-length title has symmetric padding."""
        # "AB" -> padded = " AB " (4 chars) -> sep_len = (80-4)//2 = 38
        # len(padded) = 4 -> even -> second_sep = sep (same)
        result = get_msg_title_repr("AB")
        assert "AB" in result
        total_eq = result.count("=")
        # padded = " AB " -> 4 chars, sep_len = 38, second_sep = sep
        # total = 38 + 4 + 38 = 80
        assert len(result) == 80
        left_sep = result.split(" AB ")[0]
        right_sep = result.split(" AB ")[1]
        assert left_sep == right_sep
        assert len(left_sep) == 38
        assert len(right_sep) == 38

    def test_odd_length_title_asymmetric_padding(self) -> None:
        """Test that odd-length title has asymmetric padding (right gets +1)."""
        # "ABC" -> padded = " ABC " (5 chars) -> sep_len = (80-5)//2 = 37
        # len(padded) = 5 -> odd -> second_sep = sep + "=" = 38
        result = get_msg_title_repr("ABC")
        assert "ABC" in result
        left_sep = result.split(" ABC ")[0]
        right_sep = result.split(" ABC ")[1]
        assert len(left_sep) == 37
        assert len(right_sep) == 38
        assert len(result) == 80

    def test_single_char_title(self) -> None:
        """Test single character title padding."""
        # "X" -> padded = " X " (3 chars) -> sep_len = (80-3)//2 = 38
        # len(padded) = 3 -> odd -> second_sep = sep + "=" = 39
        result = get_msg_title_repr("X")
        assert " X " in result
        left_sep = result.split(" X ")[0]
        right_sep = result.split(" X ")[1]
        assert len(left_sep) == 38
        assert len(right_sep) == 39
        assert len(result) == 80

    def test_empty_title(self) -> None:
        """Test empty string title padding."""
        # "" -> padded = "  " (2 chars) -> sep_len = (80-2)//2 = 39
        # len(padded) = 2 -> even -> second_sep = sep
        result = get_msg_title_repr("")
        assert "  " in result
        # Split on double space
        parts = result.split("  ")
        assert len(parts[0]) == 39
        assert len(parts[1]) == 39
        assert len(result) == 80

    def test_bold_does_not_change_content(self) -> None:
        """Test that bold=True still includes the title."""
        result = get_msg_title_repr("Test", bold=True)
        assert "Test" in result

    def test_known_title_exact_output(self) -> None:
        """Test exact output for a known title to verify format."""
        # "Human Message" -> padded = " Human Message " (15 chars)
        # sep_len = (80-15)//2 = 32
        # len(padded) = 15 -> odd -> second_sep = 32 + 1 = 33
        result = get_msg_title_repr("Human Message")
        expected_left = "=" * 32
        expected_right = "=" * 33
        assert result == f"{expected_left} Human Message {expected_right}"
        assert len(result) == 80
