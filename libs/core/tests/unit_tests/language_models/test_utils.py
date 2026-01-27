"""Tests for langchain_core.language_models._utils module."""

from typing import Any

from langchain_core.language_models._utils import (
    _ensure_message_copy,
    _parse_data_uri,
    _update_content_block,
    _update_message_content_to_blocks,
    is_openai_data_block,
)
from langchain_core.messages import AIMessage, HumanMessage


class TestIsOpenaiDataBlock:
    """Tests for is_openai_data_block function."""

    # --- Image URL blocks ---

    def test_image_url_block_valid(self) -> None:
        """Test valid image_url block returns True."""
        block = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        }
        assert is_openai_data_block(block) is True

    def test_image_url_block_with_detail(self) -> None:
        """Test image_url block with detail field returns True."""
        block = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png", "detail": "high"},
            "detail": "high",
        }
        assert is_openai_data_block(block) is True

    def test_image_url_block_with_filter_image(self) -> None:
        """Test image_url block with image filter returns True."""
        block = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        }
        assert is_openai_data_block(block, filter_="image") is True

    def test_image_url_block_with_filter_audio(self) -> None:
        """Test image_url block with audio filter returns False."""
        block = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        }
        assert is_openai_data_block(block, filter_="audio") is False

    def test_image_url_block_with_filter_file(self) -> None:
        """Test image_url block with file filter returns False."""
        block = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        }
        assert is_openai_data_block(block, filter_="file") is False

    def test_image_url_block_missing_url(self) -> None:
        """Test image_url block without url returns False."""
        block = {
            "type": "image_url",
            "image_url": {},
        }
        assert is_openai_data_block(block) is False

    def test_image_url_block_url_not_string(self) -> None:
        """Test image_url block with non-string url returns False."""
        block = {
            "type": "image_url",
            "image_url": {"url": 123},
        }
        assert is_openai_data_block(block) is False

    def test_image_url_block_image_url_not_dict(self) -> None:
        """Test image_url block with non-dict image_url returns False."""
        block = {
            "type": "image_url",
            "image_url": "https://example.com/image.png",
        }
        assert is_openai_data_block(block) is False

    def test_image_url_block_extra_keys(self) -> None:
        """Test image_url block with extra keys returns False."""
        block = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
            "extra_key": "value",
        }
        assert is_openai_data_block(block) is False

    # --- Input audio blocks ---

    def test_input_audio_block_valid(self) -> None:
        """Test valid input_audio block returns True."""
        block = {
            "type": "input_audio",
            "input_audio": {"data": "base64data", "format": "wav"},
        }
        assert is_openai_data_block(block) is True

    def test_input_audio_block_with_filter_audio(self) -> None:
        """Test input_audio block with audio filter returns True."""
        block = {
            "type": "input_audio",
            "input_audio": {"data": "base64data", "format": "mp3"},
        }
        assert is_openai_data_block(block, filter_="audio") is True

    def test_input_audio_block_with_filter_image(self) -> None:
        """Test input_audio block with image filter returns False."""
        block = {
            "type": "input_audio",
            "input_audio": {"data": "base64data", "format": "wav"},
        }
        assert is_openai_data_block(block, filter_="image") is False

    def test_input_audio_block_missing_data(self) -> None:
        """Test input_audio block without data returns False."""
        block = {
            "type": "input_audio",
            "input_audio": {"format": "wav"},
        }
        assert is_openai_data_block(block) is False

    def test_input_audio_block_missing_format(self) -> None:
        """Test input_audio block without format returns False."""
        block = {
            "type": "input_audio",
            "input_audio": {"data": "base64data"},
        }
        assert is_openai_data_block(block) is False

    def test_input_audio_block_data_not_string(self) -> None:
        """Test input_audio block with non-string data returns False."""
        block = {
            "type": "input_audio",
            "input_audio": {"data": 123, "format": "wav"},
        }
        assert is_openai_data_block(block) is False

    def test_input_audio_block_format_not_string(self) -> None:
        """Test input_audio block with non-string format returns False."""
        block = {
            "type": "input_audio",
            "input_audio": {"data": "base64data", "format": 123},
        }
        assert is_openai_data_block(block) is False

    def test_input_audio_block_input_audio_not_dict(self) -> None:
        """Test input_audio block with non-dict input_audio returns False."""
        block = {
            "type": "input_audio",
            "input_audio": "base64data",
        }
        assert is_openai_data_block(block) is False

    # --- File blocks ---

    def test_file_block_with_file_data(self) -> None:
        """Test file block with file_data returns True."""
        block = {
            "type": "file",
            "file": {"file_data": "base64data"},
        }
        assert is_openai_data_block(block) is True

    def test_file_block_with_file_id(self) -> None:
        """Test file block with file_id returns True."""
        block = {
            "type": "file",
            "file": {"file_id": "file-123"},
        }
        assert is_openai_data_block(block) is True

    def test_file_block_with_filter_file(self) -> None:
        """Test file block with file filter returns True."""
        block = {
            "type": "file",
            "file": {"file_data": "base64data"},
        }
        assert is_openai_data_block(block, filter_="file") is True

    def test_file_block_with_filter_image(self) -> None:
        """Test file block with image filter returns False."""
        block = {
            "type": "file",
            "file": {"file_data": "base64data"},
        }
        assert is_openai_data_block(block, filter_="image") is False

    def test_file_block_missing_file_data_and_file_id(self) -> None:
        """Test file block without file_data or file_id returns False."""
        block = {
            "type": "file",
            "file": {"filename": "test.pdf"},
        }
        assert is_openai_data_block(block) is False

    def test_file_block_file_data_not_string(self) -> None:
        """Test file block with non-string file_data returns False."""
        block = {
            "type": "file",
            "file": {"file_data": 123},
        }
        assert is_openai_data_block(block) is False

    def test_file_block_file_id_not_string(self) -> None:
        """Test file block with non-string file_id returns False."""
        block = {
            "type": "file",
            "file": {"file_id": 123},
        }
        assert is_openai_data_block(block) is False

    def test_file_block_file_not_dict(self) -> None:
        """Test file block with non-dict file returns False."""
        block = {
            "type": "file",
            "file": "base64data",
        }
        assert is_openai_data_block(block) is False

    # --- Invalid/unknown types ---

    def test_unknown_type(self) -> None:
        """Test block with unknown type returns False."""
        block = {
            "type": "unknown",
            "data": "something",
        }
        assert is_openai_data_block(block) is False

    def test_text_type(self) -> None:
        """Test text block returns False."""
        block = {
            "type": "text",
            "text": "Hello world",
        }
        assert is_openai_data_block(block) is False

    def test_missing_type(self) -> None:
        """Test block without type returns False."""
        block = {
            "image_url": {"url": "https://example.com/image.png"},
        }
        assert is_openai_data_block(block) is False

    def test_empty_block(self) -> None:
        """Test empty block returns False."""
        block: dict[str, Any] = {}
        assert is_openai_data_block(block) is False


class TestParseDataUri:
    """Tests for _parse_data_uri function."""

    def test_valid_data_uri_image_jpeg(self) -> None:
        """Test parsing valid JPEG data URI."""
        uri = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        result = _parse_data_uri(uri)
        assert result is not None
        assert result["source_type"] == "base64"
        assert result["mime_type"] == "image/jpeg"
        assert result["data"] == "/9j/4AAQSkZJRg..."

    def test_valid_data_uri_image_png(self) -> None:
        """Test parsing valid PNG data URI."""
        uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"
        result = _parse_data_uri(uri)
        assert result is not None
        assert result["source_type"] == "base64"
        assert result["mime_type"] == "image/png"
        assert result["data"] == "iVBORw0KGgoAAAANSUhEUgAAAAUA"

    def test_valid_data_uri_application_pdf(self) -> None:
        """Test parsing valid PDF data URI."""
        uri = "data:application/pdf;base64,JVBERi0xLjQKJeLjz9MKMSAwIG9iago="
        result = _parse_data_uri(uri)
        assert result is not None
        assert result["source_type"] == "base64"
        assert result["mime_type"] == "application/pdf"
        assert result["data"] == "JVBERi0xLjQKJeLjz9MKMSAwIG9iago="

    def test_valid_data_uri_audio_wav(self) -> None:
        """Test parsing valid WAV audio data URI."""
        uri = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEA"
        result = _parse_data_uri(uri)
        assert result is not None
        assert result["source_type"] == "base64"
        assert result["mime_type"] == "audio/wav"
        assert result["data"] == "UklGRiQAAABXQVZFZm10IBAAAAABAAEA"

    def test_invalid_data_uri_no_data_prefix(self) -> None:
        """Test parsing URI without data: prefix returns None."""
        uri = "https://example.com/image.png"
        result = _parse_data_uri(uri)
        assert result is None

    def test_invalid_data_uri_no_base64(self) -> None:
        """Test parsing URI without base64 encoding returns None."""
        uri = "data:image/png,rawdata"
        result = _parse_data_uri(uri)
        assert result is None

    def test_invalid_data_uri_empty_mime_type(self) -> None:
        """Test parsing URI with empty mime type returns None."""
        uri = "data:;base64,somedata"
        result = _parse_data_uri(uri)
        assert result is None

    def test_invalid_data_uri_empty_data(self) -> None:
        """Test parsing URI with empty data returns None."""
        uri = "data:image/png;base64,"
        result = _parse_data_uri(uri)
        assert result is None

    def test_invalid_data_uri_malformed(self) -> None:
        """Test parsing malformed URI returns None."""
        uri = "data:image/png"
        result = _parse_data_uri(uri)
        assert result is None

    def test_empty_string(self) -> None:
        """Test parsing empty string returns None."""
        uri = ""
        result = _parse_data_uri(uri)
        assert result is None


class TestEnsureMessageCopy:
    """Tests for _ensure_message_copy function."""

    def test_creates_copy_when_same_reference(self) -> None:
        """Test that a copy is created when message and formatted_message are same."""
        message = HumanMessage(content=[{"type": "text", "text": "Hello"}])
        formatted_message = message  # Same reference

        result = _ensure_message_copy(message, formatted_message)

        assert result is not message
        assert result.content == message.content
        assert isinstance(result.content, list)

    def test_returns_existing_copy_when_different_reference(self) -> None:
        """Test that existing copy is returned when already different."""
        message = HumanMessage(content=[{"type": "text", "text": "Hello"}])
        formatted_message = message.model_copy()
        formatted_message.content = list(formatted_message.content)

        result = _ensure_message_copy(message, formatted_message)

        assert result is formatted_message
        assert result is not message

    def test_content_is_shallow_copied(self) -> None:
        """Test that content list is shallow copied."""
        message = HumanMessage(content=[{"type": "text", "text": "Hello"}])
        formatted_message = message

        result = _ensure_message_copy(message, formatted_message)

        # Content should be a new list
        assert result.content is not message.content
        # But the dict inside should be the same reference (shallow copy)
        assert result.content[0] is message.content[0]


class TestUpdateContentBlock:
    """Tests for _update_content_block function."""

    def test_updates_content_block_at_index(self) -> None:
        """Test updating content block at specific index."""
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]
        )
        new_block = {"type": "image", "url": "https://example.com/image.png"}

        _update_content_block(message, 1, new_block)

        assert message.content[1] == new_block
        assert message.content[0] == {"type": "text", "text": "Hello"}

    def test_updates_first_block(self) -> None:
        """Test updating first content block."""
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]
        )
        new_block = {"type": "audio", "base64": "data"}

        _update_content_block(message, 0, new_block)

        assert message.content[0] == new_block


class TestUpdateMessageContentToBlocks:
    """Tests for _update_message_content_to_blocks function."""

    def test_updates_content_to_content_blocks(self) -> None:
        """Test that content is updated to content_blocks format."""
        message = AIMessage(content="Hello world")

        result = _update_message_content_to_blocks(message, "v1")

        assert result.content == message.content_blocks
        assert result.response_metadata["output_version"] == "v1"

    def test_preserves_original_message(self) -> None:
        """Test that original message is not modified."""
        message = AIMessage(content="Hello world")
        original_content = message.content

        result = _update_message_content_to_blocks(message, "v1")

        assert message.content == original_content
        assert result is not message

    def test_with_complex_content(self) -> None:
        """Test with complex content blocks."""
        message = AIMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "tool_use", "id": "123", "name": "test", "input": {}},
            ]
        )

        result = _update_message_content_to_blocks(message, "v1")

        assert result.response_metadata["output_version"] == "v1"
        assert isinstance(result.content, list)

    def test_with_different_output_version(self) -> None:
        """Test with different output version string."""
        message = AIMessage(content="Test")

        result = _update_message_content_to_blocks(message, "v2")

        assert result.response_metadata["output_version"] == "v2"

    def test_preserves_existing_response_metadata(self) -> None:
        """Test that existing response_metadata is preserved."""
        message = AIMessage(
            content="Hello",
            response_metadata={"model": "test-model", "usage": {"tokens": 10}},
        )

        result = _update_message_content_to_blocks(message, "v1")

        assert result.response_metadata["model"] == "test-model"
        assert result.response_metadata["usage"] == {"tokens": 10}
        assert result.response_metadata["output_version"] == "v1"
