"""Tests for langchain_core.messages.content module."""

import pytest

from langchain_core.messages.content import (
    # TypedDict types
    Citation,
    NonStandardAnnotation,
    TextContentBlock,
    ToolCall,
    ToolCallChunk,
    InvalidToolCall,
    ServerToolCall,
    ServerToolCallChunk,
    ServerToolResult,
    ReasoningContentBlock,
    ImageContentBlock,
    VideoContentBlock,
    AudioContentBlock,
    PlainTextContentBlock,
    FileContentBlock,
    NonStandardContentBlock,
    # Utilities
    KNOWN_BLOCK_TYPES,
    is_data_content_block,
    # Factory functions
    create_text_block,
    create_image_block,
    create_video_block,
    create_audio_block,
    create_file_block,
    create_plaintext_block,
    create_tool_call,
    create_reasoning_block,
    create_citation,
    create_non_standard_block,
)


class TestKnownBlockTypes:
    """Tests for KNOWN_BLOCK_TYPES constant."""

    def test_known_block_types_contains_text(self) -> None:
        """Test that KNOWN_BLOCK_TYPES contains 'text'."""
        assert "text" in KNOWN_BLOCK_TYPES

    def test_known_block_types_contains_reasoning(self) -> None:
        """Test that KNOWN_BLOCK_TYPES contains 'reasoning'."""
        assert "reasoning" in KNOWN_BLOCK_TYPES

    def test_known_block_types_contains_tool_call(self) -> None:
        """Test that KNOWN_BLOCK_TYPES contains 'tool_call'."""
        assert "tool_call" in KNOWN_BLOCK_TYPES

    def test_known_block_types_contains_image(self) -> None:
        """Test that KNOWN_BLOCK_TYPES contains 'image'."""
        assert "image" in KNOWN_BLOCK_TYPES

    def test_known_block_types_contains_audio(self) -> None:
        """Test that KNOWN_BLOCK_TYPES contains 'audio'."""
        assert "audio" in KNOWN_BLOCK_TYPES

    def test_known_block_types_contains_file(self) -> None:
        """Test that KNOWN_BLOCK_TYPES contains 'file'."""
        assert "file" in KNOWN_BLOCK_TYPES

    def test_known_block_types_contains_video(self) -> None:
        """Test that KNOWN_BLOCK_TYPES contains 'video'."""
        assert "video" in KNOWN_BLOCK_TYPES

    def test_known_block_types_contains_non_standard(self) -> None:
        """Test that KNOWN_BLOCK_TYPES contains 'non_standard'."""
        assert "non_standard" in KNOWN_BLOCK_TYPES


class TestIsDataContentBlock:
    """Tests for is_data_content_block function."""

    def test_image_block_with_url(self) -> None:
        """Test that image block with URL is recognized."""
        block = {"type": "image", "url": "https://example.com/image.png"}
        assert is_data_content_block(block) is True

    def test_image_block_with_base64(self) -> None:
        """Test that image block with base64 is recognized."""
        block = {"type": "image", "base64": "dGVzdA==", "mime_type": "image/png"}
        assert is_data_content_block(block) is True

    def test_image_block_with_file_id(self) -> None:
        """Test that image block with file_id is recognized."""
        block = {"type": "image", "file_id": "file-123"}
        assert is_data_content_block(block) is True

    def test_audio_block_with_url(self) -> None:
        """Test that audio block with URL is recognized."""
        block = {"type": "audio", "url": "https://example.com/audio.mp3"}
        assert is_data_content_block(block) is True

    def test_audio_block_with_base64(self) -> None:
        """Test that audio block with base64 is recognized."""
        block = {"type": "audio", "base64": "dGVzdA==", "mime_type": "audio/mp3"}
        assert is_data_content_block(block) is True

    def test_file_block_with_url(self) -> None:
        """Test that file block with URL is recognized."""
        block = {"type": "file", "url": "https://example.com/doc.pdf"}
        assert is_data_content_block(block) is True

    def test_file_block_with_base64(self) -> None:
        """Test that file block with base64 is recognized."""
        block = {"type": "file", "base64": "dGVzdA==", "mime_type": "application/pdf"}
        assert is_data_content_block(block) is True

    def test_video_block_with_url(self) -> None:
        """Test that video block with URL is recognized."""
        block = {"type": "video", "url": "https://example.com/video.mp4"}
        assert is_data_content_block(block) is True

    def test_plaintext_block(self) -> None:
        """Test that plaintext block is recognized."""
        block = {"type": "text-plain", "text": "Hello world", "mime_type": "text/plain"}
        assert is_data_content_block(block) is True

    def test_text_block_is_not_data_block(self) -> None:
        """Test that regular text block is not a data block."""
        block = {"type": "text", "text": "Hello"}
        assert is_data_content_block(block) is False

    def test_tool_call_is_not_data_block(self) -> None:
        """Test that tool_call block is not a data block."""
        block = {"type": "tool_call", "name": "test", "args": {}, "id": "1"}
        assert is_data_content_block(block) is False

    def test_v0_style_image_block_with_source_type_url(self) -> None:
        """Test v0 style image block with source_type='url'."""
        block = {"type": "image", "source_type": "url", "url": "https://example.com/img.png"}
        assert is_data_content_block(block) is True

    def test_v0_style_image_block_with_source_type_base64(self) -> None:
        """Test v0 style image block with source_type='base64'."""
        block = {"type": "image", "source_type": "base64", "data": "dGVzdA=="}
        assert is_data_content_block(block) is True

    def test_v0_style_block_with_source_type_id(self) -> None:
        """Test v0 style block with source_type='id'."""
        block = {"type": "file", "source_type": "id", "id": "file-123"}
        assert is_data_content_block(block) is True

    def test_block_without_type_is_not_data_block(self) -> None:
        """Test that block without type is not a data block."""
        block = {"url": "https://example.com/image.png"}
        assert is_data_content_block(block) is False

    def test_block_with_unknown_type(self) -> None:
        """Test that block with unknown type is not a data block."""
        block = {"type": "custom_type", "url": "https://example.com"}
        assert is_data_content_block(block) is False


class TestCreateTextBlock:
    """Tests for create_text_block factory function."""

    def test_basic_text_block(self) -> None:
        """Test creating a basic text block."""
        block = create_text_block("Hello world")
        assert block["type"] == "text"
        assert block["text"] == "Hello world"
        assert "id" in block
        assert block["id"].startswith("lc_")

    def test_text_block_with_custom_id(self) -> None:
        """Test creating a text block with custom ID."""
        block = create_text_block("Hello", id="custom-id")
        assert block["id"] == "custom-id"

    def test_text_block_with_annotations(self) -> None:
        """Test creating a text block with annotations."""
        annotations: list = [
            {"type": "citation", "url": "https://example.com"}
        ]
        block = create_text_block("Hello", annotations=annotations)
        assert block["annotations"] == annotations

    def test_text_block_with_index(self) -> None:
        """Test creating a text block with index."""
        block = create_text_block("Hello", index=0)
        assert block["index"] == 0

    def test_text_block_with_extras(self) -> None:
        """Test creating a text block with extra kwargs."""
        block = create_text_block("Hello", custom_field="custom_value")
        assert block["extras"]["custom_field"] == "custom_value"

    def test_text_block_empty_text(self) -> None:
        """Test creating a text block with empty text."""
        block = create_text_block("")
        assert block["text"] == ""


class TestCreateImageBlock:
    """Tests for create_image_block factory function."""

    def test_image_block_with_url(self) -> None:
        """Test creating an image block with URL."""
        block = create_image_block(url="https://example.com/image.png")
        assert block["type"] == "image"
        assert block["url"] == "https://example.com/image.png"
        assert "id" in block

    def test_image_block_with_base64(self) -> None:
        """Test creating an image block with base64."""
        block = create_image_block(base64="dGVzdA==", mime_type="image/png")
        assert block["type"] == "image"
        assert block["base64"] == "dGVzdA=="
        assert block["mime_type"] == "image/png"

    def test_image_block_with_file_id(self) -> None:
        """Test creating an image block with file_id."""
        block = create_image_block(file_id="file-123")
        assert block["type"] == "image"
        assert block["file_id"] == "file-123"

    def test_image_block_requires_source(self) -> None:
        """Test that create_image_block raises error without source."""
        with pytest.raises(ValueError, match="Must provide one of"):
            create_image_block()

    def test_image_block_with_custom_id(self) -> None:
        """Test creating an image block with custom ID."""
        block = create_image_block(url="https://example.com/img.png", id="img-123")
        assert block["id"] == "img-123"

    def test_image_block_with_index(self) -> None:
        """Test creating an image block with index."""
        block = create_image_block(url="https://example.com/img.png", index=0)
        assert block["index"] == 0

    def test_image_block_with_extras(self) -> None:
        """Test creating an image block with extra kwargs."""
        block = create_image_block(url="https://example.com/img.png", alt="Test image")
        assert block["extras"]["alt"] == "Test image"


class TestCreateVideoBlock:
    """Tests for create_video_block factory function."""

    def test_video_block_with_url(self) -> None:
        """Test creating a video block with URL."""
        block = create_video_block(url="https://example.com/video.mp4")
        assert block["type"] == "video"
        assert block["url"] == "https://example.com/video.mp4"

    def test_video_block_with_base64(self) -> None:
        """Test creating a video block with base64."""
        block = create_video_block(base64="dGVzdA==", mime_type="video/mp4")
        assert block["type"] == "video"
        assert block["base64"] == "dGVzdA=="
        assert block["mime_type"] == "video/mp4"

    def test_video_block_with_file_id(self) -> None:
        """Test creating a video block with file_id."""
        block = create_video_block(file_id="file-123")
        assert block["type"] == "video"
        assert block["file_id"] == "file-123"

    def test_video_block_requires_source(self) -> None:
        """Test that create_video_block raises error without source."""
        with pytest.raises(ValueError, match="Must provide one of"):
            create_video_block()

    def test_video_block_base64_requires_mime_type(self) -> None:
        """Test that base64 video requires mime_type."""
        with pytest.raises(ValueError, match="mime_type is required"):
            create_video_block(base64="dGVzdA==")

    def test_video_block_with_custom_id(self) -> None:
        """Test creating a video block with custom ID."""
        block = create_video_block(url="https://example.com/video.mp4", id="vid-123")
        assert block["id"] == "vid-123"


class TestCreateAudioBlock:
    """Tests for create_audio_block factory function."""

    def test_audio_block_with_url(self) -> None:
        """Test creating an audio block with URL."""
        block = create_audio_block(url="https://example.com/audio.mp3")
        assert block["type"] == "audio"
        assert block["url"] == "https://example.com/audio.mp3"

    def test_audio_block_with_base64(self) -> None:
        """Test creating an audio block with base64."""
        block = create_audio_block(base64="dGVzdA==", mime_type="audio/mp3")
        assert block["type"] == "audio"
        assert block["base64"] == "dGVzdA=="
        assert block["mime_type"] == "audio/mp3"

    def test_audio_block_with_file_id(self) -> None:
        """Test creating an audio block with file_id."""
        block = create_audio_block(file_id="file-123")
        assert block["type"] == "audio"
        assert block["file_id"] == "file-123"

    def test_audio_block_requires_source(self) -> None:
        """Test that create_audio_block raises error without source."""
        with pytest.raises(ValueError, match="Must provide one of"):
            create_audio_block()

    def test_audio_block_base64_requires_mime_type(self) -> None:
        """Test that base64 audio requires mime_type."""
        with pytest.raises(ValueError, match="mime_type is required"):
            create_audio_block(base64="dGVzdA==")

    def test_audio_block_with_custom_id(self) -> None:
        """Test creating an audio block with custom ID."""
        block = create_audio_block(url="https://example.com/audio.mp3", id="aud-123")
        assert block["id"] == "aud-123"


class TestCreateFileBlock:
    """Tests for create_file_block factory function."""

    def test_file_block_with_url(self) -> None:
        """Test creating a file block with URL."""
        block = create_file_block(url="https://example.com/doc.pdf")
        assert block["type"] == "file"
        assert block["url"] == "https://example.com/doc.pdf"

    def test_file_block_with_base64(self) -> None:
        """Test creating a file block with base64."""
        block = create_file_block(base64="dGVzdA==", mime_type="application/pdf")
        assert block["type"] == "file"
        assert block["base64"] == "dGVzdA=="
        assert block["mime_type"] == "application/pdf"

    def test_file_block_with_file_id(self) -> None:
        """Test creating a file block with file_id."""
        block = create_file_block(file_id="file-123")
        assert block["type"] == "file"
        assert block["file_id"] == "file-123"

    def test_file_block_requires_source(self) -> None:
        """Test that create_file_block raises error without source."""
        with pytest.raises(ValueError, match="Must provide one of"):
            create_file_block()

    def test_file_block_base64_requires_mime_type(self) -> None:
        """Test that base64 file requires mime_type."""
        with pytest.raises(ValueError, match="mime_type is required"):
            create_file_block(base64="dGVzdA==")

    def test_file_block_with_custom_id(self) -> None:
        """Test creating a file block with custom ID."""
        block = create_file_block(url="https://example.com/doc.pdf", id="file-custom")
        assert block["id"] == "file-custom"


class TestCreatePlaintextBlock:
    """Tests for create_plaintext_block factory function."""

    def test_plaintext_block_with_text(self) -> None:
        """Test creating a plaintext block with text."""
        block = create_plaintext_block(text="Hello world")
        assert block["type"] == "text-plain"
        assert block["text"] == "Hello world"
        assert block["mime_type"] == "text/plain"

    def test_plaintext_block_with_url(self) -> None:
        """Test creating a plaintext block with URL."""
        block = create_plaintext_block(url="https://example.com/file.txt")
        assert block["type"] == "text-plain"
        assert block["url"] == "https://example.com/file.txt"

    def test_plaintext_block_with_base64(self) -> None:
        """Test creating a plaintext block with base64."""
        block = create_plaintext_block(base64="SGVsbG8gd29ybGQ=")
        assert block["type"] == "text-plain"
        assert block["base64"] == "SGVsbG8gd29ybGQ="

    def test_plaintext_block_with_file_id(self) -> None:
        """Test creating a plaintext block with file_id."""
        block = create_plaintext_block(file_id="file-123")
        assert block["type"] == "text-plain"
        assert block["file_id"] == "file-123"

    def test_plaintext_block_with_title(self) -> None:
        """Test creating a plaintext block with title."""
        block = create_plaintext_block(text="Hello", title="My Document")
        assert block["title"] == "My Document"

    def test_plaintext_block_with_context(self) -> None:
        """Test creating a plaintext block with context."""
        block = create_plaintext_block(text="Hello", context="Important information")
        assert block["context"] == "Important information"

    def test_plaintext_block_with_custom_id(self) -> None:
        """Test creating a plaintext block with custom ID."""
        block = create_plaintext_block(text="Hello", id="txt-123")
        assert block["id"] == "txt-123"


class TestCreateToolCall:
    """Tests for create_tool_call factory function."""

    def test_basic_tool_call(self) -> None:
        """Test creating a basic tool call."""
        block = create_tool_call(name="test_tool", args={"param": "value"})
        assert block["type"] == "tool_call"
        assert block["name"] == "test_tool"
        assert block["args"] == {"param": "value"}
        assert "id" in block
        assert block["id"].startswith("lc_")

    def test_tool_call_with_custom_id(self) -> None:
        """Test creating a tool call with custom ID."""
        block = create_tool_call(name="test_tool", args={}, id="call-123")
        assert block["id"] == "call-123"

    def test_tool_call_with_index(self) -> None:
        """Test creating a tool call with index."""
        block = create_tool_call(name="test_tool", args={}, index=0)
        assert block["index"] == 0

    def test_tool_call_with_extras(self) -> None:
        """Test creating a tool call with extras."""
        block = create_tool_call(name="test_tool", args={}, custom="value")
        assert block["extras"]["custom"] == "value"

    def test_tool_call_empty_args(self) -> None:
        """Test creating a tool call with empty args."""
        block = create_tool_call(name="test_tool", args={})
        assert block["args"] == {}

    def test_tool_call_complex_args(self) -> None:
        """Test creating a tool call with complex args."""
        complex_args = {
            "string": "value",
            "number": 42,
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }
        block = create_tool_call(name="test_tool", args=complex_args)
        assert block["args"] == complex_args


class TestCreateReasoningBlock:
    """Tests for create_reasoning_block factory function."""

    def test_basic_reasoning_block(self) -> None:
        """Test creating a basic reasoning block."""
        block = create_reasoning_block(reasoning="Let me think about this...")
        assert block["type"] == "reasoning"
        assert block["reasoning"] == "Let me think about this..."
        assert "id" in block

    def test_reasoning_block_with_custom_id(self) -> None:
        """Test creating a reasoning block with custom ID."""
        block = create_reasoning_block(reasoning="Thinking...", id="reason-123")
        assert block["id"] == "reason-123"

    def test_reasoning_block_with_index(self) -> None:
        """Test creating a reasoning block with index."""
        block = create_reasoning_block(reasoning="Thinking...", index=0)
        assert block["index"] == 0

    def test_reasoning_block_empty_reasoning(self) -> None:
        """Test creating a reasoning block with no reasoning."""
        block = create_reasoning_block()
        assert block["reasoning"] == ""

    def test_reasoning_block_with_extras(self) -> None:
        """Test creating a reasoning block with extras."""
        block = create_reasoning_block(reasoning="Thinking...", signature="abc123")
        assert block["extras"]["signature"] == "abc123"


class TestCreateCitation:
    """Tests for create_citation factory function."""

    def test_basic_citation(self) -> None:
        """Test creating a basic citation."""
        block = create_citation(url="https://example.com/source")
        assert block["type"] == "citation"
        assert block["url"] == "https://example.com/source"
        assert "id" in block

    def test_citation_with_all_fields(self) -> None:
        """Test creating a citation with all fields."""
        block = create_citation(
            url="https://example.com/source",
            title="Source Document",
            start_index=0,
            end_index=100,
            cited_text="This is the cited text.",
            id="cite-123",
        )
        assert block["url"] == "https://example.com/source"
        assert block["title"] == "Source Document"
        assert block["start_index"] == 0
        assert block["end_index"] == 100
        assert block["cited_text"] == "This is the cited text."
        assert block["id"] == "cite-123"

    def test_citation_with_extras(self) -> None:
        """Test creating a citation with extras."""
        block = create_citation(url="https://example.com", custom_field="value")
        assert block["extras"]["custom_field"] == "value"


class TestCreateNonStandardBlock:
    """Tests for create_non_standard_block factory function."""

    def test_basic_non_standard_block(self) -> None:
        """Test creating a basic non-standard block."""
        block = create_non_standard_block(value={"custom": "data"})
        assert block["type"] == "non_standard"
        assert block["value"] == {"custom": "data"}
        assert "id" in block

    def test_non_standard_block_with_custom_id(self) -> None:
        """Test creating a non-standard block with custom ID."""
        block = create_non_standard_block(value={"key": "value"}, id="ns-123")
        assert block["id"] == "ns-123"

    def test_non_standard_block_with_index(self) -> None:
        """Test creating a non-standard block with index."""
        block = create_non_standard_block(value={"key": "value"}, index=0)
        assert block["index"] == 0

    def test_non_standard_block_complex_value(self) -> None:
        """Test creating a non-standard block with complex value."""
        complex_value = {
            "nested": {"deep": {"data": "value"}},
            "list": [1, 2, 3],
            "string": "text",
        }
        block = create_non_standard_block(value=complex_value)
        assert block["value"] == complex_value


class TestTypedDictStructures:
    """Tests to verify TypedDict structures are properly typed."""

    def test_text_content_block_structure(self) -> None:
        """Test TextContentBlock TypedDict structure."""
        block: TextContentBlock = {
            "type": "text",
            "text": "Hello",
        }
        assert block["type"] == "text"
        assert block["text"] == "Hello"

    def test_tool_call_structure(self) -> None:
        """Test ToolCall TypedDict structure."""
        block: ToolCall = {
            "type": "tool_call",
            "id": "123",
            "name": "test_tool",
            "args": {"param": "value"},
        }
        assert block["type"] == "tool_call"
        assert block["name"] == "test_tool"

    def test_tool_call_chunk_structure(self) -> None:
        """Test ToolCallChunk TypedDict structure."""
        block: ToolCallChunk = {
            "type": "tool_call_chunk",
            "id": "123",
            "name": "test_tool",
            "args": '{"param": "value"}',
        }
        assert block["type"] == "tool_call_chunk"
        assert block["args"] == '{"param": "value"}'

    def test_invalid_tool_call_structure(self) -> None:
        """Test InvalidToolCall TypedDict structure."""
        block: InvalidToolCall = {
            "type": "invalid_tool_call",
            "id": "123",
            "name": "test_tool",
            "args": "invalid json",
            "error": "JSON parse error",
        }
        assert block["type"] == "invalid_tool_call"
        assert block["error"] == "JSON parse error"

    def test_reasoning_content_block_structure(self) -> None:
        """Test ReasoningContentBlock TypedDict structure."""
        block: ReasoningContentBlock = {
            "type": "reasoning",
            "reasoning": "Let me think...",
        }
        assert block["type"] == "reasoning"

    def test_image_content_block_structure(self) -> None:
        """Test ImageContentBlock TypedDict structure."""
        block: ImageContentBlock = {
            "type": "image",
            "url": "https://example.com/image.png",
        }
        assert block["type"] == "image"

    def test_audio_content_block_structure(self) -> None:
        """Test AudioContentBlock TypedDict structure."""
        block: AudioContentBlock = {
            "type": "audio",
            "url": "https://example.com/audio.mp3",
        }
        assert block["type"] == "audio"

    def test_video_content_block_structure(self) -> None:
        """Test VideoContentBlock TypedDict structure."""
        block: VideoContentBlock = {
            "type": "video",
            "url": "https://example.com/video.mp4",
        }
        assert block["type"] == "video"

    def test_file_content_block_structure(self) -> None:
        """Test FileContentBlock TypedDict structure."""
        block: FileContentBlock = {
            "type": "file",
            "url": "https://example.com/doc.pdf",
        }
        assert block["type"] == "file"

    def test_plaintext_content_block_structure(self) -> None:
        """Test PlainTextContentBlock TypedDict structure."""
        block: PlainTextContentBlock = {
            "type": "text-plain",
            "mime_type": "text/plain",
            "text": "Hello",
        }
        assert block["type"] == "text-plain"

    def test_citation_structure(self) -> None:
        """Test Citation TypedDict structure."""
        block: Citation = {
            "type": "citation",
            "url": "https://example.com",
        }
        assert block["type"] == "citation"

    def test_non_standard_annotation_structure(self) -> None:
        """Test NonStandardAnnotation TypedDict structure."""
        block: NonStandardAnnotation = {
            "type": "non_standard_annotation",
            "value": {"custom": "data"},
        }
        assert block["type"] == "non_standard_annotation"

    def test_server_tool_call_structure(self) -> None:
        """Test ServerToolCall TypedDict structure."""
        block: ServerToolCall = {
            "type": "server_tool_call",
            "id": "stc-123",
            "name": "web_search",
            "args": {"query": "test"},
        }
        assert block["type"] == "server_tool_call"

    def test_server_tool_call_chunk_structure(self) -> None:
        """Test ServerToolCallChunk TypedDict structure."""
        block: ServerToolCallChunk = {
            "type": "server_tool_call_chunk",
        }
        assert block["type"] == "server_tool_call_chunk"

    def test_server_tool_result_structure(self) -> None:
        """Test ServerToolResult TypedDict structure."""
        block: ServerToolResult = {
            "type": "server_tool_result",
            "tool_call_id": "stc-123",
            "status": "success",
        }
        assert block["type"] == "server_tool_result"

    def test_non_standard_content_block_structure(self) -> None:
        """Test NonStandardContentBlock TypedDict structure."""
        block: NonStandardContentBlock = {
            "type": "non_standard",
            "value": {"provider": "custom", "data": "value"},
        }
        assert block["type"] == "non_standard"
