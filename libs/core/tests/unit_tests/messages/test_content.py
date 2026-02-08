"""Tests for langchain_core.messages.content module."""

import pytest

from langchain_core.messages.content import (
    # Utilities
    KNOWN_BLOCK_TYPES,
    AudioContentBlock,
    # TypedDict types
    Citation,
    FileContentBlock,
    ImageContentBlock,
    InvalidToolCall,
    NonStandardAnnotation,
    NonStandardContentBlock,
    PlainTextContentBlock,
    ReasoningContentBlock,
    ServerToolCall,
    ServerToolCallChunk,
    ServerToolResult,
    TextContentBlock,
    ToolCall,
    ToolCallChunk,
    VideoContentBlock,
    create_audio_block,
    create_citation,
    create_file_block,
    create_image_block,
    create_non_standard_block,
    create_plaintext_block,
    create_reasoning_block,
    # Factory functions
    create_text_block,
    create_tool_call,
    create_video_block,
    is_data_content_block,
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
        block = {
            "type": "image",
            "source_type": "url",
            "url": "https://example.com/img.png",
        }
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
        annotations: list = [{"type": "citation", "url": "https://example.com"}]
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


# ---------------------------------------------------------------------------
# Additional tests for untested functionality
# ---------------------------------------------------------------------------

from langchain_core.messages.content import (  # noqa: E402
    ContentBlock,
    DataContentBlock,
    _get_data_content_block_types,
)


class TestKnownBlockTypesSnapshot:
    """Snapshot-style tests for the exact contents of KNOWN_BLOCK_TYPES."""

    def test_known_block_types_exact_set(self) -> None:
        """KNOWN_BLOCK_TYPES must equal this exact frozen set of strings."""
        expected = {
            "text",
            "reasoning",
            "tool_call",
            "invalid_tool_call",
            "tool_call_chunk",
            "image",
            "audio",
            "file",
            "text-plain",
            "video",
            "server_tool_call",
            "server_tool_call_chunk",
            "server_tool_result",
            "non_standard",
        }
        assert KNOWN_BLOCK_TYPES == expected

    def test_known_block_types_count(self) -> None:
        """KNOWN_BLOCK_TYPES must contain exactly 14 entries."""
        assert len(KNOWN_BLOCK_TYPES) == 14

    def test_citation_not_in_known_block_types(self) -> None:
        """'citation' must NOT be in KNOWN_BLOCK_TYPES."""
        assert "citation" not in KNOWN_BLOCK_TYPES

    def test_non_standard_annotation_not_in_known_block_types(self) -> None:
        """'non_standard_annotation' must NOT be in KNOWN_BLOCK_TYPES."""
        assert "non_standard_annotation" not in KNOWN_BLOCK_TYPES


class TestGetDataContentBlockTypes:
    """Tests for the private _get_data_content_block_types helper."""

    def test_returns_tuple(self) -> None:
        """_get_data_content_block_types must return a tuple."""
        result = _get_data_content_block_types()
        assert isinstance(result, tuple)

    def test_exact_type_literals(self) -> None:
        """The returned tuple must contain exactly the expected type literals."""
        result = _get_data_content_block_types()
        expected = ("image", "video", "audio", "text-plain", "file")
        assert set(result) == set(expected)
        assert len(result) == len(expected)

    def test_each_member_is_string(self) -> None:
        """Every element must be a plain string."""
        for item in _get_data_content_block_types():
            assert isinstance(item, str)


class TestIsDataContentBlockAdditional:
    """Additional edge-case tests for is_data_content_block."""

    def test_plaintext_block_with_source_type_text_and_url(self) -> None:
        """A block with type 'text-plain' using source_type='text' and url."""
        block = {
            "type": "text-plain",
            "source_type": "text",
            "url": "https://example.com/file.txt",
        }
        assert is_data_content_block(block) is True

    def test_image_block_with_type_only_no_data_fields(self) -> None:
        """An image block with type but NO data fields should return False."""
        block = {"type": "image"}
        assert is_data_content_block(block) is False

    def test_video_block_with_file_id(self) -> None:
        """A video block identified by file_id is a data content block."""
        block = {"type": "video", "file_id": "vid-file-001"}
        assert is_data_content_block(block) is True

    def test_audio_block_with_file_id(self) -> None:
        """An audio block identified by file_id is a data content block."""
        block = {"type": "audio", "file_id": "aud-file-002"}
        assert is_data_content_block(block) is True


class TestCreateImageBlockBase64WithoutMimeType:
    """Confirm create_image_block behaviour with base64 but no mime_type."""

    def test_base64_without_mime_type_does_not_raise(self) -> None:
        """Unlike video/audio, create_image_block does NOT require mime_type for
        base64 data.  This test documents that intentional asymmetry."""
        block = create_image_block(base64="iVBORw0KGgo=")
        assert block["type"] == "image"
        assert block["base64"] == "iVBORw0KGgo="
        assert "mime_type" not in block


class TestCreateTextBlockNoExtras:
    """Tests for create_text_block when no extras are provided."""

    def test_text_block_with_none_extras(self) -> None:
        """When no **kwargs are passed, the 'extras' key must be absent."""
        block = create_text_block("hello")
        assert "extras" not in block


class TestCreatePlaintextBlockAllFields:
    """Tests for create_plaintext_block with all fields populated."""

    def test_all_fields_populated(self) -> None:
        """Every optional field is set; verify all are present in the result."""
        block = create_plaintext_block(
            text="some text",
            url="https://example.com/file.txt",
            base64="c29tZSB0ZXh0",
            file_id="file-xyz",
            title="My Document",
            context="Summary context",
            id="pt-999",
            index=3,
        )
        assert block["type"] == "text-plain"
        assert block["mime_type"] == "text/plain"
        assert block["text"] == "some text"
        assert block["url"] == "https://example.com/file.txt"
        assert block["base64"] == "c29tZSB0ZXh0"
        assert block["file_id"] == "file-xyz"
        assert block["title"] == "My Document"
        assert block["context"] == "Summary context"
        assert block["id"] == "pt-999"
        assert block["index"] == 3

    def test_plaintext_block_with_extras(self) -> None:
        """Extra kwargs end up in the 'extras' dict."""
        block = create_plaintext_block(
            text="hello",
            custom_key="custom_value",
            another="val",
        )
        assert block["extras"] == {
            "custom_key": "custom_value",
            "another": "val",
        }


class TestCreateFileBlockAllFields:
    """Tests for create_file_block with all supported fields."""

    def test_all_fields_populated(self) -> None:
        """URL, mime_type, index, and custom id must all be present."""
        block = create_file_block(
            url="https://example.com/report.pdf",
            mime_type="application/pdf",
            id="file-all",
            index=7,
        )
        assert block["type"] == "file"
        assert block["url"] == "https://example.com/report.pdf"
        assert block["mime_type"] == "application/pdf"
        assert block["id"] == "file-all"
        assert block["index"] == 7


class TestCreateCitationMinimal:
    """Tests for create_citation with minimal fields."""

    def test_citation_with_no_optional_fields(self) -> None:
        """A citation created with only 'type' (auto-set) and auto-generated id."""
        block = create_citation()
        assert block["type"] == "citation"
        assert "id" in block
        assert block["id"].startswith("lc_")
        # None-valued optional fields must be absent
        assert "url" not in block
        assert "title" not in block
        assert "start_index" not in block
        assert "end_index" not in block
        assert "cited_text" not in block
        assert "extras" not in block


class TestCreateNonStandardBlockEmptyDict:
    """Tests for create_non_standard_block with an empty dict value."""

    def test_empty_dict_value(self) -> None:
        """An empty dict is a valid value for NonStandardContentBlock."""
        block = create_non_standard_block(value={})
        assert block["type"] == "non_standard"
        assert block["value"] == {}
        assert "id" in block
        assert block["id"].startswith("lc_")


class TestCreateToolCallAutoId:
    """Tests for create_tool_call automatic id generation."""

    def test_auto_generates_id_when_not_provided(self) -> None:
        """When id is omitted, create_tool_call must auto-generate one."""
        block = create_tool_call(name="my_tool", args={"x": 1})
        assert block["id"] is not None
        assert isinstance(block["id"], str)
        assert block["id"].startswith("lc_")

    def test_auto_generated_ids_are_unique(self) -> None:
        """Successive calls must produce distinct ids."""
        block_a = create_tool_call(name="t", args={})
        block_b = create_tool_call(name="t", args={})
        assert block_a["id"] != block_b["id"]


class TestCreateReasoningBlockNoneReasoning:
    """Tests for create_reasoning_block when reasoning is explicitly None."""

    def test_none_reasoning_defaults_to_empty_string(self) -> None:
        """Passing reasoning=None must result in an empty string."""
        block = create_reasoning_block(reasoning=None)
        assert block["reasoning"] == ""
        assert block["type"] == "reasoning"
        assert "id" in block


class TestContentBlockUnionType:
    """Verify that representative blocks of each type satisfy ContentBlock."""

    def _assert_content_block(self, block: ContentBlock) -> None:
        """Helper: simply ensures the block can be assigned to ContentBlock."""
        assert "type" in block

    def test_text_content_block(self) -> None:
        block: ContentBlock = {"type": "text", "text": "hi"}
        self._assert_content_block(block)

    def test_tool_call_content_block(self) -> None:
        block: ContentBlock = {
            "type": "tool_call",
            "id": "tc1",
            "name": "fn",
            "args": {},
        }
        self._assert_content_block(block)

    def test_invalid_tool_call_content_block(self) -> None:
        block: ContentBlock = {
            "type": "invalid_tool_call",
            "id": "itc1",
            "name": "fn",
            "args": "bad",
            "error": "parse error",
        }
        self._assert_content_block(block)

    def test_tool_call_chunk_content_block(self) -> None:
        block: ContentBlock = {
            "type": "tool_call_chunk",
            "id": "tcc1",
            "name": "fn",
            "args": '{"a":1}',
        }
        self._assert_content_block(block)

    def test_reasoning_content_block(self) -> None:
        block: ContentBlock = {"type": "reasoning", "reasoning": "thinking"}
        self._assert_content_block(block)

    def test_image_content_block(self) -> None:
        block: ContentBlock = {
            "type": "image",
            "url": "https://example.com/img.png",
        }
        self._assert_content_block(block)

    def test_video_content_block(self) -> None:
        block: ContentBlock = {
            "type": "video",
            "url": "https://example.com/vid.mp4",
        }
        self._assert_content_block(block)

    def test_audio_content_block(self) -> None:
        block: ContentBlock = {
            "type": "audio",
            "url": "https://example.com/aud.mp3",
        }
        self._assert_content_block(block)

    def test_plaintext_content_block(self) -> None:
        block: ContentBlock = {
            "type": "text-plain",
            "mime_type": "text/plain",
            "text": "hello",
        }
        self._assert_content_block(block)

    def test_file_content_block(self) -> None:
        block: ContentBlock = {
            "type": "file",
            "url": "https://example.com/doc.pdf",
        }
        self._assert_content_block(block)

    def test_server_tool_call_content_block(self) -> None:
        block: ContentBlock = {
            "type": "server_tool_call",
            "id": "stc1",
            "name": "search",
            "args": {},
        }
        self._assert_content_block(block)

    def test_server_tool_call_chunk_content_block(self) -> None:
        block: ContentBlock = {"type": "server_tool_call_chunk"}
        self._assert_content_block(block)

    def test_server_tool_result_content_block(self) -> None:
        block: ContentBlock = {
            "type": "server_tool_result",
            "tool_call_id": "stc1",
            "status": "success",
        }
        self._assert_content_block(block)

    def test_non_standard_content_block(self) -> None:
        block: ContentBlock = {
            "type": "non_standard",
            "value": {"data": "custom"},
        }
        self._assert_content_block(block)


class TestDataContentBlockUnionType:
    """Verify DataContentBlock includes exactly image, video, audio, text-plain, file."""

    def _assert_data_block(self, block: DataContentBlock) -> None:
        """Helper: block can be assigned to DataContentBlock."""
        assert "type" in block

    def test_image_in_data_content_block(self) -> None:
        block: DataContentBlock = {
            "type": "image",
            "url": "https://example.com/img.png",
        }
        self._assert_data_block(block)

    def test_video_in_data_content_block(self) -> None:
        block: DataContentBlock = {
            "type": "video",
            "url": "https://example.com/vid.mp4",
        }
        self._assert_data_block(block)

    def test_audio_in_data_content_block(self) -> None:
        block: DataContentBlock = {
            "type": "audio",
            "url": "https://example.com/aud.mp3",
        }
        self._assert_data_block(block)

    def test_plaintext_in_data_content_block(self) -> None:
        block: DataContentBlock = {
            "type": "text-plain",
            "mime_type": "text/plain",
            "text": "hi",
        }
        self._assert_data_block(block)

    def test_file_in_data_content_block(self) -> None:
        block: DataContentBlock = {
            "type": "file",
            "url": "https://example.com/doc.pdf",
        }
        self._assert_data_block(block)

    def test_data_content_block_union_has_exactly_five_members(self) -> None:
        """DataContentBlock union must consist of exactly 5 member types."""
        from typing import get_args as _get_args

        members = _get_args(DataContentBlock)
        assert len(members) == 5
        member_names = {m.__name__ for m in members}
        assert member_names == {
            "ImageContentBlock",
            "VideoContentBlock",
            "AudioContentBlock",
            "PlainTextContentBlock",
            "FileContentBlock",
        }
