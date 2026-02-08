"""Unit tests for Generation and GenerationChunk classes."""

import pytest

from langchain_core.outputs import Generation, GenerationChunk


class TestGeneration:
    """Test suite for Generation class."""

    def test_creation_with_text_only(self) -> None:
        """Test creating a Generation with only text."""
        gen = Generation(text="Hello, world!")
        assert gen.text == "Hello, world!"
        assert gen.generation_info is None
        assert gen.type == "Generation"

    def test_creation_with_generation_info(self) -> None:
        """Test creating a Generation with generation_info."""
        gen_info = {"finish_reason": "stop", "logprobs": None}
        gen = Generation(text="Test output", generation_info=gen_info)
        assert gen.text == "Test output"
        assert gen.generation_info == gen_info
        assert gen.type == "Generation"

    def test_creation_with_empty_text(self) -> None:
        """Test creating a Generation with empty text."""
        gen = Generation(text="")
        assert gen.text == ""
        assert gen.generation_info is None

    def test_is_lc_serializable(self) -> None:
        """Test that Generation is marked as serializable."""
        assert Generation.is_lc_serializable() is True

    def test_get_lc_namespace(self) -> None:
        """Test that Generation returns correct namespace."""
        expected_namespace = ["langchain", "schema", "output"]
        assert Generation.get_lc_namespace() == expected_namespace

    def test_type_field_is_literal(self) -> None:
        """Test that type field is set correctly."""
        gen = Generation(text="test")
        assert gen.type == "Generation"


class TestGenerationChunk:
    """Test suite for GenerationChunk class."""

    def test_creation(self) -> None:
        """Test creating a GenerationChunk."""
        chunk = GenerationChunk(text="chunk")
        assert chunk.text == "chunk"
        assert chunk.generation_info is None

    def test_add_two_chunks(self) -> None:
        """Test concatenating two GenerationChunks."""
        chunk1 = GenerationChunk(text="Hello, ")
        chunk2 = GenerationChunk(text="world!")
        result = chunk1 + chunk2
        assert isinstance(result, GenerationChunk)
        assert result.text == "Hello, world!"
        assert result.generation_info is None

    def test_add_chunks_with_generation_info(self) -> None:
        """Test concatenating chunks with generation_info."""
        chunk1 = GenerationChunk(
            text="Hello",
            generation_info={"key1": "value1", "shared": "first"},
        )
        chunk2 = GenerationChunk(
            text=" world",
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
        chunk1 = GenerationChunk(text="Hello", generation_info={"key": "value"})
        chunk2 = GenerationChunk(text=" world", generation_info=None)
        result = chunk1 + chunk2
        assert result.text == "Hello world"
        assert result.generation_info == {"key": "value"}

    def test_add_chunks_both_none_generation_info(self) -> None:
        """Test concatenating chunks where both have None generation_info."""
        chunk1 = GenerationChunk(text="Hello", generation_info=None)
        chunk2 = GenerationChunk(text=" world", generation_info=None)
        result = chunk1 + chunk2
        assert result.text == "Hello world"
        assert result.generation_info is None

    def test_add_empty_chunks(self) -> None:
        """Test concatenating empty chunks."""
        chunk1 = GenerationChunk(text="")
        chunk2 = GenerationChunk(text="")
        result = chunk1 + chunk2
        assert result.text == ""

    def test_add_multiple_chunks_sequentially(self) -> None:
        """Test concatenating multiple chunks in sequence."""
        chunk1 = GenerationChunk(text="A")
        chunk2 = GenerationChunk(text="B")
        chunk3 = GenerationChunk(text="C")
        result = chunk1 + chunk2 + chunk3
        assert result.text == "ABC"

    def test_add_with_invalid_type_raises_error(self) -> None:
        """Test that adding non-GenerationChunk raises TypeError."""
        chunk = GenerationChunk(text="test")
        with pytest.raises(TypeError) as exc_info:
            _ = chunk + "invalid"  # type: ignore[operator]
        assert "unsupported operand type(s) for +" in str(exc_info.value)

    def test_add_with_generation_raises_error(self) -> None:
        """Test that adding Generation (not chunk) raises TypeError."""
        chunk = GenerationChunk(text="test")
        gen = Generation(text="other")
        with pytest.raises(TypeError) as exc_info:
            _ = chunk + gen  # type: ignore[operator]
        assert "unsupported operand type(s) for +" in str(exc_info.value)

    def test_inherits_from_generation(self) -> None:
        """Test that GenerationChunk inherits from Generation."""
        chunk = GenerationChunk(text="test")
        assert isinstance(chunk, Generation)
        assert isinstance(chunk, GenerationChunk)

    def test_is_lc_serializable_inherited(self) -> None:
        """Test that GenerationChunk inherits serializable property."""
        assert GenerationChunk.is_lc_serializable() is True

    def test_get_lc_namespace_inherited(self) -> None:
        """Test that GenerationChunk inherits namespace."""
        expected_namespace = ["langchain", "schema", "output"]
        assert GenerationChunk.get_lc_namespace() == expected_namespace

    def test_type_field_is_generation(self) -> None:
        """Test that GenerationChunk type field stays 'Generation' (inherited)."""
        chunk = GenerationChunk(text="test")
        assert chunk.type == "Generation"


class TestGenerationSerialization:
    """Test suite for Generation serialization roundtrips."""

    def test_model_dump_basic(self) -> None:
        """Test model_dump produces correct dict structure."""
        gen = Generation(text="Hello", generation_info={"reason": "stop"})
        data = gen.model_dump()
        assert data["text"] == "Hello"
        assert data["generation_info"] == {"reason": "stop"}
        assert data["type"] == "Generation"

    def test_model_dump_none_generation_info(self) -> None:
        """Test model_dump with None generation_info."""
        gen = Generation(text="Hello")
        data = gen.model_dump()
        assert data["generation_info"] is None

    def test_model_validate_roundtrip(self) -> None:
        """Test serialization roundtrip via model_dump/model_validate."""
        gen = Generation(text="test output", generation_info={"logprobs": [0.1, 0.2]})
        data = gen.model_dump()
        restored = Generation.model_validate(data)
        assert restored.text == gen.text
        assert restored.generation_info == gen.generation_info
        assert restored.type == gen.type

    def test_json_roundtrip(self) -> None:
        """Test JSON serialization roundtrip."""
        gen = Generation(
            text="json test",
            generation_info={"finish_reason": "stop", "index": 0},
        )
        json_str = gen.model_dump_json()
        restored = Generation.model_validate_json(json_str)
        assert restored.text == gen.text
        assert restored.generation_info == gen.generation_info
        assert restored.type == gen.type

    def test_generation_chunk_model_dump(self) -> None:
        """Test GenerationChunk model_dump produces correct dict."""
        chunk = GenerationChunk(text="chunk", generation_info={"key": "val"})
        data = chunk.model_dump()
        assert data["text"] == "chunk"
        assert data["generation_info"] == {"key": "val"}

    def test_generation_chunk_json_roundtrip(self) -> None:
        """Test GenerationChunk JSON roundtrip."""
        chunk = GenerationChunk(text="json chunk", generation_info={"a": 1})
        json_str = chunk.model_dump_json()
        restored = GenerationChunk.model_validate_json(json_str)
        assert restored.text == chunk.text
        assert restored.generation_info == chunk.generation_info


class TestGenerationChunkMerging:
    """Test suite for GenerationChunk merging edge cases."""

    def test_merge_generation_info_with_nested_dicts(self) -> None:
        """Test merging generation_info containing nested dictionaries."""
        chunk1 = GenerationChunk(
            text="A",
            generation_info={"nested": {"key1": "val1"}},
        )
        chunk2 = GenerationChunk(
            text="B",
            generation_info={"nested": {"key2": "val2"}},
        )
        result = chunk1 + chunk2
        assert result.generation_info is not None
        assert result.generation_info["nested"]["key1"] == "val1"
        assert result.generation_info["nested"]["key2"] == "val2"

    def test_merge_generation_info_with_list_values(self) -> None:
        """Test merging generation_info containing list values."""
        chunk1 = GenerationChunk(
            text="A",
            generation_info={"items": [1, 2]},
        )
        chunk2 = GenerationChunk(
            text="B",
            generation_info={"items": [3, 4]},
        )
        result = chunk1 + chunk2
        assert result.generation_info is not None
        assert result.generation_info["items"] == [1, 2, 3, 4]

    def test_merge_generation_info_with_int_values(self) -> None:
        """Test merging generation_info containing integer values."""
        chunk1 = GenerationChunk(
            text="A",
            generation_info={"count": 5},
        )
        chunk2 = GenerationChunk(
            text="B",
            generation_info={"count": 3},
        )
        result = chunk1 + chunk2
        assert result.generation_info is not None
        assert result.generation_info["count"] == 8

    def test_add_preserves_generation_chunk_type(self) -> None:
        """Test that addition always returns GenerationChunk, not Generation."""
        chunk1 = GenerationChunk(text="A")
        chunk2 = GenerationChunk(text="B")
        result = chunk1 + chunk2
        assert type(result) is GenerationChunk

    def test_add_with_none_type_raises_error(self) -> None:
        """Test that adding None raises TypeError."""
        chunk = GenerationChunk(text="test")
        with pytest.raises(TypeError):
            _ = chunk + None  # type: ignore[operator]

    def test_add_with_int_raises_error(self) -> None:
        """Test that adding int raises TypeError."""
        chunk = GenerationChunk(text="test")
        with pytest.raises(TypeError):
            _ = chunk + 42  # type: ignore[operator]

    def test_sequential_add_accumulates_generation_info(self) -> None:
        """Test that sequential adds properly accumulate generation_info."""
        c1 = GenerationChunk(text="A", generation_info={"k1": "v1"})
        c2 = GenerationChunk(text="B", generation_info={"k2": "v2"})
        c3 = GenerationChunk(text="C", generation_info={"k3": "v3"})
        result = c1 + c2 + c3
        assert result.text == "ABC"
        assert result.generation_info is not None
        assert result.generation_info["k1"] == "v1"
        assert result.generation_info["k2"] == "v2"
        assert result.generation_info["k3"] == "v3"

    def test_add_first_has_none_second_has_info(self) -> None:
        """Test adding when first chunk has None generation_info."""
        chunk1 = GenerationChunk(text="A", generation_info=None)
        chunk2 = GenerationChunk(text="B", generation_info={"key": "value"})
        result = chunk1 + chunk2
        assert result.generation_info == {"key": "value"}
