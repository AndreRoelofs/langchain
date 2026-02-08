"""Unit tests for ChatResult class."""

import pytest
from pydantic import BaseModel

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class TestChatResult:
    """Test suite for ChatResult class."""

    def test_creation_with_single_generation(self) -> None:
        """Test creating ChatResult with a single generation."""
        gen = ChatGeneration(message=AIMessage(content="Hello"))
        result = ChatResult(generations=[gen])
        assert len(result.generations) == 1
        assert result.generations[0] == gen
        assert result.llm_output is None

    def test_creation_with_multiple_generations(self) -> None:
        """Test creating ChatResult with multiple generations."""
        gen1 = ChatGeneration(message=AIMessage(content="Response 1"))
        gen2 = ChatGeneration(message=AIMessage(content="Response 2"))
        gen3 = ChatGeneration(message=AIMessage(content="Response 3"))
        result = ChatResult(generations=[gen1, gen2, gen3])
        assert len(result.generations) == 3
        assert result.generations[0] == gen1
        assert result.generations[1] == gen2
        assert result.generations[2] == gen3

    def test_creation_with_llm_output(self) -> None:
        """Test creating ChatResult with llm_output."""
        gen = ChatGeneration(message=AIMessage(content="Test"))
        llm_output = {
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "model_name": "gpt-4",
        }
        result = ChatResult(generations=[gen], llm_output=llm_output)
        assert result.llm_output == llm_output
        assert result.llm_output["token_usage"]["prompt_tokens"] == 10
        assert result.llm_output["model_name"] == "gpt-4"

    def test_creation_with_empty_llm_output(self) -> None:
        """Test creating ChatResult with empty llm_output dict."""
        gen = ChatGeneration(message=AIMessage(content="Test"))
        result = ChatResult(generations=[gen], llm_output={})
        assert result.llm_output == {}

    def test_llm_output_defaults_to_none(self) -> None:
        """Test that llm_output defaults to None when not provided."""
        gen = ChatGeneration(message=AIMessage(content="Test"))
        result = ChatResult(generations=[gen])
        assert result.llm_output is None

    def test_generations_with_generation_info(self) -> None:
        """Test ChatResult with generations that have generation_info."""
        gen1 = ChatGeneration(
            message=AIMessage(content="Response 1"),
            generation_info={"finish_reason": "stop"},
        )
        gen2 = ChatGeneration(
            message=AIMessage(content="Response 2"),
            generation_info={"finish_reason": "length"},
        )
        result = ChatResult(generations=[gen1, gen2])
        assert result.generations[0].generation_info["finish_reason"] == "stop"
        assert result.generations[1].generation_info["finish_reason"] == "length"

    def test_empty_generations_list(self) -> None:
        """Test creating ChatResult with empty generations list."""
        result = ChatResult(generations=[])
        assert len(result.generations) == 0
        assert result.llm_output is None

    def test_generations_preserve_message_attributes(self) -> None:
        """Test that message attributes are preserved in generations."""
        msg = AIMessage(
            content="Test response",
            additional_kwargs={"function_call": {"name": "test"}},
        )
        gen = ChatGeneration(message=msg)
        result = ChatResult(generations=[gen])
        assert result.generations[0].message.content == "Test response"
        assert result.generations[0].message.additional_kwargs == {
            "function_call": {"name": "test"}
        }

    def test_llm_output_with_various_types(self) -> None:
        """Test llm_output can contain various data types."""
        gen = ChatGeneration(message=AIMessage(content="Test"))
        llm_output = {
            "string_field": "value",
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": True,
            "list_field": [1, 2, 3],
            "nested_dict": {"key": "value"},
        }
        result = ChatResult(generations=[gen], llm_output=llm_output)
        assert result.llm_output == llm_output
        assert result.llm_output["string_field"] == "value"
        assert result.llm_output["int_field"] == 42
        assert result.llm_output["nested_dict"]["key"] == "value"

    def test_multiple_candidate_generations(self) -> None:
        """Test ChatResult with multiple candidate generations for same prompt."""
        # Simulates n>1 parameter in API calls
        candidates = [
            ChatGeneration(message=AIMessage(content="Candidate 1")),
            ChatGeneration(message=AIMessage(content="Candidate 2")),
            ChatGeneration(message=AIMessage(content="Candidate 3")),
        ]
        result = ChatResult(generations=candidates)
        assert len(result.generations) == 3
        for i, gen in enumerate(result.generations, 1):
            assert gen.text == f"Candidate {i}"


class TestChatResultSerialization:
    """Test suite for ChatResult serialization roundtrips."""

    def test_model_dump_basic(self) -> None:
        """Test model_dump for ChatResult."""
        gen = ChatGeneration(message=AIMessage(content="Hello"))
        result = ChatResult(generations=[gen])
        data = result.model_dump()
        assert "generations" in data
        assert len(data["generations"]) == 1
        assert data["llm_output"] is None

    def test_model_dump_with_llm_output(self) -> None:
        """Test model_dump includes llm_output."""
        gen = ChatGeneration(message=AIMessage(content="Hello"))
        result = ChatResult(
            generations=[gen],
            llm_output={"model": "gpt-4", "token_usage": {"total": 50}},
        )
        data = result.model_dump()
        assert data["llm_output"]["model"] == "gpt-4"
        assert data["llm_output"]["token_usage"]["total"] == 50

    def test_json_roundtrip(self) -> None:
        """Test JSON serialization roundtrip."""
        gen = ChatGeneration(
            message=AIMessage(content="test"),
            generation_info={"finish_reason": "stop"},
        )
        result = ChatResult(
            generations=[gen],
            llm_output={"model": "gpt-4"},
        )
        json_str = result.model_dump_json()
        restored = ChatResult.model_validate_json(json_str)
        assert len(restored.generations) == 1
        assert restored.generations[0].text == "test"
        assert restored.llm_output == {"model": "gpt-4"}

    def test_model_validate_from_dict(self) -> None:
        """Test model_validate from dict."""
        gen = ChatGeneration(message=AIMessage(content="test"))
        result = ChatResult(generations=[gen], llm_output={"key": "val"})
        data = result.model_dump()
        restored = ChatResult.model_validate(data)
        assert len(restored.generations) == len(result.generations)
        assert restored.llm_output == result.llm_output


class TestChatResultEquality:
    """Test suite for ChatResult equality semantics."""

    def test_equality_same_content(self) -> None:
        """Test equality for ChatResults with same content."""
        gen = ChatGeneration(message=AIMessage(content="Hello"))
        result1 = ChatResult(generations=[gen], llm_output={"model": "gpt-4"})
        result2 = ChatResult(generations=[gen], llm_output={"model": "gpt-4"})
        assert result1 == result2

    def test_inequality_different_generations(self) -> None:
        """Test inequality for ChatResults with different generations."""
        gen1 = ChatGeneration(message=AIMessage(content="Hello"))
        gen2 = ChatGeneration(message=AIMessage(content="Goodbye"))
        result1 = ChatResult(generations=[gen1])
        result2 = ChatResult(generations=[gen2])
        assert result1 != result2

    def test_inequality_different_llm_output(self) -> None:
        """Test inequality for ChatResults with different llm_output."""
        gen = ChatGeneration(message=AIMessage(content="Hello"))
        result1 = ChatResult(generations=[gen], llm_output={"model": "gpt-4"})
        result2 = ChatResult(generations=[gen], llm_output={"model": "gpt-3.5"})
        assert result1 != result2

    def test_equality_both_none_llm_output(self) -> None:
        """Test equality when both have None llm_output."""
        gen = ChatGeneration(message=AIMessage(content="Hello"))
        result1 = ChatResult(generations=[gen])
        result2 = ChatResult(generations=[gen])
        assert result1 == result2


class TestChatResultModelBehavior:
    """Test suite for ChatResult Pydantic model behavior."""

    def test_is_pydantic_model(self) -> None:
        """Test that ChatResult is a Pydantic BaseModel."""
        gen = ChatGeneration(message=AIMessage(content="test"))
        result = ChatResult(generations=[gen])
        assert isinstance(result, BaseModel)

    def test_with_chat_generation_chunk(self) -> None:
        """Test ChatResult with ChatGenerationChunk objects."""
        chunk = ChatGenerationChunk(message=AIMessageChunk(content="chunk"))
        result = ChatResult(generations=[chunk])
        assert len(result.generations) == 1
        assert isinstance(result.generations[0], ChatGenerationChunk)
        assert result.generations[0].text == "chunk"

    def test_generations_ordering_preserved(self) -> None:
        """Test that generation ordering is preserved."""
        gens = [
            ChatGeneration(message=AIMessage(content=f"Response {i}")) for i in range(5)
        ]
        result = ChatResult(generations=gens)
        for i, gen in enumerate(result.generations):
            assert gen.text == f"Response {i}"

    def test_generations_with_mixed_content_types(self) -> None:
        """Test ChatResult with generations having different content types."""
        gen_str = ChatGeneration(message=AIMessage(content="string content"))
        gen_list = ChatGeneration(
            message=AIMessage(content=[{"text": "list content", "type": "text"}])
        )
        result = ChatResult(generations=[gen_str, gen_list])
        assert result.generations[0].text == "string content"
        assert result.generations[1].text == "list content"
