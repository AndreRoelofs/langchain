"""Unit tests for ChatResult class."""

import pytest

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


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
