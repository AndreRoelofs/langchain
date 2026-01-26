"""Unit tests for LLMResult class."""

from uuid import uuid4

import pytest

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    Generation,
    GenerationChunk,
    LLMResult,
    RunInfo,
)


class TestLLMResult:
    """Test suite for LLMResult class."""

    def test_creation_with_single_prompt_single_generation(self) -> None:
        """Test creating LLMResult with single prompt and single generation."""
        gen = Generation(text="Response")
        result = LLMResult(generations=[[gen]])
        assert len(result.generations) == 1
        assert len(result.generations[0]) == 1
        assert result.generations[0][0] == gen
        assert result.llm_output is None
        assert result.run is None
        assert result.type == "LLMResult"

    def test_creation_with_multiple_prompts(self) -> None:
        """Test creating LLMResult with multiple prompts."""
        gen1 = Generation(text="Response 1")
        gen2 = Generation(text="Response 2")
        gen3 = Generation(text="Response 3")
        result = LLMResult(generations=[[gen1], [gen2], [gen3]])
        assert len(result.generations) == 3
        assert result.generations[0][0].text == "Response 1"
        assert result.generations[1][0].text == "Response 2"
        assert result.generations[2][0].text == "Response 3"

    def test_creation_with_multiple_candidates(self) -> None:
        """Test creating LLMResult with multiple candidate generations per prompt."""
        gen1 = Generation(text="Candidate 1")
        gen2 = Generation(text="Candidate 2")
        gen3 = Generation(text="Candidate 3")
        result = LLMResult(generations=[[gen1, gen2, gen3]])
        assert len(result.generations) == 1
        assert len(result.generations[0]) == 3
        assert result.generations[0][0].text == "Candidate 1"
        assert result.generations[0][1].text == "Candidate 2"
        assert result.generations[0][2].text == "Candidate 3"

    def test_creation_with_chat_generations(self) -> None:
        """Test creating LLMResult with ChatGeneration objects."""
        gen1 = ChatGeneration(message=AIMessage(content="Chat response 1"))
        gen2 = ChatGeneration(message=AIMessage(content="Chat response 2"))
        result = LLMResult(generations=[[gen1], [gen2]])
        assert len(result.generations) == 2
        assert isinstance(result.generations[0][0], ChatGeneration)
        assert result.generations[0][0].text == "Chat response 1"

    def test_creation_with_generation_chunks(self) -> None:
        """Test creating LLMResult with GenerationChunk objects."""
        chunk1 = GenerationChunk(text="Chunk 1")
        chunk2 = ChatGenerationChunk(message=AIMessageChunk(content="Chunk 2"))
        result = LLMResult(generations=[[chunk1], [chunk2]])
        assert len(result.generations) == 2
        assert isinstance(result.generations[0][0], GenerationChunk)
        assert isinstance(result.generations[1][0], ChatGenerationChunk)

    def test_creation_with_llm_output(self) -> None:
        """Test creating LLMResult with llm_output."""
        gen = Generation(text="Response")
        llm_output = {
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "model_name": "gpt-4",
        }
        result = LLMResult(generations=[[gen]], llm_output=llm_output)
        assert result.llm_output == llm_output
        assert result.llm_output["token_usage"]["prompt_tokens"] == 10

    def test_creation_with_run_info(self) -> None:
        """Test creating LLMResult with run info."""
        gen = Generation(text="Response")
        run_id = uuid4()
        run_info = RunInfo(run_id=run_id)
        result = LLMResult(generations=[[gen]], run=[run_info])
        assert result.run is not None
        assert len(result.run) == 1
        assert result.run[0].run_id == run_id

    def test_creation_with_multiple_run_infos(self) -> None:
        """Test creating LLMResult with multiple run infos."""
        gen1 = Generation(text="Response 1")
        gen2 = Generation(text="Response 2")
        run_id1 = uuid4()
        run_id2 = uuid4()
        run_info1 = RunInfo(run_id=run_id1)
        run_info2 = RunInfo(run_id=run_id2)
        result = LLMResult(generations=[[gen1], [gen2]], run=[run_info1, run_info2])
        assert result.run is not None
        assert len(result.run) == 2
        assert result.run[0].run_id == run_id1
        assert result.run[1].run_id == run_id2

    def test_flatten_single_prompt_single_generation(self) -> None:
        """Test flattening LLMResult with single prompt and generation."""
        gen = Generation(text="Response")
        result = LLMResult(generations=[[gen]])
        flattened = result.flatten()
        assert len(flattened) == 1
        assert flattened[0].generations == [[gen]]

    def test_flatten_multiple_prompts(self) -> None:
        """Test flattening LLMResult with multiple prompts."""
        gen1 = Generation(text="Response 1")
        gen2 = Generation(text="Response 2")
        gen3 = Generation(text="Response 3")
        result = LLMResult(generations=[[gen1], [gen2], [gen3]])
        flattened = result.flatten()
        assert len(flattened) == 3
        assert flattened[0].generations == [[gen1]]
        assert flattened[1].generations == [[gen2]]
        assert flattened[2].generations == [[gen3]]

    def test_flatten_preserves_llm_output_for_first(self) -> None:
        """Test that flatten preserves llm_output for first result."""
        gen1 = Generation(text="Response 1")
        gen2 = Generation(text="Response 2")
        llm_output = {"token_usage": {"total": 100}, "model": "gpt-4"}
        result = LLMResult(generations=[[gen1], [gen2]], llm_output=llm_output)
        flattened = result.flatten()
        assert flattened[0].llm_output == llm_output
        assert flattened[0].llm_output["token_usage"]["total"] == 100

    def test_flatten_clears_token_usage_for_subsequent(self) -> None:
        """Test that flatten clears token_usage for subsequent results."""
        gen1 = Generation(text="Response 1")
        gen2 = Generation(text="Response 2")
        llm_output = {"token_usage": {"total": 100}, "model": "gpt-4"}
        result = LLMResult(generations=[[gen1], [gen2]], llm_output=llm_output)
        flattened = result.flatten()
        assert flattened[1].llm_output is not None
        assert flattened[1].llm_output["token_usage"] == {}
        assert flattened[1].llm_output["model"] == "gpt-4"

    def test_flatten_handles_none_llm_output(self) -> None:
        """Test that flatten handles None llm_output correctly."""
        gen1 = Generation(text="Response 1")
        gen2 = Generation(text="Response 2")
        result = LLMResult(generations=[[gen1], [gen2]], llm_output=None)
        flattened = result.flatten()
        assert flattened[0].llm_output is None
        assert flattened[1].llm_output is None

    def test_flatten_with_multiple_candidates(self) -> None:
        """Test flattening with multiple candidate generations."""
        gen1 = Generation(text="Candidate 1")
        gen2 = Generation(text="Candidate 2")
        result = LLMResult(generations=[[gen1, gen2]])
        flattened = result.flatten()
        assert len(flattened) == 1
        assert len(flattened[0].generations[0]) == 2

    def test_equality_same_generations_and_output(self) -> None:
        """Test equality for LLMResults with same generations and output."""
        gen = Generation(text="Response")
        llm_output = {"model": "gpt-4"}
        result1 = LLMResult(generations=[[gen]], llm_output=llm_output)
        result2 = LLMResult(generations=[[gen]], llm_output=llm_output)
        assert result1 == result2

    def test_equality_different_generations(self) -> None:
        """Test inequality for LLMResults with different generations."""
        gen1 = Generation(text="Response 1")
        gen2 = Generation(text="Response 2")
        result1 = LLMResult(generations=[[gen1]])
        result2 = LLMResult(generations=[[gen2]])
        assert result1 != result2

    def test_equality_different_llm_output(self) -> None:
        """Test inequality for LLMResults with different llm_output."""
        gen = Generation(text="Response")
        result1 = LLMResult(generations=[[gen]], llm_output={"model": "gpt-4"})
        result2 = LLMResult(generations=[[gen]], llm_output={"model": "gpt-3.5"})
        assert result1 != result2

    def test_equality_ignores_run_info(self) -> None:
        """Test that equality ignores run info."""
        gen = Generation(text="Response")
        run_id1 = uuid4()
        run_id2 = uuid4()
        result1 = LLMResult(generations=[[gen]], run=[RunInfo(run_id=run_id1)])
        result2 = LLMResult(generations=[[gen]], run=[RunInfo(run_id=run_id2)])
        # Should be equal despite different run IDs
        assert result1 == result2

    def test_equality_with_none_llm_output(self) -> None:
        """Test equality when llm_output is None."""
        gen = Generation(text="Response")
        result1 = LLMResult(generations=[[gen]], llm_output=None)
        result2 = LLMResult(generations=[[gen]], llm_output=None)
        assert result1 == result2

    def test_equality_with_non_llm_result(self) -> None:
        """Test equality comparison with non-LLMResult object."""
        gen = Generation(text="Response")
        result = LLMResult(generations=[[gen]])
        assert result != "not an LLMResult"
        assert result != 42
        assert result != None  # noqa: E711

    def test_hash_is_none(self) -> None:
        """Test that LLMResult is not hashable."""
        gen = Generation(text="Response")
        result = LLMResult(generations=[[gen]])
        assert result.__hash__ is None

    def test_type_field_is_literal(self) -> None:
        """Test that type field is set correctly."""
        gen = Generation(text="Response")
        result = LLMResult(generations=[[gen]])
        assert result.type == "LLMResult"

    def test_empty_generations(self) -> None:
        """Test creating LLMResult with empty generations."""
        result = LLMResult(generations=[])
        assert len(result.generations) == 0
        flattened = result.flatten()
        assert len(flattened) == 0

    def test_mixed_generation_types(self) -> None:
        """Test LLMResult with mixed generation types in same list."""
        gen = Generation(text="Regular")
        chat_gen = ChatGeneration(message=AIMessage(content="Chat"))
        result = LLMResult(generations=[[gen], [chat_gen]])
        assert len(result.generations) == 2
        assert isinstance(result.generations[0][0], Generation)
        assert isinstance(result.generations[1][0], ChatGeneration)

    def test_complex_llm_output_structure(self) -> None:
        """Test LLMResult with complex nested llm_output."""
        gen = Generation(text="Response")
        llm_output = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "model_name": "gpt-4",
            "system_fingerprint": "fp_123",
            "metadata": {"temperature": 0.7, "top_p": 1.0},
        }
        result = LLMResult(generations=[[gen]], llm_output=llm_output)
        assert result.llm_output == llm_output
        assert result.llm_output["metadata"]["temperature"] == 0.7
