"""Snapshot tests for UsageMetadataCallbackHandler and get_usage_metadata_callback.

These tests capture the behavior of the usage tracking callback handler
including thread safety, accumulation, edge cases, and the context manager.
"""

import threading
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

from langchain_core.callbacks.usage import (
    UsageMetadataCallbackHandler,
    get_usage_metadata_callback,
)
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
    add_usage,
)
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult


def _make_llm_result(
    content: str,
    usage: UsageMetadata,
    model_name: str,
) -> LLMResult:
    """Create an LLMResult with an AIMessage carrying usage metadata."""
    msg = AIMessage(
        content=content,
        usage_metadata=usage,
        response_metadata={"model_name": model_name},
    )
    gen = ChatGeneration(message=msg)
    return LLMResult(generations=[[gen]])


class TestUsageMetadataCallbackHandlerInit:
    """Tests for handler initialization."""

    def test_empty_usage_metadata_on_init(self) -> None:
        handler = UsageMetadataCallbackHandler()
        assert handler.usage_metadata == {}

    def test_has_lock(self) -> None:
        handler = UsageMetadataCallbackHandler()
        assert hasattr(handler, "_lock")
        assert isinstance(handler._lock, type(threading.Lock()))

    def test_repr_empty(self) -> None:
        handler = UsageMetadataCallbackHandler()
        assert repr(handler) == "{}"


class TestOnLLMEnd:
    """Tests for on_llm_end collecting usage metadata."""

    def test_collects_single_response(self) -> None:
        usage = UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)
        result = _make_llm_result("hi", usage, "model-a")
        handler = UsageMetadataCallbackHandler()
        handler.on_llm_end(result, run_id=uuid4())
        assert handler.usage_metadata == {"model-a": usage}

    def test_accumulates_multiple_responses_same_model(self) -> None:
        u1 = UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)
        u2 = UsageMetadata(input_tokens=20, output_tokens=10, total_tokens=30)
        handler = UsageMetadataCallbackHandler()
        handler.on_llm_end(_make_llm_result("a", u1, "model-a"), run_id=uuid4())
        handler.on_llm_end(_make_llm_result("b", u2, "model-a"), run_id=uuid4())
        expected = add_usage(u1, u2)
        assert handler.usage_metadata["model-a"] == expected

    def test_tracks_multiple_models(self) -> None:
        u1 = UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)
        u2 = UsageMetadata(input_tokens=20, output_tokens=10, total_tokens=30)
        handler = UsageMetadataCallbackHandler()
        handler.on_llm_end(_make_llm_result("a", u1, "model-a"), run_id=uuid4())
        handler.on_llm_end(_make_llm_result("b", u2, "model-b"), run_id=uuid4())
        assert handler.usage_metadata == {"model-a": u1, "model-b": u2}

    def test_with_token_details(self) -> None:
        usage = UsageMetadata(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_token_details=InputTokenDetails(audio=3, cache_read=2),
            output_token_details=OutputTokenDetails(reasoning=4),
        )
        handler = UsageMetadataCallbackHandler()
        handler.on_llm_end(_make_llm_result("a", usage, "m"), run_id=uuid4())
        stored = handler.usage_metadata["m"]
        assert stored["input_token_details"]["audio"] == 3
        assert stored["input_token_details"]["cache_read"] == 2
        assert stored["output_token_details"]["reasoning"] == 4

    def test_empty_generations_ignored(self) -> None:
        result = LLMResult(generations=[[]])
        handler = UsageMetadataCallbackHandler()
        handler.on_llm_end(result, run_id=uuid4())
        assert handler.usage_metadata == {}

    def test_no_generations_at_all(self) -> None:
        result = LLMResult(generations=[])
        handler = UsageMetadataCallbackHandler()
        handler.on_llm_end(result, run_id=uuid4())
        assert handler.usage_metadata == {}

    def test_non_chat_generation_ignored(self) -> None:
        """Non-ChatGeneration results should not be tracked."""
        from langchain_core.outputs import Generation

        gen = Generation(text="hello")
        result = LLMResult(generations=[[gen]])
        handler = UsageMetadataCallbackHandler()
        handler.on_llm_end(result, run_id=uuid4())
        assert handler.usage_metadata == {}

    def test_missing_model_name_ignored(self) -> None:
        """If model_name is not in response_metadata, usage is not tracked."""
        msg = AIMessage(
            content="hi",
            usage_metadata=UsageMetadata(
                input_tokens=1, output_tokens=1, total_tokens=2
            ),
            response_metadata={},  # no model_name
        )
        gen = ChatGeneration(message=msg)
        result = LLMResult(generations=[[gen]])
        handler = UsageMetadataCallbackHandler()
        handler.on_llm_end(result, run_id=uuid4())
        assert handler.usage_metadata == {}

    def test_missing_usage_metadata_ignored(self) -> None:
        """If AIMessage has no usage_metadata, it is not tracked."""
        msg = AIMessage(
            content="hi",
            response_metadata={"model_name": "m"},
        )
        gen = ChatGeneration(message=msg)
        result = LLMResult(generations=[[gen]])
        handler = UsageMetadataCallbackHandler()
        handler.on_llm_end(result, run_id=uuid4())
        assert handler.usage_metadata == {}


class TestRepr:
    """Tests for __repr__."""

    def test_repr_with_data(self) -> None:
        usage = UsageMetadata(input_tokens=1, output_tokens=2, total_tokens=3)
        handler = UsageMetadataCallbackHandler()
        handler.on_llm_end(_make_llm_result("a", usage, "m"), run_id=uuid4())
        r = repr(handler)
        assert "m" in r
        assert "input_tokens" in r


class TestThreadSafety:
    """Tests for thread-safe accumulation."""

    def test_concurrent_on_llm_end_calls(self) -> None:
        handler = UsageMetadataCallbackHandler()
        usage = UsageMetadata(input_tokens=1, output_tokens=1, total_tokens=2)
        num_threads = 20

        def worker() -> None:
            for _ in range(50):
                handler.on_llm_end(_make_llm_result("x", usage, "m"), run_id=uuid4())

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stored = handler.usage_metadata["m"]
        expected_count = num_threads * 50
        assert stored["input_tokens"] == expected_count
        assert stored["output_tokens"] == expected_count
        assert stored["total_tokens"] == expected_count * 2


class TestGetUsageMetadataCallback:
    """Tests for the get_usage_metadata_callback context manager."""

    def test_yields_handler(self) -> None:
        with get_usage_metadata_callback() as cb:
            assert isinstance(cb, UsageMetadataCallbackHandler)

    def test_handler_starts_empty(self) -> None:
        with get_usage_metadata_callback() as cb:
            assert cb.usage_metadata == {}

    def test_custom_name(self) -> None:
        with get_usage_metadata_callback(name="custom") as cb:
            assert isinstance(cb, UsageMetadataCallbackHandler)

    def test_multiple_context_managers_independent(self) -> None:
        """Two context managers should yield independent handlers."""
        with get_usage_metadata_callback() as cb1:
            with get_usage_metadata_callback() as cb2:
                assert cb1 is not cb2
