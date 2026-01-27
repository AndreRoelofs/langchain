"""Tests for langchain_core.language_models.fake module (FakeListLLM, FakeStreamingListLLM)."""

import pytest

from langchain_core.language_models.fake import (
    FakeListLLM,
    FakeListLLMError,
    FakeStreamingListLLM,
)


class TestFakeListLLM:
    """Tests for FakeListLLM class."""

    def test_initialization(self) -> None:
        """Test FakeListLLM initialization."""
        llm = FakeListLLM(responses=["response1", "response2"])
        assert llm.responses == ["response1", "response2"]
        assert llm.i == 0
        assert llm.sleep is None

    def test_initialization_with_sleep(self) -> None:
        """Test FakeListLLM initialization with sleep parameter."""
        llm = FakeListLLM(responses=["response"], sleep=0.1)
        assert llm.sleep == 0.1

    def test_llm_type(self) -> None:
        """Test _llm_type property."""
        llm = FakeListLLM(responses=["response"])
        assert llm._llm_type == "fake-list"

    def test_invoke_single_response(self) -> None:
        """Test invoke with single response."""
        llm = FakeListLLM(responses=["hello"])
        result = llm.invoke("any prompt")
        assert result == "hello"

    def test_invoke_cycles_through_responses(self) -> None:
        """Test invoke cycles through responses."""
        llm = FakeListLLM(responses=["first", "second", "third"])

        assert llm.invoke("prompt1") == "first"
        assert llm.i == 1

        assert llm.invoke("prompt2") == "second"
        assert llm.i == 2

        assert llm.invoke("prompt3") == "third"
        # Should cycle back to 0
        assert llm.i == 0

        # Should start from beginning again
        assert llm.invoke("prompt4") == "first"

    def test_invoke_with_single_response_stays_at_same(self) -> None:
        """Test invoke with single response always returns same."""
        llm = FakeListLLM(responses=["only"])

        assert llm.invoke("prompt1") == "only"
        assert llm.invoke("prompt2") == "only"
        assert llm.invoke("prompt3") == "only"
        assert llm.i == 0

    async def test_ainvoke_single_response(self) -> None:
        """Test ainvoke with single response."""
        llm = FakeListLLM(responses=["async hello"])
        result = await llm.ainvoke("any prompt")
        assert result == "async hello"

    async def test_ainvoke_cycles_through_responses(self) -> None:
        """Test ainvoke cycles through responses."""
        llm = FakeListLLM(responses=["first", "second"])

        assert await llm.ainvoke("prompt1") == "first"
        assert await llm.ainvoke("prompt2") == "second"
        assert await llm.ainvoke("prompt3") == "first"

    def test_identifying_params(self) -> None:
        """Test _identifying_params property."""
        llm = FakeListLLM(responses=["a", "b", "c"])
        params = llm._identifying_params
        assert params == {"responses": ["a", "b", "c"]}

    def test_batch_processing(self) -> None:
        """Test batch processing."""
        llm = FakeListLLM(responses=["r1", "r2", "r3"])
        results = llm.batch(["p1", "p2", "p3"])
        assert results == ["r1", "r2", "r3"]

    async def test_abatch_processing(self) -> None:
        """Test async batch processing."""
        llm = FakeListLLM(responses=["r1", "r2", "r3"])
        results = await llm.abatch(["p1", "p2", "p3"])
        assert results == ["r1", "r2", "r3"]

    def test_generate_returns_llm_result(self) -> None:
        """Test generate returns LLMResult."""
        llm = FakeListLLM(responses=["response"])
        result = llm.generate(["prompt"])
        assert len(result.generations) == 1
        assert result.generations[0][0].text == "response"

    def test_call_method(self) -> None:
        """Test _call method directly."""
        llm = FakeListLLM(responses=["direct call"])
        result = llm._call("prompt")
        assert result == "direct call"

    async def test_acall_method(self) -> None:
        """Test _acall method directly."""
        llm = FakeListLLM(responses=["async direct call"])
        result = await llm._acall("prompt")
        assert result == "async direct call"


class TestFakeListLLMError:
    """Tests for FakeListLLMError exception."""

    def test_error_can_be_raised(self) -> None:
        """Test FakeListLLMError can be raised."""
        with pytest.raises(FakeListLLMError):
            raise FakeListLLMError("test error")

    def test_error_is_exception(self) -> None:
        """Test FakeListLLMError is an Exception."""
        error = FakeListLLMError("test")
        assert isinstance(error, Exception)


class TestFakeStreamingListLLM:
    """Tests for FakeStreamingListLLM class."""

    def test_initialization(self) -> None:
        """Test FakeStreamingListLLM initialization."""
        llm = FakeStreamingListLLM(responses=["response"])
        assert llm.responses == ["response"]
        assert llm.error_on_chunk_number is None

    def test_initialization_with_error_on_chunk(self) -> None:
        """Test FakeStreamingListLLM with error_on_chunk_number."""
        llm = FakeStreamingListLLM(responses=["response"], error_on_chunk_number=2)
        assert llm.error_on_chunk_number == 2

    def test_stream_yields_characters(self) -> None:
        """Test stream yields individual characters."""
        llm = FakeStreamingListLLM(responses=["hello"])
        chunks = list(llm.stream("prompt"))
        assert chunks == ["h", "e", "l", "l", "o"]

    def test_stream_cycles_through_responses(self) -> None:
        """Test stream cycles through responses."""
        llm = FakeStreamingListLLM(responses=["ab", "cd"])

        chunks1 = list(llm.stream("prompt1"))
        assert chunks1 == ["a", "b"]

        chunks2 = list(llm.stream("prompt2"))
        assert chunks2 == ["c", "d"]

        # Should cycle back
        chunks3 = list(llm.stream("prompt3"))
        assert chunks3 == ["a", "b"]

    def test_stream_with_sleep(self) -> None:
        """Test stream with sleep parameter."""
        import time

        llm = FakeStreamingListLLM(responses=["ab"], sleep=0.01)
        start = time.time()
        list(llm.stream("prompt"))
        elapsed = time.time() - start
        # Should have slept at least 0.02 seconds (2 chunks * 0.01)
        assert elapsed >= 0.02

    def test_stream_error_on_chunk(self) -> None:
        """Test stream raises error on specified chunk."""
        llm = FakeStreamingListLLM(responses=["hello"], error_on_chunk_number=2)
        stream = llm.stream("prompt")

        assert next(stream) == "h"
        assert next(stream) == "e"
        with pytest.raises(FakeListLLMError):
            next(stream)

    def test_stream_error_on_first_chunk(self) -> None:
        """Test stream raises error on first chunk."""
        llm = FakeStreamingListLLM(responses=["hello"], error_on_chunk_number=0)
        stream = llm.stream("prompt")

        with pytest.raises(FakeListLLMError):
            next(stream)

    async def test_astream_yields_characters(self) -> None:
        """Test astream yields individual characters."""
        llm = FakeStreamingListLLM(responses=["hello"])
        chunks = [chunk async for chunk in llm.astream("prompt")]
        assert chunks == ["h", "e", "l", "l", "o"]

    async def test_astream_cycles_through_responses(self) -> None:
        """Test astream cycles through responses."""
        llm = FakeStreamingListLLM(responses=["ab", "cd"])

        chunks1 = [chunk async for chunk in llm.astream("prompt1")]
        assert chunks1 == ["a", "b"]

        chunks2 = [chunk async for chunk in llm.astream("prompt2")]
        assert chunks2 == ["c", "d"]

    async def test_astream_with_sleep(self) -> None:
        """Test astream with sleep parameter."""
        import asyncio
        import time

        llm = FakeStreamingListLLM(responses=["ab"], sleep=0.01)
        start = time.time()
        chunks = [chunk async for chunk in llm.astream("prompt")]
        elapsed = time.time() - start
        # Should have slept at least 0.02 seconds (2 chunks * 0.01)
        assert elapsed >= 0.02
        assert chunks == ["a", "b"]

    async def test_astream_error_on_chunk(self) -> None:
        """Test astream raises error on specified chunk."""
        llm = FakeStreamingListLLM(responses=["hello"], error_on_chunk_number=2)

        chunks = []
        with pytest.raises(FakeListLLMError):
            async for chunk in llm.astream("prompt"):
                chunks.append(chunk)

        assert chunks == ["h", "e"]

    async def test_astream_error_on_first_chunk(self) -> None:
        """Test astream raises error on first chunk."""
        llm = FakeStreamingListLLM(responses=["hello"], error_on_chunk_number=0)

        with pytest.raises(FakeListLLMError):
            async for _ in llm.astream("prompt"):
                pass

    def test_invoke_returns_full_response(self) -> None:
        """Test invoke returns full response (not streamed)."""
        llm = FakeStreamingListLLM(responses=["hello world"])
        result = llm.invoke("prompt")
        assert result == "hello world"

    async def test_ainvoke_returns_full_response(self) -> None:
        """Test ainvoke returns full response (not streamed)."""
        llm = FakeStreamingListLLM(responses=["hello world"])
        result = await llm.ainvoke("prompt")
        assert result == "hello world"

    def test_inherits_from_fake_list_llm(self) -> None:
        """Test FakeStreamingListLLM inherits from FakeListLLM."""
        llm = FakeStreamingListLLM(responses=["test"])
        assert isinstance(llm, FakeListLLM)

    def test_llm_type(self) -> None:
        """Test _llm_type property inherited from FakeListLLM."""
        llm = FakeStreamingListLLM(responses=["test"])
        assert llm._llm_type == "fake-list"

    def test_stream_empty_response(self) -> None:
        """Test stream with empty response."""
        llm = FakeStreamingListLLM(responses=[""])
        chunks = list(llm.stream("prompt"))
        assert chunks == []

    async def test_astream_empty_response(self) -> None:
        """Test astream with empty response."""
        llm = FakeStreamingListLLM(responses=[""])
        chunks = [chunk async for chunk in llm.astream("prompt")]
        assert chunks == []

    def test_stream_unicode_characters(self) -> None:
        """Test stream with unicode characters."""
        llm = FakeStreamingListLLM(responses=["你好"])
        chunks = list(llm.stream("prompt"))
        assert chunks == ["你", "好"]

    def test_stream_with_spaces(self) -> None:
        """Test stream with spaces."""
        llm = FakeStreamingListLLM(responses=["a b"])
        chunks = list(llm.stream("prompt"))
        assert chunks == ["a", " ", "b"]

    def test_stream_with_newlines(self) -> None:
        """Test stream with newlines."""
        llm = FakeStreamingListLLM(responses=["a\nb"])
        chunks = list(llm.stream("prompt"))
        assert chunks == ["a", "\n", "b"]
