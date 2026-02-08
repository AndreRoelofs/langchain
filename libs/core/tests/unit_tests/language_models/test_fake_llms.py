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


class TestFakeListLLMSleepParameter:
    """Tests for FakeListLLM sleep parameter behavior."""

    def test_sleep_stored_but_does_not_affect_call(self) -> None:
        """Test that sleep is stored but _call does not sleep.

        The sleep parameter is stored on FakeListLLM but is only used by
        subclasses like FakeStreamingListLLM. _call should return immediately
        regardless of the sleep value.
        """
        import time

        llm = FakeListLLM(responses=["response"], sleep=10.0)
        assert llm.sleep == 10.0

        start = time.time()
        result = llm._call("prompt")
        elapsed = time.time() - start

        assert result == "response"
        # _call should complete nearly instantly, not sleep 10 seconds
        assert elapsed < 1.0

    async def test_sleep_stored_but_does_not_affect_acall(self) -> None:
        """Test that sleep is stored but _acall does not sleep."""
        import time

        llm = FakeListLLM(responses=["response"], sleep=10.0)

        start = time.time()
        result = await llm._acall("prompt")
        elapsed = time.time() - start

        assert result == "response"
        assert elapsed < 1.0


class TestFakeListLLMGenerateMultiplePrompts:
    """Tests for FakeListLLM.generate with multiple prompts."""

    def test_generate_with_multiple_prompts(self) -> None:
        """Test generate with more than one prompt.

        Each prompt should get its own generation in the result, and
        responses should cycle through the response list.
        """
        llm = FakeListLLM(responses=["alpha", "beta", "gamma"])
        result = llm.generate(["prompt1", "prompt2", "prompt3"])

        assert len(result.generations) == 3
        assert result.generations[0][0].text == "alpha"
        assert result.generations[1][0].text == "beta"
        assert result.generations[2][0].text == "gamma"

    def test_generate_with_more_prompts_than_responses(self) -> None:
        """Test generate with more prompts than responses cycles correctly."""
        llm = FakeListLLM(responses=["first", "second"])
        result = llm.generate(["p1", "p2", "p3"])

        assert len(result.generations) == 3
        assert result.generations[0][0].text == "first"
        assert result.generations[1][0].text == "second"
        # Should cycle back to first
        assert result.generations[2][0].text == "first"


class TestFakeListLLMBatchCycling:
    """Tests for FakeListLLM batch counter state after processing."""

    def test_batch_cycles_correctly_and_updates_counter(self) -> None:
        """Test that batch processes items and counter reflects state after.

        After batching through all responses, the counter should have cycled
        back to 0.
        """
        llm = FakeListLLM(responses=["r1", "r2", "r3"])
        results = llm.batch(["p1", "p2", "p3"])

        assert results == ["r1", "r2", "r3"]
        # After going through all 3 responses, counter resets to 0
        assert llm.i == 0

    def test_batch_partial_cycle_updates_counter(self) -> None:
        """Test batch with fewer prompts than responses leaves counter mid-cycle."""
        llm = FakeListLLM(responses=["r1", "r2", "r3"])
        results = llm.batch(["p1"])

        assert results == ["r1"]
        # Counter should have advanced to 1
        assert llm.i == 1


class TestFakeListLLMTwoResponsesCycling:
    """Tests for FakeListLLM with exactly two responses, verifying counter state."""

    def test_two_responses_exact_counter_state(self) -> None:
        """Verify exact counter state at each step with two responses."""
        llm = FakeListLLM(responses=["a", "b"])

        assert llm.i == 0

        result1 = llm.invoke("p1")
        assert result1 == "a"
        assert llm.i == 1

        result2 = llm.invoke("p2")
        assert result2 == "b"
        assert llm.i == 0  # Reset after reaching end

        result3 = llm.invoke("p3")
        assert result3 == "a"
        assert llm.i == 1

        result4 = llm.invoke("p4")
        assert result4 == "b"
        assert llm.i == 0


class TestFakeListLLMCallResetsCounter:
    """Tests for FakeListLLM._call and _acall counter reset at end of list."""

    def test_call_resets_counter_at_end(self) -> None:
        """When i == len(responses)-1, _call should reset counter to 0."""
        llm = FakeListLLM(responses=["only_one"])
        assert llm.i == 0

        # Single response: i == len(responses)-1 == 0, so reset to 0
        result = llm._call("prompt")
        assert result == "only_one"
        assert llm.i == 0

    def test_call_resets_counter_at_end_multiple_responses(self) -> None:
        """Test counter reset with multiple responses, advancing to the last."""
        llm = FakeListLLM(responses=["a", "b", "c"])
        # Manually set counter to last response
        llm.i = 2

        result = llm._call("prompt")
        assert result == "c"
        assert llm.i == 0  # Reset after last response

    async def test_acall_resets_counter_at_end(self) -> None:
        """Async version: when i == len(responses)-1, _acall resets counter to 0."""
        llm = FakeListLLM(responses=["only_one"])
        assert llm.i == 0

        result = await llm._acall("prompt")
        assert result == "only_one"
        assert llm.i == 0

    async def test_acall_resets_counter_at_end_multiple_responses(self) -> None:
        """Async version: counter reset with multiple responses at the last."""
        llm = FakeListLLM(responses=["x", "y", "z"])
        llm.i = 2

        result = await llm._acall("prompt")
        assert result == "z"
        assert llm.i == 0


class TestFakeListLLMGenerateLLMResultStructure:
    """Tests for FakeListLLM.generate verifying LLMResult structure."""

    def test_generate_returns_proper_llm_result_structure(self) -> None:
        """Verify generate returns LLMResult with correct generations and llm_output."""
        from langchain_core.outputs import LLMResult
        from langchain_core.outputs.generation import Generation

        llm = FakeListLLM(responses=["hello", "world"])
        result = llm.generate(["prompt1", "prompt2"])

        # Result should be an LLMResult
        assert isinstance(result, LLMResult)

        # generations is a list of lists
        assert len(result.generations) == 2

        # Each generation list should have one Generation object
        assert len(result.generations[0]) == 1
        assert len(result.generations[1]) == 1

        # Each Generation should have the text attribute
        gen0 = result.generations[0][0]
        gen1 = result.generations[1][0]
        assert isinstance(gen0, Generation)
        assert isinstance(gen1, Generation)
        assert gen0.text == "hello"
        assert gen1.text == "world"

    def test_generate_single_prompt_structure(self) -> None:
        """Verify generate with single prompt returns correct nested structure."""
        from langchain_core.outputs import LLMResult

        llm = FakeListLLM(responses=["single response"])
        result = llm.generate(["one prompt"])

        assert isinstance(result, LLMResult)
        assert len(result.generations) == 1
        assert len(result.generations[0]) == 1
        assert result.generations[0][0].text == "single response"


class TestFakeStreamingListLLMSingleCharResponse:
    """Tests for FakeStreamingListLLM stream with single character response."""

    def test_stream_single_character(self) -> None:
        """Test streaming a response that is only one character long."""
        llm = FakeStreamingListLLM(responses=["x"])
        chunks = list(llm.stream("prompt"))
        assert chunks == ["x"]
        assert len(chunks) == 1

    async def test_astream_single_character(self) -> None:
        """Test async streaming a response that is only one character long."""
        llm = FakeStreamingListLLM(responses=["y"])
        chunks = [chunk async for chunk in llm.astream("prompt")]
        assert chunks == ["y"]
        assert len(chunks) == 1


class TestFakeStreamingListLLMAstreamSleepTiming:
    """Tests for FakeStreamingListLLM astream sleep timing accuracy."""

    async def test_astream_sleep_delays_proportional_to_chunks(self) -> None:
        """Verify async sleep actually delays proportional to chunk count.

        A 5-character response with 0.02s sleep per chunk should take at
        least 0.10 seconds total.
        """
        import time

        llm = FakeStreamingListLLM(responses=["abcde"], sleep=0.02)
        start = time.time()
        chunks = [chunk async for chunk in llm.astream("prompt")]
        elapsed = time.time() - start

        assert chunks == ["a", "b", "c", "d", "e"]
        # 5 chunks * 0.02s = 0.10s minimum
        assert elapsed >= 0.10

    async def test_astream_no_sleep_is_fast(self) -> None:
        """Verify astream without sleep completes quickly."""
        import time

        llm = FakeStreamingListLLM(responses=["abcde"])
        start = time.time()
        chunks = [chunk async for chunk in llm.astream("prompt")]
        elapsed = time.time() - start

        assert chunks == ["a", "b", "c", "d", "e"]
        # Without sleep, should complete very quickly
        assert elapsed < 1.0


class TestFakeStreamingListLLMErrorOnLastChunk:
    """Tests for FakeStreamingListLLM error_on_chunk_number on the very last chunk."""

    def test_stream_error_on_exact_last_chunk(self) -> None:
        """Test error raised on the very last character of the response.

        For a 3-character response 'abc', the last chunk index is 2.
        Setting error_on_chunk_number=2 should yield 'a', 'b', then raise.
        """
        llm = FakeStreamingListLLM(responses=["abc"], error_on_chunk_number=2)

        chunks = []
        with pytest.raises(FakeListLLMError):
            for chunk in llm.stream("prompt"):
                chunks.append(chunk)

        assert chunks == ["a", "b"]

    async def test_astream_error_on_exact_last_chunk(self) -> None:
        """Test async error raised on the very last character of the response."""
        llm = FakeStreamingListLLM(responses=["abc"], error_on_chunk_number=2)

        chunks = []
        with pytest.raises(FakeListLLMError):
            async for chunk in llm.astream("prompt"):
                chunks.append(chunk)

        assert chunks == ["a", "b"]

    def test_stream_error_on_last_chunk_single_char(self) -> None:
        """Test error on the only chunk of a single-character response."""
        llm = FakeStreamingListLLM(responses=["z"], error_on_chunk_number=0)

        chunks = []
        with pytest.raises(FakeListLLMError):
            for chunk in llm.stream("prompt"):
                chunks.append(chunk)

        assert chunks == []


class TestFakeStreamingListLLMStreamPreservesCycling:
    """Tests for FakeStreamingListLLM counter advancement after streaming."""

    def test_stream_advances_counter(self) -> None:
        """After streaming a response, the counter should advance.

        Since stream() calls invoke() internally, the counter is already
        advanced before characters are yielded.
        """
        llm = FakeStreamingListLLM(responses=["ab", "cd", "ef"])

        # Stream first response
        chunks1 = list(llm.stream("prompt1"))
        assert chunks1 == ["a", "b"]
        assert llm.i == 1

        # Stream second response
        chunks2 = list(llm.stream("prompt2"))
        assert chunks2 == ["c", "d"]
        assert llm.i == 2

        # Stream third response, counter cycles back
        chunks3 = list(llm.stream("prompt3"))
        assert chunks3 == ["e", "f"]
        assert llm.i == 0


class TestFakeStreamingListLLMIdentifyingParams:
    """Tests for FakeStreamingListLLM inheriting _identifying_params."""

    def test_identifying_params_inherited(self) -> None:
        """Verify _identifying_params works through inheritance from FakeListLLM."""
        llm = FakeStreamingListLLM(responses=["hello", "world"])
        params = llm._identifying_params
        assert params == {"responses": ["hello", "world"]}

    def test_identifying_params_with_extra_attributes(self) -> None:
        """Verify _identifying_params only contains responses, not subclass attrs."""
        llm = FakeStreamingListLLM(
            responses=["test"], error_on_chunk_number=5, sleep=0.5
        )
        params = llm._identifying_params
        # _identifying_params from FakeListLLM only includes responses
        assert params == {"responses": ["test"]}
        assert "error_on_chunk_number" not in params
        assert "sleep" not in params


class TestFakeStreamingListLLMMultipleSequentialStreams:
    """Tests for FakeStreamingListLLM with multiple sequential streams cycling."""

    def test_multiple_sequential_streams_cycle(self) -> None:
        """Stream response 1, then response 2, then cycle back to response 1."""
        llm = FakeStreamingListLLM(responses=["AB", "CD"])

        # First stream: response 1
        chunks1 = list(llm.stream("prompt1"))
        assert chunks1 == ["A", "B"]

        # Second stream: response 2
        chunks2 = list(llm.stream("prompt2"))
        assert chunks2 == ["C", "D"]

        # Third stream: cycles back to response 1
        chunks3 = list(llm.stream("prompt3"))
        assert chunks3 == ["A", "B"]

        # Fourth stream: response 2 again
        chunks4 = list(llm.stream("prompt4"))
        assert chunks4 == ["C", "D"]

    async def test_multiple_sequential_astreams_cycle(self) -> None:
        """Async version: multiple sequential streams cycle correctly."""
        llm = FakeStreamingListLLM(responses=["AB", "CD"])

        chunks1 = [chunk async for chunk in llm.astream("p1")]
        assert chunks1 == ["A", "B"]

        chunks2 = [chunk async for chunk in llm.astream("p2")]
        assert chunks2 == ["C", "D"]

        chunks3 = [chunk async for chunk in llm.astream("p3")]
        assert chunks3 == ["A", "B"]


class TestFakeListLLMInvokeWithMessageInput:
    """Tests for FakeListLLM invoke with message-type input."""

    def test_invoke_with_human_message_list(self) -> None:
        """Test that message-type inputs work via _convert_input.

        LLMs accept LanguageModelInput which includes list of BaseMessage.
        """
        from langchain_core.messages import HumanMessage

        llm = FakeListLLM(responses=["message response"])
        result = llm.invoke([HumanMessage(content="Hello")])
        assert result == "message response"

    def test_invoke_with_multiple_messages(self) -> None:
        """Test invoke with a list of multiple messages."""
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = FakeListLLM(responses=["multi message response"])
        result = llm.invoke(
            [
                SystemMessage(content="You are a helper."),
                HumanMessage(content="What is 2+2?"),
            ]
        )
        assert result == "multi message response"

    async def test_ainvoke_with_human_message_list(self) -> None:
        """Test async invoke with message-type inputs."""
        from langchain_core.messages import HumanMessage

        llm = FakeListLLM(responses=["async message response"])
        result = await llm.ainvoke([HumanMessage(content="Hello")])
        assert result == "async message response"


class TestFakeStreamingListLLMSpecialCharacters:
    """Tests for FakeStreamingListLLM stream with special characters."""

    def test_stream_with_emoji(self) -> None:
        """Test stream correctly yields emoji characters."""
        llm = FakeStreamingListLLM(responses=["hi\U0001f600"])
        chunks = list(llm.stream("prompt"))
        # "hi" + grinning face emoji
        assert chunks == ["h", "i", "\U0001f600"]

    def test_stream_with_tabs(self) -> None:
        """Test stream with tab characters."""
        llm = FakeStreamingListLLM(responses=["a\tb"])
        chunks = list(llm.stream("prompt"))
        assert chunks == ["a", "\t", "b"]

    def test_stream_with_mixed_special_characters(self) -> None:
        """Test stream with a mix of special characters: emoji, newline, tab."""
        llm = FakeStreamingListLLM(responses=["\U0001f44d\n\t"])
        chunks = list(llm.stream("prompt"))
        assert chunks == ["\U0001f44d", "\n", "\t"]

    def test_stream_with_carriage_return(self) -> None:
        """Test stream with carriage return characters."""
        llm = FakeStreamingListLLM(responses=["a\r\nb"])
        chunks = list(llm.stream("prompt"))
        assert chunks == ["a", "\r", "\n", "b"]

    async def test_astream_with_emoji(self) -> None:
        """Test async stream with emoji characters."""
        llm = FakeStreamingListLLM(responses=["\U0001f680\U0001f30d"])
        chunks = [chunk async for chunk in llm.astream("prompt")]
        # Rocket emoji + Earth emoji
        assert chunks == ["\U0001f680", "\U0001f30d"]

    def test_stream_with_null_byte(self) -> None:
        """Test stream with null byte character."""
        llm = FakeStreamingListLLM(responses=["a\x00b"])
        chunks = list(llm.stream("prompt"))
        assert chunks == ["a", "\x00", "b"]
