"""Tests for langchain_core.language_models.fake_chat_models module."""

import time
from unittest.mock import MagicMock

import pytest

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import (
    FakeChatModel,
    FakeListChatModel,
    FakeListChatModelError,
    FakeMessagesListChatModel,
    GenericFakeChatModel,
    ParrotFakeChatModel,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult


class TestFakeMessagesListChatModel:
    """Tests for FakeMessagesListChatModel class."""

    def test_initialization(self) -> None:
        """Test FakeMessagesListChatModel initialization."""
        responses = [AIMessage(content="response1"), AIMessage(content="response2")]
        model = FakeMessagesListChatModel(responses=responses)
        assert model.responses == responses
        assert model.i == 0
        assert model.sleep is None

    def test_initialization_with_sleep(self) -> None:
        """Test FakeMessagesListChatModel with sleep parameter."""
        model = FakeMessagesListChatModel(
            responses=[AIMessage(content="test")], sleep=0.1
        )
        assert model.sleep == 0.1

    def test_llm_type(self) -> None:
        """Test _llm_type property."""
        model = FakeMessagesListChatModel(responses=[AIMessage(content="test")])
        assert model._llm_type == "fake-messages-list-chat-model"

    def test_invoke_returns_message(self) -> None:
        """Test invoke returns the message from responses."""
        response = AIMessage(content="hello")
        model = FakeMessagesListChatModel(responses=[response])
        result = model.invoke("any prompt")
        assert result.content == "hello"

    def test_invoke_cycles_through_responses(self) -> None:
        """Test invoke cycles through responses."""
        responses = [
            AIMessage(content="first"),
            AIMessage(content="second"),
            AIMessage(content="third"),
        ]
        model = FakeMessagesListChatModel(responses=responses)

        assert model.invoke("p1").content == "first"
        assert model.invoke("p2").content == "second"
        assert model.invoke("p3").content == "third"
        # Should cycle back
        assert model.invoke("p4").content == "first"

    def test_invoke_with_single_response(self) -> None:
        """Test invoke with single response stays at same."""
        model = FakeMessagesListChatModel(responses=[AIMessage(content="only")])

        assert model.invoke("p1").content == "only"
        assert model.invoke("p2").content == "only"
        assert model.i == 0

    def test_invoke_with_sleep(self) -> None:
        """Test invoke with sleep parameter."""
        import time

        model = FakeMessagesListChatModel(
            responses=[AIMessage(content="test")], sleep=0.05
        )
        start = time.time()
        model.invoke("prompt")
        elapsed = time.time() - start
        assert elapsed >= 0.05

    def test_generate_returns_chat_result(self) -> None:
        """Test _generate returns ChatResult."""
        model = FakeMessagesListChatModel(responses=[AIMessage(content="test")])
        result = model._generate([HumanMessage(content="hi")])
        assert len(result.generations) == 1
        assert result.generations[0].message.content == "test"


class TestFakeListChatModelError:
    """Tests for FakeListChatModelError exception."""

    def test_error_can_be_raised(self) -> None:
        """Test FakeListChatModelError can be raised."""
        with pytest.raises(FakeListChatModelError):
            raise FakeListChatModelError("test error")

    def test_error_is_exception(self) -> None:
        """Test FakeListChatModelError is an Exception."""
        error = FakeListChatModelError("test")
        assert isinstance(error, Exception)


class TestFakeListChatModel:
    """Tests for FakeListChatModel class."""

    def test_initialization(self) -> None:
        """Test FakeListChatModel initialization."""
        model = FakeListChatModel(responses=["response1", "response2"])
        assert model.responses == ["response1", "response2"]
        assert model.i == 0
        assert model.sleep is None
        assert model.error_on_chunk_number is None

    def test_llm_type(self) -> None:
        """Test _llm_type property."""
        model = FakeListChatModel(responses=["test"])
        assert model._llm_type == "fake-list-chat-model"

    def test_invoke_returns_ai_message(self) -> None:
        """Test invoke returns AIMessage."""
        model = FakeListChatModel(responses=["hello"])
        result = model.invoke("prompt")
        assert isinstance(result, AIMessage)
        assert result.content == "hello"

    def test_invoke_cycles_through_responses(self) -> None:
        """Test invoke cycles through responses."""
        model = FakeListChatModel(responses=["first", "second"])

        assert model.invoke("p1").content == "first"
        assert model.invoke("p2").content == "second"
        assert model.invoke("p3").content == "first"

    def test_stream_yields_characters(self) -> None:
        """Test stream yields individual characters."""
        model = FakeListChatModel(responses=["hello"])
        chunks = list(model.stream("prompt"))
        assert len(chunks) == 5
        assert all(isinstance(c, AIMessageChunk) for c in chunks)
        contents = [c.content for c in chunks]
        assert contents == ["h", "e", "l", "l", "o"]

    def test_stream_with_chunk_position(self) -> None:
        """Test stream sets chunk_position on last chunk."""
        model = FakeListChatModel(responses=["ab"])
        chunks = list(model.stream("prompt"))
        assert chunks[0].chunk_position is None
        assert chunks[1].chunk_position == "last"

    def test_stream_error_on_chunk(self) -> None:
        """Test stream raises error on specified chunk."""
        model = FakeListChatModel(responses=["hello"], error_on_chunk_number=2)
        stream = model.stream("prompt")

        next(stream)  # h
        next(stream)  # e
        with pytest.raises(FakeListChatModelError):
            next(stream)

    async def test_astream_yields_characters(self) -> None:
        """Test astream yields individual characters."""
        model = FakeListChatModel(responses=["hello"])
        chunks = [chunk async for chunk in model.astream("prompt")]
        assert len(chunks) == 5
        contents = [c.content for c in chunks]
        assert contents == ["h", "e", "l", "l", "o"]

    async def test_astream_error_on_chunk(self) -> None:
        """Test astream raises error on specified chunk."""
        model = FakeListChatModel(responses=["hello"], error_on_chunk_number=2)

        chunks = []
        with pytest.raises(FakeListChatModelError):
            async for chunk in model.astream("prompt"):
                chunks.append(chunk)

        assert len(chunks) == 2

    def test_identifying_params(self) -> None:
        """Test _identifying_params property."""
        model = FakeListChatModel(responses=["a", "b"])
        params = model._identifying_params
        assert params == {"responses": ["a", "b"]}

    def test_batch_preserves_order(self) -> None:
        """Test batch preserves order."""
        model = FakeListChatModel(responses=["r1", "r2", "r3"])
        results = model.batch(["p1", "p2", "p3"])
        assert [r.content for r in results] == ["r1", "r2", "r3"]

    async def test_abatch_preserves_order(self) -> None:
        """Test abatch preserves order."""
        model = FakeListChatModel(responses=["r1", "r2", "r3"])
        results = await model.abatch(["p1", "p2", "p3"])
        assert [r.content for r in results] == ["r1", "r2", "r3"]

    def test_batch_with_config_list(self) -> None:
        """Test batch with list of configs."""
        model = FakeListChatModel(responses=["r1", "r2"])
        results = model.batch(["p1", "p2"], config=[{}, {}])
        assert [r.content for r in results] == ["r1", "r2"]

    async def test_abatch_with_config_list(self) -> None:
        """Test abatch with list of configs."""
        model = FakeListChatModel(responses=["r1", "r2"])
        results = await model.abatch(["p1", "p2"], config=[{}, {}])
        assert [r.content for r in results] == ["r1", "r2"]


class TestFakeChatModel:
    """Tests for FakeChatModel class."""

    def test_initialization(self) -> None:
        """Test FakeChatModel initialization."""
        model = FakeChatModel()
        assert model._llm_type == "fake-chat-model"

    def test_invoke_returns_fake_response(self) -> None:
        """Test invoke always returns 'fake response'."""
        model = FakeChatModel()
        result = model.invoke("any prompt")
        assert result.content == "fake response"

    def test_invoke_ignores_input(self) -> None:
        """Test invoke ignores input content."""
        model = FakeChatModel()
        result1 = model.invoke("hello")
        result2 = model.invoke("goodbye")
        assert result1.content == result2.content == "fake response"

    async def test_ainvoke_returns_fake_response(self) -> None:
        """Test ainvoke returns 'fake response'."""
        model = FakeChatModel()
        result = await model.ainvoke("any prompt")
        assert result.content == "fake response"

    def test_identifying_params(self) -> None:
        """Test _identifying_params property."""
        model = FakeChatModel()
        params = model._identifying_params
        assert params == {"key": "fake"}


class TestGenericFakeChatModel:
    """Tests for GenericFakeChatModel class."""

    def test_initialization(self) -> None:
        """Test GenericFakeChatModel initialization."""
        messages = iter([AIMessage(content="test")])
        model = GenericFakeChatModel(messages=messages)
        assert model._llm_type == "generic-fake-chat-model"

    def test_invoke_returns_message_from_iterator(self) -> None:
        """Test invoke returns message from iterator."""
        messages = iter([AIMessage(content="hello")])
        model = GenericFakeChatModel(messages=messages)
        result = model.invoke("prompt")
        assert result.content == "hello"

    def test_invoke_with_string_messages(self) -> None:
        """Test invoke with string messages in iterator."""
        messages = iter(["hello", "world"])
        model = GenericFakeChatModel(messages=messages)

        result1 = model.invoke("p1")
        assert result1.content == "hello"

        result2 = model.invoke("p2")
        assert result2.content == "world"

    def test_invoke_exhausts_iterator(self) -> None:
        """Test invoke exhausts iterator."""
        messages = iter([AIMessage(content="only")])
        model = GenericFakeChatModel(messages=messages)

        model.invoke("p1")
        with pytest.raises(StopIteration):
            model.invoke("p2")

    def test_stream_splits_on_whitespace(self) -> None:
        """Test stream splits content on whitespace."""
        messages = iter([AIMessage(content="hello world")])
        model = GenericFakeChatModel(messages=messages)
        chunks = list(model.stream("prompt"))

        # Should split on whitespace, preserving it
        contents = [str(c.content) for c in chunks]
        assert "".join(contents) == "hello world"

    def test_stream_with_function_call(self) -> None:
        """Test stream with function call in additional_kwargs."""
        message = AIMessage(
            content="",
            additional_kwargs={
                "function_call": {"name": "test_func", "arguments": '{"a": 1}'}
            },
        )
        messages = iter([message])
        model = GenericFakeChatModel(messages=messages)
        chunks = list(model.stream("prompt"))

        # Should have chunks for function call
        assert len(chunks) > 0

    def test_stream_with_additional_kwargs(self) -> None:
        """Test stream with additional_kwargs."""
        message = AIMessage(
            content="", additional_kwargs={"custom_key": "custom_value"}
        )
        messages = iter([message])
        model = GenericFakeChatModel(messages=messages)
        chunks = list(model.stream("prompt"))

        # Should have chunk for additional kwargs
        assert len(chunks) > 0

    def test_stream_empty_content_raises_error(self) -> None:
        """Test stream with empty content raises ValueError."""
        messages = iter([AIMessage(content="")])
        model = GenericFakeChatModel(messages=messages)
        # Empty content yields no chunks, which raises ValueError
        with pytest.raises(ValueError, match="No generation chunks were returned"):
            list(model.stream("prompt"))


class TestParrotFakeChatModel:
    """Tests for ParrotFakeChatModel class."""

    def test_initialization(self) -> None:
        """Test ParrotFakeChatModel initialization."""
        model = ParrotFakeChatModel()
        assert model._llm_type == "parrot-fake-chat-model"

    def test_invoke_returns_last_message(self) -> None:
        """Test invoke returns the last message."""
        model = ParrotFakeChatModel()
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello!"),
        ]
        result = model.invoke(messages)
        # Should return the last message (HumanMessage)
        assert result.content == "Hello!"

    def test_invoke_with_single_message(self) -> None:
        """Test invoke with single message."""
        model = ParrotFakeChatModel()
        result = model.invoke([HumanMessage(content="Single")])
        assert result.content == "Single"

    def test_invoke_with_string_input(self) -> None:
        """Test invoke with string input."""
        model = ParrotFakeChatModel()
        result = model.invoke("Hello string")
        # String input gets converted to HumanMessage
        assert result.content == "Hello string"

    def test_invoke_preserves_message_type(self) -> None:
        """Test invoke preserves the message type in response."""
        model = ParrotFakeChatModel()
        messages = [HumanMessage(content="test")]
        result = model._generate(messages)
        # The generation should contain the original message
        assert result.generations[0].message.content == "test"

    async def test_ainvoke_returns_last_message(self) -> None:
        """Test ainvoke returns the last message."""
        model = ParrotFakeChatModel()
        messages = [
            HumanMessage(content="First"),
            HumanMessage(content="Last"),
        ]
        result = await model.ainvoke(messages)
        assert result.content == "Last"

    def test_generate_returns_chat_result(self) -> None:
        """Test _generate returns ChatResult."""
        model = ParrotFakeChatModel()
        result = model._generate([HumanMessage(content="test")])
        assert len(result.generations) == 1
        assert result.generations[0].message.content == "test"

    def test_batch_returns_last_messages(self) -> None:
        """Test batch returns last message from each input."""
        model = ParrotFakeChatModel()
        inputs = [
            [HumanMessage(content="batch1")],
            [HumanMessage(content="batch2")],
        ]
        results = model.batch(inputs)
        assert results[0].content == "batch1"
        assert results[1].content == "batch2"

    def test_with_complex_content(self) -> None:
        """Test with complex content blocks."""
        model = ParrotFakeChatModel()
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.png"},
                },
            ]
        )
        result = model.invoke([message])
        assert result.content == message.content


class TestFakeMessagesListChatModelAdditional:
    """Additional tests for FakeMessagesListChatModel class."""

    def test_generate_with_non_ai_message_response(self) -> None:
        """Test _generate can accept HumanMessage in responses list."""
        human_msg = HumanMessage(content="echoed back")
        model = FakeMessagesListChatModel(responses=[human_msg])
        result = model._generate([HumanMessage(content="hi")])
        assert len(result.generations) == 1
        assert result.generations[0].message.content == "echoed back"
        assert isinstance(result.generations[0].message, HumanMessage)

    async def test_ainvoke(self) -> None:
        """Test async invocation of FakeMessagesListChatModel."""
        responses = [
            AIMessage(content="async first"),
            AIMessage(content="async second"),
        ]
        model = FakeMessagesListChatModel(responses=responses)
        result1 = await model.ainvoke("prompt1")
        assert result1.content == "async first"
        result2 = await model.ainvoke("prompt2")
        assert result2.content == "async second"
        # Cycle back
        result3 = await model.ainvoke("prompt3")
        assert result3.content == "async first"

    def test_generate_returns_proper_chat_result_structure(self) -> None:
        """Test _generate returns proper ChatResult with generations list."""
        model = FakeMessagesListChatModel(responses=[AIMessage(content="structured")])
        result = model._generate([HumanMessage(content="hi")])
        assert isinstance(result, ChatResult)
        assert isinstance(result.generations, list)
        assert len(result.generations) == 1
        generation = result.generations[0]
        assert isinstance(generation, ChatGeneration)
        assert isinstance(generation.message, AIMessage)
        assert generation.message.content == "structured"

    def test_single_response_counter_stays_at_zero(self) -> None:
        """Test that with a single response, counter resets to 0 after each call."""
        model = FakeMessagesListChatModel(responses=[AIMessage(content="only one")])
        assert model.i == 0
        model.invoke("p1")
        assert model.i == 0
        model.invoke("p2")
        assert model.i == 0
        model.invoke("p3")
        assert model.i == 0


class TestFakeListChatModelAdditional:
    """Additional tests for FakeListChatModel class."""

    def test_call_with_sleep(self) -> None:
        """Test _call with sleep delays execution."""
        model = FakeListChatModel(responses=["hello"], sleep=0.05)
        start = time.time()
        result = model.invoke("prompt")
        elapsed = time.time() - start
        assert elapsed >= 0.05
        assert result.content == "hello"

    def test_stream_chunk_position_single_char(self) -> None:
        """Test _stream chunk_position on single char response is 'last'."""
        model = FakeListChatModel(responses=["x"])
        chunks = list(model.stream("prompt"))
        assert len(chunks) == 1
        assert chunks[0].content == "x"
        assert chunks[0].chunk_position == "last"

    async def test_astream_chunk_position_last(self) -> None:
        """Test _astream marks last chunk with chunk_position='last'."""
        model = FakeListChatModel(responses=["abc"])
        chunks = [chunk async for chunk in model.astream("prompt")]
        assert len(chunks) == 3
        # Only the last chunk should have chunk_position set
        assert chunks[0].chunk_position is None
        assert chunks[1].chunk_position is None
        assert chunks[2].chunk_position == "last"

    def test_stream_error_on_first_chunk(self) -> None:
        """Test _stream raises error on first chunk (index 0)."""
        model = FakeListChatModel(responses=["hello"], error_on_chunk_number=0)
        with pytest.raises(FakeListChatModelError):
            list(model.stream("prompt"))

    async def test_astream_error_on_first_chunk(self) -> None:
        """Test _astream raises error on first chunk (index 0)."""
        model = FakeListChatModel(responses=["hello"], error_on_chunk_number=0)
        chunks: list[AIMessageChunk] = []
        with pytest.raises(FakeListChatModelError):
            async for chunk in model.astream("prompt"):
                chunks.append(chunk)
        assert len(chunks) == 0

    def test_batch_with_single_config(self) -> None:
        """Test batch with a single config dict (not a list)."""
        model = FakeListChatModel(responses=["r1", "r2", "r3"])
        results = model.batch(["p1", "p2", "p3"], config={"metadata": {"key": "val"}})
        assert len(results) == 3
        assert [r.content for r in results] == ["r1", "r2", "r3"]

    async def test_abatch_with_single_config(self) -> None:
        """Test abatch with a single config dict (not a list)."""
        model = FakeListChatModel(responses=["r1", "r2", "r3"])
        results = await model.abatch(
            ["p1", "p2", "p3"], config={"metadata": {"key": "val"}}
        )
        assert len(results) == 3
        assert [r.content for r in results] == ["r1", "r2", "r3"]

    def test_stream_empty_string_response(self) -> None:
        """Test stream with empty string response raises ValueError."""
        model = FakeListChatModel(responses=[""])
        # Empty string has no characters, so _stream yields nothing,
        # and the base class raises ValueError for no generation chunks
        with pytest.raises(ValueError, match="No generation chunks were returned"):
            list(model.stream("prompt"))


class TestFakeChatModelAdditional:
    """Additional tests for FakeChatModel class."""

    async def test_agenerate_returns_chat_result(self) -> None:
        """Test _agenerate returns proper ChatResult structure."""
        model = FakeChatModel()
        result = await model._agenerate([HumanMessage(content="hi")])
        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1
        generation = result.generations[0]
        assert isinstance(generation, ChatGeneration)
        assert isinstance(generation.message, AIMessage)
        assert generation.message.content == "fake response"

    def test_llm_type_and_identifying_params_consistency(self) -> None:
        """Test _llm_type and _identifying_params return expected values."""
        model = FakeChatModel()
        assert model._llm_type == "fake-chat-model"
        params = model._identifying_params
        assert isinstance(params, dict)
        assert params == {"key": "fake"}
        # Verify consistency across multiple accesses
        assert model._llm_type == "fake-chat-model"
        assert model._identifying_params == {"key": "fake"}


class TestGenericFakeChatModelAdditional:
    """Additional tests for GenericFakeChatModel class."""

    def test_stream_with_multiple_words_preserves_whitespace(self) -> None:
        """Test _stream with multiple words preserves whitespace in detail."""
        messages = iter([AIMessage(content="hello world foo")])
        model = GenericFakeChatModel(messages=messages)
        chunks = list(model.stream("prompt"))
        contents = [str(c.content) for c in chunks]
        # re.split(r"(\s)", "hello world foo") produces
        # ["hello", " ", "world", " ", "foo"]
        assert contents == ["hello", " ", "world", " ", "foo"]
        # Reconstructed content matches original
        assert "".join(contents) == "hello world foo"

    def test_stream_function_call_non_string_values(self) -> None:
        """Test _stream with function_call having non-string values hits else branch."""
        message = AIMessage(
            content="",
            additional_kwargs={
                "function_call": {"name": "my_func", "parsed": {"key": "value"}}
            },
        )
        messages = iter([message])
        model = GenericFakeChatModel(messages=messages)
        chunks = list(model.stream("prompt"))
        assert len(chunks) > 0

        # Verify we get chunks for both function_call sub-keys
        all_kwargs = [c.additional_kwargs for c in chunks if c.additional_kwargs]
        # "name" is a string so it gets split by comma; "parsed" is a dict so it goes
        # through the else branch yielding a single chunk
        name_chunks = [
            kw
            for kw in all_kwargs
            if "function_call" in kw and "name" in kw["function_call"]
        ]
        parsed_chunks = [
            kw
            for kw in all_kwargs
            if "function_call" in kw and "parsed" in kw["function_call"]
        ]
        assert len(name_chunks) >= 1
        assert len(parsed_chunks) == 1
        assert parsed_chunks[0]["function_call"]["parsed"] == {"key": "value"}

    def test_stream_chunk_position_last_no_additional_kwargs(self) -> None:
        """Test _stream chunk_position on last chunk when no additional_kwargs."""
        messages = iter([AIMessage(content="hi there")])
        model = GenericFakeChatModel(messages=messages)
        chunks = list(model.stream("prompt"))
        # re.split(r"(\s)", "hi there") -> ["hi", " ", "there"]
        assert len(chunks) == 3
        # Only the last chunk should have chunk_position="last"
        assert chunks[0].chunk_position is None
        assert chunks[1].chunk_position is None
        assert chunks[2].chunk_position == "last"

    def test_stream_with_run_manager_callback(self) -> None:
        """Test _stream calls on_llm_new_token via run_manager."""
        messages = iter([AIMessage(content="hello world")])
        model = GenericFakeChatModel(messages=messages)

        # Create a mock run_manager
        run_manager = MagicMock(spec=CallbackManagerForLLMRun)

        chunks = list(
            model._stream([HumanMessage(content="prompt")], run_manager=run_manager)
        )
        # re.split(r"(\s)", "hello world") -> ["hello", " ", "world"]
        assert len(chunks) == 3

        # Verify on_llm_new_token was called for each chunk
        assert run_manager.on_llm_new_token.call_count == 3
        # Verify the tokens passed to on_llm_new_token
        tokens = [call.args[0] for call in run_manager.on_llm_new_token.call_args_list]
        assert tokens == ["hello", " ", "world"]

    def test_stream_with_content_and_additional_kwargs(self) -> None:
        """Test _stream with both content and additional_kwargs generates chunks."""
        message = AIMessage(
            content="hello",
            additional_kwargs={"custom_key": "custom_value"},
        )
        messages = iter([message])
        model = GenericFakeChatModel(messages=messages)
        chunks = list(model.stream("prompt"))

        # Content "hello" is a single word, so 1 content chunk
        # Plus 1 chunk for additional_kwargs
        assert len(chunks) >= 2

        # Verify content chunk(s)
        content_chunks = [c for c in chunks if c.content]
        assert len(content_chunks) >= 1
        assert "".join(str(c.content) for c in content_chunks) == "hello"

        # Verify additional_kwargs chunk
        kwargs_chunks = [c for c in chunks if c.additional_kwargs]
        assert len(kwargs_chunks) == 1
        assert kwargs_chunks[0].additional_kwargs == {"custom_key": "custom_value"}

        # Content chunk should NOT have chunk_position="last" since there are
        # additional_kwargs following
        last_content_chunk = content_chunks[-1]
        assert last_content_chunk.chunk_position is None


class TestParrotFakeChatModelAdditional:
    """Additional tests for ParrotFakeChatModel class."""

    def test_generate_with_multiple_messages_returns_last(self) -> None:
        """Test _generate with multiple messages returns the last one."""
        model = ParrotFakeChatModel()
        messages = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="first human"),
            HumanMessage(content="second human"),
            HumanMessage(content="last human"),
        ]
        result = model._generate(messages)
        assert len(result.generations) == 1
        assert result.generations[0].message.content == "last human"

    async def test_ainvoke_with_string(self) -> None:
        """Test ainvoke with string input echoes back the string."""
        model = ParrotFakeChatModel()
        result = await model.ainvoke("echo this string")
        assert result.content == "echo this string"
