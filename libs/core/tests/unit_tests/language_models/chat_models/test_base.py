"""Test base chat model."""

import uuid
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Literal

import pytest
from typing_extensions import override

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    BaseChatModel,
    FakeListChatModel,
    ParrotFakeChatModel,
)
from langchain_core.language_models._utils import _normalize_messages
from langchain_core.language_models.chat_models import (
    SimpleChatModel,
    _cleanup_llm_representation,
    _format_for_tracing,
    _format_ls_structured_output,
    _gen_info_and_msg_metadata,
    _generate_response_from_error,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.language_models.fake_chat_models import (
    FakeChatModel,
    FakeListChatModelError,
    GenericFakeChatModel,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.outputs.generation import Generation
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.tracers import LogStreamCallbackHandler
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.context import collect_runs
from langchain_core.tracers.event_stream import _AstreamEventsCallbackHandler
from langchain_core.tracers.schemas import Run
from tests.unit_tests.fake.callbacks import (
    BaseFakeCallbackHandler,
    FakeAsyncCallbackHandler,
    FakeCallbackHandler,
)
from tests.unit_tests.stubs import _any_id_ai_message, _any_id_ai_message_chunk

if TYPE_CHECKING:
    from langchain_core.outputs.llm_result import LLMResult


def _content_blocks_equal_ignore_id(
    actual: str | list[Any], expected: str | list[Any]
) -> bool:
    """Compare content blocks, ignoring auto-generated `id` fields.

    Args:
        actual: Actual content from response (string or list of content blocks).
        expected: Expected content to compare against (string or list of blocks).

    Returns:
        True if content matches (excluding `id` fields), `False` otherwise.

    """
    if isinstance(actual, str) or isinstance(expected, str):
        return actual == expected

    if len(actual) != len(expected):
        return False
    for actual_block, expected_block in zip(actual, expected, strict=False):
        actual_without_id = (
            {k: v for k, v in actual_block.items() if k != "id"}
            if isinstance(actual_block, dict) and "id" in actual_block
            else actual_block
        )

        if actual_without_id != expected_block:
            return False

    return True


@pytest.fixture
def messages() -> list:
    return [
        SystemMessage(content="You are a test user."),
        HumanMessage(content="Hello, I am a test user."),
    ]


@pytest.fixture
def messages_2() -> list:
    return [
        SystemMessage(content="You are a test user."),
        HumanMessage(content="Hello, I not a test user."),
    ]


def test_batch_size(messages: list, messages_2: list) -> None:
    # The base endpoint doesn't support native batching,
    # so we expect batch_size to always be 1
    llm = FakeListChatModel(responses=[str(i) for i in range(100)])
    with collect_runs() as cb:
        llm.batch([messages, messages_2], {"callbacks": [cb]})
        assert len(cb.traced_runs) == 2
        assert all((r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs)
    with collect_runs() as cb:
        llm.batch([messages], {"callbacks": [cb]})
        assert all((r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs)
        assert len(cb.traced_runs) == 1

    with collect_runs() as cb:
        llm.invoke(messages)
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1

    with collect_runs() as cb:
        list(llm.stream(messages))
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1


async def test_async_batch_size(messages: list, messages_2: list) -> None:
    llm = FakeListChatModel(responses=[str(i) for i in range(100)])
    # The base endpoint doesn't support native batching,
    # so we expect batch_size to always be 1
    with collect_runs() as cb:
        await llm.abatch([messages, messages_2], {"callbacks": [cb]})
        assert all((r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs)
        assert len(cb.traced_runs) == 2
    with collect_runs() as cb:
        await llm.abatch([messages], {"callbacks": [cb]})
        assert all((r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs)
        assert len(cb.traced_runs) == 1

    with collect_runs() as cb:
        await llm.ainvoke(messages)
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1

    with collect_runs() as cb:
        async for _ in llm.astream(messages):
            pass
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1


@pytest.mark.xfail(reason="This test is failing due to a bug in the testing code")
async def test_stream_error_callback() -> None:
    message = "test"

    def eval_response(callback: BaseFakeCallbackHandler, i: int) -> None:
        assert callback.errors == 1
        assert len(callback.errors_args) == 1
        llm_result: LLMResult = callback.errors_args[0]["kwargs"]["response"]
        if i == 0:
            assert llm_result.generations == []
        else:
            assert llm_result.generations[0][0].text == message[:i]

    for i in range(len(message)):
        llm = FakeListChatModel(
            responses=[message],
            error_on_chunk_number=i,
        )
        cb_async = FakeAsyncCallbackHandler()
        llm_astream = llm.astream("Dummy message", config={"callbacks": [cb_async]})
        for _ in range(i):
            await llm_astream.__anext__()
        with pytest.raises(FakeListChatModelError):
            await llm_astream.__anext__()
        eval_response(cb_async, i)

        cb_sync = FakeCallbackHandler()
        llm_stream = llm.stream("Dumy message", config={"callbacks": [cb_sync]})
        for _ in range(i):
            next(llm_stream)
        with pytest.raises(FakeListChatModelError):
            next(llm_stream)
        eval_response(cb_sync, i)


async def test_astream_fallback_to_ainvoke() -> None:
    """Test `astream()` uses appropriate implementation."""

    class ModelWithGenerate(BaseChatModel):
        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            """Top Level call."""
            message = AIMessage(content="hello")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    model = ModelWithGenerate()
    chunks = list(model.stream("anything"))
    # BaseChatModel.stream is typed to return Iterator[BaseMessageChunk].
    # When streaming is disabled, it returns Iterator[BaseMessage], so the type hint
    # is not strictly correct.
    # LangChain documents a pattern of adding BaseMessageChunks to accumulate a stream.
    # This may be better done with `reduce(operator.add, chunks)`.
    assert chunks == [_any_id_ai_message(content="hello")]

    chunks = [chunk async for chunk in model.astream("anything")]
    assert chunks == [_any_id_ai_message(content="hello")]


async def test_astream_implementation_fallback_to_stream() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithSyncStream(BaseChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            """Top Level call."""
            raise NotImplementedError

        @override
        def _stream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> Iterator[ChatGenerationChunk]:
            """Stream the output of the model."""
            yield ChatGenerationChunk(message=AIMessageChunk(content="a"))
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="b", chunk_position="last")
            )

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    model = ModelWithSyncStream()
    chunks = list(model.stream("anything"))
    assert chunks == [
        _any_id_ai_message_chunk(
            content="a",
        ),
        _any_id_ai_message_chunk(content="b", chunk_position="last"),
    ]
    assert len({chunk.id for chunk in chunks}) == 1
    assert type(model)._astream == BaseChatModel._astream
    astream_chunks = [chunk async for chunk in model.astream("anything")]
    assert astream_chunks == [
        _any_id_ai_message_chunk(
            content="a",
        ),
        _any_id_ai_message_chunk(content="b", chunk_position="last"),
    ]
    assert len({chunk.id for chunk in astream_chunks}) == 1


async def test_astream_implementation_uses_astream() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithAsyncStream(BaseChatModel):
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            """Top Level call."""
            raise NotImplementedError

        @override
        async def _astream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,  # type: ignore[override]
            **kwargs: Any,
        ) -> AsyncIterator[ChatGenerationChunk]:
            """Stream the output of the model."""
            yield ChatGenerationChunk(message=AIMessageChunk(content="a"))
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="b", chunk_position="last")
            )

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    model = ModelWithAsyncStream()
    chunks = [chunk async for chunk in model.astream("anything")]
    assert chunks == [
        _any_id_ai_message_chunk(
            content="a",
        ),
        _any_id_ai_message_chunk(content="b", chunk_position="last"),
    ]
    assert len({chunk.id for chunk in chunks}) == 1


class FakeTracer(BaseTracer):
    def __init__(self) -> None:
        super().__init__()
        self.traced_run_ids: list = []

    def _persist_run(self, run: Run) -> None:
        """Persist a run."""
        self.traced_run_ids.append(run.id)


def test_pass_run_id() -> None:
    llm = FakeListChatModel(responses=["a", "b", "c"])
    cb = FakeTracer()
    uid1 = uuid.uuid4()
    llm.invoke("Dummy message", {"callbacks": [cb], "run_id": uid1})
    assert cb.traced_run_ids == [uid1]
    uid2 = uuid.uuid4()
    list(llm.stream("Dummy message", {"callbacks": [cb], "run_id": uid2}))
    assert cb.traced_run_ids == [uid1, uid2]
    uid3 = uuid.uuid4()
    llm.batch([["Dummy message"]], {"callbacks": [cb], "run_id": uid3})
    assert cb.traced_run_ids == [uid1, uid2, uid3]


async def test_async_pass_run_id() -> None:
    llm = FakeListChatModel(responses=["a", "b", "c"])
    cb = FakeTracer()
    uid1 = uuid.uuid4()
    await llm.ainvoke("Dummy message", {"callbacks": [cb], "run_id": uid1})
    assert cb.traced_run_ids == [uid1]
    uid2 = uuid.uuid4()
    async for _ in llm.astream("Dummy message", {"callbacks": [cb], "run_id": uid2}):
        pass
    assert cb.traced_run_ids == [uid1, uid2]

    uid3 = uuid.uuid4()
    await llm.abatch([["Dummy message"]], {"callbacks": [cb], "run_id": uid3})
    assert cb.traced_run_ids == [uid1, uid2, uid3]


class NoStreamingModel(BaseChatModel):
    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage("invoke"))])

    @property
    def _llm_type(self) -> str:
        return "model1"


class StreamingModel(NoStreamingModel):
    streaming: bool = False

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="stream"))


@pytest.mark.parametrize("disable_streaming", [True, False, "tool_calling"])
def test_disable_streaming(
    *,
    disable_streaming: bool | Literal["tool_calling"],
) -> None:
    model = StreamingModel(disable_streaming=disable_streaming)
    assert model.invoke([]).content == "invoke"

    expected = "invoke" if disable_streaming is True else "stream"
    assert next(model.stream([])).content == expected
    assert (
        model.invoke([], config={"callbacks": [LogStreamCallbackHandler()]}).content
        == expected
    )

    expected = "invoke" if disable_streaming in {"tool_calling", True} else "stream"
    assert next(model.stream([], tools=[{"type": "function"}])).content == expected
    assert (
        model.invoke(
            [], config={"callbacks": [LogStreamCallbackHandler()]}, tools=[{}]
        ).content
        == expected
    )


@pytest.mark.parametrize("disable_streaming", [True, False, "tool_calling"])
async def test_disable_streaming_async(
    *,
    disable_streaming: bool | Literal["tool_calling"],
) -> None:
    model = StreamingModel(disable_streaming=disable_streaming)
    assert (await model.ainvoke([])).content == "invoke"

    expected = "invoke" if disable_streaming is True else "stream"
    async for c in model.astream([]):
        assert c.content == expected
        break
    assert (
        await model.ainvoke([], config={"callbacks": [_AstreamEventsCallbackHandler()]})
    ).content == expected

    expected = "invoke" if disable_streaming in {"tool_calling", True} else "stream"
    async for c in model.astream([], tools=[{}]):
        assert c.content == expected
        break
    assert (
        await model.ainvoke(
            [], config={"callbacks": [_AstreamEventsCallbackHandler()]}, tools=[{}]
        )
    ).content == expected


async def test_streaming_attribute_overrides_streaming_callback() -> None:
    model = StreamingModel(streaming=False)
    assert (
        await model.ainvoke([], config={"callbacks": [_AstreamEventsCallbackHandler()]})
    ).content == "invoke"


@pytest.mark.parametrize("disable_streaming", [True, False, "tool_calling"])
def test_disable_streaming_no_streaming_model(
    *,
    disable_streaming: bool | Literal["tool_calling"],
) -> None:
    model = NoStreamingModel(disable_streaming=disable_streaming)
    assert model.invoke([]).content == "invoke"
    assert next(model.stream([])).content == "invoke"
    assert (
        model.invoke([], config={"callbacks": [LogStreamCallbackHandler()]}).content
        == "invoke"
    )
    assert next(model.stream([], tools=[{}])).content == "invoke"


@pytest.mark.parametrize("disable_streaming", [True, False, "tool_calling"])
async def test_disable_streaming_no_streaming_model_async(
    *,
    disable_streaming: bool | Literal["tool_calling"],
) -> None:
    model = NoStreamingModel(disable_streaming=disable_streaming)
    assert (await model.ainvoke([])).content == "invoke"
    async for c in model.astream([]):
        assert c.content == "invoke"
        break
    assert (
        await model.ainvoke([], config={"callbacks": [_AstreamEventsCallbackHandler()]})
    ).content == "invoke"
    async for c in model.astream([], tools=[{}]):
        assert c.content == "invoke"
        break


class FakeChatModelStartTracer(FakeTracer):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list = []

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> Run:
        _, messages = args
        self.messages.append(messages)
        return super().on_chat_model_start(
            *args,
            **kwargs,
        )


def test_trace_images_in_openai_format() -> None:
    """Test that images are traced in OpenAI Chat Completions format."""
    llm = ParrotFakeChatModel()
    messages = [
        {
            "role": "user",
            # v0 format
            "content": [
                {
                    "type": "image",
                    "source_type": "url",
                    "url": "https://example.com/image.png",
                }
            ],
        }
    ]
    tracer = FakeChatModelStartTracer()
    llm.invoke(messages, config={"callbacks": [tracer]})
    assert tracer.messages == [
        [
            [
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.png"},
                        }
                    ]
                )
            ]
        ]
    ]


def test_trace_pdfs() -> None:
    # For backward compat
    llm = ParrotFakeChatModel()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "mime_type": "application/pdf",
                    "base64": "<base64 string>",
                }
            ],
        }
    ]
    tracer = FakeChatModelStartTracer()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        llm.invoke(messages, config={"callbacks": [tracer]})

    assert tracer.messages == [
        [
            [
                HumanMessage(
                    content=[
                        {
                            "type": "file",
                            "mime_type": "application/pdf",
                            "source_type": "base64",
                            "data": "<base64 string>",
                        }
                    ]
                )
            ]
        ]
    ]


def test_content_block_transformation_v0_to_v1_image() -> None:
    """Test that v0 format image content blocks are transformed to v1 format."""
    # Create a message with v0 format image content
    image_message = AIMessage(
        content=[
            {
                "type": "image",
                "source_type": "url",
                "url": "https://example.com/image.png",
            }
        ]
    )

    llm = GenericFakeChatModel(messages=iter([image_message]), output_version="v1")
    response = llm.invoke("test")

    # With v1 output_version, .content should be transformed
    # Check structure, ignoring auto-generated IDs
    assert len(response.content) == 1
    content_block = response.content[0]
    if isinstance(content_block, dict) and "id" in content_block:
        # Remove auto-generated id for comparison
        content_without_id = {k: v for k, v in content_block.items() if k != "id"}
        expected_content = {
            "type": "image",
            "url": "https://example.com/image.png",
        }
        assert content_without_id == expected_content
    else:
        assert content_block == {
            "type": "image",
            "url": "https://example.com/image.png",
        }


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_trace_content_blocks_with_no_type_key(output_version: str) -> None:
    """Test behavior of content blocks that don't have a `type` key.

    Only for blocks with one key, in which case, the name of the key is used as `type`.

    """
    llm = ParrotFakeChatModel(output_version=output_version)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hello",
                },
                {
                    "cachePoint": {"type": "default"},
                },
            ],
        }
    ]
    tracer = FakeChatModelStartTracer()
    response = llm.invoke(messages, config={"callbacks": [tracer]})
    assert tracer.messages == [
        [
            [
                HumanMessage(
                    [
                        {
                            "type": "text",
                            "text": "Hello",
                        },
                        {
                            "type": "cachePoint",
                            "cachePoint": {"type": "default"},
                        },
                    ]
                )
            ]
        ]
    ]

    if output_version == "v0":
        assert response.content == [
            {
                "type": "text",
                "text": "Hello",
            },
            {
                "cachePoint": {"type": "default"},
            },
        ]
    else:
        assert response.content == [
            {
                "type": "text",
                "text": "Hello",
            },
            {
                "type": "non_standard",
                "value": {
                    "cachePoint": {"type": "default"},
                },
            },
        ]

    assert response.content_blocks == [
        {
            "type": "text",
            "text": "Hello",
        },
        {
            "type": "non_standard",
            "value": {
                "cachePoint": {"type": "default"},
            },
        },
    ]


def test_extend_support_to_openai_multimodal_formats() -> None:
    """Test normalizing OpenAI audio, image, and file inputs to v1."""
    # Audio and file only (chat model default)
    messages = HumanMessage(
        content=[
            {"type": "text", "text": "Hello"},
            {  # audio-base64
                "type": "input_audio",
                "input_audio": {
                    "format": "wav",
                    "data": "<base64 string>",
                },
            },
            {  # file-base64
                "type": "file",
                "file": {
                    "filename": "draconomicon.pdf",
                    "file_data": "data:application/pdf;base64,<base64 string>",
                },
            },
            {  # file-id
                "type": "file",
                "file": {"file_id": "<file id>"},
            },
        ]
    )

    expected_content_messages = HumanMessage(
        content=[
            {"type": "text", "text": "Hello"},  # TextContentBlock
            {  # AudioContentBlock
                "type": "audio",
                "base64": "<base64 string>",
                "mime_type": "audio/wav",
            },
            {  # FileContentBlock
                "type": "file",
                "base64": "<base64 string>",
                "mime_type": "application/pdf",
                "extras": {"filename": "draconomicon.pdf"},
            },
            {  # ...
                "type": "file",
                "file_id": "<file id>",
            },
        ]
    )

    normalized_content = _normalize_messages([messages])

    # Check structure, ignoring auto-generated IDs
    assert len(normalized_content) == 1
    normalized_message = normalized_content[0]
    assert len(normalized_message.content) == len(expected_content_messages.content)

    assert _content_blocks_equal_ignore_id(
        normalized_message.content, expected_content_messages.content
    )

    messages = HumanMessage(
        content=[
            {"type": "text", "text": "Hello"},
            {  # image-url
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
            {  # image-base64
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."},
            },
            {  # audio-base64
                "type": "input_audio",
                "input_audio": {
                    "format": "wav",
                    "data": "<base64 string>",
                },
            },
            {  # file-base64
                "type": "file",
                "file": {
                    "filename": "draconomicon.pdf",
                    "file_data": "data:application/pdf;base64,<base64 string>",
                },
            },
            {  # file-id
                "type": "file",
                "file": {"file_id": "<file id>"},
            },
        ]
    )

    expected_content_messages = HumanMessage(
        content=[
            {"type": "text", "text": "Hello"},  # TextContentBlock
            {  # image-url passes through
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
            {  # image-url passes through with inline data
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."},
            },
            {  # AudioContentBlock
                "type": "audio",
                "base64": "<base64 string>",
                "mime_type": "audio/wav",
            },
            {  # FileContentBlock
                "type": "file",
                "base64": "<base64 string>",
                "mime_type": "application/pdf",
                "extras": {"filename": "draconomicon.pdf"},
            },
            {  # ...
                "type": "file",
                "file_id": "<file id>",
            },
        ]
    )

    normalized_content = _normalize_messages([messages])

    # Check structure, ignoring auto-generated IDs
    assert len(normalized_content) == 1
    normalized_message = normalized_content[0]
    assert len(normalized_message.content) == len(expected_content_messages.content)

    assert _content_blocks_equal_ignore_id(
        normalized_message.content, expected_content_messages.content
    )


def test_normalize_messages_edge_cases() -> None:
    # Test behavior of malformed/unrecognized content blocks

    messages = [
        HumanMessage(
            content=[
                {
                    "type": "input_image",  # Responses API type; not handled
                    "image_url": "uri",
                },
                {
                    # Standard OpenAI Chat Completions type but malformed structure
                    "type": "input_audio",
                    "input_audio": "uri",  # Should be nested in `audio`
                },
                {
                    "type": "file",
                    "file": "uri",  # `file` should be a dict for Chat Completions
                },
                {
                    "type": "input_file",  # Responses API type; not handled
                    "file_data": "uri",
                    "filename": "file-name",
                },
            ]
        )
    ]

    assert messages == _normalize_messages(messages)


def test_normalize_messages_v1_content_blocks_unchanged() -> None:
    """Test passing v1 content blocks to `_normalize_messages()` leaves unchanged."""
    input_messages = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Hello world",
                },
                {
                    "type": "image",
                    "url": "https://example.com/image.png",
                    "mime_type": "image/png",
                },
                {
                    "type": "audio",
                    "base64": "base64encodedaudiodata",
                    "mime_type": "audio/wav",
                },
                {
                    "type": "file",
                    "id": "file_123",
                },
                {
                    "type": "reasoning",
                    "reasoning": "Let me think about this...",
                },
            ]
        )
    ]

    result = _normalize_messages(input_messages)

    # Verify the result is identical to the input (message should not be copied)
    assert len(result) == 1
    assert result[0] is input_messages[0]
    assert result[0].content == input_messages[0].content


def test_output_version_invoke(monkeypatch: Any) -> None:
    messages = [AIMessage("hello")]

    llm = GenericFakeChatModel(messages=iter(messages), output_version="v1")
    response = llm.invoke("hello")
    assert response.content == [{"type": "text", "text": "hello"}]
    assert response.response_metadata["output_version"] == "v1"

    llm = GenericFakeChatModel(messages=iter(messages))
    response = llm.invoke("hello")
    assert response.content == "hello"

    monkeypatch.setenv("LC_OUTPUT_VERSION", "v1")
    llm = GenericFakeChatModel(messages=iter(messages))
    response = llm.invoke("hello")
    assert response.content == [{"type": "text", "text": "hello"}]
    assert response.response_metadata["output_version"] == "v1"


# -- v1 output version tests --


async def test_output_version_ainvoke(monkeypatch: Any) -> None:
    messages = [AIMessage("hello")]

    # v0
    llm = GenericFakeChatModel(messages=iter(messages))
    response = await llm.ainvoke("hello")
    assert response.content == "hello"

    # v1
    llm = GenericFakeChatModel(messages=iter(messages), output_version="v1")
    response = await llm.ainvoke("hello")
    assert response.content == [{"type": "text", "text": "hello"}]
    assert response.response_metadata["output_version"] == "v1"

    # v1 from env var
    monkeypatch.setenv("LC_OUTPUT_VERSION", "v1")
    llm = GenericFakeChatModel(messages=iter(messages))
    response = await llm.ainvoke("hello")
    assert response.content == [{"type": "text", "text": "hello"}]
    assert response.response_metadata["output_version"] == "v1"


class _AnotherFakeChatModel(BaseChatModel):
    responses: Iterator[AIMessage]
    """Responses for _generate."""

    chunks: Iterator[AIMessageChunk]
    """Responses for _stream."""

    @property
    def _llm_type(self) -> str:
        return "another-fake-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=next(self.responses))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=next(self.responses))])

    def _stream(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self.chunks:
            yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self.chunks:
            yield ChatGenerationChunk(message=chunk)


def test_output_version_stream(monkeypatch: Any) -> None:
    messages = [AIMessage("foo bar")]

    # v0
    llm = GenericFakeChatModel(messages=iter(messages))
    full = None
    for chunk in llm.stream("hello"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, str)
        assert chunk.content
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.content == "foo bar"

    # v1
    llm = GenericFakeChatModel(messages=iter(messages), output_version="v1")
    full_v1: AIMessageChunk | None = None
    for chunk in llm.stream("hello"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, list)
        assert len(chunk.content) == 1
        block = chunk.content[0]
        assert isinstance(block, dict)
        assert block["type"] == "text"
        assert block["text"]
        full_v1 = chunk if full_v1 is None else full_v1 + chunk
    assert isinstance(full_v1, AIMessageChunk)
    assert full_v1.response_metadata["output_version"] == "v1"

    assert full_v1.content == [{"type": "text", "text": "foo bar", "index": 0}]

    # Test text blocks
    llm_with_rich_content = _AnotherFakeChatModel(
        responses=iter([]),
        chunks=iter(
            [
                AIMessageChunk(content="foo "),
                AIMessageChunk(content="bar"),
            ]
        ),
        output_version="v1",
    )
    full_v1 = None
    for chunk in llm_with_rich_content.stream("hello"):
        full_v1 = chunk if full_v1 is None else full_v1 + chunk
    assert isinstance(full_v1, AIMessageChunk)
    assert full_v1.content_blocks == [{"type": "text", "text": "foo bar", "index": 0}]

    # Test content blocks of different types
    chunks = [
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "<rea"}),
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "soning>"}),
        AIMessageChunk(content="<some "),
        AIMessageChunk(content="text>"),
    ]
    llm_with_rich_content = _AnotherFakeChatModel(
        responses=iter([]),
        chunks=iter(chunks),
        output_version="v1",
    )
    full_v1 = None
    for chunk in llm_with_rich_content.stream("hello"):
        full_v1 = chunk if full_v1 is None else full_v1 + chunk
    assert isinstance(full_v1, AIMessageChunk)
    assert full_v1.content_blocks == [
        {"type": "reasoning", "reasoning": "<reasoning>", "index": 0},
        {"type": "text", "text": "<some text>", "index": 1},
    ]

    # Test invoke with stream=True
    llm_with_rich_content = _AnotherFakeChatModel(
        responses=iter([]),
        chunks=iter(chunks),
        output_version="v1",
    )
    response_v1 = llm_with_rich_content.invoke("hello", stream=True)
    assert response_v1.content_blocks == [
        {"type": "reasoning", "reasoning": "<reasoning>", "index": 0},
        {"type": "text", "text": "<some text>", "index": 1},
    ]

    # v1 from env var
    monkeypatch.setenv("LC_OUTPUT_VERSION", "v1")
    llm = GenericFakeChatModel(messages=iter(messages))
    full_env = None
    for chunk in llm.stream("hello"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, list)
        assert len(chunk.content) == 1
        block = chunk.content[0]
        assert isinstance(block, dict)
        assert block["type"] == "text"
        assert block["text"]
        full_env = chunk if full_env is None else full_env + chunk
    assert isinstance(full_env, AIMessageChunk)
    assert full_env.response_metadata["output_version"] == "v1"


async def test_output_version_astream(monkeypatch: Any) -> None:
    messages = [AIMessage("foo bar")]

    # v0
    llm = GenericFakeChatModel(messages=iter(messages))
    full = None
    async for chunk in llm.astream("hello"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, str)
        assert chunk.content
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.content == "foo bar"

    # v1
    llm = GenericFakeChatModel(messages=iter(messages), output_version="v1")
    full_v1: AIMessageChunk | None = None
    async for chunk in llm.astream("hello"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, list)
        assert len(chunk.content) == 1
        block = chunk.content[0]
        assert isinstance(block, dict)
        assert block["type"] == "text"
        assert block["text"]
        full_v1 = chunk if full_v1 is None else full_v1 + chunk
    assert isinstance(full_v1, AIMessageChunk)
    assert full_v1.response_metadata["output_version"] == "v1"

    assert full_v1.content == [{"type": "text", "text": "foo bar", "index": 0}]

    # Test text blocks
    llm_with_rich_content = _AnotherFakeChatModel(
        responses=iter([]),
        chunks=iter(
            [
                AIMessageChunk(content="foo "),
                AIMessageChunk(content="bar"),
            ]
        ),
        output_version="v1",
    )
    full_v1 = None
    async for chunk in llm_with_rich_content.astream("hello"):
        full_v1 = chunk if full_v1 is None else full_v1 + chunk
    assert isinstance(full_v1, AIMessageChunk)
    assert full_v1.content_blocks == [{"type": "text", "text": "foo bar", "index": 0}]

    # Test content blocks of different types
    chunks = [
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "<rea"}),
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "soning>"}),
        AIMessageChunk(content="<some "),
        AIMessageChunk(content="text>"),
    ]
    llm_with_rich_content = _AnotherFakeChatModel(
        responses=iter([]),
        chunks=iter(chunks),
        output_version="v1",
    )
    full_v1 = None
    async for chunk in llm_with_rich_content.astream("hello"):
        full_v1 = chunk if full_v1 is None else full_v1 + chunk
    assert isinstance(full_v1, AIMessageChunk)
    assert full_v1.content_blocks == [
        {"type": "reasoning", "reasoning": "<reasoning>", "index": 0},
        {"type": "text", "text": "<some text>", "index": 1},
    ]

    # Test invoke with stream=True
    llm_with_rich_content = _AnotherFakeChatModel(
        responses=iter([]),
        chunks=iter(chunks),
        output_version="v1",
    )
    response_v1 = await llm_with_rich_content.ainvoke("hello", stream=True)
    assert response_v1.content_blocks == [
        {"type": "reasoning", "reasoning": "<reasoning>", "index": 0},
        {"type": "text", "text": "<some text>", "index": 1},
    ]

    # v1 from env var
    monkeypatch.setenv("LC_OUTPUT_VERSION", "v1")
    llm = GenericFakeChatModel(messages=iter(messages))
    full_env = None
    async for chunk in llm.astream("hello"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, list)
        assert len(chunk.content) == 1
        block = chunk.content[0]
        assert isinstance(block, dict)
        assert block["type"] == "text"
        assert block["text"]
        full_env = chunk if full_env is None else full_env + chunk
    assert isinstance(full_env, AIMessageChunk)
    assert full_env.response_metadata["output_version"] == "v1"
    assert messages == _normalize_messages(messages)


def test_get_ls_params() -> None:
    class LSParamsModel(BaseChatModel):
        model: str = "foo"
        temperature: float = 0.1
        max_tokens: int = 1024

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            raise NotImplementedError

        @override
        def _stream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> Iterator[ChatGenerationChunk]:
            raise NotImplementedError

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    llm = LSParamsModel()

    # Test standard tracing params
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "lsparamsmodel",
        "ls_model_type": "chat",
        "ls_model_name": "foo",
        "ls_temperature": 0.1,
        "ls_max_tokens": 1024,
    }

    ls_params = llm._get_ls_params(model="bar")
    assert ls_params["ls_model_name"] == "bar"

    ls_params = llm._get_ls_params(temperature=0.2)
    assert ls_params["ls_temperature"] == 0.2

    ls_params = llm._get_ls_params(max_tokens=2048)
    assert ls_params["ls_max_tokens"] == 2048

    ls_params = llm._get_ls_params(stop=["stop"])
    assert ls_params["ls_stop"] == ["stop"]


def test_model_profiles() -> None:
    model = GenericFakeChatModel(messages=iter([]))
    assert model.profile is None

    model_with_profile = GenericFakeChatModel(
        messages=iter([]), profile={"max_input_tokens": 100}
    )
    assert model_with_profile.profile == {"max_input_tokens": 100}


class MockResponse:
    """Mock response for testing _generate_response_from_error."""

    def __init__(
        self,
        status_code: int = 400,
        headers: dict[str, str] | None = None,
        json_data: dict[str, Any] | None = None,
        json_raises: type[Exception] | None = None,
        text_raises: type[Exception] | None = None,
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self._json_data = json_data
        self._json_raises = json_raises
        self._text_raises = text_raises

    def json(self) -> dict[str, Any]:
        if self._json_raises:
            msg = "JSON parsing failed"
            raise self._json_raises(msg)
        return self._json_data or {}

    @property
    def text(self) -> str:
        if self._text_raises:
            msg = "Text access failed"
            raise self._text_raises(msg)
        return ""


class MockAPIError(Exception):
    """Mock API error with response attribute."""

    def __init__(self, message: str, response: MockResponse | None = None):
        super().__init__(message)
        self.message = message
        if response is not None:
            self.response = response


def test_generate_response_from_error_with_valid_json() -> None:
    """Test `_generate_response_from_error` with valid JSON response."""
    response = MockResponse(
        status_code=400,
        headers={"content-type": "application/json"},
        json_data={"error": {"message": "Bad request", "type": "invalid_request"}},
    )
    error = MockAPIError("API Error", response=response)

    generations = _generate_response_from_error(error)

    assert len(generations) == 1
    generation = generations[0]
    assert isinstance(generation, ChatGeneration)
    assert isinstance(generation.message, AIMessage)
    assert generation.message.content == ""

    metadata = generation.message.response_metadata
    assert metadata["body"] == {
        "error": {"message": "Bad request", "type": "invalid_request"}
    }
    assert metadata["headers"] == {"content-type": "application/json"}
    assert metadata["status_code"] == 400


def test_generate_response_from_error_handles_streaming_response_failure() -> None:
    # Simulates scenario where accessing response.json() or response.text
    # raises ResponseNotRead on streaming responses
    response = MockResponse(
        status_code=400,
        headers={"content-type": "application/json"},
        json_raises=Exception,  # Simulates ResponseNotRead or similar
        text_raises=Exception,
    )
    error = MockAPIError("API Error", response=response)

    # This should NOT raise an exception, but should handle it gracefully
    generations = _generate_response_from_error(error)

    assert len(generations) == 1
    generation = generations[0]
    metadata = generation.message.response_metadata

    # When both fail, body should be None instead of raising an exception
    assert metadata["body"] is None
    assert metadata["headers"] == {"content-type": "application/json"}
    assert metadata["status_code"] == 400


# ---------------------------------------------------------------------------
# New tests for additional coverage of chat_models.py
# ---------------------------------------------------------------------------


class TestConvertInput:
    """Tests for ``BaseChatModel._convert_input``."""

    def _get_model(self) -> BaseChatModel:
        return FakeListChatModel(responses=["hi"])

    def test_convert_input_from_string(self) -> None:
        """A plain string should be wrapped in a ``StringPromptValue``."""
        model = self._get_model()
        result = model._convert_input("hello")
        assert isinstance(result, StringPromptValue)
        assert result.text == "hello"

    def test_convert_input_from_prompt_value(self) -> None:
        """A ``PromptValue`` should be returned as-is."""
        model = self._get_model()
        pv = StringPromptValue(text="hello")
        result = model._convert_input(pv)
        assert result is pv

    def test_convert_input_from_chat_prompt_value(self) -> None:
        """A ``ChatPromptValue`` should be returned as-is."""
        model = self._get_model()
        cpv = ChatPromptValue(messages=[HumanMessage(content="hi")])
        result = model._convert_input(cpv)
        assert result is cpv

    def test_convert_input_from_message_sequence(self) -> None:
        """A list of messages should be converted to a ``ChatPromptValue``."""
        model = self._get_model()
        msgs = [HumanMessage(content="hello"), AIMessage(content="world")]
        result = model._convert_input(msgs)
        assert isinstance(result, ChatPromptValue)
        assert len(result.messages) == 2

    def test_convert_input_from_dict_messages(self) -> None:
        """A list of dicts (OpenAI-style) should be converted."""
        model = self._get_model()
        result = model._convert_input([{"role": "user", "content": "hello"}])
        assert isinstance(result, ChatPromptValue)
        assert len(result.messages) == 1
        assert isinstance(result.messages[0], HumanMessage)

    def test_convert_input_invalid_type_raises(self) -> None:
        """Non-string, non-PromptValue, non-sequence should raise ``ValueError``."""
        model = self._get_model()
        with pytest.raises(ValueError, match="Invalid input type"):
            model._convert_input(12345)  # type: ignore[arg-type]


class TestBaseChatModelDict:
    """Tests for ``BaseChatModel.dict()``."""

    def test_dict_contains_type_key(self) -> None:
        """``dict()`` should include ``_type`` from ``_llm_type``."""
        model = FakeChatModel()
        d = model.dict()
        assert "_type" in d
        assert d["_type"] == "fake-chat-model"

    def test_dict_contains_identifying_params(self) -> None:
        """``dict()`` should include the identifying params."""
        model = FakeChatModel()
        d = model.dict()
        # FakeChatModel._identifying_params returns {"key": "fake"}
        assert d.get("key") == "fake"

    def test_dict_identifying_params_for_fake_list(self) -> None:
        """``FakeListChatModel.dict()`` includes responses."""
        model = FakeListChatModel(responses=["a", "b"])
        d = model.dict()
        assert d["_type"] == "fake-list-chat-model"
        assert d["responses"] == ["a", "b"]


class TestOutputTypeProperty:
    """Tests for ``BaseChatModel.OutputType``."""

    def test_output_type_is_any_message(self) -> None:
        model = FakeListChatModel(responses=["x"])
        assert model.OutputType is AnyMessage


class TestGenerateFromStream:
    """Tests for ``generate_from_stream`` and ``agenerate_from_stream``."""

    def test_accumulates_chunks(self) -> None:
        """Should accumulate chunks and return a ``ChatResult``."""
        chunks = iter(
            [
                ChatGenerationChunk(message=AIMessageChunk(content="hel")),
                ChatGenerationChunk(message=AIMessageChunk(content="lo")),
            ]
        )
        result = generate_from_stream(chunks)
        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1
        gen = result.generations[0]
        assert isinstance(gen, ChatGeneration)
        assert isinstance(gen.message, AIMessage)
        assert gen.message.content == "hello"

    def test_single_chunk(self) -> None:
        """A single chunk should still work."""
        chunks = iter([ChatGenerationChunk(message=AIMessageChunk(content="only"))])
        result = generate_from_stream(chunks)
        assert result.generations[0].message.content == "only"

    def test_empty_stream_raises_value_error(self) -> None:
        """An empty iterator should raise ``ValueError``."""
        chunks: Iterator[ChatGenerationChunk] = iter([])
        with pytest.raises(ValueError, match="No generations found in stream"):
            generate_from_stream(chunks)

    def test_preserves_generation_info(self) -> None:
        """``generation_info`` from the accumulated chunk is preserved."""
        chunks = iter(
            [
                ChatGenerationChunk(
                    message=AIMessageChunk(content="a"),
                    generation_info={"finish_reason": "stop"},
                ),
            ]
        )
        result = generate_from_stream(chunks)
        assert result.generations[0].generation_info == {"finish_reason": "stop"}


class TestAGenerateFromStream:
    """Tests for the async ``agenerate_from_stream``."""

    async def test_accumulates_chunks(self) -> None:
        async def _stream() -> AsyncIterator[ChatGenerationChunk]:
            yield ChatGenerationChunk(message=AIMessageChunk(content="hel"))
            yield ChatGenerationChunk(message=AIMessageChunk(content="lo"))

        result = await agenerate_from_stream(_stream())
        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1
        assert result.generations[0].message.content == "hello"

    async def test_empty_stream_raises_value_error(self) -> None:
        async def _stream() -> AsyncIterator[ChatGenerationChunk]:
            return
            yield  # noqa: RET504  # make it an async generator

        with pytest.raises(ValueError, match="No generations found in stream"):
            await agenerate_from_stream(_stream())


class _MinimalSimpleChatModel(SimpleChatModel):
    """Minimal ``SimpleChatModel`` subclass for testing."""

    @property
    def _llm_type(self) -> str:
        return "minimal-simple"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        return "simple response"


class TestSimpleChatModelGenerate:
    """Tests for ``SimpleChatModel._generate``."""

    def test_generate_wraps_call_output(self) -> None:
        """``_generate`` should wrap ``_call`` output in a ``ChatResult``."""
        model = _MinimalSimpleChatModel()
        result = model._generate([HumanMessage(content="hello")])
        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1
        gen = result.generations[0]
        assert isinstance(gen, ChatGeneration)
        assert isinstance(gen.message, AIMessage)
        assert gen.message.content == "simple response"

    async def test_agenerate_falls_back_to_executor(self) -> None:
        """``SimpleChatModel._agenerate`` should fall back to running _generate."""
        model = _MinimalSimpleChatModel()
        result = await model._agenerate([HumanMessage(content="hello")])
        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1
        assert result.generations[0].message.content == "simple response"


class TestFormatForTracing:
    """Tests for ``_format_for_tracing``."""

    def test_image_block_converted_to_openai_format(self) -> None:
        """Image content blocks in v0/v1 data format get converted for tracing."""
        msg = HumanMessage(
            content=[
                {
                    "type": "image",
                    "source_type": "url",
                    "url": "https://example.com/img.png",
                }
            ]
        )
        traced = _format_for_tracing([msg])
        assert len(traced) == 1
        block = traced[0].content[0]
        assert isinstance(block, dict)
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == "https://example.com/img.png"

    def test_single_key_dict_gets_type_added(self) -> None:
        """Content blocks with a single key and no ``type`` get ``type`` added."""
        msg = HumanMessage(
            content=[
                {"cacheControl": {"kind": "ephemeral"}},
            ]
        )
        traced = _format_for_tracing([msg])
        block = traced[0].content[0]
        assert isinstance(block, dict)
        assert block["type"] == "cacheControl"
        assert block["cacheControl"] == {"kind": "ephemeral"}

    def test_string_content_unchanged(self) -> None:
        """Messages with string content pass through unchanged."""
        msg = HumanMessage(content="plain text")
        traced = _format_for_tracing([msg])
        assert traced[0].content == "plain text"
        # Should be the exact same object when no changes needed
        assert traced[0] is msg

    def test_blocks_with_type_unchanged(self) -> None:
        """Content blocks that already have ``type`` are not modified."""
        msg = HumanMessage(content=[{"type": "text", "text": "hello"}])
        traced = _format_for_tracing([msg])
        # No transformation needed, so the original message is returned
        assert traced[0] is msg

    def test_file_base64_block_converted_for_tracing(self) -> None:
        """File content blocks with base64 get converted to v0 format for tracing."""
        msg = HumanMessage(
            content=[
                {
                    "type": "file",
                    "mime_type": "application/pdf",
                    "base64": "abc123",
                }
            ]
        )
        traced = _format_for_tracing([msg])
        block = traced[0].content[0]
        assert isinstance(block, dict)
        assert block.get("data") == "abc123"
        assert block.get("source_type") == "base64"
        assert "base64" not in block

    def test_multiple_messages_processed(self) -> None:
        """Multiple messages should all be processed."""
        msgs = [
            HumanMessage(content="text"),
            HumanMessage(content=[{"marker": {"val": 1}}]),
        ]
        traced = _format_for_tracing(msgs)
        assert len(traced) == 2
        # First message unchanged
        assert traced[0] is msgs[0]
        # Second message has single-key block with type added
        assert traced[1].content[0]["type"] == "marker"


class TestGenInfoAndMsgMetadata:
    """Tests for ``_gen_info_and_msg_metadata``."""

    def test_merges_generation_info_with_response_metadata(self) -> None:
        gen = ChatGeneration(
            message=AIMessage(
                content="hi",
                response_metadata={"model": "test"},
            ),
            generation_info={"finish_reason": "stop"},
        )
        result = _gen_info_and_msg_metadata(gen)
        assert result["finish_reason"] == "stop"
        assert result["model"] == "test"

    def test_empty_generation_info(self) -> None:
        gen = ChatGeneration(
            message=AIMessage(content="hi", response_metadata={"key": "val"}),
        )
        result = _gen_info_and_msg_metadata(gen)
        assert result == {"key": "val"}

    def test_empty_response_metadata(self) -> None:
        gen = ChatGeneration(
            message=AIMessage(content="hi"),
            generation_info={"token_count": 10},
        )
        result = _gen_info_and_msg_metadata(gen)
        assert result == {"token_count": 10}

    def test_both_empty(self) -> None:
        gen = ChatGeneration(message=AIMessage(content="hi"))
        result = _gen_info_and_msg_metadata(gen)
        assert result == {}

    def test_works_with_chunk(self) -> None:
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="hi", response_metadata={"rm": 1}),
            generation_info={"gi": 2},
        )
        result = _gen_info_and_msg_metadata(chunk)
        assert result == {"gi": 2, "rm": 1}

    def test_response_metadata_overrides_generation_info(self) -> None:
        """When keys overlap, response_metadata values win (dict merge order)."""
        gen = ChatGeneration(
            message=AIMessage(
                content="",
                response_metadata={"key": "from_metadata"},
            ),
            generation_info={"key": "from_gen_info"},
        )
        result = _gen_info_and_msg_metadata(gen)
        assert result["key"] == "from_metadata"


class TestCleanupLlmRepresentation:
    """Tests for ``_cleanup_llm_representation``."""

    def test_removes_repr_from_not_implemented(self) -> None:
        serialized = {
            "type": "not_implemented",
            "repr": "<some repr string>",
        }
        _cleanup_llm_representation(serialized, 1)
        assert "repr" not in serialized
        assert serialized["type"] == "not_implemented"

    def test_removes_graph(self) -> None:
        serialized = {
            "type": "constructor",
            "graph": {"nodes": [], "edges": []},
            "kwargs": {},
        }
        _cleanup_llm_representation(serialized, 1)
        assert "graph" not in serialized

    def test_recursive_cleanup_in_kwargs(self) -> None:
        serialized = {
            "kwargs": {
                "nested": {
                    "type": "not_implemented",
                    "repr": "remove me",
                    "graph": {"stuff": True},
                }
            }
        }
        _cleanup_llm_representation(serialized, 1)
        nested = serialized["kwargs"]["nested"]
        assert "repr" not in nested
        assert "graph" not in nested

    def test_non_dict_input_is_noop(self) -> None:
        """Non-dict inputs are silently ignored."""
        _cleanup_llm_representation("not a dict", 1)
        _cleanup_llm_representation(42, 1)
        _cleanup_llm_representation(None, 1)

    def test_max_depth_guard(self) -> None:
        """Should bail out at max depth (no error, just stops recursing)."""
        deeply_nested: dict[str, Any] = {"type": "not_implemented", "repr": "keep"}
        _cleanup_llm_representation(deeply_nested, 200)
        # At depth > 100, the function returns early without modification
        assert "repr" in deeply_nested

    def test_normal_dict_without_special_keys(self) -> None:
        """A dict without ``repr``/``graph``/``not_implemented`` stays unchanged."""
        serialized = {"type": "constructor", "kwargs": {"a": 1}}
        _cleanup_llm_representation(serialized, 1)
        assert serialized == {"type": "constructor", "kwargs": {"a": 1}}


class TestShouldStream:
    """Tests for ``BaseChatModel._should_stream``."""

    def test_no_stream_implemented_returns_false(self) -> None:
        """If ``_stream`` is not overridden, should return False for sync."""
        model = NoStreamingModel()
        assert model._should_stream(async_api=False) is False

    def test_no_stream_or_astream_returns_false_for_async(self) -> None:
        """If neither ``_stream`` nor ``_astream`` is overridden, False for async."""
        model = NoStreamingModel()
        assert model._should_stream(async_api=True) is False

    def test_stream_implemented_with_streaming_callback(self) -> None:
        """With ``_stream`` overridden and a streaming callback, should be True."""
        model = StreamingModel()
        handler = LogStreamCallbackHandler()
        # Create a mock run_manager-like object that has handlers
        from unittest.mock import MagicMock

        run_manager = MagicMock()
        run_manager.handlers = [handler]
        assert model._should_stream(async_api=False, run_manager=run_manager) is True

    def test_disable_streaming_true_returns_false(self) -> None:
        """``disable_streaming=True`` should always return False."""
        model = StreamingModel(disable_streaming=True)
        assert model._should_stream(async_api=False) is False

    def test_disable_streaming_tool_calling_with_tools(self) -> None:
        """``disable_streaming='tool_calling'`` with tools kwarg returns False."""
        model = StreamingModel(disable_streaming="tool_calling")
        assert model._should_stream(async_api=False, tools=[{"type": "f"}]) is False

    def test_disable_streaming_tool_calling_without_tools(self) -> None:
        """``disable_streaming='tool_calling'`` without tools, respects other logic."""
        model = StreamingModel(disable_streaming="tool_calling")
        # No tools, no streaming callback, no stream kwarg -> depends on handlers
        assert model._should_stream(async_api=False) is False

    def test_stream_kwarg_true(self) -> None:
        """Explicit ``stream=True`` kwarg should return True."""
        model = StreamingModel()
        assert model._should_stream(async_api=False, stream=True) is True

    def test_stream_kwarg_false(self) -> None:
        """Explicit ``stream=False`` kwarg should return False."""
        model = StreamingModel()
        assert model._should_stream(async_api=False, stream=False) is False

    def test_streaming_attribute_set(self) -> None:
        """When ``streaming`` field is explicitly set, it should be respected."""
        model = StreamingModel(streaming=True)
        assert model._should_stream(async_api=False) is True

    def test_streaming_attribute_false(self) -> None:
        """When ``streaming`` is explicitly set to False."""
        model = StreamingModel(streaming=False)
        assert model._should_stream(async_api=False) is False

    def test_no_handlers_no_streaming(self) -> None:
        """With no streaming callbacks and no stream kwarg, returns False."""
        model = StreamingModel()
        assert model._should_stream(async_api=False) is False

    def test_async_with_sync_stream_only(self) -> None:
        """If only ``_stream`` is overridden (not ``_astream``),
        async should still be able to stream (falls back to sync)."""
        model = StreamingModel()
        assert model._should_stream(async_api=True, stream=True) is True


class TestBindTools:
    """Tests for ``BaseChatModel.bind_tools``."""

    def test_raises_not_implemented_by_default(self) -> None:
        """Base ``bind_tools`` should raise ``NotImplementedError``."""
        model = FakeListChatModel(responses=["hi"])
        with pytest.raises(NotImplementedError):
            model.bind_tools([{"type": "function", "function": {"name": "f"}}])


class TestWithStructuredOutput:
    """Tests for ``BaseChatModel.with_structured_output``."""

    def test_raises_not_implemented_when_bind_tools_not_overridden(self) -> None:
        """Should raise ``NotImplementedError`` when bind_tools is the base impl."""
        model = FakeListChatModel(responses=["hi"])
        with pytest.raises(NotImplementedError, match="not implemented"):
            model.with_structured_output({"type": "object", "properties": {}})


class TestCombineLlmOutputs:
    """Tests for ``BaseChatModel._combine_llm_outputs``."""

    def test_returns_empty_dict_by_default(self) -> None:
        model = FakeListChatModel(responses=["x"])
        result = model._combine_llm_outputs([{"token_usage": 10}, None])
        assert result == {}

    def test_returns_empty_dict_with_empty_list(self) -> None:
        model = FakeListChatModel(responses=["x"])
        result = model._combine_llm_outputs([])
        assert result == {}


class TestConvertCachedGenerations:
    """Tests for ``BaseChatModel._convert_cached_generations``."""

    def test_with_chat_generation_objects(self) -> None:
        """``ChatGeneration`` objects should pass through."""
        model = FakeListChatModel(responses=["x"])
        chat_gen = ChatGeneration(message=AIMessage(content="cached"))
        result = model._convert_cached_generations([chat_gen])
        assert len(result) == 1
        assert isinstance(result[0], ChatGeneration)
        assert result[0].message.content == "cached"

    def test_with_legacy_generation_objects(self) -> None:
        """Plain ``Generation`` objects (legacy) should be converted."""
        model = FakeListChatModel(responses=["x"])
        legacy_gen = Generation(text="old cached")
        result = model._convert_cached_generations([legacy_gen])
        assert len(result) == 1
        assert isinstance(result[0], ChatGeneration)
        assert isinstance(result[0].message, AIMessage)
        assert result[0].message.content == "old cached"

    def test_with_mixed_generation_objects(self) -> None:
        """Mix of ``ChatGeneration`` and ``Generation`` objects."""
        model = FakeListChatModel(responses=["x"])
        chat_gen = ChatGeneration(message=AIMessage(content="chat"))
        legacy_gen = Generation(text="legacy")
        result = model._convert_cached_generations([chat_gen, legacy_gen])
        assert len(result) == 2
        assert result[0].message.content == "chat"
        assert isinstance(result[1], ChatGeneration)
        assert result[1].message.content == "legacy"

    def test_preserves_generation_info_on_legacy(self) -> None:
        """``generation_info`` from legacy ``Generation`` is preserved."""
        model = FakeListChatModel(responses=["x"])
        legacy_gen = Generation(text="info", generation_info={"finish_reason": "stop"})
        result = model._convert_cached_generations([legacy_gen])
        assert result[0].generation_info == {"finish_reason": "stop"}

    def test_zeroes_out_total_cost_on_cache_hit(self) -> None:
        """Cached ``ChatGeneration`` with ``usage_metadata`` gets total_cost=0."""
        model = FakeListChatModel(responses=["x"])
        chat_gen = ChatGeneration(
            message=AIMessage(
                content="cached",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "total_cost": 0.005,
                },
            )
        )
        result = model._convert_cached_generations([chat_gen])
        assert result[0].message.usage_metadata["total_cost"] == 0


class TestGenerateMethod:
    """Tests for ``BaseChatModel.generate``."""

    def test_single_message_list(self) -> None:
        """Should handle a single list of messages."""
        model = FakeListChatModel(responses=["response1", "response2"])
        result = model.generate([[HumanMessage(content="hello")]])
        assert isinstance(result, LLMResult)
        assert len(result.generations) == 1
        gen = result.generations[0][0]
        assert isinstance(gen, ChatGeneration)
        assert gen.message.content == "response1"

    def test_multiple_message_lists(self) -> None:
        """Should handle multiple lists of messages."""
        model = FakeListChatModel(responses=["a", "b", "c"])
        result = model.generate(
            [
                [HumanMessage(content="first")],
                [HumanMessage(content="second")],
            ]
        )
        assert isinstance(result, LLMResult)
        assert len(result.generations) == 2
        assert result.generations[0][0].message.content == "a"
        assert result.generations[1][0].message.content == "b"

    def test_generate_returns_run_info(self) -> None:
        """Run info should be populated when callbacks are used."""
        model = FakeListChatModel(responses=["hi"])
        cb = FakeTracer()
        result = model.generate(
            [[HumanMessage(content="hello")]],
            callbacks=[cb],
        )
        assert result.run is not None
        assert len(result.run) == 1

    def test_generate_combines_llm_output(self) -> None:
        """``llm_output`` should be the result of ``_combine_llm_outputs``."""
        model = FakeListChatModel(responses=["x"])
        result = model.generate([[HumanMessage(content="hi")]])
        # _combine_llm_outputs returns {} by default
        assert result.llm_output == {}


class TestAGenerateMethod:
    """Tests for ``BaseChatModel.agenerate``."""

    async def test_single_message_list(self) -> None:
        model = FakeListChatModel(responses=["async_response"])
        result = await model.agenerate([[HumanMessage(content="hello")]])
        assert isinstance(result, LLMResult)
        assert len(result.generations) == 1
        assert result.generations[0][0].message.content == "async_response"

    async def test_multiple_message_lists(self) -> None:
        model = FakeListChatModel(responses=["a", "b", "c"])
        result = await model.agenerate(
            [
                [HumanMessage(content="first")],
                [HumanMessage(content="second")],
            ]
        )
        assert isinstance(result, LLMResult)
        assert len(result.generations) == 2

    async def test_agenerate_returns_run_info(self) -> None:
        model = FakeListChatModel(responses=["hi"])
        cb = FakeTracer()
        result = await model.agenerate(
            [[HumanMessage(content="hello")]],
            callbacks=[cb],
        )
        assert result.run is not None
        assert len(result.run) == 1


class TestFormatLsStructuredOutput:
    """Tests for ``_format_ls_structured_output``."""

    def test_with_none(self) -> None:
        result = _format_ls_structured_output(None)
        assert result == {}

    def test_with_valid_pydantic_schema(self) -> None:
        """Providing a valid Pydantic class as schema should produce valid output."""
        from pydantic import BaseModel as PydanticBaseModel

        class MySchema(PydanticBaseModel):
            answer: str
            score: int

        fmt = {"schema": MySchema, "kwargs": {"method": "function_calling"}}
        result = _format_ls_structured_output(fmt)
        assert "ls_structured_output_format" in result
        inner = result["ls_structured_output_format"]
        assert inner["kwargs"] == {"method": "function_calling"}
        assert "properties" in inner["schema"]
        assert "answer" in inner["schema"]["properties"]
        assert "score" in inner["schema"]["properties"]

    def test_with_empty_dict(self) -> None:
        """An empty dict (falsy) should return empty."""
        result = _format_ls_structured_output({})
        assert result == {}

    def test_with_invalid_schema_returns_empty(self) -> None:
        """If ``convert_to_json_schema`` raises ValueError, returns empty dict."""
        result = _format_ls_structured_output(
            {"schema": "not_a_valid_schema", "kwargs": {}}
        )
        assert result == {}


class TestSimpleChatModelFakeChatModel:
    """Tests for ``FakeChatModel`` (a ``SimpleChatModel`` subclass)."""

    def test_generate_returns_chat_result(self) -> None:
        model = FakeChatModel()
        result = model._generate([HumanMessage(content="hello")])
        assert isinstance(result, ChatResult)
        assert result.generations[0].message.content == "fake response"

    async def test_agenerate_returns_chat_result(self) -> None:
        """FakeChatModel overrides _agenerate directly."""
        model = FakeChatModel()
        result = await model._agenerate([HumanMessage(content="hello")])
        assert isinstance(result, ChatResult)
        assert result.generations[0].message.content == "fake response"
