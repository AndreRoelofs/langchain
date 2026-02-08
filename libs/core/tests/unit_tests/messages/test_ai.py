from typing import cast

from langchain_core.load import dumpd, load
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
    add_ai_message_chunks,
    add_usage,
    subtract_usage,
)
from langchain_core.messages.tool import invalid_tool_call as create_invalid_tool_call
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk


def test_serdes_message() -> None:
    msg = AIMessage(
        content=[{"text": "blah", "type": "text"}],
        tool_calls=[create_tool_call(name="foo", args={"bar": 1}, id="baz")],
        invalid_tool_calls=[
            create_invalid_tool_call(name="foobad", args="blah", id="booz", error="bad")
        ],
    )
    expected = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "type": "ai",
            "content": [{"text": "blah", "type": "text"}],
            "tool_calls": [
                {"name": "foo", "args": {"bar": 1}, "id": "baz", "type": "tool_call"}
            ],
            "invalid_tool_calls": [
                {
                    "name": "foobad",
                    "args": "blah",
                    "id": "booz",
                    "error": "bad",
                    "type": "invalid_tool_call",
                }
            ],
        },
    }
    actual = dumpd(msg)
    assert actual == expected
    assert load(actual) == msg


def test_serdes_message_chunk() -> None:
    chunk = AIMessageChunk(
        content=[{"text": "blah", "type": "text"}],
        tool_call_chunks=[
            create_tool_call_chunk(name="foo", args='{"bar": 1}', id="baz", index=0),
            create_tool_call_chunk(
                name="foobad",
                args="blah",
                id="booz",
                index=1,
            ),
        ],
    )
    expected = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", "AIMessageChunk"],
        "kwargs": {
            "type": "AIMessageChunk",
            "content": [{"text": "blah", "type": "text"}],
            "tool_calls": [
                {"name": "foo", "args": {"bar": 1}, "id": "baz", "type": "tool_call"}
            ],
            "invalid_tool_calls": [
                {
                    "name": "foobad",
                    "args": "blah",
                    "id": "booz",
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
            "tool_call_chunks": [
                {
                    "name": "foo",
                    "args": '{"bar": 1}',
                    "id": "baz",
                    "index": 0,
                    "type": "tool_call_chunk",
                },
                {
                    "name": "foobad",
                    "args": "blah",
                    "id": "booz",
                    "index": 1,
                    "type": "tool_call_chunk",
                },
            ],
        },
    }
    actual = dumpd(chunk)
    assert actual == expected
    assert load(actual) == chunk


def test_add_usage_both_none() -> None:
    result = add_usage(None, None)
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_add_usage_one_none() -> None:
    usage = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
    result = add_usage(usage, None)
    assert result == usage


def test_add_usage_both_present() -> None:
    usage1 = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
    usage2 = UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15)
    result = add_usage(usage1, usage2)
    assert result == UsageMetadata(input_tokens=15, output_tokens=30, total_tokens=45)


def test_add_usage_with_details() -> None:
    usage1 = UsageMetadata(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        input_token_details=InputTokenDetails(audio=5),
        output_token_details=OutputTokenDetails(reasoning=10),
    )
    usage2 = UsageMetadata(
        input_tokens=5,
        output_tokens=10,
        total_tokens=15,
        input_token_details=InputTokenDetails(audio=3),
        output_token_details=OutputTokenDetails(reasoning=5),
    )
    result = add_usage(usage1, usage2)
    assert result["input_token_details"]["audio"] == 8
    assert result["output_token_details"]["reasoning"] == 15


def test_subtract_usage_both_none() -> None:
    result = subtract_usage(None, None)
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_subtract_usage_one_none() -> None:
    usage = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
    result = subtract_usage(usage, None)
    assert result == usage


def test_subtract_usage_both_present() -> None:
    usage1 = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
    usage2 = UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15)
    result = subtract_usage(usage1, usage2)
    assert result == UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15)


def test_subtract_usage_with_negative_result() -> None:
    usage1 = UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15)
    usage2 = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
    result = subtract_usage(usage1, usage2)
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_add_ai_message_chunks_usage() -> None:
    chunks = [
        AIMessageChunk(content="", usage_metadata=None),
        AIMessageChunk(
            content="",
            usage_metadata=UsageMetadata(
                input_tokens=2, output_tokens=3, total_tokens=5
            ),
        ),
        AIMessageChunk(
            content="",
            usage_metadata=UsageMetadata(
                input_tokens=2,
                output_tokens=3,
                total_tokens=5,
                input_token_details=InputTokenDetails(audio=1, cache_read=1),
                output_token_details=OutputTokenDetails(audio=1, reasoning=2),
            ),
        ),
    ]
    combined = add_ai_message_chunks(*chunks)
    assert combined == AIMessageChunk(
        content="",
        usage_metadata=UsageMetadata(
            input_tokens=4,
            output_tokens=6,
            total_tokens=10,
            input_token_details=InputTokenDetails(audio=1, cache_read=1),
            output_token_details=OutputTokenDetails(audio=1, reasoning=2),
        ),
    )


def test_init_tool_calls() -> None:
    # Test we add "type" key on init
    msg = AIMessage("", tool_calls=[{"name": "foo", "args": {"a": "b"}, "id": "abc"}])
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["type"] == "tool_call"

    # Test we can assign without adding type key
    msg.tool_calls = [{"name": "bar", "args": {"c": "d"}, "id": "def"}]


def test_content_blocks() -> None:
    message = AIMessage(
        "",
        tool_calls=[
            {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"}
        ],
    )
    assert len(message.content_blocks) == 1
    assert message.content_blocks[0]["type"] == "tool_call"
    assert message.content_blocks == [
        {"type": "tool_call", "id": "abc_123", "name": "foo", "args": {"a": "b"}}
    ]
    assert message.content == ""

    message = AIMessage(
        "foo",
        tool_calls=[
            {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"}
        ],
    )
    assert len(message.content_blocks) == 2
    assert message.content_blocks[0]["type"] == "text"
    assert message.content_blocks[1]["type"] == "tool_call"
    assert message.content_blocks == [
        {"type": "text", "text": "foo"},
        {"type": "tool_call", "id": "abc_123", "name": "foo", "args": {"a": "b"}},
    ]
    assert message.content == "foo"

    # With standard blocks
    standard_content: list[types.ContentBlock] = [
        {"type": "reasoning", "reasoning": "foo"},
        {"type": "text", "text": "bar"},
        {
            "type": "text",
            "text": "baz",
            "annotations": [{"type": "citation", "url": "http://example.com"}],
        },
        {
            "type": "image",
            "url": "http://example.com/image.png",
            "extras": {"foo": "bar"},
        },
        {
            "type": "non_standard",
            "value": {"custom_key": "custom_value", "another_key": 123},
        },
        {
            "type": "tool_call",
            "name": "foo",
            "args": {"a": "b"},
            "id": "abc_123",
        },
    ]
    missing_tool_call: types.ToolCall = {
        "type": "tool_call",
        "name": "bar",
        "args": {"c": "d"},
        "id": "abc_234",
    }
    message = AIMessage(
        content_blocks=standard_content,
        tool_calls=[
            {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"},
            missing_tool_call,
        ],
    )
    assert message.content_blocks == [*standard_content, missing_tool_call]

    # Check we auto-populate tool_calls
    standard_content = [
        {"type": "text", "text": "foo"},
        {
            "type": "tool_call",
            "name": "foo",
            "args": {"a": "b"},
            "id": "abc_123",
        },
        missing_tool_call,
    ]
    message = AIMessage(content_blocks=standard_content)
    assert message.tool_calls == [
        {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"},
        missing_tool_call,
    ]

    # Chunks
    message = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "foo",
                "args": "",
                "id": "abc_123",
                "index": 0,
            }
        ],
    )
    assert len(message.content_blocks) == 1
    assert message.content_blocks[0]["type"] == "tool_call_chunk"
    assert message.content_blocks == [
        {
            "type": "tool_call_chunk",
            "name": "foo",
            "args": "",
            "id": "abc_123",
            "index": 0,
        }
    ]
    assert message.content == ""

    # Test we parse tool call chunks into tool calls for v1 content
    chunk_1 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "foo",
                "args": '{"foo": "b',
                "id": "abc_123",
                "index": 0,
            }
        ],
    )

    chunk_2 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "",
                "args": 'ar"}',
                "id": "abc_123",
                "index": 0,
            }
        ],
    )
    chunk_3 = AIMessageChunk(content="", chunk_position="last")
    chunk = chunk_1 + chunk_2 + chunk_3
    assert chunk.content == ""
    assert chunk.content_blocks == chunk.tool_calls

    # test v1 content
    chunk_1.content = cast("str | list[str | dict]", chunk_1.content_blocks)
    assert len(chunk_1.content) == 1
    chunk_1.content[0]["extras"] = {"baz": "qux"}  # type: ignore[index]
    chunk_1.response_metadata["output_version"] = "v1"
    chunk_2.content = cast("str | list[str | dict]", chunk_2.content_blocks)

    chunk = chunk_1 + chunk_2 + chunk_3
    assert chunk.content == [
        {
            "type": "tool_call",
            "name": "foo",
            "args": {"foo": "bar"},
            "id": "abc_123",
            "extras": {"baz": "qux"},
        }
    ]

    # Non-standard
    standard_content_1: list[types.ContentBlock] = [
        {"type": "non_standard", "index": 0, "value": {"foo": "bar "}}
    ]
    standard_content_2: list[types.ContentBlock] = [
        {"type": "non_standard", "index": 0, "value": {"foo": "baz"}}
    ]
    chunk_1 = AIMessageChunk(content=cast("str | list[str | dict]", standard_content_1))
    chunk_2 = AIMessageChunk(content=cast("str | list[str | dict]", standard_content_2))
    merged_chunk = chunk_1 + chunk_2
    assert merged_chunk.content == [
        {"type": "non_standard", "index": 0, "value": {"foo": "bar baz"}},
    ]

    # Test server_tool_call_chunks
    chunk_1 = AIMessageChunk(
        content=[
            {
                "type": "server_tool_call_chunk",
                "index": 0,
                "name": "foo",
            }
        ]
    )
    chunk_2 = AIMessageChunk(
        content=[{"type": "server_tool_call_chunk", "index": 0, "args": '{"a'}]
    )
    chunk_3 = AIMessageChunk(
        content=[{"type": "server_tool_call_chunk", "index": 0, "args": '": 1}'}]
    )
    merged_chunk = chunk_1 + chunk_2 + chunk_3
    assert merged_chunk.content == [
        {
            "type": "server_tool_call_chunk",
            "name": "foo",
            "index": 0,
            "args": '{"a": 1}',
        }
    ]

    full_chunk = merged_chunk + AIMessageChunk(
        content=[], chunk_position="last", response_metadata={"output_version": "v1"}
    )
    assert full_chunk.content == [
        {"type": "server_tool_call", "name": "foo", "index": 0, "args": {"a": 1}}
    ]

    # Test non-standard + non-standard
    chunk_1 = AIMessageChunk(
        content=[
            {
                "type": "non_standard",
                "index": 0,
                "value": {"type": "non_standard_tool", "foo": "bar"},
            }
        ]
    )
    chunk_2 = AIMessageChunk(
        content=[
            {
                "type": "non_standard",
                "index": 0,
                "value": {"type": "input_json_delta", "partial_json": "a"},
            }
        ]
    )
    chunk_3 = AIMessageChunk(
        content=[
            {
                "type": "non_standard",
                "index": 0,
                "value": {"type": "input_json_delta", "partial_json": "b"},
            }
        ]
    )
    merged_chunk = chunk_1 + chunk_2 + chunk_3
    assert merged_chunk.content == [
        {
            "type": "non_standard",
            "index": 0,
            "value": {"type": "non_standard_tool", "foo": "bar", "partial_json": "ab"},
        }
    ]

    # Test standard + non-standard with same index
    standard_content_1 = [
        {
            "type": "server_tool_call",
            "name": "web_search",
            "id": "ws_123",
            "args": {"query": "web query"},
            "index": 0,
        }
    ]
    standard_content_2 = [{"type": "non_standard", "value": {"foo": "bar"}, "index": 0}]
    chunk_1 = AIMessageChunk(content=cast("str | list[str | dict]", standard_content_1))
    chunk_2 = AIMessageChunk(content=cast("str | list[str | dict]", standard_content_2))
    merged_chunk = chunk_1 + chunk_2
    assert merged_chunk.content == [
        {
            "type": "server_tool_call",
            "name": "web_search",
            "id": "ws_123",
            "args": {"query": "web query"},
            "index": 0,
            "extras": {"foo": "bar"},
        }
    ]


def test_content_blocks_reasoning_extraction() -> None:
    """Test best-effort reasoning extraction from `additional_kwargs`."""
    message = AIMessage(
        content="The answer is 42.",
        additional_kwargs={"reasoning_content": "Let me think about this problem..."},
    )
    content_blocks = message.content_blocks
    assert len(content_blocks) == 2
    assert content_blocks[0]["type"] == "reasoning"
    assert content_blocks[0].get("reasoning") == "Let me think about this problem..."
    assert content_blocks[1]["type"] == "text"
    assert content_blocks[1]["text"] == "The answer is 42."

    # Test no reasoning extraction when no reasoning content
    message = AIMessage(
        content="The answer is 42.", additional_kwargs={"other_field": "some value"}
    )
    content_blocks = message.content_blocks
    assert len(content_blocks) == 1
    assert content_blocks[0]["type"] == "text"


# ---------------------------------------------------------------------------
# New tests covering additional AIMessage / AIMessageChunk functionality
# ---------------------------------------------------------------------------


def test_ai_message_type_field() -> None:
    """AIMessage type field is always 'ai'."""
    msg = AIMessage(content="hello")
    assert msg.type == "ai"

    msg_with_tools = AIMessage(
        content="",
        tool_calls=[create_tool_call(name="t", args={}, id="1")],
    )
    assert msg_with_tools.type == "ai"


def test_ai_message_lc_attributes_returns_tool_calls_and_invalid_tool_calls() -> None:
    """lc_attributes property includes both tool_calls and invalid_tool_calls."""
    msg = AIMessage(
        content="hi",
        tool_calls=[create_tool_call(name="foo", args={"x": 1}, id="tc1")],
        invalid_tool_calls=[
            create_invalid_tool_call(name="bad", args="nope", id="itc1", error="err")
        ],
    )
    attrs = msg.lc_attributes
    assert "tool_calls" in attrs
    assert "invalid_tool_calls" in attrs
    assert len(attrs["tool_calls"]) == 1
    assert attrs["tool_calls"][0]["name"] == "foo"
    assert attrs["tool_calls"][0]["args"] == {"x": 1}
    assert attrs["tool_calls"][0]["id"] == "tc1"
    assert len(attrs["invalid_tool_calls"]) == 1
    assert attrs["invalid_tool_calls"][0]["name"] == "bad"
    assert attrs["invalid_tool_calls"][0]["args"] == "nope"
    assert attrs["invalid_tool_calls"][0]["id"] == "itc1"
    assert attrs["invalid_tool_calls"][0]["error"] == "err"

    # Empty case
    empty_msg = AIMessage(content="nothing")
    assert empty_msg.lc_attributes == {
        "tool_calls": [],
        "invalid_tool_calls": [],
    }


def test_ai_message_pretty_repr_with_tool_calls() -> None:
    """pretty_repr includes a 'Tool Calls:' section for tool calls."""
    msg = AIMessage(
        content="Sure, let me call that tool.",
        tool_calls=[
            create_tool_call(name="get_weather", args={"city": "SF"}, id="call_1"),
        ],
    )
    result = msg.pretty_repr()
    assert "Tool Calls:" in result
    assert "get_weather (call_1)" in result
    assert "Call ID: call_1" in result
    assert "Args:" in result
    assert "city: SF" in result


def test_ai_message_pretty_repr_with_invalid_tool_calls() -> None:
    """pretty_repr includes an 'Invalid Tool Calls:' section for invalid calls."""
    msg = AIMessage(
        content="",
        invalid_tool_calls=[
            create_invalid_tool_call(
                name="broken", args="not json", id="call_bad", error="parse error"
            )
        ],
    )
    result = msg.pretty_repr()
    assert "Invalid Tool Calls:" in result
    assert "broken (call_bad)" in result
    assert "Call ID: call_bad" in result
    assert "Error: parse error" in result
    assert "Args:" in result
    assert "not json" in result


def test_ai_message_pretty_repr_with_string_args() -> None:
    """pretty_repr handles tool calls whose args are a raw string."""
    msg = AIMessage(
        content="",
        invalid_tool_calls=[
            create_invalid_tool_call(
                name="mytool", args="raw string args", id="id1", error=None
            )
        ],
    )
    result = msg.pretty_repr()
    assert "Invalid Tool Calls:" in result
    assert "raw string args" in result


def test_ai_message_init_with_usage_metadata() -> None:
    """AIMessage can be initialized with usage_metadata."""
    usage = UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)
    msg = AIMessage(content="hello", usage_metadata=usage)
    assert msg.usage_metadata is not None
    assert msg.usage_metadata["input_tokens"] == 10
    assert msg.usage_metadata["output_tokens"] == 5
    assert msg.usage_metadata["total_tokens"] == 15

    # None by default
    msg_no_usage = AIMessage(content="hi")
    assert msg_no_usage.usage_metadata is None


def test_ai_message_serdes_with_usage_metadata() -> None:
    """Serialization roundtrip preserves usage_metadata."""
    usage = UsageMetadata(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        input_token_details=InputTokenDetails(cache_read=20),
        output_token_details=OutputTokenDetails(reasoning=10),
    )
    msg = AIMessage(content="result", usage_metadata=usage)
    dumped = dumpd(msg)
    loaded = load(dumped)
    assert loaded.usage_metadata == usage
    assert loaded.usage_metadata["input_token_details"]["cache_read"] == 20
    assert loaded.usage_metadata["output_token_details"]["reasoning"] == 10
    assert loaded.content == "result"


def test_backwards_compat_tool_calls_from_additional_kwargs() -> None:
    """When tool calls live in additional_kwargs (OpenAI format), they are parsed."""
    msg = AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_abc",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "langchain"}',
                    },
                    "type": "function",
                }
            ]
        },
    )
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["name"] == "search"
    assert msg.tool_calls[0]["args"] == {"query": "langchain"}
    assert msg.tool_calls[0]["id"] == "call_abc"
    assert msg.tool_calls[0]["type"] == "tool_call"
    assert len(msg.invalid_tool_calls) == 0


def test_backwards_compat_invalid_json_becomes_invalid_tool_calls() -> None:
    """When additional_kwargs has tool calls with invalid JSON, they go to invalid_tool_calls."""
    msg = AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_xyz",
                    "function": {
                        "name": "broken_tool",
                        "arguments": "this is not json{{{",
                    },
                    "type": "function",
                }
            ]
        },
    )
    assert len(msg.tool_calls) == 0
    assert len(msg.invalid_tool_calls) == 1
    assert msg.invalid_tool_calls[0]["name"] == "broken_tool"
    assert msg.invalid_tool_calls[0]["args"] == "this is not json{{{"
    assert msg.invalid_tool_calls[0]["id"] == "call_xyz"
    assert msg.invalid_tool_calls[0]["type"] == "invalid_tool_call"


def test_ai_message_chunk_type_field() -> None:
    """AIMessageChunk type field is 'AIMessageChunk'."""
    chunk = AIMessageChunk(content="hi")
    assert chunk.type == "AIMessageChunk"


def test_ai_message_chunk_chunk_position_field() -> None:
    """AIMessageChunk supports a chunk_position field."""
    chunk = AIMessageChunk(content="done", chunk_position="last")
    assert chunk.chunk_position == "last"

    chunk_none = AIMessageChunk(content="partial")
    assert chunk_none.chunk_position is None


def test_init_tool_calls_populates_tool_call_chunks_from_tool_calls() -> None:
    """When AIMessageChunk has tool_calls but no tool_call_chunks, chunks are created."""
    chunk = AIMessageChunk(
        content="",
        tool_calls=[
            create_tool_call(name="my_tool", args={"key": "val"}, id="tc_1"),
        ],
    )
    assert len(chunk.tool_call_chunks) == 1
    assert chunk.tool_call_chunks[0]["name"] == "my_tool"
    assert chunk.tool_call_chunks[0]["args"] == '{"key": "val"}'
    assert chunk.tool_call_chunks[0]["id"] == "tc_1"
    assert chunk.tool_call_chunks[0]["index"] is None
    assert chunk.tool_call_chunks[0]["type"] == "tool_call_chunk"


def test_init_tool_calls_populates_tool_call_chunks_from_invalid_tool_calls() -> None:
    """When AIMessageChunk has invalid_tool_calls but no tool_call_chunks, chunks are created."""
    chunk = AIMessageChunk(
        content="",
        invalid_tool_calls=[
            create_invalid_tool_call(
                name="bad_tool", args="bad args", id="itc_1", error="fail"
            ),
        ],
    )
    assert len(chunk.tool_call_chunks) == 1
    assert chunk.tool_call_chunks[0]["name"] == "bad_tool"
    assert chunk.tool_call_chunks[0]["args"] == "bad args"
    assert chunk.tool_call_chunks[0]["id"] == "itc_1"
    assert chunk.tool_call_chunks[0]["index"] is None
    assert chunk.tool_call_chunks[0]["type"] == "tool_call_chunk"


def test_ai_message_chunk_add_with_list_of_chunks() -> None:
    """AIMessageChunk __add__ supports adding a list or tuple of chunks."""
    base = AIMessageChunk(content="Hello")
    others = [
        AIMessageChunk(content=" world"),
        AIMessageChunk(content="!"),
    ]
    result = base + others
    assert isinstance(result, AIMessageChunk)
    assert result.content == "Hello world!"

    # Also works with a tuple
    result_tuple = base + tuple(others)
    assert isinstance(result_tuple, AIMessageChunk)
    assert result_tuple.content == "Hello world!"


def test_add_ai_message_chunks_id_priority() -> None:
    """ID priority: provider-assigned > lc_run > lc_ auto."""
    provider_id = "chatcmpl-abc123"
    run_id = "lc_run-some-uuid"
    auto_id = "lc_auto-generated-uuid"

    # Provider-assigned wins over lc_run and lc_ auto
    chunk_auto = AIMessageChunk(content="a", id=auto_id)
    chunk_run = AIMessageChunk(content="b", id=run_id)
    chunk_provider = AIMessageChunk(content="c", id=provider_id)

    result = add_ai_message_chunks(chunk_auto, chunk_run, chunk_provider)
    assert result.id == provider_id

    # lc_run wins over lc_ auto
    result2 = add_ai_message_chunks(chunk_auto, chunk_run)
    assert result2.id == run_id

    # Only auto IDs: first one is picked
    chunk_auto2 = AIMessageChunk(content="d", id="lc_other-uuid")
    result3 = add_ai_message_chunks(chunk_auto, chunk_auto2)
    assert result3.id == auto_id


def test_add_ai_message_chunks_chunk_position_propagation() -> None:
    """If any chunk has chunk_position='last', the result has 'last'."""
    chunk1 = AIMessageChunk(content="a")
    chunk2 = AIMessageChunk(content="b")
    chunk3 = AIMessageChunk(content="c", chunk_position="last")

    result = add_ai_message_chunks(chunk1, chunk2, chunk3)
    assert result.chunk_position == "last"

    # Without any 'last', result is None
    result_no_last = add_ai_message_chunks(chunk1, chunk2)
    assert result_no_last.chunk_position is None


def test_subtract_usage_with_details() -> None:
    """subtract_usage correctly subtracts InputTokenDetails and OutputTokenDetails."""
    usage1 = UsageMetadata(
        input_tokens=20,
        output_tokens=30,
        total_tokens=50,
        input_token_details=InputTokenDetails(audio=10, cache_read=8),
        output_token_details=OutputTokenDetails(audio=5, reasoning=15),
    )
    usage2 = UsageMetadata(
        input_tokens=5,
        output_tokens=10,
        total_tokens=15,
        input_token_details=InputTokenDetails(audio=3, cache_read=2),
        output_token_details=OutputTokenDetails(audio=2, reasoning=5),
    )
    result = subtract_usage(usage1, usage2)
    assert result["input_tokens"] == 15
    assert result["output_tokens"] == 20
    assert result["total_tokens"] == 35
    assert result["input_token_details"]["audio"] == 7
    assert result["input_token_details"]["cache_read"] == 6
    assert result["output_token_details"]["audio"] == 3
    assert result["output_token_details"]["reasoning"] == 10


def test_subtract_usage_right_none_returns_left() -> None:
    """subtract_usage with right=None returns left unchanged."""
    usage = UsageMetadata(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        input_token_details=InputTokenDetails(cache_read=5),
    )
    result = subtract_usage(usage, None)
    assert result == usage
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 20
    assert result["total_tokens"] == 30
    assert result["input_token_details"]["cache_read"] == 5


def test_ai_message_chunk_content_blocks_with_output_version_v1() -> None:
    """With output_version=v1 in response_metadata, content_blocks returns content directly."""
    raw_content: list[dict] = [
        {"type": "text", "text": "hello"},
        {"type": "tool_call", "name": "foo", "args": {"a": 1}, "id": "tc1"},
    ]
    chunk = AIMessageChunk(
        content=raw_content,
        response_metadata={"output_version": "v1"},
    )
    blocks = chunk.content_blocks
    # Should return content as-is, without any transformation
    assert blocks == raw_content
    assert blocks[0]["type"] == "text"
    assert blocks[1]["type"] == "tool_call"


def test_ai_message_chunk_content_blocks_reasoning_from_additional_kwargs() -> None:
    """AIMessageChunk content_blocks extracts reasoning from additional_kwargs."""
    chunk = AIMessageChunk(
        content="The result is 7.",
        additional_kwargs={"reasoning_content": "I need to compute 3 + 4."},
    )
    blocks = chunk.content_blocks
    assert len(blocks) == 2
    assert blocks[0]["type"] == "reasoning"
    assert blocks[0].get("reasoning") == "I need to compute 3 + 4."
    assert blocks[1]["type"] == "text"
    assert blocks[1]["text"] == "The result is 7."

    # No reasoning key -> no reasoning block
    chunk_no_reasoning = AIMessageChunk(
        content="Just text.",
        additional_kwargs={"something_else": "value"},
    )
    blocks_no = chunk_no_reasoning.content_blocks
    assert len(blocks_no) == 1
    assert blocks_no[0]["type"] == "text"
