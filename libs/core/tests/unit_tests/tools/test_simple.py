"""Unit tests for the simple Tool class."""

from collections.abc import Awaitable
from functools import partial

import pytest
from pydantic import BaseModel, Field

from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool, ToolException


def simple_func(x: str) -> str:
    """Simple test function."""
    return f"Result: {x}"


async def async_func(x: str) -> str:
    """Async test function."""
    return f"Async result: {x}"


def multi_arg_func(x: str, y: int) -> str:
    """Function with multiple arguments."""
    return f"{x}-{y}"


class SimpleSchema(BaseModel):
    """Simple schema for testing."""

    input_str: str = Field(description="The input string")


def test_tool_initialization_positional() -> None:
    """Test Tool initialization with positional arguments."""
    tool = Tool("test_tool", simple_func, "Test description")

    assert tool.name == "test_tool"
    assert tool.description == "Test description"
    assert tool.func == simple_func
    assert tool.is_single_input


def test_tool_initialization_keyword() -> None:
    """Test Tool initialization with keyword arguments."""
    tool = Tool(name="test_tool", func=simple_func, description="Test description")

    assert tool.name == "test_tool"
    assert tool.description == "Test description"
    assert tool.func == simple_func


def test_tool_from_function() -> None:
    """Test Tool.from_function factory method."""
    tool = Tool.from_function(
        func=simple_func,
        name="test_tool",
        description="Test description",
    )

    assert tool.name == "test_tool"
    assert tool.description == "Test description"
    assert tool.func == simple_func


def test_tool_from_function_with_coroutine() -> None:
    """Test Tool.from_function with async coroutine."""
    tool = Tool.from_function(
        func=simple_func,
        name="test_tool",
        description="Test description",
        coroutine=async_func,
    )

    assert tool.name == "test_tool"
    assert tool.coroutine == async_func


def test_tool_from_function_coroutine_only() -> None:
    """Test Tool.from_function with only coroutine provided."""
    tool = Tool.from_function(
        func=None,
        name="async_tool",
        description="Async only tool",
        coroutine=async_func,
    )

    assert tool.name == "async_tool"
    assert tool.coroutine == async_func
    assert tool.func is None


def test_tool_from_function_neither_provided_raises() -> None:
    """Test that Tool.from_function raises error when neither func nor coroutine provided."""
    with pytest.raises(ValueError, match="Function and/or coroutine must be provided"):
        Tool.from_function(
            func=None,
            name="invalid_tool",
            description="Invalid tool",
            coroutine=None,
        )


def test_tool_invoke_string_input() -> None:
    """Test tool invocation with string input."""
    tool = Tool(
        name="string_tool",
        func=simple_func,
        description="Test tool",
    )

    result = tool.invoke("test")
    assert result == "Result: test"


def test_tool_invoke_dict_input() -> None:
    """Test tool invocation with dict input (single-input tool)."""
    tool = Tool(
        name="string_tool",
        func=simple_func,
        description="Test tool",
    )

    # For simple tools, dict is converted to single arg
    result = tool.invoke({"x": "test"})
    assert result == "Result: test"


def test_tool_invoke_with_tool_call() -> None:
    """Test tool invocation with ToolCall input."""
    tool = Tool(
        name="test_tool",
        func=simple_func,
        description="Test tool",
    )

    tool_call: ToolCall = {
        "name": "test_tool",
        "args": {"x": "test"},
        "id": "call_123",
        "type": "tool_call",
    }

    result = tool.invoke(tool_call)

    assert isinstance(result, ToolMessage)
    assert result.content == "Result: test"
    assert result.tool_call_id == "call_123"
    assert result.name == "test_tool"


def test_tool_run_method() -> None:
    """Test tool.run method."""
    tool = Tool(
        name="test_tool",
        func=simple_func,
        description="Test tool",
    )

    # run() without tool_call_id returns the string result, not ToolMessage
    result = tool.run("test")
    assert result == "Result: test"


async def test_tool_async_invoke() -> None:
    """Test async tool invocation."""
    tool = Tool(
        name="async_tool",
        func=simple_func,
        description="Async tool",
        coroutine=async_func,
    )

    result = await tool.ainvoke("test")
    assert result == "Async result: test"


async def test_tool_async_invoke_without_coroutine_falls_back_to_sync() -> None:
    """Test that async invoke falls back to sync func when no coroutine."""
    tool = Tool(
        name="sync_tool",
        func=simple_func,
        description="Sync tool without coroutine",
    )

    result = await tool.ainvoke("test")
    assert result == "Result: test"


async def test_tool_arun_method() -> None:
    """Test tool.arun method."""
    tool = Tool(
        name="async_tool",
        func=simple_func,
        description="Async tool",
        coroutine=async_func,
    )

    # arun() without tool_call_id returns the string result, not ToolMessage
    result = await tool.arun("test")
    assert result == "Async result: test"


def test_tool_args_property_without_schema() -> None:
    """Test Tool.args property when no args_schema provided."""
    tool = Tool(
        name="test_tool",
        func=simple_func,
        description="Test tool",
    )

    args = tool.args
    assert args == {"tool_input": {"type": "string"}}


def test_tool_args_property_with_schema() -> None:
    """Test Tool.args property when args_schema is provided."""
    tool = Tool(
        name="test_tool",
        func=simple_func,
        description="Test tool",
        args_schema=SimpleSchema,
    )

    args = tool.args
    assert "input_str" in args
    assert args["input_str"]["type"] == "string"


def test_tool_with_lambda_func() -> None:
    """Test Tool with lambda function."""
    tool = Tool(
        name="lambda_tool",
        func=lambda x: f"Lambda: {x}",
        description="Lambda tool",
    )

    result = tool.invoke("test")
    assert result == "Lambda: test"


def test_tool_with_partial_function() -> None:
    """Test Tool with partial function."""

    def full_func(x: str, prefix: str) -> str:
        return f"{prefix}: {x}"

    tool = Tool(
        name="partial_tool",
        func=partial(full_func, prefix="PREFIX"),
        description="Partial tool",
    )

    result = tool.invoke("test")
    assert result == "PREFIX: test"


def test_tool_single_input_detection() -> None:
    """Test is_single_input property."""
    tool = Tool(
        name="single_input",
        func=simple_func,
        description="Single input tool",
    )

    assert tool.is_single_input


def test_tool_with_config_parameter() -> None:
    """Test Tool with RunnableConfig parameter."""
    captured_config = {}

    def func_with_config(x: str, config: RunnableConfig) -> str:
        captured_config["config"] = config
        return x

    tool = Tool(
        name="config_tool",
        func=func_with_config,
        description="Tool with config",
    )

    result = tool.invoke(
        {"x": "test"},
        config={"configurable": {"test_key": "test_value"}},
    )

    assert result == "test"
    assert "config" in captured_config
    assert captured_config["config"]["configurable"]["test_key"] == "test_value"


def test_tool_too_many_args_raises_exception() -> None:
    """Test that providing too many arguments raises ToolException."""
    tool = Tool(
        name="single_tool",
        func=simple_func,
        description="Single input tool",
    )

    with pytest.raises(ToolException, match="Too many arguments to single-input tool"):
        tool.invoke({"x": "test", "y": "extra"})


def test_tool_without_func_raises_not_implemented() -> None:
    """Test that Tool without func raises NotImplementedError on invoke."""
    tool = Tool(
        name="async_only",
        func=None,
        description="Async only tool",
        coroutine=async_func,
    )

    with pytest.raises(
        NotImplementedError, match="Tool does not support sync invocation"
    ):
        tool.invoke("test")


def test_tool_with_return_direct() -> None:
    """Test Tool with return_direct flag."""
    tool = Tool.from_function(
        func=simple_func,
        name="direct_tool",
        description="Direct return tool",
        return_direct=True,
    )

    assert tool.return_direct is True


def test_tool_description_default_empty() -> None:
    """Test that Tool has default empty description."""
    tool = Tool(
        name="test_tool",
        func=simple_func,
        description="",
    )

    assert tool.description == ""


def test_tool_invoke_preserves_input() -> None:
    """Test that invoking tool doesn't mutate input dict."""
    tool = Tool(
        name="test_tool",
        func=simple_func,
        description="Test tool",
    )

    input_dict = {"x": "test"}
    original_input = input_dict.copy()

    tool.invoke(input_dict)

    assert input_dict == original_input


def test_tool_with_json_schema_args() -> None:
    """Test Tool with JSON schema as args_schema."""
    json_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The query"},
        },
        "required": ["query"],
    }

    tool = Tool(
        name="json_schema_tool",
        func=simple_func,
        description="Tool with JSON schema",
        args_schema=json_schema,
    )

    assert tool.args_schema == json_schema
    assert tool.args == {"query": {"type": "string", "description": "The query"}}


async def test_tool_async_with_config_parameter() -> None:
    """Test async Tool with RunnableConfig parameter."""
    captured_config = {}

    async def async_func_with_config(x: str, config: RunnableConfig) -> str:
        captured_config["config"] = config
        return x

    tool = Tool(
        name="async_config_tool",
        func=None,
        description="Async tool with config",
        coroutine=async_func_with_config,
    )

    result = await tool.ainvoke(
        {"x": "test"},
        config={"configurable": {"async_key": "async_value"}},
    )

    assert result == "test"
    assert "config" in captured_config
    assert captured_config["config"]["configurable"]["async_key"] == "async_value"


def test_tool_invoke_with_string_for_dict_arg() -> None:
    """Test that single string input works even with schema expecting dict."""
    tool = Tool(
        name="test_tool",
        func=simple_func,
        description="Test tool",
        args_schema=SimpleSchema,
    )

    # String input should work for single-input tools
    result = tool.invoke("test_string")
    assert "test_string" in result


# ---------------------------------------------------------------------------
# Error handling on Tool
# ---------------------------------------------------------------------------


def test_tool_error_handled_with_bool() -> None:
    """Test Tool with handle_tool_error=True returns error message."""

    def failing_func(x: str) -> str:
        raise ToolException("tool broke")

    tool = Tool(
        name="failing",
        func=failing_func,
        description="Fails",
        handle_tool_error=True,
    )
    result = tool.invoke("test")
    assert result == "tool broke"


def test_tool_error_handled_with_string() -> None:
    """Test Tool with handle_tool_error as string returns that string."""

    def failing_func(x: str) -> str:
        raise ToolException("original")

    tool = Tool(
        name="failing",
        func=failing_func,
        description="Fails",
        handle_tool_error="Custom error",
    )
    result = tool.invoke("test")
    assert result == "Custom error"


def test_tool_error_handled_with_callable() -> None:
    """Test Tool with handle_tool_error as callable."""

    def failing_func(x: str) -> str:
        raise ToolException("msg")

    tool = Tool(
        name="failing",
        func=failing_func,
        description="Fails",
        handle_tool_error=lambda e: f"Caught: {e.args[0]}",
    )
    result = tool.invoke("test")
    assert result == "Caught: msg"


def test_tool_error_not_handled_propagates() -> None:
    """Test that ToolException propagates when handle_tool_error is False."""

    def failing_func(x: str) -> str:
        raise ToolException("boom")

    tool = Tool(
        name="failing",
        func=failing_func,
        description="Fails",
    )
    with pytest.raises(ToolException, match="boom"):
        tool.invoke("test")


def test_tool_error_with_tool_call_returns_error_tool_message() -> None:
    """Test that handled error with ToolCall returns ToolMessage with error status."""

    def failing_func(x: str) -> str:
        raise ToolException("err")

    tool = Tool(
        name="failing",
        func=failing_func,
        description="Fails",
        handle_tool_error=True,
    )
    tc: ToolCall = {
        "name": "failing",
        "args": {"x": "test"},
        "id": "call_err",
        "type": "tool_call",
    }
    result = tool.invoke(tc)
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.content == "err"
    assert result.tool_call_id == "call_err"


# ---------------------------------------------------------------------------
# Tool with callbacks parameter in function
# ---------------------------------------------------------------------------


def test_tool_with_callbacks_parameter() -> None:
    """Test Tool function that accepts callbacks parameter."""
    captured = {}

    def func_with_callbacks(x: str, callbacks: object = None) -> str:
        captured["callbacks"] = callbacks
        return x

    tool = Tool(
        name="cb_tool",
        func=func_with_callbacks,
        description="Tool with callbacks",
    )
    result = tool.invoke("test")
    assert result == "test"


# ---------------------------------------------------------------------------
# Tool run with tool_call_id
# ---------------------------------------------------------------------------


def test_tool_run_with_tool_call_id() -> None:
    """Test Tool.run with explicit tool_call_id returns ToolMessage."""
    tool = Tool(
        name="test_tool",
        func=simple_func,
        description="Test",
    )
    result = tool.run("input", tool_call_id="run_call_1")
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "run_call_1"
    assert result.content == "Result: input"


# ---------------------------------------------------------------------------
# Tool with response_format = content_and_artifact
# ---------------------------------------------------------------------------


def test_tool_content_and_artifact_format() -> None:
    """Test Tool with response_format='content_and_artifact'."""

    def artifact_func(x: str) -> tuple[str, dict]:
        return f"content: {x}", {"raw": x}

    tool = Tool(
        name="artifact_tool",
        func=artifact_func,
        description="Artifact tool",
        response_format="content_and_artifact",
    )
    tc: ToolCall = {
        "name": "artifact_tool",
        "args": {"x": "hello"},
        "id": "call_a",
        "type": "tool_call",
    }
    result = tool.invoke(tc)
    assert isinstance(result, ToolMessage)
    assert result.content == "content: hello"
    assert result.artifact == {"raw": "hello"}


def test_tool_content_and_artifact_bad_return_raises() -> None:
    """Test Tool with content_and_artifact but non-tuple return raises."""

    def bad_func(x: str) -> str:
        return x

    tool = Tool(
        name="bad_tool",
        func=bad_func,
        description="Bad tool",
        response_format="content_and_artifact",
    )
    with pytest.raises(ValueError, match="two-tuple"):
        tool.invoke({"x": "test"})


# ---------------------------------------------------------------------------
# Tool verbose flag
# ---------------------------------------------------------------------------


def test_tool_verbose_flag() -> None:
    """Test Tool verbose flag is settable."""
    tool = Tool(
        name="verbose_tool",
        func=simple_func,
        description="Verbose tool",
        verbose=True,
    )
    assert tool.verbose is True


# ---------------------------------------------------------------------------
# Async Tool with both func and coroutine
# ---------------------------------------------------------------------------


async def test_tool_ainvoke_prefers_coroutine() -> None:
    """Test ainvoke uses coroutine when both func and coroutine are provided."""
    tool = Tool(
        name="dual_tool",
        func=simple_func,
        description="Dual tool",
        coroutine=async_func,
    )
    result = await tool.ainvoke("test")
    assert result == "Async result: test"


async def test_tool_arun_with_tool_call_id() -> None:
    """Test arun with tool_call_id returns ToolMessage."""
    tool = Tool(
        name="async_tool",
        func=simple_func,
        description="Async",
        coroutine=async_func,
    )
    result = await tool.arun("input", tool_call_id="async_call_1")
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "async_call_1"
