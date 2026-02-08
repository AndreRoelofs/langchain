"""Unit tests for the tool decorator and convert_runnable_to_tool."""

from __future__ import annotations

from typing import Annotated, Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import StructuredTool, Tool, tool
from langchain_core.tools.base import InjectedToolArg, InjectedToolCallId
from langchain_core.tools.convert import (
    _get_description_from_runnable,
    _get_schema_from_runnable_and_arg_types,
    convert_runnable_to_tool,
)

# ---------------------------------------------------------------------------
# @tool decorator — basic usage
# ---------------------------------------------------------------------------


class TestToolDecoratorBasic:
    """Tests for @tool decorator basic usage."""

    def test_decorator_without_args(self) -> None:
        @tool
        def search(query: str) -> str:
            """Search for something."""
            return f"result: {query}"

        assert isinstance(search, StructuredTool)
        assert search.name == "search"
        assert "Search for something." in search.description

    def test_decorator_with_name(self) -> None:
        @tool("custom_name")
        def search(query: str) -> str:
            """Search for something."""
            return f"result: {query}"

        assert search.name == "custom_name"

    def test_decorator_with_description(self) -> None:
        @tool(description="Custom description")
        def search(query: str) -> str:
            """Original docstring."""
            return f"result: {query}"

        assert search.description == "Custom description"

    def test_decorator_preserves_function_output(self) -> None:
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = add.invoke({"a": 1, "b": 2})
        assert result == 3

    def test_decorator_with_return_direct(self) -> None:
        @tool(return_direct=True)
        def search(query: str) -> str:
            """Search."""
            return query

        assert search.return_direct is True

    def test_decorator_with_extras(self) -> None:
        @tool(extras={"cache_control": {"type": "ephemeral"}})
        def search(query: str) -> str:
            """Search."""
            return query

        assert search.extras == {"cache_control": {"type": "ephemeral"}}


# ---------------------------------------------------------------------------
# @tool decorator — async functions
# ---------------------------------------------------------------------------


class TestToolDecoratorAsync:
    """Tests for @tool decorator with async functions."""

    def test_async_function(self) -> None:
        @tool
        async def async_search(query: str) -> str:
            """Async search."""
            return f"async: {query}"

        assert isinstance(async_search, StructuredTool)
        assert async_search.coroutine is not None
        assert async_search.func is None

    async def test_async_function_invoke(self) -> None:
        @tool
        async def async_search(query: str) -> str:
            """Async search."""
            return f"async: {query}"

        result = await async_search.ainvoke({"query": "test"})
        assert result == "async: test"


# ---------------------------------------------------------------------------
# @tool decorator — schema inference
# ---------------------------------------------------------------------------


class TestToolDecoratorSchemaInference:
    """Tests for @tool decorator schema inference."""

    def test_infer_schema_true(self) -> None:
        @tool
        def my_tool(x: int, y: str = "default") -> str:
            """My tool."""
            return f"{x}-{y}"

        schema = my_tool.args_schema.model_json_schema()
        assert "x" in schema["properties"]
        assert "y" in schema["properties"]
        assert schema["properties"]["x"]["type"] == "integer"

    def test_infer_schema_false(self) -> None:
        @tool(infer_schema=False)
        def my_tool(x: str) -> str:
            """My tool."""
            return x

        assert isinstance(my_tool, Tool)

    def test_infer_schema_false_no_docstring_raises(self) -> None:
        with pytest.raises(ValueError, match="Function must have a docstring"):

            @tool(infer_schema=False)
            def my_tool(x: str) -> str:
                return x

    def test_custom_args_schema(self) -> None:
        class MySchema(BaseModel):
            query: str = Field(description="The query")

        @tool(args_schema=MySchema)
        def search(query: str) -> str:
            """Search."""
            return query

        assert search.args_schema is MySchema


# ---------------------------------------------------------------------------
# @tool decorator — docstring parsing
# ---------------------------------------------------------------------------


class TestToolDecoratorDocstringParsing:
    """Tests for @tool decorator with parse_docstring."""

    def test_parse_docstring_extracts_arg_descriptions(self) -> None:
        @tool(parse_docstring=True)
        def my_tool(x: int, y: str) -> str:
            """My tool description.

            Args:
                x: The x value.
                y: The y value.
            """
            return f"{x}-{y}"

        schema = my_tool.args_schema.model_json_schema()
        assert schema["properties"]["x"]["description"] == "The x value."
        assert schema["properties"]["y"]["description"] == "The y value."
        assert schema["description"] == "My tool description."

    def test_parse_docstring_extracts_function_description(self) -> None:
        @tool(parse_docstring=True)
        def my_tool(x: int) -> str:
            """This is the tool description.

            Args:
                x: An integer.
            """
            return str(x)

        assert "This is the tool description." in my_tool.description

    def test_error_on_invalid_docstring(self) -> None:
        with pytest.raises(ValueError):

            @tool(parse_docstring=True, error_on_invalid_docstring=True)
            def bad_tool(x: int) -> str:
                """Bad tool.

                Args:
                    nonexistent_param: Does not exist.
                """
                return str(x)


# ---------------------------------------------------------------------------
# @tool decorator — response_format
# ---------------------------------------------------------------------------


class TestToolDecoratorResponseFormat:
    """Tests for @tool decorator with response_format."""

    def test_content_format(self) -> None:
        @tool(response_format="content")
        def my_tool(x: str) -> str:
            """My tool."""
            return x

        assert my_tool.response_format == "content"

    def test_content_and_artifact_format(self) -> None:
        @tool(response_format="content_and_artifact")
        def my_tool(x: str) -> tuple[str, dict]:
            """My tool."""
            return x, {"data": x}

        assert my_tool.response_format == "content_and_artifact"

        tc: ToolCall = {
            "name": "my_tool",
            "args": {"x": "test"},
            "id": "call_1",
            "type": "tool_call",
        }
        result = my_tool.invoke(tc)
        assert isinstance(result, ToolMessage)
        assert result.content == "test"
        assert result.artifact == {"data": "test"}


# ---------------------------------------------------------------------------
# @tool decorator — edge cases
# ---------------------------------------------------------------------------


class TestToolDecoratorEdgeCases:
    """Tests for @tool decorator edge cases."""

    def test_too_many_positional_args_raises(self) -> None:
        with pytest.raises(ValueError, match="Too many arguments"):
            tool("name", None, "extra_arg")  # type: ignore[call-overload]

    def test_runnable_without_name_raises(self) -> None:
        runnable = RunnableLambda(lambda x: x)
        with pytest.raises(ValueError, match="Runnable without name"):
            tool(runnable=runnable)  # type: ignore[call-overload]

    def test_runnable_with_non_string_name_raises(self) -> None:
        runnable = RunnableLambda(lambda x: x)
        with pytest.raises(ValueError, match="Name must be a string"):
            tool(lambda x: x, runnable)  # type: ignore[call-overload]

    def test_invalid_first_arg_type_raises(self) -> None:
        with pytest.raises(
            ValueError, match="first argument must be a string or a callable"
        ):
            tool(42)  # type: ignore[call-overload]

    def test_decorator_with_name_as_string(self) -> None:
        @tool("custom_search")
        def search(query: str) -> str:
            """Search."""
            return query

        assert search.name == "custom_search"

    def test_tool_with_no_arguments(self) -> None:
        @tool
        def no_args() -> str:
            """A tool with no arguments."""
            return "done"

        result = no_args.invoke({})
        assert result == "done"

    def test_tool_with_default_arguments(self) -> None:
        @tool
        def defaults(x: int = 5, y: str = "hello") -> str:
            """Tool with defaults."""
            return f"{x}-{y}"

        result = defaults.invoke({})
        assert result == "5-hello"

        result = defaults.invoke({"x": 10})
        assert result == "10-hello"


# ---------------------------------------------------------------------------
# @tool decorator — injected args
# ---------------------------------------------------------------------------


class TestToolDecoratorInjectedArgs:
    """Tests for @tool decorator with injected arguments."""

    def test_injected_arg_excluded_from_schema(self) -> None:
        @tool
        def my_tool(
            x: int,
            secret: Annotated[str, InjectedToolArg],
        ) -> str:
            """My tool."""
            return f"{x}-{secret}"

        schema = my_tool.tool_call_schema
        fields = schema.model_json_schema()["properties"]
        assert "x" in fields
        assert "secret" not in fields

    def test_injected_tool_call_id(self) -> None:
        @tool
        def my_tool(
            x: int,
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> str:
            """My tool."""
            return f"{x}-{tool_call_id}"

        tc: ToolCall = {
            "name": "my_tool",
            "args": {"x": 42},
            "id": "call_abc",
            "type": "tool_call",
        }
        result = my_tool.invoke(tc)
        assert isinstance(result, ToolMessage)
        assert result.content == "42-call_abc"


# ---------------------------------------------------------------------------
# @tool with RunnableConfig
# ---------------------------------------------------------------------------


class TestToolDecoratorRunnableConfig:
    """Tests for @tool decorator with RunnableConfig parameter."""

    def test_config_parameter_not_in_schema(self) -> None:
        @tool
        def my_tool(x: int, config: RunnableConfig) -> str:
            """My tool."""
            return str(x)

        schema = my_tool.args_schema.model_json_schema()
        assert "config" not in schema.get("properties", {})

    def test_config_parameter_passed_to_function(self) -> None:
        captured = {}

        @tool
        def my_tool(x: int, config: RunnableConfig) -> str:
            """My tool."""
            captured["config"] = config
            return str(x)

        my_tool.invoke(
            {"x": 1},
            config={"configurable": {"key": "value"}},
        )
        assert captured["config"]["configurable"]["key"] == "value"


# ---------------------------------------------------------------------------
# @tool with Runnable
# ---------------------------------------------------------------------------


class TestToolWithRunnable:
    """Tests for @tool with Runnable argument."""

    def test_tool_from_runnable_with_name(self) -> None:
        class InputModel(BaseModel):
            x: int

        runnable = RunnableLambda(lambda d: d["x"] * 2).with_types(
            input_type=InputModel
        )
        t = tool("doubler", runnable, description="Doubles x")
        assert t.name == "doubler"
        assert t.description == "Doubles x"

    def test_tool_from_runnable_invokes(self) -> None:
        class InputModel(BaseModel):
            x: int

        runnable = RunnableLambda(lambda d: d["x"] * 2).with_types(
            input_type=InputModel
        )
        t = tool("doubler", runnable, description="Doubles x")
        result = t.invoke({"x": 5})
        assert result == 10

    def test_runnable_without_object_schema_raises(self) -> None:
        runnable = RunnableLambda(lambda x: x)
        with pytest.raises(ValueError, match="Runnable must have an object schema"):
            tool("str_runnable", runnable, description="str input")


# ---------------------------------------------------------------------------
# convert_runnable_to_tool
# ---------------------------------------------------------------------------


class TestConvertRunnableToTool:
    """Tests for convert_runnable_to_tool."""

    def test_converts_object_schema_runnable(self) -> None:
        class InputModel(BaseModel):
            query: str

        runnable = RunnableLambda(lambda d: f"Result: {d['query']}").with_types(
            input_type=InputModel
        )
        t = convert_runnable_to_tool(runnable, name="search", description="Search tool")
        assert isinstance(t, StructuredTool)
        assert t.name == "search"
        assert t.description == "Search tool"

    def test_converts_with_default_name(self) -> None:
        class InputModel(BaseModel):
            x: int

        runnable = RunnableLambda(lambda d: d["x"]).with_types(input_type=InputModel)
        t = convert_runnable_to_tool(runnable)
        assert t.name is not None

    def test_converts_with_default_description(self) -> None:
        class InputModel(BaseModel):
            x: int

        runnable = RunnableLambda(lambda d: d["x"]).with_types(input_type=InputModel)
        t = convert_runnable_to_tool(runnable)
        assert t.description != ""

    def test_converted_tool_invocation(self) -> None:
        class InputModel(BaseModel):
            a: int
            b: int

        runnable = RunnableLambda(lambda d: d["a"] + d["b"]).with_types(
            input_type=InputModel
        )
        t = convert_runnable_to_tool(
            runnable, name="adder", description="Adds two numbers"
        )
        result = t.invoke({"a": 3, "b": 4})
        assert result == 7

    async def test_converted_tool_async_invocation(self) -> None:
        class InputModel(BaseModel):
            x: int

        runnable = RunnableLambda(lambda d: d["x"] * 10).with_types(
            input_type=InputModel
        )
        t = convert_runnable_to_tool(
            runnable, name="multiplier", description="Multiplies by 10"
        )
        result = await t.ainvoke({"x": 5})
        assert result == 50

    def test_converts_with_custom_args_schema(self) -> None:
        class MySchema(BaseModel):
            val: int

        runnable = RunnableLambda(lambda d: d["val"])
        t = convert_runnable_to_tool(
            runnable,
            args_schema=MySchema,
            name="custom",
            description="Custom schema",
        )
        result = t.invoke({"val": 42})
        assert result == 42

    def test_converts_with_arg_types(self) -> None:
        class InputModel(BaseModel):
            x: int

        runnable = RunnableLambda(lambda d: d["x"]).with_types(input_type=InputModel)
        t = convert_runnable_to_tool(
            runnable,
            name="typed",
            description="Typed tool",
            arg_types={"x": int},
        )
        assert t.name == "typed"

    def test_string_schema_creates_simple_tool(self) -> None:
        runnable = RunnableLambda(lambda x: f"echo: {x}").with_types(input_type=str)
        t = convert_runnable_to_tool(runnable, name="echo", description="Echo tool")
        assert isinstance(t, Tool)
        result = t.invoke("hello")
        assert result == "echo: hello"


# ---------------------------------------------------------------------------
# _get_description_from_runnable
# ---------------------------------------------------------------------------


class TestGetDescriptionFromRunnable:
    """Tests for _get_description_from_runnable."""

    def test_generates_description(self) -> None:
        class InputModel(BaseModel):
            x: int

        runnable = RunnableLambda(lambda d: d["x"]).with_types(input_type=InputModel)
        desc = _get_description_from_runnable(runnable)
        assert "Takes" in desc


# ---------------------------------------------------------------------------
# _get_schema_from_runnable_and_arg_types
# ---------------------------------------------------------------------------


class TestGetSchemaFromRunnableAndArgTypes:
    """Tests for _get_schema_from_runnable_and_arg_types."""

    def test_creates_model_from_arg_types(self) -> None:
        runnable = RunnableLambda(lambda d: d["x"])
        schema = _get_schema_from_runnable_and_arg_types(
            runnable, "TestTool", arg_types={"x": int, "y": str}
        )
        assert issubclass(schema, BaseModel)
        fields = schema.model_json_schema()["properties"]
        assert "x" in fields
        assert "y" in fields
