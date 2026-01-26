"""Unit tests for the StructuredTool class."""

import inspect
import textwrap

import pytest
from pydantic import BaseModel, Field

from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool, tool


class MockSchema(BaseModel):
    """Mock schema for testing."""

    arg1: int = Field(description="The first argument")
    arg2: str = Field(description="The second argument")
    arg3: dict | None = Field(default=None, description="Optional third argument")


def simple_func(arg1: int, arg2: str, arg3: dict | None = None) -> str:
    """Simple function for testing.

    Args:
        arg1: The first argument
        arg2: The second argument
        arg3: Optional third argument
    """
    return f"{arg1}-{arg2}-{arg3}"


async def async_simple_func(arg1: int, arg2: str, arg3: dict | None = None) -> str:
    """Async version of simple function."""
    return f"async-{arg1}-{arg2}-{arg3}"


def no_args_func() -> str:
    """Function with no arguments."""
    return "no args"


def test_structured_tool_from_function_basic() -> None:
    """Test StructuredTool.from_function with basic function."""
    tool_instance = StructuredTool.from_function(simple_func)

    assert tool_instance.name == "simple_func"
    assert tool_instance.func == simple_func
    assert tool_instance.args_schema is not None
    assert tool_instance.description == textwrap.dedent(simple_func.__doc__).strip()


def test_structured_tool_from_function_custom_name() -> None:
    """Test StructuredTool.from_function with custom name."""
    tool_instance = StructuredTool.from_function(
        simple_func,
        name="custom_name",
    )

    assert tool_instance.name == "custom_name"


def test_structured_tool_from_function_custom_description() -> None:
    """Test StructuredTool.from_function with custom description."""
    tool_instance = StructuredTool.from_function(
        simple_func,
        description="Custom description",
    )

    assert tool_instance.description == "Custom description"


def test_structured_tool_from_function_with_args_schema() -> None:
    """Test StructuredTool.from_function with explicit args_schema."""
    tool_instance = StructuredTool.from_function(
        simple_func,
        args_schema=MockSchema,
    )

    assert tool_instance.args_schema == MockSchema


def test_structured_tool_from_function_with_return_direct() -> None:
    """Test StructuredTool.from_function with return_direct flag."""
    tool_instance = StructuredTool.from_function(
        simple_func,
        return_direct=True,
    )

    assert tool_instance.return_direct is True


def test_structured_tool_from_function_infer_schema() -> None:
    """Test that StructuredTool infers schema from function signature."""
    tool_instance = StructuredTool.from_function(simple_func)

    args_schema = tool_instance.args_schema
    assert args_schema is not None

    schema = args_schema.model_json_schema()
    assert "arg1" in schema["properties"]
    assert "arg2" in schema["properties"]
    assert "arg3" in schema["properties"]
    assert schema["properties"]["arg1"]["type"] == "integer"
    assert schema["properties"]["arg2"]["type"] == "string"


def test_structured_tool_from_function_no_infer_schema() -> None:
    """Test StructuredTool.from_function with infer_schema=False."""
    tool_instance = StructuredTool.from_function(
        simple_func,
        infer_schema=False,
    )

    assert tool_instance.args_schema is None


def test_structured_tool_from_function_with_coroutine() -> None:
    """Test StructuredTool.from_function with async coroutine."""
    tool_instance = StructuredTool.from_function(
        func=simple_func,
        coroutine=async_simple_func,
    )

    assert tool_instance.func == simple_func
    assert tool_instance.coroutine == async_simple_func


def test_structured_tool_from_function_coroutine_only() -> None:
    """Test StructuredTool.from_function with only coroutine."""
    tool_instance = StructuredTool.from_function(
        func=None,
        coroutine=async_simple_func,
    )

    assert tool_instance.func is None
    assert tool_instance.coroutine == async_simple_func


def test_structured_tool_from_function_neither_raises() -> None:
    """Test that providing neither func nor coroutine raises ValueError."""
    with pytest.raises(ValueError, match="Function and/or coroutine must be provided"):
        StructuredTool.from_function(func=None, coroutine=None)


def test_structured_tool_from_function_no_docstring_no_description_raises() -> None:
    """Test that function without docstring and no description raises ValueError."""
    def no_doc(arg: int) -> str:
        return str(arg)

    with pytest.raises(ValueError, match="Function must have a docstring"):
        StructuredTool.from_function(no_doc)


def test_structured_tool_from_function_no_docstring_with_description() -> None:
    """Test that function without docstring but with description works."""
    def no_doc(arg: int) -> str:
        return str(arg)

    tool_instance = StructuredTool.from_function(
        no_doc,
        description="Custom description",
    )

    assert tool_instance.description == "Custom description"


def test_structured_tool_invoke_basic() -> None:
    """Test StructuredTool invocation with basic inputs."""
    tool_instance = StructuredTool.from_function(simple_func)

    result = tool_instance.invoke({"arg1": 1, "arg2": "test", "arg3": {"key": "value"}})

    assert result == "1-test-{'key': 'value'}"


def test_structured_tool_invoke_with_tool_call() -> None:
    """Test StructuredTool invocation with ToolCall."""
    tool_instance = StructuredTool.from_function(simple_func)

    tool_call: ToolCall = {
        "name": "simple_func",
        "args": {"arg1": 1, "arg2": "test"},
        "id": "call_123",
        "type": "tool_call",
    }

    result = tool_instance.invoke(tool_call)

    assert isinstance(result, ToolMessage)
    assert result.content == "1-test-None"
    assert result.tool_call_id == "call_123"


async def test_structured_tool_ainvoke() -> None:
    """Test StructuredTool async invocation."""
    tool_instance = StructuredTool.from_function(
        func=simple_func,
        coroutine=async_simple_func,
    )

    result = await tool_instance.ainvoke({"arg1": 1, "arg2": "test"})

    assert result == "async-1-test-None"


async def test_structured_tool_ainvoke_fallback_to_sync() -> None:
    """Test that ainvoke falls back to sync func when no coroutine."""
    tool_instance = StructuredTool.from_function(simple_func)

    result = await tool_instance.ainvoke({"arg1": 1, "arg2": "test"})

    assert result == "1-test-None"


def test_structured_tool_no_func_raises() -> None:
    """Test that StructuredTool without func raises NotImplementedError on invoke."""
    tool_instance = StructuredTool.from_function(
        func=None,
        coroutine=async_simple_func,
    )

    with pytest.raises(NotImplementedError, match="StructuredTool does not support sync invocation"):
        tool_instance.invoke({"arg1": 1, "arg2": "test"})


def test_structured_tool_direct_instantiation() -> None:
    """Test direct instantiation of StructuredTool."""
    tool_instance = StructuredTool(
        name="direct_tool",
        description="Directly instantiated tool",
        func=simple_func,
        args_schema=MockSchema,
    )

    assert tool_instance.name == "direct_tool"
    assert tool_instance.description == "Directly instantiated tool"
    assert tool_instance.func == simple_func
    assert tool_instance.args_schema == MockSchema


def test_structured_tool_with_response_format_content() -> None:
    """Test StructuredTool with response_format='content'."""
    tool_instance = StructuredTool.from_function(
        simple_func,
        response_format="content",
    )

    assert tool_instance.response_format == "content"

    result = tool_instance.invoke({"arg1": 1, "arg2": "test"})
    assert isinstance(result, str)


def test_structured_tool_with_response_format_content_and_artifact() -> None:
    """Test StructuredTool with response_format='content_and_artifact'."""
    def artifact_func(arg1: int, arg2: str) -> tuple[str, dict]:
        """Function returning tuple."""
        return f"{arg1}-{arg2}", {"arg1": arg1, "arg2": arg2}

    tool_instance = StructuredTool.from_function(
        artifact_func,
        response_format="content_and_artifact",
    )

    assert tool_instance.response_format == "content_and_artifact"

    tool_call: ToolCall = {
        "name": "artifact_func",
        "args": {"arg1": 1, "arg2": "test"},
        "id": "call_456",
        "type": "tool_call",
    }

    result = tool_instance.invoke(tool_call)
    assert isinstance(result, ToolMessage)
    assert result.content == "1-test"
    assert result.artifact == {"arg1": 1, "arg2": "test"}


def test_structured_tool_from_function_with_parse_docstring() -> None:
    """Test StructuredTool.from_function with parse_docstring=True."""
    tool_instance = StructuredTool.from_function(
        simple_func,
        parse_docstring=True,
    )

    schema = tool_instance.args_schema.model_json_schema()

    # Check that descriptions were parsed from docstring
    assert schema["properties"]["arg1"]["description"] == "The first argument"
    assert schema["properties"]["arg2"]["description"] == "The second argument"


def test_structured_tool_from_function_parameterless() -> None:
    """Test StructuredTool.from_function with parameterless function."""
    tool_instance = StructuredTool.from_function(no_args_func)

    result = tool_instance.invoke({})
    assert result == "no args"


def test_structured_tool_with_config_parameter() -> None:
    """Test StructuredTool with RunnableConfig parameter."""
    captured_config = {}

    def func_with_config(arg1: int, config: RunnableConfig) -> str:
        """Function with config parameter."""
        captured_config["config"] = config
        return str(arg1)

    tool_instance = StructuredTool.from_function(func_with_config)

    result = tool_instance.invoke(
        {"arg1": 42},
        config={"configurable": {"test_key": "test_value"}},
    )

    assert result == "42"
    assert "config" in captured_config
    assert captured_config["config"]["configurable"]["test_key"] == "test_value"


def test_structured_tool_description_from_args_schema() -> None:
    """Test that description is taken from args_schema if no other source."""
    class SchemaWithDoc(BaseModel):
        """Schema documentation."""

        arg: int

    def no_doc_func(arg: int) -> str:
        return str(arg)

    tool_instance = StructuredTool.from_function(
        no_doc_func,
        args_schema=SchemaWithDoc,
    )

    assert tool_instance.description == "Schema documentation."


def test_structured_tool_strips_pydantic_base_description() -> None:
    """Test that generic Pydantic base model descriptions are stripped."""
    class EmptyDocSchema(BaseModel):
        """A base class for creating Pydantic models with strict behavior."""

        arg: int

    def no_doc_func(arg: int) -> str:
        return str(arg)

    tool_instance = StructuredTool.from_function(
        no_doc_func,
        args_schema=EmptyDocSchema,
    )

    # Should be empty, not the Pydantic base description
    assert tool_instance.description == ""


def test_structured_tool_json_schema_args() -> None:
    """Test StructuredTool with JSON schema as args_schema."""
    json_schema = {
        "type": "object",
        "properties": {
            "arg1": {"type": "integer"},
            "arg2": {"type": "string"},
        },
        "required": ["arg1", "arg2"],
        "description": "JSON schema description",
    }

    tool_instance = StructuredTool(
        name="json_tool",
        description="Tool with JSON schema",
        func=simple_func,
        args_schema=json_schema,
    )

    assert tool_instance.args_schema == json_schema
    assert tool_instance.description == "Tool with JSON schema"


def test_structured_tool_json_schema_description_extraction() -> None:
    """Test that description can be extracted from JSON schema args."""
    json_schema = {
        "type": "object",
        "properties": {"arg": {"type": "integer"}},
        "description": "Schema description",
    }

    def no_doc_func(arg: int) -> str:
        return str(arg)

    tool_instance = StructuredTool.from_function(
        no_doc_func,
        args_schema=json_schema,
    )

    assert tool_instance.description == "Schema description"


def test_structured_tool_invalid_args_schema_raises() -> None:
    """Test that invalid args_schema raises TypeError."""
    def simple(arg: int) -> str:
        """Simple function."""
        return str(arg)

    with pytest.raises(TypeError, match="args_schema must be a subclass of pydantic BaseModel"):
        StructuredTool.from_function(
            simple,
            args_schema="invalid",  # type: ignore
        )


def test_structured_tool_injected_args_keys() -> None:
    """Test _injected_args_keys property."""
    from typing import Annotated
    from langchain_core.tools.base import InjectedToolArg

    def func_with_injected(
        arg1: int,
        injected_arg: Annotated[str, InjectedToolArg],
    ) -> str:
        """Function with injected arg."""
        return f"{arg1}-{injected_arg}"

    tool_instance = StructuredTool.from_function(func_with_injected)

    # _injected_args_keys should identify the injected argument
    assert "injected_arg" in tool_instance._injected_args_keys


def test_structured_tool_preserves_input() -> None:
    """Test that StructuredTool doesn't mutate input dict."""
    tool_instance = StructuredTool.from_function(simple_func)

    input_dict = {"arg1": 1, "arg2": "test", "arg3": {"key": "value"}}
    original_input = input_dict.copy()

    tool_instance.invoke(input_dict)

    assert input_dict == original_input


def test_structured_tool_extras_field() -> None:
    """Test StructuredTool with extras field."""
    tool_instance = StructuredTool.from_function(
        simple_func,
        extras={"custom_field": "custom_value"},
    )

    assert tool_instance.extras == {"custom_field": "custom_value"}
