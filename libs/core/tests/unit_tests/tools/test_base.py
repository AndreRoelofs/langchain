"""Unit tests for tools base module utilities and classes."""

from __future__ import annotations

import json
from typing import Annotated, Any, TypeVar

import pytest
from pydantic import BaseModel, Field, ValidationError
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import override

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    StructuredTool,
    ToolException,
    create_schema_from_function,
)
from langchain_core.tools.base import (
    BaseToolkit,
    InjectedToolArg,
    InjectedToolCallId,
    SchemaAnnotationError,
    _format_output,
    _get_annotation_description,
    _get_filtered_args,
    _get_runnable_config_param,
    _get_type_hints,
    _handle_tool_error,
    _handle_validation_error,
    _is_annotated_type,
    _is_injected_arg_type,
    _is_message_content_block,
    _is_message_content_type,
    _is_tool_call,
    _prep_run_args,
    _replace_type_vars,
    _stringify,
    get_all_basemodel_annotations,
)

# ---------------------------------------------------------------------------
# _is_annotated_type
# ---------------------------------------------------------------------------


class TestIsAnnotatedType:
    """Tests for _is_annotated_type."""

    def test_annotated_type_returns_true(self) -> None:
        assert _is_annotated_type(Annotated[int, "desc"]) is True

    def test_plain_type_returns_false(self) -> None:
        assert _is_annotated_type(int) is False

    def test_list_type_returns_false(self) -> None:
        assert _is_annotated_type(list[int]) is False

    def test_optional_type_returns_false(self) -> None:
        assert _is_annotated_type(int | None) is False

    def test_annotated_with_multiple_metadata(self) -> None:
        assert _is_annotated_type(Annotated[str, "desc", 42]) is True


# ---------------------------------------------------------------------------
# _get_annotation_description
# ---------------------------------------------------------------------------


class TestGetAnnotationDescription:
    """Tests for _get_annotation_description."""

    def test_returns_string_description(self) -> None:
        assert (
            _get_annotation_description(Annotated[int, "my description"])
            == "my description"
        )

    def test_returns_none_for_plain_type(self) -> None:
        assert _get_annotation_description(int) is None

    def test_returns_first_string_metadata(self) -> None:
        result = _get_annotation_description(Annotated[int, 42, "first string"])
        assert result == "first string"

    def test_returns_none_when_no_string_metadata(self) -> None:
        assert _get_annotation_description(Annotated[int, 42, 3.14]) is None


# ---------------------------------------------------------------------------
# _is_tool_call
# ---------------------------------------------------------------------------


class TestIsToolCall:
    """Tests for _is_tool_call."""

    def test_valid_tool_call(self) -> None:
        tc: ToolCall = {
            "name": "test",
            "args": {"x": 1},
            "id": "id1",
            "type": "tool_call",
        }
        assert _is_tool_call(tc) is True

    def test_dict_without_type(self) -> None:
        assert _is_tool_call({"name": "test", "args": {}}) is False

    def test_dict_with_wrong_type(self) -> None:
        assert _is_tool_call({"type": "not_tool_call"}) is False

    def test_string_input(self) -> None:
        assert _is_tool_call("hello") is False

    def test_none_input(self) -> None:
        assert _is_tool_call(None) is False

    def test_list_input(self) -> None:
        assert _is_tool_call([1, 2, 3]) is False


# ---------------------------------------------------------------------------
# _stringify
# ---------------------------------------------------------------------------


class TestStringify:
    """Tests for _stringify."""

    def test_dict_uses_json(self) -> None:
        result = _stringify({"key": "value"})
        assert result == json.dumps({"key": "value"}, ensure_ascii=False)

    def test_list_uses_json(self) -> None:
        result = _stringify([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_number_uses_json(self) -> None:
        assert _stringify(42) == "42"

    def test_non_serializable_falls_back_to_str(self) -> None:
        obj = object()
        result = _stringify(obj)
        assert result == str(obj)

    def test_unicode_content_preserved(self) -> None:
        result = _stringify({"text": "hello"})
        assert "hello" in result

    def test_bool_value(self) -> None:
        assert _stringify(True) == "true"

    def test_none_value(self) -> None:
        assert _stringify(None) == "null"


# ---------------------------------------------------------------------------
# _is_message_content_type / _is_message_content_block
# ---------------------------------------------------------------------------


class TestIsMessageContentType:
    """Tests for _is_message_content_type."""

    def test_string_is_valid(self) -> None:
        assert _is_message_content_type("hello") is True

    def test_list_of_strings_is_valid(self) -> None:
        assert _is_message_content_type(["hello", "world"]) is True

    def test_list_of_text_blocks_is_valid(self) -> None:
        assert _is_message_content_type([{"type": "text", "text": "hello"}]) is True

    def test_list_of_image_url_blocks_is_valid(self) -> None:
        assert (
            _is_message_content_type(
                [{"type": "image_url", "image_url": {"url": "..."}}]
            )
            is True
        )

    def test_integer_is_invalid(self) -> None:
        assert _is_message_content_type(42) is False

    def test_dict_is_invalid(self) -> None:
        assert _is_message_content_type({"key": "value"}) is False

    def test_empty_list_is_valid(self) -> None:
        assert _is_message_content_type([]) is True

    def test_list_with_invalid_block_is_invalid(self) -> None:
        assert _is_message_content_type([{"type": "unknown_type"}]) is False

    def test_list_with_mixed_valid_blocks(self) -> None:
        blocks = [
            "plain text",
            {"type": "text", "text": "hello"},
            {"type": "json", "data": {}},
        ]
        assert _is_message_content_type(blocks) is True


class TestIsMessageContentBlock:
    """Tests for _is_message_content_block."""

    def test_string_is_valid(self) -> None:
        assert _is_message_content_block("hello") is True

    def test_text_block(self) -> None:
        assert _is_message_content_block({"type": "text"}) is True

    def test_image_block(self) -> None:
        assert _is_message_content_block({"type": "image"}) is True

    def test_json_block(self) -> None:
        assert _is_message_content_block({"type": "json"}) is True

    def test_image_url_block(self) -> None:
        assert _is_message_content_block({"type": "image_url"}) is True

    def test_search_result_block(self) -> None:
        assert _is_message_content_block({"type": "search_result"}) is True

    def test_document_block(self) -> None:
        assert _is_message_content_block({"type": "document"}) is True

    def test_file_block(self) -> None:
        assert _is_message_content_block({"type": "file"}) is True

    def test_custom_tool_call_output_block(self) -> None:
        assert _is_message_content_block({"type": "custom_tool_call_output"}) is True

    def test_unknown_type_block(self) -> None:
        assert _is_message_content_block({"type": "unknown"}) is False

    def test_dict_without_type(self) -> None:
        assert _is_message_content_block({"key": "value"}) is False

    def test_integer_is_invalid(self) -> None:
        assert _is_message_content_block(42) is False


# ---------------------------------------------------------------------------
# _handle_validation_error
# ---------------------------------------------------------------------------


class TestHandleValidationError:
    """Tests for _handle_validation_error."""

    def _make_validation_error(self) -> ValidationError:
        class M(BaseModel):
            x: int

        try:
            M(x="not_an_int")  # type: ignore[arg-type]
        except ValidationError as e:
            return e
        msg = "Should have raised"
        raise AssertionError(msg)  # pragma: no cover

    def test_bool_flag_returns_generic_message(self) -> None:
        e = self._make_validation_error()
        result = _handle_validation_error(e, flag=True)
        assert result == "Tool input validation error"

    def test_string_flag_returns_custom_message(self) -> None:
        e = self._make_validation_error()
        result = _handle_validation_error(e, flag="Custom error message")
        assert result == "Custom error message"

    def test_callable_flag_receives_error(self) -> None:
        e = self._make_validation_error()
        result = _handle_validation_error(
            e, flag=lambda err: f"Error: {type(err).__name__}"
        )
        assert result == "Error: ValidationError"

    def test_invalid_flag_raises(self) -> None:
        e = self._make_validation_error()
        with pytest.raises(ValueError, match="Got unexpected type"):
            _handle_validation_error(e, flag=42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _handle_tool_error
# ---------------------------------------------------------------------------


class TestHandleToolError:
    """Tests for _handle_tool_error."""

    def test_bool_flag_returns_exception_message(self) -> None:
        e = ToolException("Something went wrong")
        result = _handle_tool_error(e, flag=True)
        assert result == "Something went wrong"

    def test_bool_flag_empty_args_returns_generic_message(self) -> None:
        e = ToolException()
        result = _handle_tool_error(e, flag=True)
        assert result == "Tool execution error"

    def test_string_flag_returns_custom_message(self) -> None:
        e = ToolException("original")
        result = _handle_tool_error(e, flag="Custom error")
        assert result == "Custom error"

    def test_callable_flag_receives_error(self) -> None:
        e = ToolException("err_msg")
        result = _handle_tool_error(e, flag=lambda err: f"Handled: {err.args[0]}")
        assert result == "Handled: err_msg"

    def test_invalid_flag_raises(self) -> None:
        e = ToolException("err")
        with pytest.raises(ValueError, match="Got unexpected type"):
            _handle_tool_error(e, flag=42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _format_output
# ---------------------------------------------------------------------------


class TestFormatOutput:
    """Tests for _format_output."""

    def test_returns_tool_message_when_tool_call_id_present(self) -> None:
        result = _format_output("content", None, "call_1", "tool_name", "success")
        assert isinstance(result, ToolMessage)
        assert result.content == "content"
        assert result.tool_call_id == "call_1"
        assert result.name == "tool_name"
        assert result.status == "success"

    def test_returns_raw_content_when_no_tool_call_id(self) -> None:
        result = _format_output("content", None, None, "tool_name", "success")
        assert result == "content"

    def test_includes_artifact(self) -> None:
        result = _format_output(
            "content", {"data": 1}, "call_1", "tool_name", "success"
        )
        assert isinstance(result, ToolMessage)
        assert result.artifact == {"data": 1}

    def test_non_content_type_gets_stringified(self) -> None:
        result = _format_output(42, None, "call_1", "tool_name", "success")
        assert isinstance(result, ToolMessage)
        assert result.content == "42"

    def test_list_content_blocks_preserved(self) -> None:
        content = [{"type": "text", "text": "hello"}]
        result = _format_output(content, None, "call_1", "tool_name", "success")
        assert isinstance(result, ToolMessage)
        assert result.content == content

    def test_error_status(self) -> None:
        result = _format_output("error msg", None, "call_1", "tool_name", "error")
        assert isinstance(result, ToolMessage)
        assert result.status == "error"

    def test_dict_content_gets_stringified(self) -> None:
        result = _format_output({"key": "val"}, None, "call_1", "t", "success")
        assert isinstance(result, ToolMessage)
        # dict is not valid message content so should be JSON stringified
        assert result.content == json.dumps({"key": "val"}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# _prep_run_args
# ---------------------------------------------------------------------------


class TestPrepRunArgs:
    """Tests for _prep_run_args."""

    def test_string_input(self) -> None:
        tool_input, kwargs = _prep_run_args("hello", None)
        assert tool_input == "hello"
        assert kwargs["tool_call_id"] is None

    def test_dict_input(self) -> None:
        tool_input, kwargs = _prep_run_args({"x": 1}, None)
        assert tool_input == {"x": 1}
        assert kwargs["tool_call_id"] is None

    def test_tool_call_input(self) -> None:
        tc: ToolCall = {
            "name": "test",
            "args": {"x": 1},
            "id": "call_123",
            "type": "tool_call",
        }
        tool_input, kwargs = _prep_run_args(tc, None)
        assert tool_input == {"x": 1}
        assert kwargs["tool_call_id"] == "call_123"

    def test_tool_call_args_are_copied(self) -> None:
        original_args = {"x": 1}
        tc: ToolCall = {
            "name": "test",
            "args": original_args,
            "id": "call_123",
            "type": "tool_call",
        }
        tool_input, _ = _prep_run_args(tc, None)
        assert tool_input is not original_args
        assert tool_input == original_args

    def test_config_is_propagated(self) -> None:
        config: RunnableConfig = {
            "tags": ["tag1"],
            "metadata": {"key": "value"},
            "callbacks": None,
        }
        _, kwargs = _prep_run_args("hello", config)
        assert kwargs["tags"] == ["tag1"]
        assert kwargs["metadata"] == {"key": "value"}


# ---------------------------------------------------------------------------
# _get_runnable_config_param
# ---------------------------------------------------------------------------


class TestGetRunnableConfigParam:
    """Tests for _get_runnable_config_param."""

    def test_detects_config_param(self) -> None:
        def func(x: int, config: RunnableConfig) -> str:
            return str(x)

        assert _get_runnable_config_param(func) == "config"

    def test_returns_none_when_no_config(self) -> None:
        def func(x: int) -> str:
            return str(x)

        assert _get_runnable_config_param(func) is None

    def test_detects_custom_named_config(self) -> None:
        def func(x: int, my_config: RunnableConfig) -> str:
            return str(x)

        assert _get_runnable_config_param(func) == "my_config"


# ---------------------------------------------------------------------------
# _get_type_hints
# ---------------------------------------------------------------------------


class TestGetTypeHints:
    """Tests for _get_type_hints."""

    def test_regular_function(self) -> None:
        def func(x: int, y: str) -> bool:
            return True

        hints = _get_type_hints(func)
        assert hints is not None
        assert hints["x"] is int
        assert hints["y"] is str

    def test_partial_function(self) -> None:
        import functools

        def func(x: int, y: str) -> bool:
            return True

        partial_func = functools.partial(func, x=1)
        hints = _get_type_hints(partial_func)
        assert hints is not None
        assert hints["x"] is int

    def test_lambda_returns_empty_hints(self) -> None:
        hints = _get_type_hints(lambda x: x)
        assert hints is not None
        assert len(hints) == 0


# ---------------------------------------------------------------------------
# _is_injected_arg_type
# ---------------------------------------------------------------------------


class TestIsInjectedArgType:
    """Tests for _is_injected_arg_type."""

    def test_injected_tool_arg_instance(self) -> None:
        t = Annotated[str, InjectedToolArg()]
        assert _is_injected_arg_type(t) is True

    def test_injected_tool_arg_class(self) -> None:
        t = Annotated[str, InjectedToolArg]
        assert _is_injected_arg_type(t) is True

    def test_injected_tool_call_id(self) -> None:
        t = Annotated[str, InjectedToolCallId()]
        assert _is_injected_arg_type(t) is True

    def test_specific_injected_type_check(self) -> None:
        t = Annotated[str, InjectedToolCallId()]
        assert _is_injected_arg_type(t, injected_type=InjectedToolCallId) is True

    def test_non_injected_type(self) -> None:
        t = Annotated[str, "description"]
        assert _is_injected_arg_type(t) is False

    def test_plain_type_not_injected(self) -> None:
        assert _is_injected_arg_type(int) is False

    def test_injected_subclass(self) -> None:
        class CustomInjected(InjectedToolArg):
            pass

        t = Annotated[str, CustomInjected()]
        assert _is_injected_arg_type(t) is True


# ---------------------------------------------------------------------------
# SchemaAnnotationError
# ---------------------------------------------------------------------------


class TestSchemaAnnotationError:
    """Tests for SchemaAnnotationError."""

    def test_is_type_error_subclass(self) -> None:
        assert issubclass(SchemaAnnotationError, TypeError)

    def test_raised_for_bad_annotation(self) -> None:
        # __init_subclass__ checks if the raw annotation in __annotations__
        # is exactly BaseModel. This only triggers when annotations are NOT
        # deferred (i.e., without `from __future__ import annotations`).
        # We must use a subprocess or exec without future annotations.
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pydantic import BaseModel\n"
                    "from langchain_core.tools.base import BaseTool, SchemaAnnotationError\n"
                    "try:\n"
                    "    class BadTool(BaseTool):\n"
                    "        name: str = 'bad'\n"
                    "        description: str = 'bad'\n"
                    "        args_schema: BaseModel = None\n"
                    "        def _run(self) -> str:\n"
                    "            return ''\n"
                    "except SchemaAnnotationError:\n"
                    "    print('RAISED')\n"
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert "RAISED" in result.stdout


# ---------------------------------------------------------------------------
# ToolException
# ---------------------------------------------------------------------------


class TestToolException:
    """Tests for ToolException."""

    def test_is_exception(self) -> None:
        assert issubclass(ToolException, Exception)

    def test_message_accessible(self) -> None:
        e = ToolException("Tool failed")
        assert str(e) == "Tool failed"

    def test_empty_exception(self) -> None:
        e = ToolException()
        assert e.args == ()


# ---------------------------------------------------------------------------
# create_schema_from_function
# ---------------------------------------------------------------------------


class TestCreateSchemaFromFunction:
    """Tests for create_schema_from_function."""

    def test_basic_function(self) -> None:
        def func(a: int, b: str) -> str:
            """Test function."""
            return f"{a}{b}"

        schema = create_schema_from_function("TestSchema", func)
        assert issubclass(schema, BaseModel)
        fields = schema.model_json_schema()["properties"]
        assert "a" in fields
        assert "b" in fields

    def test_filters_run_manager(self) -> None:
        def func(a: int, run_manager: Any = None) -> str:
            """Test function."""
            return str(a)

        schema = create_schema_from_function("TestSchema", func)
        fields = schema.model_json_schema()["properties"]
        assert "a" in fields
        assert "run_manager" not in fields

    def test_filters_callbacks(self) -> None:
        def func(a: int, callbacks: Any = None) -> str:
            """Test function."""
            return str(a)

        schema = create_schema_from_function("TestSchema", func)
        fields = schema.model_json_schema()["properties"]
        assert "a" in fields
        assert "callbacks" not in fields

    def test_custom_filter_args(self) -> None:
        def func(a: int, b: str, c: float) -> str:
            """Test function."""
            return ""

        schema = create_schema_from_function("TestSchema", func, filter_args=["b"])
        fields = schema.model_json_schema()["properties"]
        assert "a" in fields
        assert "c" in fields
        assert "b" not in fields

    def test_parse_docstring_extracts_descriptions(self) -> None:
        def func(a: int, b: str) -> str:
            """Test function.

            Args:
                a: The integer argument.
                b: The string argument.
            """
            return ""

        schema = create_schema_from_function("TestSchema", func, parse_docstring=True)
        props = schema.model_json_schema()["properties"]
        assert props["a"]["description"] == "The integer argument."
        assert props["b"]["description"] == "The string argument."

    def test_optional_arguments(self) -> None:
        def func(a: int, b: str = "default") -> str:
            """Test function."""
            return ""

        schema = create_schema_from_function("TestSchema", func)
        json_schema = schema.model_json_schema()
        # a should be required, b should not
        assert "a" in json_schema.get("required", [])

    def test_include_injected_true_keeps_injected_args(self) -> None:
        def func(
            a: int,
            injected: Annotated[str, InjectedToolArg],
        ) -> str:
            """Test function."""
            return ""

        schema = create_schema_from_function("TestSchema", func, include_injected=True)
        fields = schema.model_json_schema()["properties"]
        assert "a" in fields
        assert "injected" in fields

    def test_annotated_description_used_when_no_docstring(self) -> None:
        def func(a: Annotated[int, "the a parameter"]) -> str:
            """Test function."""
            return ""

        schema = create_schema_from_function("TestSchema", func)
        props = schema.model_json_schema()["properties"]
        assert props["a"]["description"] == "the a parameter"


# ---------------------------------------------------------------------------
# BaseTool core functionality
# ---------------------------------------------------------------------------


class SimpleTool(BaseTool):
    """A simple tool for testing."""

    name: str = "simple_tool"
    description: str = "A simple tool"

    @override
    def _run(self, x: str, **kwargs: Any) -> str:
        return f"result: {x}"


class AsyncTool(BaseTool):
    """A tool with async implementation."""

    name: str = "async_tool"
    description: str = "An async tool"

    @override
    def _run(self, x: str, **kwargs: Any) -> str:
        return f"sync: {x}"

    @override
    async def _arun(self, x: str, **kwargs: Any) -> str:
        return f"async: {x}"


class TestBaseToolProperties:
    """Tests for BaseTool properties and initialization."""

    def test_is_single_input(self) -> None:
        tool = SimpleTool()
        assert tool.is_single_input is True

    def test_args_property_from_run_method(self) -> None:
        tool = SimpleTool()
        args = tool.args
        assert "x" in args

    def test_default_field_values(self) -> None:
        tool = SimpleTool()
        assert tool.return_direct is False
        assert tool.verbose is False
        assert tool.callbacks is None
        assert tool.tags is None
        assert tool.metadata is None
        assert tool.handle_tool_error is False
        assert tool.handle_validation_error is False
        assert tool.response_format == "content"
        assert tool.extras is None

    def test_custom_tags_and_metadata(self) -> None:
        tool = SimpleTool(tags=["tag1"], metadata={"key": "value"})
        assert tool.tags == ["tag1"]
        assert tool.metadata == {"key": "value"}

    def test_extras_field(self) -> None:
        tool = SimpleTool(extras={"cache_control": {"type": "ephemeral"}})
        assert tool.extras == {"cache_control": {"type": "ephemeral"}}


# ---------------------------------------------------------------------------
# BaseTool.invoke / run with error handling
# ---------------------------------------------------------------------------


class ToolExceptionTool(BaseTool):
    """Tool that raises ToolException."""

    name: str = "exception_tool"
    description: str = "Raises ToolException"

    @override
    def _run(self, **kwargs: Any) -> str:
        raise ToolException("Tool failed!")


class ValidationExceptionTool(BaseTool):
    """Tool with strict validation."""

    name: str = "validation_tool"
    description: str = "Tool with validation"
    args_schema: type[BaseModel] = type(
        "StrictSchema",
        (BaseModel,),
        {"__annotations__": {"x": int}},
    )

    @override
    def _run(self, x: int, **kwargs: Any) -> str:
        return str(x)


class TestBaseToolErrorHandling:
    """Tests for BaseTool error handling."""

    def test_tool_exception_propagates_without_handler(self) -> None:
        tool = ToolExceptionTool()
        with pytest.raises(ToolException, match="Tool failed!"):
            tool.invoke({})

    def test_tool_exception_handled_with_bool(self) -> None:
        tool = ToolExceptionTool(handle_tool_error=True)
        result = tool.invoke({})
        assert result == "Tool failed!"

    def test_tool_exception_handled_with_string(self) -> None:
        tool = ToolExceptionTool(handle_tool_error="Custom error message")
        result = tool.invoke({})
        assert result == "Custom error message"

    def test_tool_exception_handled_with_callable(self) -> None:
        tool = ToolExceptionTool(handle_tool_error=lambda e: f"Handled: {e.args[0]}")
        result = tool.invoke({})
        assert result == "Handled: Tool failed!"

    def test_tool_exception_handled_with_tool_call_returns_tool_message(self) -> None:
        tool = ToolExceptionTool(handle_tool_error=True)
        tc: ToolCall = {
            "name": "exception_tool",
            "args": {},
            "id": "call_1",
            "type": "tool_call",
        }
        result = tool.invoke(tc)
        assert isinstance(result, ToolMessage)
        assert result.content == "Tool failed!"
        assert result.status == "error"
        assert result.tool_call_id == "call_1"

    def test_validation_error_propagates_without_handler(self) -> None:
        tool = ValidationExceptionTool()
        with pytest.raises(ValidationError):
            tool.invoke({"x": "not_an_int"})

    def test_validation_error_handled_with_bool(self) -> None:
        tool = ValidationExceptionTool(handle_validation_error=True)
        result = tool.invoke({"x": "not_an_int"})
        assert result == "Tool input validation error"

    def test_validation_error_handled_with_string(self) -> None:
        tool = ValidationExceptionTool(handle_validation_error="Bad input")
        result = tool.invoke({"x": "not_an_int"})
        assert result == "Bad input"

    def test_validation_error_handled_with_callable(self) -> None:
        tool = ValidationExceptionTool(
            handle_validation_error=lambda e: f"Validation: {type(e).__name__}"
        )
        result = tool.invoke({"x": "not_an_int"})
        assert result == "Validation: ValidationError"


# ---------------------------------------------------------------------------
# BaseTool async error handling
# ---------------------------------------------------------------------------


class AsyncToolExceptionTool(BaseTool):
    """Async tool that raises ToolException."""

    name: str = "async_exception_tool"
    description: str = "Raises ToolException async"

    @override
    def _run(self, **kwargs: Any) -> str:
        raise ToolException("sync fail")

    @override
    async def _arun(self, **kwargs: Any) -> str:
        raise ToolException("Async tool failed!")


class TestBaseToolAsyncErrorHandling:
    """Tests for BaseTool async error handling."""

    async def test_async_tool_exception_propagates(self) -> None:
        tool = AsyncToolExceptionTool()
        with pytest.raises(ToolException, match="Async tool failed!"):
            await tool.ainvoke({})

    async def test_async_tool_exception_handled_with_bool(self) -> None:
        tool = AsyncToolExceptionTool(handle_tool_error=True)
        result = await tool.ainvoke({})
        assert result == "Async tool failed!"

    async def test_async_tool_exception_handled_with_string(self) -> None:
        tool = AsyncToolExceptionTool(handle_tool_error="Async error handled")
        result = await tool.ainvoke({})
        assert result == "Async error handled"


# ---------------------------------------------------------------------------
# BaseTool.response_format = "content_and_artifact"
# ---------------------------------------------------------------------------


class ArtifactTool(BaseTool):
    """Tool that returns content and artifact."""

    name: str = "artifact_tool"
    description: str = "Returns artifact"
    response_format: str = "content_and_artifact"

    @override
    def _run(self, **kwargs: Any) -> tuple[str, dict]:
        return "content", {"artifact_key": "artifact_value"}


class BadArtifactTool(BaseTool):
    """Tool that returns wrong type for content_and_artifact."""

    name: str = "bad_artifact_tool"
    description: str = "Returns bad artifact"
    response_format: str = "content_and_artifact"

    @override
    def _run(self, **kwargs: Any) -> str:
        return "not a tuple"


class TestBaseToolResponseFormat:
    """Tests for BaseTool response_format handling."""

    def test_content_and_artifact_with_tool_call(self) -> None:
        tool = ArtifactTool()
        tc: ToolCall = {
            "name": "artifact_tool",
            "args": {},
            "id": "call_1",
            "type": "tool_call",
        }
        result = tool.invoke(tc)
        assert isinstance(result, ToolMessage)
        assert result.content == "content"
        assert result.artifact == {"artifact_key": "artifact_value"}

    def test_content_and_artifact_without_tool_call(self) -> None:
        tool = ArtifactTool()
        result = tool.invoke({})
        assert result == "content"

    def test_bad_artifact_format_raises(self) -> None:
        tool = BadArtifactTool()
        with pytest.raises(ValueError, match="two-tuple"):
            tool.invoke({})


# ---------------------------------------------------------------------------
# BaseTool async response_format = "content_and_artifact"
# ---------------------------------------------------------------------------


class AsyncArtifactTool(BaseTool):
    """Async tool returning content and artifact."""

    name: str = "async_artifact_tool"
    description: str = "Async artifact tool"
    response_format: str = "content_and_artifact"

    @override
    def _run(self, **kwargs: Any) -> tuple[str, dict]:
        return "sync content", {"sync": True}

    @override
    async def _arun(self, **kwargs: Any) -> tuple[str, dict]:
        return "async content", {"async": True}


class TestBaseToolAsyncResponseFormat:
    """Tests for BaseTool async response_format handling."""

    async def test_async_content_and_artifact(self) -> None:
        tool = AsyncArtifactTool()
        tc: ToolCall = {
            "name": "async_artifact_tool",
            "args": {},
            "id": "call_1",
            "type": "tool_call",
        }
        result = await tool.ainvoke(tc)
        assert isinstance(result, ToolMessage)
        assert result.content == "async content"
        assert result.artifact == {"async": True}


# ---------------------------------------------------------------------------
# BaseTool._to_args_and_kwargs
# ---------------------------------------------------------------------------


class NoArgsTool(BaseTool):
    """Tool with empty schema (no args)."""

    name: str = "no_args"
    description: str = "No args tool"
    args_schema: type[BaseModel] = type(
        "EmptySchema",
        (BaseModel,),
        {},
    )

    @override
    def _run(self, **kwargs: Any) -> str:
        return "no args result"


class TestBaseToolToArgsAndKwargs:
    """Tests for BaseTool._to_args_and_kwargs."""

    def test_empty_schema_returns_no_args(self) -> None:
        tool = NoArgsTool()
        args, kwargs = tool._to_args_and_kwargs({}, None)
        assert args == ()
        assert kwargs == {}

    def test_string_input_becomes_positional_arg(self) -> None:
        tool = SimpleTool()
        args, kwargs = tool._to_args_and_kwargs("hello", None)
        assert args == ("hello",)
        assert kwargs == {}

    def test_dict_input_becomes_kwargs(self) -> None:
        tool = SimpleTool()
        args, kwargs = tool._to_args_and_kwargs({"x": "hello"}, None)
        assert args == ()
        assert kwargs == {"x": "hello"}


# ---------------------------------------------------------------------------
# BaseTool._parse_input with InjectedToolCallId
# ---------------------------------------------------------------------------


class InjectedToolCallIdSchema(BaseModel):
    """Schema with injected tool call id."""

    x: int
    tool_call_id: Annotated[str, InjectedToolCallId]


class InjectedTool(BaseTool):
    """Tool with InjectedToolCallId."""

    name: str = "injected_tool"
    description: str = "Tool with injected call id"
    args_schema: type[BaseModel] = InjectedToolCallIdSchema

    @override
    def _run(self, x: int, tool_call_id: str, **kwargs: Any) -> str:
        return f"{x}-{tool_call_id}"


class TestBaseToolParseInputInjection:
    """Tests for BaseTool._parse_input with injected args."""

    def test_injected_tool_call_id_is_set(self) -> None:
        tool = InjectedTool()
        result = tool._parse_input({"x": 1}, tool_call_id="call_abc")
        assert result["tool_call_id"] == "call_abc"
        assert result["x"] == 1

    def test_missing_tool_call_id_raises(self) -> None:
        tool = InjectedTool()
        with pytest.raises(ValueError, match="InjectedToolCallId"):
            tool._parse_input({"x": 1}, tool_call_id=None)


# ---------------------------------------------------------------------------
# BaseTool._parse_input with JSON schema args_schema
# ---------------------------------------------------------------------------


class TestBaseToolParseInputJsonSchema:
    """Tests for BaseTool._parse_input with JSON schema."""

    def test_string_input_with_json_schema_raises(self) -> None:
        json_schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }

        class JsonTool(BaseTool):
            name: str = "json_tool"
            description: str = "test"
            args_schema: dict = json_schema

            @override
            def _run(self, **kwargs: Any) -> str:
                return ""

        tool = JsonTool()
        with pytest.raises(ValueError, match="String tool inputs are not allowed"):
            tool._parse_input("hello", None)

    def test_dict_input_with_json_schema_passthrough(self) -> None:
        json_schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }

        class JsonTool(BaseTool):
            name: str = "json_tool"
            description: str = "test"
            args_schema: dict = json_schema

            @override
            def _run(self, **kwargs: Any) -> str:
                return ""

        tool = JsonTool()
        result = tool._parse_input({"x": 1}, None)
        assert result == {"x": 1}


# ---------------------------------------------------------------------------
# BaseTool._filter_injected_args
# ---------------------------------------------------------------------------


class TestBaseToolFilterInjectedArgs:
    """Tests for BaseTool._filter_injected_args."""

    def test_filters_run_manager_and_callbacks(self) -> None:
        tool = SimpleTool()
        filtered = tool._filter_injected_args(
            {"x": 1, "run_manager": "rm", "callbacks": "cb", "other": 2}
        )
        assert filtered == {"x": 1, "other": 2}

    def test_filters_injected_tool_arg(self) -> None:
        tool = InjectedTool()
        filtered = tool._filter_injected_args({"x": 1, "tool_call_id": "call_1"})
        assert "tool_call_id" not in filtered
        assert filtered == {"x": 1}


# ---------------------------------------------------------------------------
# BaseTool.tool_call_schema
# ---------------------------------------------------------------------------


class TestBaseToolToolCallSchema:
    """Tests for BaseTool.tool_call_schema."""

    def test_excludes_injected_args(self) -> None:
        class SchemaWithInjected(BaseModel):
            x: int
            injected: Annotated[str, InjectedToolArg]

        class InjectedSchemaBaseTool(BaseTool):
            name: str = "test"
            description: str = "test"
            args_schema: type[BaseModel] = SchemaWithInjected

            @override
            def _run(self, **kwargs: Any) -> str:
                return ""

        tool = InjectedSchemaBaseTool()
        schema = tool.tool_call_schema
        assert issubclass(schema, BaseModel)
        fields = schema.model_json_schema()["properties"]
        assert "x" in fields
        assert "injected" not in fields

    def test_json_schema_adds_description(self) -> None:
        json_schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

        class JsonSchemaTool(BaseTool):
            name: str = "test"
            description: str = "My description"
            args_schema: dict = json_schema

            @override
            def _run(self, **kwargs: Any) -> str:
                return ""

        tool = JsonSchemaTool()
        result = tool.tool_call_schema
        assert isinstance(result, dict)
        assert result["description"] == "My description"

    def test_json_schema_without_description(self) -> None:
        json_schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

        class JsonSchemaTool(BaseTool):
            name: str = "test"
            description: str = ""
            args_schema: dict = json_schema

            @override
            def _run(self, **kwargs: Any) -> str:
                return ""

        tool = JsonSchemaTool()
        result = tool.tool_call_schema
        assert isinstance(result, dict)
        assert "description" not in result


# ---------------------------------------------------------------------------
# BaseTool.get_input_schema
# ---------------------------------------------------------------------------


class TestBaseToolGetInputSchema:
    """Tests for BaseTool.get_input_schema."""

    def test_returns_args_schema_when_set(self) -> None:
        class MySchema(BaseModel):
            x: int

        class SchemaTool(BaseTool):
            name: str = "test"
            description: str = "test"
            args_schema: type[BaseModel] = MySchema

            @override
            def _run(self, x: int, **kwargs: Any) -> str:
                return str(x)

        tool = SchemaTool()
        assert tool.get_input_schema() is MySchema

    def test_infers_from_run_when_no_schema(self) -> None:
        tool = SimpleTool()
        schema = tool.get_input_schema()
        assert issubclass(schema, BaseModel)


# ---------------------------------------------------------------------------
# BaseTool.__init__ validates args_schema type
# ---------------------------------------------------------------------------


class TestBaseToolInitValidation:
    """Tests for BaseTool.__init__ validation."""

    def test_invalid_args_schema_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="args_schema must be a subclass"):

            class InvalidSchemaTool(BaseTool):
                name: str = "test"
                description: str = "test"

                @override
                def _run(self, **kwargs: Any) -> str:
                    return ""

            InvalidSchemaTool(args_schema="not_a_schema")  # type: ignore[arg-type]

    def test_none_args_schema_is_allowed(self) -> None:
        class NullSchemaTool(BaseTool):
            name: str = "test"
            description: str = "test"

            @override
            def _run(self, **kwargs: Any) -> str:
                return ""

        tool = NullSchemaTool(args_schema=None)
        assert tool.args_schema is None

    def test_dict_args_schema_is_allowed(self) -> None:
        class DictSchemaTool(BaseTool):
            name: str = "test"
            description: str = "test"

            @override
            def _run(self, **kwargs: Any) -> str:
                return ""

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        tool = DictSchemaTool(args_schema=schema)
        assert tool.args_schema == schema


# ---------------------------------------------------------------------------
# BaseToolkit
# ---------------------------------------------------------------------------


class TestBaseToolkit:
    """Tests for BaseToolkit."""

    def test_toolkit_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseToolkit()  # type: ignore[abstract]

    def test_toolkit_subclass_returns_tools(self) -> None:
        class MyToolkit(BaseToolkit):
            @override
            def get_tools(self) -> list[BaseTool]:
                return [SimpleTool()]

        toolkit = MyToolkit()
        tools = toolkit.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "simple_tool"


# ---------------------------------------------------------------------------
# get_all_basemodel_annotations
# ---------------------------------------------------------------------------


class TestGetAllBasemodelAnnotations:
    """Tests for get_all_basemodel_annotations."""

    def test_simple_model(self) -> None:
        class Simple(BaseModel):
            x: int
            y: str

        annotations = get_all_basemodel_annotations(Simple)
        assert "x" in annotations
        assert "y" in annotations

    def test_inherited_model(self) -> None:
        class Parent(BaseModel):
            x: int

        class Child(Parent):
            y: str

        annotations = get_all_basemodel_annotations(Child)
        assert "x" in annotations
        assert "y" in annotations


# ---------------------------------------------------------------------------
# _replace_type_vars
# ---------------------------------------------------------------------------


class TestReplaceTypeVars:
    """Tests for _replace_type_vars."""

    def test_replaces_typevar_from_map(self) -> None:
        T = TypeVar("T")
        result = _replace_type_vars(T, {T: int})
        assert result is int

    def test_unresolved_typevar_defaults_to_bound(self) -> None:
        T = TypeVar("T", bound=str)
        result = _replace_type_vars(T, default_to_bound=True)
        assert result is str

    def test_unresolved_typevar_defaults_to_any(self) -> None:
        T = TypeVar("T")
        result = _replace_type_vars(T, default_to_bound=True)
        assert result is Any

    def test_unresolved_typevar_preserved_when_no_default(self) -> None:
        T = TypeVar("T")
        result = _replace_type_vars(T, default_to_bound=False)
        assert result is T

    def test_plain_type_unchanged(self) -> None:
        result = _replace_type_vars(int)
        assert result is int


# ---------------------------------------------------------------------------
# BaseTool.run with run_manager callback integration
# ---------------------------------------------------------------------------


class ToolWithRunManager(BaseTool):
    """Tool that accepts run_manager."""

    name: str = "rm_tool"
    description: str = "Tool with run manager"

    @override
    def _run(
        self,
        x: str,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        return f"result: {x}"


class TestBaseToolRunManager:
    """Tests for BaseTool interaction with run_manager."""

    def test_tool_with_run_manager_invokes_successfully(self) -> None:
        tool = ToolWithRunManager()
        result = tool.invoke({"x": "test"})
        assert result == "result: test"


# ---------------------------------------------------------------------------
# BaseTool async invocation
# ---------------------------------------------------------------------------


class TestBaseToolAsyncInvoke:
    """Tests for BaseTool ainvoke/arun."""

    async def test_ainvoke_calls_arun(self) -> None:
        tool = AsyncTool()
        result = await tool.ainvoke({"x": "hello"})
        assert result == "async: hello"

    async def test_ainvoke_with_tool_call(self) -> None:
        tool = AsyncTool()
        tc: ToolCall = {
            "name": "async_tool",
            "args": {"x": "hello"},
            "id": "call_1",
            "type": "tool_call",
        }
        result = await tool.ainvoke(tc)
        assert isinstance(result, ToolMessage)
        assert result.content == "async: hello"

    async def test_arun_returns_result(self) -> None:
        tool = AsyncTool()
        result = await tool.arun({"x": "hello"})
        assert result == "async: hello"

    async def test_default_arun_falls_back_to_sync(self) -> None:
        tool = SimpleTool()
        result = await tool.ainvoke({"x": "test"})
        assert result == "result: test"
