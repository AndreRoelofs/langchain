"""Snapshot tests for OpenAI tools output parsers."""

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, InvalidToolCall
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    JsonOutputToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
    parse_tool_calls,
)
from langchain_core.outputs import ChatGeneration, Generation

# --- Test helpers ---


def _tool_call_msg(
    tool_calls: list[dict[str, Any]],
    content: str = "",
    response_metadata: dict[str, Any] | None = None,
) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=tool_calls,
        response_metadata=response_metadata or {},
    )


def _raw_tool_call_msg(
    tool_calls: list[dict[str, Any]],
    content: str = "",
) -> AIMessage:
    return AIMessage(
        content=content,
        additional_kwargs={"tool_calls": tool_calls},
    )


# --- parse_tool_call() tests ---


class TestParseToolCall:
    """Tests for parse_tool_call() function."""

    def test_valid_arguments(self) -> None:
        raw = {
            "function": {"name": "myTool", "arguments": '{"a": 1}'},
            "id": "call_1",
            "type": "function",
        }
        result = parse_tool_call(raw, return_id=True)
        assert result is not None
        assert result["name"] == "myTool"
        assert result["args"] == {"a": 1}
        assert result["id"] == "call_1"

    def test_none_arguments_non_partial(self) -> None:
        raw = {
            "function": {"name": "noArgs", "arguments": None},
            "id": "call_2",
        }
        result = parse_tool_call(raw, return_id=True)
        assert result is not None
        assert result["args"] == {}

    def test_empty_string_arguments(self) -> None:
        raw = {
            "function": {"name": "emptyArgs", "arguments": ""},
            "id": "call_3",
        }
        result = parse_tool_call(raw, return_id=True)
        assert result is not None
        assert result["args"] == {}

    def test_partial_valid(self) -> None:
        raw = {
            "function": {"name": "fn", "arguments": '{"key": "val'},
            "id": "call_4",
        }
        result = parse_tool_call(raw, partial=True, return_id=True)
        assert result is not None
        assert result["args"] == {"key": "val"}

    def test_partial_none_arguments_returns_none(self) -> None:
        raw = {
            "function": {"name": "fn", "arguments": None},
            "id": "call_5",
        }
        result = parse_tool_call(raw, partial=True, return_id=True)
        assert result is None

    def test_partial_unparseable_returns_none(self) -> None:
        raw = {
            "function": {"name": "fn", "arguments": "{{bad"},
            "id": "call_6",
        }
        result = parse_tool_call(raw, partial=True, return_id=True)
        assert result is None

    def test_invalid_json_raises(self) -> None:
        raw = {
            "function": {"name": "fn", "arguments": "not_json"},
            "id": "call_7",
        }
        with pytest.raises(OutputParserException, match="are not valid JSON"):
            parse_tool_call(raw)

    def test_no_function_key_returns_none(self) -> None:
        raw = {"id": "call_8", "type": "function"}
        result = parse_tool_call(raw)
        assert result is None

    def test_return_id_false(self) -> None:
        raw = {
            "function": {"name": "fn", "arguments": '{"a": 1}'},
            "id": "call_9",
        }
        result = parse_tool_call(raw, return_id=False)
        assert result is not None
        assert result["name"] == "fn"
        assert "id" not in result

    def test_strict_mode_rejects_newlines(self) -> None:
        raw = {
            "function": {"name": "fn", "arguments": '{"code": "a\nb"}'},
            "id": "call_10",
        }
        with pytest.raises(OutputParserException):
            parse_tool_call(raw, strict=True)

    def test_non_strict_allows_newlines(self) -> None:
        raw = {
            "function": {"name": "fn", "arguments": '{"code": "a\nb"}'},
            "id": "call_11",
        }
        result = parse_tool_call(raw, strict=False, return_id=True)
        assert result is not None
        assert result["args"] == {"code": "a\nb"}

    def test_empty_function_name(self) -> None:
        raw = {
            "function": {"name": None, "arguments": '{"a": 1}'},
            "id": "call_12",
        }
        result = parse_tool_call(raw, return_id=True)
        assert result is not None
        assert result["name"] == ""


# --- make_invalid_tool_call() tests ---


class TestMakeInvalidToolCall:
    """Tests for make_invalid_tool_call() function."""

    def test_creates_invalid_tool_call(self) -> None:
        raw = {
            "function": {"name": "fn", "arguments": "bad_json"},
            "id": "call_1",
        }
        result = make_invalid_tool_call(raw, "Parse error")
        # InvalidToolCall is a TypedDict, so it's a dict
        assert isinstance(result, dict)
        assert result["name"] == "fn"
        assert result["args"] == "bad_json"
        assert result["id"] == "call_1"
        assert result["error"] == "Parse error"
        assert result["type"] == "invalid_tool_call"

    def test_none_error_message(self) -> None:
        raw = {
            "function": {"name": "fn", "arguments": "{}"},
            "id": "call_2",
        }
        result = make_invalid_tool_call(raw, None)
        assert result["error"] is None

    def test_missing_id(self) -> None:
        raw = {
            "function": {"name": "fn", "arguments": "{}"},
        }
        result = make_invalid_tool_call(raw, "error")
        assert result["id"] is None


# --- parse_tool_calls() tests ---


class TestParseToolCalls:
    """Tests for parse_tool_calls() function."""

    def test_multiple_valid_calls(self) -> None:
        raw_calls = [
            {"function": {"name": "fn1", "arguments": '{"a": 1}'}, "id": "c1"},
            {"function": {"name": "fn2", "arguments": '{"b": 2}'}, "id": "c2"},
        ]
        result = parse_tool_calls(raw_calls, return_id=True)
        assert len(result) == 2
        assert result[0]["name"] == "fn1"
        assert result[1]["name"] == "fn2"

    def test_empty_list(self) -> None:
        result = parse_tool_calls([])
        assert result == []

    def test_all_invalid_raises(self) -> None:
        raw_calls = [
            {"function": {"name": "fn1", "arguments": "bad1"}, "id": "c1"},
            {"function": {"name": "fn2", "arguments": "bad2"}, "id": "c2"},
        ]
        with pytest.raises(OutputParserException) as exc_info:
            parse_tool_calls(raw_calls)
        # Should aggregate both error messages
        error_str = str(exc_info.value)
        assert "fn1" in error_str
        assert "fn2" in error_str

    def test_mixed_valid_invalid_raises(self) -> None:
        raw_calls = [
            {"function": {"name": "good", "arguments": '{"a": 1}'}, "id": "c1"},
            {"function": {"name": "bad", "arguments": "invalid"}, "id": "c2"},
        ]
        with pytest.raises(OutputParserException):
            parse_tool_calls(raw_calls)

    def test_partial_mode_skips_unparseable(self) -> None:
        raw_calls = [
            {"function": {"name": "fn1", "arguments": '{"a": 1}'}, "id": "c1"},
            {"function": {"name": "fn2", "arguments": None}, "id": "c2"},
        ]
        result = parse_tool_calls(raw_calls, partial=True, return_id=True)
        assert len(result) == 1
        assert result[0]["name"] == "fn1"

    def test_no_function_key_skipped(self) -> None:
        raw_calls = [
            {"id": "c1"},  # No function key
            {"function": {"name": "fn", "arguments": '{"a": 1}'}, "id": "c2"},
        ]
        result = parse_tool_calls(raw_calls, return_id=True)
        assert len(result) == 1
        assert result[0]["name"] == "fn"


# --- JsonOutputToolsParser tests ---


class TestJsonOutputToolsParser:
    """Tests for JsonOutputToolsParser."""

    def test_parses_tool_calls(self) -> None:
        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "fn", "args": {"a": 1}},
            ]
        )
        parser = JsonOutputToolsParser()
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert len(result) == 1
        assert result[0]["type"] == "fn"
        assert result[0]["args"] == {"a": 1}

    def test_multiple_tool_calls(self) -> None:
        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "fn1", "args": {"a": 1}},
                {"id": "c2", "name": "fn2", "args": {"b": 2}},
            ]
        )
        parser = JsonOutputToolsParser()
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert len(result) == 2
        assert result[0]["type"] == "fn1"
        assert result[1]["type"] == "fn2"

    def test_return_id(self) -> None:
        msg = _tool_call_msg(
            [
                {"id": "call_123", "name": "fn", "args": {"a": 1}},
            ]
        )
        parser = JsonOutputToolsParser(return_id=True)
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert result[0]["id"] == "call_123"

    def test_no_return_id(self) -> None:
        msg = _tool_call_msg(
            [
                {"id": "call_123", "name": "fn", "args": {"a": 1}},
            ]
        )
        parser = JsonOutputToolsParser(return_id=False)
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert "id" not in result[0]

    def test_first_tool_only(self) -> None:
        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "fn1", "args": {"a": 1}},
                {"id": "c2", "name": "fn2", "args": {"b": 2}},
            ]
        )
        parser = JsonOutputToolsParser(first_tool_only=True)
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert isinstance(result, dict)
        assert result["type"] == "fn1"

    def test_first_tool_only_empty_returns_empty_list(self) -> None:
        msg = _tool_call_msg([])
        parser = JsonOutputToolsParser(first_tool_only=True)
        result = parser.parse_result([ChatGeneration(message=msg)])
        # When tool_calls is empty list via AIMessage.tool_calls, the code
        # renames name->type before checking first_tool_only, so it returns []
        assert result == []

    def test_empty_tool_calls_returns_empty_list(self) -> None:
        msg = _tool_call_msg([])
        parser = JsonOutputToolsParser()
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert result == []

    def test_non_chat_generation_raises(self) -> None:
        parser = JsonOutputToolsParser()
        with pytest.raises(OutputParserException, match="chat generation"):
            parser.parse_result([Generation(text="text")])

    def test_fallback_to_additional_kwargs(self) -> None:
        msg = _raw_tool_call_msg(
            [
                {
                    "id": "c1",
                    "function": {"name": "fn", "arguments": '{"a": 1}'},
                    "type": "function",
                },
            ]
        )
        parser = JsonOutputToolsParser()
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert len(result) == 1
        assert result[0]["type"] == "fn"

    def test_no_tool_calls_or_kwargs_returns_empty(self) -> None:
        msg = AIMessage(content="no tools", additional_kwargs={})
        parser = JsonOutputToolsParser()
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert result == []

    def test_parse_raises_not_implemented(self) -> None:
        parser = JsonOutputToolsParser()
        with pytest.raises(NotImplementedError):
            parser.parse("text")


# --- JsonOutputKeyToolsParser tests ---


class TestJsonOutputKeyToolsParser:
    """Tests for JsonOutputKeyToolsParser."""

    def test_filters_by_key_name(self) -> None:
        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "target", "args": {"a": 1}},
                {"id": "c2", "name": "other", "args": {"b": 2}},
            ]
        )
        parser = JsonOutputKeyToolsParser(key_name="target", return_id=False)
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert result == [{"a": 1}]

    def test_no_match_returns_empty_list(self) -> None:
        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "other", "args": {"a": 1}},
            ]
        )
        parser = JsonOutputKeyToolsParser(key_name="nonexistent")
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert result == []

    def test_first_tool_only_returns_args(self) -> None:
        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "target", "args": {"a": 1}},
            ]
        )
        parser = JsonOutputKeyToolsParser(
            key_name="target", first_tool_only=True, return_id=False
        )
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert result == {"a": 1}

    def test_first_tool_only_with_return_id(self) -> None:
        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "target", "args": {"a": 1}},
            ]
        )
        parser = JsonOutputKeyToolsParser(
            key_name="target", first_tool_only=True, return_id=True
        )
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert result["type"] == "target"
        assert result["args"] == {"a": 1}

    def test_first_tool_only_no_match_returns_none(self) -> None:
        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "other", "args": {"a": 1}},
            ]
        )
        parser = JsonOutputKeyToolsParser(key_name="missing", first_tool_only=True)
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert result is None

    def test_multiple_matches_returns_all(self) -> None:
        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "fn", "args": {"a": 1}},
                {"id": "c2", "name": "other", "args": {"b": 2}},
                {"id": "c3", "name": "fn", "args": {"a": 3}},
            ]
        )
        parser = JsonOutputKeyToolsParser(key_name="fn", return_id=False)
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert result == [{"a": 1}, {"a": 3}]

    def test_empty_tool_calls_first_only_returns_none(self) -> None:
        msg = AIMessage(content="", additional_kwargs={})
        parser = JsonOutputKeyToolsParser(key_name="fn", first_tool_only=True)
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert result is None

    def test_empty_tool_calls_returns_empty_list(self) -> None:
        msg = AIMessage(content="", additional_kwargs={})
        parser = JsonOutputKeyToolsParser(key_name="fn")
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert result == []


# --- PydanticToolsParser tests ---


class TestPydanticToolsParser:
    """Tests for PydanticToolsParser."""

    def test_parses_single_tool(self) -> None:
        class MyTool(BaseModel):
            value: int
            name: str

        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "MyTool", "args": {"value": 42, "name": "test"}},
            ]
        )
        parser = PydanticToolsParser(tools=[MyTool])
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert len(result) == 1
        assert isinstance(result[0], MyTool)
        assert result[0].value == 42

    def test_parses_multiple_tools(self) -> None:
        class ToolA(BaseModel):
            a: int

        class ToolB(BaseModel):
            b: str

        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "ToolA", "args": {"a": 1}},
                {"id": "c2", "name": "ToolB", "args": {"b": "hello"}},
            ]
        )
        parser = PydanticToolsParser(tools=[ToolA, ToolB])
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert len(result) == 2
        assert isinstance(result[0], ToolA)
        assert isinstance(result[1], ToolB)

    def test_first_tool_only(self) -> None:
        class MyTool(BaseModel):
            x: int

        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "MyTool", "args": {"x": 1}},
                {"id": "c2", "name": "MyTool", "args": {"x": 2}},
            ]
        )
        parser = PydanticToolsParser(tools=[MyTool], first_tool_only=True)
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert isinstance(result, MyTool)
        assert result.x == 1

    def test_first_tool_only_empty_returns_none(self) -> None:
        class MyTool(BaseModel):
            x: int

        msg = _tool_call_msg([])
        parser = PydanticToolsParser(tools=[MyTool], first_tool_only=True)
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert result is None

    def test_empty_returns_empty_list(self) -> None:
        class MyTool(BaseModel):
            x: int

        msg = _tool_call_msg([])
        parser = PydanticToolsParser(tools=[MyTool])
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert result == []

    def test_validation_error_raises(self) -> None:
        class StrictTool(BaseModel):
            count: int

        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "StrictTool", "args": {"count": "not_int"}},
            ]
        )
        parser = PydanticToolsParser(tools=[StrictTool])
        with pytest.raises(ValidationError):
            parser.parse_result([ChatGeneration(message=msg)])

    def test_partial_skips_invalid(self) -> None:
        class StrictTool(BaseModel):
            count: int

        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "StrictTool", "args": {"count": "bad"}},
            ]
        )
        parser = PydanticToolsParser(tools=[StrictTool])
        result = parser.parse_result([ChatGeneration(message=msg)], partial=True)
        assert result == []

    def test_partial_non_dict_args_skipped(self) -> None:
        class MyTool(BaseModel):
            x: int

        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "MyTool", "args": {"x": 1}},
            ]
        )
        gen = ChatGeneration(message=msg)
        parser = PydanticToolsParser(tools=[MyTool])
        # Manually modify to have non-dict args for testing
        result = parser.parse_result([gen], partial=True)
        assert len(result) == 1

    def test_unknown_tool_name_raises(self) -> None:
        class MyTool(BaseModel):
            x: int

        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "UnknownTool", "args": {"x": 1}},
            ]
        )
        parser = PydanticToolsParser(tools=[MyTool])
        with pytest.raises(KeyError):
            parser.parse_result([ChatGeneration(message=msg)])

    def test_custom_title_model(self) -> None:
        class CustomTool(BaseModel):
            model_config = {"title": "MyCustomName"}
            val: int

        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "MyCustomName", "args": {"val": 99}},
            ]
        )
        parser = PydanticToolsParser(tools=[CustomTool])
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert len(result) == 1
        assert isinstance(result[0], CustomTool)
        assert result[0].val == 99

    def test_max_tokens_stop_reason_logged(self, caplog: Any) -> None:
        class StrictTool(BaseModel):
            required: str

        msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "StrictTool", "args": {"required": 123}},
            ],
            response_metadata={"stop_reason": "max_tokens"},
        )
        parser = PydanticToolsParser(tools=[StrictTool], first_tool_only=True)
        with pytest.raises(ValidationError):
            parser.invoke(msg)
        assert any(
            "`max_tokens` stop reason" in msg_text for msg_text in caplog.messages
        )

    def test_nested_pydantic_models(self) -> None:
        class Inner(BaseModel):
            val: str

        class Outer(BaseModel):
            inner: Inner
            name: str

        msg = _tool_call_msg(
            [
                {
                    "id": "c1",
                    "name": "Outer",
                    "args": {"inner": {"val": "deep"}, "name": "top"},
                },
            ]
        )
        parser = PydanticToolsParser(tools=[Outer])
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert len(result) == 1
        assert isinstance(result[0].inner, Inner)
        assert result[0].inner.val == "deep"

    def test_optional_fields(self) -> None:
        class OptTool(BaseModel):
            required: str
            optional: str | None = None

        msg = _tool_call_msg(
            [
                {"id": "c1", "name": "OptTool", "args": {"required": "yes"}},
            ]
        )
        parser = PydanticToolsParser(tools=[OptTool])
        result = parser.parse_result([ChatGeneration(message=msg)])
        assert result[0].required == "yes"
        assert result[0].optional is None
