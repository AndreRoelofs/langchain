"""Snapshot tests for OpenAI functions output parsers."""

import json
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic.v1 import BaseModel as V1BaseModel

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers.openai_functions import (
    JsonKeyOutputFunctionsParser,
    JsonOutputFunctionsParser,
    OutputFunctionsParser,
    PydanticAttrOutputFunctionsParser,
    PydanticOutputFunctionsParser,
)
from langchain_core.outputs import ChatGeneration, Generation

# --- Test helpers ---


def _make_fn_message(name: str, arguments: str, content: str = "test") -> AIMessage:
    return AIMessage(
        content=content,
        additional_kwargs={"function_call": {"name": name, "arguments": arguments}},
    )


def _make_chat_gen(message: AIMessage) -> ChatGeneration:
    return ChatGeneration(message=message)


# --- OutputFunctionsParser tests ---


class TestOutputFunctionsParser:
    """Tests for OutputFunctionsParser."""

    def test_args_only_returns_arguments_string(self) -> None:
        msg = _make_fn_message("fn", '{"a": 1}')
        parser = OutputFunctionsParser(args_only=True)
        result = parser.parse_result([_make_chat_gen(msg)])
        assert result == '{"a": 1}'

    def test_full_output_returns_function_call(self) -> None:
        msg = _make_fn_message("fn", '{"a": 1}')
        parser = OutputFunctionsParser(args_only=False)
        result = parser.parse_result([_make_chat_gen(msg)])
        assert result == {"name": "fn", "arguments": '{"a": 1}'}

    def test_non_chat_generation_raises(self) -> None:
        parser = OutputFunctionsParser()
        gen = Generation(text="plain text")
        with pytest.raises(OutputParserException, match="chat generation"):
            parser.parse_result([gen])

    def test_missing_function_call_raises(self) -> None:
        msg = AIMessage(content="no function call", additional_kwargs={})
        parser = OutputFunctionsParser()
        with pytest.raises(
            OutputParserException, match="Could not parse function call"
        ):
            parser.parse_result([_make_chat_gen(msg)])

    def test_does_not_modify_original_message(self) -> None:
        msg = _make_fn_message("fn", '{"a": 1}')
        original_kwargs = json.loads(json.dumps(msg.additional_kwargs))
        parser = OutputFunctionsParser(args_only=False)
        parser.parse_result([_make_chat_gen(msg)])
        assert msg.additional_kwargs == original_kwargs


# --- JsonOutputFunctionsParser tests ---


class TestJsonOutputFunctionsParser:
    """Tests for JsonOutputFunctionsParser."""

    def test_args_only_parses_json(self) -> None:
        msg = _make_fn_message("fn", '{"key": "value"}')
        parser = JsonOutputFunctionsParser(args_only=True)
        result = parser.parse_result([_make_chat_gen(msg)])
        assert result == {"key": "value"}

    def test_full_output_parses_arguments(self) -> None:
        msg = _make_fn_message("fn", '{"key": "value"}')
        parser = JsonOutputFunctionsParser(args_only=False)
        result = parser.parse_result([_make_chat_gen(msg)])
        assert result == {"name": "fn", "arguments": {"key": "value"}}

    def test_non_strict_allows_newlines(self) -> None:
        msg = _make_fn_message("fn", '{"code": "line1\nline2"}')
        parser = JsonOutputFunctionsParser(strict=False)
        result = parser.parse_result([_make_chat_gen(msg)])
        assert result == {"code": "line1\nline2"}

    def test_strict_rejects_newlines(self) -> None:
        msg = _make_fn_message("fn", '{"code": "line1\nline2"}')
        parser = JsonOutputFunctionsParser(strict=True)
        with pytest.raises(OutputParserException, match="Could not parse"):
            parser.parse_result([_make_chat_gen(msg)])

    def test_non_strict_allows_unicode(self) -> None:
        msg = _make_fn_message("fn", '{"text": "你好"}')
        parser = JsonOutputFunctionsParser(strict=False)
        result = parser.parse_result([_make_chat_gen(msg)])
        assert result == {"text": "你好"}

    def test_non_chat_generation_raises(self) -> None:
        parser = JsonOutputFunctionsParser()
        gen = Generation(text="text")
        with pytest.raises(OutputParserException, match="chat generation"):
            parser.parse_result([gen])

    def test_missing_function_call_raises(self) -> None:
        msg = AIMessage(content="no fn", additional_kwargs={})
        parser = JsonOutputFunctionsParser()
        with pytest.raises(OutputParserException, match="Could not parse"):
            parser.parse_result([_make_chat_gen(msg)])

    def test_missing_function_call_partial_returns_none(self) -> None:
        msg = AIMessage(content="no fn", additional_kwargs={})
        parser = JsonOutputFunctionsParser()
        result = parser.parse_result([_make_chat_gen(msg)], partial=True)
        assert result is None

    def test_invalid_json_raises(self) -> None:
        msg = _make_fn_message("fn", "not_json")
        parser = JsonOutputFunctionsParser()
        with pytest.raises(
            OutputParserException, match="Could not parse function call data"
        ):
            parser.parse_result([_make_chat_gen(msg)])

    def test_invalid_json_full_output_raises(self) -> None:
        msg = _make_fn_message("fn", "bad_json")
        parser = JsonOutputFunctionsParser(args_only=False)
        with pytest.raises(
            OutputParserException, match="Could not parse function call data"
        ):
            parser.parse_result([_make_chat_gen(msg)])

    def test_partial_valid_json(self) -> None:
        msg = _make_fn_message("fn", '{"key": "val')
        parser = JsonOutputFunctionsParser()
        result = parser.parse_result([_make_chat_gen(msg)], partial=True)
        assert result == {"key": "val"}

    def test_partial_invalid_json_returns_none(self) -> None:
        msg = _make_fn_message("fn", "{{bad")
        parser = JsonOutputFunctionsParser()
        result = parser.parse_result([_make_chat_gen(msg)], partial=True)
        assert result is None

    def test_partial_full_output(self) -> None:
        msg = _make_fn_message("fn", '{"key": "val')
        parser = JsonOutputFunctionsParser(args_only=False)
        result = parser.parse_result([_make_chat_gen(msg)], partial=True)
        assert result == {"name": "fn", "arguments": {"key": "val"}}

    def test_missing_arguments_key_returns_none(self) -> None:
        msg = AIMessage(
            content="test",
            additional_kwargs={"function_call": {"name": "fn"}},
        )
        parser = JsonOutputFunctionsParser()
        result = parser.parse_result([_make_chat_gen(msg)])
        assert result is None

    def test_multiple_results_raises(self) -> None:
        msg1 = _make_fn_message("fn", '{"a": 1}')
        msg2 = _make_fn_message("fn", '{"b": 2}')
        parser = JsonOutputFunctionsParser()
        with pytest.raises(OutputParserException, match="Expected exactly one result"):
            parser.parse_result([_make_chat_gen(msg1), _make_chat_gen(msg2)])

    def test_type_property(self) -> None:
        parser = JsonOutputFunctionsParser()
        assert parser._type == "json_functions"

    def test_parse_raises_not_implemented(self) -> None:
        parser = JsonOutputFunctionsParser()
        with pytest.raises(NotImplementedError):
            parser.parse("text")

    def test_diff_method(self) -> None:
        parser = JsonOutputFunctionsParser()
        diff = parser._diff({"a": 1}, {"a": 1, "b": 2})
        assert isinstance(diff, list)
        assert any(op["op"] == "add" for op in diff)


# --- JsonKeyOutputFunctionsParser tests ---


class TestJsonKeyOutputFunctionsParser:
    """Tests for JsonKeyOutputFunctionsParser."""

    def test_extracts_key(self) -> None:
        msg = _make_fn_message("fn", '{"key1": "val1", "key2": "val2"}')
        parser = JsonKeyOutputFunctionsParser(key_name="key1")
        result = parser.parse_result([_make_chat_gen(msg)])
        assert result == "val1"

    def test_extracts_nested_key(self) -> None:
        msg = _make_fn_message("fn", '{"data": {"nested": true}, "other": 1}')
        parser = JsonKeyOutputFunctionsParser(key_name="data")
        result = parser.parse_result([_make_chat_gen(msg)])
        assert result == {"nested": True}

    def test_missing_key_raises(self) -> None:
        msg = _make_fn_message("fn", '{"a": 1}')
        parser = JsonKeyOutputFunctionsParser(key_name="missing")
        with pytest.raises(KeyError):
            parser.parse_result([_make_chat_gen(msg)])

    def test_partial_returns_none_when_key_missing(self) -> None:
        msg = _make_fn_message("fn", '{"a": 1}')
        parser = JsonKeyOutputFunctionsParser(key_name="missing")
        result = parser.parse_result([_make_chat_gen(msg)], partial=True)
        assert result is None

    def test_partial_returns_value_when_key_present(self) -> None:
        msg = _make_fn_message("fn", '{"target": "val')
        parser = JsonKeyOutputFunctionsParser(key_name="target")
        result = parser.parse_result([_make_chat_gen(msg)], partial=True)
        assert result == "val"

    def test_partial_with_no_function_call_returns_none(self) -> None:
        msg = AIMessage(content="no fn", additional_kwargs={})
        parser = JsonKeyOutputFunctionsParser(key_name="key")
        result = parser.parse_result([_make_chat_gen(msg)], partial=True)
        assert result is None


# --- PydanticOutputFunctionsParser tests ---


class TestPydanticOutputFunctionsParser:
    """Tests for PydanticOutputFunctionsParser."""

    def test_single_schema_parses(self) -> None:
        class Model(BaseModel):
            name: str
            age: int

        msg = _make_fn_message("Model", json.dumps({"name": "Alice", "age": 30}))
        parser = PydanticOutputFunctionsParser(pydantic_schema=Model)
        result = parser.parse_result([_make_chat_gen(msg)])
        assert isinstance(result, Model)
        assert result.name == "Alice"
        assert result.age == 30

    def test_multiple_schemas_selects_by_name(self) -> None:
        class Cat(BaseModel):
            breed: str

        class Dog(BaseModel):
            species: str

        msg = _make_fn_message("cat", json.dumps({"breed": "Siamese"}))
        parser = PydanticOutputFunctionsParser(pydantic_schema={"cat": Cat, "dog": Dog})
        result = parser.parse_result([_make_chat_gen(msg)])
        assert isinstance(result, Cat)
        assert result.breed == "Siamese"

    def test_single_schema_sets_args_only_true(self) -> None:
        class Model(BaseModel):
            x: int

        parser = PydanticOutputFunctionsParser(pydantic_schema=Model)
        assert parser.args_only is True

    def test_dict_schema_sets_args_only_false(self) -> None:
        class Model(BaseModel):
            x: int

        parser = PydanticOutputFunctionsParser(pydantic_schema={"model": Model})
        assert parser.args_only is False

    def test_dict_schema_with_args_only_true_raises(self) -> None:
        class Model(BaseModel):
            x: int

        with pytest.raises(ValueError, match="args_only should be"):
            PydanticOutputFunctionsParser(
                pydantic_schema={"model": Model}, args_only=True
            )

    def test_validation_error_raises(self) -> None:
        class Model(BaseModel):
            x: int

        msg = _make_fn_message("Model", '{"x": "not_int"}')
        parser = PydanticOutputFunctionsParser(pydantic_schema=Model)
        with pytest.raises(Exception):
            parser.parse_result([_make_chat_gen(msg)])


# --- PydanticAttrOutputFunctionsParser tests ---


class TestPydanticAttrOutputFunctionsParser:
    """Tests for PydanticAttrOutputFunctionsParser."""

    def test_extracts_attribute(self) -> None:
        class Model(BaseModel):
            name: str
            value: int

        msg = _make_fn_message("Model", json.dumps({"name": "test", "value": 42}))
        parser = PydanticAttrOutputFunctionsParser(
            pydantic_schema=Model, attr_name="value"
        )
        result = parser.parse_result([_make_chat_gen(msg)])
        assert result == 42

    def test_extracts_string_attribute(self) -> None:
        class Model(BaseModel):
            name: str
            value: int

        msg = _make_fn_message("Model", json.dumps({"name": "hello", "value": 1}))
        parser = PydanticAttrOutputFunctionsParser(
            pydantic_schema=Model, attr_name="name"
        )
        result = parser.parse_result([_make_chat_gen(msg)])
        assert result == "hello"

    def test_extracts_list_attribute(self) -> None:
        class Model(BaseModel):
            items: list[str]
            count: int

        msg = _make_fn_message("Model", json.dumps({"items": ["a", "b"], "count": 2}))
        parser = PydanticAttrOutputFunctionsParser(
            pydantic_schema=Model, attr_name="items"
        )
        result = parser.parse_result([_make_chat_gen(msg)])
        assert result == ["a", "b"]

    def test_invalid_attribute_raises(self) -> None:
        class Model(BaseModel):
            x: int

        msg = _make_fn_message("Model", json.dumps({"x": 1}))
        parser = PydanticAttrOutputFunctionsParser(
            pydantic_schema=Model, attr_name="nonexistent"
        )
        with pytest.raises(AttributeError):
            parser.parse_result([_make_chat_gen(msg)])
