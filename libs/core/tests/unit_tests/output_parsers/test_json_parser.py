"""Snapshot tests for JsonOutputParser."""

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from pydantic import BaseModel, Field

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.json import (
    JsonOutputParser,
    SimpleJsonOutputParser,
)
from langchain_core.outputs import Generation

# --- Models for testing ---


class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline")


class NestedModel(BaseModel):
    name: str
    details: dict[str, Any] = Field(default_factory=dict)


class UnicodeModel(BaseModel):
    title: str = Field(description="科学文章的标题")
    content: str = Field(description="文章内容")


# --- JsonOutputParser.parse() tests ---


class TestJsonOutputParserParse:
    """Tests for JsonOutputParser.parse() method."""

    def test_parse_valid_json(self) -> None:
        parser = JsonOutputParser()
        result = parser.parse('{"foo": "bar"}')
        assert result == {"foo": "bar"}

    def test_parse_json_in_code_block(self) -> None:
        parser = JsonOutputParser()
        result = parser.parse('```json\n{"foo": "bar"}\n```')
        assert result == {"foo": "bar"}

    def test_parse_json_in_plain_code_block(self) -> None:
        parser = JsonOutputParser()
        result = parser.parse('```\n{"foo": "bar"}\n```')
        assert result == {"foo": "bar"}

    def test_parse_json_with_surrounding_text(self) -> None:
        parser = JsonOutputParser()
        result = parser.parse('Some text\n```\n{"foo": "bar"}\n```\nMore text')
        assert result == {"foo": "bar"}

    def test_parse_invalid_json_raises(self) -> None:
        parser = JsonOutputParser()
        with pytest.raises(OutputParserException, match="Invalid json output"):
            parser.parse("not json at all")

    def test_parse_nested_json(self) -> None:
        parser = JsonOutputParser()
        text = '{"outer": {"inner": {"deep": "value"}}}'
        result = parser.parse(text)
        assert result == {"outer": {"inner": {"deep": "value"}}}

    def test_parse_json_with_array(self) -> None:
        parser = JsonOutputParser()
        text = '{"items": [1, 2, 3], "name": "test"}'
        result = parser.parse(text)
        assert result == {"items": [1, 2, 3], "name": "test"}

    def test_parse_json_with_newlines_in_values(self) -> None:
        parser = JsonOutputParser()
        text = '{"code": "line1\\nline2"}'
        result = parser.parse(text)
        assert result == {"code": "line1\nline2"}

    def test_parse_json_with_unicode(self) -> None:
        parser = JsonOutputParser()
        text = '{"name": "你好世界"}'
        result = parser.parse(text)
        assert result == {"name": "你好世界"}

    def test_parse_json_with_whitespace(self) -> None:
        parser = JsonOutputParser()
        text = '  \n  {"foo": "bar"}  \n  '
        result = parser.parse(text)
        assert result == {"foo": "bar"}

    def test_parse_json_with_boolean_and_null(self) -> None:
        parser = JsonOutputParser()
        text = '{"active": true, "deleted": false, "metadata": null}'
        result = parser.parse(text)
        assert result == {"active": True, "deleted": False, "metadata": None}

    def test_parse_json_numeric_values(self) -> None:
        parser = JsonOutputParser()
        text = '{"int": 42, "float": 3.14, "negative": -1}'
        result = parser.parse(text)
        assert result == {"int": 42, "float": 3.14, "negative": -1}


class TestJsonOutputParserParseResult:
    """Tests for JsonOutputParser.parse_result() method."""

    def test_parse_result_full(self) -> None:
        parser = JsonOutputParser()
        gen = Generation(text='{"key": "value"}')
        result = parser.parse_result([gen])
        assert result == {"key": "value"}

    def test_parse_result_partial_valid(self) -> None:
        parser = JsonOutputParser()
        gen = Generation(text='{"key": "val')
        result = parser.parse_result([gen], partial=True)
        assert result == {"key": "val"}

    def test_parse_result_partial_returns_none_for_unparseable(self) -> None:
        parser = JsonOutputParser()
        gen = Generation(text="not json")
        result = parser.parse_result([gen], partial=True)
        assert result is None

    def test_parse_result_non_partial_raises_on_invalid(self) -> None:
        parser = JsonOutputParser()
        gen = Generation(text="not json")
        with pytest.raises(OutputParserException):
            parser.parse_result([gen])


# --- JsonOutputParser.get_format_instructions() tests ---


class TestJsonOutputParserFormatInstructions:
    """Tests for JsonOutputParser.get_format_instructions()."""

    def test_no_pydantic_object(self) -> None:
        parser = JsonOutputParser()
        instructions = parser.get_format_instructions()
        assert instructions == "Return a JSON object."

    def test_with_pydantic_object(self) -> None:
        parser = JsonOutputParser(pydantic_object=Joke)
        instructions = parser.get_format_instructions()
        assert "setup" in instructions
        assert "punchline" in instructions
        assert "{schema}" not in instructions  # placeholder should be filled

    def test_unicode_preserved_in_instructions(self) -> None:
        parser = JsonOutputParser(pydantic_object=UnicodeModel)
        instructions = parser.get_format_instructions()
        assert "科学文章的标题" in instructions
        assert "文章内容" in instructions

    def test_format_instructions_do_not_alter_schema(self) -> None:
        initial_schema = Joke.model_json_schema()
        _ = JsonOutputParser(pydantic_object=Joke).get_format_instructions()
        assert Joke.model_json_schema() == initial_schema


# --- JsonOutputParser._type property ---


class TestJsonOutputParserType:
    """Tests for JsonOutputParser._type property."""

    def test_type(self) -> None:
        parser = JsonOutputParser()
        assert parser._type == "simple_json_output_parser"


# --- JsonOutputParser._get_schema() ---


class TestJsonOutputParserGetSchema:
    """Tests for JsonOutputParser._get_schema() static method."""

    def test_get_schema_v2(self) -> None:
        schema = JsonOutputParser._get_schema(Joke)
        assert "properties" in schema
        assert "setup" in schema["properties"]
        assert "punchline" in schema["properties"]


# --- SimpleJsonOutputParser alias ---


class TestSimpleJsonOutputParser:
    """Tests that SimpleJsonOutputParser is an alias for JsonOutputParser."""

    def test_is_alias(self) -> None:
        assert SimpleJsonOutputParser is JsonOutputParser

    def test_parse(self) -> None:
        parser = SimpleJsonOutputParser()
        assert parser.parse('{"a": 1}') == {"a": 1}


# --- JsonOutputParser streaming tests ---


class TestJsonOutputParserStreaming:
    """Tests for JsonOutputParser streaming via transform."""

    def test_streaming_accumulates(self) -> None:
        parser = JsonOutputParser()
        tokens = ["{", '"key"', ": ", '"val', 'ue"', "}"]

        def input_iter(_: Any) -> Iterator[str]:
            yield from tokens

        chain = input_iter | parser
        results = list(chain.stream(None))

        # Should accumulate and parse partial JSON
        assert len(results) > 0
        # Final result should be complete
        assert results[-1] == {"key": "value"}

    def test_streaming_diff_mode(self) -> None:
        parser = JsonOutputParser(diff=True)
        tokens = ['{"a":', ' "1"', ', "b":', ' "2"', "}"]

        def input_iter(_: Any) -> Iterator[str]:
            yield from tokens

        chain = input_iter | parser
        results = list(chain.stream(None))

        # Diff mode returns JSON patch operations
        assert len(results) > 0
        # Each result should be a list of patch operations
        for r in results:
            assert isinstance(r, list)

    async def test_streaming_async(self) -> None:
        parser = JsonOutputParser()
        tokens = ['{"x":', " 42", "}"]

        async def input_iter(_: Any) -> AsyncIterator[str]:
            for t in tokens:
                yield t

        chain = input_iter | parser
        results = [r async for r in chain.astream(None)]

        assert len(results) > 0
        assert results[-1] == {"x": 42}


# --- JsonOutputParser._diff() ---


class TestJsonOutputParserDiff:
    """Tests for JsonOutputParser._diff() method."""

    def test_diff_add_key(self) -> None:
        parser = JsonOutputParser()
        prev = {"a": 1}
        next_ = {"a": 1, "b": 2}
        diff = parser._diff(prev, next_)
        assert isinstance(diff, list)
        assert any(op["op"] == "add" and op["path"] == "/b" for op in diff)

    def test_diff_replace_value(self) -> None:
        parser = JsonOutputParser()
        prev = {"a": 1}
        next_ = {"a": 2}
        diff = parser._diff(prev, next_)
        assert isinstance(diff, list)
        assert any(op["op"] == "replace" for op in diff)

    def test_diff_remove_key(self) -> None:
        parser = JsonOutputParser()
        prev = {"a": 1, "b": 2}
        next_ = {"a": 1}
        diff = parser._diff(prev, next_)
        assert isinstance(diff, list)
        assert any(op["op"] == "remove" and op["path"] == "/b" for op in diff)

    def test_diff_from_none(self) -> None:
        parser = JsonOutputParser()
        diff = parser._diff(None, {"a": 1})
        assert isinstance(diff, list)

    def test_diff_no_change(self) -> None:
        parser = JsonOutputParser()
        diff = parser._diff({"a": 1}, {"a": 1})
        assert diff == []
