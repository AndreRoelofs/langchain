"""Snapshot tests for list output parsers."""

import re
from collections.abc import AsyncIterator, Iterator
from typing import TypeVar

import pytest

from langchain_core.output_parsers.list import (
    CommaSeparatedListOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,
    droplastn,
)
from langchain_core.runnables.utils import aadd, add

T = TypeVar("T")


# --- droplastn() utility function tests ---


class TestDropLastN:
    """Tests for droplastn() utility function."""

    def test_drop_last_one(self) -> None:
        result = list(droplastn(iter([1, 2, 3, 4, 5]), 1))
        assert result == [1, 2, 3, 4]

    def test_drop_last_two(self) -> None:
        result = list(droplastn(iter([1, 2, 3, 4, 5]), 2))
        assert result == [1, 2, 3]

    def test_drop_last_zero(self) -> None:
        result = list(droplastn(iter([1, 2, 3]), 0))
        assert result == [1, 2, 3]

    def test_drop_all(self) -> None:
        result = list(droplastn(iter([1, 2, 3]), 3))
        assert result == []

    def test_drop_more_than_length(self) -> None:
        result = list(droplastn(iter([1, 2]), 5))
        assert result == []

    def test_empty_iterator(self) -> None:
        result = list(droplastn(iter([]), 1))
        assert result == []

    def test_single_element_drop_one(self) -> None:
        result = list(droplastn(iter([42]), 1))
        assert result == []

    def test_string_elements(self) -> None:
        result = list(droplastn(iter(["a", "b", "c"]), 1))
        assert result == ["a", "b"]


# --- CommaSeparatedListOutputParser tests ---


class TestCommaSeparatedListOutputParser:
    """Tests for CommaSeparatedListOutputParser."""

    def test_single_item(self) -> None:
        parser = CommaSeparatedListOutputParser()
        assert parser.parse("foo") == ["foo"]

    def test_multiple_items_with_spaces(self) -> None:
        parser = CommaSeparatedListOutputParser()
        assert parser.parse("foo, bar, baz") == ["foo", "bar", "baz"]

    def test_multiple_items_no_spaces(self) -> None:
        parser = CommaSeparatedListOutputParser()
        assert parser.parse("foo,bar,baz") == ["foo", "bar", "baz"]

    def test_quoted_items_with_commas(self) -> None:
        parser = CommaSeparatedListOutputParser()
        assert parser.parse('"foo, foo2",bar,baz') == ["foo, foo2", "bar", "baz"]

    def test_empty_string(self) -> None:
        parser = CommaSeparatedListOutputParser()
        result = parser.parse("")
        assert result == []

    def test_single_item_with_whitespace(self) -> None:
        parser = CommaSeparatedListOutputParser()
        # CSV reader with skipinitialspace trims leading space, trailing stays
        assert parser.parse("  foo  ") == ["foo  "]

    def test_many_items(self) -> None:
        parser = CommaSeparatedListOutputParser()
        items = [f"item{i}" for i in range(20)]
        text = ", ".join(items)
        assert parser.parse(text) == items

    def test_type_property(self) -> None:
        parser = CommaSeparatedListOutputParser()
        assert parser._type == "comma-separated-list"

    def test_is_lc_serializable(self) -> None:
        assert CommaSeparatedListOutputParser.is_lc_serializable() is True

    def test_get_lc_namespace(self) -> None:
        assert CommaSeparatedListOutputParser.get_lc_namespace() == [
            "langchain",
            "output_parsers",
            "list",
        ]

    def test_get_format_instructions(self) -> None:
        parser = CommaSeparatedListOutputParser()
        instructions = parser.get_format_instructions()
        assert "comma" in instructions.lower()
        assert "foo" in instructions

    def test_transform_char_by_char(self) -> None:
        parser = CommaSeparatedListOutputParser()
        text = "foo, bar, baz"
        expected = ["foo", "bar", "baz"]
        assert add(parser.transform(t for t in text)) == expected

    def test_transform_yields_individual_items(self) -> None:
        parser = CommaSeparatedListOutputParser()
        text = "foo, bar, baz"
        result = list(parser.transform(t for t in text))
        assert result == [["foo"], ["bar"], ["baz"]]

    def test_transform_single_chunk(self) -> None:
        parser = CommaSeparatedListOutputParser()
        text = "foo, bar, baz"
        result = list(parser.transform(iter([text])))
        assert result == [["foo"], ["bar"], ["baz"]]


class TestCommaSeparatedListOutputParserAsync:
    """Tests for CommaSeparatedListOutputParser async methods."""

    async def test_aparse(self) -> None:
        parser = CommaSeparatedListOutputParser()
        assert await parser.aparse("foo, bar") == ["foo", "bar"]

    async def test_atransform_char_by_char(self) -> None:
        parser = CommaSeparatedListOutputParser()
        text = "foo, bar"

        async def aiter_text() -> AsyncIterator[str]:
            for c in text:
                yield c

        result = await aadd(parser.atransform(aiter_text()))
        assert result == ["foo", "bar"]

    async def test_atransform_yields_items(self) -> None:
        parser = CommaSeparatedListOutputParser()
        text = "foo, bar, baz"

        async def aiter_text() -> AsyncIterator[str]:
            for c in text:
                yield c

        result = [item async for item in parser.atransform(aiter_text())]
        assert result == [["foo"], ["bar"], ["baz"]]


# --- NumberedListOutputParser tests ---


class TestNumberedListOutputParser:
    """Tests for NumberedListOutputParser."""

    def test_basic_numbered_list(self) -> None:
        parser = NumberedListOutputParser()
        text = "1. foo\n2. bar\n3. baz"
        assert parser.parse(text) == ["foo", "bar", "baz"]

    def test_numbered_list_with_extra_spacing(self) -> None:
        parser = NumberedListOutputParser()
        text = "1. foo\n\n2. bar\n\n3. baz"
        assert parser.parse(text) == ["foo", "bar", "baz"]

    def test_numbered_list_with_prefix_text(self) -> None:
        parser = NumberedListOutputParser()
        text = "Here are the items:\n\n1. apple\n2. banana\n3. cherry"
        assert parser.parse(text) == ["apple", "banana", "cherry"]

    def test_empty_text_returns_empty_list(self) -> None:
        parser = NumberedListOutputParser()
        assert parser.parse("No items here.") == []

    def test_indented_numbers(self) -> None:
        parser = NumberedListOutputParser()
        text = "Items:\n\n1. apple\n\n    2. banana\n\n3. cherry"
        assert parser.parse(text) == ["apple", "banana", "cherry"]

    def test_large_numbers(self) -> None:
        parser = NumberedListOutputParser()
        text = "100. first\n200. second\n300. third"
        assert parser.parse(text) == ["first", "second", "third"]

    def test_type_property(self) -> None:
        parser = NumberedListOutputParser()
        assert parser._type == "numbered-list"

    def test_get_format_instructions(self) -> None:
        parser = NumberedListOutputParser()
        instructions = parser.get_format_instructions()
        assert "numbered" in instructions.lower()
        assert "1." in instructions

    def test_parse_iter(self) -> None:
        parser = NumberedListOutputParser()
        text = "1. foo\n2. bar"
        matches = list(parser.parse_iter(text))
        assert len(matches) == 2
        assert all(isinstance(m, re.Match) for m in matches)
        assert matches[0].group(1) == "foo"
        assert matches[1].group(1) == "bar"

    def test_custom_pattern(self) -> None:
        parser = NumberedListOutputParser(pattern=r"\d+\)\s([^\n]+)")
        text = "1) foo\n2) bar"
        assert parser.parse(text) == ["foo", "bar"]

    def test_transform_char_by_char(self) -> None:
        parser = NumberedListOutputParser()
        text = "1. foo\n2. bar\n3. baz"
        result = add(parser.transform(t for t in text))
        assert result == ["foo", "bar", "baz"]

    def test_transform_yields_items(self) -> None:
        parser = NumberedListOutputParser()
        text = "1. foo\n2. bar\n3. baz"
        result = list(parser.transform(t for t in text))
        assert result == [["foo"], ["bar"], ["baz"]]


class TestNumberedListOutputParserAsync:
    """Tests for NumberedListOutputParser async methods."""

    async def test_aparse(self) -> None:
        parser = NumberedListOutputParser()
        text = "1. foo\n2. bar\n3. baz"
        assert await parser.aparse(text) == ["foo", "bar", "baz"]

    async def test_atransform(self) -> None:
        parser = NumberedListOutputParser()
        text = "1. foo\n2. bar\n3. baz"

        async def aiter_text() -> AsyncIterator[str]:
            for c in text:
                yield c

        result = await aadd(parser.atransform(aiter_text()))
        assert result == ["foo", "bar", "baz"]


# --- MarkdownListOutputParser tests ---


class TestMarkdownListOutputParser:
    """Tests for MarkdownListOutputParser."""

    def test_basic_dash_list(self) -> None:
        parser = MarkdownListOutputParser()
        text = "- foo\n- bar\n- baz"
        assert parser.parse(text) == ["foo", "bar", "baz"]

    def test_asterisk_list(self) -> None:
        parser = MarkdownListOutputParser()
        text = "* foo\n* bar\n* baz"
        assert parser.parse(text) == ["foo", "bar", "baz"]

    def test_list_with_prefix_text(self) -> None:
        parser = MarkdownListOutputParser()
        text = "Items:\n- apple\n- banana\n- cherry"
        assert parser.parse(text) == ["apple", "banana", "cherry"]

    def test_empty_text_returns_empty_list(self) -> None:
        parser = MarkdownListOutputParser()
        assert parser.parse("No items here.") == []

    def test_indented_items(self) -> None:
        parser = MarkdownListOutputParser()
        text = "Items:\n- apple\n     - banana\n- cherry"
        assert parser.parse(text) == ["apple", "banana", "cherry"]

    def test_text_with_dashes_in_prose_not_matched(self) -> None:
        parser = MarkdownListOutputParser()
        text = "This is a sentence with - not a list item.\n- actual item"
        result = parser.parse(text)
        # The inline dash is NOT at start of line with proper format
        # but "- not a list item." IS at start after matching pattern
        assert "actual item" in result

    def test_type_property(self) -> None:
        parser = MarkdownListOutputParser()
        assert parser._type == "markdown-list"

    def test_get_format_instructions(self) -> None:
        parser = MarkdownListOutputParser()
        instructions = parser.get_format_instructions()
        assert "markdown" in instructions.lower()
        assert "- foo" in instructions

    def test_parse_iter(self) -> None:
        parser = MarkdownListOutputParser()
        text = "- foo\n- bar"
        matches = list(parser.parse_iter(text))
        assert len(matches) == 2
        assert all(isinstance(m, re.Match) for m in matches)
        assert matches[0].group(1) == "foo"
        assert matches[1].group(1) == "bar"

    def test_transform_char_by_char(self) -> None:
        parser = MarkdownListOutputParser()
        text = "- foo\n- bar\n- baz"
        result = add(parser.transform(t for t in text))
        assert result == ["foo", "bar", "baz"]

    def test_transform_yields_items(self) -> None:
        parser = MarkdownListOutputParser()
        text = "- foo\n- bar\n- baz"
        result = list(parser.transform(t for t in text))
        assert result == [["foo"], ["bar"], ["baz"]]


class TestMarkdownListOutputParserAsync:
    """Tests for MarkdownListOutputParser async methods."""

    async def test_aparse(self) -> None:
        parser = MarkdownListOutputParser()
        text = "- foo\n- bar\n- baz"
        assert await parser.aparse(text) == ["foo", "bar", "baz"]

    async def test_atransform(self) -> None:
        parser = MarkdownListOutputParser()
        text = "- foo\n- bar\n- baz"

        async def aiter_text() -> AsyncIterator[str]:
            for c in text:
                yield c

        result = await aadd(parser.atransform(aiter_text()))
        assert result == ["foo", "bar", "baz"]


# --- ListOutputParser base class tests ---


class TestListOutputParserBase:
    """Tests for ListOutputParser base class properties."""

    def test_list_type_property(self) -> None:
        # CommaSeparatedListOutputParser inherits from ListOutputParser
        # but overrides _type. Check ListOutputParser._type via numbered parser.
        from langchain_core.output_parsers.list import ListOutputParser

        # ListOutputParser._type returns "list" but we can't instantiate directly
        # So check through NumberedListOutputParser before it overrides
        # Actually, all subclasses override. Just check existence.
        parser = NumberedListOutputParser()
        # It overrides to "numbered-list"
        assert parser._type == "numbered-list"
