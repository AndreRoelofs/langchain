"""Test BaseTransformOutputParser and BaseCumulativeTransformOutputParser."""

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from typing_extensions import override

from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.output_parsers.transform import (
    BaseCumulativeTransformOutputParser,
    BaseTransformOutputParser,
)
from langchain_core.outputs import ChatGeneration, Generation


class UpperCaseParser(BaseTransformOutputParser[str]):
    """Test parser that converts text to uppercase."""

    @override
    def parse(self, text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()


class ReverseParser(BaseTransformOutputParser[str]):
    """Test parser that reverses text."""

    @override
    def parse(self, text: str) -> str:
        """Reverse the text."""
        return text[::-1]


class WordCountParser(BaseTransformOutputParser[int]):
    """Test parser that counts words."""

    @override
    def parse(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())


class CumulativeUpperCaseParser(BaseCumulativeTransformOutputParser[str]):
    """Test cumulative parser that converts to uppercase."""

    @override
    def parse(self, text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()


class CumulativeWordCountParser(BaseCumulativeTransformOutputParser[int]):
    """Test cumulative parser that counts words."""

    @override
    def parse(self, text: str) -> int:
        """Count words in accumulated text."""
        return len(text.split())


class CumulativeDictParser(BaseCumulativeTransformOutputParser[dict[str, Any]]):
    """Test cumulative parser that returns a dict."""

    @override
    def parse(self, text: str) -> dict[str, Any]:
        """Parse text into a dict with word count and length."""
        return {"word_count": len(text.split()), "length": len(text)}

    @override
    def _diff(self, prev: dict[str, Any] | None, next: dict[str, Any]) -> dict[str, Any]:  # noqa: A002
        """Return the difference between two dicts."""
        if prev is None:
            return {"added": next, "removed": {}}

        added = {k: v for k, v in next.items() if k not in prev or prev[k] != v}
        removed = {k: v for k, v in prev.items() if k not in next}
        return {"added": added, "removed": removed}


# Tests for BaseTransformOutputParser


def test_base_transform_parser_parse() -> None:
    """Test BaseTransformOutputParser.parse() method."""
    parser = UpperCaseParser()
    assert parser.parse("hello world") == "HELLO WORLD"


def test_base_transform_parser_invoke_with_string() -> None:
    """Test BaseTransformOutputParser.invoke() with string input."""
    parser = UpperCaseParser()
    result = parser.invoke("test string")
    assert result == "TEST STRING"


def test_base_transform_parser_invoke_with_message() -> None:
    """Test BaseTransformOutputParser.invoke() with message input."""
    parser = UpperCaseParser()
    message = AIMessage(content="hello from ai")
    result = parser.invoke(message)
    assert result == "HELLO FROM AI"


async def test_base_transform_parser_ainvoke_with_string() -> None:
    """Test BaseTransformOutputParser.ainvoke() with string input."""
    parser = UpperCaseParser()
    result = await parser.ainvoke("async test")
    assert result == "ASYNC TEST"


async def test_base_transform_parser_ainvoke_with_message() -> None:
    """Test BaseTransformOutputParser.ainvoke() with message input."""
    parser = UpperCaseParser()
    message = AIMessage(content="async message")
    result = await parser.ainvoke(message)
    assert result == "ASYNC MESSAGE"


def test_base_transform_parser_parse_result() -> None:
    """Test BaseTransformOutputParser.parse_result() method."""
    parser = UpperCaseParser()
    generation = Generation(text="generation text")
    result = parser.parse_result([generation])
    assert result == "GENERATION TEXT"


def test_base_transform_parser_transform_strings() -> None:
    """Test BaseTransformOutputParser.transform() with string chunks."""
    parser = UpperCaseParser()
    chunks = ["hello", " ", "world"]

    def input_iter() -> Iterator[str]:
        yield from chunks

    result = list(parser.transform(input_iter()))
    assert result == ["HELLO", " ", "WORLD"]


def test_base_transform_parser_transform_messages() -> None:
    """Test BaseTransformOutputParser.transform() with message chunks."""
    parser = UpperCaseParser()
    chunks = [
        AIMessageChunk(content="hello"),
        AIMessageChunk(content=" "),
        AIMessageChunk(content="world"),
    ]

    def input_iter() -> Iterator[BaseMessage]:
        yield from chunks

    result = list(parser.transform(input_iter()))
    assert result == ["HELLO", " ", "WORLD"]


async def test_base_transform_parser_atransform_strings() -> None:
    """Test BaseTransformOutputParser.atransform() with string chunks."""
    parser = UpperCaseParser()
    chunks = ["async", " ", "test"]

    async def input_iter() -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    result = [chunk async for chunk in parser.atransform(input_iter())]
    assert result == ["ASYNC", " ", "TEST"]


async def test_base_transform_parser_atransform_messages() -> None:
    """Test BaseTransformOutputParser.atransform() with message chunks."""
    parser = UpperCaseParser()
    chunks = [
        AIMessageChunk(content="async"),
        AIMessageChunk(content=" "),
        AIMessageChunk(content="messages"),
    ]

    async def input_iter() -> AsyncIterator[BaseMessage]:
        for chunk in chunks:
            yield chunk

    result = [chunk async for chunk in parser.atransform(input_iter())]
    assert result == ["ASYNC", " ", "MESSAGES"]


def test_base_transform_parser_with_model() -> None:
    """Test BaseTransformOutputParser chained with a model."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="model output")]))
    parser = UpperCaseParser()
    chain = model | parser

    result = chain.invoke("input")
    assert result == "MODEL OUTPUT"


def test_base_transform_parser_with_model_stream() -> None:
    """Test BaseTransformOutputParser streaming with a model."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="streaming test")]))
    parser = UpperCaseParser()
    chain = model | parser

    chunks = list(chain.stream("input"))
    assert chunks == ["STREAMING", " ", "TEST"]


def test_reverse_parser() -> None:
    """Test ReverseParser implementation."""
    parser = ReverseParser()
    assert parser.parse("hello") == "olleh"
    assert parser.invoke("world") == "dlrow"


def test_word_count_parser() -> None:
    """Test WordCountParser implementation."""
    parser = WordCountParser()
    assert parser.parse("one two three") == 3
    assert parser.invoke("single") == 1
    assert parser.invoke("") == 0


def test_word_count_parser_transform() -> None:
    """Test WordCountParser with transform."""
    parser = WordCountParser()
    chunks = ["one two", " three", " four"]

    def input_iter() -> Iterator[str]:
        yield from chunks

    result = list(parser.transform(input_iter()))
    # Each chunk is parsed independently
    assert result == [2, 1, 1]


# Tests for BaseCumulativeTransformOutputParser


def test_cumulative_parser_parse() -> None:
    """Test BaseCumulativeTransformOutputParser.parse() method."""
    parser = CumulativeUpperCaseParser()
    assert parser.parse("hello world") == "HELLO WORLD"


def test_cumulative_parser_invoke() -> None:
    """Test BaseCumulativeTransformOutputParser.invoke() with string."""
    parser = CumulativeUpperCaseParser()
    result = parser.invoke("test string")
    assert result == "TEST STRING"


def test_cumulative_parser_transform_accumulates() -> None:
    """Test BaseCumulativeTransformOutputParser accumulates chunks."""
    parser = CumulativeUpperCaseParser()
    chunks = ["hello", " ", "world"]

    def input_iter() -> Iterator[str]:
        yield from chunks

    result = list(parser.transform(input_iter()))
    # Should accumulate: "hello", "hello ", "hello world"
    assert result == ["HELLO", "HELLO ", "HELLO WORLD"]


def test_cumulative_parser_transform_messages() -> None:
    """Test BaseCumulativeTransformOutputParser with message chunks."""
    parser = CumulativeUpperCaseParser()
    chunks = [
        AIMessageChunk(content="hello"),
        AIMessageChunk(content=" "),
        AIMessageChunk(content="world"),
    ]

    def input_iter() -> Iterator[BaseMessage]:
        yield from chunks

    result = list(parser.transform(input_iter()))
    assert result == ["HELLO", "HELLO ", "HELLO WORLD"]


async def test_cumulative_parser_atransform() -> None:
    """Test BaseCumulativeTransformOutputParser.atransform()."""
    parser = CumulativeUpperCaseParser()
    chunks = ["async", " ", "test"]

    async def input_iter() -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    result = [chunk async for chunk in parser.atransform(input_iter())]
    assert result == ["ASYNC", "ASYNC ", "ASYNC TEST"]


def test_cumulative_word_count_parser() -> None:
    """Test CumulativeWordCountParser accumulates correctly."""
    parser = CumulativeWordCountParser()
    chunks = ["one", " two", " three"]

    def input_iter() -> Iterator[str]:
        yield from chunks

    result = list(parser.transform(input_iter()))
    # Accumulates: "one" (1), "one two" (2), "one two three" (3)
    assert result == [1, 2, 3]


def test_cumulative_parser_with_model_stream() -> None:
    """Test BaseCumulativeTransformOutputParser with model streaming."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="hello world")]))
    parser = CumulativeUpperCaseParser()
    chain = model | parser

    chunks = list(chain.stream("input"))
    # Model splits by whitespace, parser accumulates
    assert chunks == ["HELLO", "HELLO ", "HELLO WORLD"]


def test_cumulative_parser_diff_mode() -> None:
    """Test BaseCumulativeTransformOutputParser with diff=True."""
    parser = CumulativeDictParser(diff=True)
    chunks = ["one", " two", " three"]

    def input_iter() -> Iterator[str]:
        yield from chunks

    result = list(parser.transform(input_iter()))

    # First chunk: no previous, so added contains the full dict
    assert result[0]["added"] == {"word_count": 1, "length": 3}
    assert result[0]["removed"] == {}

    # Second chunk: word_count changes from 1 to 2, length from 3 to 7
    assert result[1]["added"]["word_count"] == 2
    assert result[1]["added"]["length"] == 7

    # Third chunk: word_count changes from 2 to 3, length from 7 to 13
    assert result[2]["added"]["word_count"] == 3
    assert result[2]["added"]["length"] == 13


def test_cumulative_parser_no_diff_mode() -> None:
    """Test BaseCumulativeTransformOutputParser with diff=False (default)."""
    parser = CumulativeDictParser(diff=False)
    chunks = ["one", " two", " three"]

    def input_iter() -> Iterator[str]:
        yield from chunks

    result = list(parser.transform(input_iter()))

    # Without diff mode, should return full parsed objects
    assert result[0] == {"word_count": 1, "length": 3}
    assert result[1] == {"word_count": 2, "length": 7}
    assert result[2] == {"word_count": 3, "length": 13}


def test_cumulative_parser_skips_unchanged() -> None:
    """Test BaseCumulativeTransformOutputParser skips unchanged results."""
    parser = CumulativeWordCountParser()
    # These chunks don't add new words, so word count stays the same
    chunks = ["one", " ", " ", " "]

    def input_iter() -> Iterator[str]:
        yield from chunks

    result = list(parser.transform(input_iter()))

    # Should only yield when the count changes
    # "one" -> 1, "one " -> 1 (skip), "one  " -> 1 (skip), "one   " -> 1 (skip)
    assert result == [1]


def test_cumulative_parser_with_empty_chunks() -> None:
    """Test BaseCumulativeTransformOutputParser with empty chunks."""
    parser = CumulativeUpperCaseParser()
    chunks = ["hello", "", " ", "", "world"]

    def input_iter() -> Iterator[str]:
        yield from chunks

    result = list(parser.transform(input_iter()))
    # Empty chunks still accumulate
    assert "HELLO" in result
    assert "HELLO WORLD" in result


async def test_cumulative_parser_atransform_diff_mode() -> None:
    """Test BaseCumulativeTransformOutputParser.atransform() with diff mode."""
    parser = CumulativeDictParser(diff=True)
    chunks = ["one", " two"]

    async def input_iter() -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    result = [chunk async for chunk in parser.atransform(input_iter())]

    assert len(result) == 2
    assert result[0]["added"] == {"word_count": 1, "length": 3}
    assert result[1]["added"]["word_count"] == 2


def test_cumulative_parser_partial_support() -> None:
    """Test BaseCumulativeTransformOutputParser calls parse_result with partial=True."""

    class PartialAwareParser(BaseCumulativeTransformOutputParser[str]):
        """Parser that tracks if partial flag is set."""

        partial_calls: list[bool] = []

        @override
        def parse_result(
            self, result: list[Generation], *, partial: bool = False
        ) -> str:
            """Track partial flag and parse."""
            self.partial_calls.append(partial)
            return result[0].text.upper()

        @override
        def parse(self, text: str) -> str:
            """Convert to uppercase."""
            return text.upper()

    parser = PartialAwareParser()
    chunks = ["hello", " world"]

    def input_iter() -> Iterator[str]:
        yield from chunks

    list(parser.transform(input_iter()))

    # All calls during streaming should have partial=True
    assert all(parser.partial_calls)
    assert len(parser.partial_calls) == 2


def test_base_transform_parser_empty_input() -> None:
    """Test BaseTransformOutputParser with empty input."""
    parser = UpperCaseParser()

    def input_iter() -> Iterator[str]:
        return
        yield  # Make it a generator

    result = list(parser.transform(input_iter()))
    assert result == []


async def test_base_transform_parser_empty_async_input() -> None:
    """Test BaseTransformOutputParser.atransform() with empty input."""
    parser = UpperCaseParser()

    async def input_iter() -> AsyncIterator[str]:
        return
        yield  # Make it an async generator

    result = [chunk async for chunk in parser.atransform(input_iter())]
    assert result == []


def test_cumulative_parser_empty_input() -> None:
    """Test BaseCumulativeTransformOutputParser with empty input."""
    parser = CumulativeUpperCaseParser()

    def input_iter() -> Iterator[str]:
        return
        yield  # Make it a generator

    result = list(parser.transform(input_iter()))
    assert result == []


def test_base_transform_parser_with_config() -> None:
    """Test BaseTransformOutputParser.transform() with config."""
    parser = UpperCaseParser()
    chunks = ["hello", " world"]

    def input_iter() -> Iterator[str]:
        yield from chunks

    # Should work with config parameter
    result = list(parser.transform(input_iter(), config={}))
    assert result == ["HELLO", " WORLD"]


async def test_base_transform_parser_atransform_with_config() -> None:
    """Test BaseTransformOutputParser.atransform() with config."""
    parser = UpperCaseParser()
    chunks = ["async", " test"]

    async def input_iter() -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    # Should work with config parameter
    result = [chunk async for chunk in parser.atransform(input_iter(), config={})]
    assert result == ["ASYNC", " TEST"]
