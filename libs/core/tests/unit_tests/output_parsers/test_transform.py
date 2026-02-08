"""Snapshot tests for BaseTransformOutputParser and BaseCumulativeTransformOutputParser."""

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

# --- Concrete implementations for testing ---


class UpperParser(BaseTransformOutputParser[str]):
    """Converts to uppercase."""

    @override
    def parse(self, text: str) -> str:
        return text.upper()


class ReverseParser(BaseTransformOutputParser[str]):
    """Reverses text."""

    @override
    def parse(self, text: str) -> str:
        return text[::-1]


class WordCountParser(BaseTransformOutputParser[int]):
    """Counts words."""

    @override
    def parse(self, text: str) -> int:
        return len(text.split())


class CumUpperParser(BaseCumulativeTransformOutputParser[str]):
    """Cumulative uppercase."""

    @override
    def parse(self, text: str) -> str:
        return text.upper()


class CumWordCountParser(BaseCumulativeTransformOutputParser[int]):
    """Cumulative word counter."""

    @override
    def parse(self, text: str) -> int:
        return len(text.split())


class CumDictParser(BaseCumulativeTransformOutputParser[dict[str, Any]]):
    """Cumulative parser returning a dict with word count and length."""

    @override
    def parse(self, text: str) -> dict[str, Any]:
        return {"word_count": len(text.split()), "length": len(text)}

    @override
    def _diff(
        self, prev: dict[str, Any] | None, next: dict[str, Any]
    ) -> dict[str, Any]:
        if prev is None:
            return {"added": next, "removed": {}}
        added = {k: v for k, v in next.items() if k not in prev or prev[k] != v}
        removed = {k: v for k, v in prev.items() if k not in next}
        return {"added": added, "removed": removed}


# --- BaseTransformOutputParser tests ---


class TestBaseTransformOutputParserParse:
    """Tests for BaseTransformOutputParser.parse()."""

    def test_parse(self) -> None:
        assert UpperParser().parse("hello") == "HELLO"

    def test_parse_empty(self) -> None:
        assert UpperParser().parse("") == ""

    def test_reverse_parse(self) -> None:
        assert ReverseParser().parse("hello") == "olleh"

    def test_word_count_parse(self) -> None:
        assert WordCountParser().parse("one two three") == 3


class TestBaseTransformOutputParserInvoke:
    """Tests for BaseTransformOutputParser.invoke()."""

    def test_invoke_string(self) -> None:
        assert UpperParser().invoke("test") == "TEST"

    def test_invoke_message(self) -> None:
        msg = AIMessage(content="hello")
        assert UpperParser().invoke(msg) == "HELLO"

    async def test_ainvoke_string(self) -> None:
        assert await UpperParser().ainvoke("test") == "TEST"

    async def test_ainvoke_message(self) -> None:
        msg = AIMessage(content="hello")
        assert await UpperParser().ainvoke(msg) == "HELLO"


class TestBaseTransformOutputParserParseResult:
    """Tests for BaseTransformOutputParser.parse_result()."""

    def test_parse_result_generation(self) -> None:
        gen = Generation(text="text")
        assert UpperParser().parse_result([gen]) == "TEXT"

    def test_parse_result_chat_generation(self) -> None:
        msg = AIMessage(content="text")
        gen = ChatGeneration(message=msg)
        assert UpperParser().parse_result([gen]) == "TEXT"


class TestBaseTransformOutputParserTransform:
    """Tests for BaseTransformOutputParser.transform()."""

    def test_transform_strings(self) -> None:
        result = list(UpperParser().transform(iter(["hello", " ", "world"])))
        assert result == ["HELLO", " ", "WORLD"]

    def test_transform_messages(self) -> None:
        chunks = [
            AIMessageChunk(content="hello"),
            AIMessageChunk(content=" "),
            AIMessageChunk(content="world"),
        ]
        result = list(UpperParser().transform(iter(chunks)))
        assert result == ["HELLO", " ", "WORLD"]

    def test_transform_empty(self) -> None:
        def empty() -> Iterator[str]:
            return
            yield  # noqa: RET504

        assert list(UpperParser().transform(empty())) == []

    def test_transform_with_config(self) -> None:
        result = list(UpperParser().transform(iter(["test"]), config={}))
        assert result == ["TEST"]

    def test_word_count_transform(self) -> None:
        result = list(WordCountParser().transform(iter(["one two", " three"])))
        assert result == [2, 1]


class TestBaseTransformOutputParserAtransform:
    """Tests for BaseTransformOutputParser.atransform()."""

    async def test_atransform_strings(self) -> None:
        async def aiter() -> AsyncIterator[str]:
            for s in ["hello", " ", "world"]:
                yield s

        result = [c async for c in UpperParser().atransform(aiter())]
        assert result == ["HELLO", " ", "WORLD"]

    async def test_atransform_messages(self) -> None:
        async def aiter() -> AsyncIterator[AIMessageChunk]:
            for s in ["hello", " ", "world"]:
                yield AIMessageChunk(content=s)

        result = [c async for c in UpperParser().atransform(aiter())]
        assert result == ["HELLO", " ", "WORLD"]

    async def test_atransform_empty(self) -> None:
        async def aiter() -> AsyncIterator[str]:
            return
            yield  # noqa: RET504

        result = [c async for c in UpperParser().atransform(aiter())]
        assert result == []

    async def test_atransform_with_config(self) -> None:
        async def aiter() -> AsyncIterator[str]:
            yield "test"

        result = [c async for c in UpperParser().atransform(aiter(), config={})]
        assert result == ["TEST"]


class TestBaseTransformOutputParserChaining:
    """Tests for BaseTransformOutputParser chained with models."""

    def test_chain_invoke(self) -> None:
        model = GenericFakeChatModel(messages=iter([AIMessage(content="hello")]))
        chain = model | UpperParser()
        assert chain.invoke("input") == "HELLO"

    def test_chain_stream(self) -> None:
        model = GenericFakeChatModel(messages=iter([AIMessage(content="hello world")]))
        chain = model | UpperParser()
        chunks = list(chain.stream("input"))
        assert chunks == ["HELLO", " ", "WORLD"]


# --- BaseCumulativeTransformOutputParser tests ---


class TestCumulativeParserParse:
    """Tests for BaseCumulativeTransformOutputParser.parse()."""

    def test_parse(self) -> None:
        assert CumUpperParser().parse("hello") == "HELLO"

    def test_invoke(self) -> None:
        assert CumUpperParser().invoke("test") == "TEST"


class TestCumulativeParserTransformAccumulation:
    """Tests for BaseCumulativeTransformOutputParser accumulation behavior."""

    def test_accumulates_chunks(self) -> None:
        result = list(CumUpperParser().transform(iter(["hello", " ", "world"])))
        assert result == ["HELLO", "HELLO ", "HELLO WORLD"]

    def test_accumulates_messages(self) -> None:
        chunks = [
            AIMessageChunk(content="hello"),
            AIMessageChunk(content=" "),
            AIMessageChunk(content="world"),
        ]
        result = list(CumUpperParser().transform(iter(chunks)))
        assert result == ["HELLO", "HELLO ", "HELLO WORLD"]

    def test_accumulates_plain_base_messages(self) -> None:
        """Test that non-chunk BaseMessages are also properly accumulated."""
        from langchain_core.messages import AIMessage

        msgs: list[AIMessage] = [
            AIMessage(content="hello"),
            AIMessage(content=" world"),
        ]
        result = list(CumUpperParser().transform(iter(msgs)))
        # AIMessage (not chunk) should still accumulate through conversion
        assert len(result) >= 1
        assert "HELLO" in result[-1]

    def test_word_count_accumulates(self) -> None:
        result = list(CumWordCountParser().transform(iter(["one", " two", " three"])))
        assert result == [1, 2, 3]

    def test_skips_unchanged_results(self) -> None:
        # Spaces don't change word count
        result = list(CumWordCountParser().transform(iter(["one", " ", " ", " "])))
        assert result == [1]

    def test_empty_input(self) -> None:
        def empty() -> Iterator[str]:
            return
            yield  # noqa: RET504

        assert list(CumUpperParser().transform(empty())) == []

    def test_empty_chunks_still_accumulate(self) -> None:
        result = list(CumUpperParser().transform(iter(["hello", "", " ", "", "world"])))
        assert "HELLO" in result
        assert "HELLO WORLD" in result


class TestCumulativeParserDiffMode:
    """Tests for BaseCumulativeTransformOutputParser diff mode."""

    def test_diff_mode(self) -> None:
        parser = CumDictParser(diff=True)
        result = list(parser.transform(iter(["one", " two", " three"])))

        assert result[0]["added"] == {"word_count": 1, "length": 3}
        assert result[0]["removed"] == {}
        assert result[1]["added"]["word_count"] == 2
        assert result[2]["added"]["word_count"] == 3

    def test_no_diff_mode(self) -> None:
        parser = CumDictParser(diff=False)
        result = list(parser.transform(iter(["one", " two", " three"])))

        assert result[0] == {"word_count": 1, "length": 3}
        assert result[1] == {"word_count": 2, "length": 7}
        assert result[2] == {"word_count": 3, "length": 13}

    def test_diff_default_is_false(self) -> None:
        parser = CumDictParser()
        assert parser.diff is False


class TestCumulativeParserAtransform:
    """Tests for BaseCumulativeTransformOutputParser.atransform()."""

    async def test_atransform_accumulates(self) -> None:
        async def aiter() -> AsyncIterator[str]:
            for s in ["hello", " ", "world"]:
                yield s

        result = [c async for c in CumUpperParser().atransform(aiter())]
        assert result == ["HELLO", "HELLO ", "HELLO WORLD"]

    async def test_atransform_diff_mode(self) -> None:
        parser = CumDictParser(diff=True)

        async def aiter() -> AsyncIterator[str]:
            for s in ["one", " two"]:
                yield s

        result = [c async for c in parser.atransform(aiter())]
        assert len(result) == 2
        assert result[0]["added"] == {"word_count": 1, "length": 3}

    async def test_atransform_accumulates_messages(self) -> None:
        async def aiter() -> AsyncIterator[AIMessageChunk]:
            for s in ["hello", " ", "world"]:
                yield AIMessageChunk(content=s)

        result = [c async for c in CumUpperParser().atransform(aiter())]
        assert result == ["HELLO", "HELLO ", "HELLO WORLD"]


class TestCumulativeParserPartialSupport:
    """Tests for BaseCumulativeTransformOutputParser partial flag."""

    def test_parse_result_called_with_partial(self) -> None:
        class PartialTracker(BaseCumulativeTransformOutputParser[str]):
            calls: list[bool] = []

            @override
            def parse_result(
                self, result: list[Generation], *, partial: bool = False
            ) -> str:
                self.calls.append(partial)
                return result[0].text.upper()

            @override
            def parse(self, text: str) -> str:
                return text.upper()

        parser = PartialTracker()
        list(parser.transform(iter(["a", "b"])))
        assert all(parser.calls)
        assert len(parser.calls) == 2


class TestCumulativeParserChaining:
    """Tests for BaseCumulativeTransformOutputParser chained with models."""

    def test_chain_stream(self) -> None:
        model = GenericFakeChatModel(messages=iter([AIMessage(content="hello world")]))
        chain = model | CumUpperParser()
        chunks = list(chain.stream("input"))
        assert chunks == ["HELLO", "HELLO ", "HELLO WORLD"]


class TestCumulativeParserDiffNotImplemented:
    """Tests for _diff raising NotImplementedError when not overridden."""

    def test_diff_not_implemented(self) -> None:
        parser = CumUpperParser(diff=True)
        with pytest.raises(NotImplementedError):
            list(parser.transform(iter(["a", "b"])))
