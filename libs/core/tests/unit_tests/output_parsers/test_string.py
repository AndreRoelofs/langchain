"""Snapshot tests for StrOutputParser."""

from collections.abc import AsyncIterator, Iterator

import pytest

from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.outputs import ChatGeneration, Generation


class TestStrOutputParserParse:
    """Tests for StrOutputParser.parse() identity behavior."""

    def test_returns_input_unchanged(self) -> None:
        parser = StrOutputParser()
        assert parser.parse("Hello, world!") == "Hello, world!"

    def test_empty_string(self) -> None:
        parser = StrOutputParser()
        assert parser.parse("") == ""

    def test_multiline_text(self) -> None:
        parser = StrOutputParser()
        text = "Line 1\nLine 2\nLine 3"
        assert parser.parse(text) == text

    def test_special_characters(self) -> None:
        parser = StrOutputParser()
        text = "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        assert parser.parse(text) == text

    def test_unicode(self) -> None:
        parser = StrOutputParser()
        text = "Unicode: 你好, こんにちは, नमस्ते, 🎉"
        assert parser.parse(text) == text

    def test_whitespace_only(self) -> None:
        parser = StrOutputParser()
        text = "   \n\t  "
        assert parser.parse(text) == text

    def test_preserves_formatting(self) -> None:
        parser = StrOutputParser()
        text = "    indented\n        double-indented\n"
        assert parser.parse(text) == text

    def test_code_block_not_parsed(self) -> None:
        parser = StrOutputParser()
        text = '```python\ndef hello():\n    print("Hello")\n```'
        assert parser.parse(text) == text

    def test_json_string_not_parsed(self) -> None:
        parser = StrOutputParser()
        text = '{"key": "value", "number": 42}'
        assert parser.parse(text) == text

    def test_xml_string_not_parsed(self) -> None:
        parser = StrOutputParser()
        text = "<root><child>value</child></root>"
        assert parser.parse(text) == text

    def test_long_text(self) -> None:
        parser = StrOutputParser()
        text = "A" * 10000
        assert parser.parse(text) == text
        assert len(parser.parse(text)) == 10000


class TestStrOutputParserInvoke:
    """Tests for StrOutputParser.invoke() with string and message inputs."""

    def test_invoke_string(self) -> None:
        parser = StrOutputParser()
        assert parser.invoke("Test string") == "Test string"

    def test_invoke_ai_message(self) -> None:
        parser = StrOutputParser()
        msg = AIMessage(content="Hello from AI")
        assert parser.invoke(msg) == "Hello from AI"

    def test_invoke_human_message(self) -> None:
        parser = StrOutputParser()
        msg = HumanMessage(content="Hello from human")
        assert parser.invoke(msg) == "Hello from human"

    def test_invoke_empty_content_message(self) -> None:
        parser = StrOutputParser()
        msg = AIMessage(content="")
        assert parser.invoke(msg) == ""


class TestStrOutputParserAsync:
    """Tests for StrOutputParser async methods."""

    async def test_ainvoke_string(self) -> None:
        parser = StrOutputParser()
        result = await parser.ainvoke("Async test")
        assert result == "Async test"

    async def test_ainvoke_message(self) -> None:
        parser = StrOutputParser()
        msg = AIMessage(content="Async AI")
        result = await parser.ainvoke(msg)
        assert result == "Async AI"

    async def test_aparse(self) -> None:
        parser = StrOutputParser()
        result = await parser.aparse("Async parse")
        assert result == "Async parse"


class TestStrOutputParserParseResult:
    """Tests for StrOutputParser.parse_result()."""

    def test_with_generation(self) -> None:
        parser = StrOutputParser()
        gen = Generation(text="Generated text")
        assert parser.parse_result([gen]) == "Generated text"

    def test_with_chat_generation(self) -> None:
        parser = StrOutputParser()
        msg = AIMessage(content="Chat generated text")
        gen = ChatGeneration(message=msg)
        assert parser.parse_result([gen]) == "Chat generated text"

    def test_multiple_generations_uses_first(self) -> None:
        parser = StrOutputParser()
        generations = [
            Generation(text="First"),
            Generation(text="Second"),
            Generation(text="Third"),
        ]
        assert parser.parse_result(generations) == "First"


class TestStrOutputParserTransform:
    """Tests for StrOutputParser.transform() streaming."""

    def test_string_chunks(self) -> None:
        parser = StrOutputParser()
        chunks = ["Hello", " ", "world", "!"]

        result = list(parser.transform(iter(chunks)))
        assert result == chunks

    def test_message_chunks(self) -> None:
        parser = StrOutputParser()
        chunks = [
            AIMessageChunk(content="Hello"),
            AIMessageChunk(content=" "),
            AIMessageChunk(content="world"),
        ]

        result = list(parser.transform(iter(chunks)))
        assert result == ["Hello", " ", "world"]

    def test_mixed_chunks(self) -> None:
        parser = StrOutputParser()

        def input_iter() -> Iterator[str | AIMessageChunk]:
            yield "Start "
            yield AIMessageChunk(content="middle")
            yield " end"

        result = list(parser.transform(input_iter()))
        assert result == ["Start ", "middle", " end"]

    def test_empty_iterator(self) -> None:
        parser = StrOutputParser()

        def input_iter() -> Iterator[str]:
            return
            yield  # noqa: RET504

        result = list(parser.transform(input_iter()))
        assert result == []


class TestStrOutputParserAtransform:
    """Tests for StrOutputParser.atransform() async streaming."""

    async def test_string_chunks(self) -> None:
        parser = StrOutputParser()
        chunks = ["Async", " ", "test"]

        async def input_iter() -> AsyncIterator[str]:
            for chunk in chunks:
                yield chunk

        result = [chunk async for chunk in parser.atransform(input_iter())]
        assert result == chunks

    async def test_message_chunks(self) -> None:
        parser = StrOutputParser()
        chunks = [
            AIMessageChunk(content="Async"),
            AIMessageChunk(content=" "),
            AIMessageChunk(content="messages"),
        ]

        async def input_iter() -> AsyncIterator[AIMessageChunk]:
            for chunk in chunks:
                yield chunk

        result = [chunk async for chunk in parser.atransform(input_iter())]
        assert result == ["Async", " ", "messages"]

    async def test_empty_iterator(self) -> None:
        parser = StrOutputParser()

        async def input_iter() -> AsyncIterator[str]:
            return
            yield  # noqa: RET504

        result = [chunk async for chunk in parser.atransform(input_iter())]
        assert result == []


class TestStrOutputParserChaining:
    """Tests for StrOutputParser chained with models."""

    def test_with_model_invoke(self) -> None:
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Model output")]))
        parser = StrOutputParser()
        chain = model | parser
        assert chain.invoke("input") == "Model output"

    def test_with_model_stream(self) -> None:
        model = GenericFakeChatModel(
            messages=iter([AIMessage(content="Streaming output")])
        )
        parser = StrOutputParser()
        chain = model | parser
        chunks = list(chain.stream("input"))
        assert chunks == ["Streaming", " ", "output"]

    async def test_with_model_astream(self) -> None:
        model = GenericFakeChatModel(
            messages=iter([AIMessage(content="Async streaming")])
        )
        parser = StrOutputParser()
        chain = model | parser
        chunks = [chunk async for chunk in chain.astream("input")]
        assert chunks == ["Async", " ", "streaming"]


class TestStrOutputParserSerialization:
    """Tests for StrOutputParser serialization properties."""

    def test_is_lc_serializable(self) -> None:
        assert StrOutputParser.is_lc_serializable() is True

    def test_get_lc_namespace(self) -> None:
        assert StrOutputParser.get_lc_namespace() == [
            "langchain",
            "schema",
            "output_parser",
        ]

    def test_type_property(self) -> None:
        parser = StrOutputParser()
        assert parser._type == "default"

    def test_dict_method(self) -> None:
        parser = StrOutputParser()
        d = parser.dict()
        assert isinstance(d, dict)
        assert d["_type"] == "default"
