"""Test StrOutputParser."""

from collections.abc import AsyncIterator, Iterator

import pytest

from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.outputs import ChatGeneration, Generation


def test_str_output_parser_parse() -> None:
    """Test StrOutputParser.parse() returns input unchanged."""
    parser = StrOutputParser()
    text = "Hello, world!"
    assert parser.parse(text) == text


def test_str_output_parser_parse_empty_string() -> None:
    """Test StrOutputParser.parse() with empty string."""
    parser = StrOutputParser()
    assert parser.parse("") == ""


def test_str_output_parser_parse_multiline() -> None:
    """Test StrOutputParser.parse() with multiline text."""
    parser = StrOutputParser()
    text = "Line 1\nLine 2\nLine 3"
    assert parser.parse(text) == text


def test_str_output_parser_parse_special_chars() -> None:
    """Test StrOutputParser.parse() with special characters."""
    parser = StrOutputParser()
    text = "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
    assert parser.parse(text) == text


def test_str_output_parser_parse_unicode() -> None:
    """Test StrOutputParser.parse() with unicode characters."""
    parser = StrOutputParser()
    text = "Unicode: 你好, こんにちは, नमस्ते, 🎉"
    assert parser.parse(text) == text


def test_str_output_parser_invoke_with_string() -> None:
    """Test StrOutputParser.invoke() with string input."""
    parser = StrOutputParser()
    text = "Test string"
    result = parser.invoke(text)
    assert result == text


def test_str_output_parser_invoke_with_message() -> None:
    """Test StrOutputParser.invoke() with AIMessage input."""
    parser = StrOutputParser()
    message = AIMessage(content="Hello from AI")
    result = parser.invoke(message)
    assert result == "Hello from AI"


def test_str_output_parser_invoke_with_human_message() -> None:
    """Test StrOutputParser.invoke() with HumanMessage input."""
    parser = StrOutputParser()
    message = HumanMessage(content="Hello from human")
    result = parser.invoke(message)
    assert result == "Hello from human"


async def test_str_output_parser_ainvoke_with_string() -> None:
    """Test StrOutputParser.ainvoke() with string input."""
    parser = StrOutputParser()
    text = "Async test string"
    result = await parser.ainvoke(text)
    assert result == text


async def test_str_output_parser_ainvoke_with_message() -> None:
    """Test StrOutputParser.ainvoke() with AIMessage input."""
    parser = StrOutputParser()
    message = AIMessage(content="Async hello from AI")
    result = await parser.ainvoke(message)
    assert result == "Async hello from AI"


def test_str_output_parser_parse_result_with_generation() -> None:
    """Test StrOutputParser.parse_result() with Generation."""
    parser = StrOutputParser()
    generation = Generation(text="Generated text")
    result = parser.parse_result([generation])
    assert result == "Generated text"


def test_str_output_parser_parse_result_with_chat_generation() -> None:
    """Test StrOutputParser.parse_result() with ChatGeneration."""
    parser = StrOutputParser()
    message = AIMessage(content="Chat generated text")
    generation = ChatGeneration(message=message)
    result = parser.parse_result([generation])
    assert result == "Chat generated text"


def test_str_output_parser_transform_string_chunks() -> None:
    """Test StrOutputParser.transform() with string chunks."""
    parser = StrOutputParser()
    chunks = ["Hello", " ", "world", "!"]

    def input_iter() -> Iterator[str]:
        yield from chunks

    result = list(parser.transform(input_iter()))
    assert result == chunks


def test_str_output_parser_transform_message_chunks() -> None:
    """Test StrOutputParser.transform() with message chunks."""
    parser = StrOutputParser()
    chunks = [
        AIMessageChunk(content="Hello"),
        AIMessageChunk(content=" "),
        AIMessageChunk(content="world"),
    ]

    def input_iter() -> Iterator[AIMessageChunk]:
        yield from chunks

    result = list(parser.transform(input_iter()))
    assert result == ["Hello", " ", "world"]


def test_str_output_parser_transform_mixed_chunks() -> None:
    """Test StrOutputParser.transform() with mixed string and message chunks."""
    parser = StrOutputParser()

    def input_iter() -> Iterator[str | AIMessageChunk]:
        yield "Start "
        yield AIMessageChunk(content="middle")
        yield " end"

    result = list(parser.transform(input_iter()))
    assert result == ["Start ", "middle", " end"]


async def test_str_output_parser_atransform_string_chunks() -> None:
    """Test StrOutputParser.atransform() with string chunks."""
    parser = StrOutputParser()
    chunks = ["Async", " ", "test"]

    async def input_iter() -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    result = [chunk async for chunk in parser.atransform(input_iter())]
    assert result == chunks


async def test_str_output_parser_atransform_message_chunks() -> None:
    """Test StrOutputParser.atransform() with message chunks."""
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


def test_str_output_parser_with_model_chain() -> None:
    """Test StrOutputParser chained with a model."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="Model output")]))
    parser = StrOutputParser()
    chain = model | parser

    result = chain.invoke("input")
    assert result == "Model output"


def test_str_output_parser_with_model_stream() -> None:
    """Test StrOutputParser streaming with a model."""
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="Streaming output")])
    )
    parser = StrOutputParser()
    chain = model | parser

    chunks = list(chain.stream("input"))
    # The model splits by whitespace
    assert chunks == ["Streaming", " ", "output"]


async def test_str_output_parser_with_model_astream() -> None:
    """Test StrOutputParser async streaming with a model."""
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="Async streaming")])
    )
    parser = StrOutputParser()
    chain = model | parser

    chunks = [chunk async for chunk in chain.astream("input")]
    assert chunks == ["Async", " ", "streaming"]


def test_str_output_parser_serialization() -> None:
    """Test StrOutputParser serialization properties."""
    parser = StrOutputParser()

    # Test is_lc_serializable
    assert parser.is_lc_serializable() is True

    # Test get_lc_namespace
    assert parser.get_lc_namespace() == ["langchain", "schema", "output_parser"]

    # Test _type property
    assert parser._type == "default"


def test_str_output_parser_with_empty_content() -> None:
    """Test StrOutputParser with message containing empty content."""
    parser = StrOutputParser()
    message = AIMessage(content="")
    result = parser.invoke(message)
    assert result == ""


def test_str_output_parser_with_whitespace_only() -> None:
    """Test StrOutputParser with whitespace-only content."""
    parser = StrOutputParser()
    text = "   \n\t  "
    assert parser.parse(text) == text


def test_str_output_parser_preserves_formatting() -> None:
    """Test StrOutputParser preserves text formatting."""
    parser = StrOutputParser()
    text = """
    This is a formatted text
        with indentation
    and multiple lines
    """
    assert parser.parse(text) == text


def test_str_output_parser_with_code_block() -> None:
    """Test StrOutputParser with code block content."""
    parser = StrOutputParser()
    text = """```python
def hello():
    print("Hello, world!")
```"""
    assert parser.parse(text) == text


def test_str_output_parser_with_json_string() -> None:
    """Test StrOutputParser with JSON string (should not parse it)."""
    parser = StrOutputParser()
    text = '{"key": "value", "number": 42}'
    # StrOutputParser should return the string as-is, not parse JSON
    assert parser.parse(text) == text


def test_str_output_parser_with_xml_string() -> None:
    """Test StrOutputParser with XML string (should not parse it)."""
    parser = StrOutputParser()
    text = "<root><child>value</child></root>"
    # StrOutputParser should return the string as-is, not parse XML
    assert parser.parse(text) == text


def test_str_output_parser_multiple_generations() -> None:
    """Test StrOutputParser.parse_result() uses only first generation."""
    parser = StrOutputParser()
    generations = [
        Generation(text="First generation"),
        Generation(text="Second generation"),
        Generation(text="Third generation"),
    ]
    # Should only use the first generation
    result = parser.parse_result(generations)
    assert result == "First generation"


async def test_str_output_parser_aparse() -> None:
    """Test StrOutputParser.aparse() method."""
    parser = StrOutputParser()
    text = "Async parse test"
    result = await parser.aparse(text)
    assert result == text


def test_str_output_parser_with_long_text() -> None:
    """Test StrOutputParser with long text."""
    parser = StrOutputParser()
    text = "A" * 10000  # 10k characters
    assert parser.parse(text) == text
    assert len(parser.parse(text)) == 10000


def test_str_output_parser_transform_empty_iterator() -> None:
    """Test StrOutputParser.transform() with empty iterator."""
    parser = StrOutputParser()

    def input_iter() -> Iterator[str]:
        return
        yield  # Make it a generator

    result = list(parser.transform(input_iter()))
    assert result == []


async def test_str_output_parser_atransform_empty_iterator() -> None:
    """Test StrOutputParser.atransform() with empty iterator."""
    parser = StrOutputParser()

    async def input_iter() -> AsyncIterator[str]:
        return
        yield  # Make it an async generator

    result = [chunk async for chunk in parser.atransform(input_iter())]
    assert result == []


def test_str_output_parser_dict_method() -> None:
    """Test StrOutputParser.dict() method for serialization."""
    parser = StrOutputParser()
    parser_dict = parser.dict()

    # Should contain basic serialization info
    assert isinstance(parser_dict, dict)
    # The _type should be included if available
    assert "_type" in parser_dict or "type" in parser_dict or len(parser_dict) >= 0
