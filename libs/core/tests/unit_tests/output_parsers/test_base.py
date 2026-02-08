"""Snapshot tests for base output parser classes."""

import contextlib
from typing import Any

import pytest
from typing_extensions import override

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers.base import (
    BaseGenerationOutputParser,
    BaseLLMOutputParser,
    BaseOutputParser,
)
from langchain_core.outputs import ChatGeneration, Generation

# --- Concrete subclasses for testing ---


class IntParser(BaseOutputParser[int]):
    """Parses string to int."""

    @override
    def parse(self, text: str) -> int:
        try:
            return int(text.strip())
        except ValueError as e:
            msg = f"Cannot parse '{text}' to int"
            raise OutputParserException(msg) from e

    @property
    def _type(self) -> str:
        return "int_parser"


class BoolParser(BaseOutputParser[bool]):
    """Parses YES/NO to bool."""

    true_val: str = "YES"
    false_val: str = "NO"

    @override
    def parse(self, text: str) -> bool:
        cleaned = text.strip().upper()
        if cleaned == self.true_val.upper():
            return True
        if cleaned == self.false_val.upper():
            return False
        msg = f"Expected {self.true_val} or {self.false_val}, got '{text}'"
        raise OutputParserException(msg)

    @property
    def _type(self) -> str:
        return "bool_parser"


class NoTypeParser(BaseOutputParser[str]):
    """Parser without _type property override."""

    @override
    def parse(self, text: str) -> str:
        return text


class ChatOnlyGenerationParser(BaseGenerationOutputParser[str]):
    """Generation parser that extracts content from chat generations."""

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> str:
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            msg = "This parser only works with ChatGeneration."
            raise OutputParserException(msg)
        return str(generation.message.content)


# --- BaseOutputParser tests ---


class TestBaseOutputParserParse:
    """Tests for BaseOutputParser.parse() and parse_result()."""

    def test_parse_valid_int(self) -> None:
        parser = IntParser()
        assert parser.parse("42") == 42

    def test_parse_invalid_int_raises(self) -> None:
        parser = IntParser()
        with pytest.raises(OutputParserException, match="Cannot parse"):
            parser.parse("not_a_number")

    def test_parse_with_whitespace(self) -> None:
        parser = IntParser()
        assert parser.parse("  42  ") == 42

    def test_parse_result_uses_first_generation(self) -> None:
        parser = IntParser()
        generations = [Generation(text="10"), Generation(text="20")]
        result = parser.parse_result(generations)
        assert result == 10

    def test_parse_result_single_generation(self) -> None:
        parser = IntParser()
        result = parser.parse_result([Generation(text="99")])
        assert result == 99

    def test_parse_result_with_chat_generation(self) -> None:
        parser = IntParser()
        message = AIMessage(content="55")
        gen = ChatGeneration(message=message)
        result = parser.parse_result([gen])
        assert result == 55

    def test_bool_parser_true(self) -> None:
        parser = BoolParser()
        assert parser.parse("YES") is True

    def test_bool_parser_false(self) -> None:
        parser = BoolParser()
        assert parser.parse("NO") is False

    def test_bool_parser_case_insensitive(self) -> None:
        parser = BoolParser()
        assert parser.parse("yes") is True
        assert parser.parse("no") is False

    def test_bool_parser_invalid(self) -> None:
        parser = BoolParser()
        with pytest.raises(OutputParserException, match="Expected"):
            parser.parse("MAYBE")


class TestBaseOutputParserInvoke:
    """Tests for BaseOutputParser.invoke() with different input types."""

    def test_invoke_with_string(self) -> None:
        parser = IntParser()
        assert parser.invoke("42") == 42

    def test_invoke_with_ai_message(self) -> None:
        parser = IntParser()
        msg = AIMessage(content="42")
        assert parser.invoke(msg) == 42

    def test_invoke_with_human_message(self) -> None:
        parser = IntParser()
        msg = HumanMessage(content="42")
        assert parser.invoke(msg) == 42


class TestBaseOutputParserAsync:
    """Tests for BaseOutputParser async methods."""

    async def test_ainvoke_string(self) -> None:
        parser = IntParser()
        result = await parser.ainvoke("42")
        assert result == 42

    async def test_ainvoke_message(self) -> None:
        parser = IntParser()
        msg = AIMessage(content="42")
        result = await parser.ainvoke(msg)
        assert result == 42

    async def test_aparse(self) -> None:
        parser = IntParser()
        result = await parser.aparse("42")
        assert result == 42

    async def test_aparse_result(self) -> None:
        parser = IntParser()
        result = await parser.aparse_result([Generation(text="42")])
        assert result == 42

    async def test_aparse_result_partial_flag(self) -> None:
        parser = IntParser()
        result = await parser.aparse_result([Generation(text="42")], partial=True)
        assert result == 42


class TestBaseOutputParserParseWithPrompt:
    """Tests for BaseOutputParser.parse_with_prompt()."""

    def test_parse_with_prompt_ignores_prompt(self) -> None:
        from langchain_core.prompt_values import StringPromptValue

        parser = IntParser()
        prompt = StringPromptValue(text="Give me a number")
        result = parser.parse_with_prompt("42", prompt)
        assert result == 42


class TestBaseOutputParserGetFormatInstructions:
    """Tests for BaseOutputParser.get_format_instructions()."""

    def test_get_format_instructions_raises(self) -> None:
        parser = IntParser()
        with pytest.raises(NotImplementedError):
            parser.get_format_instructions()


class TestBaseOutputParserType:
    """Tests for BaseOutputParser._type property."""

    def test_type_returns_value(self) -> None:
        parser = IntParser()
        assert parser._type == "int_parser"

    def test_type_not_implemented_raises(self) -> None:
        parser = NoTypeParser()
        with pytest.raises(
            NotImplementedError, match="_type property is not implemented"
        ):
            _ = parser._type


class TestBaseOutputParserDict:
    """Tests for BaseOutputParser.dict() method."""

    def test_dict_includes_type(self) -> None:
        parser = IntParser()
        d = parser.dict()
        assert isinstance(d, dict)
        assert d["_type"] == "int_parser"

    def test_dict_suppresses_not_implemented_type(self) -> None:
        parser = NoTypeParser()
        d = parser.dict()
        assert isinstance(d, dict)
        assert "_type" not in d


class TestBaseOutputParserOutputType:
    """Tests for BaseOutputParser.OutputType property."""

    def test_output_type_int(self) -> None:
        parser = IntParser()
        assert parser.OutputType is int

    def test_output_type_bool(self) -> None:
        parser = BoolParser()
        assert parser.OutputType is bool


class TestBaseOutputParserInputType:
    """Tests for BaseOutputParser.InputType property."""

    def test_input_type(self) -> None:
        parser = IntParser()
        # InputType should accept str or BaseMessage
        assert parser.InputType is not None


# --- BaseGenerationOutputParser tests ---


class TestBaseGenerationOutputParser:
    """Tests for BaseGenerationOutputParser."""

    def test_invoke_with_ai_message(self) -> None:
        parser = ChatOnlyGenerationParser()
        msg = AIMessage(content="hello")
        assert parser.invoke(msg) == "hello"

    def test_invoke_with_string(self) -> None:
        parser = ChatOnlyGenerationParser()
        # String input creates a plain Generation, not ChatGeneration
        with pytest.raises(
            OutputParserException, match="only works with ChatGeneration"
        ):
            parser.invoke("hello")

    async def test_ainvoke_with_ai_message(self) -> None:
        parser = ChatOnlyGenerationParser()
        msg = AIMessage(content="async hello")
        result = await parser.ainvoke(msg)
        assert result == "async hello"

    async def test_ainvoke_with_string_raises(self) -> None:
        parser = ChatOnlyGenerationParser()
        with pytest.raises(
            OutputParserException, match="only works with ChatGeneration"
        ):
            await parser.ainvoke("hello")

    def test_input_type(self) -> None:
        parser = ChatOnlyGenerationParser()
        assert parser.InputType is not None

    def test_output_type(self) -> None:
        parser = ChatOnlyGenerationParser()
        # Generic T, returns T
        assert parser.OutputType is not None


# --- BaseLLMOutputParser tests ---


class TestBaseLLMOutputParser:
    """Tests for BaseLLMOutputParser abstract interface."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseLLMOutputParser()  # type: ignore[abstract]

    async def test_aparse_result_delegates_to_sync(self) -> None:
        """Test that default aparse_result calls parse_result in executor."""

        class SimpleParser(BaseLLMOutputParser[str]):
            @override
            def parse_result(
                self, result: list[Generation], *, partial: bool = False
            ) -> str:
                return result[0].text.upper()

        parser = SimpleParser()
        result = await parser.aparse_result([Generation(text="hello")])
        assert result == "HELLO"

    async def test_aparse_result_partial_flag(self) -> None:
        """Test that partial flag is forwarded to parse_result."""

        class PartialTracker(BaseLLMOutputParser[str]):
            received_partial: bool = False

            @override
            def parse_result(
                self, result: list[Generation], *, partial: bool = False
            ) -> str:
                self.received_partial = partial
                return result[0].text

        parser = PartialTracker()
        await parser.aparse_result([Generation(text="test")], partial=True)
        assert parser.received_partial is True
