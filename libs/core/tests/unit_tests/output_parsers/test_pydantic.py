"""Snapshot tests for PydanticOutputParser."""

import json
from enum import Enum
from typing import Any, Literal

import pydantic
import pytest
from pydantic import BaseModel, Field

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import ParrotFakeChatModel
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.outputs import Generation
from langchain_core.prompts.prompt import PromptTemplate

# --- Test models ---


class SimpleModel(BaseModel):
    name: str
    age: int


class OptionalFieldModel(BaseModel):
    required_field: str
    optional_field: str | None = None
    default_field: int = 42


class NestedAddress(BaseModel):
    street: str
    city: str
    zip_code: str


class PersonWithAddress(BaseModel):
    name: str
    age: int
    address: NestedAddress


class ColorEnum(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class WithEnumModel(BaseModel):
    name: str
    color: ColorEnum


class WithLiteralModel(BaseModel):
    status: Literal["active", "inactive", "pending"]
    name: str


class WithListModel(BaseModel):
    tags: list[str]
    scores: list[int] = Field(default_factory=list)


class DeepNested(BaseModel):
    level3: str


class MidNested(BaseModel):
    level2: DeepNested


class TopNested(BaseModel):
    level1: MidNested


class UnicodeModel(BaseModel):
    title: str = Field(description="科学文章的标题")
    author: str = Field(description="作者姓名")


# --- PydanticOutputParser.parse() tests ---


class TestPydanticOutputParserParse:
    """Tests for PydanticOutputParser.parse() method."""

    def test_parse_simple_model(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        result = parser.parse('{"name": "Alice", "age": 30}')
        assert isinstance(result, SimpleModel)
        assert result.name == "Alice"
        assert result.age == 30

    def test_parse_in_code_block(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        text = '```json\n{"name": "Bob", "age": 25}\n```'
        result = parser.parse(text)
        assert result.name == "Bob"
        assert result.age == 25

    def test_parse_optional_fields_provided(self) -> None:
        parser = PydanticOutputParser(pydantic_object=OptionalFieldModel)
        text = '{"required_field": "hello", "optional_field": "world", "default_field": 100}'
        result = parser.parse(text)
        assert result.required_field == "hello"
        assert result.optional_field == "world"
        assert result.default_field == 100

    def test_parse_optional_fields_omitted(self) -> None:
        parser = PydanticOutputParser(pydantic_object=OptionalFieldModel)
        text = '{"required_field": "hello"}'
        result = parser.parse(text)
        assert result.required_field == "hello"
        assert result.optional_field is None
        assert result.default_field == 42

    def test_parse_nested_model(self) -> None:
        parser = PydanticOutputParser(pydantic_object=PersonWithAddress)
        text = json.dumps(
            {
                "name": "Alice",
                "age": 30,
                "address": {
                    "street": "123 Main St",
                    "city": "Springfield",
                    "zip_code": "12345",
                },
            }
        )
        result = parser.parse(text)
        assert isinstance(result, PersonWithAddress)
        assert isinstance(result.address, NestedAddress)
        assert result.address.city == "Springfield"

    def test_parse_enum_field(self) -> None:
        parser = PydanticOutputParser(pydantic_object=WithEnumModel)
        text = '{"name": "test", "color": "red"}'
        result = parser.parse(text)
        assert result.color == ColorEnum.RED

    def test_parse_literal_field(self) -> None:
        parser = PydanticOutputParser(pydantic_object=WithLiteralModel)
        text = '{"status": "active", "name": "test"}'
        result = parser.parse(text)
        assert result.status == "active"

    def test_parse_list_field(self) -> None:
        parser = PydanticOutputParser(pydantic_object=WithListModel)
        text = '{"tags": ["a", "b", "c"], "scores": [1, 2, 3]}'
        result = parser.parse(text)
        assert result.tags == ["a", "b", "c"]
        assert result.scores == [1, 2, 3]

    def test_parse_deeply_nested(self) -> None:
        parser = PydanticOutputParser(pydantic_object=TopNested)
        text = '{"level1": {"level2": {"level3": "deep_value"}}}'
        result = parser.parse(text)
        assert result.level1.level2.level3 == "deep_value"


class TestPydanticOutputParserErrors:
    """Tests for PydanticOutputParser error handling."""

    def test_invalid_json_raises(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        with pytest.raises(OutputParserException, match="Invalid json output"):
            parser.parse("not json")

    def test_validation_error_raises(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        with pytest.raises(OutputParserException, match="Failed to parse SimpleModel"):
            parser.parse('{"name": "Alice", "age": "not_a_number"}')

    def test_missing_required_field_raises(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        with pytest.raises(OutputParserException, match="Failed to parse SimpleModel"):
            parser.parse('{"name": "Alice"}')

    def test_invalid_enum_raises(self) -> None:
        parser = PydanticOutputParser(pydantic_object=WithEnumModel)
        with pytest.raises(OutputParserException, match="Failed to parse"):
            parser.parse('{"name": "test", "color": "purple"}')

    def test_invalid_literal_raises(self) -> None:
        parser = PydanticOutputParser(pydantic_object=WithLiteralModel)
        with pytest.raises(OutputParserException, match="Failed to parse"):
            parser.parse('{"status": "unknown", "name": "test"}')

    def test_parser_exception_contains_llm_output(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        with pytest.raises(OutputParserException) as exc_info:
            parser.parse('{"name": 123, "age": "bad"}')
        assert exc_info.value.llm_output is not None


class TestPydanticOutputParserParseResult:
    """Tests for PydanticOutputParser.parse_result()."""

    def test_parse_result_valid(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        gen = Generation(text='{"name": "Alice", "age": 30}')
        result = parser.parse_result([gen])
        assert isinstance(result, SimpleModel)
        assert result.name == "Alice"

    def test_parse_result_partial_valid(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        gen = Generation(text='{"name": "Alice", "age": 30}')
        result = parser.parse_result([gen], partial=True)
        assert isinstance(result, SimpleModel)

    def test_parse_result_partial_invalid_returns_none(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        gen = Generation(text="not json")
        result = parser.parse_result([gen], partial=True)
        assert result is None

    def test_parse_result_non_partial_invalid_raises(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        gen = Generation(text="not json")
        with pytest.raises(OutputParserException):
            parser.parse_result([gen])


class TestPydanticOutputParserParseObj:
    """Tests for PydanticOutputParser._parse_obj() internal method."""

    def test_parse_obj_valid(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        result = parser._parse_obj({"name": "Alice", "age": 30})
        assert isinstance(result, SimpleModel)
        assert result.name == "Alice"

    def test_parse_obj_invalid_raises(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        with pytest.raises(OutputParserException, match="Failed to parse"):
            parser._parse_obj({"name": "Alice", "age": "not_int"})


class TestPydanticOutputParserParserException:
    """Tests for PydanticOutputParser._parser_exception() internal method."""

    def test_parser_exception_message(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        exc = parser._parser_exception(
            ValueError("test error"), {"name": "Alice", "age": "bad"}
        )
        assert isinstance(exc, OutputParserException)
        assert "Failed to parse SimpleModel" in str(exc)
        assert "test error" in str(exc)
        assert exc.llm_output is not None


class TestPydanticOutputParserFormatInstructions:
    """Tests for PydanticOutputParser.get_format_instructions()."""

    def test_contains_schema_fields(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        instructions = parser.get_format_instructions()
        assert "name" in instructions
        assert "age" in instructions

    def test_does_not_contain_title(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        instructions = parser.get_format_instructions()
        # Title and type should be removed from schema
        json_part = instructions.split("```")[1] if "```" in instructions else ""
        if json_part:
            schema_dict = json.loads(json_part)
            assert "title" not in schema_dict
            assert "type" not in schema_dict

    def test_unicode_preserved(self) -> None:
        parser = PydanticOutputParser(pydantic_object=UnicodeModel)
        instructions = parser.get_format_instructions()
        assert "科学文章的标题" in instructions
        assert "作者姓名" in instructions

    def test_instructions_contain_example(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        instructions = parser.get_format_instructions()
        assert "example" in instructions.lower()
        assert "well-formatted" in instructions.lower()

    def test_does_not_alter_original_schema(self) -> None:
        original_schema = SimpleModel.model_json_schema()
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        _ = parser.get_format_instructions()
        assert SimpleModel.model_json_schema() == original_schema


class TestPydanticOutputParserProperties:
    """Tests for PydanticOutputParser properties."""

    def test_type_property(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        assert parser._type == "pydantic"

    def test_output_type(self) -> None:
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        assert parser.OutputType is SimpleModel

    def test_output_type_nested(self) -> None:
        parser = PydanticOutputParser(pydantic_object=PersonWithAddress)
        assert parser.OutputType is PersonWithAddress


class TestPydanticOutputParserChaining:
    """Tests for PydanticOutputParser chained with models."""

    def test_chain_with_prompt_and_model(self) -> None:
        prompt = PromptTemplate(
            template='{{"name": "Alice", "age": 30}}',
            input_variables=[],
        )
        model = ParrotFakeChatModel()
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        chain = prompt | model | parser

        result = chain.invoke({})
        assert isinstance(result, SimpleModel)
        assert result.name == "Alice"
        assert result.age == 30

    def test_chain_validation_failure(self) -> None:
        prompt = PromptTemplate(
            template='{{"name": 123, "age": "bad"}}',
            input_variables=[],
        )
        model = ParrotFakeChatModel()
        parser = PydanticOutputParser(pydantic_object=SimpleModel)
        chain = prompt | model | parser

        with pytest.raises(OutputParserException):
            chain.invoke({})


class TestPydanticOutputParserTypeInference:
    """Tests for PydanticOutputParser type inference."""

    def test_get_output_schema(self) -> None:
        parser = PydanticOutputParser[SimpleModel](pydantic_object=SimpleModel)
        schema = parser.get_output_schema().model_json_schema()
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
