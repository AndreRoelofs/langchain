"""Test base prompt template functionality."""

import pytest

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import (
    BasePromptTemplate,
    aformat_document,
    format_document,
)


def test_format_document_basic() -> None:
    """Test basic document formatting."""
    doc = Document(page_content="Hello world", metadata={"page": "1"})
    prompt = PromptTemplate.from_template("Page {page}: {page_content}")
    result = format_document(doc, prompt)
    assert result == "Page 1: Hello world"


async def test_aformat_document_basic() -> None:
    """Test async document formatting."""
    doc = Document(page_content="Hello world", metadata={"page": "1"})
    prompt = PromptTemplate.from_template("Page {page}: {page_content}")
    result = await aformat_document(doc, prompt)
    assert result == "Page 1: Hello world"


def test_format_document_multiple_metadata() -> None:
    """Test document formatting with multiple metadata fields."""
    doc = Document(
        page_content="Test content",
        metadata={"page": "5", "chapter": "Introduction", "author": "John"},
    )
    prompt = PromptTemplate.from_template(
        "Chapter: {chapter}, Page {page}, Author: {author}\nContent: {page_content}"
    )
    result = format_document(doc, prompt)
    expected = "Chapter: Introduction, Page 5, Author: John\nContent: Test content"
    assert result == expected


def test_format_document_missing_metadata() -> None:
    """Test document formatting with missing metadata raises error."""
    doc = Document(page_content="Hello", metadata={"page": "1"})
    prompt = PromptTemplate.from_template("Page {page}, Chapter {chapter}: {page_content}")

    with pytest.raises(ValueError, match="missing metadata: \\['chapter'\\]"):
        format_document(doc, prompt)


def test_format_document_only_page_content() -> None:
    """Test document formatting with only page_content variable."""
    doc = Document(page_content="Just content", metadata={"extra": "ignored"})
    prompt = PromptTemplate.from_template("Content: {page_content}")
    result = format_document(doc, prompt)
    assert result == "Content: Just content"


async def test_aformat_document_missing_metadata() -> None:
    """Test async document formatting with missing metadata raises error."""
    doc = Document(page_content="Hello", metadata={"page": "1"})
    prompt = PromptTemplate.from_template("Page {page}, Chapter {chapter}: {page_content}")

    with pytest.raises(ValueError, match="missing metadata: \\['chapter'\\]"):
        await aformat_document(doc, prompt)


def test_base_prompt_template_validation_stop_in_input_variables() -> None:
    """Test that 'stop' cannot be used as an input variable."""
    with pytest.raises(ValueError, match="Cannot have an input variable named 'stop'"):
        PromptTemplate(
            template="This is a {stop} test",
            input_variables=["stop"],
        )


def test_base_prompt_template_validation_stop_in_partial_variables() -> None:
    """Test that 'stop' cannot be used as a partial variable."""
    with pytest.raises(
        ValueError, match="Cannot have an partial variable named 'stop'"
    ):
        PromptTemplate(
            template="This is a test",
            input_variables=[],
            partial_variables={"stop": "value"},
        )


def test_base_prompt_template_invoke_with_single_variable_non_dict() -> None:
    """Test invoking a single-variable template with a non-dict value."""
    template = PromptTemplate.from_template("Say {text}")
    result = template.invoke("hello")
    assert result.to_string() == "Say hello"


def test_base_prompt_template_invoke_with_multiple_variables_non_dict() -> None:
    """Test invoking a multi-variable template with non-dict raises error."""
    template = PromptTemplate.from_template("Say {text} and {more}")

    with pytest.raises(TypeError, match="Expected mapping type as input"):
        template.invoke("hello")


async def test_base_prompt_template_ainvoke_with_single_variable() -> None:
    """Test async invoking a single-variable template with a non-dict value."""
    template = PromptTemplate.from_template("Say {text}")
    result = await template.ainvoke("hello")
    assert result.to_string() == "Say hello"


def test_base_prompt_template_invoke_missing_variables() -> None:
    """Test invoking template with missing variables raises KeyError."""
    template = PromptTemplate.from_template("Say {text} and {more}")

    with pytest.raises(KeyError, match="more"):
        template.invoke({"text": "hello"})


def test_base_prompt_template_partial_basic() -> None:
    """Test partial variable substitution."""
    template = PromptTemplate.from_template("Say {greeting} {name}")
    partial_template = template.partial(greeting="Hello")

    assert partial_template.input_variables == ["name"]
    assert partial_template.partial_variables == {"greeting": "Hello"}

    result = partial_template.invoke({"name": "World"})
    assert result.to_string() == "Say Hello World"


def test_base_prompt_template_partial_with_function() -> None:
    """Test partial variable substitution with a callable."""
    def get_greeting() -> str:
        return "Hi"

    template = PromptTemplate.from_template("Say {greeting} {name}")
    partial_template = template.partial(greeting=get_greeting)

    assert partial_template.input_variables == ["name"]

    result = partial_template.invoke({"name": "World"})
    assert result.to_string() == "Say Hi World"


def test_base_prompt_template_save_with_partial_variables_raises() -> None:
    """Test that saving a prompt with partial variables raises error."""
    template = PromptTemplate.from_template("Say {greeting} {name}")
    partial_template = template.partial(greeting="Hello")

    with pytest.raises(ValueError, match="Cannot save prompt with partial variables"):
        partial_template.save("test.json")


def test_base_prompt_template_get_input_schema() -> None:
    """Test get_input_schema returns correct Pydantic model."""
    template = PromptTemplate.from_template("Say {text} and {more}")
    schema = template.get_input_schema()

    # Check that the schema has the required fields
    assert "text" in schema.model_fields
    assert "more" in schema.model_fields

    # Verify it can be instantiated
    instance = schema(text="hello", more="world")
    assert instance.text == "hello"
    assert instance.more == "world"


def test_base_prompt_template_get_input_schema_with_types() -> None:
    """Test get_input_schema with input_types specified."""
    template = PromptTemplate(
        template="Count {number}",
        input_variables=["number"],
        input_types={"number": int},
    )
    schema = template.get_input_schema()

    # Check that the schema has the correct type
    field_info = schema.model_fields["number"]
    assert field_info.annotation == int


def test_base_prompt_template_metadata_and_tags() -> None:
    """Test that metadata and tags are preserved."""
    template = PromptTemplate(
        template="Say {text}",
        input_variables=["text"],
        metadata={"version": "1.0", "author": "test"},
        tags=["test", "example"],
    )

    assert template.metadata == {"version": "1.0", "author": "test"}
    assert template.tags == ["test", "example"]


async def test_base_prompt_template_ainvoke_with_metadata() -> None:
    """Test that ainvoke includes metadata in config."""
    from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

    template = PromptTemplate(
        template="Say {text}",
        input_variables=["text"],
        metadata={"version": "1.0"},
        tags=["test"],
    )

    tracer = RunCollectorCallbackHandler()
    result = await template.ainvoke(
        {"text": "hello"},
        {"callbacks": [tracer]},
    )

    assert result.to_string() == "Say hello"
    assert len(tracer.traced_runs) == 1
    assert tracer.traced_runs[0].extra["metadata"]["version"] == "1.0"
    assert "test" in tracer.traced_runs[0].tags


def test_base_prompt_template_is_lc_serializable() -> None:
    """Test that BasePromptTemplate is serializable."""
    assert BasePromptTemplate.is_lc_serializable()


def test_base_prompt_template_lc_namespace() -> None:
    """Test that BasePromptTemplate has correct namespace."""
    assert BasePromptTemplate.get_lc_namespace() == [
        "langchain",
        "schema",
        "prompt_template",
    ]
