"""Expanded tests for PromptTemplate functionality."""

from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from langchain_core.prompts.prompt import PromptTemplate

# --- PromptTemplate properties ---


def test_prompt_type_property() -> None:
    """Test _prompt_type returns 'prompt'."""
    template = PromptTemplate.from_template("Hello {name}")
    assert template._prompt_type == "prompt"


def test_lc_attributes_property() -> None:
    """Test lc_attributes includes template_format."""
    template = PromptTemplate.from_template("Hello {name}")
    assert template.lc_attributes == {"template_format": "f-string"}


def test_lc_attributes_mustache() -> None:
    """Test lc_attributes with mustache format."""
    template = PromptTemplate.from_template(
        "Hello {{name}}", template_format="mustache"
    )
    assert template.lc_attributes == {"template_format": "mustache"}


def test_lc_namespace() -> None:
    """Test get_lc_namespace returns correct namespace."""
    assert PromptTemplate.get_lc_namespace() == ["langchain", "prompts", "prompt"]


# --- pre_init_validation ---


def test_pre_init_validation_auto_infers_variables() -> None:
    """Test that input_variables are auto-inferred from template."""
    template = PromptTemplate(template="Hello {name} {greeting}")
    assert sorted(template.input_variables) == ["greeting", "name"]


def test_pre_init_validation_with_partial_variables() -> None:
    """Test that partial variables are excluded from input_variables."""
    template = PromptTemplate(
        template="Hello {name} {greeting}",
        partial_variables={"greeting": "Hi"},
    )
    assert template.input_variables == ["name"]


def test_validate_template_true_fstring() -> None:
    """Test validate_template=True checks f-string template."""
    # Valid case
    template = PromptTemplate(
        template="Hello {name}",
        input_variables=["name"],
        validate_template=True,
    )
    assert template.input_variables == ["name"]


def test_validate_template_true_missing_vars() -> None:
    """Test validate_template=True raises on missing variables."""
    with pytest.raises(ValueError, match="Invalid prompt schema"):
        PromptTemplate(
            template="Hello {name}",
            input_variables=[],
            validate_template=True,
        )


def test_validate_template_mustache_raises() -> None:
    """Test validate_template=True raises for mustache format."""
    with pytest.raises(ValueError, match="Mustache templates cannot be validated"):
        PromptTemplate(
            template="Hello {{name}}",
            input_variables=["name"],
            template_format="mustache",
            validate_template=True,
        )


def test_validate_template_true_without_input_variables() -> None:
    """Test validate_template=True requires input_variables."""
    with pytest.raises(
        ValueError, match="Input variables must be provided to validate"
    ):
        PromptTemplate(
            template="Hello {name}",
            template_format="f-string",
            validate_template=True,
        )


def test_no_template_returns_early() -> None:
    """Test that pre_init_validation returns early if template is None."""
    # This tests the guard clause; template=None should let pydantic fail
    with pytest.raises(Exception):
        PromptTemplate(template=None)  # type: ignore[arg-type]


# --- format ---


def test_format_basic_fstring() -> None:
    """Test basic f-string formatting."""
    template = PromptTemplate.from_template("Hello {name}")
    result = template.format(name="World")
    assert result == "Hello World"


def test_format_basic_mustache() -> None:
    """Test basic mustache formatting."""
    template = PromptTemplate.from_template(
        "Hello {{name}}", template_format="mustache"
    )
    result = template.format(name="World")
    assert result == "Hello World"


def test_format_with_partial_variables() -> None:
    """Test formatting with partial variables."""
    template = PromptTemplate(
        template="Hello {greeting} {name}",
        input_variables=["name"],
        partial_variables={"greeting": "dear"},
    )
    result = template.format(name="World")
    assert result == "Hello dear World"


def test_format_with_partial_callable() -> None:
    """Test formatting with callable partial variables."""
    template = PromptTemplate(
        template="Count: {count}",
        input_variables=[],
        partial_variables={"count": lambda: "42"},
    )
    result = template.format()
    assert result == "Count: 42"


# --- __add__ ---


def test_add_two_prompt_templates() -> None:
    """Test adding two PromptTemplates."""
    t1 = PromptTemplate.from_template("Hello {name}")
    t2 = PromptTemplate.from_template(" and {greeting}")
    combined = t1 + t2
    assert isinstance(combined, PromptTemplate)
    assert sorted(combined.input_variables) == ["greeting", "name"]
    assert combined.format(name="World", greeting="Hi") == "Hello World and Hi"


def test_add_prompt_with_string() -> None:
    """Test adding a PromptTemplate with a string."""
    t1 = PromptTemplate.from_template("Hello {name}")
    combined = t1 + " and {greeting}"
    assert isinstance(combined, PromptTemplate)
    result = combined.format(name="World", greeting="Hi")
    assert result == "Hello World and Hi"


def test_add_different_formats_raises() -> None:
    """Test adding templates with different formats raises error."""
    t1 = PromptTemplate.from_template("Hello {name}")
    t2 = PromptTemplate.from_template("Hello {{name}}", template_format="mustache")
    with pytest.raises(ValueError, match="Cannot add templates of different formats"):
        t1 + t2


def test_add_with_unsupported_type_raises() -> None:
    """Test adding a PromptTemplate with unsupported type raises error."""
    t1 = PromptTemplate.from_template("Hello {name}")
    with pytest.raises(NotImplementedError, match="Unsupported operand type"):
        t1 + 42  # type: ignore[operator]


def test_add_with_conflicting_partial_variables_raises() -> None:
    """Test adding templates with conflicting partial variables raises."""
    t1 = PromptTemplate(
        template="Hello {name} {a}",
        input_variables=["name"],
        partial_variables={"a": "1"},
    )
    t2 = PromptTemplate(
        template=" {greeting} {a}",
        input_variables=["greeting"],
        partial_variables={"a": "2"},
    )
    with pytest.raises(ValueError, match="Cannot have same variable partialed twice"):
        t1 + t2


def test_add_combines_partial_variables() -> None:
    """Test adding templates merges partial variables."""
    t1 = PromptTemplate(
        template="{a} ",
        input_variables=[],
        partial_variables={"a": "1"},
    )
    t2 = PromptTemplate(
        template="{b}",
        input_variables=[],
        partial_variables={"b": "2"},
    )
    combined = t1 + t2
    assert combined.format() == "1 2"


def test_add_validate_template_propagation() -> None:
    """Test that validate_template is ANDed when adding templates."""
    t1 = PromptTemplate(
        template="Hello {name}",
        input_variables=["name"],
        validate_template=True,
    )
    t2 = PromptTemplate.from_template(" {greeting}")
    combined = t1 + t2
    assert combined.validate_template is False


# --- from_examples ---


def test_from_examples_basic() -> None:
    """Test creating prompt from examples."""
    examples = ["Example 1: A", "Example 2: B"]
    suffix = "Question: {question}"
    prompt = PromptTemplate.from_examples(
        examples=examples,
        suffix=suffix,
        input_variables=["question"],
        prefix="Instructions:",
    )
    result = prompt.format(question="What?")
    assert "Instructions:" in result
    assert "Example 1: A" in result
    assert "Example 2: B" in result
    assert "Question: What?" in result


def test_from_examples_custom_separator() -> None:
    """Test from_examples with custom separator."""
    examples = ["Ex1", "Ex2"]
    prompt = PromptTemplate.from_examples(
        examples=examples,
        suffix="End: {q}",
        input_variables=["q"],
        example_separator=" | ",
    )
    result = prompt.format(q="test")
    assert " | " in result


# --- from_file ---


def test_from_file_reads_template(tmp_path: Path) -> None:
    """Test from_file reads template from disk."""
    template_file = tmp_path / "template.txt"
    template_file.write_text("Hello {name}", encoding="utf-8")
    prompt = PromptTemplate.from_file(template_file)
    assert prompt.format(name="World") == "Hello World"


def test_from_file_with_encoding(tmp_path: Path) -> None:
    """Test from_file with specific encoding."""
    template_file = tmp_path / "template.txt"
    template_file.write_text("Hello {name} €", encoding="utf-8")
    prompt = PromptTemplate.from_file(template_file, encoding="utf-8")
    assert prompt.format(name="World") == "Hello World €"


def test_from_file_with_template_format(tmp_path: Path) -> None:
    """Test from_file respects template_format kwarg."""
    template_file = tmp_path / "template.txt"
    template_file.write_text("Hello {{name}}", encoding="utf-8")
    prompt = PromptTemplate.from_file(template_file, template_format="mustache")
    assert prompt.template_format == "mustache"
    assert prompt.format(name="World") == "Hello World"


# --- from_template ---


def test_from_template_with_partial_variables() -> None:
    """Test from_template with partial_variables kwarg."""
    prompt = PromptTemplate.from_template(
        "Hello {name} {greeting}",
        partial_variables={"greeting": "Hi"},
    )
    assert prompt.input_variables == ["name"]
    assert prompt.format(name="World") == "Hello World Hi"


def test_from_template_fstring_no_vars() -> None:
    """Test from_template with no variables."""
    prompt = PromptTemplate.from_template("Hello World")
    assert prompt.input_variables == []
    assert prompt.format() == "Hello World"


def test_from_template_mustache() -> None:
    """Test from_template with mustache format."""
    prompt = PromptTemplate.from_template("Hello {{name}}", template_format="mustache")
    assert prompt.input_variables == ["name"]
    assert prompt.template_format == "mustache"


# --- get_input_schema ---


def test_get_input_schema_fstring() -> None:
    """Test get_input_schema for f-string template."""
    prompt = PromptTemplate.from_template("Hello {name}")
    schema = prompt.get_input_schema()
    assert "name" in schema.model_fields


def test_get_input_schema_mustache() -> None:
    """Test get_input_schema for mustache template returns mustache_schema."""
    prompt = PromptTemplate.from_template("Hello {{name}}", template_format="mustache")
    schema = prompt.get_input_schema()
    assert "name" in schema.model_fields


def test_get_input_schema_mustache_nested() -> None:
    """Test get_input_schema for nested mustache template."""
    prompt = PromptTemplate.from_template(
        "{{person.name}} is {{person.age}}",
        template_format="mustache",
    )
    schema = prompt.get_input_schema()
    assert "person" in schema.model_fields


# --- Invalid template format ---


def test_invalid_template_format_raises() -> None:
    """Test that an invalid template_format raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported template format"):
        PromptTemplate(
            template="Hello {name}",
            input_variables=["name"],
            template_format="invalid",  # type: ignore[arg-type]
        )


# --- Serialization round-trip ---


def test_serialization_round_trip() -> None:
    """Test that PromptTemplate can be serialized and deserialized."""
    from langchain_core.load.dump import dumps
    from langchain_core.load.load import loads

    template = PromptTemplate.from_template("Hello {name}")
    serialized = dumps(template)
    deserialized = loads(serialized)
    assert deserialized.format(name="World") == "Hello World"
    assert deserialized.input_variables == ["name"]


def test_serialization_round_trip_with_metadata() -> None:
    """Test serialization preserves metadata and tags."""
    from langchain_core.load.dump import dumps
    from langchain_core.load.load import loads

    template = PromptTemplate(
        template="Hello {name}",
        input_variables=["name"],
        metadata={"version": "1"},
        tags=["test"],
    )
    serialized = dumps(template)
    deserialized = loads(serialized)
    assert deserialized.metadata == {"version": "1"}
    assert deserialized.tags == ["test"]
