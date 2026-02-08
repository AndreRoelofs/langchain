"""Additional expanded tests for string prompt utilities."""

import pytest

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
    StringPromptTemplate,
    is_subsequence,
    mustache_schema,
    mustache_template_vars,
)

# --- is_subsequence ---


def test_is_subsequence_true() -> None:
    """Test is_subsequence returns True for valid subsequence."""
    assert is_subsequence(("a",), ("a", "b")) is True
    assert is_subsequence(("a", "b"), ("a", "b", "c")) is True


def test_is_subsequence_false() -> None:
    """Test is_subsequence returns False when not a subsequence."""
    assert is_subsequence(("a", "b"), ("c", "d")) is False
    assert is_subsequence(("b",), ("a", "c")) is False


def test_is_subsequence_equal() -> None:
    """Test is_subsequence with equal sequences."""
    assert is_subsequence(("a", "b"), ("a", "b")) is True


def test_is_subsequence_child_longer_than_parent() -> None:
    """Test is_subsequence returns False when child is longer."""
    assert is_subsequence(("a", "b", "c"), ("a", "b")) is False


def test_is_subsequence_empty_child() -> None:
    """Test is_subsequence with empty child returns False."""
    assert is_subsequence((), ("a", "b")) is False


def test_is_subsequence_empty_parent() -> None:
    """Test is_subsequence with empty parent returns False."""
    assert is_subsequence(("a",), ()) is False


def test_is_subsequence_both_empty() -> None:
    """Test is_subsequence with both empty returns False."""
    assert is_subsequence((), ()) is False


# --- mustache_schema ---


def test_mustache_schema_simple() -> None:
    """Test mustache_schema with simple variables."""
    schema = mustache_schema("Hello {{name}}")
    assert "name" in schema.model_fields


def test_mustache_schema_nested() -> None:
    """Test mustache_schema with nested variables."""
    schema = mustache_schema("{{person.name}} is {{person.age}}")
    assert "person" in schema.model_fields
    # person should be a nested model with name and age
    person_type = schema.model_fields["person"].annotation
    assert hasattr(person_type, "model_fields")


def test_mustache_schema_sections() -> None:
    """Test mustache_schema with sections."""
    schema = mustache_schema("{{#items}}{{name}}{{/items}}")
    assert "items" in schema.model_fields


def test_mustache_schema_empty_template() -> None:
    """Test mustache_schema with no variables."""
    schema = mustache_schema("Hello World")
    assert len(schema.model_fields) == 0


def test_mustache_schema_inverted_section() -> None:
    """Test mustache_schema with inverted section."""
    schema = mustache_schema("{{^empty}}content{{/empty}}")
    assert "empty" in schema.model_fields


def test_mustache_schema_deeply_nested() -> None:
    """Test mustache_schema with deeply nested structure."""
    template = "{{#a}}{{#b}}{{c}}{{/b}}{{/a}}"
    schema = mustache_schema(template)
    assert "a" in schema.model_fields


# --- mustache_template_vars edge cases ---


def test_mustache_template_vars_empty_template() -> None:
    """Test empty template returns empty set."""
    assert mustache_template_vars("") == set()


def test_mustache_template_vars_no_variables() -> None:
    """Test template with no variables returns empty set."""
    assert mustache_template_vars("Hello World") == set()


def test_mustache_template_vars_dot_only() -> None:
    """Test that {{.}} is excluded."""
    assert mustache_template_vars("{{.}}") == set()


def test_mustache_template_vars_nested_sections_only_top_level() -> None:
    """Test that only top-level variables are returned."""
    vars_ = mustache_template_vars("{{top}} {{#section}}{{nested}}{{/section}}")
    assert "top" in vars_
    assert "section" in vars_
    # nested should NOT be included (it's inside a section)
    assert "nested" not in vars_


def test_mustache_template_vars_triple_mustache() -> None:
    """Test triple mustache (no escape) syntax."""
    vars_ = mustache_template_vars("{{{raw}}}")
    assert "raw" in vars_


# --- StringPromptTemplate ---


def test_string_prompt_template_format_prompt() -> None:
    """Test StringPromptTemplate.format_prompt returns StringPromptValue."""
    from langchain_core.prompt_values import StringPromptValue

    template = PromptTemplate.from_template("Hello {name}")
    result = template.format_prompt(name="World")
    assert isinstance(result, StringPromptValue)
    assert result.to_string() == "Hello World"


async def test_string_prompt_template_aformat_prompt() -> None:
    """Test StringPromptTemplate.aformat_prompt returns StringPromptValue."""
    from langchain_core.prompt_values import StringPromptValue

    template = PromptTemplate.from_template("Hello {name}")
    result = await template.aformat_prompt(name="World")
    assert isinstance(result, StringPromptValue)
    assert result.to_string() == "Hello World"


def test_string_prompt_template_pretty_repr_no_html() -> None:
    """Test pretty_repr without HTML."""
    template = PromptTemplate.from_template("Hello {name} and {friend}")
    result = template.pretty_repr(html=False)
    assert "{name}" in result
    assert "{friend}" in result


def test_string_prompt_template_pretty_repr_html() -> None:
    """Test pretty_repr with HTML highlights variables."""
    template = PromptTemplate.from_template("Hello {name}")
    result = template.pretty_repr(html=True)
    assert "name" in result


def test_string_prompt_template_pretty_print(
    capsys: pytest.CaptureFixture,
) -> None:
    """Test pretty_print outputs to stdout."""
    template = PromptTemplate.from_template("Hello {name}")
    template.pretty_print()
    captured = capsys.readouterr()
    assert "name" in captured.out


# --- StringPromptTemplate as abstract ---


def test_string_prompt_template_requires_format() -> None:
    """Test that StringPromptTemplate requires format to be implemented."""
    with pytest.raises(TypeError):
        StringPromptTemplate(input_variables=[])  # type: ignore[abstract]


# --- get_lc_namespace ---


def test_string_prompt_template_lc_namespace_via_class() -> None:
    """Test StringPromptTemplate.get_lc_namespace() returns correct value."""
    assert StringPromptTemplate.get_lc_namespace() == [
        "langchain",
        "prompts",
        "base",
    ]
