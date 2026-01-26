"""Expanded test suite for dict prompt template."""

import pytest

from langchain_core.load import dumpd, load
from langchain_core.prompts.dict import DictPromptTemplate


def test_dict_prompt_template_basic_fstring() -> None:
    """Test basic dict prompt template with f-string format."""
    template = {"message": "Hello {name}"}
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    result = prompt.format(name="World")
    assert result == {"message": "Hello World"}


def test_dict_prompt_template_basic_mustache() -> None:
    """Test basic dict prompt template with mustache format."""
    template = {"message": "Hello {{name}}"}
    prompt = DictPromptTemplate(template=template, template_format="mustache")
    result = prompt.format(name="World")
    assert result == {"message": "Hello World"}


def test_dict_prompt_template_nested_dict() -> None:
    """Test dict prompt template with nested dictionaries."""
    template = {
        "outer": {
            "inner": {
                "message": "Hello {name}",
            }
        }
    }
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    result = prompt.format(name="World")
    assert result == {"outer": {"inner": {"message": "Hello World"}}}


def test_dict_prompt_template_with_list() -> None:
    """Test dict prompt template with list values."""
    template = {
        "items": [
            "First: {first}",
            "Second: {second}",
        ]
    }
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    result = prompt.format(first="A", second="B")
    assert result == {"items": ["First: A", "Second: B"]}


def test_dict_prompt_template_with_nested_list_dicts() -> None:
    """Test dict prompt template with nested lists containing dicts."""
    template = {
        "items": [
            {"name": "{name1}"},
            {"name": "{name2}"},
        ]
    }
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    result = prompt.format(name1="Alice", name2="Bob")
    assert result == {"items": [{"name": "Alice"}, {"name": "Bob"}]}


def test_dict_prompt_template_with_tuple() -> None:
    """Test dict prompt template with tuple values."""
    template = {
        "items": (
            "First: {first}",
            "Second: {second}",
        )
    }
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    result = prompt.format(first="A", second="B")
    # Result should maintain tuple type
    assert result == {"items": ("First: A", "Second: B")}


def test_dict_prompt_template_with_non_string_values() -> None:
    """Test dict prompt template with non-string values."""
    template = {
        "number": 42,
        "boolean": True,
        "null": None,
        "message": "Hello {name}",
    }
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    result = prompt.format(name="World")
    assert result == {
        "number": 42,
        "boolean": True,
        "null": None,
        "message": "Hello World",
    }


def test_dict_prompt_template_input_variables_single() -> None:
    """Test extracting input variables from simple template."""
    template = {"message": "Hello {name}"}
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    assert prompt.input_variables == ["name"]


def test_dict_prompt_template_input_variables_multiple() -> None:
    """Test extracting multiple input variables."""
    template = {
        "greeting": "Hello {name}",
        "farewell": "Goodbye {name}, see you in {place}",
    }
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    assert set(prompt.input_variables) == {"name", "place"}


def test_dict_prompt_template_input_variables_nested() -> None:
    """Test extracting input variables from nested dicts."""
    template = {
        "outer": {
            "inner1": "Value {var1}",
            "inner2": "Value {var2}",
        }
    }
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    assert set(prompt.input_variables) == {"var1", "var2"}


def test_dict_prompt_template_input_variables_in_lists() -> None:
    """Test extracting input variables from lists."""
    template = {
        "items": [
            "First {a}",
            "Second {b}",
        ]
    }
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    assert set(prompt.input_variables) == {"a", "b"}


async def test_dict_prompt_template_aformat() -> None:
    """Test async formatting of dict prompt template."""
    template = {"message": "Hello {name}"}
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    result = await prompt.aformat(name="Async")
    assert result == {"message": "Hello Async"}


def test_dict_prompt_template_invoke() -> None:
    """Test invoking dict prompt template."""
    template = {"message": "Hello {name}"}
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    result = prompt.invoke({"name": "World"})
    assert result == {"message": "Hello World"}


def test_dict_prompt_template_mustache_nested_variables() -> None:
    """Test mustache template with nested variable access."""
    template = {"message": "Hello {{person.name}}"}
    prompt = DictPromptTemplate(template=template, template_format="mustache")
    result = prompt.format(person={"name": "Alice"})
    assert result == {"message": "Hello Alice"}


def test_dict_prompt_template_mustache_sections() -> None:
    """Test mustache template with sections."""
    template = {
        "message": "{{#show}}Visible: {{value}}{{/show}}",
        "value": "{{value}}",
    }
    prompt = DictPromptTemplate(template=template, template_format="mustache")
    result = prompt.format(show=True, value="test")
    assert result == {
        "message": "Visible: test",
        "value": "test",
    }


def test_dict_prompt_template_serialization() -> None:
    """Test serialization and deserialization of dict prompt template."""
    template = {"message": "Hello {name}"}
    prompt = DictPromptTemplate(template=template, template_format="f-string")

    serialized = dumpd(prompt)
    deserialized = load(serialized)

    assert isinstance(deserialized, DictPromptTemplate)
    assert deserialized.template == template
    assert deserialized.template_format == "f-string"


def test_dict_prompt_template_is_lc_serializable() -> None:
    """Test that DictPromptTemplate is serializable."""
    assert DictPromptTemplate.is_lc_serializable()


def test_dict_prompt_template_lc_namespace() -> None:
    """Test DictPromptTemplate namespace."""
    assert DictPromptTemplate.get_lc_namespace() == [
        "langchain_core",
        "prompts",
        "dict",
    ]


def test_dict_prompt_template_prompt_type() -> None:
    """Test _prompt_type property."""
    template = {"message": "Hello {name}"}
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    assert prompt._prompt_type == "dict-prompt"


def test_dict_prompt_template_pretty_repr_not_implemented() -> None:
    """Test that pretty_repr raises NotImplementedError."""
    template = {"message": "Hello {name}"}
    prompt = DictPromptTemplate(template=template, template_format="f-string")

    with pytest.raises(NotImplementedError):
        prompt.pretty_repr()


def test_dict_prompt_template_empty_template() -> None:
    """Test dict prompt template with empty template."""
    template = {}
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    result = prompt.format()
    assert result == {}


def test_dict_prompt_template_no_variables() -> None:
    """Test dict prompt template with no variables."""
    template = {"message": "Static message"}
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    assert prompt.input_variables == []
    result = prompt.format()
    assert result == {"message": "Static message"}


def test_dict_prompt_template_multiple_occurrences() -> None:
    """Test dict prompt template with repeated variable."""
    template = {
        "message1": "Hello {name}",
        "message2": "Goodbye {name}",
    }
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    result = prompt.format(name="World")
    assert result == {
        "message1": "Hello World",
        "message2": "Goodbye World",
    }


def test_dict_prompt_template_complex_nested() -> None:
    """Test complex nested structure."""
    template = {
        "level1": {
            "level2": {
                "items": [
                    {"id": "{id1}", "name": "{name1}"},
                    {"id": "{id2}", "name": "{name2}"},
                ],
                "message": "Total: {count}",
            }
        }
    }
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    result = prompt.format(id1="1", name1="A", id2="2", name2="B", count="2")
    assert result == {
        "level1": {
            "level2": {
                "items": [
                    {"id": "1", "name": "A"},
                    {"id": "2", "name": "B"},
                ],
                "message": "Total: 2",
            }
        }
    }


def test_dict_prompt_template_with_special_characters() -> None:
    """Test dict prompt template with special characters in values."""
    template = {
        "message": "Hello {name}!",
        "question": "How are you {name}?",
    }
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    result = prompt.format(name="World")
    assert result == {
        "message": "Hello World!",
        "question": "How are you World?",
    }
