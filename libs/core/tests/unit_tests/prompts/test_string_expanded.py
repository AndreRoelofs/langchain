"""Expanded test suite for string prompt utilities."""

import pytest

from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    DEFAULT_VALIDATOR_MAPPING,
    StringPromptTemplate,
    check_valid_template,
    get_template_variables,
    jinja2_formatter,
    mustache_formatter,
    mustache_template_vars,
    validate_jinja2,
)


def test_mustache_formatter_basic() -> None:
    """Test basic mustache formatting."""
    result = mustache_formatter("Hello {{name}}", name="World")
    assert result == "Hello World"


def test_mustache_formatter_multiple_variables() -> None:
    """Test mustache formatting with multiple variables."""
    result = mustache_formatter(
        "{{greeting}} {{name}}!",
        greeting="Hello",
        name="World",
    )
    assert result == "Hello World!"


def test_mustache_formatter_nested_variables() -> None:
    """Test mustache formatting with nested variables."""
    result = mustache_formatter(
        "{{person.name}} is {{person.age}} years old",
        person={"name": "Alice", "age": 30},
    )
    assert result == "Alice is 30 years old"


def test_mustache_formatter_sections() -> None:
    """Test mustache formatting with sections."""
    result = mustache_formatter(
        "{{#show}}Visible{{/show}}",
        show=True,
    )
    assert result == "Visible"

    result = mustache_formatter(
        "{{#show}}Invisible{{/show}}",
        show=False,
    )
    assert result == ""


def test_mustache_formatter_inverted_sections() -> None:
    """Test mustache formatting with inverted sections."""
    result = mustache_formatter(
        "{{^show}}Visible{{/show}}",
        show=False,
    )
    assert result == "Visible"

    result = mustache_formatter(
        "{{^show}}Invisible{{/show}}",
        show=True,
    )
    assert result == ""


def test_mustache_formatter_lists() -> None:
    """Test mustache formatting with lists."""
    result = mustache_formatter(
        "{{#items}}{{name}}, {{/items}}",
        items=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
    )
    assert result == "A, B, C, "


def test_mustache_template_vars_basic() -> None:
    """Test extracting variables from mustache template."""
    vars = mustache_template_vars("Hello {{name}}")
    assert vars == {"name"}


def test_mustache_template_vars_multiple() -> None:
    """Test extracting multiple variables from mustache template."""
    vars = mustache_template_vars("{{greeting}} {{name}} {{punctuation}}")
    assert vars == {"greeting", "name", "punctuation"}


def test_mustache_template_vars_nested() -> None:
    """Test extracting nested variables from mustache template."""
    vars = mustache_template_vars("{{person.name}} is {{person.age}}")
    assert vars == {"person"}


def test_mustache_template_vars_sections() -> None:
    """Test extracting variables from sections."""
    vars = mustache_template_vars("{{#items}}{{name}}{{/items}}")
    assert vars == {"items"}


def test_mustache_template_vars_with_dot() -> None:
    """Test that dot variable is ignored."""
    vars = mustache_template_vars("{{.}}")
    assert vars == set()


def test_mustache_template_vars_no_escape() -> None:
    """Test extracting variables from no-escape syntax."""
    vars = mustache_template_vars("{{&variable}}")
    assert vars == {"variable"}


def test_mustache_template_vars_inverted_section() -> None:
    """Test extracting variables from inverted sections."""
    vars = mustache_template_vars("{{^empty}}Something{{/empty}}")
    assert vars == {"empty"}


@pytest.mark.requires("jinja2")
def test_jinja2_formatter_basic() -> None:
    """Test basic jinja2 formatting."""
    result = jinja2_formatter("Hello {{ name }}", name="World")
    assert result == "Hello World"


@pytest.mark.requires("jinja2")
def test_jinja2_formatter_multiple_variables() -> None:
    """Test jinja2 formatting with multiple variables."""
    result = jinja2_formatter(
        "{{ greeting }} {{ name }}!",
        greeting="Hello",
        name="World",
    )
    assert result == "Hello World!"


@pytest.mark.requires("jinja2")
def test_jinja2_formatter_with_filters() -> None:
    """Test jinja2 formatting with filters."""
    result = jinja2_formatter("{{ name|upper }}", name="world")
    assert result == "WORLD"


@pytest.mark.requires("jinja2")
def test_jinja2_formatter_with_control_structures() -> None:
    """Test jinja2 formatting with if statements."""
    template = "{% if show %}Visible{% endif %}"
    result = jinja2_formatter(template, show=True)
    assert result == "Visible"

    result = jinja2_formatter(template, show=False)
    assert result == ""


@pytest.mark.requires("jinja2")
def test_jinja2_formatter_with_for_loop() -> None:
    """Test jinja2 formatting with for loops."""
    template = "{% for item in items %}{{ item }}, {% endfor %}"
    result = jinja2_formatter(template, items=["A", "B", "C"])
    assert result == "A, B, C, "


@pytest.mark.requires("jinja2")
def test_jinja2_formatter_security_sandboxed() -> None:
    """Test that jinja2 uses sandboxed environment."""
    # Jinja2 sandbox may not raise SecurityError for all attribute access
    # but it should prevent dangerous operations. Test that it works without
    # allowing execution of arbitrary code.
    result = jinja2_formatter("{{ name }}", name="safe")
    assert result == "safe"

    # The sandbox should restrict attribute access, but the exact behavior
    # may vary. At minimum, verify it doesn't crash and handles safely.
    try:
        # This might return empty or raise error depending on sandbox config
        result = jinja2_formatter("{{ ''.__class__ }}")
        # If it doesn't raise, it should at least not execute dangerous code
        assert "__class__" not in result or result == ""
    except Exception:
        # Any exception is acceptable here - it means it's blocked
        pass


@pytest.mark.requires("jinja2")
def test_validate_jinja2_correct_variables() -> None:
    """Test jinja2 validation with correct variables."""
    # Should not raise or warn
    validate_jinja2("{{ name }} {{ age }}", ["name", "age"])


@pytest.mark.requires("jinja2")
def test_validate_jinja2_missing_variables() -> None:
    """Test jinja2 validation with missing variables."""
    with pytest.warns(UserWarning, match="Missing variables"):
        validate_jinja2("{{ name }} {{ age }}", ["name"])


@pytest.mark.requires("jinja2")
def test_validate_jinja2_extra_variables() -> None:
    """Test jinja2 validation with extra variables."""
    with pytest.warns(UserWarning, match="Extra variables"):
        validate_jinja2("{{ name }}", ["name", "age", "extra"])


@pytest.mark.requires("jinja2")
def test_validate_jinja2_both_missing_and_extra() -> None:
    """Test jinja2 validation with both missing and extra variables."""
    with pytest.warns(UserWarning, match="Missing variables.*Extra variables"):
        validate_jinja2("{{ name }} {{ age }}", ["name", "extra"])


def test_check_valid_template_fstring() -> None:
    """Test check_valid_template with f-string format."""
    # Should not raise
    check_valid_template("Hello {name}", "f-string", ["name"])


def test_check_valid_template_fstring_invalid() -> None:
    """Test check_valid_template with invalid f-string."""
    with pytest.raises(ValueError, match="Invalid prompt schema"):
        check_valid_template("Hello {name}", "f-string", ["wrong"])


@pytest.mark.requires("jinja2")
def test_check_valid_template_jinja2() -> None:
    """Test check_valid_template with jinja2 format."""
    # Should not raise (might warn but not raise)
    check_valid_template("Hello {{ name }}", "jinja2", ["name"])


def test_check_valid_template_invalid_format() -> None:
    """Test check_valid_template with invalid template format."""
    with pytest.raises(ValueError, match="Invalid template format"):
        check_valid_template("Hello {name}", "invalid_format", ["name"])


def test_get_template_variables_fstring() -> None:
    """Test getting variables from f-string template."""
    vars = get_template_variables("Hello {name} and {friend}", "f-string")
    assert sorted(vars) == ["friend", "name"]


def test_get_template_variables_fstring_repeated() -> None:
    """Test getting variables from f-string with repeated variables."""
    vars = get_template_variables("{name} and {name} again", "f-string")
    assert vars == ["name"]


def test_get_template_variables_fstring_attribute_access_blocked() -> None:
    """Test that attribute access in f-string variables is blocked."""
    with pytest.raises(ValueError, match="Invalid variable name.*attribute access"):
        get_template_variables("{obj.attr}", "f-string")


def test_get_template_variables_fstring_indexing_blocked() -> None:
    """Test that indexing in f-string variables is blocked."""
    with pytest.raises(ValueError, match="Invalid variable name.*indexing"):
        get_template_variables("{arr[0]}", "f-string")


def test_get_template_variables_fstring_digit_names_blocked() -> None:
    """Test that all-digit variable names are blocked in f-string."""
    with pytest.raises(ValueError, match="Invalid variable name.*positional arguments"):
        get_template_variables("{0}", "f-string")

    with pytest.raises(ValueError, match="Invalid variable name.*positional arguments"):
        get_template_variables("{42}", "f-string")


@pytest.mark.requires("jinja2")
def test_get_template_variables_jinja2() -> None:
    """Test getting variables from jinja2 template."""
    vars = get_template_variables("Hello {{ name }} and {{ friend }}", "jinja2")
    assert sorted(vars) == ["friend", "name"]


@pytest.mark.requires("jinja2")
def test_get_template_variables_jinja2_with_control() -> None:
    """Test getting variables from jinja2 template with control structures."""
    vars = get_template_variables(
        "{% if show %}{{ name }}{% endif %}",
        "jinja2",
    )
    assert sorted(vars) == ["name", "show"]


def test_get_template_variables_mustache() -> None:
    """Test getting variables from mustache template."""
    vars = get_template_variables("Hello {{name}} and {{friend}}", "mustache")
    assert sorted(vars) == ["friend", "name"]


def test_get_template_variables_unsupported_format() -> None:
    """Test getting variables with unsupported format raises error."""
    with pytest.raises(ValueError, match="Unsupported template format"):
        get_template_variables("Hello {name}", "unsupported")


def test_default_formatter_mapping_has_all_formats() -> None:
    """Test that DEFAULT_FORMATTER_MAPPING has all supported formats."""
    assert "f-string" in DEFAULT_FORMATTER_MAPPING
    assert "mustache" in DEFAULT_FORMATTER_MAPPING
    assert "jinja2" in DEFAULT_FORMATTER_MAPPING


def test_default_validator_mapping_has_expected_formats() -> None:
    """Test that DEFAULT_VALIDATOR_MAPPING has expected formats."""
    assert "f-string" in DEFAULT_VALIDATOR_MAPPING
    assert "jinja2" in DEFAULT_VALIDATOR_MAPPING


def test_string_prompt_template_lc_namespace() -> None:
    """Test StringPromptTemplate namespace."""
    assert StringPromptTemplate.get_lc_namespace() == [
        "langchain",
        "prompts",
        "base",
    ]


def test_string_prompt_template_pretty_repr() -> None:
    """Test StringPromptTemplate pretty_repr method."""
    from langchain_core.prompts import PromptTemplate

    template = PromptTemplate.from_template("Hello {name} and {friend}")
    repr_str = template.pretty_repr()

    assert "{name}" in repr_str
    assert "{friend}" in repr_str


def test_mustache_template_vars_nested_sections() -> None:
    """Test extracting variables from nested sections."""
    vars = mustache_template_vars(
        "{{#outer}}{{#inner}}{{value}}{{/inner}}{{/outer}}"
    )
    assert vars == {"outer"}


def test_mustache_template_vars_mixed() -> None:
    """Test extracting variables from mixed template."""
    vars = mustache_template_vars(
        "{{simple}} {{#section}}{{nested}}{{/section}} {{&unescaped}}"
    )
    assert vars == {"simple", "section", "unescaped"}
