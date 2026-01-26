"""Tests for formatting utilities."""

import pytest

from langchain_core.utils.formatting import StrictFormatter, formatter


def test_strict_formatter_basic() -> None:
    """Test basic string formatting with StrictFormatter."""
    fmt = StrictFormatter()
    result = fmt.format("Hello {name}", name="World")
    assert result == "Hello World"


def test_strict_formatter_multiple_variables() -> None:
    """Test formatting with multiple variables."""
    fmt = StrictFormatter()
    result = fmt.format("{greeting} {name}!", greeting="Hello", name="World")
    assert result == "Hello World!"


def test_strict_formatter_rejects_positional_args() -> None:
    """Test that StrictFormatter rejects positional arguments."""
    fmt = StrictFormatter()
    with pytest.raises(
        ValueError,
        match="No arguments should be provided, everything should be passed as keyword arguments",
    ):
        fmt.format("Hello {}", "World")


def test_strict_formatter_validate_input_variables() -> None:
    """Test validate_input_variables method."""
    fmt = StrictFormatter()
    # Should not raise for valid variables
    fmt.validate_input_variables("Hello {name}", ["name"])

    # Should raise for missing variables
    with pytest.raises(KeyError):
        fmt.validate_input_variables("Hello {name}", ["other"])


def test_strict_formatter_validate_input_variables_multiple() -> None:
    """Test validation with multiple input variables."""
    fmt = StrictFormatter()
    fmt.validate_input_variables(
        "{greeting} {name}!",
        ["greeting", "name"]
    )


def test_formatter_is_strict_formatter() -> None:
    """Test that the exported formatter is a StrictFormatter instance."""
    assert isinstance(formatter, StrictFormatter)


def test_strict_formatter_with_nested_braces() -> None:
    """Test formatting with nested braces."""
    fmt = StrictFormatter()
    result = fmt.format("{{literal}} {var}", var="value")
    assert result == "{literal} value"


def test_strict_formatter_empty_string() -> None:
    """Test formatting an empty string."""
    fmt = StrictFormatter()
    result = fmt.format("", name="World")
    assert result == ""


def test_strict_formatter_no_placeholders() -> None:
    """Test formatting string with no placeholders."""
    fmt = StrictFormatter()
    result = fmt.format("Hello World", name="unused")
    assert result == "Hello World"
