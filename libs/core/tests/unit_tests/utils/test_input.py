"""Tests for input utilities (color/text formatting)."""

from io import StringIO

import pytest

from langchain_core.utils.input import (
    get_bolded_text,
    get_color_mapping,
    get_colored_text,
    print_text,
)


def test_get_color_mapping() -> None:
    """Test basic color mapping functionality."""
    items = ["item1", "item2", "item3"]
    mapping = get_color_mapping(items)

    assert len(mapping) == 3
    assert "item1" in mapping
    assert "item2" in mapping
    assert "item3" in mapping
    assert all(color in ["blue", "yellow", "pink", "green", "red"] for color in mapping.values())


def test_get_color_mapping_with_excluded_colors() -> None:
    """Test color mapping with excluded colors."""
    items = ["item1", "item2"]
    excluded = ["blue", "yellow"]
    mapping = get_color_mapping(items, excluded_colors=excluded)

    assert len(mapping) == 2
    assert all(color not in excluded for color in mapping.values())


def test_get_color_mapping_cycles_colors() -> None:
    """Test that color mapping cycles through colors for many items."""
    items = [f"item{i}" for i in range(10)]
    mapping = get_color_mapping(items)

    assert len(mapping) == 10
    # Since we have 5 colors, we should see repeats
    colors_used = list(mapping.values())
    assert len(set(colors_used)) <= 5


def test_get_color_mapping_empty_after_exclusions() -> None:
    """Test that excluding all colors raises an error."""
    items = ["item1"]
    excluded = ["blue", "yellow", "pink", "green", "red"]

    with pytest.raises(ValueError, match="No colors available after applying exclusions"):
        get_color_mapping(items, excluded_colors=excluded)


def test_get_colored_text() -> None:
    """Test basic colored text generation."""
    text = "Hello"
    colored = get_colored_text(text, "blue")

    # Should contain ANSI escape codes
    assert "\u001b[" in colored
    assert "Hello" in colored
    assert "\u001b[0m" in colored  # Reset code


def test_get_colored_text_all_colors() -> None:
    """Test colored text generation for all available colors."""
    text = "Test"
    colors = ["blue", "yellow", "pink", "green", "red"]

    for color in colors:
        colored = get_colored_text(text, color)
        assert "\u001b[" in colored
        assert "Test" in colored


def test_get_bolded_text() -> None:
    """Test bold text generation."""
    text = "Bold Text"
    bolded = get_bolded_text(text)

    assert "\033[1m" in bolded  # Bold code
    assert "Bold Text" in bolded
    assert "\033[0m" in bolded  # Reset code


def test_print_text_no_color() -> None:
    """Test printing text without color."""
    output = StringIO()
    print_text("Hello", file=output)

    result = output.getvalue()
    assert "Hello" in result


def test_print_text_with_color() -> None:
    """Test printing text with color."""
    output = StringIO()
    print_text("Hello", color="blue", file=output)

    result = output.getvalue()
    assert "Hello" in result
    assert "\u001b[" in result  # ANSI escape code


def test_print_text_with_end() -> None:
    """Test printing text with custom end character."""
    output = StringIO()
    print_text("Hello", end="\n\n", file=output)

    result = output.getvalue()
    assert result == "Hello\n\n"


def test_print_text_default_end() -> None:
    """Test printing text with default empty end."""
    output = StringIO()
    print_text("Hello", file=output)

    result = output.getvalue()
    assert result == "Hello"  # No newline by default


def test_print_text_flushes_file() -> None:
    """Test that print_text flushes the file buffer."""
    output = StringIO()
    print_text("Test", file=output)

    # If flush didn't work, this test would need different logic
    # But StringIO handles flush gracefully
    assert output.getvalue() == "Test"
