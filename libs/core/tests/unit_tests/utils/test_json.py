"""Tests for JSON parsing utilities."""

import json

import pytest

from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import (
    parse_and_check_json_markdown,
    parse_json_markdown,
    parse_partial_json,
)


def test_parse_partial_json_valid() -> None:
    """Test parsing valid JSON."""
    result = parse_partial_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_parse_partial_json_missing_closing_brace() -> None:
    """Test parsing JSON with missing closing brace."""
    result = parse_partial_json('{"key": "value"')
    assert result == {"key": "value"}


def test_parse_partial_json_missing_closing_bracket() -> None:
    """Test parsing JSON array with missing closing bracket."""
    result = parse_partial_json('[1, 2, 3')
    assert result == [1, 2, 3]


def test_parse_partial_json_nested_missing_braces() -> None:
    """Test parsing nested JSON with missing braces."""
    result = parse_partial_json('{"outer": {"inner": "value"')
    assert result == {"outer": {"inner": "value"}}


def test_parse_partial_json_with_newlines_in_string() -> None:
    """Test parsing JSON with unescaped newlines in strings."""
    # The function should handle this by escaping the newline
    result = parse_partial_json('{"key": "value\nwith newline"}')
    # The newline is preserved in the parsed value (not double-escaped)
    assert result == {"key": "value\nwith newline"}


def test_parse_partial_json_unterminated_string() -> None:
    """Test parsing JSON with unterminated string."""
    result = parse_partial_json('{"key": "unterminated')
    assert result == {"key": "unterminated"}


def test_parse_partial_json_empty_string() -> None:
    """Test parsing empty JSON."""
    result = parse_partial_json('{}')
    assert result == {}


def test_parse_partial_json_empty_array() -> None:
    """Test parsing empty array."""
    result = parse_partial_json('[]')
    assert result == []


def test_parse_json_markdown_plain_json() -> None:
    """Test parsing plain JSON without markdown."""
    result = parse_json_markdown('{"key": "value"}')
    assert result == {"key": "value"}


def test_parse_json_markdown_with_code_block() -> None:
    """Test parsing JSON from markdown code block."""
    markdown = """```json
{
    "key": "value"
}
```"""
    result = parse_json_markdown(markdown)
    assert result == {"key": "value"}


def test_parse_json_markdown_with_plain_code_block() -> None:
    """Test parsing JSON from plain markdown code block (no language)."""
    markdown = """```
{
    "key": "value"
}
```"""
    result = parse_json_markdown(markdown)
    assert result == {"key": "value"}


def test_parse_json_markdown_with_surrounding_text() -> None:
    """Test parsing JSON from markdown with surrounding text."""
    markdown = """Here is some JSON:
```json
{"key": "value"}
```
And here is more text."""
    result = parse_json_markdown(markdown)
    assert result == {"key": "value"}


def test_parse_json_markdown_with_whitespace() -> None:
    """Test parsing JSON with extra whitespace."""
    result = parse_json_markdown('  \n\n  {"key": "value"}  \n\n  ')
    assert result == {"key": "value"}


def test_parse_json_markdown_with_backticks() -> None:
    """Test parsing JSON surrounded by backticks."""
    result = parse_json_markdown('```{"key": "value"}```')
    assert result == {"key": "value"}


def test_parse_and_check_json_markdown_valid() -> None:
    """Test parsing and checking valid JSON."""
    result = parse_and_check_json_markdown(
        '{"key": "value"}',
        expected_keys=["key"]
    )
    assert result == {"key": "value"}


def test_parse_and_check_json_markdown_missing_key() -> None:
    """Test parsing JSON missing expected key."""
    with pytest.raises(OutputParserException, match="Expected key `missing`"):
        parse_and_check_json_markdown(
            '{"key": "value"}',
            expected_keys=["missing"]
        )


def test_parse_and_check_json_markdown_invalid_json() -> None:
    """Test parsing invalid JSON."""
    with pytest.raises(OutputParserException, match="Got invalid JSON object"):
        parse_and_check_json_markdown(
            "not json",
            expected_keys=["key"]
        )


def test_parse_and_check_json_markdown_not_dict() -> None:
    """Test parsing JSON that's not a dictionary."""
    with pytest.raises(OutputParserException, match="Expected JSON object"):
        parse_and_check_json_markdown(
            '["array", "not", "dict"]',
            expected_keys=["key"]
        )


def test_parse_and_check_json_markdown_multiple_keys() -> None:
    """Test parsing and checking JSON with multiple expected keys."""
    result = parse_and_check_json_markdown(
        '{"key1": "value1", "key2": "value2"}',
        expected_keys=["key1", "key2"]
    )
    assert result == {"key1": "value1", "key2": "value2"}


def test_parse_json_markdown_with_action_input() -> None:
    """Test custom parser for action_input with unescaped newlines."""
    json_str = '{"action": "test", "action_input": "line1\nline2"}'
    result = parse_json_markdown(json_str)
    # The _custom_parser should escape newlines in action_input
    assert result["action"] == "test"
    assert "line1" in result["action_input"]


def test_parse_partial_json_with_booleans() -> None:
    """Test parsing JSON with boolean values."""
    result = parse_partial_json('{"true_val": true, "false_val": false}')
    assert result == {"true_val": True, "false_val": False}


def test_parse_partial_json_with_null() -> None:
    """Test parsing JSON with null value."""
    result = parse_partial_json('{"null_val": null}')
    assert result == {"null_val": None}


def test_parse_partial_json_with_numbers() -> None:
    """Test parsing JSON with various number types."""
    result = parse_partial_json('{"int": 42, "float": 3.14, "neg": -5}')
    assert result == {"int": 42, "float": 3.14, "neg": -5}


def test_parse_partial_json_complex_nested() -> None:
    """Test parsing complex nested JSON."""
    json_str = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob"}]}'
    result = parse_partial_json(json_str)
    assert result == {
        "users": [
            {"name": "Alice", "age": 30},
            {"name": "Bob"}
        ]
    }


def test_parse_partial_json_with_escaped_quotes() -> None:
    """Test parsing JSON with escaped quotes."""
    result = parse_partial_json('{"key": "value with \\"quotes\\""}')
    assert result == {"key": 'value with "quotes"'}
