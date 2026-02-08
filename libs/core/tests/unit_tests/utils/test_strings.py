"""Test string utilities."""

from langchain_core.utils.strings import (
    comma_list,
    sanitize_for_postgres,
    stringify_dict,
    stringify_value,
)


def test_sanitize_for_postgres() -> None:
    """Test sanitizing text for PostgreSQL compatibility."""
    # Test with NUL bytes
    text_with_nul = "Hello\x00world\x00test"
    expected = "Helloworldtest"
    assert sanitize_for_postgres(text_with_nul) == expected

    # Test with replacement character
    expected_with_replacement = "Hello world test"
    assert sanitize_for_postgres(text_with_nul, " ") == expected_with_replacement

    # Test with text without NUL bytes
    clean_text = "Hello world"
    assert sanitize_for_postgres(clean_text) == clean_text

    # Test empty string
    assert not sanitize_for_postgres("")

    # Test with multiple consecutive NUL bytes
    text_with_multiple_nuls = "Hello\x00\x00\x00world"
    assert sanitize_for_postgres(text_with_multiple_nuls) == "Helloworld"
    assert sanitize_for_postgres(text_with_multiple_nuls, "-") == "Hello---world"


def test_existing_string_functions() -> None:
    """Test existing string functions still work."""
    # Test comma_list
    assert comma_list([1, 2, 3]) == "1, 2, 3"
    assert comma_list(["a", "b", "c"]) == "a, b, c"

    # Test stringify_value
    assert stringify_value("hello") == "hello"
    assert stringify_value(42) == "42"

    # Test stringify_dict
    data = {"key": "value", "number": 123}
    result = stringify_dict(data)
    assert "key: value" in result
    assert "number: 123" in result


def test_stringify_value_nested_structures() -> None:
    """Test stringifying nested structures."""
    # Test nested dict in list
    nested_data = {
        "users": [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
        ],
        "metadata": {"total_users": 2, "active": True},
    }

    result = stringify_value(nested_data)

    # Should contain all the nested values
    assert "users:" in result
    assert "name: Alice" in result
    assert "name: Bob" in result
    assert "metadata:" in result
    assert "total_users: 2" in result
    assert "active: True" in result

    # Test list of mixed types
    mixed_list = ["string", 42, {"key": "value"}, ["nested", "list"]]
    result = stringify_value(mixed_list)

    assert "string" in result
    assert "42" in result
    assert "key: value" in result
    assert "nested" in result
    assert "list" in result


# ---------------------------------------------------------------------------
# comma_list edge cases
# ---------------------------------------------------------------------------
def test_comma_list_empty() -> None:
    """comma_list returns empty string for empty list."""
    assert comma_list([]) == ""


def test_comma_list_single_item() -> None:
    """comma_list returns single item without comma."""
    assert comma_list(["only"]) == "only"


def test_comma_list_non_string_items() -> None:
    """comma_list stringifies non-string items."""
    assert comma_list([1, 2.5, True, None]) == "1, 2.5, True, None"


# ---------------------------------------------------------------------------
# stringify_value edge cases
# ---------------------------------------------------------------------------
def test_stringify_value_with_string() -> None:
    """stringify_value returns strings as-is."""
    assert stringify_value("hello") == "hello"
    assert stringify_value("") == ""


def test_stringify_value_with_int() -> None:
    """stringify_value converts int to string."""
    assert stringify_value(42) == "42"
    assert stringify_value(0) == "0"
    assert stringify_value(-1) == "-1"


def test_stringify_value_with_float() -> None:
    """stringify_value converts float to string."""
    assert stringify_value(3.14) == "3.14"


def test_stringify_value_with_bool() -> None:
    """stringify_value converts bool to string."""
    assert stringify_value(True) == "True"
    assert stringify_value(False) == "False"


def test_stringify_value_with_none() -> None:
    """stringify_value converts None to string."""
    assert stringify_value(None) == "None"


def test_stringify_value_with_empty_dict() -> None:
    """stringify_value returns newline + empty for empty dict."""
    result = stringify_value({})
    assert result == "\n"


def test_stringify_value_with_empty_list() -> None:
    """stringify_value returns empty string for empty list."""
    assert stringify_value([]) == ""


def test_stringify_value_with_list_of_strings() -> None:
    """stringify_value joins list items with newlines."""
    result = stringify_value(["a", "b", "c"])
    assert result == "a\nb\nc"


def test_stringify_value_with_nested_dict() -> None:
    """stringify_value recursively stringifies nested dicts."""
    result = stringify_value({"outer": {"inner": "value"}})
    assert "outer:" in result
    assert "inner: value" in result


# ---------------------------------------------------------------------------
# stringify_dict edge cases
# ---------------------------------------------------------------------------
def test_stringify_dict_empty() -> None:
    """stringify_dict returns empty string for empty dict."""
    assert stringify_dict({}) == ""


def test_stringify_dict_single_key() -> None:
    """stringify_dict formats a single key-value pair."""
    assert stringify_dict({"key": "value"}) == "key: value\n"


def test_stringify_dict_with_int_values() -> None:
    """stringify_dict converts int values to strings."""
    result = stringify_dict({"count": 42})
    assert "count: 42" in result


def test_stringify_dict_with_nested_dict_value() -> None:
    """stringify_dict handles nested dict values."""
    result = stringify_dict({"parent": {"child": "val"}})
    assert "parent:" in result
    assert "child: val" in result


def test_stringify_dict_with_list_value() -> None:
    """stringify_dict handles list values."""
    result = stringify_dict({"items": ["a", "b"]})
    assert "items:" in result
    assert "a" in result
    assert "b" in result


# ---------------------------------------------------------------------------
# sanitize_for_postgres edge cases
# ---------------------------------------------------------------------------
def test_sanitize_for_postgres_only_nul_bytes() -> None:
    """sanitize_for_postgres handles string of only NUL bytes."""
    assert sanitize_for_postgres("\x00\x00\x00") == ""


def test_sanitize_for_postgres_custom_replacement() -> None:
    """sanitize_for_postgres replaces NUL bytes with custom string."""
    assert sanitize_for_postgres("a\x00b", "X") == "aXb"
    assert sanitize_for_postgres("a\x00b", "??") == "a??b"


def test_sanitize_for_postgres_unicode() -> None:
    """sanitize_for_postgres preserves unicode characters."""
    text = "héllo wörld \x00 café"
    assert sanitize_for_postgres(text) == "héllo wörld  café"


def test_sanitize_for_postgres_no_nul() -> None:
    """sanitize_for_postgres returns identical string when no NUL bytes."""
    text = "clean text"
    result = sanitize_for_postgres(text)
    assert result == text
