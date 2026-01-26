"""Test format instructions module."""

from langchain_core.output_parsers.format_instructions import JSON_FORMAT_INSTRUCTIONS


def test_json_format_instructions_exists() -> None:
    """Test that JSON_FORMAT_INSTRUCTIONS constant exists."""
    assert JSON_FORMAT_INSTRUCTIONS is not None
    assert isinstance(JSON_FORMAT_INSTRUCTIONS, str)


def test_json_format_instructions_content() -> None:
    """Test that JSON_FORMAT_INSTRUCTIONS contains expected content."""
    # Should contain key instructions
    assert "JSON" in JSON_FORMAT_INSTRUCTIONS
    assert "schema" in JSON_FORMAT_INSTRUCTIONS

    # Should mention not to include markdown
    assert "Do not wrap" in JSON_FORMAT_INSTRUCTIONS or "not include" in JSON_FORMAT_INSTRUCTIONS

    # Should have placeholder for schema
    assert "{schema}" in JSON_FORMAT_INSTRUCTIONS


def test_json_format_instructions_no_markdown_mention() -> None:
    """Test that instructions explicitly mention not using markdown."""
    instructions_lower = JSON_FORMAT_INSTRUCTIONS.lower()
    assert "markdown" in instructions_lower or "```" in JSON_FORMAT_INSTRUCTIONS


def test_json_format_instructions_example() -> None:
    """Test that instructions contain an example."""
    # Should contain an example of well-formatted output
    assert "example" in JSON_FORMAT_INSTRUCTIONS.lower()
    assert "well-formatted" in JSON_FORMAT_INSTRUCTIONS.lower()


def test_json_format_instructions_strict_output() -> None:
    """Test that instructions emphasize strict output format."""
    instructions_upper = JSON_FORMAT_INSTRUCTIONS.upper()
    assert "STRICT" in instructions_upper or "only" in JSON_FORMAT_INSTRUCTIONS.lower()


def test_json_format_instructions_no_additional_text() -> None:
    """Test that instructions mention not including additional text."""
    instructions_lower = JSON_FORMAT_INSTRUCTIONS.lower()
    assert "additional" in instructions_lower or "only" in instructions_lower


def test_json_format_instructions_schema_placeholder() -> None:
    """Test that the schema placeholder is properly formatted."""
    # The placeholder should be in curly braces for string formatting
    assert "{schema}" in JSON_FORMAT_INSTRUCTIONS
    # Should not have double braces (which would escape them)
    assert "{{schema}}" not in JSON_FORMAT_INSTRUCTIONS


def test_json_format_instructions_formatting() -> None:
    """Test that instructions can be formatted with a schema."""
    test_schema = '{"properties": {"foo": {"type": "string"}}, "required": ["foo"]}'
    formatted = JSON_FORMAT_INSTRUCTIONS.format(schema=test_schema)

    # Should contain the schema
    assert test_schema in formatted
    # Should not contain the placeholder anymore
    assert "{schema}" not in formatted


def test_json_format_instructions_multiline() -> None:
    """Test that instructions are multiline for readability."""
    # Should have multiple lines
    assert "\n" in JSON_FORMAT_INSTRUCTIONS
    # Should have at least a few lines
    assert len(JSON_FORMAT_INSTRUCTIONS.split("\n")) > 3


def test_json_format_instructions_mentions_code_fence() -> None:
    """Test that instructions mention not using code fences."""
    # Should explicitly mention not using ``` or code fences
    assert "```" in JSON_FORMAT_INSTRUCTIONS or "code fence" in JSON_FORMAT_INSTRUCTIONS.lower()


def test_json_format_instructions_mentions_no_prepend() -> None:
    """Test that instructions mention not prepending text."""
    instructions_lower = JSON_FORMAT_INSTRUCTIONS.lower()
    assert "prepend" in instructions_lower or "append" in instructions_lower or "additional text" in instructions_lower


def test_json_format_instructions_example_format() -> None:
    """Test that the example in instructions shows proper format."""
    # Should show an example with properties and required fields
    assert "properties" in JSON_FORMAT_INSTRUCTIONS
    assert "required" in JSON_FORMAT_INSTRUCTIONS


def test_json_format_instructions_mentions_single_value() -> None:
    """Test that instructions mention returning a single JSON value."""
    instructions_lower = JSON_FORMAT_INSTRUCTIONS.lower()
    assert "single" in instructions_lower or "one" in instructions_lower or "only" in instructions_lower


def test_json_format_instructions_no_trailing_commas() -> None:
    """Test that instructions mention no trailing commas."""
    instructions_lower = JSON_FORMAT_INSTRUCTIONS.lower()
    assert "trailing" in instructions_lower or "comma" in instructions_lower or "comment" in instructions_lower


def test_json_format_instructions_conforms_to_schema() -> None:
    """Test that instructions mention conforming to schema."""
    instructions_lower = JSON_FORMAT_INSTRUCTIONS.lower()
    assert "conform" in instructions_lower or "match" in instructions_lower or "schema" in instructions_lower


def test_json_format_instructions_with_complex_schema() -> None:
    """Test formatting with a complex schema."""
    complex_schema = """{
        "properties": {
            "name": {"type": "string", "description": "Person's name"},
            "age": {"type": "integer", "minimum": 0},
            "emails": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name", "age"]
    }"""

    formatted = JSON_FORMAT_INSTRUCTIONS.format(schema=complex_schema)

    # Should contain all parts of the complex schema
    assert "name" in formatted
    assert "age" in formatted
    assert "emails" in formatted
    assert "array" in formatted


def test_json_format_instructions_with_unicode_schema() -> None:
    """Test formatting with unicode characters in schema."""
    unicode_schema = '{"properties": {"名前": {"type": "string", "description": "用户名"}}}'

    formatted = JSON_FORMAT_INSTRUCTIONS.format(schema=unicode_schema)

    # Should preserve unicode characters
    assert "名前" in formatted
    assert "用户名" in formatted


def test_json_format_instructions_length() -> None:
    """Test that instructions are comprehensive but not too long."""
    # Should be detailed enough (at least 100 characters)
    assert len(JSON_FORMAT_INSTRUCTIONS) > 100
    # But not excessively long (less than 5000 characters)
    assert len(JSON_FORMAT_INSTRUCTIONS) < 5000


def test_json_format_instructions_no_html() -> None:
    """Test that instructions don't contain HTML tags."""
    # Should not have HTML tags
    assert "<html>" not in JSON_FORMAT_INSTRUCTIONS.lower()
    assert "<div>" not in JSON_FORMAT_INSTRUCTIONS.lower()
    assert "<p>" not in JSON_FORMAT_INSTRUCTIONS.lower()


def test_json_format_instructions_readable() -> None:
    """Test that instructions are human-readable."""
    # Should not be all caps (except for emphasis)
    all_caps_lines = [
        line for line in JSON_FORMAT_INSTRUCTIONS.split("\n")
        if line.strip() and line.strip().isupper() and len(line.strip()) > 20
    ]
    # At most one or two lines should be all caps (for emphasis)
    assert len(all_caps_lines) <= 2


def test_json_format_instructions_mentions_top_level() -> None:
    """Test that instructions mention top-level JSON value."""
    instructions_lower = JSON_FORMAT_INSTRUCTIONS.lower()
    assert "top-level" in instructions_lower or "top level" in instructions_lower or "single" in instructions_lower


def test_json_format_instructions_example_shows_bad_format() -> None:
    """Test that instructions show what NOT to do."""
    instructions_lower = JSON_FORMAT_INSTRUCTIONS.lower()
    # Should mention what is NOT well-formatted
    assert "not well-formatted" in instructions_lower or "not" in instructions_lower


def test_json_format_instructions_code_block_mention() -> None:
    """Test that instructions mention code blocks in the example."""
    # The instructions should show the schema in a code block for readability
    # but tell users not to include code blocks in their output
    assert "```" in JSON_FORMAT_INSTRUCTIONS
    # Should clarify this is for readability only
    assert "readability" in JSON_FORMAT_INSTRUCTIONS.lower() or "shown" in JSON_FORMAT_INSTRUCTIONS.lower()
