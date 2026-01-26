"""Tests for Mustache template rendering."""

import pytest

from langchain_core.utils.mustache import ChevronError, render, tokenize


def test_render_basic_variable() -> None:
    """Test basic variable rendering."""
    template = "Hello {{name}}!"
    data = {"name": "World"}
    result = render(template, data)
    assert result == "Hello World!"


def test_render_multiple_variables() -> None:
    """Test rendering multiple variables."""
    template = "{{greeting}} {{name}}!"
    data = {"greeting": "Hello", "name": "World"}
    result = render(template, data)
    assert result == "Hello World!"


def test_render_section() -> None:
    """Test rendering a section."""
    template = "{{#items}}{{.}} {{/items}}"
    data = {"items": ["a", "b", "c"]}
    result = render(template, data)
    assert result == "a b c "


def test_render_section_with_dicts() -> None:
    """Test rendering section with dictionaries."""
    template = "{{#users}}{{name}} {{/users}}"
    data = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
    result = render(template, data)
    assert result == "Alice Bob "


def test_render_inverted_section() -> None:
    """Test rendering inverted section."""
    template = "{{^items}}No items{{/items}}"
    data = {"items": []}
    result = render(template, data)
    assert result == "No items"


def test_render_inverted_section_with_items() -> None:
    """Test inverted section when items exist."""
    template = "{{^items}}No items{{/items}}"
    data = {"items": ["something"]}
    result = render(template, data)
    assert result == ""


def test_render_comment() -> None:
    """Test that comments are ignored."""
    template = "Hello {{! This is a comment }}World"
    data = {}
    result = render(template, data)
    assert result == "Hello World"


def test_render_no_escape() -> None:
    """Test rendering without HTML escaping."""
    template = "{{& html}}"
    data = {"html": "<b>Bold</b>"}
    result = render(template, data)
    assert result == "<b>Bold</b>"


def test_render_triple_mustache() -> None:
    """Test triple mustache (no escape)."""
    template = "{{{html}}}"
    data = {"html": "<b>Bold</b>"}
    result = render(template, data)
    assert result == "<b>Bold</b>"


def test_render_html_escape() -> None:
    """Test HTML escaping with regular mustache."""
    template = "{{html}}"
    data = {"html": "<b>Bold</b>"}
    result = render(template, data)
    assert result == "&lt;b&gt;Bold&lt;/b&gt;"


def test_render_nested_context() -> None:
    """Test accessing nested context."""
    template = "{{user.name}}"
    data = {"user": {"name": "Alice"}}
    result = render(template, data)
    assert result == "Alice"


def test_render_deeply_nested() -> None:
    """Test deeply nested context access."""
    template = "{{a.b.c.d}}"
    data = {"a": {"b": {"c": {"d": "value"}}}}
    result = render(template, data)
    assert result == "value"


def test_render_missing_variable() -> None:
    """Test rendering with missing variable returns empty string."""
    template = "Hello {{name}}!"
    data = {}
    result = render(template, data)
    assert result == "Hello !"


def test_render_partial() -> None:
    """Test rendering with partials."""
    template = "{{> partial}}"
    data = {"name": "World"}
    partials = {"partial": "Hello {{name}}!"}
    result = render(template, data, partials_dict=partials)
    assert result == "Hello World!"


def test_render_dot_notation() -> None:
    """Test dot notation for current context."""
    template = "{{#items}}{{.}} {{/items}}"
    data = {"items": [1, 2, 3]}
    result = render(template, data)
    assert result == "1 2 3 "


def test_render_boolean_true() -> None:
    """Test rendering boolean true value."""
    template = "{{#flag}}Yes{{/flag}}"
    data = {"flag": True}
    result = render(template, data)
    assert result == "Yes"


def test_render_boolean_false() -> None:
    """Test rendering boolean false value (falsy section)."""
    template = "{{#flag}}Yes{{/flag}}"
    data = {"flag": False}
    result = render(template, data)
    assert result == ""


def test_render_empty_list() -> None:
    """Test rendering empty list (falsy)."""
    template = "{{#items}}Item{{/items}}"
    data = {"items": []}
    result = render(template, data)
    assert result == ""


def test_render_zero_value() -> None:
    """Test rendering zero value (should not be falsy)."""
    template = "{{value}}"
    data = {"value": 0}
    result = render(template, data)
    assert result == "0"


def test_tokenize_basic() -> None:
    """Test basic tokenization."""
    template = "Hello {{name}}!"
    tokens = list(tokenize(template))
    assert len(tokens) == 3
    assert tokens[0] == ("literal", "Hello ")
    assert tokens[1] == ("variable", "name")
    assert tokens[2] == ("literal", "!")


def test_tokenize_section() -> None:
    """Test tokenizing sections."""
    template = "{{#section}}content{{/section}}"
    tokens = list(tokenize(template))
    assert ("section", "section") in tokens
    assert ("end", "section") in tokens


def test_tokenize_unclosed_tag() -> None:
    """Test tokenizing unclosed tag raises error."""
    template = "{{unclosed"
    with pytest.raises(ChevronError, match="unclosed tag"):
        list(tokenize(template))


def test_tokenize_unclosed_section() -> None:
    """Test tokenizing unclosed section raises error."""
    template = "{{#section}}content"
    with pytest.raises(ChevronError, match="was never closed"):
        list(tokenize(template))


def test_tokenize_mismatched_section() -> None:
    """Test tokenizing mismatched section tags."""
    template = "{{#section1}}{{/section2}}"
    with pytest.raises(ChevronError, match="Trying to close tag"):
        list(tokenize(template))


def test_render_with_custom_delimiters() -> None:
    """Test rendering with custom delimiters."""
    template = "[[name]]"
    data = {"name": "World"}
    result = render(template, data, def_ldel="[[", def_rdel="]]")
    assert result == "World"


def test_render_set_delimiter() -> None:
    """Test set delimiter tag."""
    template = "{{name}} {{=<% %>=}} <%name%>"
    data = {"name": "World"}
    result = render(template, data)
    # Set delimiter tags are standalone and may add extra whitespace
    assert result == "World  World"


def test_render_warn_missing_key() -> None:
    """Test warning about missing keys."""
    template = "{{missing}}"
    data = {}
    # Should not raise, just return empty string
    result = render(template, data, warn=True)
    assert result == ""


def test_render_keep_unreplaced() -> None:
    """Test keeping unreplaced tags."""
    template = "{{missing}}"
    data = {}
    result = render(template, data, keep=True)
    assert result == "{{ missing }}"


def test_render_callable_section() -> None:
    """Test section with callable."""
    template = "{{#wrapper}}{{name}}{{/wrapper}}"

    def wrapper(text: str, render_func):  # type: ignore[no-untyped-def]
        return f"<b>{render_func(text)}</b>"

    data = {"name": "World", "wrapper": wrapper}
    result = render(template, data)
    assert result == "<b>World</b>"


def test_render_list_index_access() -> None:
    """Test accessing list elements by index."""
    template = "{{items.0}}"
    data = {"items": ["first", "second", "third"]}
    result = render(template, data)
    assert result == "first"


def test_render_security_no_object_traversal() -> None:
    """Test that arbitrary object traversal is prevented."""
    class CustomObj:
        secret = "should not access"

    template = "{{obj.secret}}"
    data = {"obj": CustomObj()}

    # Should either return empty string or raise TypeError for security
    # The implementation returns empty string when attribute access fails
    result = render(template, data)
    assert result == ""


def test_render_tuple_access() -> None:
    """Test accessing tuple elements."""
    template = "{{items.0}}"
    data = {"items": ("first", "second")}
    result = render(template, data)
    assert result == "first"
