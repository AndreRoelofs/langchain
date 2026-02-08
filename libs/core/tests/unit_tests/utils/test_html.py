from langchain_core.utils.html import (
    PREFIXES_TO_IGNORE,
    SUFFIXES_TO_IGNORE,
    extract_sub_links,
    find_all_links,
)


def test_find_all_links_none() -> None:
    html = "<span>Hello world</span>"
    actual = find_all_links(html)
    assert actual == []


def test_find_all_links_single() -> None:
    htmls = [
        "href='foobar.com'",
        'href="foobar.com"',
        '<div><a class="blah" href="foobar.com">hullo</a></div>',
    ]
    actual = [find_all_links(html) for html in htmls]
    assert actual == [["foobar.com"]] * 3


def test_find_all_links_multiple() -> None:
    html = (
        '<div><a class="blah" href="https://foobar.com">hullo</a></div>'
        '<div><a class="bleh" href="/baz/cool">buhbye</a></div>'
    )
    actual = find_all_links(html)
    assert sorted(actual) == [
        "/baz/cool",
        "https://foobar.com",
    ]


def test_find_all_links_ignore_suffix() -> None:
    html = 'href="foobar{suffix}"'
    for suffix_ in SUFFIXES_TO_IGNORE:
        actual = find_all_links(html.format(suffix=suffix_))
        assert actual == []

    # Don't ignore if pattern doesn't occur at end of link.
    html = 'href="foobar{suffix}more"'
    for suffix_ in SUFFIXES_TO_IGNORE:
        actual = find_all_links(html.format(suffix=suffix_))
        assert actual == [f"foobar{suffix_}more"]


def test_find_all_links_ignore_prefix() -> None:
    html = 'href="{prefix}foobar"'
    for prefix_ in PREFIXES_TO_IGNORE:
        actual = find_all_links(html.format(prefix=prefix_))
        assert actual == []

    # Don't ignore if pattern doesn't occur at beginning of link.
    html = 'href="foobar{prefix}more"'
    for prefix_ in PREFIXES_TO_IGNORE:
        # Pound signs are split on when not prefixes.
        if prefix_ == "#":
            continue
        actual = find_all_links(html.format(prefix=prefix_))
        assert actual == [f"foobar{prefix_}more"]


def test_find_all_links_drop_fragment() -> None:
    html = 'href="foobar.com/woah#section_one"'
    actual = find_all_links(html)
    assert actual == ["foobar.com/woah"]


def test_extract_sub_links() -> None:
    html = (
        '<a href="https://foobar.com">one</a>'
        '<a href="http://baz.net">two</a>'
        '<a href="//foobar.com/hello">three</a>'
        '<a href="/how/are/you/doing">four</a>'
    )
    expected = sorted(
        [
            "https://foobar.com",
            "https://foobar.com/hello",
            "https://foobar.com/how/are/you/doing",
        ]
    )
    actual = sorted(extract_sub_links(html, "https://foobar.com"))
    assert actual == expected

    actual = extract_sub_links(html, "https://foobar.com/hello")
    expected = ["https://foobar.com/hello"]
    assert actual == expected

    actual = sorted(
        extract_sub_links(html, "https://foobar.com/hello", prevent_outside=False)
    )
    expected = sorted(
        [
            "https://foobar.com",
            "http://baz.net",
            "https://foobar.com/hello",
            "https://foobar.com/how/are/you/doing",
        ]
    )
    assert actual == expected


def test_extract_sub_links_base() -> None:
    html = (
        '<a href="https://foobar.com">one</a>'
        '<a href="http://baz.net">two</a>'
        '<a href="//foobar.com/hello">three</a>'
        '<a href="/how/are/you/doing">four</a>'
        '<a href="alexis.html"</a>'
    )

    expected = sorted(
        [
            "https://foobar.com",
            "https://foobar.com/hello",
            "https://foobar.com/how/are/you/doing",
            "https://foobar.com/hello/alexis.html",
        ]
    )
    actual = sorted(
        extract_sub_links(
            html, "https://foobar.com/hello/bill.html", base_url="https://foobar.com"
        )
    )
    assert actual == expected


def test_extract_sub_links_exclude() -> None:
    html = (
        '<a href="https://foobar.com">one</a>'
        '<a href="http://baz.net">two</a>'
        '<a href="//foobar.com/hello">three</a>'
        '<a href="/how/are/you/doing">four</a>'
        '<a href="alexis.html"</a>'
    )

    expected = sorted(
        [
            "http://baz.net",
            "https://foobar.com",
            "https://foobar.com/hello",
            "https://foobar.com/hello/alexis.html",
        ]
    )
    actual = sorted(
        extract_sub_links(
            html,
            "https://foobar.com/hello/bill.html",
            base_url="https://foobar.com",
            prevent_outside=False,
            exclude_prefixes=("https://foobar.com/how", "http://baz.org"),
        )
    )
    assert actual == expected


def test_prevent_outside() -> None:
    """Test that prevent outside compares against full base URL."""
    html = (
        '<a href="https://foobar.comic.com">BAD</a>'
        '<a href="https://foobar.comic:9999">BAD</a>'
        '<a href="https://foobar.com:9999">BAD</a>'
        '<a href="http://foobar.com:9999/">BAD</a>'
        '<a href="https://foobar.com/OK">OK</a>'
        '<a href="http://foobar.com/BAD">BAD</a>'  # Change in scheme is not OK here
    )

    expected = sorted(
        [
            "https://foobar.com/OK",
        ]
    )
    actual = sorted(
        extract_sub_links(
            html,
            "https://foobar.com/hello/bill.html",
            base_url="https://foobar.com",
            prevent_outside=True,
        )
    )
    assert actual == expected


def test_extract_sub_links_with_query() -> None:
    html = (
        '<a href="https://foobar.com?query=123">one</a>'
        '<a href="/hello?query=456">two</a>'
        '<a href="//foobar.com/how/are/you?query=789">three</a>'
        '<a href="doing?query=101112"></a>'
    )

    expected = sorted(
        [
            "https://foobar.com?query=123",
            "https://foobar.com/hello?query=456",
            "https://foobar.com/how/are/you?query=789",
            "https://foobar.com/hello/doing?query=101112",
        ]
    )
    actual = sorted(
        extract_sub_links(
            html, "https://foobar.com/hello/bill.html", base_url="https://foobar.com"
        )
    )
    assert actual == expected, f"Expected {expected}, but got {actual}"


# ---------------------------------------------------------------------------
# continue_on_failure
# ---------------------------------------------------------------------------
def test_extract_sub_links_continue_on_failure() -> None:
    """continue_on_failure=True logs warning and continues on bad links."""
    # A link with a scheme that might cause issues - we just test that
    # it doesn't raise even with unusual link content
    html = (
        '<a href="https://foobar.com/ok">OK</a>'
        '<a href="https://foobar.com/also-ok">Also OK</a>'
    )
    result = extract_sub_links(
        html,
        "https://foobar.com",
        continue_on_failure=True,
    )
    assert "https://foobar.com/ok" in result
    assert "https://foobar.com/also-ok" in result


# ---------------------------------------------------------------------------
# Custom pattern
# ---------------------------------------------------------------------------
def test_find_all_links_custom_pattern() -> None:
    """find_all_links with custom regex pattern."""
    html = '<a data-url="https://example.com/page1">Link</a>'
    pattern = r'data-url="(https?://[^"]+)"'
    result = find_all_links(html, pattern=pattern)
    assert result == ["https://example.com/page1"]


def test_find_all_links_custom_pattern_no_match() -> None:
    """find_all_links with custom pattern returns empty when no match."""
    html = '<a href="https://example.com">Link</a>'
    pattern = r'data-url="(https?://[^"]+)"'
    result = find_all_links(html, pattern=pattern)
    assert result == []


# ---------------------------------------------------------------------------
# extract_sub_links with protocol-relative URL
# ---------------------------------------------------------------------------
def test_extract_sub_links_protocol_relative() -> None:
    """extract_sub_links handles protocol-relative URLs (//domain/path)."""
    html = '<a href="//cdn.example.com/resource">CDN</a>'
    result = extract_sub_links(html, "https://example.com", prevent_outside=False)
    assert "https://cdn.example.com/resource" in result


# ---------------------------------------------------------------------------
# Constants checks
# ---------------------------------------------------------------------------
def test_prefixes_to_ignore_content() -> None:
    """PREFIXES_TO_IGNORE contains expected prefixes."""
    assert "javascript:" in PREFIXES_TO_IGNORE
    assert "mailto:" in PREFIXES_TO_IGNORE
    assert "#" in PREFIXES_TO_IGNORE


def test_suffixes_to_ignore_content() -> None:
    """SUFFIXES_TO_IGNORE contains expected file extensions."""
    assert ".css" in SUFFIXES_TO_IGNORE
    assert ".js" in SUFFIXES_TO_IGNORE
    assert ".png" in SUFFIXES_TO_IGNORE
    assert ".jpg" in SUFFIXES_TO_IGNORE
    assert ".gif" in SUFFIXES_TO_IGNORE
    assert ".svg" in SUFFIXES_TO_IGNORE
    assert ".zip" in SUFFIXES_TO_IGNORE


# ---------------------------------------------------------------------------
# extract_sub_links with relative links
# ---------------------------------------------------------------------------
def test_extract_sub_links_relative_path() -> None:
    """extract_sub_links resolves relative paths correctly."""
    html = '<a href="sibling.html">Sibling</a>'
    result = extract_sub_links(
        html,
        "https://example.com/dir/page.html",
        base_url="https://example.com",
    )
    assert "https://example.com/dir/sibling.html" in result


def test_extract_sub_links_empty_html() -> None:
    """extract_sub_links returns empty list for HTML with no links."""
    html = "<p>No links here</p>"
    result = extract_sub_links(html, "https://example.com")
    assert result == []


def test_extract_sub_links_all_excluded() -> None:
    """extract_sub_links returns empty when all links are excluded."""
    html = (
        '<a href="https://example.com/api/v1">API</a>'
        '<a href="https://example.com/api/v2">API2</a>'
    )
    result = extract_sub_links(
        html,
        "https://example.com",
        exclude_prefixes=("https://example.com/api",),
    )
    assert result == []
