"""Snapshot tests for XMLOutputParser."""

import importlib
from collections.abc import AsyncIterator, Iterable

import pytest

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.xml import (
    XML_FORMAT_INSTRUCTIONS,
    XMLOutputParser,
    _StreamingParser,
    nested_element,
)
from langchain_core.runnables.utils import AddableDict

# --- Test data ---

SIMPLE_XML = "<root><child>value</child></root>"

NESTED_XML = """<foo>
    <bar>
        <baz>inner</baz>
    </bar>
</foo>"""

MULTI_CHILD_XML = """<parent>
    <child1>val1</child1>
    <child2>val2</child2>
    <child3>val3</child3>
</parent>"""

ROOT_TEXT_XML = "<body>Text of the body.</body>"

EMPTY_ELEMENT_XML = "<root><empty></empty></root>"

XML_WITH_HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<root><item>data</item></root>'
)

XML_IN_CODE_BLOCK = "```xml\n<root><item>data</item></root>\n```"

XML_WITH_SURROUNDING_TEXT = (
    "Some text\n```xml\n<root><item>data</item></root>\n```\nMore text"
)


# --- Helpers ---


async def _as_iter(iterable: Iterable[str]) -> AsyncIterator[str]:
    for item in iterable:
        yield item


# --- XMLOutputParser.parse() tests ---


class TestXMLOutputParserParse:
    """Tests for XMLOutputParser.parse() method."""

    def test_simple_xml(self) -> None:
        parser = XMLOutputParser(parser="xml")
        result = parser.parse(SIMPLE_XML)
        assert result == {"root": [{"child": "value"}]}

    def test_root_text_only(self) -> None:
        parser = XMLOutputParser(parser="xml")
        result = parser.parse(ROOT_TEXT_XML)
        assert result == {"body": "Text of the body."}

    def test_nested_xml(self) -> None:
        parser = XMLOutputParser(parser="xml")
        result = parser.parse(NESTED_XML)
        assert result == {"foo": [{"bar": [{"baz": "inner"}]}]}

    def test_multiple_children(self) -> None:
        parser = XMLOutputParser(parser="xml")
        result = parser.parse(MULTI_CHILD_XML)
        assert result == {
            "parent": [
                {"child1": "val1"},
                {"child2": "val2"},
                {"child3": "val3"},
            ]
        }

    def test_empty_element(self) -> None:
        parser = XMLOutputParser(parser="xml")
        result = parser.parse(EMPTY_ELEMENT_XML)
        assert result == {"root": [{"empty": None}]}

    def test_xml_with_header(self) -> None:
        parser = XMLOutputParser(parser="xml")
        result = parser.parse(XML_WITH_HEADER)
        assert result == {"root": [{"item": "data"}]}

    def test_xml_in_code_block(self) -> None:
        parser = XMLOutputParser(parser="xml")
        result = parser.parse(XML_IN_CODE_BLOCK)
        assert result == {"root": [{"item": "data"}]}

    def test_xml_with_surrounding_text(self) -> None:
        parser = XMLOutputParser(parser="xml")
        result = parser.parse(XML_WITH_SURROUNDING_TEXT)
        assert result == {"root": [{"item": "data"}]}

    def test_mixed_children_and_nested(self) -> None:
        parser = XMLOutputParser(parser="xml")
        xml = """<root>
            <simple>text</simple>
            <complex><inner>deep</inner></complex>
        </root>"""
        result = parser.parse(xml)
        assert result == {
            "root": [
                {"simple": "text"},
                {"complex": [{"inner": "deep"}]},
            ]
        }

    def test_multiple_same_tag_children(self) -> None:
        parser = XMLOutputParser(parser="xml")
        xml = """<root>
            <item>first</item>
            <item>second</item>
            <item>third</item>
        </root>"""
        result = parser.parse(xml)
        assert result == {
            "root": [
                {"item": "first"},
                {"item": "second"},
                {"item": "third"},
            ]
        }


class TestXMLOutputParserParseErrors:
    """Tests for XMLOutputParser error handling."""

    def test_invalid_xml_raises(self) -> None:
        parser = XMLOutputParser(parser="xml")
        with pytest.raises(OutputParserException, match="Failed to parse"):
            parser.parse("not xml at all")

    def test_unclosed_tag_raises(self) -> None:
        parser = XMLOutputParser(parser="xml")
        with pytest.raises(OutputParserException, match="Failed to parse"):
            parser.parse("<foo><bar></foo>")

    def test_malformed_tag_raises(self) -> None:
        parser = XMLOutputParser(parser="xml")
        with pytest.raises(OutputParserException, match="Failed to parse"):
            parser.parse("foo></foo>")

    def test_no_closing_tag_raises(self) -> None:
        parser = XMLOutputParser(parser="xml")
        with pytest.raises(OutputParserException, match="Failed to parse"):
            parser.parse("<foo></foo")

    def test_parser_exception_contains_llm_output(self) -> None:
        parser = XMLOutputParser(parser="xml")
        with pytest.raises(OutputParserException) as exc_info:
            parser.parse("bad xml")
        assert exc_info.value.llm_output is not None


class TestXMLOutputParserAsync:
    """Tests for XMLOutputParser async methods."""

    async def test_aparse(self) -> None:
        parser = XMLOutputParser(parser="xml")
        result = await parser.aparse(SIMPLE_XML)
        assert result == {"root": [{"child": "value"}]}

    async def test_aparse_root_text(self) -> None:
        parser = XMLOutputParser(parser="xml")
        result = await parser.aparse(ROOT_TEXT_XML)
        assert result == {"body": "Text of the body."}


class TestXMLOutputParserTransform:
    """Tests for XMLOutputParser streaming via transform."""

    def test_stream_char_by_char(self) -> None:
        parser = XMLOutputParser(parser="xml")
        result = list(parser.transform(iter(SIMPLE_XML)))
        assert result == [{"root": [{"child": "value"}]}]

    def test_stream_root_text_single_chunk(self) -> None:
        parser = XMLOutputParser(parser="xml")
        # char-by-char streaming of root-text-only XML yields empty
        # because the streaming parser only yields leaf elements with paths
        result = list(parser.transform(iter([ROOT_TEXT_XML])))
        assert result == [{"body": "Text of the body."}]

    def test_stream_nested_xml(self) -> None:
        parser = XMLOutputParser(parser="xml")
        xml = """<foo>
            <bar>
                <baz></baz>
                <baz>slim.shady</baz>
            </bar>
            <baz>tag</baz>
        </foo>"""
        result = list(parser.transform(iter(xml)))
        assert result == [
            {"foo": [{"bar": [{"baz": None}]}]},
            {"foo": [{"bar": [{"baz": "slim.shady"}]}]},
            {"foo": [{"baz": "tag"}]},
        ]

    async def test_atransform_char_by_char(self) -> None:
        parser = XMLOutputParser(parser="xml")
        result = [chunk async for chunk in parser.atransform(_as_iter(SIMPLE_XML))]
        assert result == [{"root": [{"child": "value"}]}]

    async def test_atransform_nested(self) -> None:
        parser = XMLOutputParser(parser="xml")
        xml = """<foo>
            <bar>
                <baz></baz>
                <baz>slim.shady</baz>
            </bar>
            <baz>tag</baz>
        </foo>"""
        result = [chunk async for chunk in parser.atransform(_as_iter(xml))]
        assert result == [
            {"foo": [{"bar": [{"baz": None}]}]},
            {"foo": [{"bar": [{"baz": "slim.shady"}]}]},
            {"foo": [{"baz": "tag"}]},
        ]


class TestXMLOutputParserDefusedXml:
    """Tests for XMLOutputParser with defusedxml backend."""

    @pytest.mark.skipif(
        importlib.util.find_spec("defusedxml") is None,
        reason="defusedxml is not installed",
    )
    def test_parse_simple(self) -> None:
        parser = XMLOutputParser(parser="defusedxml")
        result = parser.parse(SIMPLE_XML)
        assert result == {"root": [{"child": "value"}]}

    @pytest.mark.skipif(
        importlib.util.find_spec("defusedxml") is None,
        reason="defusedxml is not installed",
    )
    def test_parse_nested(self) -> None:
        parser = XMLOutputParser(parser="defusedxml")
        result = parser.parse(NESTED_XML)
        assert result == {"foo": [{"bar": [{"baz": "inner"}]}]}

    @pytest.mark.skipif(
        importlib.util.find_spec("defusedxml") is None,
        reason="defusedxml is not installed",
    )
    def test_stream_simple(self) -> None:
        parser = XMLOutputParser(parser="defusedxml")
        result = list(parser.transform(iter(SIMPLE_XML)))
        assert result == [{"root": [{"child": "value"}]}]


class TestXMLOutputParserSecurity:
    """Tests for XMLOutputParser security features."""

    MALICIOUS_XML = """<?xml version="1.0"?>
<!DOCTYPE lolz [<!ENTITY lol "lol"><!ELEMENT lolz (#PCDATA)>
 <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
 <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">
 <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
 <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
 <!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;">
 <!ENTITY lol6 "&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;">
 <!ENTITY lol7 "&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;">
 <!ENTITY lol8 "&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;">
 <!ENTITY lol9 "&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;">
]>
<lolz>&lol9;</lolz>"""

    def test_billion_laughs_sync(self) -> None:
        parser = XMLOutputParser(parser="xml")
        with pytest.raises(OutputParserException):
            parser.parse(self.MALICIOUS_XML)

    async def test_billion_laughs_async(self) -> None:
        parser = XMLOutputParser(parser="xml")
        with pytest.raises(OutputParserException):
            await parser.aparse(self.MALICIOUS_XML)


class TestXMLOutputParserProperties:
    """Tests for XMLOutputParser properties."""

    def test_type_property(self) -> None:
        parser = XMLOutputParser(parser="xml")
        assert parser._type == "xml"

    def test_default_parser_is_defusedxml(self) -> None:
        parser = XMLOutputParser()
        assert parser.parser == "defusedxml"

    def test_tags_default_none(self) -> None:
        parser = XMLOutputParser(parser="xml")
        assert parser.tags is None

    def test_get_format_instructions(self) -> None:
        parser = XMLOutputParser(parser="xml", tags=["foo", "bar"])
        instructions = parser.get_format_instructions()
        assert "foo" in instructions
        assert "bar" in instructions

    def test_get_format_instructions_with_none_tags(self) -> None:
        parser = XMLOutputParser(parser="xml")
        instructions = parser.get_format_instructions()
        assert "None" in instructions or "tags" in instructions.lower()


# --- nested_element() function tests ---


class TestNestedElement:
    """Tests for nested_element() utility function."""

    def test_empty_path(self) -> None:
        import xml.etree.ElementTree as ET

        elem = ET.Element("tag")
        elem.text = "value"
        result = nested_element([], elem)
        assert isinstance(result, AddableDict)
        assert result == {"tag": "value"}

    def test_single_level_path(self) -> None:
        import xml.etree.ElementTree as ET

        elem = ET.Element("child")
        elem.text = "content"
        result = nested_element(["parent"], elem)
        assert isinstance(result, AddableDict)
        assert result == {"parent": [{"child": "content"}]}

    def test_multi_level_path(self) -> None:
        import xml.etree.ElementTree as ET

        elem = ET.Element("leaf")
        elem.text = "data"
        result = nested_element(["root", "mid"], elem)
        assert result == {"root": [{"mid": [{"leaf": "data"}]}]}

    def test_none_text_element(self) -> None:
        import xml.etree.ElementTree as ET

        elem = ET.Element("empty")
        result = nested_element([], elem)
        assert result == {"empty": None}


# --- _StreamingParser tests ---


class TestStreamingParser:
    """Tests for _StreamingParser internal class."""

    def test_parses_simple_xml(self) -> None:
        sp = _StreamingParser(parser="xml")
        results = list(sp.parse("<root><child>val</child></root>"))
        sp.close()
        assert len(results) == 1
        assert results[0] == {"root": [{"child": "val"}]}

    def test_ignores_text_before_xml(self) -> None:
        sp = _StreamingParser(parser="xml")
        results = []
        results.extend(sp.parse("Some preamble text "))
        results.extend(sp.parse("<root>"))
        results.extend(sp.parse("<item>data</item>"))
        results.extend(sp.parse("</root>"))
        sp.close()
        assert len(results) == 1
        assert results[0] == {"root": [{"item": "data"}]}

    def test_handles_incremental_chunks(self) -> None:
        sp = _StreamingParser(parser="xml")
        all_results = []
        for chunk in ["<ro", "ot>", "<ch", "ild>", "val", "</child>", "</root>"]:
            all_results.extend(sp.parse(chunk))
        sp.close()
        assert len(all_results) == 1
        assert all_results[0] == {"root": [{"child": "val"}]}

    def test_handles_message_input(self) -> None:
        from langchain_core.messages import AIMessageChunk

        sp = _StreamingParser(parser="xml")
        msg = AIMessageChunk(content="<root><item>text</item></root>")
        results = list(sp.parse(msg))
        sp.close()
        assert len(results) == 1

    def test_handles_non_string_message_content(self) -> None:
        from langchain_core.messages import AIMessageChunk

        sp = _StreamingParser(parser="xml")
        msg = AIMessageChunk(content=[{"type": "text", "text": "hello"}])
        results = list(sp.parse(msg))
        sp.close()
        assert results == []

    def test_close_is_safe_after_incomplete(self) -> None:
        sp = _StreamingParser(parser="xml")
        list(sp.parse("<root><item>"))
        # Should not raise
        sp.close()


# --- XML_FORMAT_INSTRUCTIONS constant tests ---


class TestXMLFormatInstructions:
    """Tests for XML_FORMAT_INSTRUCTIONS constant."""

    def test_exists_and_is_string(self) -> None:
        assert isinstance(XML_FORMAT_INSTRUCTIONS, str)

    def test_contains_xml_keywords(self) -> None:
        assert "XML" in XML_FORMAT_INSTRUCTIONS
        assert "tags" in XML_FORMAT_INSTRUCTIONS.lower()

    def test_has_placeholder(self) -> None:
        assert "{tags}" in XML_FORMAT_INSTRUCTIONS

    def test_can_be_formatted(self) -> None:
        formatted = XML_FORMAT_INSTRUCTIONS.format(tags='["foo", "bar"]')
        assert "foo" in formatted
        assert "bar" in formatted
        assert "{tags}" not in formatted

    def test_contains_examples(self) -> None:
        assert "well-formatted" in XML_FORMAT_INSTRUCTIONS.lower()
        assert "badly-formatted" in XML_FORMAT_INSTRUCTIONS.lower()
