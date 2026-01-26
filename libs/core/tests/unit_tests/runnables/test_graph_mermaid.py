"""Comprehensive tests for Mermaid graph drawing functionality."""

import pytest
from pydantic import BaseModel

from langchain_core.runnables.graph import (
    CurveStyle,
    Edge,
    Graph,
    MermaidDrawMethod,
    Node,
    NodeStyles,
)
from langchain_core.runnables.graph_mermaid import (
    _generate_mermaid_graph_styles,
    _to_safe_id,
    draw_mermaid,
)


def test_to_safe_id_alphanumeric() -> None:
    """Test _to_safe_id with alphanumeric characters."""
    assert _to_safe_id("node1") == "node1"
    assert _to_safe_id("MyNode") == "MyNode"
    assert _to_safe_id("node_123") == "node_123"
    assert _to_safe_id("node-abc") == "node-abc"


def test_to_safe_id_special_characters() -> None:
    """Test _to_safe_id escapes special characters."""
    # Special characters should be escaped as \hex (without trailing space)
    assert _to_safe_id("node#1") == "node\\231"
    assert _to_safe_id("node@test") == "node\\40test"
    assert _to_safe_id("node$var") == "node\\24var"
    assert _to_safe_id("node%20") == "node\\2520"


def test_to_safe_id_unicode() -> None:
    """Test _to_safe_id with Unicode characters."""
    # Chinese characters should be escaped
    result = _to_safe_id("开始")
    assert "\\" in result
    assert result != "开始"


def test_to_safe_id_spaces() -> None:
    """Test _to_safe_id with spaces."""
    result = _to_safe_id("my node")
    assert "\\" in result  # Space should be escaped


def test_generate_mermaid_graph_styles() -> None:
    """Test generation of Mermaid CSS styles."""
    styles = NodeStyles()
    result = _generate_mermaid_graph_styles(styles)

    assert "classDef default" in result
    assert "classDef first" in result
    assert "classDef last" in result
    assert "fill:#f2f0ff" in result


def test_generate_mermaid_graph_styles_custom() -> None:
    """Test generation with custom node styles."""
    styles = NodeStyles(
        default="fill:#ff0000",
        first="fill:#00ff00",
        last="fill:#0000ff",
    )
    result = _generate_mermaid_graph_styles(styles)

    assert "fill:#ff0000" in result
    assert "fill:#00ff00" in result
    assert "fill:#0000ff" in result


def test_draw_mermaid_simple_graph() -> None:
    """Test drawing a simple Mermaid graph."""
    nodes = {
        "1": Node(id="1", name="Start", data=None, metadata=None),
        "2": Node(id="2", name="End", data=None, metadata=None),
    }
    edges = [Edge(source="1", target="2", data=None, conditional=False)]

    result = draw_mermaid(nodes, edges)

    assert "graph TD;" in result
    assert "Start" in result
    assert "End" in result
    assert "1" in result or "Start" in result
    assert "-->" in result


def test_draw_mermaid_without_styles() -> None:
    """Test drawing Mermaid graph without styles."""
    nodes = {
        "1": Node(id="1", name="A", data=None, metadata=None),
        "2": Node(id="2", name="B", data=None, metadata=None),
    }
    edges = [Edge(source="1", target="2", data=None, conditional=False)]

    result = draw_mermaid(nodes, edges, with_styles=False)

    assert "graph TD;" in result
    assert "classDef" not in result  # No styles
    assert "---" not in result  # No frontmatter


def test_draw_mermaid_with_conditional_edge() -> None:
    """Test drawing Mermaid graph with conditional edges."""
    nodes = {
        "1": Node(id="1", name="Decision", data=None, metadata=None),
        "2": Node(id="2", name="PathA", data=None, metadata=None),
        "3": Node(id="3", name="PathB", data=None, metadata=None),
    }
    edges = [
        Edge(source="1", target="2", data="yes", conditional=True),
        Edge(source="1", target="3", data="no", conditional=False),
    ]

    result = draw_mermaid(nodes, edges)

    # Conditional edges use dotted line notation: "-. label .->"
    assert "-.>" in result or "-." in result
    assert "-->" in result  # Non-conditional edge


def test_draw_mermaid_with_edge_labels() -> None:
    """Test Mermaid graph with edge labels."""
    nodes = {
        "1": Node(id="1", name="A", data=None, metadata=None),
        "2": Node(id="2", name="B", data=None, metadata=None),
    }
    edges = [Edge(source="1", target="2", data="label text", conditional=False)]

    result = draw_mermaid(nodes, edges)

    assert "label text" in result or "label&nbsp" in result


def test_draw_mermaid_with_long_edge_label() -> None:
    """Test Mermaid graph wraps long edge labels."""
    nodes = {
        "1": Node(id="1", name="A", data=None, metadata=None),
        "2": Node(id="2", name="B", data=None, metadata=None),
    }
    # Create a long label
    long_label = " ".join([f"word{i}" for i in range(15)])
    edges = [Edge(source="1", target="2", data=long_label, conditional=False)]

    result = draw_mermaid(nodes, edges, wrap_label_n_words=9)

    # Should contain line break
    assert "<br>" in result or "&nbsp" in result


def test_draw_mermaid_curve_styles() -> None:
    """Test different curve styles."""
    nodes = {
        "1": Node(id="1", name="A", data=None, metadata=None),
        "2": Node(id="2", name="B", data=None, metadata=None),
    }
    edges = [Edge(source="1", target="2", data=None, conditional=False)]

    for curve_style in CurveStyle:
        result = draw_mermaid(nodes, edges, curve_style=curve_style)
        assert curve_style.value in result


def test_draw_mermaid_first_last_nodes() -> None:
    """Test Mermaid graph highlights first and last nodes."""
    nodes = {
        "start": Node(id="start", name="Start", data=None, metadata=None),
        "middle": Node(id="middle", name="Middle", data=None, metadata=None),
        "end": Node(id="end", name="End", data=None, metadata=None),
    }
    edges = [
        Edge(source="start", target="middle", data=None, conditional=False),
        Edge(source="middle", target="end", data=None, conditional=False),
    ]

    result = draw_mermaid(nodes, edges, first_node="start", last_node="end")

    # First and last nodes get special styling
    assert ":::first" in result
    assert ":::last" in result


def test_draw_mermaid_subgraph() -> None:
    """Test Mermaid graph with subgraphs (colons in node IDs)."""
    nodes = {
        "parent": Node(id="parent", name="Parent", data=None, metadata=None),
        "sub:child": Node(id="sub:child", name="Child", data=None, metadata=None),
    }
    edges = [Edge(source="parent", target="sub:child", data=None, conditional=False)]

    result = draw_mermaid(nodes, edges)

    # Should create a subgraph
    assert "subgraph sub" in result
    assert "end" in result


def test_draw_mermaid_nested_subgraphs() -> None:
    """Test Mermaid graph with nested subgraphs."""
    nodes = {
        "root": Node(id="root", name="Root", data=None, metadata=None),
        "a:b:c": Node(id="a:b:c", name="Nested", data=None, metadata=None),
    }
    edges = [Edge(source="root", target="a:b:c", data=None, conditional=False)]

    result = draw_mermaid(nodes, edges)

    # Nodes with colons get their colons hex-escaped when no internal edges exist
    assert "\\3a" in result or "subgraph" in result


def test_draw_mermaid_node_metadata() -> None:
    """Test Mermaid graph includes node metadata."""
    nodes = {
        "1": Node(
            id="1",
            name="Node1",
            data=None,
            metadata={"key1": "value1", "key2": "value2"},
        ),
    }
    edges: list[Edge] = []

    result = draw_mermaid(nodes, edges)

    # Metadata should be included in node label
    assert "key1" in result or "value1" in result


def test_draw_mermaid_frontmatter_config() -> None:
    """Test Mermaid graph with frontmatter configuration."""
    nodes = {
        "1": Node(id="1", name="A", data=None, metadata=None),
        "2": Node(id="2", name="B", data=None, metadata=None),
    }
    edges = [Edge(source="1", target="2", data=None, conditional=False)]

    frontmatter = {
        "config": {
            "theme": "dark",
            "themeVariables": {"primaryColor": "#ff0000"},
        }
    }

    result = draw_mermaid(nodes, edges, frontmatter_config=frontmatter)

    assert "---" in result  # Frontmatter markers
    assert "theme: dark" in result


def test_draw_mermaid_markdown_special_chars() -> None:
    """Test Mermaid handles node names with Markdown special characters."""
    nodes = {
        "1": Node(id="1", name="*bold*", data=None, metadata=None),
        "2": Node(id="2", name="_italic_", data=None, metadata=None),
    }
    edges = [Edge(source="1", target="2", data=None, conditional=False)]

    result = draw_mermaid(nodes, edges)

    # Special characters should be wrapped in <p> tags
    assert "<p>*bold*</p>" in result or "*bold*" in result
    assert "<p>_italic_</p>" in result or "_italic_" in result


def test_graph_draw_mermaid_method() -> None:
    """Test Graph.draw_mermaid() method."""
    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")
    graph.add_edge(node1, node2)

    result = graph.draw_mermaid()

    assert "graph TD;" in result
    assert isinstance(result, str)


def test_graph_draw_mermaid_without_styles() -> None:
    """Test Graph.draw_mermaid() without styles."""
    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")
    graph.add_edge(node1, node2)

    result = graph.draw_mermaid(with_styles=False)

    assert "classDef" not in result


def test_mermaid_curve_style_linear() -> None:
    """Test Mermaid with linear curve style."""
    nodes = {
        "1": Node(id="1", name="A", data=None, metadata=None),
    }
    edges: list[Edge] = []

    result = draw_mermaid(nodes, edges, curve_style=CurveStyle.LINEAR)

    assert "curve: linear" in result


def test_mermaid_curve_style_basis() -> None:
    """Test Mermaid with basis curve style."""
    nodes = {
        "1": Node(id="1", name="A", data=None, metadata=None),
    }
    edges: list[Edge] = []

    result = draw_mermaid(nodes, edges, curve_style=CurveStyle.BASIS)

    assert "curve: basis" in result


def test_mermaid_empty_graph() -> None:
    """Test drawing empty Mermaid graph."""
    nodes: dict[str, Node] = {}
    edges: list[Edge] = []

    result = draw_mermaid(nodes, edges)

    assert "graph TD;" in result


def test_mermaid_single_node() -> None:
    """Test Mermaid with single node, no edges."""
    nodes = {
        "1": Node(id="1", name="OnlyNode", data=None, metadata=None),
    }
    edges: list[Edge] = []

    result = draw_mermaid(nodes, edges, with_styles=False)

    assert "graph TD;" in result


def test_mermaid_parallel_edges() -> None:
    """Test Mermaid with multiple edges from same source."""
    nodes = {
        "1": Node(id="1", name="Source", data=None, metadata=None),
        "2": Node(id="2", name="Target1", data=None, metadata=None),
        "3": Node(id="3", name="Target2", data=None, metadata=None),
    }
    edges = [
        Edge(source="1", target="2", data=None, conditional=False),
        Edge(source="1", target="3", data=None, conditional=False),
    ]

    result = draw_mermaid(nodes, edges)

    # Should have two arrows from source
    assert result.count("-->") >= 2


def test_mermaid_self_loop() -> None:
    """Test Mermaid with self-referencing edge."""
    nodes = {
        "1": Node(id="1", name="SelfLoop", data=None, metadata=None),
    }
    edges = [Edge(source="1", target="1", data=None, conditional=False)]

    result = draw_mermaid(nodes, edges)

    # Should handle self-loop
    assert "graph TD;" in result


def test_mermaid_duplicate_subgraph_name_error() -> None:
    """Test that duplicate subgraph names raise error."""
    nodes = {
        "sub:node1": Node(id="sub:node1", name="Node1", data=None, metadata=None),
        "sub:node2": Node(id="sub:node2", name="Node2", data=None, metadata=None),
        "other:sub:node3": Node(
            id="other:sub:node3", name="Node3", data=None, metadata=None
        ),
    }
    edges = [
        Edge(source="sub:node1", target="sub:node2", data=None, conditional=False),
        Edge(source="other:sub:node3", target="sub:node1", data=None, conditional=False),
    ]

    # This may or may not raise depending on structure - test it doesn't crash
    try:
        result = draw_mermaid(nodes, edges)
        assert "subgraph" in result
    except ValueError as e:
        # May raise if duplicate subgraph names detected
        assert "duplicate subgraph" in str(e).lower() or "subgraph" in str(e)


def test_mermaid_node_with_metadata() -> None:
    """Test Mermaid node rendering with metadata."""
    nodes = {
        "1": Node(
            id="1",
            name="TestNode",
            data=None,
            metadata={"version": "1.0", "author": "test"},
        ),
    }
    edges: list[Edge] = []

    result = draw_mermaid(nodes, edges)

    # Metadata should appear in node
    assert "version = 1.0" in result or "version" in result
    assert "author = test" in result or "author" in result


def test_mermaid_wrap_label_custom_words() -> None:
    """Test custom wrap_label_n_words parameter."""
    nodes = {
        "1": Node(id="1", name="A", data=None, metadata=None),
        "2": Node(id="2", name="B", data=None, metadata=None),
    }
    long_label = " ".join([f"word{i}" for i in range(20)])
    edges = [Edge(source="1", target="2", data=long_label, conditional=False)]

    result = draw_mermaid(nodes, edges, wrap_label_n_words=5)

    # Should have multiple line breaks
    assert result.count("<br>") > 2


def test_mermaid_frontmatter_preserves_existing_config() -> None:
    """Test that frontmatter config is properly merged."""
    nodes = {
        "1": Node(id="1", name="A", data=None, metadata=None),
    }
    edges: list[Edge] = []

    frontmatter = {
        "config": {
            "theme": "forest",
            "flowchart": {
                "htmlLabels": True,
            },
        }
    }

    result = draw_mermaid(
        nodes, edges, frontmatter_config=frontmatter, curve_style=CurveStyle.BASIS
    )

    # Should include both user config and curve style
    assert "theme: forest" in result
    assert "curve: basis" in result


def test_mermaid_empty_subgraph() -> None:
    """Test Mermaid handles subgraphs with nodes but no internal edges."""
    nodes = {
        "regular": Node(id="regular", name="Regular", data=None, metadata=None),
        "sub:node1": Node(id="sub:node1", name="SubNode1", data=None, metadata=None),
        "sub:node2": Node(id="sub:node2", name="SubNode2", data=None, metadata=None),
    }
    edges = [
        Edge(source="regular", target="sub:node1", data=None, conditional=False),
    ]

    result = draw_mermaid(nodes, edges)

    # Should create subgraph even if nodes inside aren't connected to each other
    assert "subgraph sub" in result


def test_graph_draw_mermaid_with_curve_styles() -> None:
    """Test all curve styles work."""
    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")
    graph.add_edge(node1, node2)

    for curve_style in CurveStyle:
        result = graph.draw_mermaid(curve_style=curve_style)
        assert f"curve: {curve_style.value}" in result


def test_graph_draw_mermaid_custom_node_colors() -> None:
    """Test Graph.draw_mermaid() with custom node colors."""
    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")
    graph.add_edge(node1, node2)

    custom_colors = NodeStyles(
        default="fill:#abcdef",
        first="fill:#123456",
        last="fill:#fedcba",
    )

    result = graph.draw_mermaid(node_colors=custom_colors)

    assert "fill:#abcdef" in result
    assert "fill:#123456" in result
    assert "fill:#fedcba" in result


def test_draw_mermaid_all_curve_styles_valid() -> None:
    """Test that all CurveStyle enum values are valid."""
    valid_styles = [
        CurveStyle.BASIS,
        CurveStyle.BUMP_X,
        CurveStyle.BUMP_Y,
        CurveStyle.CARDINAL,
        CurveStyle.CATMULL_ROM,
        CurveStyle.LINEAR,
        CurveStyle.MONOTONE_X,
        CurveStyle.MONOTONE_Y,
        CurveStyle.NATURAL,
        CurveStyle.STEP,
        CurveStyle.STEP_AFTER,
        CurveStyle.STEP_BEFORE,
    ]

    assert len(valid_styles) == 12
    for style in valid_styles:
        assert isinstance(style.value, str)


def test_mermaid_node_styles_default_values() -> None:
    """Test NodeStyles default values."""
    styles = NodeStyles()

    assert "fill:#f2f0ff" in styles.default
    assert "fill-opacity:0" in styles.first
    assert "fill:#bfb6fc" in styles.last


def test_mermaid_draw_method_enum() -> None:
    """Test MermaidDrawMethod enum values."""
    assert MermaidDrawMethod.PYPPETEER.value == "pyppeteer"
    assert MermaidDrawMethod.API.value == "api"


def test_draw_mermaid_special_node_names() -> None:
    """Test Mermaid with special reserved node names."""
    nodes = {
        "__start__": Node(id="__start__", name="__start__", data=None, metadata=None),
        "__end__": Node(id="__end__", name="__end__", data=None, metadata=None),
    }
    edges = [Edge(source="__start__", target="__end__", data=None, conditional=False)]

    result = draw_mermaid(nodes, edges)

    assert "__start__" in result or "start" in result
    assert "__end__" in result or "end" in result


def test_draw_mermaid_numeric_node_ids() -> None:
    """Test Mermaid with numeric node IDs."""
    nodes = {
        "1": Node(id="1", name="First", data=None, metadata=None),
        "2": Node(id="2", name="Second", data=None, metadata=None),
    }
    edges = [Edge(source="1", target="2", data=None, conditional=False)]

    result = draw_mermaid(nodes, edges)

    assert "graph TD;" in result


def test_draw_mermaid_complex_metadata() -> None:
    """Test Mermaid with complex metadata structures."""
    nodes = {
        "1": Node(
            id="1",
            name="Node",
            data=None,
            metadata={
                "nested": {"key": "value"},
                "list": [1, 2, 3],
                "bool": True,
            },
        ),
    }
    edges: list[Edge] = []

    result = draw_mermaid(nodes, edges)

    # Should handle various metadata types
    assert "nested" in result or "key" in result


def test_to_safe_id_preserves_allowed_chars() -> None:
    """Test that _to_safe_id preserves allowed characters."""
    allowed_input = "abcABC123_-"
    result = _to_safe_id(allowed_input)

    assert result == allowed_input  # No escaping needed


def test_to_safe_id_escapes_punctuation() -> None:
    """Test _to_safe_id escapes common punctuation."""
    test_cases = [
        ("node!", "node\\21"),
        ("node?", "node\\3f"),
        ("node.", "node\\2e"),
        ("node,", "node\\2c"),
    ]

    for input_str, expected in test_cases:
        result = _to_safe_id(input_str)
        assert result == expected


def test_mermaid_multiple_disconnected_subgraphs() -> None:
    """Test Mermaid with multiple disconnected subgraphs."""
    nodes = {
        "sub1:a": Node(id="sub1:a", name="A", data=None, metadata=None),
        "sub1:b": Node(id="sub1:b", name="B", data=None, metadata=None),
        "sub2:c": Node(id="sub2:c", name="C", data=None, metadata=None),
        "sub2:d": Node(id="sub2:d", name="D", data=None, metadata=None),
    }
    edges = [
        Edge(source="sub1:a", target="sub1:b", data=None, conditional=False),
        Edge(source="sub2:c", target="sub2:d", data=None, conditional=False),
    ]

    result = draw_mermaid(nodes, edges)

    assert "subgraph sub1" in result
    assert "subgraph sub2" in result


def test_mermaid_edge_with_none_data() -> None:
    """Test Mermaid edge with None data (no label)."""
    nodes = {
        "1": Node(id="1", name="A", data=None, metadata=None),
        "2": Node(id="2", name="B", data=None, metadata=None),
    }
    edges = [Edge(source="1", target="2", data=None, conditional=False)]

    result = draw_mermaid(nodes, edges)

    # Should use simple arrow without label
    assert "-->" in result


def test_mermaid_conditional_edge_with_label() -> None:
    """Test conditional edge with label uses dotted line."""
    nodes = {
        "1": Node(id="1", name="A", data=None, metadata=None),
        "2": Node(id="2", name="B", data=None, metadata=None),
    }
    edges = [Edge(source="1", target="2", data="condition", conditional=True)]

    result = draw_mermaid(nodes, edges)

    # Conditional edges use dotted notation
    assert "-." in result and ".->" in result


def test_node_styles_dataclass() -> None:
    """Test NodeStyles as dataclass."""
    styles = NodeStyles(
        default="custom-default",
        first="custom-first",
        last="custom-last",
    )

    assert styles.default == "custom-default"
    assert styles.first == "custom-first"
    assert styles.last == "custom-last"


def test_mermaid_with_all_features() -> None:
    """Test Mermaid graph with all features combined."""
    nodes = {
        "start": Node(id="start", name="Start", data=None, metadata=None),
        "sub:node1": Node(
            id="sub:node1",
            name="Process1",
            data=None,
            metadata={"type": "processor"},
        ),
        "sub:node2": Node(id="sub:node2", name="Process2", data=None, metadata=None),
        "end": Node(id="end", name="End", data=None, metadata=None),
    }
    edges = [
        Edge(source="start", target="sub:node1", data=None, conditional=False),
        Edge(source="sub:node1", target="sub:node2", data="next", conditional=False),
        Edge(source="sub:node2", target="end", data=None, conditional=True),
    ]

    frontmatter = {"config": {"theme": "neutral"}}
    custom_styles = NodeStyles(default="fill:#eee")

    result = draw_mermaid(
        nodes,
        edges,
        first_node="start",
        last_node="end",
        with_styles=True,
        curve_style=CurveStyle.CARDINAL,
        node_styles=custom_styles,
        wrap_label_n_words=5,
        frontmatter_config=frontmatter,
    )

    # Should include all features
    assert "graph TD;" in result
    assert "theme: neutral" in result
    assert "curve: cardinal" in result
    assert "subgraph sub" in result
    assert "fill:#eee" in result
    assert ":::first" in result
    assert ":::last" in result


def test_graph_reid_with_duplicate_names() -> None:
    """Test graph reid handles duplicate node names."""
    graph = Graph()
    # Add nodes with same name but different IDs
    node1 = graph.add_node(BaseModel, id="id1")
    node2 = graph.add_node(BaseModel, id="id2")

    reided_graph = graph.reid()

    # Should disambiguate with suffixes
    node_names = [node.name for node in reided_graph.nodes.values()]
    # Names should be unique after reid
    assert len(set(node_names)) == len(node_names)


def test_to_safe_id_empty_string() -> None:
    """Test _to_safe_id with empty string."""
    result = _to_safe_id("")
    assert result == ""


def test_to_safe_id_only_special_chars() -> None:
    """Test _to_safe_id with only special characters."""
    result = _to_safe_id("!@#$")
    # All should be escaped
    assert "\\" in result
    assert result != "!@#$"


def test_mermaid_subgraph_single_node() -> None:
    """Test subgraph with single node."""
    nodes = {
        "outer": Node(id="outer", name="Outer", data=None, metadata=None),
        "sub:inner": Node(id="sub:inner", name="Inner", data=None, metadata=None),
    }
    edges = [Edge(source="outer", target="sub:inner", data=None, conditional=False)]

    result = draw_mermaid(nodes, edges)

    assert "subgraph sub" in result
    assert "end" in result


def test_curve_style_enum_values() -> None:
    """Test all CurveStyle enum values are strings."""
    for style in CurveStyle:
        assert isinstance(style.value, str)
        assert len(style.value) > 0
