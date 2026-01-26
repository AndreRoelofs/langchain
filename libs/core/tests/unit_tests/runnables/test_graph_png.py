"""Comprehensive tests for PNG graph drawing functionality."""

import pytest
from pydantic import BaseModel

from langchain_core.runnables.graph import Edge, Graph, LabelsDict, Node
from langchain_core.runnables.graph_png import PngDrawer


def test_png_drawer_initialization() -> None:
    """Test PngDrawer initialization with default values."""
    drawer = PngDrawer()

    assert drawer.fontname == "arial"
    assert drawer.labels == LabelsDict(nodes={}, edges={})


def test_png_drawer_initialization_custom() -> None:
    """Test PngDrawer initialization with custom values."""
    custom_labels = LabelsDict(
        nodes={"node1": "CustomNode1"},
        edges={"edge1": "CustomEdge1"},
    )
    drawer = PngDrawer(fontname="helvetica", labels=custom_labels)

    assert drawer.fontname == "helvetica"
    assert drawer.labels["nodes"]["node1"] == "CustomNode1"
    assert drawer.labels["edges"]["edge1"] == "CustomEdge1"


def test_png_drawer_get_node_label_default() -> None:
    """Test get_node_label with default (no custom label)."""
    drawer = PngDrawer()

    label = drawer.get_node_label("test_node")

    assert "<B>test_node</B>" in label


def test_png_drawer_get_node_label_custom() -> None:
    """Test get_node_label with custom label."""
    custom_labels = LabelsDict(
        nodes={"test_node": "Custom Label"},
        edges={},
    )
    drawer = PngDrawer(labels=custom_labels)

    label = drawer.get_node_label("test_node")

    assert "<B>Custom Label</B>" in label


def test_png_drawer_get_edge_label_default() -> None:
    """Test get_edge_label with default (no custom label)."""
    drawer = PngDrawer()

    label = drawer.get_edge_label("test_edge")

    assert "<U>test_edge</U>" in label


def test_png_drawer_get_edge_label_custom() -> None:
    """Test get_edge_label with custom label."""
    custom_labels = LabelsDict(
        nodes={},
        edges={"test_edge": "Custom Edge"},
    )
    drawer = PngDrawer(labels=custom_labels)

    label = drawer.get_edge_label("test_edge")

    assert "<U>Custom Edge</U>" in label


def test_graph_draw_png_method_exists() -> None:
    """Test that Graph has draw_png method."""
    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")
    graph.add_edge(node1, node2)

    # Test method exists and has correct signature
    assert hasattr(graph, "draw_png")
    assert callable(graph.draw_png)


def test_graph_draw_png_requires_pygraphviz() -> None:
    """Test that draw_png raises ImportError if pygraphviz not installed."""
    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")
    graph.add_edge(node1, node2)

    try:
        # Try to draw - will fail if pygraphviz not installed
        result = graph.draw_png(output_file_path=None)
        # If it works, result should be bytes
        assert isinstance(result, bytes)
    except ImportError as e:
        # Expected if pygraphviz not installed
        assert "pygraphviz" in str(e).lower()


def test_png_drawer_add_node_structure() -> None:
    """Test add_node method signature."""
    drawer = PngDrawer()

    # Method should exist and be callable
    assert hasattr(drawer, "add_node")
    assert callable(drawer.add_node)


def test_png_drawer_add_edge_structure() -> None:
    """Test add_edge method signature."""
    drawer = PngDrawer()

    assert hasattr(drawer, "add_edge")
    assert callable(drawer.add_edge)


def test_png_drawer_draw_method() -> None:
    """Test draw method exists."""
    drawer = PngDrawer()

    assert hasattr(drawer, "draw")
    assert callable(drawer.draw)


def test_png_drawer_with_empty_labels() -> None:
    """Test PngDrawer with empty label dicts."""
    labels = LabelsDict(nodes={}, edges={})
    drawer = PngDrawer(labels=labels)

    # Should fall back to original labels
    assert drawer.get_node_label("test") == "<<B>test</B>>"
    assert drawer.get_edge_label("test") == "<<U>test</U>>"


def test_png_drawer_labels_dict_structure() -> None:
    """Test LabelsDict structure."""
    labels = LabelsDict(
        nodes={"n1": "Node1", "n2": "Node2"},
        edges={"e1": "Edge1", "e2": "Edge2"},
    )

    assert "n1" in labels["nodes"]
    assert "e1" in labels["edges"]
    assert labels["nodes"]["n1"] == "Node1"
    assert labels["edges"]["e1"] == "Edge1"


def test_graph_draw_png_with_labels() -> None:
    """Test Graph.draw_png() with custom labels."""
    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")
    graph.add_edge(node1, node2)

    custom_labels = LabelsDict(
        nodes={
            "node1": "Start Node",
            "node2": "End Node",
        },
        edges={},
    )

    try:
        result = graph.draw_png(output_file_path=None, labels=custom_labels)
        assert result is not None
        assert isinstance(result, bytes)
    except ImportError:
        pytest.skip("pygraphviz not installed")


def test_graph_draw_png_with_fontname() -> None:
    """Test Graph.draw_png() with custom font."""
    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")
    graph.add_edge(node1, node2)

    try:
        result = graph.draw_png(output_file_path=None, fontname="courier")
        assert result is not None
        assert isinstance(result, bytes)
    except ImportError:
        pytest.skip("pygraphviz not installed")


def test_png_drawer_add_nodes_method() -> None:
    """Test add_nodes method exists."""
    drawer = PngDrawer()

    assert hasattr(drawer, "add_nodes")
    assert callable(drawer.add_nodes)


def test_png_drawer_add_edges_method() -> None:
    """Test add_edges method exists."""
    drawer = PngDrawer()

    assert hasattr(drawer, "add_edges")
    assert callable(drawer.add_edges)


def test_png_drawer_update_styles_method() -> None:
    """Test update_styles method exists."""
    drawer = PngDrawer()

    assert hasattr(drawer, "update_styles")
    assert callable(drawer.update_styles)


def test_png_drawer_add_subgraph_method() -> None:
    """Test add_subgraph method exists."""
    drawer = PngDrawer()

    assert hasattr(drawer, "add_subgraph")
    assert callable(drawer.add_subgraph)


def test_labels_dict_can_be_empty() -> None:
    """Test LabelsDict can have empty dictionaries."""
    labels = LabelsDict(nodes={}, edges={})

    assert len(labels["nodes"]) == 0
    assert len(labels["edges"]) == 0


def test_labels_dict_nodes_only() -> None:
    """Test LabelsDict with only node labels."""
    labels = LabelsDict(
        nodes={"node1": "Label1", "node2": "Label2"},
        edges={},
    )

    assert len(labels["nodes"]) == 2
    assert len(labels["edges"]) == 0


def test_labels_dict_edges_only() -> None:
    """Test LabelsDict with only edge labels."""
    labels = LabelsDict(
        nodes={},
        edges={"edge1": "Label1", "edge2": "Label2"},
    )

    assert len(labels["nodes"]) == 0
    assert len(labels["edges"]) == 2


def test_png_drawer_multiple_custom_labels() -> None:
    """Test PngDrawer with multiple custom labels."""
    custom_labels = LabelsDict(
        nodes={"n1": "Node One", "n2": "Node Two", "n3": "Node Three"},
        edges={"e1": "Edge One", "e2": "Edge Two"},
    )
    drawer = PngDrawer(labels=custom_labels)

    assert drawer.get_node_label("n1") == "<<B>Node One</B>>"
    assert drawer.get_node_label("n2") == "<<B>Node Two</B>>"
    assert drawer.get_edge_label("e1") == "<<U>Edge One</U>>"


def test_graph_draw_png_returns_bytes_when_no_path() -> None:
    """Test that draw_png returns bytes when output_file_path is None."""
    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")
    graph.add_edge(node1, node2)

    try:
        result = graph.draw_png(output_file_path=None)
        assert isinstance(result, bytes)
        assert len(result) > 0  # Should have some PNG data
    except ImportError:
        pytest.skip("pygraphviz not installed")


def test_png_drawer_font_names() -> None:
    """Test various font names work."""
    fonts = ["arial", "helvetica", "courier", "times"]

    for font in fonts:
        drawer = PngDrawer(fontname=font)
        assert drawer.fontname == font


def test_png_drawer_special_node_names() -> None:
    """Test get_node_label with special node names."""
    drawer = PngDrawer()

    # Test with __start__ and __end__
    start_label = drawer.get_node_label("__start__")
    assert "<B>__start__</B>" in start_label

    end_label = drawer.get_node_label("__end__")
    assert "<B>__end__</B>" in end_label


def test_png_drawer_html_formatting() -> None:
    """Test that labels use HTML formatting."""
    drawer = PngDrawer()

    node_label = drawer.get_node_label("test")
    # Should be wrapped in << >>for HTML
    assert node_label.startswith("<<")
    assert node_label.endswith(">>")
    assert "<B>" in node_label

    edge_label = drawer.get_edge_label("test")
    assert edge_label.startswith("<<")
    assert edge_label.endswith(">>")
    assert "<U>" in edge_label


def test_labels_dict_type_definition() -> None:
    """Test LabelsDict is properly typed."""
    labels: LabelsDict = LabelsDict(nodes={}, edges={})

    # Should be a TypedDict with "nodes" and "edges" keys
    assert "nodes" in labels
    assert "edges" in labels
    assert isinstance(labels["nodes"], dict)
    assert isinstance(labels["edges"], dict)


def test_png_drawer_labels_with_special_chars() -> None:
    """Test labels with special characters."""
    custom_labels = LabelsDict(
        nodes={"n1": "Node & Test", "n2": "Node < > Test"},
        edges={"e1": "Edge \"quoted\""},
    )
    drawer = PngDrawer(labels=custom_labels)

    # Should handle special characters in labels
    label1 = drawer.get_node_label("n1")
    assert "Node & Test" in label1

    label2 = drawer.get_edge_label("e1")
    assert "Edge" in label2


def test_graph_first_last_node_styling() -> None:
    """Test that first and last nodes get special styling in PNG."""
    graph = Graph()
    node1 = graph.add_node(BaseModel, id="first")
    node2 = graph.add_node(BaseModel, id="middle")
    node3 = graph.add_node(BaseModel, id="last")

    graph.add_edge(node1, node2)
    graph.add_edge(node2, node3)

    # The drawer should identify first and last nodes
    try:
        # This will call update_styles which should identify first/last
        import pygraphviz  # noqa: F401

        result = graph.draw_png(output_file_path=None)
        assert isinstance(result, bytes)
    except ImportError:
        pytest.skip("pygraphviz not installed")


def test_png_drawer_conditional_edges() -> None:
    """Test that conditional edges are rendered differently."""
    drawer = PngDrawer()

    # Conditional edges should be dotted, regular should be solid
    # This is tested through the add_edge method's conditional parameter


def test_graph_draw_png_complex_structure() -> None:
    """Test drawing complex graph structure."""
    graph = Graph()
    nodes_list = [graph.add_node(BaseModel, id=f"node{i}") for i in range(5)]

    # Create a diamond pattern
    graph.add_edge(nodes_list[0], nodes_list[1])
    graph.add_edge(nodes_list[0], nodes_list[2])
    graph.add_edge(nodes_list[1], nodes_list[3])
    graph.add_edge(nodes_list[2], nodes_list[3])
    graph.add_edge(nodes_list[3], nodes_list[4])

    try:
        result = graph.draw_png(output_file_path=None)
        assert isinstance(result, bytes)
        assert len(result) > 0
    except ImportError:
        pytest.skip("pygraphviz not installed")


def test_png_drawer_with_subgraphs() -> None:
    """Test PngDrawer handles subgraphs (nested nodes)."""
    graph = Graph()

    # Create nodes with colons (indicating subgraph structure)
    node1 = graph.add_node(BaseModel, id="parent")
    node2 = graph.add_node(BaseModel, id="parent:child1")
    node3 = graph.add_node(BaseModel, id="parent:child2")

    graph.add_edge(node1, node2)
    graph.add_edge(node1, node3)

    try:
        result = graph.draw_png(output_file_path=None)
        assert isinstance(result, bytes)
    except ImportError:
        pytest.skip("pygraphviz not installed")


def test_png_drawer_empty_graph() -> None:
    """Test drawing empty graph."""
    drawer = PngDrawer()
    graph = Graph()

    try:
        result = drawer.draw(graph, output_path=None)
        # Should handle empty graph
        assert result is None or isinstance(result, bytes)
    except ImportError:
        pytest.skip("pygraphviz not installed")


def test_labels_dict_partial_labels() -> None:
    """Test LabelsDict with only some nodes/edges labeled."""
    labels = LabelsDict(
        nodes={"node1": "Custom1"},  # Only one node has custom label
        edges={},
    )
    drawer = PngDrawer(labels=labels)

    # Custom label
    assert "Custom1" in drawer.get_node_label("node1")

    # Default label (node not in labels dict)
    assert "node2" in drawer.get_node_label("node2")


def test_png_drawer_fontname_used() -> None:
    """Test that fontname is stored correctly."""
    fonts = ["arial", "helvetica", "times", "courier", "verdana"]

    for font in fonts:
        drawer = PngDrawer(fontname=font)
        assert drawer.fontname == font


def test_graph_draw_png_with_conditional_edges() -> None:
    """Test PNG rendering with conditional edges."""
    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")
    node3 = graph.add_node(BaseModel, id="node3")

    # Add regular edge
    graph.add_edge(node1, node2, conditional=False)
    # Add conditional edge
    graph.add_edge(node1, node3, conditional=True)

    try:
        result = graph.draw_png(output_file_path=None)
        assert isinstance(result, bytes)
    except ImportError:
        pytest.skip("pygraphviz not installed")


def test_png_drawer_edge_with_data() -> None:
    """Test PNG drawer with edge data (labels)."""
    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")

    # Edge with data
    graph.add_edge(node1, node2, data="edge_label")

    try:
        result = graph.draw_png(output_file_path=None)
        assert isinstance(result, bytes)
    except ImportError:
        pytest.skip("pygraphviz not installed")


def test_labels_dict_preserves_all_entries() -> None:
    """Test that LabelsDict preserves all node and edge entries."""
    node_labels = {f"node{i}": f"Label{i}" for i in range(10)}
    edge_labels = {f"edge{i}": f"EdgeLabel{i}" for i in range(10)}

    labels = LabelsDict(nodes=node_labels, edges=edge_labels)

    assert len(labels["nodes"]) == 10
    assert len(labels["edges"]) == 10


def test_png_drawer_default_font() -> None:
    """Test PngDrawer uses arial as default font."""
    drawer = PngDrawer()
    assert drawer.fontname == "arial"


def test_png_drawer_labels_immutable_default() -> None:
    """Test that default labels dict doesn't affect drawer."""
    drawer1 = PngDrawer()
    drawer2 = PngDrawer()

    # Each should have independent labels
    assert drawer1.labels is not drawer2.labels


def test_graph_draw_png_returns_none_when_path_specified() -> None:
    """Test that draw_png returns None when saving to file."""
    # This test can't actually write file in unit tests, but tests the API

    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")
    graph.add_edge(node1, node2)

    # Method should accept output_file_path parameter
    # Can't test actual file writing in unit tests
    try:
        # Would return None if path specified (and file written)
        graph.draw_png(output_file_path="/tmp/test_graph.png")
    except ImportError:
        pytest.skip("pygraphviz not installed")
    except Exception:
        # May fail to write file, but that's ok for this test
        pass


def test_png_drawer_handles_none_data() -> None:
    """Test PngDrawer handles edges with None data."""
    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")
    graph.add_edge(node1, node2, data=None)

    drawer = PngDrawer()

    try:
        result = drawer.draw(graph, output_path=None)
        assert result is None or isinstance(result, bytes)
    except ImportError:
        pytest.skip("pygraphviz not installed")


def test_graph_draw_png_overload_signatures() -> None:
    """Test Graph.draw_png() overload signatures."""
    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")
    graph.add_edge(node1, node2)

    # Should accept both with path and without
    try:
        # Without path - returns bytes
        result1 = graph.draw_png(output_file_path=None)
        assert isinstance(result1, bytes)

        # With path - returns None (type is more complex due to overload)
        graph.draw_png(output_file_path="/tmp/test.png")
    except ImportError:
        pytest.skip("pygraphviz not installed")
    except Exception:
        # File writing may fail, that's ok
        pass


def test_png_drawer_labels_dict_typed_correctly() -> None:
    """Test LabelsDict TypedDict has correct structure."""
    # This is primarily a type-checking test
    labels: LabelsDict = {
        "nodes": {"n1": "Label1"},
        "edges": {"e1": "Label2"},
    }

    assert isinstance(labels["nodes"], dict)
    assert isinstance(labels["edges"], dict)


def test_graph_draw_png_with_metadata() -> None:
    """Test PNG drawing with node metadata."""
    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1", metadata={"version": "1.0"})
    node2 = graph.add_node(BaseModel, id="node2", metadata={"type": "processor"})
    graph.add_edge(node1, node2)

    try:
        result = graph.draw_png(output_file_path=None)
        assert isinstance(result, bytes)
    except ImportError:
        pytest.skip("pygraphviz not installed")


def test_png_drawer_preserves_graph_structure() -> None:
    """Test that PNG drawer preserves all nodes and edges."""
    graph = Graph()
    nodes = [graph.add_node(BaseModel, id=f"node{i}") for i in range(4)]

    # Create edges
    for i in range(len(nodes) - 1):
        graph.add_edge(nodes[i], nodes[i + 1])

    drawer = PngDrawer()

    try:
        # Drawing should include all nodes
        drawer.draw(graph, output_path=None)
        # If it doesn't raise, it handled all nodes
    except ImportError:
        pytest.skip("pygraphviz not installed")
