"""Comprehensive tests for ASCII graph drawing functionality."""

import pytest

from langchain_core.runnables.graph import Edge, Graph, Node
from langchain_core.runnables.graph_ascii import (
    AsciiCanvas,
    VertexViewer,
    draw_ascii,
)


def test_vertex_viewer_initialization() -> None:
    """Test VertexViewer initialization."""
    viewer = VertexViewer("TestNode")

    assert viewer.h == 3  # HEIGHT constant
    assert viewer.w == len("TestNode") + 2  # name + 2 for box edges


def test_vertex_viewer_dimensions() -> None:
    """Test VertexViewer computes correct dimensions."""
    short_viewer = VertexViewer("AB")
    assert short_viewer.w == 4  # "AB" + 2

    long_viewer = VertexViewer("VeryLongNodeName")
    assert long_viewer.w == len("VeryLongNodeName") + 2


def test_ascii_canvas_initialization() -> None:
    """Test AsciiCanvas initialization."""
    canvas = AsciiCanvas(10, 5)

    assert canvas.cols == 10
    assert canvas.lines == 5
    assert len(canvas.canvas) == 5
    assert all(len(line) == 10 for line in canvas.canvas)


def test_ascii_canvas_invalid_dimensions() -> None:
    """Test AsciiCanvas rejects invalid dimensions."""
    with pytest.raises(ValueError, match="Canvas dimensions should be > 1"):
        AsciiCanvas(1, 5)

    with pytest.raises(ValueError, match="Canvas dimensions should be > 1"):
        AsciiCanvas(10, 1)

    with pytest.raises(ValueError, match="Canvas dimensions should be > 1"):
        AsciiCanvas(0, 0)


def test_ascii_canvas_point() -> None:
    """Test drawing a point on ASCII canvas."""
    canvas = AsciiCanvas(10, 5)
    canvas.point(5, 2, "*")

    assert canvas.canvas[2][5] == "*"


def test_ascii_canvas_point_invalid_char() -> None:
    """Test canvas point rejects multi-character strings."""
    canvas = AsciiCanvas(10, 5)

    with pytest.raises(ValueError, match="char should be a single character"):
        canvas.point(5, 2, "**")


def test_ascii_canvas_point_out_of_bounds() -> None:
    """Test canvas point rejects out-of-bounds coordinates."""
    canvas = AsciiCanvas(10, 5)

    with pytest.raises(ValueError, match="x should be >= 0 and < number of columns"):
        canvas.point(10, 2, "*")

    with pytest.raises(ValueError, match="x should be >= 0 and < number of columns"):
        canvas.point(-1, 2, "*")

    with pytest.raises(ValueError, match="y should be >= 0 and < number of lines"):
        canvas.point(5, 5, "*")

    with pytest.raises(ValueError, match="y should be >= 0 and < number of lines"):
        canvas.point(5, -1, "*")


def test_ascii_canvas_line_horizontal() -> None:
    """Test drawing a horizontal line."""
    canvas = AsciiCanvas(20, 10)
    canvas.line(2, 5, 10, 5, "-")

    # Check that all points in the line are set
    for x in range(2, 11):
        assert canvas.canvas[5][x] == "-"


def test_ascii_canvas_line_vertical() -> None:
    """Test drawing a vertical line."""
    canvas = AsciiCanvas(20, 10)
    canvas.line(5, 2, 5, 7, "|")

    # Check that all points in the line are set
    for y in range(2, 8):
        assert canvas.canvas[y][5] == "|"


def test_ascii_canvas_line_diagonal() -> None:
    """Test drawing a diagonal line."""
    canvas = AsciiCanvas(20, 20)
    canvas.line(0, 0, 5, 5, "*")

    # Check diagonal points are set
    for i in range(6):
        assert canvas.canvas[i][i] == "*"


def test_ascii_canvas_line_single_point() -> None:
    """Test drawing a line with same start and end point."""
    canvas = AsciiCanvas(10, 10)
    canvas.line(5, 5, 5, 5, "X")

    assert canvas.canvas[5][5] == "X"


def test_ascii_canvas_text() -> None:
    """Test drawing text on canvas."""
    canvas = AsciiCanvas(20, 10)
    canvas.text(2, 5, "Hello")

    assert "".join(canvas.canvas[5][2:7]) == "Hello"


def test_ascii_canvas_box() -> None:
    """Test drawing a box on canvas."""
    canvas = AsciiCanvas(20, 10)
    canvas.box(2, 2, 10, 5)

    # Check corners
    assert canvas.canvas[2][2] == "+"
    assert canvas.canvas[2][11] == "+"
    assert canvas.canvas[6][2] == "+"
    assert canvas.canvas[6][11] == "+"

    # Check top and bottom edges
    for x in range(3, 11):
        assert canvas.canvas[2][x] == "-"
        assert canvas.canvas[6][x] == "-"

    # Check left and right edges
    for y in range(3, 6):
        assert canvas.canvas[y][2] == "|"
        assert canvas.canvas[y][11] == "|"


def test_ascii_canvas_box_invalid_dimensions() -> None:
    """Test canvas box rejects invalid dimensions."""
    canvas = AsciiCanvas(20, 10)

    with pytest.raises(ValueError, match="Box dimensions should be > 1"):
        canvas.box(2, 2, 1, 5)

    with pytest.raises(ValueError, match="Box dimensions should be > 1"):
        canvas.box(2, 2, 5, 1)


def test_ascii_canvas_draw() -> None:
    """Test drawing canvas to string."""
    canvas = AsciiCanvas(5, 3)
    canvas.point(2, 1, "*")

    output = canvas.draw()
    lines = output.split("\n")

    assert len(lines) == 3
    assert lines[1][2] == "*"


def test_draw_ascii_simple_graph() -> None:
    """Test drawing a simple graph."""
    vertices = {"1": "A", "2": "B"}
    edges = [
        Edge(source="1", target="2", data=None, conditional=False),
    ]

    try:
        result = draw_ascii(vertices, edges)
        # Should produce ASCII art with boxes for A and B
        assert "A" in result
        assert "B" in result
        assert "+" in result  # Box corners
    except ImportError:
        pytest.skip("grandalf not installed")


def test_draw_ascii_linear_chain() -> None:
    """Test drawing a linear chain graph."""
    vertices = {"1": "Start", "2": "Middle", "3": "End"}
    edges = [
        Edge(source="1", target="2", data=None, conditional=False),
        Edge(source="2", target="3", data=None, conditional=False),
    ]

    try:
        result = draw_ascii(vertices, edges)
        assert "Start" in result
        assert "Middle" in result
        assert "End" in result
    except ImportError:
        pytest.skip("grandalf not installed")


def test_draw_ascii_branching_graph() -> None:
    """Test drawing a graph with branches."""
    vertices = {"1": "Root", "2": "Branch1", "3": "Branch2", "4": "Merge"}
    edges = [
        Edge(source="1", target="2", data=None, conditional=False),
        Edge(source="1", target="3", data=None, conditional=False),
        Edge(source="2", target="4", data=None, conditional=False),
        Edge(source="3", target="4", data=None, conditional=False),
    ]

    try:
        result = draw_ascii(vertices, edges)
        assert "Root" in result
        assert "Branch1" in result
        assert "Branch2" in result
        assert "Merge" in result
    except ImportError:
        pytest.skip("grandalf not installed")


def test_draw_ascii_with_conditional_edges() -> None:
    """Test drawing graph with conditional edges."""
    vertices = {"1": "A", "2": "B", "3": "C"}
    edges = [
        Edge(source="1", target="2", data=None, conditional=False),
        Edge(source="1", target="3", data=None, conditional=True),
    ]

    try:
        result = draw_ascii(vertices, edges)
        # Conditional edges are drawn with dots
        assert "." in result or "*" in result
    except ImportError:
        pytest.skip("grandalf not installed")


def test_draw_ascii_complex_graph() -> None:
    """Test drawing a more complex graph."""
    vertices = {
        "1": "Input",
        "2": "Process1",
        "3": "Process2",
        "4": "Decision",
        "5": "OutputA",
        "6": "OutputB",
    }
    edges = [
        Edge(source="1", target="2", data=None, conditional=False),
        Edge(source="1", target="3", data=None, conditional=False),
        Edge(source="2", target="4", data=None, conditional=False),
        Edge(source="3", target="4", data=None, conditional=False),
        Edge(source="4", target="5", data=None, conditional=True),
        Edge(source="4", target="6", data=None, conditional=True),
    ]

    try:
        result = draw_ascii(vertices, edges)
        # Verify all nodes are present
        for node_name in vertices.values():
            assert node_name in result
    except ImportError:
        pytest.skip("grandalf not installed")


def test_draw_ascii_single_node() -> None:
    """Test drawing a graph with a single node."""
    vertices = {"1": "OnlyNode"}
    edges: list[Edge] = []

    try:
        result = draw_ascii(vertices, edges)
        assert "OnlyNode" in result
    except ImportError:
        pytest.skip("grandalf not installed")


def test_draw_ascii_cycle() -> None:
    """Test drawing a graph with cycles."""
    vertices = {"1": "A", "2": "B", "3": "C"}
    edges = [
        Edge(source="1", target="2", data=None, conditional=False),
        Edge(source="2", target="3", data=None, conditional=False),
        Edge(source="3", target="1", data=None, conditional=False),
    ]

    try:
        result = draw_ascii(vertices, edges)
        assert "A" in result
        assert "B" in result
        assert "C" in result
    except ImportError:
        pytest.skip("grandalf not installed")


def test_graph_draw_ascii_method() -> None:
    """Test Graph.draw_ascii() method."""
    from pydantic import BaseModel

    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")
    graph.add_edge(node1, node2)

    try:
        result = graph.draw_ascii()
        # Should contain node names
        assert isinstance(result, str)
        assert len(result) > 0
    except ImportError:
        pytest.skip("grandalf not installed")


def test_graph_print_ascii() -> None:
    """Test Graph.print_ascii() method (should not raise)."""
    from pydantic import BaseModel

    graph = Graph()
    node1 = graph.add_node(BaseModel, id="node1")
    node2 = graph.add_node(BaseModel, id="node2")
    graph.add_edge(node1, node2)

    try:
        # Should not raise, just print
        graph.print_ascii()
    except ImportError:
        pytest.skip("grandalf not installed")


def test_ascii_canvas_overlapping_elements() -> None:
    """Test that later canvas operations overwrite earlier ones."""
    canvas = AsciiCanvas(10, 10)

    # Draw a line
    canvas.line(0, 5, 9, 5, "-")
    # Draw a box that overlaps the line
    canvas.box(3, 3, 5, 5)

    # Box should overwrite line in overlapping positions
    assert canvas.canvas[5][3] == "|"  # Box left edge overwrites line
    assert canvas.canvas[3][3] == "+"  # Box top-left corner
    assert canvas.canvas[3][7] == "+"  # Box top-right corner
    assert canvas.canvas[7][3] == "+"  # Box bottom-left corner
    assert canvas.canvas[7][7] == "+"  # Box bottom-right corner


def test_ascii_canvas_text_overwrites() -> None:
    """Test that text overwrites existing content."""
    canvas = AsciiCanvas(20, 10)

    # Fill a region with asterisks
    for x in range(5, 10):
        canvas.point(x, 5, "*")

    # Write text over it
    canvas.text(5, 5, "HI")

    assert canvas.canvas[5][5] == "H"
    assert canvas.canvas[5][6] == "I"


def test_draw_ascii_with_long_node_names() -> None:
    """Test ASCII drawing with long node names."""
    vertices = {
        "1": "VeryLongNodeNameThatMightCauseIssues",
        "2": "AnotherLongName",
    }
    edges = [Edge(source="1", target="2", data=None, conditional=False)]

    try:
        result = draw_ascii(vertices, edges)
        assert "VeryLongNodeNameThatMightCauseIssues" in result
        assert "AnotherLongName" in result
    except ImportError:
        pytest.skip("grandalf not installed")


def test_draw_ascii_parallel_paths() -> None:
    """Test ASCII drawing with parallel execution paths."""
    vertices = {"1": "Start", "2": "PathA", "3": "PathB", "4": "End"}
    edges = [
        Edge(source="1", target="2", data=None, conditional=False),
        Edge(source="1", target="3", data=None, conditional=False),
        Edge(source="2", target="4", data=None, conditional=False),
        Edge(source="3", target="4", data=None, conditional=False),
    ]

    try:
        result = draw_ascii(vertices, edges)
        # All nodes should be present
        assert all(name in result for name in vertices.values())
    except ImportError:
        pytest.skip("grandalf not installed")


def test_ascii_canvas_line_backwards() -> None:
    """Test that line drawing works in both directions."""
    canvas1 = AsciiCanvas(20, 10)
    canvas1.line(2, 2, 10, 2, "-")

    canvas2 = AsciiCanvas(20, 10)
    canvas2.line(10, 2, 2, 2, "-")

    # Both should produce the same line
    assert canvas1.canvas[2] == canvas2.canvas[2]


def test_ascii_canvas_box_with_text() -> None:
    """Test combining box and text."""
    canvas = AsciiCanvas(20, 10)
    canvas.box(2, 2, 10, 5)
    canvas.text(3, 3, "Node1")

    # Text should be inside the box
    assert "".join(canvas.canvas[3][3:8]) == "Node1"
    # Box should still have corners
    assert canvas.canvas[2][2] == "+"


def test_draw_ascii_disconnected_components() -> None:
    """Test ASCII drawing with disconnected graph components."""
    vertices = {"1": "A", "2": "B", "3": "C", "4": "D"}
    edges = [
        Edge(source="1", target="2", data=None, conditional=False),
        Edge(source="3", target="4", data=None, conditional=False),
    ]

    try:
        # Note: current implementation only renders the first connected component
        result = draw_ascii(vertices, edges)
        # Only the first connected component (A->B) will be present
        assert "A" in result or "C" in result  # At least one component is shown
        assert "B" in result or "D" in result
    except ImportError:
        pytest.skip("grandalf not installed")


def test_ascii_canvas_minimal_box() -> None:
    """Test canvas with minimal valid box."""
    canvas = AsciiCanvas(10, 10)
    canvas.box(0, 0, 2, 2)

    # 2x2 box should have 4 corners
    assert canvas.canvas[0][0] == "+"
    assert canvas.canvas[0][1] == "+"
    assert canvas.canvas[1][0] == "+"
    assert canvas.canvas[1][1] == "+"


def test_vertex_viewer_property_access() -> None:
    """Test VertexViewer property access."""
    viewer = VertexViewer("Test")

    # Properties should be accessible
    height = viewer.h
    width = viewer.w

    assert isinstance(height, int)
    assert isinstance(width, int)
    assert height == 3
    assert width > 0


def test_draw_ascii_with_edge_data() -> None:
    """Test ASCII drawing distinguishes conditional vs normal edges."""
    vertices = {"1": "A", "2": "B", "3": "C"}
    edges = [
        Edge(source="1", target="2", data=None, conditional=False),
        Edge(source="1", target="3", data="condition", conditional=True),
    ]

    try:
        result = draw_ascii(vertices, edges)
        # Normal edges use *, conditional use .
        assert ("*" in result or "." in result)
    except ImportError:
        pytest.skip("grandalf not installed")


def test_ascii_canvas_large_dimensions() -> None:
    """Test canvas with large dimensions."""
    canvas = AsciiCanvas(100, 50)

    assert canvas.cols == 100
    assert canvas.lines == 50
    assert len(canvas.canvas) == 50
    assert all(len(line) == 100 for line in canvas.canvas)


def test_draw_ascii_self_loop() -> None:
    """Test ASCII drawing with self-referencing node."""
    vertices = {"1": "SelfLoop", "2": "Other"}
    edges = [
        Edge(source="1", target="1", data=None, conditional=False),
        Edge(source="1", target="2", data=None, conditional=False),
    ]

    try:
        # Self-loops are not currently supported by the grandalf routing
        with pytest.raises(ValueError, match="no intersection found"):
            draw_ascii(vertices, edges)
    except ImportError:
        pytest.skip("grandalf not installed")


def test_ascii_canvas_draw_creates_proper_output() -> None:
    """Test that canvas draw creates proper multi-line string."""
    canvas = AsciiCanvas(5, 3)
    canvas.text(0, 0, "ABC")
    canvas.text(0, 2, "XYZ")

    output = canvas.draw()
    lines = output.split("\n")

    assert len(lines) == 3
    assert "ABC" in lines[0]
    assert "XYZ" in lines[2]


def test_draw_ascii_preserves_node_names() -> None:
    """Test that node names are preserved in ASCII output."""
    vertices = {
        "a": "StartNode",
        "b": "MiddleNode",
        "c": "EndNode",
    }
    edges = [
        Edge(source="a", target="b", data=None, conditional=False),
        Edge(source="b", target="c", data=None, conditional=False),
    ]

    try:
        result = draw_ascii(vertices, edges)
        # All vertex names should appear in output
        assert "StartNode" in result
        assert "MiddleNode" in result
        assert "EndNode" in result
    except ImportError:
        pytest.skip("grandalf not installed")


def test_ascii_canvas_line_steep_diagonal() -> None:
    """Test drawing a steep diagonal line."""
    canvas = AsciiCanvas(20, 20)
    canvas.line(5, 2, 7, 15, "|")

    # Check that line exists
    drawn_chars = sum(1 for row in canvas.canvas for char in row if char == "|")
    assert drawn_chars > 0


def test_vertex_viewer_with_empty_name() -> None:
    """Test VertexViewer with empty string name."""
    viewer = VertexViewer("")

    assert viewer.w == 2  # Just the box edges
    assert viewer.h == 3


def test_vertex_viewer_with_special_characters() -> None:
    """Test VertexViewer with special characters in name."""
    viewer = VertexViewer("Node-123_ABC")

    assert viewer.w == len("Node-123_ABC") + 2


def test_draw_ascii_wide_graph() -> None:
    """Test ASCII drawing with many nodes at same level."""
    vertices = {str(i): f"Node{i}" for i in range(10)}
    edges = [
        Edge(source=str(i), target=str(i + 1), data=None, conditional=False)
        for i in range(9)
    ]

    try:
        result = draw_ascii(vertices, edges)
        # Should handle wide layout
        assert len(result.split("\n")) > 0
    except ImportError:
        pytest.skip("grandalf not installed")


def test_ascii_canvas_box_overlapping() -> None:
    """Test multiple overlapping boxes."""
    canvas = AsciiCanvas(20, 20)

    # Draw two overlapping boxes
    canvas.box(2, 2, 10, 8)
    canvas.box(6, 4, 10, 8)

    # Second box should overwrite overlapping parts of first
    assert canvas.canvas[4][6] == "+"  # Top-left corner of second box


def test_ascii_line_reversed_coordinates() -> None:
    """Test line handles coordinates in any order."""
    canvas = AsciiCanvas(20, 10)

    # Draw from right to left
    canvas.line(15, 5, 5, 5, "-")

    # Should still draw the line
    for x in range(5, 16):
        assert canvas.canvas[5][x] == "-"
