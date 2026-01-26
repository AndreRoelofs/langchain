"""Unit tests for tools rendering functionality."""

from inspect import signature

import pytest
from pydantic import BaseModel
from typing_extensions import override

from langchain_core.tools import BaseTool, render_text_description, render_text_description_and_args


class MockSchema(BaseModel):
    """Mock schema for testing."""

    arg1: int
    arg2: str


class MockToolWithFunc(BaseTool):
    """Mock tool with a func attribute for testing."""

    name: str = "mock_tool_with_func"
    description: str = "A mock tool with func"
    args_schema: type[BaseModel] = MockSchema

    def dummy_func(self, arg1: int, arg2: str) -> str:
        """Dummy function."""
        return f"{arg1} {arg2}"

    @override
    def _run(self, arg1: int, arg2: str) -> str:
        return self.dummy_func(arg1, arg2)

    @property
    def func(self) -> callable:  # noqa: A003
        """Return the dummy function."""
        return self.dummy_func


class MockToolWithoutFunc(BaseTool):
    """Mock tool without a func attribute."""

    name: str = "mock_tool_without_func"
    description: str = "A mock tool without func"
    args_schema: type[BaseModel] = MockSchema

    @override
    def _run(self, arg1: int, arg2: str) -> str:
        return f"{arg1} {arg2}"


def test_render_text_description_with_func() -> None:
    """Test rendering tool with func attribute shows signature."""
    tool_with_func = MockToolWithFunc()

    result = render_text_description([tool_with_func])

    sig = signature(tool_with_func.func)
    expected = f"mock_tool_with_func{sig} - A mock tool with func"
    assert result == expected


def test_render_text_description_without_func() -> None:
    """Test rendering tool without func attribute shows name and description."""
    tool_without_func = MockToolWithoutFunc()

    result = render_text_description([tool_without_func])

    expected = "mock_tool_without_func - A mock tool without func"
    assert result == expected


def test_render_text_description_multiple_tools() -> None:
    """Test rendering multiple tools with newline separator."""
    tool1 = MockToolWithoutFunc()
    tool2 = MockToolWithFunc()

    result = render_text_description([tool1, tool2])

    lines = result.split("\n")
    assert len(lines) == 2
    assert "mock_tool_without_func" in lines[0]
    assert "mock_tool_with_func" in lines[1]


def test_render_text_description_empty_list() -> None:
    """Test rendering empty tool list returns empty string."""
    result = render_text_description([])
    assert result == ""


def test_render_text_description_and_args_with_func() -> None:
    """Test rendering tool with func attribute and args schema."""
    tool_with_func = MockToolWithFunc()

    result = render_text_description_and_args([tool_with_func])

    sig = signature(tool_with_func.func)
    assert f"mock_tool_with_func{sig} - A mock tool with func" in result
    assert "args:" in result
    assert "'arg1'" in result
    assert "'arg2'" in result


def test_render_text_description_and_args_without_func() -> None:
    """Test rendering tool without func attribute and args schema."""
    tool_without_func = MockToolWithoutFunc()

    result = render_text_description_and_args([tool_without_func])

    assert "mock_tool_without_func - A mock tool without func" in result
    assert "args:" in result
    assert "'arg1'" in result
    assert "'arg2'" in result


def test_render_text_description_and_args_multiple_tools() -> None:
    """Test rendering multiple tools with args."""
    tool1 = MockToolWithoutFunc()
    tool2 = MockToolWithFunc()

    result = render_text_description_and_args([tool1, tool2])

    lines = result.split("\n")
    assert len(lines) == 2
    assert "mock_tool_without_func" in lines[0]
    assert "mock_tool_with_func" in lines[1]
    assert all("args:" in line for line in lines)


def test_render_text_description_and_args_empty_list() -> None:
    """Test rendering empty tool list with args returns empty string."""
    result = render_text_description_and_args([])
    assert result == ""


def test_render_text_description_and_args_shows_all_properties() -> None:
    """Test that args rendering includes all schema properties."""
    tool = MockToolWithoutFunc()

    result = render_text_description_and_args([tool])

    # Check that both args are present in the output
    args_str = str(tool.args)
    assert args_str in result


def test_render_functions_preserve_order() -> None:
    """Test that rendering preserves the order of tools."""
    tools = [
        MockToolWithoutFunc(name=f"tool_{i}", description=f"Tool {i}")
        for i in range(5)
    ]

    result_desc = render_text_description(tools)
    result_args = render_text_description_and_args(tools)

    lines_desc = result_desc.split("\n")
    lines_args = result_args.split("\n")

    assert len(lines_desc) == 5
    assert len(lines_args) == 5

    for i, (line_desc, line_args) in enumerate(zip(lines_desc, lines_args)):
        assert f"tool_{i}" in line_desc
        assert f"tool_{i}" in line_args
        assert f"Tool {i}" in line_desc
        assert f"Tool {i}" in line_args
