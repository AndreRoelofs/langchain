"""Unit tests for StdOutCallbackHandler."""

from io import StringIO
from unittest.mock import patch
from uuid import uuid4

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.stdout import StdOutCallbackHandler


@patch("sys.stdout", new_callable=StringIO)
def test_stdout_callback_handler_chain_start(mock_stdout: StringIO) -> None:
    """Test chain start output."""
    handler = StdOutCallbackHandler()

    handler.on_chain_start(
        serialized={"name": "TestChain"},
        inputs={"input": "test"},
        run_id=uuid4(),
    )

    output = mock_stdout.getvalue()
    assert "Entering new TestChain chain" in output


@patch("sys.stdout", new_callable=StringIO)
def test_stdout_callback_handler_chain_start_with_name_kwarg(
    mock_stdout: StringIO,
) -> None:
    """Test chain start with name in kwargs."""
    handler = StdOutCallbackHandler()

    handler.on_chain_start(
        serialized={},
        inputs={},
        run_id=uuid4(),
        name="CustomName",
    )

    output = mock_stdout.getvalue()
    assert "Entering new CustomName chain" in output


@patch("sys.stdout", new_callable=StringIO)
def test_stdout_callback_handler_chain_start_with_id(mock_stdout: StringIO) -> None:
    """Test chain start falls back to serialized id."""
    handler = StdOutCallbackHandler()

    handler.on_chain_start(
        serialized={"id": ["module", "ClassName"]},
        inputs={},
        run_id=uuid4(),
    )

    output = mock_stdout.getvalue()
    assert "Entering new ClassName chain" in output


@patch("sys.stdout", new_callable=StringIO)
def test_stdout_callback_handler_chain_start_unknown(mock_stdout: StringIO) -> None:
    """Test chain start with no name available."""
    handler = StdOutCallbackHandler()

    handler.on_chain_start(
        serialized={},
        inputs={},
        run_id=uuid4(),
    )

    output = mock_stdout.getvalue()
    assert "Entering new <unknown> chain" in output


@patch("sys.stdout", new_callable=StringIO)
def test_stdout_callback_handler_chain_end(mock_stdout: StringIO) -> None:
    """Test chain end output."""
    handler = StdOutCallbackHandler()

    handler.on_chain_end(
        outputs={"output": "result"},
        run_id=uuid4(),
    )

    output = mock_stdout.getvalue()
    assert "Finished chain" in output


@patch("sys.stdout", new_callable=StringIO)
def test_stdout_callback_handler_agent_action(mock_stdout: StringIO) -> None:
    """Test agent action output."""
    handler = StdOutCallbackHandler()

    action = AgentAction(
        tool="TestTool",
        tool_input={"query": "test"},
        log="Invoking TestTool with query",
    )

    handler.on_agent_action(action, run_id=uuid4())

    output = mock_stdout.getvalue()
    assert "Invoking TestTool with query" in output


@patch("sys.stdout", new_callable=StringIO)
def test_stdout_callback_handler_agent_action_with_color(
    mock_stdout: StringIO,
) -> None:
    """Test agent action with color override."""
    handler = StdOutCallbackHandler(color="green")

    action = AgentAction(
        tool="TestTool",
        tool_input={"query": "test"},
        log="Action log",
    )

    handler.on_agent_action(action, color="red", run_id=uuid4())

    output = mock_stdout.getvalue()
    assert "Action log" in output


@patch("sys.stdout", new_callable=StringIO)
def test_stdout_callback_handler_tool_end(mock_stdout: StringIO) -> None:
    """Test tool end output."""
    handler = StdOutCallbackHandler()

    handler.on_tool_end(
        output="Tool result",
        run_id=uuid4(),
    )

    output = mock_stdout.getvalue()
    assert "Tool result" in output


@patch("sys.stdout", new_callable=StringIO)
def test_stdout_callback_handler_tool_end_with_prefixes(
    mock_stdout: StringIO,
) -> None:
    """Test tool end with observation and llm prefixes."""
    handler = StdOutCallbackHandler()

    handler.on_tool_end(
        output="Tool result",
        observation_prefix="Observation:",
        llm_prefix="Thought:",
        run_id=uuid4(),
    )

    output = mock_stdout.getvalue()
    assert "Observation:" in output
    assert "Tool result" in output
    assert "Thought:" in output


@patch("sys.stdout", new_callable=StringIO)
def test_stdout_callback_handler_on_text(mock_stdout: StringIO) -> None:
    """Test arbitrary text output."""
    handler = StdOutCallbackHandler()

    handler.on_text(
        text="Custom text",
        run_id=uuid4(),
    )

    output = mock_stdout.getvalue()
    assert "Custom text" in output


@patch("sys.stdout", new_callable=StringIO)
def test_stdout_callback_handler_on_text_with_end(mock_stdout: StringIO) -> None:
    """Test text output with custom end character."""
    handler = StdOutCallbackHandler()

    handler.on_text(
        text="Line 1",
        end="\n",
        run_id=uuid4(),
    )
    handler.on_text(
        text="Line 2",
        end="",
        run_id=uuid4(),
    )

    output = mock_stdout.getvalue()
    assert "Line 1\n" in output
    assert "Line 2" in output


@patch("sys.stdout", new_callable=StringIO)
def test_stdout_callback_handler_agent_finish(mock_stdout: StringIO) -> None:
    """Test agent finish output."""
    handler = StdOutCallbackHandler()

    finish = AgentFinish(
        return_values={"output": "final result"},
        log="Agent completed successfully",
    )

    handler.on_agent_finish(finish, run_id=uuid4())

    output = mock_stdout.getvalue()
    assert "Agent completed successfully" in output


@patch("sys.stdout", new_callable=StringIO)
def test_stdout_callback_handler_with_default_color(mock_stdout: StringIO) -> None:
    """Test handler with default color."""
    handler = StdOutCallbackHandler(color="blue")
    assert handler.color == "blue"

    handler.on_text("Colored text", run_id=uuid4())

    output = mock_stdout.getvalue()
    assert "Colored text" in output


@patch("sys.stdout", new_callable=StringIO)
def test_stdout_callback_handler_no_color(mock_stdout: StringIO) -> None:
    """Test handler without color."""
    handler = StdOutCallbackHandler()
    assert handler.color is None

    handler.on_text("Plain text", run_id=uuid4())

    output = mock_stdout.getvalue()
    assert "Plain text" in output


@patch("sys.stdout", new_callable=StringIO)
def test_stdout_callback_handler_color_override(mock_stdout: StringIO) -> None:
    """Test color override in method calls."""
    handler = StdOutCallbackHandler(color="green")

    # Test tool_end color override
    handler.on_tool_end(
        output="Result",
        color="red",
        run_id=uuid4(),
    )

    output = mock_stdout.getvalue()
    assert "Result" in output
