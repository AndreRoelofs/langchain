"""Unit tests for FileCallbackHandler."""

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.file import FileCallbackHandler


def test_file_callback_handler_context_manager() -> None:
    """Test FileCallbackHandler as a context manager."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        with FileCallbackHandler(temp_path, mode="w") as handler:
            handler.on_text("Hello, world!", run_id=uuid4())

        # File should be closed after context manager exits
        assert handler.file.closed

        # Verify content was written
        with Path(temp_path).open("r") as f:
            content = f.read()
            assert "Hello, world!" in content
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_file_callback_handler_write_chain_start() -> None:
    """Test writing chain start events."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        with FileCallbackHandler(temp_path, mode="w") as handler:
            handler.on_chain_start(
                serialized={"name": "TestChain"},
                inputs={"input": "test"},
                run_id=uuid4(),
            )

        with Path(temp_path).open("r") as f:
            content = f.read()
            assert "Entering new TestChain chain" in content
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_file_callback_handler_write_chain_end() -> None:
    """Test writing chain end events."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        with FileCallbackHandler(temp_path, mode="w") as handler:
            handler.on_chain_end(
                outputs={"output": "result"},
                run_id=uuid4(),
            )

        with Path(temp_path).open("r") as f:
            content = f.read()
            assert "Finished chain" in content
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_file_callback_handler_agent_action() -> None:
    """Test writing agent actions."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        with FileCallbackHandler(temp_path, mode="w") as handler:
            action = AgentAction(
                tool="TestTool",
                tool_input={"query": "test"},
                log="Invoking TestTool",
            )
            handler.on_agent_action(action, run_id=uuid4())

        with Path(temp_path).open("r") as f:
            content = f.read()
            assert "Invoking TestTool" in content
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_file_callback_handler_agent_finish() -> None:
    """Test writing agent finish."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        with FileCallbackHandler(temp_path, mode="w") as handler:
            finish = AgentFinish(
                return_values={"output": "final result"},
                log="Agent finished successfully",
            )
            handler.on_agent_finish(finish, run_id=uuid4())

        with Path(temp_path).open("r") as f:
            content = f.read()
            assert "Agent finished successfully" in content
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_file_callback_handler_tool_end() -> None:
    """Test writing tool end events."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        with FileCallbackHandler(temp_path, mode="w") as handler:
            handler.on_tool_end(
                output="Tool output",
                observation_prefix="Observation:",
                llm_prefix="Thought:",
                run_id=uuid4(),
            )

        with Path(temp_path).open("r") as f:
            content = f.read()
            assert "Observation:" in content
            assert "Tool output" in content
            assert "Thought:" in content
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_file_callback_handler_on_text() -> None:
    """Test writing arbitrary text."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        with FileCallbackHandler(temp_path, mode="w") as handler:
            handler.on_text("Custom text", end="\n", run_id=uuid4())

        with Path(temp_path).open("r") as f:
            content = f.read()
            assert "Custom text" in content
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_file_callback_handler_append_mode() -> None:
    """Test that append mode works correctly."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Write initial content
        with FileCallbackHandler(temp_path, mode="w") as handler:
            handler.on_text("First line", run_id=uuid4())

        # Append more content
        with FileCallbackHandler(temp_path, mode="a") as handler:
            handler.on_text("Second line", run_id=uuid4())

        with Path(temp_path).open("r") as f:
            content = f.read()
            assert "First line" in content
            assert "Second line" in content
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_file_callback_handler_color_parameter() -> None:
    """Test that color parameter is stored."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        with FileCallbackHandler(temp_path, mode="w", color="green") as handler:
            assert handler.color == "green"
            handler.on_text("Colored text", run_id=uuid4())

        # File should contain the text (color codes depend on terminal support)
        with Path(temp_path).open("r") as f:
            content = f.read()
            assert "Colored text" in content
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_file_callback_handler_close_method() -> None:
    """Test that close() method works."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        handler = FileCallbackHandler(temp_path, mode="w")
        assert not handler.file.closed

        handler.close()
        assert handler.file.closed

        # Calling close again should not raise
        handler.close()
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_file_callback_handler_error_when_file_closed() -> None:
    """Test that writing to closed file raises error."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        handler = FileCallbackHandler(temp_path, mode="w")
        handler.close()

        with pytest.raises(RuntimeError, match="File is not open"):
            handler.on_text("Test", run_id=uuid4())
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_file_callback_handler_chain_start_with_name_kwarg() -> None:
    """Test chain start with name in kwargs."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        with FileCallbackHandler(temp_path, mode="w") as handler:
            handler.on_chain_start(
                serialized={},
                inputs={},
                run_id=uuid4(),
                name="CustomName",
            )

        with Path(temp_path).open("r") as f:
            content = f.read()
            assert "Entering new CustomName chain" in content
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_file_callback_handler_chain_start_with_serialized_id() -> None:
    """Test chain start falls back to serialized id."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        with FileCallbackHandler(temp_path, mode="w") as handler:
            handler.on_chain_start(
                serialized={"id": ["module", "ClassName"]},
                inputs={},
                run_id=uuid4(),
            )

        with Path(temp_path).open("r") as f:
            content = f.read()
            assert "Entering new ClassName chain" in content
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_file_callback_handler_chain_start_unknown() -> None:
    """Test chain start with no name available."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        with FileCallbackHandler(temp_path, mode="w") as handler:
            handler.on_chain_start(
                serialized={},
                inputs={},
                run_id=uuid4(),
            )

        with Path(temp_path).open("r") as f:
            content = f.read()
            assert "Entering new <unknown> chain" in content
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_file_callback_handler_multiple_calls() -> None:
    """Test multiple callback calls in sequence."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        with FileCallbackHandler(temp_path, mode="w") as handler:
            handler.on_chain_start(
                serialized={"name": "Chain1"},
                inputs={},
                run_id=uuid4(),
            )
            handler.on_text("Processing...", run_id=uuid4())
            handler.on_chain_end(outputs={}, run_id=uuid4())

        with Path(temp_path).open("r") as f:
            content = f.read()
            assert "Entering new Chain1 chain" in content
            assert "Processing..." in content
            assert "Finished chain" in content
    finally:
        Path(temp_path).unlink(missing_ok=True)
