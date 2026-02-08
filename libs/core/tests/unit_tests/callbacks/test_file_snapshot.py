"""Snapshot tests for FileCallbackHandler.

These tests capture the exact behavior of the FileCallbackHandler including
context manager usage, file modes, output formatting, cleanup, and
deprecation warnings.
"""

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.file import FileCallbackHandler


def _temp_path() -> str:
    """Create a temporary file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
    f.close()
    return f.name


class TestInit:
    """Tests for handler initialization."""

    def test_inherits_base_handler(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path) as handler:
                assert isinstance(handler, BaseCallbackHandler)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_stores_filename(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path) as handler:
                assert handler.filename == path
        finally:
            Path(path).unlink(missing_ok=True)

    def test_stores_mode(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                assert handler.mode == "w"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_default_mode_is_append(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path) as handler:
                assert handler.mode == "a"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_stores_color(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, color="green") as handler:
                assert handler.color == "green"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_default_color_is_none(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path) as handler:
                assert handler.color is None
        finally:
            Path(path).unlink(missing_ok=True)

    def test_file_is_opened_on_init(self) -> None:
        path = _temp_path()
        try:
            handler = FileCallbackHandler(path)
            assert not handler.file.closed
            handler.close()
        finally:
            Path(path).unlink(missing_ok=True)


class TestContextManager:
    """Tests for context manager usage."""

    def test_file_closed_on_exit(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path) as handler:
                assert not handler.file.closed
            assert handler.file.closed
        finally:
            Path(path).unlink(missing_ok=True)

    def test_returns_self(self) -> None:
        path = _temp_path()
        try:
            handler = FileCallbackHandler(path)
            with handler as h:
                assert h is handler
        finally:
            Path(path).unlink(missing_ok=True)

    def test_sets_file_opened_in_context_flag(self) -> None:
        path = _temp_path()
        try:
            handler = FileCallbackHandler(path)
            assert handler._file_opened_in_context is False
            with handler:
                assert handler._file_opened_in_context is True
        finally:
            Path(path).unlink(missing_ok=True)

    def test_file_closed_even_on_exception(self) -> None:
        path = _temp_path()
        try:
            with pytest.raises(ValueError):
                with FileCallbackHandler(path) as handler:
                    raise ValueError("test")
            assert handler.file.closed
        finally:
            Path(path).unlink(missing_ok=True)


class TestClose:
    """Tests for close() method."""

    def test_close_closes_file(self) -> None:
        path = _temp_path()
        try:
            handler = FileCallbackHandler(path)
            handler.close()
            assert handler.file.closed
        finally:
            Path(path).unlink(missing_ok=True)

    def test_close_idempotent(self) -> None:
        path = _temp_path()
        try:
            handler = FileCallbackHandler(path)
            handler.close()
            handler.close()  # should not raise
            handler.close()
            assert handler.file.closed
        finally:
            Path(path).unlink(missing_ok=True)


class TestWriteToClosedFile:
    """Tests for error when writing to closed file."""

    def test_on_text_raises_runtime_error(self) -> None:
        path = _temp_path()
        try:
            handler = FileCallbackHandler(path)
            handler.close()
            with pytest.raises(RuntimeError, match="File is not open"):
                handler.on_text("test", run_id=uuid4())
        finally:
            Path(path).unlink(missing_ok=True)

    def test_on_chain_start_raises_runtime_error(self) -> None:
        path = _temp_path()
        try:
            handler = FileCallbackHandler(path)
            handler.close()
            with pytest.raises(RuntimeError, match="File is not open"):
                handler.on_chain_start({}, {}, run_id=uuid4())
        finally:
            Path(path).unlink(missing_ok=True)


class TestOnChainStart:
    """Tests for on_chain_start file output."""

    def test_with_serialized_name(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_chain_start({"name": "TestChain"}, {}, run_id=uuid4())
            content = Path(path).read_text()
            assert "Entering new TestChain chain..." in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_with_name_kwarg(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_chain_start({}, {}, run_id=uuid4(), name="KwName")
            content = Path(path).read_text()
            assert "Entering new KwName chain..." in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_with_serialized_id_fallback(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_chain_start({"id": ["mod", "ClassName"]}, {}, run_id=uuid4())
            content = Path(path).read_text()
            assert "Entering new ClassName chain..." in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_unknown_when_no_name(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_chain_start({}, {}, run_id=uuid4())
            content = Path(path).read_text()
            assert "Entering new <unknown> chain..." in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_name_kwarg_precedence(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_chain_start({"name": "Ser"}, {}, run_id=uuid4(), name="Kw")
            content = Path(path).read_text()
            assert "Kw" in content
            assert "Ser" not in content
        finally:
            Path(path).unlink(missing_ok=True)


class TestOnChainEnd:
    """Tests for on_chain_end file output."""

    def test_outputs_finished_chain(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_chain_end({}, run_id=uuid4())
            content = Path(path).read_text()
            assert "Finished chain." in content
        finally:
            Path(path).unlink(missing_ok=True)


class TestOnAgentAction:
    """Tests for on_agent_action file output."""

    def test_writes_action_log(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                action = AgentAction(tool="t", tool_input="i", log="Action log")
                handler.on_agent_action(action, run_id=uuid4())
            content = Path(path).read_text()
            assert "Action log" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_color_override(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w", color="green") as handler:
                action = AgentAction(tool="t", tool_input="i", log="log")
                handler.on_agent_action(action, color="red", run_id=uuid4())
            content = Path(path).read_text()
            assert "log" in content
        finally:
            Path(path).unlink(missing_ok=True)


class TestOnAgentFinish:
    """Tests for on_agent_finish file output."""

    def test_writes_finish_log(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                finish = AgentFinish(return_values={}, log="Done!")
                handler.on_agent_finish(finish, run_id=uuid4())
            content = Path(path).read_text()
            assert "Done!" in content
        finally:
            Path(path).unlink(missing_ok=True)


class TestOnToolEnd:
    """Tests for on_tool_end file output."""

    def test_writes_output(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_tool_end("Tool result", run_id=uuid4())
            content = Path(path).read_text()
            assert "Tool result" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_with_observation_prefix(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_tool_end(
                    "result",
                    observation_prefix="Obs:",
                    run_id=uuid4(),
                )
            content = Path(path).read_text()
            assert "Obs:" in content
            assert "result" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_with_llm_prefix(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_tool_end(
                    "result",
                    llm_prefix="Think:",
                    run_id=uuid4(),
                )
            content = Path(path).read_text()
            assert "Think:" in content
            assert "result" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_without_prefixes(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_tool_end("result", run_id=uuid4())
            content = Path(path).read_text()
            assert "result" in content
        finally:
            Path(path).unlink(missing_ok=True)


class TestOnText:
    """Tests for on_text file output."""

    def test_writes_text(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_text("hello", run_id=uuid4())
            content = Path(path).read_text()
            assert "hello" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_custom_end(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_text("line1", end="\n", run_id=uuid4())
                handler.on_text("line2", end="", run_id=uuid4())
            content = Path(path).read_text()
            assert "line1" in content
            assert "line2" in content
        finally:
            Path(path).unlink(missing_ok=True)


class TestFileModes:
    """Tests for different file modes."""

    def test_write_mode_truncates(self) -> None:
        path = _temp_path()
        try:
            # Write initial content
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_text("first", run_id=uuid4())

            # Write again in write mode (should truncate)
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_text("second", run_id=uuid4())

            content = Path(path).read_text()
            assert "first" not in content
            assert "second" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_append_mode_preserves(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_text("first", run_id=uuid4())

            with FileCallbackHandler(path, mode="a") as handler:
                handler.on_text("second", run_id=uuid4())

            content = Path(path).read_text()
            assert "first" in content
            assert "second" in content
        finally:
            Path(path).unlink(missing_ok=True)


class TestMultipleCallSequence:
    """Tests for realistic multi-call sequences."""

    def test_full_chain_lifecycle(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_chain_start(
                    {"name": "TestChain"}, {"input": "test"}, run_id=uuid4()
                )
                handler.on_text("Processing...", run_id=uuid4())
                handler.on_chain_end({"output": "done"}, run_id=uuid4())

            content = Path(path).read_text()
            assert "Entering new TestChain chain" in content
            assert "Processing..." in content
            assert "Finished chain" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_agent_lifecycle(self) -> None:
        path = _temp_path()
        try:
            with FileCallbackHandler(path, mode="w") as handler:
                handler.on_chain_start({"name": "Agent"}, {}, run_id=uuid4())
                action = AgentAction(
                    tool="search", tool_input="query", log="Searching..."
                )
                handler.on_agent_action(action, run_id=uuid4())
                handler.on_tool_end(
                    "Search result",
                    observation_prefix="Obs:",
                    llm_prefix="Think:",
                    run_id=uuid4(),
                )
                finish = AgentFinish(
                    return_values={"output": "answer"}, log="Final: answer"
                )
                handler.on_agent_finish(finish, run_id=uuid4())
                handler.on_chain_end({}, run_id=uuid4())

            content = Path(path).read_text()
            assert "Agent" in content
            assert "Searching..." in content
            assert "Search result" in content
            assert "Obs:" in content
            assert "Think:" in content
            assert "Final: answer" in content
            assert "Finished chain" in content
        finally:
            Path(path).unlink(missing_ok=True)


class TestDestructor:
    """Tests for __del__ cleanup."""

    def test_del_closes_file(self) -> None:
        path = _temp_path()
        try:
            handler = FileCallbackHandler(path)
            file_ref = handler.file
            assert not file_ref.closed
            handler.__del__()
            assert file_ref.closed
        finally:
            Path(path).unlink(missing_ok=True)

    def test_del_after_close_no_error(self) -> None:
        path = _temp_path()
        try:
            handler = FileCallbackHandler(path)
            handler.close()
            handler.__del__()  # should not raise
        finally:
            Path(path).unlink(missing_ok=True)
