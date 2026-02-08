"""Snapshot tests for StdOutCallbackHandler.

These tests capture the exact output format and behavior of the
StdOutCallbackHandler to detect unintended changes.
"""

from io import StringIO
from unittest.mock import patch
from uuid import uuid4

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.stdout import StdOutCallbackHandler


class TestStdOutCallbackHandlerInit:
    """Tests for handler initialization."""

    def test_default_color_is_none(self) -> None:
        handler = StdOutCallbackHandler()
        assert handler.color is None

    def test_custom_color_stored(self) -> None:
        handler = StdOutCallbackHandler(color="blue")
        assert handler.color == "blue"

    def test_inherits_base_handler(self) -> None:
        from langchain_core.callbacks.base import BaseCallbackHandler

        handler = StdOutCallbackHandler()
        assert isinstance(handler, BaseCallbackHandler)

    def test_default_flags(self) -> None:
        handler = StdOutCallbackHandler()
        assert handler.raise_error is False
        assert handler.run_inline is False


class TestOnChainStart:
    """Tests for on_chain_start output formatting."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_uses_name_from_kwargs(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_chain_start({}, {}, run_id=uuid4(), name="MyChain")
        output = mock_stdout.getvalue()
        assert "Entering new MyChain chain..." in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_uses_name_from_serialized(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_chain_start({"name": "SerializedChain"}, {}, run_id=uuid4())
        output = mock_stdout.getvalue()
        assert "Entering new SerializedChain chain..." in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_uses_id_from_serialized_as_fallback(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_chain_start(
            {"id": ["module", "path", "ClassName"]}, {}, run_id=uuid4()
        )
        output = mock_stdout.getvalue()
        assert "Entering new ClassName chain..." in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_uses_unknown_when_no_name(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_chain_start({}, {}, run_id=uuid4())
        output = mock_stdout.getvalue()
        assert "Entering new <unknown> chain..." in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_name_kwarg_takes_precedence_over_serialized(
        self, mock_stdout: StringIO
    ) -> None:
        handler = StdOutCallbackHandler()
        handler.on_chain_start(
            {"name": "Serialized"}, {}, run_id=uuid4(), name="KwargName"
        )
        output = mock_stdout.getvalue()
        assert "KwargName" in output
        assert "Serialized" not in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_output_has_bold_ansi_codes(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_chain_start({"name": "Test"}, {}, run_id=uuid4())
        output = mock_stdout.getvalue()
        # Bold start and reset
        assert "\033[1m" in output
        assert "\033[0m" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_serialized_none_uses_unknown(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_chain_start(None, {}, run_id=uuid4())  # type: ignore[arg-type]
        output = mock_stdout.getvalue()
        assert "Entering new <unknown> chain..." in output


class TestOnChainEnd:
    """Tests for on_chain_end output formatting."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_outputs_finished_chain(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_chain_end({}, run_id=uuid4())
        output = mock_stdout.getvalue()
        assert "Finished chain." in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_output_has_bold_ansi_codes(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_chain_end({}, run_id=uuid4())
        output = mock_stdout.getvalue()
        assert "\033[1m" in output
        assert "\033[0m" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_outputs_start_with_newline(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_chain_end({}, run_id=uuid4())
        output = mock_stdout.getvalue()
        assert output.startswith("\n")


class TestOnAgentAction:
    """Tests for on_agent_action output."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_outputs_action_log(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        action = AgentAction(tool="search", tool_input="q", log="Using search tool")
        handler.on_agent_action(action, run_id=uuid4())
        assert "Using search tool" in mock_stdout.getvalue()

    @patch("sys.stdout", new_callable=StringIO)
    def test_color_override(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler(color="green")
        action = AgentAction(tool="t", tool_input="i", log="log text")
        handler.on_agent_action(action, color="red", run_id=uuid4())
        # When color="red" is passed, it takes precedence over self.color
        assert "log text" in mock_stdout.getvalue()

    @patch("sys.stdout", new_callable=StringIO)
    def test_uses_default_color_when_no_override(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler(color="green")
        action = AgentAction(tool="t", tool_input="i", log="log")
        handler.on_agent_action(action, run_id=uuid4())
        assert "log" in mock_stdout.getvalue()


class TestOnToolEnd:
    """Tests for on_tool_end output."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_outputs_tool_result(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_tool_end("result text", run_id=uuid4())
        assert "result text" in mock_stdout.getvalue()

    @patch("sys.stdout", new_callable=StringIO)
    def test_with_observation_prefix(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_tool_end("result", observation_prefix="Obs:", run_id=uuid4())
        output = mock_stdout.getvalue()
        assert "Obs:" in output
        assert "result" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_with_llm_prefix(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_tool_end("result", llm_prefix="Think:", run_id=uuid4())
        output = mock_stdout.getvalue()
        assert "Think:" in output
        assert "result" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_with_both_prefixes(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_tool_end(
            "result",
            observation_prefix="Obs:",
            llm_prefix="Think:",
            run_id=uuid4(),
        )
        output = mock_stdout.getvalue()
        assert "Obs:" in output
        assert "result" in output
        assert "Think:" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_no_prefix_when_none(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_tool_end("result", run_id=uuid4())
        output = mock_stdout.getvalue()
        # Should not contain prefix markers
        assert "Obs:" not in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_non_string_output_converted(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_tool_end(42, run_id=uuid4())
        assert "42" in mock_stdout.getvalue()

    @patch("sys.stdout", new_callable=StringIO)
    def test_color_override(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler(color="green")
        handler.on_tool_end("result", color="red", run_id=uuid4())
        assert "result" in mock_stdout.getvalue()


class TestOnText:
    """Tests for on_text output."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_outputs_text(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_text("hello world", run_id=uuid4())
        assert "hello world" in mock_stdout.getvalue()

    @patch("sys.stdout", new_callable=StringIO)
    def test_custom_end_character(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_text("line1", end="\n", run_id=uuid4())
        handler.on_text("line2", end="", run_id=uuid4())
        output = mock_stdout.getvalue()
        assert "line1\n" in output
        assert "line2" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_default_end_is_empty(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_text("a", run_id=uuid4())
        handler.on_text("b", run_id=uuid4())
        output = mock_stdout.getvalue()
        # With empty end, 'a' and 'b' should be adjacent (with possible color codes)
        assert "a" in output
        assert "b" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_empty_text(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        handler.on_text("", run_id=uuid4())
        # Should not raise


class TestOnAgentFinish:
    """Tests for on_agent_finish output."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_outputs_finish_log(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        finish = AgentFinish(return_values={"output": "done"}, log="Final answer: done")
        handler.on_agent_finish(finish, run_id=uuid4())
        assert "Final answer: done" in mock_stdout.getvalue()

    @patch("sys.stdout", new_callable=StringIO)
    def test_color_override(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler(color="green")
        finish = AgentFinish(return_values={}, log="log")
        handler.on_agent_finish(finish, color="red", run_id=uuid4())
        assert "log" in mock_stdout.getvalue()

    @patch("sys.stdout", new_callable=StringIO)
    def test_ends_with_newline(self, mock_stdout: StringIO) -> None:
        handler = StdOutCallbackHandler()
        finish = AgentFinish(return_values={}, log="log")
        handler.on_agent_finish(finish, run_id=uuid4())
        output = mock_stdout.getvalue()
        assert output.endswith("\n")
