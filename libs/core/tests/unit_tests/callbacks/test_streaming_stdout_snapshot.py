"""Snapshot tests for StreamingStdOutCallbackHandler.

These tests capture the exact behavior of the StreamingStdOutCallbackHandler:
only on_llm_new_token produces output; all other methods are no-ops.
"""

from io import StringIO
from unittest.mock import patch
from uuid import uuid4

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult


class TestInit:
    """Tests for handler initialization."""

    def test_inherits_base_handler(self) -> None:
        handler = StreamingStdOutCallbackHandler()
        assert isinstance(handler, BaseCallbackHandler)

    def test_default_flags(self) -> None:
        handler = StreamingStdOutCallbackHandler()
        assert handler.raise_error is False
        assert handler.run_inline is False


class TestOnLLMNewToken:
    """Tests for the main streaming functionality."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_single_token(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_llm_new_token("Hello", run_id=uuid4())
        assert mock_stdout.getvalue() == "Hello"

    @patch("sys.stdout", new_callable=StringIO)
    def test_multiple_tokens_concatenate(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        for token in ["The", " ", "answer", " ", "is", " ", "42"]:
            handler.on_llm_new_token(token, run_id=uuid4())
        assert mock_stdout.getvalue() == "The answer is 42"

    @patch("sys.stdout", new_callable=StringIO)
    def test_empty_token(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_llm_new_token("", run_id=uuid4())
        assert mock_stdout.getvalue() == ""

    @patch("sys.stdout", new_callable=StringIO)
    def test_token_with_newlines(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_llm_new_token("line1\nline2\n", run_id=uuid4())
        assert mock_stdout.getvalue() == "line1\nline2\n"

    @patch("sys.stdout", new_callable=StringIO)
    def test_token_with_special_characters(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_llm_new_token("日本語テスト", run_id=uuid4())
        assert mock_stdout.getvalue() == "日本語テスト"

    @patch("sys.stdout", new_callable=StringIO)
    def test_flush_after_each_token(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_llm_new_token("A", run_id=uuid4())
        assert mock_stdout.getvalue() == "A"
        handler.on_llm_new_token("B", run_id=uuid4())
        assert mock_stdout.getvalue() == "AB"


class TestNoOpMethods:
    """Verify all non-token methods produce no output."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_on_llm_start_noop(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_llm_start({}, ["prompt"], run_id=uuid4())
        assert mock_stdout.getvalue() == ""

    @patch("sys.stdout", new_callable=StringIO)
    def test_on_chat_model_start_noop(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_chat_model_start({}, [[HumanMessage(content="hi")]], run_id=uuid4())
        assert mock_stdout.getvalue() == ""

    @patch("sys.stdout", new_callable=StringIO)
    def test_on_llm_end_noop(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_llm_end(LLMResult(generations=[]), run_id=uuid4())
        assert mock_stdout.getvalue() == ""

    @patch("sys.stdout", new_callable=StringIO)
    def test_on_llm_error_noop(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_llm_error(ValueError("err"), run_id=uuid4())
        assert mock_stdout.getvalue() == ""

    @patch("sys.stdout", new_callable=StringIO)
    def test_on_chain_start_noop(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_chain_start({}, {}, run_id=uuid4())
        assert mock_stdout.getvalue() == ""

    @patch("sys.stdout", new_callable=StringIO)
    def test_on_chain_end_noop(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_chain_end({}, run_id=uuid4())
        assert mock_stdout.getvalue() == ""

    @patch("sys.stdout", new_callable=StringIO)
    def test_on_chain_error_noop(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_chain_error(ValueError(), run_id=uuid4())
        assert mock_stdout.getvalue() == ""

    @patch("sys.stdout", new_callable=StringIO)
    def test_on_tool_start_noop(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_tool_start({}, "input", run_id=uuid4())
        assert mock_stdout.getvalue() == ""

    @patch("sys.stdout", new_callable=StringIO)
    def test_on_tool_end_noop(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_tool_end("output", run_id=uuid4())
        assert mock_stdout.getvalue() == ""

    @patch("sys.stdout", new_callable=StringIO)
    def test_on_tool_error_noop(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_tool_error(ValueError(), run_id=uuid4())
        assert mock_stdout.getvalue() == ""

    @patch("sys.stdout", new_callable=StringIO)
    def test_on_text_noop(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        handler.on_text("text", run_id=uuid4())
        assert mock_stdout.getvalue() == ""

    @patch("sys.stdout", new_callable=StringIO)
    def test_on_agent_action_noop(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        action = AgentAction(tool="t", tool_input="i", log="l")
        handler.on_agent_action(action, run_id=uuid4())
        assert mock_stdout.getvalue() == ""

    @patch("sys.stdout", new_callable=StringIO)
    def test_on_agent_finish_noop(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        finish = AgentFinish(return_values={}, log="done")
        handler.on_agent_finish(finish, run_id=uuid4())
        assert mock_stdout.getvalue() == ""


class TestTokenSequences:
    """Tests for realistic streaming sequences."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_word_by_word_streaming(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        sentence = "The quick brown fox jumps over the lazy dog"
        tokens = sentence.split(" ")
        for i, token in enumerate(tokens):
            prefix = " " if i > 0 else ""
            handler.on_llm_new_token(prefix + token, run_id=uuid4())
        assert mock_stdout.getvalue() == sentence

    @patch("sys.stdout", new_callable=StringIO)
    def test_character_by_character(self, mock_stdout: StringIO) -> None:
        handler = StreamingStdOutCallbackHandler()
        text = "Hello"
        for char in text:
            handler.on_llm_new_token(char, run_id=uuid4())
        assert mock_stdout.getvalue() == text

    @patch("sys.stdout", new_callable=StringIO)
    def test_mixed_with_noop_methods(self, mock_stdout: StringIO) -> None:
        """No-op methods interleaved with tokens should not affect output."""
        handler = StreamingStdOutCallbackHandler()
        handler.on_llm_start({}, ["p"], run_id=uuid4())
        handler.on_llm_new_token("A", run_id=uuid4())
        handler.on_text("ignored", run_id=uuid4())
        handler.on_llm_new_token("B", run_id=uuid4())
        handler.on_llm_end(LLMResult(generations=[]), run_id=uuid4())
        assert mock_stdout.getvalue() == "AB"
