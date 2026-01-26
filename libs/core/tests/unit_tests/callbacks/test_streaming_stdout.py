"""Unit tests for StreamingStdOutCallbackHandler."""

from io import StringIO
from unittest.mock import patch
from uuid import uuid4

from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult


@patch("sys.stdout", new_callable=StringIO)
def test_streaming_stdout_on_llm_new_token(mock_stdout: StringIO) -> None:
    """Test streaming tokens to stdout."""
    handler = StreamingStdOutCallbackHandler()

    handler.on_llm_new_token("Hello", run_id=uuid4())
    handler.on_llm_new_token(" ", run_id=uuid4())
    handler.on_llm_new_token("world", run_id=uuid4())

    output = mock_stdout.getvalue()
    assert output == "Hello world"


@patch("sys.stdout", new_callable=StringIO)
def test_streaming_stdout_on_llm_start(mock_stdout: StringIO) -> None:
    """Test that on_llm_start is a no-op."""
    handler = StreamingStdOutCallbackHandler()

    handler.on_llm_start(
        serialized={},
        prompts=["test"],
        run_id=uuid4(),
    )

    # Should not output anything
    output = mock_stdout.getvalue()
    assert output == ""


@patch("sys.stdout", new_callable=StringIO)
def test_streaming_stdout_on_chat_model_start(mock_stdout: StringIO) -> None:
    """Test that on_chat_model_start is a no-op."""
    handler = StreamingStdOutCallbackHandler()

    handler.on_chat_model_start(
        serialized={},
        messages=[[HumanMessage(content="test")]],
        run_id=uuid4(),
    )

    # Should not output anything
    output = mock_stdout.getvalue()
    assert output == ""


@patch("sys.stdout", new_callable=StringIO)
def test_streaming_stdout_on_llm_end(mock_stdout: StringIO) -> None:
    """Test that on_llm_end is a no-op."""
    handler = StreamingStdOutCallbackHandler()

    handler.on_llm_end(
        response=LLMResult(generations=[[]], llm_output=None),
        run_id=uuid4(),
    )

    # Should not output anything
    output = mock_stdout.getvalue()
    assert output == ""


@patch("sys.stdout", new_callable=StringIO)
def test_streaming_stdout_on_llm_error(mock_stdout: StringIO) -> None:
    """Test that on_llm_error is a no-op."""
    handler = StreamingStdOutCallbackHandler()

    handler.on_llm_error(
        error=ValueError("test error"),
        run_id=uuid4(),
    )

    # Should not output anything
    output = mock_stdout.getvalue()
    assert output == ""


@patch("sys.stdout", new_callable=StringIO)
def test_streaming_stdout_on_chain_start(mock_stdout: StringIO) -> None:
    """Test that on_chain_start is a no-op."""
    handler = StreamingStdOutCallbackHandler()

    handler.on_chain_start(
        serialized={},
        inputs={},
        run_id=uuid4(),
    )

    # Should not output anything
    output = mock_stdout.getvalue()
    assert output == ""


@patch("sys.stdout", new_callable=StringIO)
def test_streaming_stdout_on_chain_end(mock_stdout: StringIO) -> None:
    """Test that on_chain_end is a no-op."""
    handler = StreamingStdOutCallbackHandler()

    handler.on_chain_end(
        outputs={},
        run_id=uuid4(),
    )

    # Should not output anything
    output = mock_stdout.getvalue()
    assert output == ""


@patch("sys.stdout", new_callable=StringIO)
def test_streaming_stdout_on_chain_error(mock_stdout: StringIO) -> None:
    """Test that on_chain_error is a no-op."""
    handler = StreamingStdOutCallbackHandler()

    handler.on_chain_error(
        error=ValueError("test"),
        run_id=uuid4(),
    )

    # Should not output anything
    output = mock_stdout.getvalue()
    assert output == ""


@patch("sys.stdout", new_callable=StringIO)
def test_streaming_stdout_on_tool_start(mock_stdout: StringIO) -> None:
    """Test that on_tool_start is a no-op."""
    handler = StreamingStdOutCallbackHandler()

    handler.on_tool_start(
        serialized={},
        input_str="test",
        run_id=uuid4(),
    )

    # Should not output anything
    output = mock_stdout.getvalue()
    assert output == ""


@patch("sys.stdout", new_callable=StringIO)
def test_streaming_stdout_on_tool_end(mock_stdout: StringIO) -> None:
    """Test that on_tool_end is a no-op."""
    handler = StreamingStdOutCallbackHandler()

    handler.on_tool_end(
        output="result",
        run_id=uuid4(),
    )

    # Should not output anything
    output = mock_stdout.getvalue()
    assert output == ""


@patch("sys.stdout", new_callable=StringIO)
def test_streaming_stdout_on_tool_error(mock_stdout: StringIO) -> None:
    """Test that on_tool_error is a no-op."""
    handler = StreamingStdOutCallbackHandler()

    handler.on_tool_error(
        error=ValueError("test"),
        run_id=uuid4(),
    )

    # Should not output anything
    output = mock_stdout.getvalue()
    assert output == ""


@patch("sys.stdout", new_callable=StringIO)
def test_streaming_stdout_on_text(mock_stdout: StringIO) -> None:
    """Test that on_text is a no-op."""
    handler = StreamingStdOutCallbackHandler()

    handler.on_text(
        text="test",
        run_id=uuid4(),
    )

    # Should not output anything
    output = mock_stdout.getvalue()
    assert output == ""


@patch("sys.stdout", new_callable=StringIO)
def test_streaming_stdout_multiple_tokens(mock_stdout: StringIO) -> None:
    """Test streaming multiple tokens in sequence."""
    handler = StreamingStdOutCallbackHandler()

    tokens = ["The", " ", "quick", " ", "brown", " ", "fox"]
    for token in tokens:
        handler.on_llm_new_token(token, run_id=uuid4())

    output = mock_stdout.getvalue()
    assert output == "The quick brown fox"


@patch("sys.stdout", new_callable=StringIO)
def test_streaming_stdout_flush_behavior(mock_stdout: StringIO) -> None:
    """Test that output is flushed immediately."""
    handler = StreamingStdOutCallbackHandler()

    # Each token should be immediately available after call
    handler.on_llm_new_token("Token1", run_id=uuid4())
    assert "Token1" in mock_stdout.getvalue()

    handler.on_llm_new_token("Token2", run_id=uuid4())
    assert "Token1Token2" in mock_stdout.getvalue()
