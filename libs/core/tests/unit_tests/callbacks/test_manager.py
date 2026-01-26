"""Unit tests for callback run managers."""

from typing import Any
from uuid import UUID, uuid4

import pytest

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForLLMRun,
    AsyncCallbackManagerForRetrieverRun,
    AsyncCallbackManagerForToolRun,
    AsyncParentRunManager,
    AsyncRunManager,
    BaseRunManager,
    CallbackManager,
    CallbackManagerForChainRun,
    CallbackManagerForLLMRun,
    CallbackManagerForRetrieverRun,
    CallbackManagerForToolRun,
    ParentRunManager,
    RunManager,
)
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult


def test_base_run_manager_initialization() -> None:
    """Test BaseRunManager initialization."""
    run_id = uuid4()
    parent_run_id = uuid4()
    handler = BaseCallbackHandler()

    manager = BaseRunManager(
        run_id=run_id,
        handlers=[handler],
        inheritable_handlers=[handler],
        parent_run_id=parent_run_id,
        tags=["tag1"],
        inheritable_tags=["tag2"],
        metadata={"key": "value"},
        inheritable_metadata={"key2": "value2"},
    )

    assert manager.run_id == run_id
    assert manager.parent_run_id == parent_run_id
    assert handler in manager.handlers
    assert handler in manager.inheritable_handlers
    assert "tag1" in manager.tags
    assert "tag2" in manager.inheritable_tags
    assert manager.metadata["key"] == "value"
    assert manager.inheritable_metadata["key2"] == "value2"


def test_base_run_manager_get_noop_manager() -> None:
    """Test creating a no-op manager."""
    manager = BaseRunManager.get_noop_manager()

    assert isinstance(manager.run_id, UUID)
    assert len(manager.handlers) == 0
    assert len(manager.inheritable_handlers) == 0


def test_run_manager_on_text() -> None:
    """Test RunManager on_text method."""
    run_id = uuid4()

    class TextTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.texts: list[str] = []

        def on_text(self, text: str, **kwargs: Any) -> None:
            self.texts.append(text)

    tracker = TextTracker()
    manager = RunManager(
        run_id=run_id,
        handlers=[tracker],
        inheritable_handlers=[],
    )

    manager.on_text("Hello")
    manager.on_text("World")

    assert tracker.texts == ["Hello", "World"]


def test_run_manager_empty_handlers() -> None:
    """Test RunManager with no handlers."""
    manager = RunManager(
        run_id=uuid4(),
        handlers=[],
        inheritable_handlers=[],
    )

    # Should not raise
    manager.on_text("test")


def test_parent_run_manager_get_child() -> None:
    """Test ParentRunManager creating child managers."""
    handler = BaseCallbackHandler()
    parent_manager = ParentRunManager(
        run_id=uuid4(),
        handlers=[handler],
        inheritable_handlers=[handler],
        tags=["parent_tag"],
        inheritable_tags=["inheritable_tag"],
        metadata={"key": "value"},
        inheritable_metadata={"key2": "value2"},
    )

    child = parent_manager.get_child()

    assert isinstance(child, CallbackManager)
    assert child.parent_run_id == parent_manager.run_id
    assert handler in child.handlers
    assert "inheritable_tag" in child.tags
    assert child.metadata["key2"] == "value2"


def test_parent_run_manager_get_child_with_tag() -> None:
    """Test ParentRunManager creating child with additional tag."""
    parent_manager = ParentRunManager(
        run_id=uuid4(),
        handlers=[],
        inheritable_handlers=[],
    )

    child = parent_manager.get_child(tag="child_tag")

    assert "child_tag" in child.tags
    # Child tag should not be inheritable
    assert "child_tag" not in child.inheritable_tags


async def test_async_run_manager_on_text() -> None:
    """Test AsyncRunManager on_text method."""
    run_id = uuid4()

    class AsyncTextTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.texts: list[str] = []

        async def on_text(self, text: str, **kwargs: Any) -> None:
            self.texts.append(text)

    tracker = AsyncTextTracker()

    class TestAsyncRunManager(AsyncRunManager):
        def get_sync(self) -> RunManager:
            return RunManager(
                run_id=self.run_id,
                handlers=self.handlers,
                inheritable_handlers=self.inheritable_handlers,
            )

    manager = TestAsyncRunManager(
        run_id=run_id,
        handlers=[tracker],
        inheritable_handlers=[],
    )

    await manager.on_text("Hello")
    await manager.on_text("World")

    assert tracker.texts == ["Hello", "World"]


async def test_async_parent_run_manager_get_child() -> None:
    """Test AsyncCallbackManagerForChainRun (which extends AsyncParentRunManager) creating child managers."""
    from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun

    handler = BaseCallbackHandler()
    # Use AsyncCallbackManagerForChainRun which is a concrete implementation
    parent_manager = AsyncCallbackManagerForChainRun(
        run_id=uuid4(),
        handlers=[handler],
        inheritable_handlers=[handler],
        inheritable_tags=["inheritable_tag"],
    )

    child = parent_manager.get_child()

    assert isinstance(child, AsyncCallbackManager)
    assert child.parent_run_id == parent_manager.run_id
    assert handler in child.handlers


def test_callback_manager_for_llm_run_on_llm_new_token() -> None:
    """Test CallbackManagerForLLMRun on_llm_new_token."""

    class TokenTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.tokens: list[str] = []

        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            self.tokens.append(token)

    tracker = TokenTracker()
    manager = CallbackManagerForLLMRun(
        run_id=uuid4(),
        handlers=[tracker],
        inheritable_handlers=[],
    )

    manager.on_llm_new_token("Hello")
    manager.on_llm_new_token(" ")
    manager.on_llm_new_token("World")

    assert tracker.tokens == ["Hello", " ", "World"]


def test_callback_manager_for_llm_run_on_llm_end() -> None:
    """Test CallbackManagerForLLMRun on_llm_end."""

    class EndTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.ended = False

        def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
            self.ended = True

    tracker = EndTracker()
    manager = CallbackManagerForLLMRun(
        run_id=uuid4(),
        handlers=[tracker],
        inheritable_handlers=[],
    )

    manager.on_llm_end(LLMResult(generations=[[]], llm_output=None))

    assert tracker.ended is True


def test_callback_manager_for_llm_run_on_llm_error() -> None:
    """Test CallbackManagerForLLMRun on_llm_error."""

    class ErrorTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.error: BaseException | None = None

        def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
            self.error = error

    tracker = ErrorTracker()
    manager = CallbackManagerForLLMRun(
        run_id=uuid4(),
        handlers=[tracker],
        inheritable_handlers=[],
    )

    test_error = ValueError("Test error")
    manager.on_llm_error(test_error)

    assert tracker.error is test_error


async def test_async_callback_manager_for_llm_run_get_sync() -> None:
    """Test AsyncCallbackManagerForLLMRun.get_sync()."""
    run_id = uuid4()
    handler = BaseCallbackHandler()

    async_manager = AsyncCallbackManagerForLLMRun(
        run_id=run_id,
        handlers=[handler],
        inheritable_handlers=[handler],
        tags=["test"],
    )

    sync_manager = async_manager.get_sync()

    assert isinstance(sync_manager, CallbackManagerForLLMRun)
    assert sync_manager.run_id == run_id
    assert handler in sync_manager.handlers
    assert "test" in sync_manager.tags


def test_callback_manager_for_chain_run_on_chain_end() -> None:
    """Test CallbackManagerForChainRun on_chain_end."""

    class ChainTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.outputs: dict[str, Any] | None = None

        def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
            self.outputs = outputs

    tracker = ChainTracker()
    manager = CallbackManagerForChainRun(
        run_id=uuid4(),
        handlers=[tracker],
        inheritable_handlers=[],
    )

    test_outputs = {"result": "success"}
    manager.on_chain_end(test_outputs)

    assert tracker.outputs == test_outputs


def test_callback_manager_for_chain_run_on_chain_error() -> None:
    """Test CallbackManagerForChainRun on_chain_error."""

    class ChainErrorTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.error: BaseException | None = None

        def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
            self.error = error

    tracker = ChainErrorTracker()
    manager = CallbackManagerForChainRun(
        run_id=uuid4(),
        handlers=[tracker],
        inheritable_handlers=[],
    )

    test_error = ValueError("Chain failed")
    manager.on_chain_error(test_error)

    assert tracker.error is test_error


def test_callback_manager_for_tool_run_on_tool_end() -> None:
    """Test CallbackManagerForToolRun on_tool_end."""

    class ToolTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.output: Any = None

        def on_tool_end(self, output: Any, **kwargs: Any) -> None:
            self.output = output

    tracker = ToolTracker()
    manager = CallbackManagerForToolRun(
        run_id=uuid4(),
        handlers=[tracker],
        inheritable_handlers=[],
    )

    manager.on_tool_end("Tool result")

    assert tracker.output == "Tool result"


def test_callback_manager_for_tool_run_on_tool_error() -> None:
    """Test CallbackManagerForToolRun on_tool_error."""

    class ToolErrorTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.error: BaseException | None = None

        def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
            self.error = error

    tracker = ToolErrorTracker()
    manager = CallbackManagerForToolRun(
        run_id=uuid4(),
        handlers=[tracker],
        inheritable_handlers=[],
    )

    test_error = ValueError("Tool failed")
    manager.on_tool_error(test_error)

    assert tracker.error is test_error


def test_callback_manager_for_retriever_run_on_retriever_end() -> None:
    """Test CallbackManagerForRetrieverRun on_retriever_end."""

    class RetrieverTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.documents: list[Document] = []

        def on_retriever_end(
            self,
            documents: list[Document],
            **kwargs: Any,
        ) -> None:
            self.documents = documents

    tracker = RetrieverTracker()
    manager = CallbackManagerForRetrieverRun(
        run_id=uuid4(),
        handlers=[tracker],
        inheritable_handlers=[],
    )

    docs = [Document(page_content="test")]
    manager.on_retriever_end(docs)

    assert tracker.documents == docs


def test_callback_manager_for_retriever_run_on_retriever_error() -> None:
    """Test CallbackManagerForRetrieverRun on_retriever_error."""

    class RetrieverErrorTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.error: BaseException | None = None

        def on_retriever_error(self, error: BaseException, **kwargs: Any) -> None:
            self.error = error

    tracker = RetrieverErrorTracker()
    manager = CallbackManagerForRetrieverRun(
        run_id=uuid4(),
        handlers=[tracker],
        inheritable_handlers=[],
    )

    test_error = ValueError("Retriever failed")
    manager.on_retriever_error(test_error)

    assert tracker.error is test_error


def test_callback_manager_on_llm_start_single_prompt() -> None:
    """Test CallbackManager.on_llm_start with single prompt."""

    class LLMStartTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.prompts: list[list[str]] = []

        def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            **kwargs: Any,
        ) -> None:
            self.prompts.append(prompts)

    tracker = LLMStartTracker()
    manager = CallbackManager(handlers=[tracker])

    run_managers = manager.on_llm_start(
        serialized={},
        prompts=["Test prompt"],
    )

    assert len(run_managers) == 1
    assert isinstance(run_managers[0], CallbackManagerForLLMRun)
    assert tracker.prompts == [["Test prompt"]]


def test_callback_manager_on_llm_start_multiple_prompts() -> None:
    """Test CallbackManager.on_llm_start with multiple prompts."""

    class LLMStartTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.call_count = 0

        def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            **kwargs: Any,
        ) -> None:
            self.call_count += 1

    tracker = LLMStartTracker()
    manager = CallbackManager(handlers=[tracker])

    run_managers = manager.on_llm_start(
        serialized={},
        prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
    )

    assert len(run_managers) == 3
    assert tracker.call_count == 3


def test_callback_manager_on_chain_start() -> None:
    """Test CallbackManager.on_chain_start."""

    class ChainStartTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.called = False

        def on_chain_start(
            self,
            serialized: dict[str, Any] | None,
            inputs: dict[str, Any] | Any,
            **kwargs: Any,
        ) -> None:
            self.called = True

    tracker = ChainStartTracker()
    manager = CallbackManager(handlers=[tracker])

    run_manager = manager.on_chain_start(
        serialized={},
        inputs={"input": "test"},
    )

    assert isinstance(run_manager, CallbackManagerForChainRun)
    assert tracker.called is True


def test_callback_manager_on_tool_start() -> None:
    """Test CallbackManager.on_tool_start."""

    class ToolStartTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.input_str: str | None = None

        def on_tool_start(
            self,
            serialized: dict[str, Any] | None,
            input_str: str,
            **kwargs: Any,
        ) -> None:
            self.input_str = input_str

    tracker = ToolStartTracker()
    manager = CallbackManager(handlers=[tracker])

    run_manager = manager.on_tool_start(
        serialized={},
        input_str="test input",
    )

    assert isinstance(run_manager, CallbackManagerForToolRun)
    assert tracker.input_str == "test input"


def test_callback_manager_on_retriever_start() -> None:
    """Test CallbackManager.on_retriever_start."""

    class RetrieverStartTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.query: str | None = None

        def on_retriever_start(
            self,
            serialized: dict[str, Any] | None,
            query: str,
            **kwargs: Any,
        ) -> None:
            self.query = query

    tracker = RetrieverStartTracker()
    manager = CallbackManager(handlers=[tracker])

    run_manager = manager.on_retriever_start(
        serialized={},
        query="search query",
    )

    assert isinstance(run_manager, CallbackManagerForRetrieverRun)
    assert tracker.query == "search query"


async def test_async_callback_manager_is_async() -> None:
    """Test AsyncCallbackManager.is_async property."""
    manager = AsyncCallbackManager(handlers=[])
    assert manager.is_async is True


async def test_async_callback_manager_on_llm_start() -> None:
    """Test AsyncCallbackManager.on_llm_start."""

    class AsyncLLMStartTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.called = False

        async def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            **kwargs: Any,
        ) -> None:
            self.called = True

    tracker = AsyncLLMStartTracker()
    manager = AsyncCallbackManager(handlers=[tracker])

    run_managers = await manager.on_llm_start(
        serialized={},
        prompts=["Test prompt"],
    )

    assert len(run_managers) == 1
    assert isinstance(run_managers[0], AsyncCallbackManagerForLLMRun)
    assert tracker.called is True


async def test_async_callback_manager_on_chain_start() -> None:
    """Test AsyncCallbackManager.on_chain_start."""

    class AsyncChainStartTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.called = False

        async def on_chain_start(
            self,
            serialized: dict[str, Any] | None,
            inputs: dict[str, Any] | Any,
            **kwargs: Any,
        ) -> None:
            self.called = True

    tracker = AsyncChainStartTracker()
    manager = AsyncCallbackManager(handlers=[tracker])

    run_manager = await manager.on_chain_start(
        serialized={},
        inputs={"input": "test"},
    )

    assert isinstance(run_manager, AsyncCallbackManagerForChainRun)
    assert tracker.called is True
