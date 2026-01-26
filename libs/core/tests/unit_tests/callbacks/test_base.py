"""Unit tests for base callback handlers and mixins."""

from typing import Any
from uuid import UUID, uuid4

import pytest

from langchain_core.callbacks.base import (
    AsyncCallbackHandler,
    BaseCallbackHandler,
    BaseCallbackManager,
    CallbackManagerMixin,
    ChainManagerMixin,
    LLMManagerMixin,
    RetrieverManagerMixin,
    RunManagerMixin,
    ToolManagerMixin,
)
from langchain_core.documents import Document


def test_base_callback_handler_defaults() -> None:
    """Test default values of BaseCallbackHandler."""
    handler = BaseCallbackHandler()

    assert handler.raise_error is False
    assert handler.run_inline is False
    assert handler.ignore_llm is False
    assert handler.ignore_retry is False
    assert handler.ignore_chain is False
    assert handler.ignore_agent is False
    assert handler.ignore_retriever is False
    assert handler.ignore_chat_model is False
    assert handler.ignore_custom_event is False


def test_base_callback_handler_custom_properties() -> None:
    """Test that ignore properties can be customized."""

    class CustomHandler(BaseCallbackHandler):
        @property
        def ignore_llm(self) -> bool:
            return True

        @property
        def ignore_chain(self) -> bool:
            return True

    handler = CustomHandler()
    assert handler.ignore_llm is True
    assert handler.ignore_chain is True
    assert handler.ignore_agent is False


def test_base_callback_handler_raise_error() -> None:
    """Test raise_error flag."""
    handler = BaseCallbackHandler()
    handler.raise_error = True
    assert handler.raise_error is True


def test_base_callback_handler_run_inline() -> None:
    """Test run_inline flag."""
    handler = BaseCallbackHandler()
    handler.run_inline = True
    assert handler.run_inline is True


def test_callback_manager_mixin_on_chat_model_start_not_implemented() -> None:
    """Test that on_chat_model_start raises NotImplementedError by default."""

    class TestMixin(CallbackManagerMixin):
        pass

    mixin = TestMixin()
    run_id = uuid4()

    with pytest.raises(NotImplementedError) as exc_info:
        mixin.on_chat_model_start(
            serialized={},
            messages=[],
            run_id=run_id,
        )

    assert "does not implement `on_chat_model_start`" in str(exc_info.value)


async def test_async_callback_handler_on_chat_model_start_not_implemented() -> None:
    """Test that async on_chat_model_start raises NotImplementedError by default."""
    handler = AsyncCallbackHandler()
    run_id = uuid4()

    with pytest.raises(NotImplementedError) as exc_info:
        await handler.on_chat_model_start(
            serialized={},
            messages=[],
            run_id=run_id,
        )

    assert "does not implement `on_chat_model_start`" in str(exc_info.value)


def test_llm_manager_mixin() -> None:
    """Test LLMManagerMixin methods are callable."""

    class TestLLMManager(LLMManagerMixin):
        pass

    manager = TestLLMManager()
    run_id = uuid4()

    # These should not raise - they're no-op by default
    result = manager.on_llm_new_token("token", run_id=run_id)
    assert result is None

    result = manager.on_llm_end(response={"generations": []}, run_id=run_id)  # type: ignore[arg-type]
    assert result is None

    result = manager.on_llm_error(error=ValueError("test"), run_id=run_id)
    assert result is None


def test_chain_manager_mixin() -> None:
    """Test ChainManagerMixin methods are callable."""

    class TestChainManager(ChainManagerMixin):
        pass

    manager = TestChainManager()
    run_id = uuid4()

    # These should not raise - they're no-op by default
    result = manager.on_chain_end(outputs={"result": "test"}, run_id=run_id)
    assert result is None

    result = manager.on_chain_error(error=ValueError("test"), run_id=run_id)
    assert result is None


def test_tool_manager_mixin() -> None:
    """Test ToolManagerMixin methods are callable."""

    class TestToolManager(ToolManagerMixin):
        pass

    manager = TestToolManager()
    run_id = uuid4()

    result = manager.on_tool_end(output="result", run_id=run_id)
    assert result is None

    result = manager.on_tool_error(error=ValueError("test"), run_id=run_id)
    assert result is None


def test_retriever_manager_mixin() -> None:
    """Test RetrieverManagerMixin methods are callable."""

    class TestRetrieverManager(RetrieverManagerMixin):
        pass

    manager = TestRetrieverManager()
    run_id = uuid4()

    documents = [Document(page_content="test")]

    result = manager.on_retriever_end(documents=documents, run_id=run_id)
    assert result is None

    result = manager.on_retriever_error(error=ValueError("test"), run_id=run_id)
    assert result is None


def test_run_manager_mixin() -> None:
    """Test RunManagerMixin methods are callable."""

    class TestRunManager(RunManagerMixin):
        pass

    manager = TestRunManager()
    run_id = uuid4()

    result = manager.on_text(text="test", run_id=run_id)
    assert result is None


def test_base_callback_manager_initialization() -> None:
    """Test BaseCallbackManager initialization."""
    handler1 = BaseCallbackHandler()
    handler2 = BaseCallbackHandler()

    manager = BaseCallbackManager(
        handlers=[handler1],
        inheritable_handlers=[handler2],
        tags=["tag1"],
        inheritable_tags=["tag2"],
        metadata={"key": "value"},
        inheritable_metadata={"key2": "value2"},
    )

    assert len(manager.handlers) == 1
    assert handler1 in manager.handlers
    assert len(manager.inheritable_handlers) == 1
    assert handler2 in manager.inheritable_handlers
    assert "tag1" in manager.tags
    assert "tag2" in manager.inheritable_tags
    assert manager.metadata["key"] == "value"
    assert manager.inheritable_metadata["key2"] == "value2"


def test_base_callback_manager_copy() -> None:
    """Test BaseCallbackManager.copy()."""
    handler = BaseCallbackHandler()
    manager = BaseCallbackManager(
        handlers=[handler],
        inheritable_handlers=[handler],
        tags=["tag1"],
        metadata={"key": "value"},
    )

    copied = manager.copy()

    assert copied is not manager
    assert copied.handlers == manager.handlers
    assert copied.inheritable_handlers == manager.inheritable_handlers
    assert copied.tags == manager.tags
    assert copied.metadata == manager.metadata

    # Verify it's a shallow copy (lists are new, but contents are same)
    assert copied.handlers is not manager.handlers
    assert copied.handlers[0] is manager.handlers[0]


def test_base_callback_manager_add_handler() -> None:
    """Test adding handlers to callback manager."""
    handler1 = BaseCallbackHandler()
    handler2 = BaseCallbackHandler()

    manager = BaseCallbackManager(handlers=[])

    manager.add_handler(handler1, inherit=True)
    assert handler1 in manager.handlers
    assert handler1 in manager.inheritable_handlers

    manager.add_handler(handler2, inherit=False)
    assert handler2 in manager.handlers
    assert handler2 not in manager.inheritable_handlers

    # Adding same handler again should not duplicate
    manager.add_handler(handler1, inherit=True)
    assert manager.handlers.count(handler1) == 1


def test_base_callback_manager_set_handlers() -> None:
    """Test set_handlers replaces all handlers."""
    handler1 = BaseCallbackHandler()
    handler2 = BaseCallbackHandler()
    handler3 = BaseCallbackHandler()

    manager = BaseCallbackManager(handlers=[handler1], inheritable_handlers=[handler1])

    manager.set_handlers([handler2, handler3], inherit=True)

    assert handler1 not in manager.handlers
    assert handler2 in manager.handlers
    assert handler3 in manager.handlers
    assert handler2 in manager.inheritable_handlers
    assert handler3 in manager.inheritable_handlers


def test_base_callback_manager_set_handler() -> None:
    """Test set_handler sets a single handler."""
    handler1 = BaseCallbackHandler()
    handler2 = BaseCallbackHandler()

    manager = BaseCallbackManager(handlers=[handler1])
    manager.set_handler(handler2)

    assert handler1 not in manager.handlers
    assert handler2 in manager.handlers


def test_base_callback_manager_add_tags() -> None:
    """Test adding tags to callback manager."""
    manager = BaseCallbackManager(handlers=[])

    manager.add_tags(["tag1", "tag2"], inherit=True)
    assert "tag1" in manager.tags
    assert "tag2" in manager.tags
    assert "tag1" in manager.inheritable_tags
    assert "tag2" in manager.inheritable_tags

    manager.add_tags(["tag3"], inherit=False)
    assert "tag3" in manager.tags
    assert "tag3" not in manager.inheritable_tags


def test_base_callback_manager_add_tags_removes_duplicates() -> None:
    """Test that adding existing tag removes it first."""
    manager = BaseCallbackManager(handlers=[], tags=["tag1"])

    # Add tag1 again - should remove and re-add
    initial_count = len(manager.tags)
    manager.add_tags(["tag1"])

    # Should still have just one tag1
    assert manager.tags.count("tag1") == 1


def test_base_callback_manager_remove_tags() -> None:
    """Test removing tags from callback manager."""
    manager = BaseCallbackManager(
        handlers=[],
        tags=["tag1", "tag2"],
        inheritable_tags=["tag1", "tag2"],
    )

    manager.remove_tags(["tag1"])

    assert "tag1" not in manager.tags
    assert "tag2" in manager.tags
    assert "tag1" not in manager.inheritable_tags
    assert "tag2" in manager.inheritable_tags


def test_base_callback_manager_add_metadata() -> None:
    """Test adding metadata to callback manager."""
    manager = BaseCallbackManager(handlers=[])

    manager.add_metadata({"key1": "value1"}, inherit=True)
    assert manager.metadata["key1"] == "value1"
    assert manager.inheritable_metadata["key1"] == "value1"

    manager.add_metadata({"key2": "value2"}, inherit=False)
    assert manager.metadata["key2"] == "value2"
    assert "key2" not in manager.inheritable_metadata


def test_base_callback_manager_remove_metadata() -> None:
    """Test removing metadata from callback manager."""
    manager = BaseCallbackManager(
        handlers=[],
        metadata={"key1": "value1", "key2": "value2"},
        inheritable_metadata={"key1": "value1"},
    )

    manager.remove_metadata(["key1"])

    assert "key1" not in manager.metadata
    assert "key2" in manager.metadata
    assert "key1" not in manager.inheritable_metadata


def test_base_callback_manager_is_async() -> None:
    """Test is_async property returns False for base manager."""
    manager = BaseCallbackManager(handlers=[])
    assert manager.is_async is False


def test_async_callback_handler_all_methods_are_async() -> None:
    """Test that AsyncCallbackHandler methods are async."""
    import inspect

    handler = AsyncCallbackHandler()

    # Check key methods are coroutine functions
    assert inspect.iscoroutinefunction(handler.on_llm_start)
    assert inspect.iscoroutinefunction(handler.on_chat_model_start)
    assert inspect.iscoroutinefunction(handler.on_llm_new_token)
    assert inspect.iscoroutinefunction(handler.on_llm_end)
    assert inspect.iscoroutinefunction(handler.on_llm_error)
    assert inspect.iscoroutinefunction(handler.on_chain_start)
    assert inspect.iscoroutinefunction(handler.on_chain_end)
    assert inspect.iscoroutinefunction(handler.on_chain_error)
    assert inspect.iscoroutinefunction(handler.on_tool_start)
    assert inspect.iscoroutinefunction(handler.on_tool_end)
    assert inspect.iscoroutinefunction(handler.on_tool_error)
    assert inspect.iscoroutinefunction(handler.on_text)


async def test_async_callback_handler_methods_callable() -> None:
    """Test that async callback handler methods can be called."""
    handler = AsyncCallbackHandler()
    run_id = uuid4()

    # These should not raise - they're no-op by default
    await handler.on_llm_start(serialized={}, prompts=["test"], run_id=run_id)
    await handler.on_llm_new_token(token="test", run_id=run_id)
    await handler.on_llm_end(response={"generations": []}, run_id=run_id)  # type: ignore[arg-type]
    await handler.on_llm_error(error=ValueError("test"), run_id=run_id)
    await handler.on_chain_start(serialized={}, inputs={}, run_id=run_id)
    await handler.on_chain_end(outputs={}, run_id=run_id)
    await handler.on_chain_error(error=ValueError("test"), run_id=run_id)
    await handler.on_tool_start(serialized={}, input_str="test", run_id=run_id)
    await handler.on_tool_end(output="test", run_id=run_id)
    await handler.on_tool_error(error=ValueError("test"), run_id=run_id)
    await handler.on_text(text="test", run_id=run_id)
    await handler.on_retriever_start(serialized={}, query="test", run_id=run_id)
    await handler.on_retriever_end(documents=[], run_id=run_id)
    await handler.on_retriever_error(error=ValueError("test"), run_id=run_id)
    await handler.on_custom_event(name="test", data={}, run_id=run_id)


def test_callback_manager_with_parent_run_id() -> None:
    """Test callback manager with parent_run_id."""
    parent_run_id = uuid4()
    manager = BaseCallbackManager(handlers=[], parent_run_id=parent_run_id)

    assert manager.parent_run_id == parent_run_id
