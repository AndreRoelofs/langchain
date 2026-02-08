"""Snapshot tests for base callback handlers, mixins, and manager classes.

These tests capture the current behavior of the base callback module
to detect unintended changes.
"""

import inspect
from typing import Any
from uuid import uuid4

import pytest

from langchain_core.callbacks.base import (
    AsyncCallbackHandler,
    BaseCallbackHandler,
    BaseCallbackManager,
    CallbackManagerMixin,
    Callbacks,
    ChainManagerMixin,
    LLMManagerMixin,
    RetrieverManagerMixin,
    RunManagerMixin,
    ToolManagerMixin,
)
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Callbacks type alias
# ---------------------------------------------------------------------------


class TestCallbacksTypeAlias:
    """Verify the Callbacks type alias accepts the documented types."""

    def test_callbacks_accepts_none(self) -> None:
        value: Callbacks = None
        assert value is None

    def test_callbacks_accepts_handler_list(self) -> None:
        value: Callbacks = [BaseCallbackHandler()]
        assert isinstance(value, list)

    def test_callbacks_accepts_manager(self) -> None:
        value: Callbacks = BaseCallbackManager(handlers=[])
        assert isinstance(value, BaseCallbackManager)


# ---------------------------------------------------------------------------
# Mixin classes – verify signatures and default returns
# ---------------------------------------------------------------------------


class TestRetrieverManagerMixin:
    """Tests for RetrieverManagerMixin."""

    def test_on_retriever_error_returns_none(self) -> None:
        mixin = RetrieverManagerMixin()
        result = mixin.on_retriever_error(
            ValueError("err"), run_id=uuid4(), parent_run_id=uuid4()
        )
        assert result is None

    def test_on_retriever_end_returns_none(self) -> None:
        mixin = RetrieverManagerMixin()
        docs = [Document(page_content="doc1")]
        result = mixin.on_retriever_end(docs, run_id=uuid4(), parent_run_id=None)
        assert result is None

    def test_on_retriever_end_accepts_empty_documents(self) -> None:
        mixin = RetrieverManagerMixin()
        result = mixin.on_retriever_end([], run_id=uuid4())
        assert result is None

    def test_on_retriever_error_accepts_base_exception(self) -> None:
        mixin = RetrieverManagerMixin()
        result = mixin.on_retriever_error(KeyboardInterrupt(), run_id=uuid4())
        assert result is None


class TestLLMManagerMixin:
    """Tests for LLMManagerMixin."""

    def test_on_llm_new_token_returns_none(self) -> None:
        mixin = LLMManagerMixin()
        result = mixin.on_llm_new_token("tok", run_id=uuid4())
        assert result is None

    def test_on_llm_new_token_with_chunk(self) -> None:
        mixin = LLMManagerMixin()
        result = mixin.on_llm_new_token(
            "tok", chunk=None, run_id=uuid4(), parent_run_id=uuid4()
        )
        assert result is None

    def test_on_llm_end_returns_none(self) -> None:
        mixin = LLMManagerMixin()
        result = mixin.on_llm_end(
            response=None,
            run_id=uuid4(),  # type: ignore[arg-type]
        )
        assert result is None

    def test_on_llm_error_returns_none(self) -> None:
        mixin = LLMManagerMixin()
        result = mixin.on_llm_error(RuntimeError("err"), run_id=uuid4())
        assert result is None

    def test_on_llm_error_with_parent_run_id(self) -> None:
        mixin = LLMManagerMixin()
        parent = uuid4()
        result = mixin.on_llm_error(
            RuntimeError("err"), run_id=uuid4(), parent_run_id=parent
        )
        assert result is None


class TestChainManagerMixin:
    """Tests for ChainManagerMixin."""

    def test_on_chain_end_returns_none(self) -> None:
        mixin = ChainManagerMixin()
        result = mixin.on_chain_end({"out": "val"}, run_id=uuid4())
        assert result is None

    def test_on_chain_error_returns_none(self) -> None:
        mixin = ChainManagerMixin()
        result = mixin.on_chain_error(ValueError("err"), run_id=uuid4())
        assert result is None

    def test_on_agent_action_returns_none(self) -> None:
        from langchain_core.agents import AgentAction

        mixin = ChainManagerMixin()
        action = AgentAction(tool="t", tool_input="i", log="l")
        result = mixin.on_agent_action(action, run_id=uuid4())
        assert result is None

    def test_on_agent_finish_returns_none(self) -> None:
        from langchain_core.agents import AgentFinish

        mixin = ChainManagerMixin()
        finish = AgentFinish(return_values={"out": "v"}, log="done")
        result = mixin.on_agent_finish(finish, run_id=uuid4())
        assert result is None

    def test_on_chain_end_with_kwargs(self) -> None:
        mixin = ChainManagerMixin()
        result = mixin.on_chain_end(
            {"k": "v"}, run_id=uuid4(), parent_run_id=uuid4(), extra="data"
        )
        assert result is None


class TestToolManagerMixin:
    """Tests for ToolManagerMixin."""

    def test_on_tool_end_returns_none(self) -> None:
        mixin = ToolManagerMixin()
        result = mixin.on_tool_end("output", run_id=uuid4())
        assert result is None

    def test_on_tool_error_returns_none(self) -> None:
        mixin = ToolManagerMixin()
        result = mixin.on_tool_error(TypeError("err"), run_id=uuid4())
        assert result is None

    def test_on_tool_end_accepts_any_output(self) -> None:
        mixin = ToolManagerMixin()
        for output in [42, {"key": "val"}, [1, 2], None]:
            result = mixin.on_tool_end(output, run_id=uuid4())
            assert result is None


class TestRunManagerMixin:
    """Tests for RunManagerMixin."""

    def test_on_text_returns_none(self) -> None:
        mixin = RunManagerMixin()
        result = mixin.on_text("hello", run_id=uuid4())
        assert result is None

    def test_on_retry_returns_none(self) -> None:
        mixin = RunManagerMixin()
        # Pass a mock-like retry_state
        result = mixin.on_retry(None, run_id=uuid4())  # type: ignore[arg-type]
        assert result is None

    def test_on_custom_event_returns_none(self) -> None:
        mixin = RunManagerMixin()
        result = mixin.on_custom_event("my_event", {"data": 1}, run_id=uuid4())
        assert result is None

    def test_on_custom_event_with_tags_and_metadata(self) -> None:
        mixin = RunManagerMixin()
        result = mixin.on_custom_event(
            "evt",
            {"x": 1},
            run_id=uuid4(),
            tags=["t1"],
            metadata={"m": "v"},
        )
        assert result is None


class TestCallbackManagerMixin:
    """Tests for CallbackManagerMixin."""

    def test_on_llm_start_returns_none(self) -> None:
        mixin = CallbackManagerMixin()
        result = mixin.on_llm_start(
            serialized={"id": ["test"]},
            prompts=["hello"],
            run_id=uuid4(),
        )
        assert result is None

    def test_on_chat_model_start_raises_not_implemented(self) -> None:
        mixin = CallbackManagerMixin()
        with pytest.raises(NotImplementedError, match="does not implement"):
            mixin.on_chat_model_start(serialized={}, messages=[], run_id=uuid4())

    def test_on_retriever_start_returns_none(self) -> None:
        mixin = CallbackManagerMixin()
        result = mixin.on_retriever_start(serialized={}, query="q", run_id=uuid4())
        assert result is None

    def test_on_chain_start_returns_none(self) -> None:
        mixin = CallbackManagerMixin()
        result = mixin.on_chain_start(serialized={}, inputs={"k": "v"}, run_id=uuid4())
        assert result is None

    def test_on_tool_start_returns_none(self) -> None:
        mixin = CallbackManagerMixin()
        result = mixin.on_tool_start(serialized={}, input_str="inp", run_id=uuid4())
        assert result is None

    def test_on_tool_start_with_inputs(self) -> None:
        mixin = CallbackManagerMixin()
        result = mixin.on_tool_start(
            serialized={},
            input_str="inp",
            run_id=uuid4(),
            inputs={"arg": "val"},
        )
        assert result is None

    def test_on_llm_start_with_tags_and_metadata(self) -> None:
        mixin = CallbackManagerMixin()
        result = mixin.on_llm_start(
            serialized={},
            prompts=["p"],
            run_id=uuid4(),
            tags=["t"],
            metadata={"k": "v"},
        )
        assert result is None


# ---------------------------------------------------------------------------
# BaseCallbackHandler – property and inheritance tests
# ---------------------------------------------------------------------------


class TestBaseCallbackHandler:
    """Tests for BaseCallbackHandler."""

    def test_inherits_all_mixins(self) -> None:
        handler = BaseCallbackHandler()
        assert isinstance(handler, LLMManagerMixin)
        assert isinstance(handler, ChainManagerMixin)
        assert isinstance(handler, ToolManagerMixin)
        assert isinstance(handler, RetrieverManagerMixin)
        assert isinstance(handler, CallbackManagerMixin)
        assert isinstance(handler, RunManagerMixin)

    def test_default_ignore_properties(self) -> None:
        handler = BaseCallbackHandler()
        assert handler.ignore_llm is False
        assert handler.ignore_retry is False
        assert handler.ignore_chain is False
        assert handler.ignore_agent is False
        assert handler.ignore_retriever is False
        assert handler.ignore_chat_model is False
        assert handler.ignore_custom_event is False

    def test_default_flags(self) -> None:
        handler = BaseCallbackHandler()
        assert handler.raise_error is False
        assert handler.run_inline is False

    def test_flags_are_mutable(self) -> None:
        handler = BaseCallbackHandler()
        handler.raise_error = True
        handler.run_inline = True
        assert handler.raise_error is True
        assert handler.run_inline is True

    def test_custom_ignore_overrides(self) -> None:
        class IgnoreAllHandler(BaseCallbackHandler):
            @property
            def ignore_llm(self) -> bool:
                return True

            @property
            def ignore_chain(self) -> bool:
                return True

            @property
            def ignore_agent(self) -> bool:
                return True

            @property
            def ignore_retriever(self) -> bool:
                return True

            @property
            def ignore_chat_model(self) -> bool:
                return True

            @property
            def ignore_retry(self) -> bool:
                return True

            @property
            def ignore_custom_event(self) -> bool:
                return True

        handler = IgnoreAllHandler()
        assert handler.ignore_llm is True
        assert handler.ignore_chain is True
        assert handler.ignore_agent is True
        assert handler.ignore_retriever is True
        assert handler.ignore_chat_model is True
        assert handler.ignore_retry is True
        assert handler.ignore_custom_event is True

    def test_on_chat_model_start_raises_not_implemented(self) -> None:
        handler = BaseCallbackHandler()
        with pytest.raises(NotImplementedError) as exc_info:
            handler.on_chat_model_start(serialized={}, messages=[], run_id=uuid4())
        assert "BaseCallbackHandler" in str(exc_info.value)


# ---------------------------------------------------------------------------
# AsyncCallbackHandler – async method verification
# ---------------------------------------------------------------------------


class TestAsyncCallbackHandler:
    """Tests for AsyncCallbackHandler."""

    def test_inherits_base(self) -> None:
        handler = AsyncCallbackHandler()
        assert isinstance(handler, BaseCallbackHandler)

    def test_all_callback_methods_are_coroutines(self) -> None:
        handler = AsyncCallbackHandler()
        async_methods = [
            "on_llm_start",
            "on_chat_model_start",
            "on_llm_new_token",
            "on_llm_end",
            "on_llm_error",
            "on_chain_start",
            "on_chain_end",
            "on_chain_error",
            "on_tool_start",
            "on_tool_end",
            "on_tool_error",
            "on_text",
            "on_retry",
            "on_agent_action",
            "on_agent_finish",
            "on_retriever_start",
            "on_retriever_end",
            "on_retriever_error",
            "on_custom_event",
        ]
        for method_name in async_methods:
            method = getattr(handler, method_name)
            assert inspect.iscoroutinefunction(method), (
                f"{method_name} should be a coroutine function"
            )

    async def test_on_chat_model_start_raises_not_implemented(self) -> None:
        handler = AsyncCallbackHandler()
        with pytest.raises(NotImplementedError) as exc_info:
            await handler.on_chat_model_start(
                serialized={}, messages=[], run_id=uuid4()
            )
        assert "AsyncCallbackHandler" in str(exc_info.value)

    async def test_all_no_op_methods_callable(self) -> None:
        """Verify every async no-op method runs without error."""
        handler = AsyncCallbackHandler()
        rid = uuid4()

        await handler.on_llm_start({}, ["p"], run_id=rid)
        await handler.on_llm_new_token("t", run_id=rid)
        await handler.on_llm_end(None, run_id=rid)  # type: ignore[arg-type]
        await handler.on_llm_error(ValueError(), run_id=rid)
        await handler.on_chain_start({}, {}, run_id=rid)
        await handler.on_chain_end({}, run_id=rid)
        await handler.on_chain_error(ValueError(), run_id=rid)
        await handler.on_tool_start({}, "inp", run_id=rid)
        await handler.on_tool_end("out", run_id=rid)
        await handler.on_tool_error(ValueError(), run_id=rid)
        await handler.on_text("txt", run_id=rid)
        await handler.on_retry(None, run_id=rid)  # type: ignore[arg-type]
        await handler.on_agent_action(None, run_id=rid)  # type: ignore[arg-type]
        await handler.on_agent_finish(None, run_id=rid)  # type: ignore[arg-type]
        await handler.on_retriever_start({}, "q", run_id=rid)
        await handler.on_retriever_end([], run_id=rid)
        await handler.on_retriever_error(ValueError(), run_id=rid)
        await handler.on_custom_event("evt", {}, run_id=rid)

    async def test_custom_async_handler_subclass(self) -> None:
        """Custom async handler can override methods and capture data."""

        class MyHandler(AsyncCallbackHandler):
            def __init__(self) -> None:
                self.captured: list[str] = []

            async def on_llm_start(
                self,
                serialized: dict[str, Any],
                prompts: list[str],
                **kwargs: Any,
            ) -> None:
                self.captured.extend(prompts)

        h = MyHandler()
        await h.on_llm_start({}, ["a", "b"], run_id=uuid4())
        assert h.captured == ["a", "b"]


# ---------------------------------------------------------------------------
# BaseCallbackManager – comprehensive manager operations
# ---------------------------------------------------------------------------


class TestBaseCallbackManager:
    """Tests for BaseCallbackManager."""

    def test_default_initialization(self) -> None:
        mgr = BaseCallbackManager(handlers=[])
        assert mgr.handlers == []
        assert mgr.inheritable_handlers == []
        assert mgr.parent_run_id is None
        assert mgr.tags == []
        assert mgr.inheritable_tags == []
        assert mgr.metadata == {}
        assert mgr.inheritable_metadata == {}

    def test_is_async_false(self) -> None:
        mgr = BaseCallbackManager(handlers=[])
        assert mgr.is_async is False

    # -- Handler management --

    def test_add_handler_inherit_true(self) -> None:
        h = BaseCallbackHandler()
        mgr = BaseCallbackManager(handlers=[])
        mgr.add_handler(h, inherit=True)
        assert h in mgr.handlers
        assert h in mgr.inheritable_handlers

    def test_add_handler_inherit_false(self) -> None:
        h = BaseCallbackHandler()
        mgr = BaseCallbackManager(handlers=[])
        mgr.add_handler(h, inherit=False)
        assert h in mgr.handlers
        assert h not in mgr.inheritable_handlers

    def test_add_handler_no_duplicates(self) -> None:
        h = BaseCallbackHandler()
        mgr = BaseCallbackManager(handlers=[])
        mgr.add_handler(h)
        mgr.add_handler(h)
        assert mgr.handlers.count(h) == 1
        assert mgr.inheritable_handlers.count(h) == 1

    def test_remove_handler_from_both_lists(self) -> None:
        h = BaseCallbackHandler()
        mgr = BaseCallbackManager(handlers=[h], inheritable_handlers=[h])
        mgr.remove_handler(h)
        assert h not in mgr.handlers
        assert h not in mgr.inheritable_handlers

    def test_remove_handler_only_in_handlers(self) -> None:
        h = BaseCallbackHandler()
        mgr = BaseCallbackManager(handlers=[h])
        mgr.remove_handler(h)
        assert h not in mgr.handlers

    def test_remove_handler_not_present(self) -> None:
        """Removing a handler that doesn't exist should not raise."""
        h = BaseCallbackHandler()
        mgr = BaseCallbackManager(handlers=[])
        mgr.remove_handler(h)  # should not raise

    def test_set_handlers_replaces_all(self) -> None:
        h1 = BaseCallbackHandler()
        h2 = BaseCallbackHandler()
        mgr = BaseCallbackManager(handlers=[h1], inheritable_handlers=[h1])
        mgr.set_handlers([h2])
        assert h1 not in mgr.handlers
        assert h2 in mgr.handlers
        assert h2 in mgr.inheritable_handlers

    def test_set_handlers_no_inherit(self) -> None:
        h = BaseCallbackHandler()
        mgr = BaseCallbackManager(handlers=[])
        mgr.set_handlers([h], inherit=False)
        assert h in mgr.handlers
        assert h not in mgr.inheritable_handlers

    def test_set_handler_single(self) -> None:
        h1 = BaseCallbackHandler()
        h2 = BaseCallbackHandler()
        mgr = BaseCallbackManager(handlers=[h1])
        mgr.set_handler(h2)
        assert mgr.handlers == [h2]
        assert mgr.inheritable_handlers == [h2]

    # -- Tag management --

    def test_add_tags_inherit_true(self) -> None:
        mgr = BaseCallbackManager(handlers=[])
        mgr.add_tags(["a", "b"], inherit=True)
        assert "a" in mgr.tags
        assert "b" in mgr.tags
        assert "a" in mgr.inheritable_tags
        assert "b" in mgr.inheritable_tags

    def test_add_tags_inherit_false(self) -> None:
        mgr = BaseCallbackManager(handlers=[])
        mgr.add_tags(["x"], inherit=False)
        assert "x" in mgr.tags
        assert "x" not in mgr.inheritable_tags

    def test_add_tags_deduplicates_existing(self) -> None:
        mgr = BaseCallbackManager(handlers=[], tags=["dup"])
        mgr.add_tags(["dup"])
        assert mgr.tags.count("dup") == 1

    def test_remove_tags(self) -> None:
        mgr = BaseCallbackManager(
            handlers=[],
            tags=["a", "b"],
            inheritable_tags=["a"],
        )
        mgr.remove_tags(["a"])
        assert "a" not in mgr.tags
        assert "a" not in mgr.inheritable_tags
        assert "b" in mgr.tags

    def test_remove_tags_not_present(self) -> None:
        """Removing tags that don't exist should not raise."""
        mgr = BaseCallbackManager(handlers=[])
        mgr.remove_tags(["nonexistent"])  # should not raise

    # -- Metadata management --

    def test_add_metadata_inherit_true(self) -> None:
        mgr = BaseCallbackManager(handlers=[])
        mgr.add_metadata({"k": "v"}, inherit=True)
        assert mgr.metadata["k"] == "v"
        assert mgr.inheritable_metadata["k"] == "v"

    def test_add_metadata_inherit_false(self) -> None:
        mgr = BaseCallbackManager(handlers=[])
        mgr.add_metadata({"k": "v"}, inherit=False)
        assert mgr.metadata["k"] == "v"
        assert "k" not in mgr.inheritable_metadata

    def test_add_metadata_overwrites(self) -> None:
        mgr = BaseCallbackManager(
            handlers=[], metadata={"k": "old"}, inheritable_metadata={"k": "old"}
        )
        mgr.add_metadata({"k": "new"})
        assert mgr.metadata["k"] == "new"
        assert mgr.inheritable_metadata["k"] == "new"

    def test_remove_metadata(self) -> None:
        mgr = BaseCallbackManager(
            handlers=[],
            metadata={"a": 1, "b": 2},
            inheritable_metadata={"a": 1},
        )
        mgr.remove_metadata(["a"])
        assert "a" not in mgr.metadata
        assert "a" not in mgr.inheritable_metadata
        assert "b" in mgr.metadata

    def test_remove_metadata_not_present(self) -> None:
        """Removing metadata that doesn't exist should not raise."""
        mgr = BaseCallbackManager(handlers=[])
        mgr.remove_metadata(["nonexistent"])  # should not raise

    # -- Copy --

    def test_copy_creates_independent_lists(self) -> None:
        h = BaseCallbackHandler()
        mgr = BaseCallbackManager(
            handlers=[h],
            inheritable_handlers=[h],
            tags=["t"],
            inheritable_tags=["it"],
            metadata={"k": "v"},
            inheritable_metadata={"ik": "iv"},
        )
        cp = mgr.copy()

        # Same contents
        assert cp.handlers == mgr.handlers
        assert cp.tags == mgr.tags
        assert cp.metadata == mgr.metadata

        # But different list/dict objects
        assert cp.handlers is not mgr.handlers
        assert cp.inheritable_handlers is not mgr.inheritable_handlers
        assert cp.tags is not mgr.tags
        assert cp.inheritable_tags is not mgr.inheritable_tags
        assert cp.metadata is not mgr.metadata
        assert cp.inheritable_metadata is not mgr.inheritable_metadata

    def test_copy_preserves_parent_run_id(self) -> None:
        pid = uuid4()
        mgr = BaseCallbackManager(handlers=[], parent_run_id=pid)
        cp = mgr.copy()
        assert cp.parent_run_id == pid

    # -- Merge --

    def test_merge_combines_tags(self) -> None:
        m1 = BaseCallbackManager(handlers=[], tags=["a"])
        m2 = BaseCallbackManager(handlers=[], tags=["b"])
        merged = m1.merge(m2)
        assert "a" in merged.tags
        assert "b" in merged.tags

    def test_merge_combines_metadata(self) -> None:
        m1 = BaseCallbackManager(handlers=[], metadata={"k1": "v1"})
        m2 = BaseCallbackManager(handlers=[], metadata={"k2": "v2"})
        merged = m1.merge(m2)
        assert merged.metadata["k1"] == "v1"
        assert merged.metadata["k2"] == "v2"

    def test_merge_other_metadata_wins_on_conflict(self) -> None:
        m1 = BaseCallbackManager(handlers=[], metadata={"k": "v1"})
        m2 = BaseCallbackManager(handlers=[], metadata={"k": "v2"})
        merged = m1.merge(m2)
        assert merged.metadata["k"] == "v2"

    def test_merge_combines_handlers(self) -> None:
        h1 = BaseCallbackHandler()
        h2 = BaseCallbackHandler()
        m1 = BaseCallbackManager(handlers=[h1])
        m2 = BaseCallbackManager(handlers=[h2])
        merged = m1.merge(m2)
        assert h1 in merged.handlers
        assert h2 in merged.handlers

    def test_merge_preserves_parent_run_id_from_self(self) -> None:
        pid = uuid4()
        m1 = BaseCallbackManager(handlers=[], parent_run_id=pid)
        m2 = BaseCallbackManager(handlers=[])
        merged = m1.merge(m2)
        assert merged.parent_run_id == pid

    def test_merge_uses_other_parent_run_id_when_self_is_none(self) -> None:
        pid = uuid4()
        m1 = BaseCallbackManager(handlers=[])
        m2 = BaseCallbackManager(handlers=[], parent_run_id=pid)
        merged = m1.merge(m2)
        assert merged.parent_run_id == pid

    def test_merge_deduplicates_tags(self) -> None:
        m1 = BaseCallbackManager(handlers=[], tags=["shared"])
        m2 = BaseCallbackManager(handlers=[], tags=["shared"])
        merged = m1.merge(m2)
        assert merged.tags.count("shared") == 1

    def test_merge_combines_inheritable_tags(self) -> None:
        m1 = BaseCallbackManager(handlers=[], inheritable_tags=["a"])
        m2 = BaseCallbackManager(handlers=[], inheritable_tags=["b"])
        merged = m1.merge(m2)
        assert "a" in merged.inheritable_tags
        assert "b" in merged.inheritable_tags
