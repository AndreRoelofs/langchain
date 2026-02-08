"""Snapshot tests for callback managers and event handling.

These tests capture the current behavior of the manager module to detect
unintended changes in event dispatching, ignore conditions, error handling,
run manager lifecycles, and chain group managers.
"""

from typing import Any
from uuid import UUID, uuid4

import pytest

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainGroup,
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForLLMRun,
    AsyncCallbackManagerForRetrieverRun,
    AsyncCallbackManagerForToolRun,
    AsyncParentRunManager,
    BaseRunManager,
    CallbackManager,
    CallbackManagerForChainGroup,
    CallbackManagerForChainRun,
    CallbackManagerForLLMRun,
    CallbackManagerForRetrieverRun,
    CallbackManagerForToolRun,
    ParentRunManager,
    RunManager,
    handle_event,
)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult

# ---------------------------------------------------------------------------
# Shared test handler that records all events
# ---------------------------------------------------------------------------


class RecordingHandler(BaseCallbackHandler):
    """Sync handler that records every callback invocation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.events: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.events.append((name, args, kwargs))

    def on_llm_start(self, serialized: Any, prompts: Any, **kw: Any) -> None:
        self._record("on_llm_start", serialized, prompts, **kw)

    def on_llm_new_token(self, token: str, **kw: Any) -> None:
        self._record("on_llm_new_token", token, **kw)

    def on_llm_end(self, response: Any, **kw: Any) -> None:
        self._record("on_llm_end", response, **kw)

    def on_llm_error(self, error: BaseException, **kw: Any) -> None:
        self._record("on_llm_error", error, **kw)

    def on_chain_start(self, serialized: Any, inputs: Any, **kw: Any) -> None:
        self._record("on_chain_start", serialized, inputs, **kw)

    def on_chain_end(self, outputs: Any, **kw: Any) -> None:
        self._record("on_chain_end", outputs, **kw)

    def on_chain_error(self, error: BaseException, **kw: Any) -> None:
        self._record("on_chain_error", error, **kw)

    def on_tool_start(self, serialized: Any, input_str: str, **kw: Any) -> None:
        self._record("on_tool_start", serialized, input_str, **kw)

    def on_tool_end(self, output: Any, **kw: Any) -> None:
        self._record("on_tool_end", output, **kw)

    def on_tool_error(self, error: BaseException, **kw: Any) -> None:
        self._record("on_tool_error", error, **kw)

    def on_retriever_start(self, serialized: Any, query: str, **kw: Any) -> None:
        self._record("on_retriever_start", serialized, query, **kw)

    def on_retriever_end(self, documents: Any, **kw: Any) -> None:
        self._record("on_retriever_end", documents, **kw)

    def on_retriever_error(self, error: BaseException, **kw: Any) -> None:
        self._record("on_retriever_error", error, **kw)

    def on_text(self, text: str, **kw: Any) -> None:
        self._record("on_text", text, **kw)

    def on_retry(self, retry_state: Any, **kw: Any) -> None:
        self._record("on_retry", retry_state, **kw)

    def on_agent_action(self, action: Any, **kw: Any) -> None:
        self._record("on_agent_action", action, **kw)

    def on_agent_finish(self, finish: Any, **kw: Any) -> None:
        self._record("on_agent_finish", finish, **kw)

    def on_custom_event(self, name: str, data: Any, **kw: Any) -> None:
        self._record("on_custom_event", name, data, **kw)

    def on_chat_model_start(self, serialized: Any, messages: Any, **kw: Any) -> None:
        self._record("on_chat_model_start", serialized, messages, **kw)


class AsyncRecordingHandler(AsyncCallbackHandler):
    """Async handler that records every callback invocation."""

    def __init__(self) -> None:
        self.events: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.events.append((name, args, kwargs))

    async def on_llm_start(self, serialized: Any, prompts: Any, **kw: Any) -> None:
        self._record("on_llm_start", serialized, prompts, **kw)

    async def on_llm_new_token(self, token: str, **kw: Any) -> None:
        self._record("on_llm_new_token", token, **kw)

    async def on_llm_end(self, response: Any, **kw: Any) -> None:
        self._record("on_llm_end", response, **kw)

    async def on_llm_error(self, error: BaseException, **kw: Any) -> None:
        self._record("on_llm_error", error, **kw)

    async def on_chain_start(self, serialized: Any, inputs: Any, **kw: Any) -> None:
        self._record("on_chain_start", serialized, inputs, **kw)

    async def on_chain_end(self, outputs: Any, **kw: Any) -> None:
        self._record("on_chain_end", outputs, **kw)

    async def on_chain_error(self, error: BaseException, **kw: Any) -> None:
        self._record("on_chain_error", error, **kw)

    async def on_tool_start(self, serialized: Any, input_str: str, **kw: Any) -> None:
        self._record("on_tool_start", serialized, input_str, **kw)

    async def on_tool_end(self, output: Any, **kw: Any) -> None:
        self._record("on_tool_end", output, **kw)

    async def on_tool_error(self, error: BaseException, **kw: Any) -> None:
        self._record("on_tool_error", error, **kw)

    async def on_retriever_start(self, serialized: Any, query: str, **kw: Any) -> None:
        self._record("on_retriever_start", serialized, query, **kw)

    async def on_retriever_end(self, documents: Any, **kw: Any) -> None:
        self._record("on_retriever_end", documents, **kw)

    async def on_retriever_error(self, error: BaseException, **kw: Any) -> None:
        self._record("on_retriever_error", error, **kw)

    async def on_text(self, text: str, **kw: Any) -> None:
        self._record("on_text", text, **kw)

    async def on_retry(self, retry_state: Any, **kw: Any) -> None:
        self._record("on_retry", retry_state, **kw)

    async def on_agent_action(self, action: Any, **kw: Any) -> None:
        self._record("on_agent_action", action, **kw)

    async def on_agent_finish(self, finish: Any, **kw: Any) -> None:
        self._record("on_agent_finish", finish, **kw)

    async def on_custom_event(self, name: str, data: Any, **kw: Any) -> None:
        self._record("on_custom_event", name, data, **kw)

    async def on_chat_model_start(
        self, serialized: Any, messages: Any, **kw: Any
    ) -> None:
        self._record("on_chat_model_start", serialized, messages, **kw)


# ---------------------------------------------------------------------------
# handle_event – sync event dispatching
# ---------------------------------------------------------------------------


class TestHandleEvent:
    """Tests for the sync handle_event function."""

    def test_dispatches_to_handler(self) -> None:
        rec = RecordingHandler()
        handle_event([rec], "on_text", None, "hello", run_id=uuid4())
        assert len(rec.events) == 1
        assert rec.events[0][0] == "on_text"

    def test_respects_ignore_condition(self) -> None:
        class IgnoreLLM(RecordingHandler):
            @property
            def ignore_llm(self) -> bool:
                return True

        rec = IgnoreLLM()
        handle_event(
            [rec], "on_llm_start", "ignore_llm", {}, ["prompt"], run_id=uuid4()
        )
        assert len(rec.events) == 0

    def test_ignore_condition_none_always_dispatches(self) -> None:
        rec = RecordingHandler()
        handle_event([rec], "on_text", None, "txt", run_id=uuid4())
        assert len(rec.events) == 1

    def test_multiple_handlers(self) -> None:
        r1 = RecordingHandler()
        r2 = RecordingHandler()
        handle_event([r1, r2], "on_text", None, "hello", run_id=uuid4())
        assert len(r1.events) == 1
        assert len(r2.events) == 1

    def test_handler_error_logged_not_raised(self) -> None:
        class FailHandler(BaseCallbackHandler):
            def on_text(self, text: str, **kwargs: Any) -> None:
                msg = "fail"
                raise RuntimeError(msg)

        handler = FailHandler()
        # Should not raise because raise_error is False
        handle_event([handler], "on_text", None, "x", run_id=uuid4())

    def test_handler_error_raised_when_raise_error_true(self) -> None:
        class FailHandler(BaseCallbackHandler):
            raise_error = True

            def on_text(self, text: str, **kwargs: Any) -> None:
                msg = "fail"
                raise RuntimeError(msg)

        handler = FailHandler()
        with pytest.raises(RuntimeError, match="fail"):
            handle_event([handler], "on_text", None, "x", run_id=uuid4())

    def test_chat_model_start_fallback_to_llm_start(self) -> None:
        """When on_chat_model_start raises NotImplementedError, falls back to on_llm_start."""
        rec = RecordingHandler()

        # Remove the on_chat_model_start override so the mixin raises NotImplementedError
        class FallbackHandler(BaseCallbackHandler):
            def __init__(self) -> None:
                self.called_with: list[str] = []

            def on_llm_start(
                self, serialized: Any, prompts: list[str], **kw: Any
            ) -> None:
                self.called_with.extend(prompts)

        handler = FallbackHandler()
        messages = [[HumanMessage(content="hi")]]
        handle_event(
            [handler],
            "on_chat_model_start",
            "ignore_chat_model",
            {},
            messages,
            run_id=uuid4(),
        )
        # Should have been called via on_llm_start with message strings
        assert len(handler.called_with) == 1

    def test_empty_handlers_no_error(self) -> None:
        handle_event([], "on_text", None, "hello", run_id=uuid4())


# ---------------------------------------------------------------------------
# RunManager – sync run manager
# ---------------------------------------------------------------------------


class TestRunManager:
    """Tests for sync RunManager."""

    def test_on_text_passes_run_id_and_parent_run_id(self) -> None:
        rec = RecordingHandler()
        run_id = uuid4()
        parent = uuid4()
        mgr = RunManager(
            run_id=run_id,
            handlers=[rec],
            inheritable_handlers=[],
            parent_run_id=parent,
            tags=["t1"],
        )
        mgr.on_text("hi")
        assert len(rec.events) == 1
        _, _, kw = rec.events[0]
        assert kw["run_id"] == run_id
        assert kw["parent_run_id"] == parent
        assert "t1" in kw["tags"]

    def test_on_retry_uses_ignore_retry(self) -> None:
        class IgnoreRetry(RecordingHandler):
            @property
            def ignore_retry(self) -> bool:
                return True

        rec = IgnoreRetry()
        mgr = RunManager(run_id=uuid4(), handlers=[rec], inheritable_handlers=[])
        mgr.on_retry(None)  # type: ignore[arg-type]
        assert len(rec.events) == 0

    def test_on_text_empty_handlers_no_error(self) -> None:
        mgr = RunManager(run_id=uuid4(), handlers=[], inheritable_handlers=[])
        mgr.on_text("test")  # should not raise

    def test_on_retry_empty_handlers_no_error(self) -> None:
        mgr = RunManager(run_id=uuid4(), handlers=[], inheritable_handlers=[])
        mgr.on_retry(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ParentRunManager – child creation
# ---------------------------------------------------------------------------


class TestParentRunManager:
    """Tests for sync ParentRunManager."""

    def test_get_child_inherits_handlers(self) -> None:
        h = BaseCallbackHandler()
        mgr = ParentRunManager(
            run_id=uuid4(),
            handlers=[h],
            inheritable_handlers=[h],
            inheritable_tags=["it"],
            inheritable_metadata={"ik": "iv"},
        )
        child = mgr.get_child()
        assert isinstance(child, CallbackManager)
        assert h in child.handlers
        assert h in child.inheritable_handlers
        assert "it" in child.tags
        assert "ik" in child.metadata

    def test_get_child_sets_parent_run_id(self) -> None:
        run_id = uuid4()
        mgr = ParentRunManager(run_id=run_id, handlers=[], inheritable_handlers=[])
        child = mgr.get_child()
        assert child.parent_run_id == run_id

    def test_get_child_tag_not_inheritable(self) -> None:
        mgr = ParentRunManager(run_id=uuid4(), handlers=[], inheritable_handlers=[])
        child = mgr.get_child(tag="local")
        assert "local" in child.tags
        assert "local" not in child.inheritable_tags

    def test_get_child_without_tag(self) -> None:
        mgr = ParentRunManager(run_id=uuid4(), handlers=[], inheritable_handlers=[])
        child = mgr.get_child()
        # Should work without extra tags
        assert isinstance(child, CallbackManager)


# ---------------------------------------------------------------------------
# CallbackManagerForLLMRun – LLM run lifecycle
# ---------------------------------------------------------------------------


class TestCallbackManagerForLLMRun:
    """Tests for sync LLM run manager."""

    def test_on_llm_new_token_respects_ignore_llm(self) -> None:
        class IgnoreLLM(RecordingHandler):
            @property
            def ignore_llm(self) -> bool:
                return True

        rec = IgnoreLLM()
        mgr = CallbackManagerForLLMRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        mgr.on_llm_new_token("tok")
        assert len(rec.events) == 0

    def test_on_llm_end_respects_ignore_llm(self) -> None:
        class IgnoreLLM(RecordingHandler):
            @property
            def ignore_llm(self) -> bool:
                return True

        rec = IgnoreLLM()
        mgr = CallbackManagerForLLMRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        mgr.on_llm_end(LLMResult(generations=[]))
        assert len(rec.events) == 0

    def test_on_llm_error_respects_ignore_llm(self) -> None:
        class IgnoreLLM(RecordingHandler):
            @property
            def ignore_llm(self) -> bool:
                return True

        rec = IgnoreLLM()
        mgr = CallbackManagerForLLMRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        mgr.on_llm_error(ValueError("err"))
        assert len(rec.events) == 0

    def test_on_llm_new_token_with_chunk(self) -> None:
        rec = RecordingHandler()
        mgr = CallbackManagerForLLMRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        mgr.on_llm_new_token("tok", chunk=None)
        assert len(rec.events) == 1

    def test_empty_handlers_noop(self) -> None:
        mgr = CallbackManagerForLLMRun(
            run_id=uuid4(), handlers=[], inheritable_handlers=[]
        )
        mgr.on_llm_new_token("tok")
        mgr.on_llm_end(LLMResult(generations=[]))
        mgr.on_llm_error(ValueError())


# ---------------------------------------------------------------------------
# CallbackManagerForChainRun – chain run lifecycle
# ---------------------------------------------------------------------------


class TestCallbackManagerForChainRun:
    """Tests for sync chain run manager."""

    def test_on_chain_end_respects_ignore_chain(self) -> None:
        class IgnoreChain(RecordingHandler):
            @property
            def ignore_chain(self) -> bool:
                return True

        rec = IgnoreChain()
        mgr = CallbackManagerForChainRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        mgr.on_chain_end({"out": "val"})
        assert len(rec.events) == 0

    def test_on_chain_error_respects_ignore_chain(self) -> None:
        class IgnoreChain(RecordingHandler):
            @property
            def ignore_chain(self) -> bool:
                return True

        rec = IgnoreChain()
        mgr = CallbackManagerForChainRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        mgr.on_chain_error(ValueError())
        assert len(rec.events) == 0

    def test_on_agent_action_respects_ignore_agent(self) -> None:
        class IgnoreAgent(RecordingHandler):
            @property
            def ignore_agent(self) -> bool:
                return True

        rec = IgnoreAgent()
        mgr = CallbackManagerForChainRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        action = AgentAction(tool="t", tool_input="i", log="l")
        mgr.on_agent_action(action)
        assert len(rec.events) == 0

    def test_on_agent_finish_respects_ignore_agent(self) -> None:
        class IgnoreAgent(RecordingHandler):
            @property
            def ignore_agent(self) -> bool:
                return True

        rec = IgnoreAgent()
        mgr = CallbackManagerForChainRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        finish = AgentFinish(return_values={}, log="done")
        mgr.on_agent_finish(finish)
        assert len(rec.events) == 0

    def test_on_agent_action_dispatches(self) -> None:
        rec = RecordingHandler()
        mgr = CallbackManagerForChainRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        action = AgentAction(tool="t", tool_input="i", log="l")
        mgr.on_agent_action(action)
        assert len(rec.events) == 1
        assert rec.events[0][0] == "on_agent_action"

    def test_on_agent_finish_dispatches(self) -> None:
        rec = RecordingHandler()
        mgr = CallbackManagerForChainRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        finish = AgentFinish(return_values={"out": "v"}, log="done")
        mgr.on_agent_finish(finish)
        assert len(rec.events) == 1
        assert rec.events[0][0] == "on_agent_finish"

    def test_empty_handlers_noop(self) -> None:
        mgr = CallbackManagerForChainRun(
            run_id=uuid4(), handlers=[], inheritable_handlers=[]
        )
        mgr.on_chain_end({})
        mgr.on_chain_error(ValueError())
        mgr.on_agent_action(AgentAction(tool="t", tool_input="i", log="l"))
        mgr.on_agent_finish(AgentFinish(return_values={}, log="d"))

    def test_is_parent_run_manager(self) -> None:
        """CallbackManagerForChainRun extends ParentRunManager."""
        mgr = CallbackManagerForChainRun(
            run_id=uuid4(), handlers=[], inheritable_handlers=[]
        )
        assert isinstance(mgr, ParentRunManager)
        child = mgr.get_child()
        assert isinstance(child, CallbackManager)


# ---------------------------------------------------------------------------
# CallbackManagerForToolRun – tool run lifecycle
# ---------------------------------------------------------------------------


class TestCallbackManagerForToolRun:
    """Tests for sync tool run manager."""

    def test_on_tool_end_respects_ignore_agent(self) -> None:
        class IgnoreAgent(RecordingHandler):
            @property
            def ignore_agent(self) -> bool:
                return True

        rec = IgnoreAgent()
        mgr = CallbackManagerForToolRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        mgr.on_tool_end("output")
        assert len(rec.events) == 0

    def test_on_tool_error_respects_ignore_agent(self) -> None:
        class IgnoreAgent(RecordingHandler):
            @property
            def ignore_agent(self) -> bool:
                return True

        rec = IgnoreAgent()
        mgr = CallbackManagerForToolRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        mgr.on_tool_error(ValueError())
        assert len(rec.events) == 0

    def test_empty_handlers_noop(self) -> None:
        mgr = CallbackManagerForToolRun(
            run_id=uuid4(), handlers=[], inheritable_handlers=[]
        )
        mgr.on_tool_end("out")
        mgr.on_tool_error(ValueError())

    def test_is_parent_run_manager(self) -> None:
        mgr = CallbackManagerForToolRun(
            run_id=uuid4(), handlers=[], inheritable_handlers=[]
        )
        assert isinstance(mgr, ParentRunManager)


# ---------------------------------------------------------------------------
# CallbackManagerForRetrieverRun – retriever run lifecycle
# ---------------------------------------------------------------------------


class TestCallbackManagerForRetrieverRun:
    """Tests for sync retriever run manager."""

    def test_on_retriever_end_respects_ignore_retriever(self) -> None:
        class IgnoreRetriever(RecordingHandler):
            @property
            def ignore_retriever(self) -> bool:
                return True

        rec = IgnoreRetriever()
        mgr = CallbackManagerForRetrieverRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        mgr.on_retriever_end([Document(page_content="d")])
        assert len(rec.events) == 0

    def test_on_retriever_error_respects_ignore_retriever(self) -> None:
        class IgnoreRetriever(RecordingHandler):
            @property
            def ignore_retriever(self) -> bool:
                return True

        rec = IgnoreRetriever()
        mgr = CallbackManagerForRetrieverRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        mgr.on_retriever_error(ValueError())
        assert len(rec.events) == 0

    def test_empty_handlers_noop(self) -> None:
        mgr = CallbackManagerForRetrieverRun(
            run_id=uuid4(), handlers=[], inheritable_handlers=[]
        )
        mgr.on_retriever_end([])
        mgr.on_retriever_error(ValueError())

    def test_is_parent_run_manager(self) -> None:
        mgr = CallbackManagerForRetrieverRun(
            run_id=uuid4(), handlers=[], inheritable_handlers=[]
        )
        assert isinstance(mgr, ParentRunManager)


# ---------------------------------------------------------------------------
# CallbackManager – main sync callback manager
# ---------------------------------------------------------------------------


class TestCallbackManager:
    """Tests for the main sync CallbackManager."""

    def test_is_async_false(self) -> None:
        mgr = CallbackManager(handlers=[])
        assert mgr.is_async is False

    def test_on_llm_start_returns_managers_per_prompt(self) -> None:
        rec = RecordingHandler()
        mgr = CallbackManager(handlers=[rec])
        managers = mgr.on_llm_start({}, ["p1", "p2", "p3"])
        assert len(managers) == 3
        for m in managers:
            assert isinstance(m, CallbackManagerForLLMRun)

    def test_on_llm_start_uses_provided_run_id_for_first(self) -> None:
        rid = uuid4()
        mgr = CallbackManager(handlers=[RecordingHandler()])
        managers = mgr.on_llm_start({}, ["p1", "p2"], run_id=rid)
        assert managers[0].run_id == rid
        assert managers[1].run_id != rid

    def test_on_llm_start_generates_run_id_when_none(self) -> None:
        mgr = CallbackManager(handlers=[RecordingHandler()])
        managers = mgr.on_llm_start({}, ["p1"])
        assert isinstance(managers[0].run_id, UUID)

    def test_on_chat_model_start_returns_managers_per_message_list(self) -> None:
        rec = RecordingHandler()
        mgr = CallbackManager(handlers=[rec])
        msgs = [[HumanMessage(content="a")], [HumanMessage(content="b")]]
        managers = mgr.on_chat_model_start({}, msgs)
        assert len(managers) == 2
        for m in managers:
            assert isinstance(m, CallbackManagerForLLMRun)

    def test_on_chat_model_start_uses_provided_run_id_for_first(self) -> None:
        rid = uuid4()
        rec = RecordingHandler()
        mgr = CallbackManager(handlers=[rec])
        msgs = [[HumanMessage(content="a")], [HumanMessage(content="b")]]
        managers = mgr.on_chat_model_start({}, msgs, run_id=rid)
        assert managers[0].run_id == rid
        assert managers[1].run_id != rid

    def test_on_chain_start_returns_chain_run_manager(self) -> None:
        rec = RecordingHandler()
        mgr = CallbackManager(handlers=[rec])
        rm = mgr.on_chain_start({}, {"input": "x"})
        assert isinstance(rm, CallbackManagerForChainRun)

    def test_on_chain_start_uses_provided_run_id(self) -> None:
        rid = uuid4()
        mgr = CallbackManager(handlers=[RecordingHandler()])
        rm = mgr.on_chain_start({}, {}, run_id=rid)
        assert rm.run_id == rid

    def test_on_tool_start_returns_tool_run_manager(self) -> None:
        rec = RecordingHandler()
        mgr = CallbackManager(handlers=[rec])
        rm = mgr.on_tool_start({}, "input")
        assert isinstance(rm, CallbackManagerForToolRun)

    def test_on_tool_start_uses_provided_run_id(self) -> None:
        rid = uuid4()
        mgr = CallbackManager(handlers=[RecordingHandler()])
        rm = mgr.on_tool_start({}, "input", run_id=rid)
        assert rm.run_id == rid

    def test_on_retriever_start_returns_retriever_run_manager(self) -> None:
        rec = RecordingHandler()
        mgr = CallbackManager(handlers=[rec])
        rm = mgr.on_retriever_start({}, "query")
        assert isinstance(rm, CallbackManagerForRetrieverRun)

    def test_on_retriever_start_uses_provided_run_id(self) -> None:
        rid = uuid4()
        mgr = CallbackManager(handlers=[RecordingHandler()])
        rm = mgr.on_retriever_start({}, "q", run_id=rid)
        assert rm.run_id == rid

    def test_on_custom_event_dispatches(self) -> None:
        rec = RecordingHandler()
        mgr = CallbackManager(handlers=[rec])
        mgr.on_custom_event("evt", {"data": 1})
        assert len(rec.events) == 1
        assert rec.events[0][0] == "on_custom_event"

    def test_on_custom_event_rejects_extra_kwargs(self) -> None:
        rec = RecordingHandler()
        mgr = CallbackManager(handlers=[rec])
        with pytest.raises(ValueError, match="does not accept additional keyword"):
            mgr.on_custom_event("evt", {}, extra="bad")

    def test_on_custom_event_empty_handlers_noop(self) -> None:
        mgr = CallbackManager(handlers=[])
        mgr.on_custom_event("evt", {})  # should not raise

    def test_on_custom_event_respects_ignore_custom_event(self) -> None:
        class IgnoreCustom(RecordingHandler):
            @property
            def ignore_custom_event(self) -> bool:
                return True

        rec = IgnoreCustom()
        mgr = CallbackManager(handlers=[rec])
        mgr.on_custom_event("evt", {})
        assert len(rec.events) == 0

    def test_run_managers_inherit_tags_and_metadata(self) -> None:
        mgr = CallbackManager(
            handlers=[RecordingHandler()],
            tags=["t1"],
            inheritable_tags=["it1"],
            metadata={"k": "v"},
            inheritable_metadata={"ik": "iv"},
        )
        rm = mgr.on_chain_start({}, {})
        assert "t1" in rm.tags
        assert "it1" in rm.inheritable_tags
        assert rm.metadata["k"] == "v"
        assert rm.inheritable_metadata["ik"] == "iv"

    def test_on_llm_start_propagates_tags_metadata(self) -> None:
        rec = RecordingHandler()
        mgr = CallbackManager(handlers=[rec], tags=["tag"], metadata={"mk": "mv"})
        mgr.on_llm_start({}, ["p"])
        _, _, kw = rec.events[0]
        assert "tag" in kw["tags"]
        assert kw["metadata"]["mk"] == "mv"


# ---------------------------------------------------------------------------
# CallbackManagerForChainGroup – sync chain group
# ---------------------------------------------------------------------------


class TestCallbackManagerForChainGroup:
    """Tests for sync CallbackManagerForChainGroup."""

    def _make_group(
        self,
    ) -> tuple[
        CallbackManagerForChainGroup,
        RecordingHandler,
        CallbackManagerForChainRun,
    ]:
        rec = RecordingHandler()
        parent_rm = CallbackManagerForChainRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[rec]
        )
        group = CallbackManagerForChainGroup(
            handlers=[rec],
            inheritable_handlers=[rec],
            parent_run_id=parent_rm.run_id,
            parent_run_manager=parent_rm,
        )
        return group, rec, parent_rm

    def test_on_chain_end_delegates_to_parent(self) -> None:
        group, rec, _ = self._make_group()
        group.on_chain_end({"result": "ok"})
        # Parent's on_chain_end was called
        chain_end_events = [e for e in rec.events if e[0] == "on_chain_end"]
        assert len(chain_end_events) >= 1

    def test_on_chain_end_sets_ended(self) -> None:
        group, _, _ = self._make_group()
        assert group.ended is False
        group.on_chain_end({})
        assert group.ended is True

    def test_on_chain_error_delegates_to_parent(self) -> None:
        group, rec, _ = self._make_group()
        group.on_chain_error(ValueError("err"))
        chain_error_events = [e for e in rec.events if e[0] == "on_chain_error"]
        assert len(chain_error_events) >= 1

    def test_on_chain_error_sets_ended(self) -> None:
        group, _, _ = self._make_group()
        group.on_chain_error(ValueError())
        assert group.ended is True

    def test_copy_preserves_parent_run_manager(self) -> None:
        group, _, parent_rm = self._make_group()
        cp = group.copy()
        assert isinstance(cp, CallbackManagerForChainGroup)
        assert cp.parent_run_manager is parent_rm

    def test_merge_preserves_parent_run_manager(self) -> None:
        group, _, parent_rm = self._make_group()
        other = CallbackManager(handlers=[], tags=["extra"])
        merged = group.merge(other)
        assert isinstance(merged, CallbackManagerForChainGroup)
        assert merged.parent_run_manager is parent_rm
        assert "extra" in merged.tags


# ---------------------------------------------------------------------------
# BaseRunManager – noop manager
# ---------------------------------------------------------------------------


class TestBaseRunManager:
    """Tests for BaseRunManager."""

    def test_get_noop_manager(self) -> None:
        mgr = BaseRunManager.get_noop_manager()
        assert isinstance(mgr.run_id, UUID)
        assert mgr.handlers == []
        assert mgr.inheritable_handlers == []
        assert mgr.tags == []
        assert mgr.metadata == {}

    def test_initialization_defaults(self) -> None:
        rid = uuid4()
        mgr = BaseRunManager(run_id=rid, handlers=[], inheritable_handlers=[])
        assert mgr.run_id == rid
        assert mgr.parent_run_id is None
        assert mgr.tags == []
        assert mgr.inheritable_tags == []
        assert mgr.metadata == {}
        assert mgr.inheritable_metadata == {}


# ---------------------------------------------------------------------------
# Async run managers – get_sync conversions
# ---------------------------------------------------------------------------


class TestAsyncRunManagerGetSync:
    """Tests for get_sync() on async run managers."""

    def test_async_llm_run_get_sync(self) -> None:
        rid = uuid4()
        h = BaseCallbackHandler()
        async_mgr = AsyncCallbackManagerForLLMRun(
            run_id=rid,
            handlers=[h],
            inheritable_handlers=[h],
            tags=["t"],
            inheritable_tags=["it"],
            metadata={"k": "v"},
            inheritable_metadata={"ik": "iv"},
        )
        sync_mgr = async_mgr.get_sync()
        assert isinstance(sync_mgr, CallbackManagerForLLMRun)
        assert sync_mgr.run_id == rid
        assert h in sync_mgr.handlers
        assert "t" in sync_mgr.tags
        assert sync_mgr.metadata["k"] == "v"

    def test_async_chain_run_get_sync(self) -> None:
        rid = uuid4()
        h = BaseCallbackHandler()
        async_mgr = AsyncCallbackManagerForChainRun(
            run_id=rid,
            handlers=[h],
            inheritable_handlers=[h],
            tags=["t"],
        )
        sync_mgr = async_mgr.get_sync()
        assert isinstance(sync_mgr, CallbackManagerForChainRun)
        assert sync_mgr.run_id == rid

    def test_async_tool_run_get_sync(self) -> None:
        rid = uuid4()
        async_mgr = AsyncCallbackManagerForToolRun(
            run_id=rid, handlers=[], inheritable_handlers=[]
        )
        sync_mgr = async_mgr.get_sync()
        assert isinstance(sync_mgr, CallbackManagerForToolRun)
        assert sync_mgr.run_id == rid

    def test_async_retriever_run_get_sync(self) -> None:
        rid = uuid4()
        async_mgr = AsyncCallbackManagerForRetrieverRun(
            run_id=rid, handlers=[], inheritable_handlers=[]
        )
        sync_mgr = async_mgr.get_sync()
        assert isinstance(sync_mgr, CallbackManagerForRetrieverRun)
        assert sync_mgr.run_id == rid


# ---------------------------------------------------------------------------
# AsyncCallbackManager – main async callback manager
# ---------------------------------------------------------------------------


class TestAsyncCallbackManager:
    """Tests for the main async CallbackManager."""

    def test_is_async_true(self) -> None:
        mgr = AsyncCallbackManager(handlers=[])
        assert mgr.is_async is True

    async def test_on_llm_start_returns_async_managers(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManager(handlers=[rec])
        managers = await mgr.on_llm_start({}, ["p1", "p2"])
        assert len(managers) == 2
        for m in managers:
            assert isinstance(m, AsyncCallbackManagerForLLMRun)

    async def test_on_llm_start_uses_provided_run_id(self) -> None:
        rid = uuid4()
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManager(handlers=[rec])
        managers = await mgr.on_llm_start({}, ["p1", "p2"], run_id=rid)
        assert managers[0].run_id == rid
        assert managers[1].run_id != rid

    async def test_on_chat_model_start_returns_async_managers(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManager(handlers=[rec])
        msgs = [[HumanMessage(content="a")], [HumanMessage(content="b")]]
        managers = await mgr.on_chat_model_start({}, msgs)
        assert len(managers) == 2
        for m in managers:
            assert isinstance(m, AsyncCallbackManagerForLLMRun)

    async def test_on_chain_start_returns_async_chain_manager(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManager(handlers=[rec])
        rm = await mgr.on_chain_start({}, {"input": "x"})
        assert isinstance(rm, AsyncCallbackManagerForChainRun)

    async def test_on_chain_start_uses_provided_run_id(self) -> None:
        rid = uuid4()
        mgr = AsyncCallbackManager(handlers=[AsyncRecordingHandler()])
        rm = await mgr.on_chain_start({}, {}, run_id=rid)
        assert rm.run_id == rid

    async def test_on_tool_start_returns_async_tool_manager(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManager(handlers=[rec])
        rm = await mgr.on_tool_start({}, "input")
        assert isinstance(rm, AsyncCallbackManagerForToolRun)

    async def test_on_retriever_start_returns_async_retriever_manager(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManager(handlers=[rec])
        rm = await mgr.on_retriever_start({}, "query")
        assert isinstance(rm, AsyncCallbackManagerForRetrieverRun)

    async def test_on_custom_event_dispatches(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManager(handlers=[rec])
        await mgr.on_custom_event("evt", {"data": 1})
        assert len(rec.events) == 1
        assert rec.events[0][0] == "on_custom_event"

    async def test_on_custom_event_rejects_extra_kwargs(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManager(handlers=[rec])
        with pytest.raises(ValueError, match="does not accept additional keyword"):
            await mgr.on_custom_event("evt", {}, extra="bad")

    async def test_on_custom_event_empty_handlers_noop(self) -> None:
        mgr = AsyncCallbackManager(handlers=[])
        await mgr.on_custom_event("evt", {})  # should not raise

    async def test_on_custom_event_respects_ignore(self) -> None:
        class IgnoreCustomAsync(AsyncRecordingHandler):
            @property
            def ignore_custom_event(self) -> bool:
                return True

        rec = IgnoreCustomAsync()
        mgr = AsyncCallbackManager(handlers=[rec])
        await mgr.on_custom_event("evt", {})
        assert len(rec.events) == 0


# ---------------------------------------------------------------------------
# Async LLM run manager
# ---------------------------------------------------------------------------


class TestAsyncCallbackManagerForLLMRun:
    """Tests for async LLM run manager."""

    async def test_on_llm_new_token(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManagerForLLMRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        await mgr.on_llm_new_token("tok")
        assert len(rec.events) == 1
        assert rec.events[0][0] == "on_llm_new_token"

    async def test_on_llm_end(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManagerForLLMRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        await mgr.on_llm_end(LLMResult(generations=[]))
        assert any(e[0] == "on_llm_end" for e in rec.events)

    async def test_on_llm_error(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManagerForLLMRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        await mgr.on_llm_error(ValueError("err"))
        assert any(e[0] == "on_llm_error" for e in rec.events)

    async def test_empty_handlers_noop(self) -> None:
        mgr = AsyncCallbackManagerForLLMRun(
            run_id=uuid4(), handlers=[], inheritable_handlers=[]
        )
        await mgr.on_llm_new_token("tok")
        await mgr.on_llm_end(LLMResult(generations=[]))
        await mgr.on_llm_error(ValueError())

    async def test_ignore_llm_skips_all(self) -> None:
        class IgnoreLLM(AsyncRecordingHandler):
            @property
            def ignore_llm(self) -> bool:
                return True

        rec = IgnoreLLM()
        mgr = AsyncCallbackManagerForLLMRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        await mgr.on_llm_new_token("tok")
        await mgr.on_llm_end(LLMResult(generations=[]))
        await mgr.on_llm_error(ValueError())
        assert len(rec.events) == 0


# ---------------------------------------------------------------------------
# Async chain run manager
# ---------------------------------------------------------------------------


class TestAsyncCallbackManagerForChainRun:
    """Tests for async chain run manager."""

    async def test_on_chain_end(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManagerForChainRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        await mgr.on_chain_end({"out": "val"})
        assert any(e[0] == "on_chain_end" for e in rec.events)

    async def test_on_chain_error(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManagerForChainRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        await mgr.on_chain_error(ValueError())
        assert any(e[0] == "on_chain_error" for e in rec.events)

    async def test_on_agent_action(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManagerForChainRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        action = AgentAction(tool="t", tool_input="i", log="l")
        await mgr.on_agent_action(action)
        assert any(e[0] == "on_agent_action" for e in rec.events)

    async def test_on_agent_finish(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManagerForChainRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        finish = AgentFinish(return_values={}, log="done")
        await mgr.on_agent_finish(finish)
        assert any(e[0] == "on_agent_finish" for e in rec.events)

    async def test_ignore_chain_skips_end_and_error(self) -> None:
        class IgnoreChain(AsyncRecordingHandler):
            @property
            def ignore_chain(self) -> bool:
                return True

        rec = IgnoreChain()
        mgr = AsyncCallbackManagerForChainRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        await mgr.on_chain_end({})
        await mgr.on_chain_error(ValueError())
        assert len(rec.events) == 0

    async def test_ignore_agent_skips_action_and_finish(self) -> None:
        class IgnoreAgent(AsyncRecordingHandler):
            @property
            def ignore_agent(self) -> bool:
                return True

        rec = IgnoreAgent()
        mgr = AsyncCallbackManagerForChainRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        await mgr.on_agent_action(AgentAction(tool="t", tool_input="i", log="l"))
        await mgr.on_agent_finish(AgentFinish(return_values={}, log="d"))
        assert len(rec.events) == 0

    async def test_empty_handlers_noop(self) -> None:
        mgr = AsyncCallbackManagerForChainRun(
            run_id=uuid4(), handlers=[], inheritable_handlers=[]
        )
        await mgr.on_chain_end({})
        await mgr.on_chain_error(ValueError())
        await mgr.on_agent_action(AgentAction(tool="t", tool_input="i", log="l"))
        await mgr.on_agent_finish(AgentFinish(return_values={}, log="d"))

    async def test_get_child_returns_async_manager(self) -> None:
        mgr = AsyncCallbackManagerForChainRun(
            run_id=uuid4(), handlers=[], inheritable_handlers=[]
        )
        child = mgr.get_child()
        assert isinstance(child, AsyncCallbackManager)


# ---------------------------------------------------------------------------
# Async tool run manager
# ---------------------------------------------------------------------------


class TestAsyncCallbackManagerForToolRun:
    """Tests for async tool run manager."""

    async def test_on_tool_end(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManagerForToolRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        await mgr.on_tool_end("result")
        assert any(e[0] == "on_tool_end" for e in rec.events)

    async def test_on_tool_error(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManagerForToolRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        await mgr.on_tool_error(ValueError())
        assert any(e[0] == "on_tool_error" for e in rec.events)

    async def test_ignore_agent_skips_both(self) -> None:
        class IgnoreAgent(AsyncRecordingHandler):
            @property
            def ignore_agent(self) -> bool:
                return True

        rec = IgnoreAgent()
        mgr = AsyncCallbackManagerForToolRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        await mgr.on_tool_end("out")
        await mgr.on_tool_error(ValueError())
        assert len(rec.events) == 0

    async def test_empty_handlers_noop(self) -> None:
        mgr = AsyncCallbackManagerForToolRun(
            run_id=uuid4(), handlers=[], inheritable_handlers=[]
        )
        await mgr.on_tool_end("out")
        await mgr.on_tool_error(ValueError())


# ---------------------------------------------------------------------------
# Async retriever run manager
# ---------------------------------------------------------------------------


class TestAsyncCallbackManagerForRetrieverRun:
    """Tests for async retriever run manager."""

    async def test_on_retriever_end(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManagerForRetrieverRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        docs = [Document(page_content="d")]
        await mgr.on_retriever_end(docs)
        assert any(e[0] == "on_retriever_end" for e in rec.events)

    async def test_on_retriever_error(self) -> None:
        rec = AsyncRecordingHandler()
        mgr = AsyncCallbackManagerForRetrieverRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        await mgr.on_retriever_error(ValueError())
        assert any(e[0] == "on_retriever_error" for e in rec.events)

    async def test_ignore_retriever_skips_both(self) -> None:
        class IgnoreRetriever(AsyncRecordingHandler):
            @property
            def ignore_retriever(self) -> bool:
                return True

        rec = IgnoreRetriever()
        mgr = AsyncCallbackManagerForRetrieverRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[]
        )
        await mgr.on_retriever_end([])
        await mgr.on_retriever_error(ValueError())
        assert len(rec.events) == 0

    async def test_empty_handlers_noop(self) -> None:
        mgr = AsyncCallbackManagerForRetrieverRun(
            run_id=uuid4(), handlers=[], inheritable_handlers=[]
        )
        await mgr.on_retriever_end([])
        await mgr.on_retriever_error(ValueError())


# ---------------------------------------------------------------------------
# AsyncCallbackManagerForChainGroup
# ---------------------------------------------------------------------------


class TestAsyncCallbackManagerForChainGroup:
    """Tests for async chain group manager."""

    async def _make_group(
        self,
    ) -> tuple[
        AsyncRecordingHandler,
        AsyncCallbackManagerForChainRun,
    ]:
        rec = AsyncRecordingHandler()
        parent_rm = AsyncCallbackManagerForChainRun(
            run_id=uuid4(), handlers=[rec], inheritable_handlers=[rec]
        )
        group = AsyncCallbackManagerForChainGroup(
            handlers=[rec],
            inheritable_handlers=[rec],
            parent_run_id=parent_rm.run_id,
            parent_run_manager=parent_rm,
        )
        return group, rec, parent_rm

    async def test_on_chain_end_delegates_to_parent(self) -> None:
        group, rec, _ = await self._make_group()
        await group.on_chain_end({"result": "ok"})
        chain_end_events = [e for e in rec.events if e[0] == "on_chain_end"]
        assert len(chain_end_events) >= 1

    async def test_on_chain_end_sets_ended(self) -> None:
        group, _, _ = await self._make_group()
        assert group.ended is False
        await group.on_chain_end({})
        assert group.ended is True

    async def test_on_chain_error_delegates_to_parent(self) -> None:
        group, rec, _ = await self._make_group()
        await group.on_chain_error(ValueError("err"))
        chain_error_events = [e for e in rec.events if e[0] == "on_chain_error"]
        assert len(chain_error_events) >= 1

    async def test_on_chain_error_sets_ended(self) -> None:
        group, _, _ = await self._make_group()
        await group.on_chain_error(ValueError())
        assert group.ended is True

    async def test_copy_preserves_parent_run_manager(self) -> None:
        group, _, parent_rm = await self._make_group()
        cp = group.copy()
        assert isinstance(cp, AsyncCallbackManagerForChainGroup)
        assert cp.parent_run_manager is parent_rm

    async def test_merge_preserves_parent_run_manager(self) -> None:
        group, _, parent_rm = await self._make_group()
        other = AsyncCallbackManager(handlers=[], tags=["extra"])
        merged = group.merge(other)
        assert isinstance(merged, AsyncCallbackManagerForChainGroup)
        assert merged.parent_run_manager is parent_rm
        assert "extra" in merged.tags


# ---------------------------------------------------------------------------
# AsyncParentRunManager
# ---------------------------------------------------------------------------


class TestAsyncParentRunManager:
    """Tests for AsyncParentRunManager child creation."""

    async def test_get_child_returns_async_callback_manager(self) -> None:
        h = BaseCallbackHandler()
        mgr = AsyncCallbackManagerForChainRun(
            run_id=uuid4(),
            handlers=[h],
            inheritable_handlers=[h],
            inheritable_tags=["it"],
            inheritable_metadata={"ik": "iv"},
        )
        child = mgr.get_child()
        assert isinstance(child, AsyncCallbackManager)
        assert child.parent_run_id == mgr.run_id
        assert h in child.handlers
        assert "it" in child.tags

    async def test_get_child_with_tag(self) -> None:
        mgr = AsyncCallbackManagerForChainRun(
            run_id=uuid4(), handlers=[], inheritable_handlers=[]
        )
        child = mgr.get_child(tag="local")
        assert "local" in child.tags
        assert "local" not in child.inheritable_tags
