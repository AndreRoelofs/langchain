from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import (
    Any,
)

import pytest
from pydantic import BaseModel
from syrupy.assertion import SnapshotAssertion
from typing_extensions import override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import (
    BaseChatModel,
    FakeListLLM,
    LanguageModelInput,
)
from langchain_core.load import dumps
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableGenerator,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableWithFallbacks,
)
from langchain_core.tools import BaseTool


@pytest.fixture
def llm() -> RunnableWithFallbacks:
    error_llm = FakeListLLM(responses=["foo"], i=1)
    pass_llm = FakeListLLM(responses=["bar"])

    return error_llm.with_fallbacks([pass_llm])


@pytest.fixture
def llm_multi() -> RunnableWithFallbacks:
    error_llm = FakeListLLM(responses=["foo"], i=1)
    error_llm_2 = FakeListLLM(responses=["baz"], i=1)
    pass_llm = FakeListLLM(responses=["bar"])

    return error_llm.with_fallbacks([error_llm_2, pass_llm])


@pytest.fixture
def chain() -> Runnable:
    error_llm = FakeListLLM(responses=["foo"], i=1)
    pass_llm = FakeListLLM(responses=["bar"])

    prompt = PromptTemplate.from_template("what did baz say to {buz}")
    return RunnableParallel({"buz": lambda x: x}) | (prompt | error_llm).with_fallbacks(
        [prompt | pass_llm]
    )


def _raise_error(_: dict) -> str:
    raise ValueError


def _dont_raise_error(inputs: dict) -> str:
    if "exception" in inputs:
        return "bar"
    raise ValueError


@pytest.fixture
def chain_pass_exceptions() -> Runnable:
    fallback = RunnableLambda(_dont_raise_error)
    return {"text": RunnablePassthrough()} | RunnableLambda(
        _raise_error
    ).with_fallbacks([fallback], exception_key="exception")


@pytest.mark.parametrize(
    "runnable",
    ["llm", "llm_multi", "chain", "chain_pass_exceptions"],
)
def test_fallbacks(
    runnable: RunnableWithFallbacks, request: Any, snapshot: SnapshotAssertion
) -> None:
    runnable = request.getfixturevalue(runnable)
    assert runnable.invoke("hello") == "bar"
    assert runnable.batch(["hi", "hey", "bye"]) == ["bar"] * 3
    assert list(runnable.stream("hello")) == ["bar"]
    assert dumps(runnable, pretty=True) == snapshot


@pytest.mark.parametrize(
    "runnable",
    ["llm", "llm_multi", "chain", "chain_pass_exceptions"],
)
async def test_fallbacks_async(runnable: RunnableWithFallbacks, request: Any) -> None:
    runnable = request.getfixturevalue(runnable)
    assert await runnable.ainvoke("hello") == "bar"
    assert await runnable.abatch(["hi", "hey", "bye"]) == ["bar"] * 3
    assert list(await runnable.ainvoke("hello")) == list("bar")


def _runnable(inputs: dict) -> str:
    if inputs["text"] == "foo":
        return "first"
    if "exception" not in inputs:
        msg = "missing exception"
        raise ValueError(msg)
    if inputs["text"] == "bar":
        return "second"
    if isinstance(inputs["exception"], ValueError):
        raise RuntimeError  # noqa: TRY004
    return "third"


def _assert_potential_error(actual: list, expected: list) -> None:
    for x, y in zip(actual, expected, strict=False):
        if isinstance(x, Exception):
            assert isinstance(y, type(x))
        else:
            assert x == y


def test_invoke_with_exception_key() -> None:
    runnable = RunnableLambda(_runnable)
    runnable_with_single = runnable.with_fallbacks(
        [runnable], exception_key="exception"
    )
    with pytest.raises(ValueError, match="missing exception"):
        runnable_with_single.invoke({"text": "baz"})

    actual = runnable_with_single.invoke({"text": "bar"})
    expected = "second"
    _assert_potential_error([actual], [expected])

    runnable_with_double = runnable.with_fallbacks(
        [runnable, runnable], exception_key="exception"
    )
    actual = runnable_with_double.invoke({"text": "baz"})

    expected = "third"
    _assert_potential_error([actual], [expected])


async def test_ainvoke_with_exception_key() -> None:
    runnable = RunnableLambda(_runnable)
    runnable_with_single = runnable.with_fallbacks(
        [runnable], exception_key="exception"
    )
    with pytest.raises(ValueError, match="missing exception"):
        await runnable_with_single.ainvoke({"text": "baz"})

    actual = await runnable_with_single.ainvoke({"text": "bar"})
    expected = "second"
    _assert_potential_error([actual], [expected])

    runnable_with_double = runnable.with_fallbacks(
        [runnable, runnable], exception_key="exception"
    )
    actual = await runnable_with_double.ainvoke({"text": "baz"})
    expected = "third"
    _assert_potential_error([actual], [expected])


def test_batch() -> None:
    runnable = RunnableLambda(_runnable)
    with pytest.raises(ValueError, match="missing exception"):
        runnable.batch([{"text": "foo"}, {"text": "bar"}, {"text": "baz"}])
    actual = runnable.batch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )
    expected = ["first", ValueError(), ValueError()]
    _assert_potential_error(actual, expected)

    runnable_with_single = runnable.with_fallbacks(
        [runnable], exception_key="exception"
    )
    with pytest.raises(RuntimeError):
        runnable_with_single.batch([{"text": "foo"}, {"text": "bar"}, {"text": "baz"}])
    actual = runnable_with_single.batch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )
    expected = ["first", "second", RuntimeError()]
    _assert_potential_error(actual, expected)

    runnable_with_double = runnable.with_fallbacks(
        [runnable, runnable], exception_key="exception"
    )
    actual = runnable_with_double.batch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )

    expected = ["first", "second", "third"]
    _assert_potential_error(actual, expected)

    runnable_with_double = runnable.with_fallbacks(
        [runnable, runnable],
        exception_key="exception",
        exceptions_to_handle=(ValueError,),
    )
    actual = runnable_with_double.batch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )

    expected = ["first", "second", RuntimeError()]
    _assert_potential_error(actual, expected)


async def test_abatch() -> None:
    runnable = RunnableLambda(_runnable)
    with pytest.raises(ValueError, match="missing exception"):
        await runnable.abatch([{"text": "foo"}, {"text": "bar"}, {"text": "baz"}])
    actual = await runnable.abatch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )
    expected = ["first", ValueError(), ValueError()]
    _assert_potential_error(actual, expected)

    runnable_with_single = runnable.with_fallbacks(
        [runnable], exception_key="exception"
    )
    with pytest.raises(RuntimeError):
        await runnable_with_single.abatch(
            [
                {"text": "foo"},
                {"text": "bar"},
                {"text": "baz"},
            ]
        )
    actual = await runnable_with_single.abatch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )
    expected = ["first", "second", RuntimeError()]
    _assert_potential_error(actual, expected)

    runnable_with_double = runnable.with_fallbacks(
        [runnable, runnable], exception_key="exception"
    )
    actual = await runnable_with_double.abatch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )

    expected = ["first", "second", "third"]
    _assert_potential_error(actual, expected)

    runnable_with_double = runnable.with_fallbacks(
        [runnable, runnable],
        exception_key="exception",
        exceptions_to_handle=(ValueError,),
    )
    actual = await runnable_with_double.abatch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )

    expected = ["first", "second", RuntimeError()]
    _assert_potential_error(actual, expected)


def _generate(_: Iterator) -> Iterator[str]:
    yield from "foo bar"


def _error(msg: str) -> None:
    raise ValueError(msg)


def _generate_immediate_error(_: Iterator) -> Iterator[str]:
    _error("immediate error")
    yield ""


def _generate_delayed_error(_: Iterator) -> Iterator[str]:
    yield ""
    _error("delayed error")


def test_fallbacks_stream() -> None:
    runnable = RunnableGenerator(_generate_immediate_error).with_fallbacks(
        [RunnableGenerator(_generate)]
    )
    assert list(runnable.stream({})) == list("foo bar")

    runnable = RunnableGenerator(_generate_delayed_error).with_fallbacks(
        [RunnableGenerator(_generate)]
    )
    with pytest.raises(ValueError, match="delayed error"):
        list(runnable.stream({}))


async def _agenerate(_: AsyncIterator) -> AsyncIterator[str]:
    for c in "foo bar":
        yield c


async def _agenerate_immediate_error(_: AsyncIterator) -> AsyncIterator[str]:
    _error("immediate error")
    yield ""


async def _agenerate_delayed_error(_: AsyncIterator) -> AsyncIterator[str]:
    yield ""
    _error("delayed error")


async def test_fallbacks_astream() -> None:
    runnable = RunnableGenerator(_agenerate_immediate_error).with_fallbacks(
        [RunnableGenerator(_agenerate)]
    )
    expected = (c for c in "foo bar")
    async for c in runnable.astream({}):
        assert c == next(expected)

    runnable = RunnableGenerator(_agenerate_delayed_error).with_fallbacks(
        [RunnableGenerator(_agenerate)]
    )
    with pytest.raises(ValueError, match="delayed error"):
        _ = [_ async for _ in runnable.astream({})]


class FakeStructuredOutputModel(BaseChatModel):
    foo: int

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call."""
        return ChatResult(generations=[])

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(tools=tools)

    @override
    def with_structured_output(
        self, schema: dict | type[BaseModel], **kwargs: Any
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        return RunnableLambda(lambda _: {"foo": self.foo})

    @property
    def _llm_type(self) -> str:
        return "fake1"


class FakeModel(BaseChatModel):
    bar: int

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call."""
        return ChatResult(generations=[])

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(tools=tools)

    @property
    def _llm_type(self) -> str:
        return "fake2"


def test_fallbacks_getattr() -> None:
    llm_with_fallbacks = FakeStructuredOutputModel(foo=3).with_fallbacks(
        [FakeModel(bar=4)]
    )
    assert llm_with_fallbacks.foo == 3

    with pytest.raises(AttributeError):
        assert llm_with_fallbacks.bar == 4


def test_fallbacks_getattr_runnable_output() -> None:
    llm_with_fallbacks = FakeStructuredOutputModel(foo=3).with_fallbacks(
        [FakeModel(bar=4)]
    )
    llm_with_fallbacks_with_tools = llm_with_fallbacks.bind_tools([])
    assert isinstance(llm_with_fallbacks_with_tools, RunnableWithFallbacks)
    assert isinstance(llm_with_fallbacks_with_tools.runnable, RunnableBinding)
    assert all(
        isinstance(fallback, RunnableBinding)
        for fallback in llm_with_fallbacks_with_tools.fallbacks
    )
    assert llm_with_fallbacks_with_tools.runnable.kwargs["tools"] == []


# ---------------------------------------------------------------------------
# Tests for _returns_runnable / _is_runnable_type
# ---------------------------------------------------------------------------
import typing

from langchain_core.runnables.fallbacks import _is_runnable_type, _returns_runnable


def test_is_runnable_type_with_runnable_class() -> None:
    assert _is_runnable_type(Runnable) is True


def test_is_runnable_type_with_non_runnable_class() -> None:
    assert _is_runnable_type(str) is False


def test_is_runnable_type_with_generic_runnable() -> None:
    """A generic Runnable type (e.g., Runnable[str, str]) is recognized."""
    from langchain_core.runnables.base import RunnableBinding

    assert _is_runnable_type(RunnableBinding) is True


def test_is_runnable_type_with_none() -> None:
    assert _is_runnable_type(None) is False


def test_returns_runnable_non_callable() -> None:
    assert _returns_runnable("not a callable") is False


def test_returns_runnable_callable_returning_runnable() -> None:
    def func() -> RunnableLambda:
        return RunnableLambda(lambda x: x)

    assert _returns_runnable(func) is True


def test_returns_runnable_callable_returning_non_runnable() -> None:
    def func() -> str:
        return "hello"

    assert _returns_runnable(func) is False


def test_returns_runnable_callable_no_return_annotation() -> None:
    def func():  # type: ignore[no-untyped-def]
        pass

    assert _returns_runnable(func) is False


# ---------------------------------------------------------------------------
# Tests for RunnableWithFallbacks properties and edge cases
# ---------------------------------------------------------------------------


def test_runnables_property() -> None:
    """The runnables property yields the main runnable then fallbacks."""
    main = RunnableLambda(lambda x: x)
    fb1 = RunnableLambda(lambda x: x + "1")
    fb2 = RunnableLambda(lambda x: x + "2")
    rwf = main.with_fallbacks([fb1, fb2])
    runnables = list(rwf.runnables)
    assert len(runnables) == 3
    assert runnables[0] is main
    assert runnables[1] is fb1
    assert runnables[2] is fb2


def test_input_output_types() -> None:
    """InputType and OutputType are delegated to the main runnable."""
    main = RunnableLambda(lambda x: x)
    fb = RunnableLambda(lambda x: x)
    rwf = main.with_fallbacks([fb])
    assert rwf.InputType == main.InputType
    assert rwf.OutputType == main.OutputType


def test_config_specs_merged() -> None:
    """config_specs should merge specs from main and fallback runnables."""
    main = RunnableLambda(lambda x: x)
    fb = RunnableLambda(lambda x: x)
    rwf = main.with_fallbacks([fb])
    # Both have the same specs, so result should be the unique union
    specs = rwf.config_specs
    assert isinstance(specs, list)


def test_is_lc_serializable() -> None:
    assert RunnableWithFallbacks.is_lc_serializable() is True


def test_get_lc_namespace() -> None:
    assert RunnableWithFallbacks.get_lc_namespace() == [
        "langchain",
        "schema",
        "runnable",
    ]


def test_get_input_schema() -> None:
    """get_input_schema delegates to the main runnable."""
    main = RunnableLambda(lambda x: x)
    fb = RunnableLambda(lambda x: x)
    rwf = main.with_fallbacks([fb])
    schema = rwf.get_input_schema()
    assert schema is not None


def test_get_output_schema() -> None:
    """get_output_schema delegates to the main runnable."""
    main = RunnableLambda(lambda x: x)
    fb = RunnableLambda(lambda x: x)
    rwf = main.with_fallbacks([fb])
    schema = rwf.get_output_schema()
    assert schema is not None


# ---------------------------------------------------------------------------
# Tests for exception_key validation
# ---------------------------------------------------------------------------


def test_invoke_exception_key_non_dict_raises() -> None:
    """When exception_key is set and input is not a dict, ValueError is raised."""
    main = RunnableLambda(lambda x: x)
    fb = RunnableLambda(lambda x: x)
    rwf = main.with_fallbacks([fb], exception_key="error")
    with pytest.raises(ValueError, match="exception_key"):
        rwf.invoke("non_dict_input")


async def test_ainvoke_exception_key_non_dict_raises() -> None:
    """Async version: exception_key with non-dict input raises ValueError."""
    main = RunnableLambda(lambda x: x)
    fb = RunnableLambda(lambda x: x)
    rwf = main.with_fallbacks([fb], exception_key="error")
    with pytest.raises(ValueError, match="exception_key"):
        await rwf.ainvoke("non_dict_input")


def test_stream_exception_key_non_dict_raises() -> None:
    """Stream with exception_key and non-dict input raises ValueError."""
    main = RunnableLambda(lambda x: x)
    fb = RunnableLambda(lambda x: x)
    rwf = main.with_fallbacks([fb], exception_key="error")
    with pytest.raises(ValueError, match="exception_key"):
        list(rwf.stream("non_dict_input"))


async def test_astream_exception_key_non_dict_raises() -> None:
    """Async stream with exception_key and non-dict input raises ValueError."""
    main = RunnableLambda(lambda x: x)
    fb = RunnableLambda(lambda x: x)
    rwf = main.with_fallbacks([fb], exception_key="error")
    with pytest.raises(ValueError, match="exception_key"):
        async for _ in rwf.astream("non_dict_input"):
            pass


def test_batch_exception_key_non_dict_raises() -> None:
    """Batch with exception_key and non-dict inputs raises ValueError."""
    main = RunnableLambda(lambda x: x)
    fb = RunnableLambda(lambda x: x)
    rwf = main.with_fallbacks([fb], exception_key="error")
    with pytest.raises(ValueError, match="exception_key"):
        rwf.batch(["non_dict1", "non_dict2"])


async def test_abatch_exception_key_non_dict_raises() -> None:
    """Async batch with exception_key and non-dict inputs raises ValueError."""
    main = RunnableLambda(lambda x: x)
    fb = RunnableLambda(lambda x: x)
    rwf = main.with_fallbacks([fb], exception_key="error")
    with pytest.raises(ValueError, match="exception_key"):
        await rwf.abatch(["non_dict1", "non_dict2"])


# ---------------------------------------------------------------------------
# Tests for custom exceptions_to_handle
# ---------------------------------------------------------------------------


def test_custom_exceptions_to_handle() -> None:
    """Only specified exception types trigger fallback."""
    call_count = 0

    def fail_with_value_error(x: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("value err")
        return "recovered"

    def fail_with_type_error(x: str) -> str:
        raise TypeError("type err")

    # ValueError should trigger fallback
    main = RunnableLambda(fail_with_value_error)
    fb = RunnableLambda(lambda x: "fallback_result")
    rwf = main.with_fallbacks([fb], exceptions_to_handle=(ValueError,))
    assert rwf.invoke("test") == "fallback_result"

    # TypeError should NOT trigger fallback
    main2 = RunnableLambda(fail_with_type_error)
    rwf2 = main2.with_fallbacks([fb], exceptions_to_handle=(ValueError,))
    with pytest.raises(TypeError, match="type err"):
        rwf2.invoke("test")


def test_fallbacks_empty_batch() -> None:
    """Empty batch input returns empty list."""
    main = RunnableLambda(lambda x: x)
    fb = RunnableLambda(lambda x: x)
    rwf = main.with_fallbacks([fb])
    assert rwf.batch([]) == []


async def test_fallbacks_empty_abatch() -> None:
    """Empty async batch input returns empty list."""
    main = RunnableLambda(lambda x: x)
    fb = RunnableLambda(lambda x: x)
    rwf = main.with_fallbacks([fb])
    assert await rwf.abatch([]) == []


def test_fallbacks_all_succeed_uses_first() -> None:
    """When the main runnable succeeds, fallbacks are not called."""
    call_log: list[str] = []

    def main_fn(x: str) -> str:
        call_log.append("main")
        return "main_result"

    def fallback_fn(x: str) -> str:
        call_log.append("fallback")
        return "fallback_result"

    rwf = RunnableLambda(main_fn).with_fallbacks([RunnableLambda(fallback_fn)])
    result = rwf.invoke("test")
    assert result == "main_result"
    assert call_log == ["main"]


def test_fallbacks_chain_of_failures() -> None:
    """When all runnables fail, the first error is raised."""
    errors: list[Exception] = []

    def fail1(x: str) -> str:
        e = ValueError("error1")
        errors.append(e)
        raise e

    def fail2(x: str) -> str:
        e = ValueError("error2")
        errors.append(e)
        raise e

    rwf = RunnableLambda(fail1).with_fallbacks([RunnableLambda(fail2)])
    with pytest.raises(ValueError, match="error1"):
        rwf.invoke("test")
