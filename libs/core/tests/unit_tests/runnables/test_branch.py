"""Comprehensive tests for RunnableBranch functionality."""

import asyncio
from typing import Any

import pytest

from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.runnables.config import RunnableConfig


def test_branch_initialization() -> None:
    """Test RunnableBranch initialization."""
    branch = RunnableBranch(
        (lambda x: x > 0, lambda x: x + 1),
        (lambda x: x < 0, lambda x: x - 1),
        lambda x: x,
    )

    assert isinstance(branch, RunnableBranch)
    assert len(branch.branches) == 2
    assert branch.default is not None


def test_branch_requires_minimum_branches() -> None:
    """Test that RunnableBranch requires at least 2 branches (1 condition + default)."""
    # Only one branch (no default) should fail
    with pytest.raises(ValueError, match="at least two branches"):
        RunnableBranch((lambda x: x > 0, lambda x: x + 1))

    # Only default (no conditions) should fail
    with pytest.raises(ValueError, match="at least two branches"):
        RunnableBranch(lambda x: x)


def test_branch_invoke_first_condition_true() -> None:
    """Test branch invocation when first condition is True."""
    branch = RunnableBranch(
        (lambda x: x > 0, lambda x: x + 1),
        (lambda x: x < 0, lambda x: x - 1),
        lambda x: x * 10,
    )

    result = branch.invoke(5)
    assert result == 6  # First condition (x > 0) matched


def test_branch_invoke_second_condition_true() -> None:
    """Test branch invocation when second condition is True."""
    branch = RunnableBranch(
        (lambda x: x > 10, lambda x: x + 1),
        (lambda x: x > 0, lambda x: x * 2),
        lambda x: x - 1,
    )

    result = branch.invoke(5)
    assert result == 10  # Second condition (x > 0) matched


def test_branch_invoke_default() -> None:
    """Test branch invocation when no condition is True."""
    branch = RunnableBranch(
        (lambda x: x > 10, lambda x: x + 1),
        (lambda x: x < 0, lambda x: x - 1),
        lambda x: x * 100,
    )

    result = branch.invoke(5)
    assert result == 500  # No condition matched, use default


async def test_branch_ainvoke_first_condition() -> None:
    """Test async branch invocation."""

    async def condition1(x: int) -> bool:
        return x > 0

    async def action1(x: int) -> int:
        return x + 1

    async def default_action(x: int) -> int:
        return x * 10

    branch = RunnableBranch((condition1, action1), default_action)

    result = await branch.ainvoke(5)
    assert result == 6


async def test_branch_ainvoke_default() -> None:
    """Test async branch default path."""

    async def condition1(x: int) -> bool:
        return x > 100

    async def action1(x: int) -> int:
        return x + 1

    async def default_action(x: int) -> int:
        return x * 10

    branch = RunnableBranch((condition1, action1), default_action)

    result = await branch.ainvoke(5)
    assert result == 50


def test_branch_batch() -> None:
    """Test RunnableBranch batch operation."""
    branch = RunnableBranch(
        (lambda x: x > 5, lambda x: x * 2),
        (lambda x: x > 0, lambda x: x + 10),
        lambda x: x - 10,
    )

    results = branch.batch([1, 3, 7, 10, -5])
    assert results == [11, 13, 14, 20, -15]


async def test_branch_abatch() -> None:
    """Test async RunnableBranch batch."""
    branch = RunnableBranch(
        (lambda x: x > 5, lambda x: x * 2),
        (lambda x: x > 0, lambda x: x + 10),
        lambda x: x - 10,
    )

    results = await branch.abatch([1, 3, 7, 10, -5])
    assert results == [11, 13, 14, 20, -15]


def test_branch_stream() -> None:
    """Test RunnableBranch streaming."""
    branch = RunnableBranch(
        (lambda x: x > 0, lambda x: x + 1),
        lambda x: x - 1,
    )

    result = list(branch.stream(5))
    assert result == [6]


async def test_branch_astream() -> None:
    """Test async RunnableBranch streaming."""
    branch = RunnableBranch(
        (lambda x: x > 0, lambda x: x + 1),
        lambda x: x - 1,
    )

    result = [x async for x in branch.astream(5)]
    assert result == [6]


def test_branch_with_runnable_objects() -> None:
    """Test RunnableBranch with Runnable objects instead of lambdas."""
    condition = RunnableLambda(lambda x: x > 0)
    action_true = RunnableLambda(lambda x: x + 1)
    action_false = RunnableLambda(lambda x: x - 1)

    branch = RunnableBranch((condition, action_true), action_false)

    assert branch.invoke(5) == 6
    assert branch.invoke(-5) == -6


def test_branch_multiple_conditions() -> None:
    """Test branch with many conditions."""
    branch = RunnableBranch(
        (lambda x: x > 100, lambda x: "very large"),
        (lambda x: x > 50, lambda x: "large"),
        (lambda x: x > 10, lambda x: "medium"),
        (lambda x: x > 0, lambda x: "small"),
        lambda x: "negative or zero",
    )

    assert branch.invoke(150) == "very large"
    assert branch.invoke(75) == "large"
    assert branch.invoke(25) == "medium"
    assert branch.invoke(5) == "small"
    assert branch.invoke(-5) == "negative or zero"
    assert branch.invoke(0) == "negative or zero"


def test_branch_with_dict_input() -> None:
    """Test RunnableBranch with dict input."""
    branch = RunnableBranch(
        (lambda x: x["type"] == "add", lambda x: x["a"] + x["b"]),
        (lambda x: x["type"] == "multiply", lambda x: x["a"] * x["b"]),
        lambda x: 0,
    )

    assert branch.invoke({"type": "add", "a": 5, "b": 3}) == 8
    assert branch.invoke({"type": "multiply", "a": 5, "b": 3}) == 15
    assert branch.invoke({"type": "unknown", "a": 5, "b": 3}) == 0


def test_branch_exception_in_condition() -> None:
    """Test branch when condition raises exception."""

    def failing_condition(_: int) -> bool:
        msg = "Condition failed"
        raise ValueError(msg)

    branch = RunnableBranch(
        (failing_condition, lambda x: x + 1),
        lambda x: x,
    )

    with pytest.raises(ValueError, match="Condition failed"):
        branch.invoke(5)


def test_branch_exception_in_action() -> None:
    """Test branch when action raises exception."""

    def failing_action(_: int) -> int:
        msg = "Action failed"
        raise ValueError(msg)

    branch = RunnableBranch(
        (lambda x: x > 0, failing_action),
        lambda x: x,
    )

    with pytest.raises(ValueError, match="Action failed"):
        branch.invoke(5)


async def test_branch_exception_in_async_action() -> None:
    """Test async branch when action raises exception."""

    async def failing_action(_: int) -> int:
        msg = "Action failed"
        raise ValueError(msg)

    branch = RunnableBranch(
        (lambda x: x > 0, failing_action),
        lambda x: x,
    )

    with pytest.raises(ValueError, match="Action failed"):
        await branch.ainvoke(5)


def test_branch_conditions_evaluated_in_order() -> None:
    """Test that conditions are evaluated in order and stop at first match."""
    evaluations: list[int] = []

    def condition1(x: int) -> bool:
        evaluations.append(1)
        return x > 10

    def condition2(x: int) -> bool:
        evaluations.append(2)
        return x > 5

    def condition3(x: int) -> bool:
        evaluations.append(3)
        return x > 0

    branch = RunnableBranch(
        (condition1, lambda x: "first"),
        (condition2, lambda x: "second"),
        (condition3, lambda x: "third"),
        lambda x: "default",
    )

    # Input > 10: only first condition should be evaluated
    evaluations.clear()
    result = branch.invoke(15)
    assert result == "first"
    assert evaluations == [1]

    # Input between 5 and 10: first two conditions evaluated
    evaluations.clear()
    result = branch.invoke(7)
    assert result == "second"
    assert evaluations == [1, 2]

    # Input between 0 and 5: all three conditions evaluated
    evaluations.clear()
    result = branch.invoke(3)
    assert result == "third"
    assert evaluations == [1, 2, 3]

    # Input <= 0: all conditions evaluated, use default
    evaluations.clear()
    result = branch.invoke(-5)
    assert result == "default"
    assert evaluations == [1, 2, 3]


def test_branch_with_streaming_output() -> None:
    """Test branch with action that streams."""
    from langchain_core.language_models import FakeStreamingListLLM

    llm = FakeStreamingListLLM(responses=["hello world"])

    branch = RunnableBranch(
        (lambda x: x == "generate", lambda _: llm),
        lambda x: x,
    )

    result = list(branch.stream("generate"))
    # Should stream from the LLM
    assert len(result) > 1
    assert "".join(result) == "hello world"


async def test_branch_with_streaming_output_async() -> None:
    """Test async branch with streaming output."""
    from langchain_core.language_models import FakeStreamingListLLM

    llm = FakeStreamingListLLM(responses=["hello world"])

    branch = RunnableBranch(
        (lambda x: x == "generate", lambda _: llm),
        lambda x: x,
    )

    result = [x async for x in branch.astream("generate")]
    assert len(result) > 1
    assert "".join(result) == "hello world"


def test_branch_config_propagation() -> None:
    """Test that config is propagated through branch."""
    configs_seen: list[RunnableConfig] = []

    def track_config(x: int, config: RunnableConfig) -> int:
        configs_seen.append(config)
        return x + 1

    branch = RunnableBranch(
        (lambda x: x > 0, track_config),
        lambda x: x,
    )

    config: RunnableConfig = {"tags": ["my-tag"]}
    result = branch.invoke(5, config)

    assert result == 6
    assert len(configs_seen) == 1
    # Tags should be propagated
    assert "my-tag" in configs_seen[0]["tags"]


def test_branch_get_input_schema() -> None:
    """Test branch input schema inference."""
    branch = RunnableBranch(
        (lambda x: x > 0, lambda x: x + 1),
        lambda x: x - 1,
    )

    schema = branch.get_input_schema()
    # Schema should be generated successfully
    json_schema = schema.model_json_schema()
    assert json_schema is not None


def test_branch_config_specs() -> None:
    """Test branch aggregates config specs from all branches."""
    from langchain_core.language_models import FakeListLLM
    from langchain_core.runnables import ConfigurableField

    llm1 = FakeListLLM(responses=["a"]).configurable_fields(
        responses=ConfigurableField(id="llm1_responses")
    )
    llm2 = FakeListLLM(responses=["b"]).configurable_fields(
        responses=ConfigurableField(id="llm2_responses")
    )
    llm3 = FakeListLLM(responses=["c"]).configurable_fields(
        responses=ConfigurableField(id="llm3_responses")
    )

    branch = RunnableBranch(
        (lambda _: True, llm1),
        (lambda _: False, llm2),
        llm3,
    )

    specs = branch.config_specs
    spec_ids = [spec.id for spec in specs]
    assert "llm1_responses" in spec_ids
    assert "llm2_responses" in spec_ids
    assert "llm3_responses" in spec_ids


def test_branch_serialization() -> None:
    """Test RunnableBranch serialization."""
    assert RunnableBranch.is_lc_serializable()
    assert RunnableBranch.get_lc_namespace() == ["langchain", "schema", "runnable"]


def test_branch_with_complex_conditions() -> None:
    """Test branch with complex condition logic."""

    def is_even_and_positive(x: int) -> bool:
        return x > 0 and x % 2 == 0

    def is_odd_and_positive(x: int) -> bool:
        return x > 0 and x % 2 == 1

    branch = RunnableBranch(
        (is_even_and_positive, lambda x: f"even: {x}"),
        (is_odd_and_positive, lambda x: f"odd: {x}"),
        lambda x: f"non-positive: {x}",
    )

    assert branch.invoke(4) == "even: 4"
    assert branch.invoke(5) == "odd: 5"
    assert branch.invoke(0) == "non-positive: 0"
    assert branch.invoke(-3) == "non-positive: -3"


def test_branch_batch_different_routes() -> None:
    """Test branch batch where inputs route to different branches."""
    branch = RunnableBranch(
        (lambda x: x > 10, lambda x: "large"),
        (lambda x: x > 0, lambda x: "small"),
        lambda x: "negative",
    )

    results = branch.batch([15, 5, -3, 0, 20])
    assert results == ["large", "small", "negative", "negative", "large"]


def test_branch_invalid_branch_structure() -> None:
    """Test branch initialization with invalid branch structures."""
    # Branch must be a tuple or list
    with pytest.raises(TypeError, match="must be tuples or lists"):
        RunnableBranch(  # type: ignore[arg-type]
            lambda x: x > 0,  # Not a tuple
            lambda x: x,
        )

    # Branch must have exactly 2 elements
    with pytest.raises(ValueError, match="length 2"):
        RunnableBranch(  # type: ignore[arg-type]
            (lambda x: x > 0,),  # Only 1 element
            lambda x: x,
        )

    # Branch must have exactly 2 elements
    with pytest.raises(ValueError, match="length 2"):
        RunnableBranch(  # type: ignore[arg-type]
            (lambda x: x > 0, lambda x: x + 1, lambda x: x + 2),  # 3 elements
            lambda x: x,
        )


def test_branch_invalid_default() -> None:
    """Test branch with invalid default."""
    # Default must be Runnable, callable, or mapping
    with pytest.raises(TypeError, match="must be Runnable, callable or mapping"):
        RunnableBranch(  # type: ignore[arg-type]
            (lambda x: x > 0, lambda x: x + 1),
            42,  # Invalid default (not callable)
        )


def test_branch_stream_accumulates_output() -> None:
    """Test that branch stream properly accumulates output."""
    from langchain_core.language_models import FakeStreamingListLLM

    llm1 = FakeStreamingListLLM(responses=["response one"])
    llm2 = FakeStreamingListLLM(responses=["response two"])

    branch = RunnableBranch(
        (lambda x: x == "a", lambda _: llm1),
        lambda _: llm2,
    )

    # Route to llm1
    result1 = list(branch.stream("a"))
    assert "".join(result1) == "response one"

    # Route to llm2 (default)
    result2 = list(branch.stream("b"))
    assert "".join(result2) == "response two"


async def test_branch_astream_accumulates_output() -> None:
    """Test that async branch stream accumulates output."""
    from langchain_core.language_models import FakeStreamingListLLM

    llm1 = FakeStreamingListLLM(responses=["response one"])
    llm2 = FakeStreamingListLLM(responses=["response two"])

    branch = RunnableBranch(
        (lambda x: x == "a", lambda _: llm1),
        lambda _: llm2,
    )

    result1 = [x async for x in branch.astream("a")]
    assert "".join(result1) == "response one"

    result2 = [x async for x in branch.astream("b")]
    assert "".join(result2) == "response two"


def test_branch_with_runnables_in_chain() -> None:
    """Test RunnableBranch as part of a larger chain."""
    # Preprocessing
    preprocess = RunnableLambda(lambda x: x["value"])

    # Branch
    branch = RunnableBranch(
        (lambda x: x > 10, lambda x: x * 2),
        (lambda x: x > 0, lambda x: x + 10),
        lambda x: x - 10,
    )

    # Postprocess
    postprocess = RunnableLambda(lambda x: f"Result: {x}")

    chain = preprocess | branch | postprocess

    assert chain.invoke({"value": 15}) == "Result: 30"
    assert chain.invoke({"value": 5}) == "Result: 15"
    assert chain.invoke({"value": -5}) == "Result: -15"


async def test_branch_coerces_conditions_and_actions() -> None:
    """Test that branch coerces callables to Runnables."""
    # Using plain functions (not wrapped in RunnableLambda)
    branch = RunnableBranch(
        (lambda x: x > 0, lambda x: x + 1),
        lambda x: x - 1,
    )

    # All should be coerced to Runnables
    assert all(
        isinstance(cond, RunnableLambda) and isinstance(action, RunnableLambda)
        for cond, action in branch.branches
    )
    assert isinstance(branch.default, RunnableLambda)


def test_branch_with_type_annotations() -> None:
    """Test branch with properly typed functions."""

    def condition_typed(x: int) -> bool:
        return x > 0

    def action_typed(x: int) -> str:
        return f"positive: {x}"

    def default_typed(x: int) -> str:
        return f"non-positive: {x}"

    branch = RunnableBranch[int, str](
        (condition_typed, action_typed),
        default_typed,
    )

    assert branch.invoke(5) == "positive: 5"
    assert branch.invoke(-5) == "non-positive: -5"


def test_branch_empty_batch() -> None:
    """Test branch with empty batch input."""
    branch = RunnableBranch(
        (lambda x: x > 0, lambda x: x + 1),
        lambda x: x,
    )

    results = branch.batch([])
    assert results == []


def test_branch_preserves_intermediate_types() -> None:
    """Test that branch preserves types through conditions and actions."""

    def str_condition(x: str) -> bool:
        return len(x) > 5

    def long_string_action(x: str) -> str:
        return x.upper()

    def short_string_action(x: str) -> str:
        return x.lower()

    branch = RunnableBranch(
        (str_condition, long_string_action),
        short_string_action,
    )

    assert branch.invoke("hello world") == "HELLO WORLD"
    assert branch.invoke("hi") == "hi"


async def test_branch_mixed_sync_async() -> None:
    """Test branch with mix of sync and async functions."""

    def sync_condition(x: int) -> bool:
        return x > 0

    async def async_action(x: int) -> int:
        return x + 1

    def sync_default(x: int) -> int:
        return x - 1

    branch = RunnableBranch(
        (sync_condition, async_action),
        sync_default,
    )

    # Should work via ainvoke
    result = await branch.ainvoke(5)
    assert result == 6

    result2 = await branch.ainvoke(-5)
    assert result2 == -6


def test_branch_condition_returning_non_bool() -> None:
    """Test branch with condition that returns truthy/falsy values."""
    # Python treats non-zero as True, zero as False
    branch = RunnableBranch(
        (lambda x: x, lambda x: "truthy"),  # x itself is the condition
        lambda x: "falsy",
    )

    assert branch.invoke(1) == "truthy"
    assert branch.invoke(5) == "truthy"
    assert branch.invoke(0) == "falsy"


def test_branch_with_callbacks() -> None:
    """Test branch with callbacks/tracing."""
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.callbacks.manager import CallbackManagerForChainRun

    class FakeTracer(BaseCallbackHandler):
        def __init__(self) -> None:
            super().__init__()
            self.runs: list[Any] = []

    tracer = FakeTracer()

    branch = RunnableBranch(
        (lambda x: x > 0, lambda x: x + 1),
        lambda x: x - 1,
    )

    # Should not raise, even with tracer
    result = branch.invoke(5, {"callbacks": [tracer]})
    assert result == 6


def test_branch_all_conditions_false_uses_default() -> None:
    """Test that default is used when all conditions are False."""
    branch = RunnableBranch(
        (lambda x: x > 100, lambda x: "a"),
        (lambda x: x > 50, lambda x: "b"),
        (lambda x: x > 25, lambda x: "c"),
        lambda x: "default",
    )

    result = branch.invoke(10)
    assert result == "default"


def test_branch_init_with_list_branches() -> None:
    """Test branch can be initialized with list tuples."""
    # Use list instead of tuple for branches
    branch = RunnableBranch(
        [lambda x: x > 0, lambda x: x + 1],  # type: ignore[list-item]
        [lambda x: x < 0, lambda x: x - 1],  # type: ignore[list-item]
        lambda x: x,
    )

    assert branch.invoke(5) == 6
    assert branch.invoke(-5) == -6
    assert branch.invoke(0) == 0


def test_branch_with_complex_return_types() -> None:
    """Test branch with complex return types."""

    branch = RunnableBranch(
        (lambda x: x["type"] == "list", lambda x: [x["value"], x["value"] * 2]),
        (lambda x: x["type"] == "dict", lambda x: {"result": x["value"]}),
        lambda x: x["value"],
    )

    assert branch.invoke({"type": "list", "value": 5}) == [5, 10]
    assert branch.invoke({"type": "dict", "value": 5}) == {"result": 5}
    assert branch.invoke({"type": "other", "value": 5}) == 5


async def test_branch_batch_preserves_order() -> None:
    """Test that async batch preserves input order."""
    branch = RunnableBranch(
        (lambda x: x > 5, lambda x: x * 2),
        lambda x: x + 1,
    )

    inputs = [1, 10, 3, 8, 2]
    results = await branch.abatch(inputs)

    # Order should be preserved
    assert results == [2, 20, 4, 16, 3]


def test_branch_with_dict_as_default() -> None:
    """Test branch with dict mapping as default (coerced to RunnableParallel)."""
    branch = RunnableBranch(
        (lambda x: x > 10, lambda x: {"result": "large", "value": x}),
        {"result": lambda x: "small", "value": lambda x: x},  # Will be coerced to RunnableParallel
    )

    result1 = branch.invoke(15)
    assert result1 == {"result": "large", "value": 15}

    result2 = branch.invoke(5)
    assert result2 == {"result": "small", "value": 5}


def test_branch_short_circuit_evaluation() -> None:
    """Test that branch stops evaluating conditions after first match."""
    condition_calls: list[str] = []

    def track_condition(name: str) -> Any:
        def condition(x: int) -> bool:
            condition_calls.append(name)
            return x > 0

        return condition

    branch = RunnableBranch(
        (track_condition("first"), lambda x: x + 1),
        (track_condition("second"), lambda x: x + 2),
        (track_condition("third"), lambda x: x + 3),
        lambda x: x,
    )

    condition_calls.clear()
    result = branch.invoke(5)
    assert result == 6
    # Should only evaluate first condition
    assert condition_calls == ["first"]


def test_branch_with_none_output() -> None:
    """Test branch that can return None."""
    branch = RunnableBranch(
        (lambda x: x == "return_none", lambda x: None),
        lambda x: x,
    )

    result = branch.invoke("return_none")
    assert result is None

    result2 = branch.invoke("other")
    assert result2 == "other"


async def test_branch_abatch_concurrent_execution() -> None:
    """Test that abatch executes concurrently."""
    import time

    start_time = time.time()

    async def slow_action(x: int) -> int:
        await asyncio.sleep(0.1)
        return x + 1

    branch = RunnableBranch(
        (lambda x: x > 0, slow_action),
        lambda x: x,
    )

    results = await branch.abatch([1, 2, 3, 4, 5])
    elapsed = time.time() - start_time

    assert results == [2, 3, 4, 5, 6]
    # Should execute concurrently, so ~0.1s not ~0.5s
    # Allow more margin for test stability
    assert elapsed < 0.5
