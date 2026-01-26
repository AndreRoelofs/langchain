"""Comprehensive tests for RunnableRetry functionality."""

import asyncio
from typing import Any

import pytest
from pytest_mock import MockerFixture

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.retry import RunnableRetry


def test_retry_initialization() -> None:
    """Test RunnableRetry initialization with various parameters."""

    def simple_func(x: int) -> int:
        return x + 1

    runnable = RunnableLambda(simple_func)

    # Test default initialization
    retry_runnable = runnable.with_retry()
    assert isinstance(retry_runnable, RunnableRetry)
    assert retry_runnable.max_attempt_number == 3
    assert retry_runnable.wait_exponential_jitter is True

    # Test custom parameters
    retry_runnable = runnable.with_retry(
        stop_after_attempt=5,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError, TypeError),
    )
    assert retry_runnable.max_attempt_number == 5
    assert retry_runnable.wait_exponential_jitter is False
    assert retry_runnable.retry_exception_types == (ValueError, TypeError)


def test_retry_invoke_success_no_retry(mocker: MockerFixture) -> None:
    """Test that successful invocations don't trigger retries."""

    def succeeds(x: int) -> int:
        return x * 2

    mock_func = mocker.Mock(side_effect=succeeds)
    runnable = RunnableLambda(mock_func)
    retry_runnable = runnable.with_retry(stop_after_attempt=3)

    result = retry_runnable.invoke(5)
    assert result == 10
    assert mock_func.call_count == 1


def test_retry_invoke_with_retryable_exception(mocker: MockerFixture) -> None:
    """Test that retryable exceptions trigger retries."""
    call_count = 0

    def fails_twice(x: int) -> int:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            msg = f"Attempt {call_count} failed"
            raise ValueError(msg)
        return x * 2

    runnable = RunnableLambda(fails_twice)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    )

    result = retry_runnable.invoke(5)
    assert result == 10
    assert call_count == 3


def test_retry_invoke_exhausts_retries(mocker: MockerFixture) -> None:
    """Test that exceptions are raised after exhausting retries."""

    def always_fails(_: int) -> int:
        msg = "Always fails"
        raise ValueError(msg)

    mock_func = mocker.Mock(side_effect=always_fails)
    runnable = RunnableLambda(mock_func)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    )

    with pytest.raises(ValueError, match="Always fails"):
        retry_runnable.invoke(5)

    assert mock_func.call_count == 2


def test_retry_invoke_non_retryable_exception(mocker: MockerFixture) -> None:
    """Test that non-retryable exceptions are not retried."""

    def raises_runtime_error(_: int) -> int:
        msg = "Runtime error"
        raise RuntimeError(msg)

    mock_func = mocker.Mock(side_effect=raises_runtime_error)
    runnable = RunnableLambda(mock_func)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    )

    with pytest.raises(RuntimeError, match="Runtime error"):
        retry_runnable.invoke(5)

    assert mock_func.call_count == 1


async def test_retry_ainvoke_success_no_retry(mocker: MockerFixture) -> None:
    """Test that successful async invocations don't trigger retries."""
    call_count = 0

    async def succeeds(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    runnable = RunnableLambda(succeeds)
    retry_runnable = runnable.with_retry(stop_after_attempt=3)

    result = await retry_runnable.ainvoke(5)
    assert result == 10
    assert call_count == 1


async def test_retry_ainvoke_with_retryable_exception(mocker: MockerFixture) -> None:
    """Test that async retryable exceptions trigger retries."""
    call_count = 0

    async def fails_twice(x: int) -> int:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            msg = f"Attempt {call_count} failed"
            raise ValueError(msg)
        return x * 2

    runnable = RunnableLambda(fails_twice)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    )

    result = await retry_runnable.ainvoke(5)
    assert result == 10
    assert call_count == 3


async def test_retry_ainvoke_exhausts_retries() -> None:
    """Test that async exceptions are raised after exhausting retries."""

    async def always_fails(_: int) -> int:
        msg = "Always fails"
        raise ValueError(msg)

    runnable = RunnableLambda(always_fails)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    )

    with pytest.raises(ValueError, match="Always fails"):
        await retry_runnable.ainvoke(5)


def test_retry_batch_partial_failures() -> None:
    """Test batch with some inputs failing and being retried."""
    call_counts: dict[int, int] = {}

    def sometimes_fails(x: int) -> int:
        call_counts[x] = call_counts.get(x, 0) + 1
        # Fail on first attempt for input 1 and 2
        if x in {1, 2} and call_counts[x] < 2:
            msg = f"Fail {x} on attempt {call_counts[x]}"
            raise ValueError(msg)
        return x * 2

    runnable = RunnableLambda(sometimes_fails)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    )

    results = retry_runnable.batch([0, 1, 2, 3])
    assert results == [0, 2, 4, 6]
    assert call_counts[0] == 1  # No retry needed
    assert call_counts[1] == 2  # Retried once
    assert call_counts[2] == 2  # Retried once
    assert call_counts[3] == 1  # No retry needed


def test_retry_batch_with_return_exceptions() -> None:
    """Test batch with return_exceptions=True."""

    def fails_on_one(_x: int) -> int:
        if _x == 1:
            msg = "Always fails on 1"
            raise ValueError(msg)
        return _x * 2

    runnable = RunnableLambda(fails_on_one)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    )

    results = retry_runnable.batch([0, 1, 2], return_exceptions=True)
    assert results[0] == 0
    assert isinstance(results[1], ValueError)
    assert results[2] == 4


async def test_retry_abatch_partial_failures() -> None:
    """Test async batch with some inputs failing and being retried."""
    call_counts: dict[int, int] = {}

    async def sometimes_fails(x: int) -> int:
        call_counts[x] = call_counts.get(x, 0) + 1
        # Fail on first attempt for input 1 and 2
        if x in {1, 2} and call_counts[x] < 2:
            msg = f"Fail {x} on attempt {call_counts[x]}"
            raise ValueError(msg)
        return x * 2

    runnable = RunnableLambda(sometimes_fails)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    )

    results = await retry_runnable.abatch([0, 1, 2, 3])
    assert results == [0, 2, 4, 6]
    assert call_counts[0] == 1  # No retry needed
    assert call_counts[1] == 2  # Retried once
    assert call_counts[2] == 2  # Retried once
    assert call_counts[3] == 1  # No retry needed


async def test_retry_abatch_with_return_exceptions() -> None:
    """Test async batch with return_exceptions=True."""

    async def fails_on_one(_x: int) -> int:
        if _x == 1:
            msg = "Always fails on 1"
            raise ValueError(msg)
        return _x * 2

    runnable = RunnableLambda(fails_on_one)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    )

    results = await retry_runnable.abatch([0, 1, 2], return_exceptions=True)
    assert results[0] == 0
    assert isinstance(results[1], ValueError)
    assert results[2] == 4


def test_retry_with_exponential_jitter() -> None:
    """Test retry with exponential jitter enabled."""
    call_count = 0

    def fails_once(_: int) -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = "First attempt fails"
            raise ValueError(msg)
        return 42

    runnable = RunnableLambda(fails_once)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": 0.01, "max": 0.1},
        retry_if_exception_type=(ValueError,),
    )

    result = retry_runnable.invoke(1)
    assert result == 42
    assert call_count == 2


async def test_retry_async_with_exponential_jitter() -> None:
    """Test async retry with exponential jitter enabled."""
    call_count = 0

    async def fails_once(_: int) -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = "First attempt fails"
            raise ValueError(msg)
        return 42

    runnable = RunnableLambda(fails_once)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": 0.01, "max": 0.1},
        retry_if_exception_type=(ValueError,),
    )

    result = await retry_runnable.ainvoke(1)
    assert result == 42
    assert call_count == 2


def test_retry_with_config(mocker: MockerFixture) -> None:
    """Test that retry works properly with RunnableConfig."""

    def increment(x: int) -> int:
        return x + 1

    spy = mocker.spy(RunnableLambda, "invoke")
    runnable = RunnableLambda(increment)
    retry_runnable = runnable.with_retry(stop_after_attempt=2)

    config: RunnableConfig = {"tags": ["test-tag"], "metadata": {"key": "value"}}
    result = retry_runnable.invoke(5, config)

    assert result == 6
    # Verify config was passed through
    assert "test-tag" in spy.call_args[0][2]["tags"]


async def test_retry_async_with_config(mocker: MockerFixture) -> None:
    """Test that async retry works properly with RunnableConfig."""

    async def increment(x: int) -> int:
        return x + 1

    spy = mocker.spy(RunnableLambda, "ainvoke")
    runnable = RunnableLambda(increment)
    retry_runnable = runnable.with_retry(stop_after_attempt=2)

    config: RunnableConfig = {"tags": ["test-tag"], "metadata": {"key": "value"}}
    result = await retry_runnable.ainvoke(5, config)

    assert result == 6
    # Verify config was passed through
    assert "test-tag" in spy.call_args[0][2]["tags"]


def test_retry_multiple_exception_types() -> None:
    """Test retry with multiple exception types."""
    call_count = 0

    def fails_with_different_errors(x: int) -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = "ValueError"
            raise ValueError(msg)
        if call_count == 2:
            msg = "TypeError"
            raise TypeError(msg)
        return x * 2

    runnable = RunnableLambda(fails_with_different_errors)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError, TypeError),
    )

    result = retry_runnable.invoke(5)
    assert result == 10
    assert call_count == 3


def test_retry_batch_preserves_order() -> None:
    """Test that retry batch preserves input order even when some fail."""
    first_fail: set[int] = {1}

    def sometimes_fail(x: int) -> int:
        if x in first_fail:
            first_fail.remove(x)
            msg = "fail once"
            raise ValueError(msg)
        return x

    runnable = RunnableLambda(sometimes_fail)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    )

    results = retry_runnable.batch([0, 1, 2])
    assert results == [0, 1, 2]


async def test_retry_abatch_preserves_order() -> None:
    """Test that async retry batch preserves input order."""
    first_fail: set[int] = {1}

    async def sometimes_fail(x: int) -> int:
        if x in first_fail:
            first_fail.remove(x)
            msg = "fail once"
            raise ValueError(msg)
        return x

    runnable = RunnableLambda(sometimes_fail)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    )

    results = await retry_runnable.abatch([0, 1, 2])
    assert results == [0, 1, 2]


def test_retry_does_not_catch_base_exception() -> None:
    """Test that retry doesn't catch BaseException like KeyboardInterrupt."""

    def raises_keyboard_interrupt(_: Any) -> Any:
        raise KeyboardInterrupt

    runnable = RunnableLambda(raises_keyboard_interrupt)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=3,
        retry_if_exception_type=(Exception,),
    )

    with pytest.raises(KeyboardInterrupt):
        retry_runnable.invoke(1)


def test_retry_with_bind() -> None:
    """Test that retry works with bound kwargs."""

    def add_numbers(x: int, *, extra: int = 0) -> int:
        return x + extra

    runnable = RunnableLambda(add_numbers).bind(extra=10)
    retry_runnable = runnable.with_retry(stop_after_attempt=2)

    result = retry_runnable.invoke(5)
    assert result == 15


async def test_retry_async_with_bind() -> None:
    """Test that async retry works with bound kwargs."""

    async def add_numbers(x: int, *, extra: int = 0) -> int:
        return x + extra

    runnable = RunnableLambda(add_numbers).bind(extra=10)
    retry_runnable = runnable.with_retry(stop_after_attempt=2)

    result = await retry_runnable.ainvoke(5)
    assert result == 15


def test_retry_chain_composition() -> None:
    """Test retry in a chain composition."""
    call_count = 0

    def unreliable_step(x: int) -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = "First attempt fails"
            raise ValueError(msg)
        return x * 2

    def reliable_step(x: int) -> int:
        return x + 1

    chain = (
        RunnableLambda(reliable_step)
        | RunnableLambda(unreliable_step).with_retry(
            stop_after_attempt=2,
            wait_exponential_jitter=False,
            retry_if_exception_type=(ValueError,),
        )
        | RunnableLambda(reliable_step)
    )

    result = chain.invoke(5)
    assert result == 13  # (5 + 1) * 2 + 1
    assert call_count == 2


def test_retry_batch_all_fail() -> None:
    """Test batch where all inputs fail even after retries."""

    def always_fails(_: int) -> int:
        msg = "Always fails"
        raise ValueError(msg)

    runnable = RunnableLambda(always_fails)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    )

    with pytest.raises(ValueError, match="Always fails"):
        retry_runnable.batch([1, 2, 3])


def test_retry_batch_empty_input() -> None:
    """Test retry batch with empty input."""

    def identity(x: int) -> int:
        return x

    runnable = RunnableLambda(identity)
    retry_runnable = runnable.with_retry()

    results = retry_runnable.batch([])
    assert results == []


async def test_retry_abatch_empty_input() -> None:
    """Test async retry batch with empty input."""

    async def identity(x: int) -> int:
        return x

    runnable = RunnableLambda(identity)
    retry_runnable = runnable.with_retry()

    results = await retry_runnable.abatch([])
    assert results == []


def test_retry_preserves_schemas() -> None:
    """Test that retry preserves input and output schemas."""

    def typed_func(x: int) -> str:
        return str(x)

    runnable = RunnableLambda(typed_func)
    retry_runnable = runnable.with_retry()

    assert retry_runnable.get_input_schema() == runnable.get_input_schema()
    assert retry_runnable.get_output_schema() == runnable.get_output_schema()


def test_retry_stream_and_transform_not_retried() -> None:
    """Test that stream and transform are not retried (as per implementation)."""
    call_count = 0

    def fails_once(x: int) -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = "First attempt fails"
            raise ValueError(msg)
        return x * 2

    runnable = RunnableLambda(fails_once)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    )

    # Stream should not retry, so it should fail immediately
    with pytest.raises(ValueError, match="First attempt fails"):
        list(retry_runnable.stream(5))

    assert call_count == 1


def test_retry_config_propagation() -> None:
    """Test that config is properly propagated through retries."""
    attempts: list[dict[str, Any]] = []

    def track_config(x: int, config: RunnableConfig) -> int:
        attempts.append({"x": x, "tags": config.get("tags", [])})
        if len(attempts) < 2:
            msg = "First attempt fails"
            raise ValueError(msg)
        return x * 2

    runnable = RunnableLambda(track_config)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    )

    config: RunnableConfig = {"tags": ["my-tag"]}
    result = retry_runnable.invoke(5, config)

    assert result == 10
    assert len(attempts) == 2
    # Both attempts should have the tag
    assert all("my-tag" in attempt["tags"] for attempt in attempts)


def test_retry_nested_with_other_bindings() -> None:
    """Test retry combined with other binding types."""

    def add_one(x: int) -> int:
        return x + 1

    runnable = RunnableLambda(add_one)
    # Combine with_config, bind, and with_retry
    complex_runnable = (
        runnable.with_config(tags=["base-tag"])
        .bind()
        .with_retry(stop_after_attempt=2)
    )

    result = complex_runnable.invoke(5)
    assert result == 6


def test_retry_batch_individual_tracking() -> None:
    """Test that each input in batch is tracked separately for retries."""
    call_tracker: dict[int, list[int]] = {0: [], 1: [], 2: []}

    def track_calls(x: int) -> int:
        call_tracker[x].append(x)
        # First input fails twice, second input fails once, third succeeds immediately
        if x == 0 and len(call_tracker[x]) < 3:
            msg = "Fail twice"
            raise ValueError(msg)
        if x == 1 and len(call_tracker[x]) < 2:
            msg = "Fail once"
            raise ValueError(msg)
        return x * 2

    runnable = RunnableLambda(track_calls)
    retry_runnable = runnable.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    )

    results = retry_runnable.batch([0, 1, 2])
    assert results == [0, 2, 4]
    assert len(call_tracker[0]) == 3  # Failed twice, succeeded on third
    assert len(call_tracker[1]) == 2  # Failed once, succeeded on second
    assert len(call_tracker[2]) == 1  # Succeeded immediately
