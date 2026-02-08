"""Test concurrency behavior of batch and async batch operations."""

import asyncio
import time
from threading import Lock
from typing import TYPE_CHECKING, Any

import pytest

from langchain_core.runnables import RunnableConfig, RunnableLambda

if TYPE_CHECKING:
    from langchain_core.runnables.base import Runnable


@pytest.mark.asyncio
async def test_abatch_concurrency() -> None:
    """Test that abatch respects max_concurrency."""
    running_tasks = 0
    max_running_tasks = 0
    lock = asyncio.Lock()

    async def tracked_function(x: Any) -> str:
        nonlocal running_tasks, max_running_tasks
        async with lock:
            running_tasks += 1
            max_running_tasks = max(max_running_tasks, running_tasks)

        await asyncio.sleep(0.1)  # Simulate work

        async with lock:
            running_tasks -= 1

        return f"Completed {x}"

    runnable: Runnable = RunnableLambda(tracked_function)
    num_tasks = 10
    max_concurrency = 3

    config = RunnableConfig(max_concurrency=max_concurrency)
    results = await runnable.abatch(list(range(num_tasks)), config=config)

    assert len(results) == num_tasks
    assert max_running_tasks <= max_concurrency


@pytest.mark.asyncio
async def test_abatch_as_completed_concurrency() -> None:
    """Test that abatch_as_completed respects max_concurrency."""
    running_tasks = 0
    max_running_tasks = 0
    lock = asyncio.Lock()

    async def tracked_function(x: Any) -> str:
        nonlocal running_tasks, max_running_tasks
        async with lock:
            running_tasks += 1
            max_running_tasks = max(max_running_tasks, running_tasks)

        await asyncio.sleep(0.1)  # Simulate work

        async with lock:
            running_tasks -= 1

        return f"Completed {x}"

    runnable: Runnable = RunnableLambda(tracked_function)
    num_tasks = 10
    max_concurrency = 3

    config = RunnableConfig(max_concurrency=max_concurrency)
    results = []
    async for _idx, result in runnable.abatch_as_completed(
        list(range(num_tasks)), config=config
    ):
        results.append(result)

    assert len(results) == num_tasks
    assert max_running_tasks <= max_concurrency


def test_batch_concurrency() -> None:
    """Test that batch respects max_concurrency."""
    running_tasks = 0
    max_running_tasks = 0

    lock = Lock()

    def tracked_function(x: Any) -> str:
        nonlocal running_tasks, max_running_tasks
        with lock:
            running_tasks += 1
            max_running_tasks = max(max_running_tasks, running_tasks)

        time.sleep(0.1)  # Simulate work

        with lock:
            running_tasks -= 1

        return f"Completed {x}"

    runnable: Runnable = RunnableLambda(tracked_function)
    num_tasks = 10
    max_concurrency = 3

    config = RunnableConfig(max_concurrency=max_concurrency)
    results = runnable.batch(list(range(num_tasks)), config=config)

    assert len(results) == num_tasks
    assert max_running_tasks <= max_concurrency


def test_batch_as_completed_concurrency() -> None:
    """Test that batch_as_completed respects max_concurrency."""
    running_tasks = 0
    max_running_tasks = 0

    lock = Lock()

    def tracked_function(x: Any) -> str:
        nonlocal running_tasks, max_running_tasks
        with lock:
            running_tasks += 1
            max_running_tasks = max(max_running_tasks, running_tasks)

        time.sleep(0.1)  # Simulate work

        with lock:
            running_tasks -= 1

        return f"Completed {x}"

    runnable: Runnable = RunnableLambda(tracked_function)
    num_tasks = 10
    max_concurrency = 3

    config = RunnableConfig(max_concurrency=max_concurrency)
    results = []
    for _idx, result in runnable.batch_as_completed(
        list(range(num_tasks)), config=config
    ):
        results.append(result)

    assert len(results) == num_tasks
    assert max_running_tasks <= max_concurrency


# ---------------------------------------------------------------------------
# Extended concurrency tests
# ---------------------------------------------------------------------------


def test_batch_empty_input() -> None:
    """batch with empty input returns empty list."""
    runnable: Runnable = RunnableLambda(lambda x: x)
    results = runnable.batch([])
    assert results == []


@pytest.mark.asyncio
async def test_abatch_empty_input() -> None:
    """abatch with empty input returns empty list."""

    async def identity(x: Any) -> Any:
        return x

    runnable: Runnable = RunnableLambda(identity)
    results = await runnable.abatch([])
    assert results == []


def test_batch_single_item_no_threading() -> None:
    """batch with a single item should not use threading overhead."""

    def func(x: int) -> int:
        return x * 2

    runnable: Runnable = RunnableLambda(func)
    results = runnable.batch([5])
    assert results == [10]


def test_batch_preserves_order() -> None:
    """batch results should correspond to input order."""

    def slow_func(x: int) -> int:
        time.sleep(0.01 * (10 - x))  # Reverse sleep so later items finish first
        return x

    runnable: Runnable = RunnableLambda(slow_func)
    inputs = list(range(10))
    results = runnable.batch(inputs)
    assert results == inputs


@pytest.mark.asyncio
async def test_abatch_preserves_order() -> None:
    """abatch results should correspond to input order."""

    async def slow_func(x: int) -> int:
        await asyncio.sleep(0.01 * (10 - x))
        return x

    runnable: Runnable = RunnableLambda(slow_func)
    inputs = list(range(10))
    results = await runnable.abatch(inputs)
    assert results == inputs


def test_batch_with_return_exceptions() -> None:
    """batch with return_exceptions=True captures exceptions in order."""
    call_count = 0

    def sometimes_fail(x: int) -> int:
        nonlocal call_count
        call_count += 1
        if x % 2 == 0:
            msg = f"failed on {x}"
            raise ValueError(msg)
        return x

    runnable: Runnable = RunnableLambda(sometimes_fail)
    results = runnable.batch([1, 2, 3, 4], return_exceptions=True)
    assert results[0] == 1
    assert isinstance(results[1], ValueError)
    assert results[2] == 3
    assert isinstance(results[3], ValueError)


@pytest.mark.asyncio
async def test_abatch_with_return_exceptions() -> None:
    """abatch with return_exceptions=True captures exceptions in order."""

    async def sometimes_fail(x: int) -> int:
        if x % 2 == 0:
            msg = f"failed on {x}"
            raise ValueError(msg)
        return x

    runnable: Runnable = RunnableLambda(sometimes_fail)
    results = await runnable.abatch([1, 2, 3, 4], return_exceptions=True)
    assert results[0] == 1
    assert isinstance(results[1], ValueError)
    assert results[2] == 3
    assert isinstance(results[3], ValueError)


def test_batch_no_concurrency_limit() -> None:
    """batch without max_concurrency runs all items."""

    def func(x: int) -> int:
        return x + 1

    runnable: Runnable = RunnableLambda(func)
    results = runnable.batch(list(range(20)))
    assert results == list(range(1, 21))


def test_batch_concurrency_of_one() -> None:
    """batch with max_concurrency=1 runs items sequentially."""
    order: list[int] = []

    lock = Lock()

    def track_order(x: int) -> int:
        with lock:
            order.append(x)
        time.sleep(0.01)
        return x

    runnable: Runnable = RunnableLambda(track_order)
    config = RunnableConfig(max_concurrency=1)
    results = runnable.batch(list(range(5)), config=config)
    assert results == list(range(5))
    # With concurrency of 1, order should be sequential
    assert order == list(range(5))


def test_batch_as_completed_returns_all() -> None:
    """batch_as_completed returns all results (possibly out of order)."""

    def func(x: int) -> int:
        time.sleep(0.01 * (10 - x))
        return x * 2

    runnable: Runnable = RunnableLambda(func)
    collected = dict(runnable.batch_as_completed(list(range(5))))
    assert len(collected) == 5
    for i in range(5):
        assert collected[i] == i * 2


@pytest.mark.asyncio
async def test_abatch_as_completed_returns_all() -> None:
    """abatch_as_completed returns all results."""

    async def func(x: int) -> int:
        await asyncio.sleep(0.01 * (5 - x))
        return x * 2

    runnable: Runnable = RunnableLambda(func)
    collected: dict[int, Any] = {}
    async for idx, result in runnable.abatch_as_completed(list(range(5))):
        collected[idx] = result
    assert len(collected) == 5
    for i in range(5):
        assert collected[i] == i * 2
