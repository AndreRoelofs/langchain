"""Extended tests for InMemoryRateLimiter covering edge cases and comprehensive scenarios."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from freezegun import freeze_time

from langchain_core.rate_limiters import InMemoryRateLimiter


def test_initialization_defaults() -> None:
    """Test that InMemoryRateLimiter initializes with default values."""
    rate_limiter = InMemoryRateLimiter()
    assert rate_limiter.requests_per_second == 1
    assert rate_limiter.check_every_n_seconds == 0.1
    assert rate_limiter.max_bucket_size == 1
    assert rate_limiter.available_tokens == 0.0
    assert rate_limiter.last is None


def test_initialization_custom_values() -> None:
    """Test initialization with custom parameter values."""
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=5.5,
        check_every_n_seconds=0.05,
        max_bucket_size=100,
    )
    assert rate_limiter.requests_per_second == 5.5
    assert rate_limiter.check_every_n_seconds == 0.05
    assert rate_limiter.max_bucket_size == 100
    assert rate_limiter.available_tokens == 0.0
    assert rate_limiter.last is None


def test_first_acquire_non_blocking_fails() -> None:
    """Test that the first non-blocking acquire fails as bucket is empty."""
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=10, check_every_n_seconds=0.1, max_bucket_size=10
    )
    # First call should fail because bucket starts empty
    assert rate_limiter.acquire(blocking=False) is False
    assert rate_limiter.available_tokens == 0.0


async def test_first_aacquire_non_blocking_fails() -> None:
    """Test that the first async non-blocking acquire fails as bucket is empty."""
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=10, check_every_n_seconds=0.1, max_bucket_size=10
    )
    # First call should fail because bucket starts empty
    assert await rate_limiter.aacquire(blocking=False) is False
    assert rate_limiter.available_tokens == 0.0


def test_blocking_acquire_waits_and_succeeds() -> None:
    """Test that blocking acquire waits for tokens and succeeds."""
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=10, check_every_n_seconds=0.01, max_bucket_size=10
        )
        rate_limiter.last = time.time()

        # Start blocking acquire in a thread
        def blocking_call() -> bool:
            return rate_limiter.acquire(blocking=True)

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(blocking_call)

        # Give thread time to start
        time.sleep(0.02)

        # Advance time to allow token generation
        frozen_time.tick(0.15)

        # Wait for blocking call to complete
        result = future.result(timeout=1)
        assert result is True
        executor.shutdown(wait=True)


async def test_blocking_aacquire_waits_and_succeeds() -> None:
    """Test that async blocking acquire waits for tokens and succeeds."""
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=10, check_every_n_seconds=0.01, max_bucket_size=10
        )
        rate_limiter.last = time.time()

        # Create async task
        async def blocking_call() -> bool:
            return await rate_limiter.aacquire(blocking=True)

        task = asyncio.create_task(blocking_call())

        # Give task time to start
        await asyncio.sleep(0)

        # Advance time to allow token generation
        frozen_time.tick(0.15)

        # Wait for async call to complete
        result = await asyncio.wait_for(task, timeout=1)
        assert result is True


def test_thread_safety_concurrent_acquires() -> None:
    """Test that the rate limiter is thread-safe with concurrent acquire calls."""
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=100, check_every_n_seconds=0.01, max_bucket_size=50
    )

    # Initialize the bucket with some time passing
    rate_limiter.last = time.monotonic()
    time.sleep(0.5)  # Wait for tokens to accumulate

    successful_acquires = []

    def acquire_token(index: int) -> None:
        if rate_limiter.acquire(blocking=False):
            successful_acquires.append(index)

    # Launch many threads
    threads = []
    num_threads = 100
    for i in range(num_threads):
        thread = threading.Thread(target=acquire_token, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Should have acquired at most max_bucket_size tokens
    assert len(successful_acquires) <= rate_limiter.max_bucket_size
    # All acquires should be unique (no double-counting)
    assert len(successful_acquires) == len(set(successful_acquires))


async def test_async_thread_safety_concurrent_aacquires() -> None:
    """Test that the rate limiter is thread-safe with concurrent async acquires."""
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=100, check_every_n_seconds=0.01, max_bucket_size=50
    )

    # Initialize the bucket with some time passing
    rate_limiter.last = time.monotonic()
    await asyncio.sleep(0.5)  # Wait for tokens to accumulate

    successful_acquires = []

    async def acquire_token(index: int) -> None:
        if await rate_limiter.aacquire(blocking=False):
            successful_acquires.append(index)

    # Launch many async tasks
    tasks = []
    num_tasks = 100
    for i in range(num_tasks):
        task = asyncio.create_task(acquire_token(i))
        tasks.append(task)

    await asyncio.gather(*tasks)

    # Should have acquired at most max_bucket_size tokens
    assert len(successful_acquires) <= rate_limiter.max_bucket_size
    # All acquires should be unique (no double-counting)
    assert len(successful_acquires) == len(set(successful_acquires))


def test_token_bucket_refill_gradual() -> None:
    """Test that tokens are refilled gradually based on elapsed time."""
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=10
        )
        rate_limiter.last = time.time()

        # Initially no tokens
        assert rate_limiter.available_tokens == 0.0

        # After 0.5 seconds, should have 1 token (2 requests/sec * 0.5 sec = 1)
        frozen_time.tick(0.5)
        assert rate_limiter.acquire(blocking=False) is True
        assert rate_limiter.available_tokens == 0.0

        # After another 1 second, should have 2 tokens
        frozen_time.tick(1.0)
        assert rate_limiter.acquire(blocking=False) is True
        assert rate_limiter.available_tokens == 1.0

        # Consume the remaining token
        assert rate_limiter.acquire(blocking=False) is True
        assert rate_limiter.available_tokens == 0.0


async def test_async_token_bucket_refill_gradual() -> None:
    """Test that tokens are refilled gradually in async context."""
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=10
        )
        rate_limiter.last = time.time()

        # Initially no tokens
        assert rate_limiter.available_tokens == 0.0

        # After 0.5 seconds, should have 1 token
        frozen_time.tick(0.5)
        assert await rate_limiter.aacquire(blocking=False) is True
        assert rate_limiter.available_tokens == 0.0

        # After another 1 second, should have 2 tokens
        frozen_time.tick(1.0)
        assert await rate_limiter.aacquire(blocking=False) is True
        assert rate_limiter.available_tokens == 1.0


def test_max_bucket_size_caps_tokens() -> None:
    """Test that max_bucket_size properly caps the number of tokens."""
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=10, check_every_n_seconds=0.1, max_bucket_size=5
        )
        rate_limiter.last = time.time()

        # Wait for a very long time (100 seconds)
        # Should generate 1000 tokens, but capped at max_bucket_size
        frozen_time.tick(100)

        # First acquire
        assert rate_limiter.acquire(blocking=False) is True
        # Should have max_bucket_size - 1 tokens left
        assert rate_limiter.available_tokens == 4.0

        # Continue acquiring until exhausted
        for _ in range(4):
            assert rate_limiter.acquire(blocking=False) is True

        # All tokens consumed
        assert rate_limiter.available_tokens == 0.0
        # Next acquire should fail
        assert rate_limiter.acquire(blocking=False) is False


def test_fractional_requests_per_second() -> None:
    """Test rate limiter with fractional requests per second."""
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        # 0.5 requests per second = 1 request every 2 seconds
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=0.5, check_every_n_seconds=0.1, max_bucket_size=10
        )
        rate_limiter.last = time.time()

        # After 1 second, should have 0.5 tokens (not enough)
        frozen_time.tick(1.0)
        assert rate_limiter.acquire(blocking=False) is False

        # After 2 seconds total, should have 1 token
        frozen_time.tick(1.0)
        assert rate_limiter.acquire(blocking=False) is True
        assert rate_limiter.available_tokens == 0.0

        # After another 4 seconds, should have 2 tokens
        frozen_time.tick(4.0)
        assert rate_limiter.acquire(blocking=False) is True
        assert rate_limiter.available_tokens == 1.0


def test_very_small_check_interval() -> None:
    """Test rate limiter with very small check interval."""
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=100, check_every_n_seconds=0.001, max_bucket_size=10
    )
    # Should initialize without errors
    assert rate_limiter.check_every_n_seconds == 0.001


def test_consume_initializes_last_on_first_call() -> None:
    """Test that _consume initializes 'last' attribute on first call."""
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=10, check_every_n_seconds=0.1, max_bucket_size=10
    )
    assert rate_limiter.last is None

    # First call should initialize 'last'
    rate_limiter.acquire(blocking=False)
    assert rate_limiter.last is not None


def test_multiple_acquires_deplete_bucket() -> None:
    """Test that multiple acquires properly deplete the token bucket."""
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=10, check_every_n_seconds=0.1, max_bucket_size=5
        )
        rate_limiter.last = time.time()

        # Generate 5 tokens (at max capacity)
        frozen_time.tick(0.5)

        # Acquire all 5 tokens
        for i in range(5):
            assert rate_limiter.acquire(blocking=False) is True
            assert rate_limiter.available_tokens == 4 - i

        # Next acquire should fail (bucket empty)
        assert rate_limiter.acquire(blocking=False) is False
        assert rate_limiter.available_tokens == 0.0


async def test_async_multiple_acquires_deplete_bucket() -> None:
    """Test that multiple async acquires properly deplete the token bucket."""
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=10, check_every_n_seconds=0.1, max_bucket_size=5
        )
        rate_limiter.last = time.time()

        # Generate 5 tokens (at max capacity)
        frozen_time.tick(0.5)

        # Acquire all 5 tokens
        for i in range(5):
            assert await rate_limiter.aacquire(blocking=False) is True
            assert rate_limiter.available_tokens == 4 - i

        # Next acquire should fail (bucket empty)
        assert await rate_limiter.aacquire(blocking=False) is False
        assert rate_limiter.available_tokens == 0.0


def test_high_request_rate() -> None:
    """Test rate limiter with high requests per second."""
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=1000, check_every_n_seconds=0.01, max_bucket_size=100
        )
        rate_limiter.last = time.time()

        # After 1 second, should have 100 tokens (capped at max)
        frozen_time.tick(1.0)

        # Should be able to acquire 100 tokens
        for _ in range(100):
            assert rate_limiter.acquire(blocking=False) is True

        # 101st acquire should fail
        assert rate_limiter.acquire(blocking=False) is False


def test_token_accumulation_respects_elapsed_threshold() -> None:
    """Test that tokens only accumulate when elapsed time threshold is met."""
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=10, check_every_n_seconds=0.1, max_bucket_size=10
        )
        rate_limiter.last = time.time()

        # Small time increment (less than needed for 1 token)
        frozen_time.tick(0.05)  # Only 0.5 tokens worth
        assert rate_limiter.acquire(blocking=False) is False
        assert rate_limiter.available_tokens == 0.0

        # Another small increment (still less than 1 token total)
        frozen_time.tick(0.04)  # Total 0.9 tokens worth
        assert rate_limiter.acquire(blocking=False) is False
        assert rate_limiter.available_tokens == 0.0

        # Final increment to cross threshold
        frozen_time.tick(0.02)  # Total 1.1 tokens worth
        assert rate_limiter.acquire(blocking=False) is True
        # Use approximate comparison due to floating point precision
        assert abs(rate_limiter.available_tokens - 0.1) < 0.001


def test_bucket_size_one_single_token_available() -> None:
    """Test rate limiter with max_bucket_size of 1."""
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=1
        )
        rate_limiter.last = time.time()

        # Wait long enough to generate multiple tokens, but capped at 1
        frozen_time.tick(10)

        # First acquire succeeds
        assert rate_limiter.acquire(blocking=False) is True
        assert rate_limiter.available_tokens == 0.0

        # Second acquire immediately fails (bucket size is 1)
        assert rate_limiter.acquire(blocking=False) is False
