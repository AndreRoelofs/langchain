"""Snapshot tests for rate_limiters module.

These tests capture the complete behavior of BaseRateLimiter and
InMemoryRateLimiter to detect any functional regressions.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Tests for module-level __all__ exports."""

    def test_all_exports(self) -> None:
        """Verify __all__ contains exactly the expected public symbols."""
        from langchain_core import rate_limiters

        assert set(rate_limiters.__all__) == {
            "BaseRateLimiter",
            "InMemoryRateLimiter",
        }

    def test_base_rate_limiter_importable(self) -> None:
        """BaseRateLimiter can be imported from the package."""
        from langchain_core.rate_limiters import BaseRateLimiter as _BRL

        assert _BRL is BaseRateLimiter

    def test_in_memory_rate_limiter_importable(self) -> None:
        """InMemoryRateLimiter can be imported from the package."""
        from langchain_core.rate_limiters import InMemoryRateLimiter as _IMRL

        assert _IMRL is InMemoryRateLimiter


# ---------------------------------------------------------------------------
# BaseRateLimiter
# ---------------------------------------------------------------------------


class TestBaseRateLimiter:
    """Tests for the BaseRateLimiter abstract base class."""

    def test_cannot_instantiate(self) -> None:
        """BaseRateLimiter is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseRateLimiter()  # type: ignore[abstract]

    def test_missing_acquire_raises(self) -> None:
        """Subclass without acquire() cannot be instantiated."""

        class Incomplete(BaseRateLimiter):
            async def aacquire(self, *, blocking: bool = True) -> bool:
                return True

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Incomplete()  # type: ignore[abstract]

    def test_missing_aacquire_raises(self) -> None:
        """Subclass without aacquire() cannot be instantiated."""

        class Incomplete(BaseRateLimiter):
            def acquire(self, *, blocking: bool = True) -> bool:
                return True

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Incomplete()  # type: ignore[abstract]

    def test_complete_subclass(self) -> None:
        """A subclass implementing both methods can be instantiated."""

        class Complete(BaseRateLimiter):
            def acquire(self, *, blocking: bool = True) -> bool:
                return True

            async def aacquire(self, *, blocking: bool = True) -> bool:
                return True

        limiter = Complete()
        assert isinstance(limiter, BaseRateLimiter)
        assert limiter.acquire() is True
        assert limiter.acquire(blocking=False) is True

    async def test_complete_subclass_async(self) -> None:
        """Async acquire works on a properly implemented subclass."""

        class Complete(BaseRateLimiter):
            def acquire(self, *, blocking: bool = True) -> bool:
                return blocking

            async def aacquire(self, *, blocking: bool = True) -> bool:
                return blocking

        limiter = Complete()
        assert await limiter.aacquire(blocking=True) is True
        assert await limiter.aacquire(blocking=False) is False

    def test_in_memory_rate_limiter_is_subclass(self) -> None:
        """InMemoryRateLimiter is a subclass of BaseRateLimiter."""
        assert issubclass(InMemoryRateLimiter, BaseRateLimiter)

    def test_in_memory_rate_limiter_instance(self) -> None:
        """InMemoryRateLimiter instance passes isinstance check."""
        limiter = InMemoryRateLimiter()
        assert isinstance(limiter, BaseRateLimiter)


# ---------------------------------------------------------------------------
# InMemoryRateLimiter — initialization
# ---------------------------------------------------------------------------


class TestInMemoryRateLimiterInit:
    """Tests for InMemoryRateLimiter initialization."""

    def test_default_parameters(self) -> None:
        """Default init values match documented defaults."""
        limiter = InMemoryRateLimiter()
        assert limiter.requests_per_second == 1
        assert limiter.check_every_n_seconds == 0.1
        assert limiter.max_bucket_size == 1
        assert limiter.available_tokens == 0.0
        assert limiter.last is None

    def test_custom_parameters(self) -> None:
        """Custom parameters are stored correctly."""
        limiter = InMemoryRateLimiter(
            requests_per_second=5.5,
            check_every_n_seconds=0.05,
            max_bucket_size=100,
        )
        assert limiter.requests_per_second == 5.5
        assert limiter.check_every_n_seconds == 0.05
        assert limiter.max_bucket_size == 100

    def test_keyword_only_parameters(self) -> None:
        """All __init__ parameters are keyword-only."""
        with pytest.raises(TypeError):
            InMemoryRateLimiter(1, 0.1, 1)  # type: ignore[misc]

    def test_has_threading_lock(self) -> None:
        """The _consume_lock attribute is a threading.Lock."""
        limiter = InMemoryRateLimiter()
        assert isinstance(limiter._consume_lock, type(threading.Lock()))

    def test_initial_available_tokens_is_zero(self) -> None:
        """Bucket starts empty regardless of config."""
        limiter = InMemoryRateLimiter(requests_per_second=1000, max_bucket_size=500)
        assert limiter.available_tokens == 0.0

    def test_initial_last_is_none(self) -> None:
        """last is None before any acquire call."""
        limiter = InMemoryRateLimiter()
        assert limiter.last is None


# ---------------------------------------------------------------------------
# InMemoryRateLimiter — _consume() internal method
# ---------------------------------------------------------------------------


class TestConsumeMethod:
    """Tests for the _consume() internal token-consumption logic."""

    def test_first_call_initializes_last(self) -> None:
        """On first _consume(), `last` is set to current monotonic time."""
        limiter = InMemoryRateLimiter()
        assert limiter.last is None
        limiter._consume()
        assert limiter.last is not None

    def test_first_call_returns_false(self) -> None:
        """First _consume() returns False because no time has elapsed."""
        limiter = InMemoryRateLimiter(requests_per_second=10, max_bucket_size=10)
        assert limiter._consume() is False

    def test_first_call_does_not_add_tokens(self) -> None:
        """First _consume() initializes but adds no tokens (anti-burst)."""
        limiter = InMemoryRateLimiter(requests_per_second=10, max_bucket_size=10)
        limiter._consume()
        assert limiter.available_tokens == 0.0

    def test_tokens_not_added_below_threshold(self) -> None:
        """Tokens are NOT added when elapsed * rate < 1."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=1, max_bucket_size=10)
            limiter.last = time.time()

            # 0.5 seconds * 1 req/s = 0.5 < 1 → no tokens added
            frozen_time.tick(0.5)
            assert limiter._consume() is False
            assert limiter.available_tokens == 0.0

    def test_tokens_added_at_threshold(self) -> None:
        """Tokens are added when elapsed * rate >= 1."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=1, max_bucket_size=10)
            limiter.last = time.time()

            # 1.0 seconds * 1 req/s = 1.0 >= 1 → tokens added
            frozen_time.tick(1.0)
            assert limiter._consume() is True
            assert limiter.available_tokens == 0.0

    def test_last_not_updated_below_threshold(self) -> None:
        """self.last is NOT updated when elapsed * rate < 1."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=1, max_bucket_size=10)
            limiter.last = time.time()
            original_last = limiter.last

            # 0.5s * 1 = 0.5 < 1 → last should stay the same
            frozen_time.tick(0.5)
            limiter._consume()
            assert limiter.last == original_last

    def test_last_updated_at_threshold(self) -> None:
        """self.last IS updated when elapsed * rate >= 1."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=1, max_bucket_size=10)
            limiter.last = time.time()
            original_last = limiter.last

            # 1.5s * 1 = 1.5 >= 1 → last should be updated
            frozen_time.tick(1.5)
            limiter._consume()
            assert limiter.last != original_last

    def test_tokens_capped_at_max_bucket_size(self) -> None:
        """available_tokens never exceeds max_bucket_size."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=100, max_bucket_size=5)
            limiter.last = time.time()

            # 10s * 100 = 1000 tokens generated, but capped at 5
            frozen_time.tick(10)
            limiter._consume()  # consumes 1 → 4 left
            assert limiter.available_tokens == 4.0

    def test_consume_decrements_by_one(self) -> None:
        """Each successful _consume() decrements available_tokens by exactly 1."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=10, max_bucket_size=10)
            limiter.last = time.time()
            frozen_time.tick(1.0)  # 10 tokens

            assert limiter._consume() is True
            assert limiter.available_tokens == 9.0
            assert limiter._consume() is True
            assert limiter.available_tokens == 8.0

    def test_repeated_sub_threshold_calls_never_add_tokens(self) -> None:
        """Many rapid _consume() calls without enough elapsed time add no tokens."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=1, max_bucket_size=10)
            limiter.last = time.time()

            # Each tick is 0.3s (0.3 * 1 = 0.3 < 1), call _consume
            for _ in range(3):
                frozen_time.tick(0.3)
                result = limiter._consume()
                # First two: 0.3 and 0.6 → still below threshold
                # Third: 0.9 → still below threshold
            assert result is False
            assert limiter.available_tokens == 0.0

    def test_accumulated_time_crosses_threshold(self) -> None:
        """Time accumulates across sub-threshold calls until threshold is met.

        Because `last` is only updated when the threshold is crossed,
        sub-threshold calls don't reset the clock. Time accumulates properly.
        """
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=1, max_bucket_size=10)
            limiter.last = time.time()

            # 0.3s → no tokens (0.3 < 1)
            frozen_time.tick(0.3)
            assert limiter._consume() is False

            # 0.3s more → still no tokens (0.6 < 1), last not updated
            frozen_time.tick(0.3)
            assert limiter._consume() is False

            # 0.3s more → still no tokens (0.9 < 1)
            frozen_time.tick(0.3)
            assert limiter._consume() is False

            # 0.2s more → now 1.1 >= 1, tokens added
            frozen_time.tick(0.2)
            assert limiter._consume() is True

    def test_exact_token_calculation(self) -> None:
        """Verify token count = elapsed * requests_per_second (capped)."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=4, max_bucket_size=100)
            limiter.last = time.time()

            # 2.5s * 4 req/s = 10 tokens
            frozen_time.tick(2.5)
            limiter._consume()  # consumes 1 → 9 left
            assert limiter.available_tokens == 9.0


# ---------------------------------------------------------------------------
# InMemoryRateLimiter — acquire() sync
# ---------------------------------------------------------------------------


class TestAcquireSync:
    """Tests for acquire() synchronous method."""

    def test_non_blocking_empty_bucket_returns_false(self) -> None:
        """Non-blocking acquire on empty bucket returns False."""
        limiter = InMemoryRateLimiter(requests_per_second=10, max_bucket_size=10)
        assert limiter.acquire(blocking=False) is False

    def test_non_blocking_with_tokens_returns_true(self) -> None:
        """Non-blocking acquire with available tokens returns True."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=10, max_bucket_size=10)
            limiter.last = time.time()
            frozen_time.tick(1.0)
            assert limiter.acquire(blocking=False) is True

    def test_blocking_always_returns_true(self) -> None:
        """Blocking acquire always returns True (it waits until success)."""
        limiter = InMemoryRateLimiter(
            requests_per_second=100, check_every_n_seconds=0.01, max_bucket_size=10
        )
        result = limiter.acquire(blocking=True)
        assert result is True

    def test_blocking_default_is_true(self) -> None:
        """Default blocking parameter is True."""
        limiter = InMemoryRateLimiter(
            requests_per_second=100, check_every_n_seconds=0.01, max_bucket_size=10
        )
        # acquire() with no args should block (same as blocking=True)
        result = limiter.acquire()
        assert result is True

    def test_non_blocking_returns_false_then_true_after_time(self) -> None:
        """Non-blocking returns False first, True after enough time passes."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=2, max_bucket_size=5)
            limiter.last = time.time()

            assert limiter.acquire(blocking=False) is False

            frozen_time.tick(0.5)
            assert limiter.acquire(blocking=False) is True

    def test_blocking_calls_sleep(self) -> None:
        """Blocking acquire calls time.sleep with check_every_n_seconds."""
        limiter = InMemoryRateLimiter(
            requests_per_second=100,
            check_every_n_seconds=0.05,
            max_bucket_size=10,
        )
        sleep_calls: list[float] = []
        original_sleep = time.sleep

        def tracking_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            # Inject tokens so the next _consume succeeds
            limiter.available_tokens = 2.0

        with patch(
            "langchain_core.rate_limiters.time.sleep", side_effect=tracking_sleep
        ):
            limiter.last = time.monotonic()
            limiter.acquire(blocking=True)

        assert len(sleep_calls) >= 1
        assert sleep_calls[0] == 0.05

    def test_depleting_bucket_sync(self) -> None:
        """Consuming all tokens one by one depletes the bucket correctly."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=5, max_bucket_size=5)
            limiter.last = time.time()
            frozen_time.tick(1.0)  # 5 tokens

            for i in range(5):
                assert limiter.acquire(blocking=False) is True
                assert limiter.available_tokens == pytest.approx(4.0 - i)

            assert limiter.acquire(blocking=False) is False
            assert limiter.available_tokens == 0.0


# ---------------------------------------------------------------------------
# InMemoryRateLimiter — aacquire() async
# ---------------------------------------------------------------------------


class TestAcquireAsync:
    """Tests for aacquire() asynchronous method."""

    async def test_non_blocking_empty_bucket_returns_false(self) -> None:
        """Non-blocking aacquire on empty bucket returns False."""
        limiter = InMemoryRateLimiter(requests_per_second=10, max_bucket_size=10)
        assert await limiter.aacquire(blocking=False) is False

    async def test_non_blocking_with_tokens_returns_true(self) -> None:
        """Non-blocking aacquire with available tokens returns True."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=10, max_bucket_size=10)
            limiter.last = time.time()
            frozen_time.tick(1.0)
            assert await limiter.aacquire(blocking=False) is True

    async def test_blocking_always_returns_true(self) -> None:
        """Blocking aacquire always returns True (it waits until success)."""
        limiter = InMemoryRateLimiter(
            requests_per_second=100, check_every_n_seconds=0.01, max_bucket_size=10
        )
        result = await limiter.aacquire(blocking=True)
        assert result is True

    async def test_blocking_default_is_true(self) -> None:
        """Default blocking parameter is True for aacquire."""
        limiter = InMemoryRateLimiter(
            requests_per_second=100, check_every_n_seconds=0.01, max_bucket_size=10
        )
        result = await limiter.aacquire()
        assert result is True

    async def test_non_blocking_returns_false_then_true_after_time(self) -> None:
        """Non-blocking aacquire returns False first, True after time passes."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=2, max_bucket_size=5)
            limiter.last = time.time()

            assert await limiter.aacquire(blocking=False) is False

            frozen_time.tick(0.5)
            assert await limiter.aacquire(blocking=False) is True

    async def test_blocking_calls_asyncio_sleep(self) -> None:
        """Blocking aacquire uses asyncio.sleep with check_every_n_seconds."""
        limiter = InMemoryRateLimiter(
            requests_per_second=100,
            check_every_n_seconds=0.07,
            max_bucket_size=10,
        )
        sleep_calls: list[float] = []
        original_sleep = asyncio.sleep

        async def tracking_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            # Advance time so _consume will succeed next iteration
            limiter.available_tokens = 2.0
            await original_sleep(0)

        with patch(
            "langchain_core.rate_limiters.asyncio.sleep", side_effect=tracking_sleep
        ):
            limiter.last = time.monotonic()
            await limiter.aacquire(blocking=True)

        assert len(sleep_calls) >= 1
        assert sleep_calls[0] == 0.07

    async def test_depleting_bucket_async(self) -> None:
        """Consuming all tokens one by one depletes the bucket correctly (async)."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=5, max_bucket_size=5)
            limiter.last = time.time()
            frozen_time.tick(1.0)  # 5 tokens

            for i in range(5):
                assert await limiter.aacquire(blocking=False) is True
                assert limiter.available_tokens == pytest.approx(4.0 - i)

            assert await limiter.aacquire(blocking=False) is False
            assert limiter.available_tokens == 0.0


# ---------------------------------------------------------------------------
# InMemoryRateLimiter — token bucket refill logic
# ---------------------------------------------------------------------------


class TestTokenBucketRefill:
    """Tests for token bucket refill behavior."""

    def test_gradual_refill(self) -> None:
        """Tokens accumulate gradually based on elapsed time."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=2, max_bucket_size=10)
            limiter.last = time.time()

            # 0.5s * 2 = 1 token
            frozen_time.tick(0.5)
            assert limiter.acquire(blocking=False) is True
            assert limiter.available_tokens == 0.0

            # 1s * 2 = 2 more tokens
            frozen_time.tick(1.0)
            assert limiter.acquire(blocking=False) is True
            assert limiter.available_tokens == 1.0

    def test_max_bucket_size_caps_tokens(self) -> None:
        """Long idle periods don't exceed max_bucket_size."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=10, max_bucket_size=5)
            limiter.last = time.time()
            frozen_time.tick(100)  # 1000 tokens theoretically, capped at 5

            assert limiter.acquire(blocking=False) is True
            assert limiter.available_tokens == 4.0

            for _ in range(4):
                assert limiter.acquire(blocking=False) is True

            assert limiter.available_tokens == 0.0
            assert limiter.acquire(blocking=False) is False

    def test_fractional_requests_per_second(self) -> None:
        """Fractional rate (< 1 req/s) requires longer waits."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            # 0.5 req/s → 1 request every 2 seconds
            limiter = InMemoryRateLimiter(requests_per_second=0.5, max_bucket_size=10)
            limiter.last = time.time()

            # 1 second: 0.5 tokens → not enough
            frozen_time.tick(1.0)
            assert limiter.acquire(blocking=False) is False

            # 2 seconds total: 1.0 token → enough
            frozen_time.tick(1.0)
            assert limiter.acquire(blocking=False) is True

    def test_refill_after_depletion(self) -> None:
        """Tokens refill after being fully consumed."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=2, max_bucket_size=2)
            limiter.last = time.time()

            # Fill and deplete
            frozen_time.tick(1.0)
            assert limiter.acquire(blocking=False) is True
            assert limiter.acquire(blocking=False) is True
            assert limiter.acquire(blocking=False) is False

            # Refill
            frozen_time.tick(1.0)
            assert limiter.acquire(blocking=False) is True

    def test_high_rate(self) -> None:
        """High request rate generates many tokens quickly."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=1000, max_bucket_size=100)
            limiter.last = time.time()

            frozen_time.tick(1.0)
            for _ in range(100):
                assert limiter.acquire(blocking=False) is True

            assert limiter.acquire(blocking=False) is False

    def test_bucket_size_one(self) -> None:
        """Max bucket size of 1 allows only one request at a time."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=2, max_bucket_size=1)
            limiter.last = time.time()

            frozen_time.tick(10)  # Would be 20 tokens, capped at 1
            assert limiter.acquire(blocking=False) is True
            assert limiter.available_tokens == 0.0
            assert limiter.acquire(blocking=False) is False


# ---------------------------------------------------------------------------
# InMemoryRateLimiter — sync/async parity
# ---------------------------------------------------------------------------


class TestSyncAsyncParity:
    """Tests that sync and async paths produce identical behavior."""

    async def test_non_blocking_parity(self) -> None:
        """acquire(blocking=False) and aacquire(blocking=False) behave the same."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            sync_limiter = InMemoryRateLimiter(requests_per_second=2, max_bucket_size=5)
            async_limiter = InMemoryRateLimiter(
                requests_per_second=2, max_bucket_size=5
            )
            sync_limiter.last = time.time()
            async_limiter.last = time.time()

            # Both should fail initially
            assert sync_limiter.acquire(blocking=False) is False
            assert await async_limiter.aacquire(blocking=False) is False

            frozen_time.tick(0.5)

            # Both should succeed with 1 token
            assert sync_limiter.acquire(blocking=False) is True
            assert await async_limiter.aacquire(blocking=False) is True

            # Both should have same state
            assert sync_limiter.available_tokens == async_limiter.available_tokens

    async def test_token_state_parity(self) -> None:
        """Token counts remain identical for sync and async paths."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            sync_limiter = InMemoryRateLimiter(
                requests_per_second=3, max_bucket_size=10
            )
            async_limiter = InMemoryRateLimiter(
                requests_per_second=3, max_bucket_size=10
            )
            sync_limiter.last = time.time()
            async_limiter.last = time.time()

            frozen_time.tick(2.0)  # 6 tokens each

            for _ in range(3):
                sync_limiter.acquire(blocking=False)
                await async_limiter.aacquire(blocking=False)

            assert sync_limiter.available_tokens == async_limiter.available_tokens


# ---------------------------------------------------------------------------
# InMemoryRateLimiter — thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for thread safety guarantees."""

    def test_concurrent_non_blocking_acquires(self) -> None:
        """Concurrent non-blocking acquires don't exceed bucket capacity."""
        limiter = InMemoryRateLimiter(requests_per_second=100, max_bucket_size=50)
        limiter.last = time.monotonic()
        time.sleep(0.5)  # ~50 tokens

        successes: list[int] = []
        lock = threading.Lock()

        def try_acquire(idx: int) -> None:
            if limiter.acquire(blocking=False):
                with lock:
                    successes.append(idx)

        threads = [threading.Thread(target=try_acquire, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Cannot exceed max_bucket_size
        assert len(successes) <= 50
        # No duplicates
        assert len(successes) == len(set(successes))

    async def test_concurrent_async_non_blocking_acquires(self) -> None:
        """Concurrent async non-blocking acquires don't exceed bucket capacity."""
        limiter = InMemoryRateLimiter(requests_per_second=100, max_bucket_size=50)
        limiter.last = time.monotonic()
        await asyncio.sleep(0.5)  # ~50 tokens

        successes: list[int] = []

        async def try_acquire(idx: int) -> None:
            if await limiter.aacquire(blocking=False):
                successes.append(idx)

        tasks = [asyncio.create_task(try_acquire(i)) for i in range(100)]
        await asyncio.gather(*tasks)

        assert len(successes) <= 50
        assert len(successes) == len(set(successes))

    def test_lock_prevents_race_condition(self) -> None:
        """The lock prevents two threads from consuming the same token."""
        limiter = InMemoryRateLimiter(requests_per_second=1, max_bucket_size=1)
        # Pre-load exactly 1 token by manipulating internal state directly
        limiter.last = time.monotonic() - 2.0  # 2 seconds ago → 2 tokens, capped to 1

        results: list[bool] = []
        lock = threading.Lock()

        def try_acquire() -> None:
            result = limiter.acquire(blocking=False)
            with lock:
                results.append(result)

        t1 = threading.Thread(target=try_acquire)
        t2 = threading.Thread(target=try_acquire)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one should succeed, one should fail
        assert sorted(results) == [False, True]


# ---------------------------------------------------------------------------
# InMemoryRateLimiter — edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_very_low_rate(self) -> None:
        """Very low rate (0.1 req/s) requires 10 seconds per request."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=0.1, max_bucket_size=10)
            limiter.last = time.time()

            # 5 seconds → 0.5 tokens → not enough
            frozen_time.tick(5.0)
            assert limiter.acquire(blocking=False) is False

            # 10 seconds total → 1.0 token → enough
            frozen_time.tick(5.0)
            assert limiter.acquire(blocking=False) is True

    def test_very_high_rate(self) -> None:
        """Very high rate (10000 req/s) still respects max_bucket_size."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=10000, max_bucket_size=3)
            limiter.last = time.time()
            frozen_time.tick(1.0)

            assert limiter.acquire(blocking=False) is True
            assert limiter.acquire(blocking=False) is True
            assert limiter.acquire(blocking=False) is True
            assert limiter.acquire(blocking=False) is False

    def test_large_bucket_size(self) -> None:
        """Large max_bucket_size allows bursts."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=2, max_bucket_size=500)
            limiter.last = time.time()
            frozen_time.tick(100)  # 200 tokens

            assert limiter.acquire(blocking=False) is True
            assert limiter.available_tokens == 199.0

            frozen_time.tick(10000)  # Would add 20000, capped at 500
            assert limiter.acquire(blocking=False) is True
            assert limiter.available_tokens == 499.0

    def test_acquire_after_long_idle(self) -> None:
        """Rate limiter works correctly after a very long idle period."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=1, max_bucket_size=5)
            limiter.last = time.time()

            # Idle for 1000 seconds
            frozen_time.tick(1000)
            assert limiter.acquire(blocking=False) is True
            # Capped at max_bucket_size, minus 1 consumed
            assert limiter.available_tokens == 4.0

    def test_immediate_second_acquire_fails_with_bucket_size_one(self) -> None:
        """With max_bucket_size=1, second immediate acquire fails."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(requests_per_second=1, max_bucket_size=1)
            limiter.last = time.time()
            frozen_time.tick(10)

            assert limiter.acquire(blocking=False) is True
            assert limiter.acquire(blocking=False) is False

    def test_check_every_n_seconds_does_not_affect_non_blocking(self) -> None:
        """check_every_n_seconds only affects blocking mode, not non-blocking."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(
                requests_per_second=10,
                check_every_n_seconds=999,  # Very large sleep interval
                max_bucket_size=10,
            )
            limiter.last = time.time()
            frozen_time.tick(1.0)

            # Non-blocking should return immediately regardless of check interval
            assert limiter.acquire(blocking=False) is True

    async def test_check_every_n_seconds_does_not_affect_async_non_blocking(
        self,
    ) -> None:
        """check_every_n_seconds only affects async blocking mode."""
        with freeze_time("2023-01-01 00:00:00") as frozen_time:
            limiter = InMemoryRateLimiter(
                requests_per_second=10,
                check_every_n_seconds=999,
                max_bucket_size=10,
            )
            limiter.last = time.time()
            frozen_time.tick(1.0)

            assert await limiter.aacquire(blocking=False) is True

    def test_consume_with_zero_elapsed_time(self) -> None:
        """_consume at exact same time as last returns False (no new tokens)."""
        limiter = InMemoryRateLimiter(requests_per_second=10, max_bucket_size=10)
        now = time.monotonic()
        limiter.last = now

        with patch("langchain_core.rate_limiters.time.monotonic", return_value=now):
            assert limiter._consume() is False
            assert limiter.available_tokens == 0.0

    def test_token_precision_with_small_increments(self) -> None:
        """Token calculation handles small floating-point increments correctly."""
        limiter = InMemoryRateLimiter(requests_per_second=10, max_bucket_size=100)
        base_time = 1000.0
        limiter.last = base_time

        # 0.1s * 10 = 1.0 → exactly at threshold
        with patch(
            "langchain_core.rate_limiters.time.monotonic",
            return_value=base_time + 0.1,
        ):
            assert limiter.acquire(blocking=False) is True
            assert limiter.available_tokens == pytest.approx(0.0, abs=0.01)

    async def test_async_blocking_acquire_after_depletion(self) -> None:
        """Async blocking acquire works after bucket is depleted."""
        limiter = InMemoryRateLimiter(
            requests_per_second=100, check_every_n_seconds=0.01, max_bucket_size=1
        )
        # First call depletes bucket (blocking waits for first token)
        result = await limiter.aacquire(blocking=True)
        assert result is True

        # Second call should also eventually succeed
        result = await limiter.aacquire(blocking=True)
        assert result is True

    def test_sync_blocking_acquire_after_depletion(self) -> None:
        """Sync blocking acquire works after bucket is depleted."""
        limiter = InMemoryRateLimiter(
            requests_per_second=100, check_every_n_seconds=0.01, max_bucket_size=1
        )
        result = limiter.acquire(blocking=True)
        assert result is True

        result = limiter.acquire(blocking=True)
        assert result is True
