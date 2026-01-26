"""Test BaseRateLimiter abstract base class."""

import pytest

from langchain_core.rate_limiters import BaseRateLimiter


def test_base_rate_limiter_is_abstract() -> None:
    """Test that BaseRateLimiter cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseRateLimiter()  # type: ignore[abstract]


def test_base_rate_limiter_requires_acquire() -> None:
    """Test that subclasses must implement acquire method."""

    class IncompleteRateLimiter(BaseRateLimiter):
        async def aacquire(self, *, blocking: bool = True) -> bool:
            return True

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteRateLimiter()  # type: ignore[abstract]


def test_base_rate_limiter_requires_aacquire() -> None:
    """Test that subclasses must implement aacquire method."""

    class IncompleteRateLimiter(BaseRateLimiter):
        def acquire(self, *, blocking: bool = True) -> bool:
            return True

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteRateLimiter()  # type: ignore[abstract]


def test_base_rate_limiter_can_be_subclassed() -> None:
    """Test that BaseRateLimiter can be properly subclassed."""

    class CompleteRateLimiter(BaseRateLimiter):
        def acquire(self, *, blocking: bool = True) -> bool:
            return True

        async def aacquire(self, *, blocking: bool = True) -> bool:
            return True

    # Should not raise an error
    limiter = CompleteRateLimiter()
    assert limiter.acquire() is True
    assert limiter.acquire(blocking=False) is True


async def test_base_rate_limiter_async_method() -> None:
    """Test that async methods work properly in subclasses."""

    class CompleteRateLimiter(BaseRateLimiter):
        def acquire(self, *, blocking: bool = True) -> bool:
            return True

        async def aacquire(self, *, blocking: bool = True) -> bool:
            return blocking

    limiter = CompleteRateLimiter()
    assert await limiter.aacquire() is True
    assert await limiter.aacquire(blocking=False) is False
