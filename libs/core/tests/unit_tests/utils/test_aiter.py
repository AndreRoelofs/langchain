from collections.abc import AsyncIterator

import pytest

from langchain_core.utils.aiter import NoLock, Tee, abatch_iterate, aclosing, atee


async def test_tee_async_basic() -> None:
    """Test basic async tee functionality."""
    async def async_range(n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i

    tee_obj = Tee(async_range(5), n=2)
    iter1, iter2 = tee_obj

    # Both iterators should produce same results
    result1 = [x async for x in iter1]
    result2 = [x async for x in iter2]

    assert result1 == [0, 1, 2, 3, 4]
    assert result2 == [0, 1, 2, 3, 4]


async def test_tee_async_len() -> None:
    """Test async Tee __len__ method."""
    async def async_range(n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i

    tee_obj = Tee(async_range(3), n=3)
    assert len(tee_obj) == 3


async def test_tee_async_getitem() -> None:
    """Test async Tee indexing."""
    async def async_range(n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i

    tee_obj = Tee(async_range(3), n=3)
    iter0 = tee_obj[0]
    result = [x async for x in iter0]
    assert result == [0, 1, 2]


async def test_tee_async_context_manager() -> None:
    """Test async Tee as context manager."""
    async def async_range(n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i

    async with Tee(async_range(3), n=2) as tee_obj:
        iter1, iter2 = tee_obj
        # Both should work within context
        assert await iter1.__anext__() == 0
        assert await iter2.__anext__() == 0


async def test_tee_async_different_speeds() -> None:
    """Test async tee handles iterators advancing at different speeds."""
    async def async_range(n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i

    tee_obj = Tee(async_range(5), n=2)
    iter1, iter2 = tee_obj

    # Advance iter1 ahead
    assert await iter1.__anext__() == 0
    assert await iter1.__anext__() == 1
    assert await iter1.__anext__() == 2

    # iter2 should still have all items
    result2 = [x async for x in iter2]
    assert result2 == [0, 1, 2, 3, 4]

    # iter1 should have remaining items
    result1 = [x async for x in iter1]
    assert result1 == [3, 4]


async def test_atee() -> None:
    """Test that atee is an alias for Tee."""
    assert atee is Tee


async def test_aclosing_basic() -> None:
    """Test basic aclosing functionality."""
    async def async_generator() -> AsyncIterator[int]:
        for i in range(3):
            yield i

    gen = async_generator()
    async with aclosing(gen) as agen:
        result = [x async for x in agen]

    assert result == [0, 1, 2]


async def test_aclosing_enters_correctly() -> None:
    """Test aclosing __aenter__ returns the wrapped object."""
    async def async_generator() -> AsyncIterator[int]:
        for i in range(3):
            yield i

    gen = async_generator()
    async with aclosing(gen) as agen:
        assert agen is gen


async def test_no_lock_async() -> None:
    """Test NoLock async context manager."""
    lock = NoLock()

    # Should enter and exit without error
    async with lock:
        pass

    # __aexit__ should return False (not suppressing exceptions)
    result = await lock.__aexit__(None, None, None)
    assert result is False


async def test_tee_async_empty_iterator() -> None:
    """Test async tee with empty iterator."""
    async def empty_gen() -> AsyncIterator[int]:
        return
        yield  # Make it a generator

    tee_obj = Tee(empty_gen(), n=2)
    iter1, iter2 = tee_obj

    result1 = [x async for x in iter1]
    result2 = [x async for x in iter2]

    assert result1 == []
    assert result2 == []


@pytest.mark.parametrize(
    ("input_size", "input_iterable", "expected_output"),
    [
        (2, [1, 2, 3, 4, 5], [[1, 2], [3, 4], [5]]),
        (3, [10, 20, 30, 40, 50], [[10, 20, 30], [40, 50]]),
        (1, [100, 200, 300], [[100], [200], [300]]),
        (4, [], []),
    ],
)
async def test_abatch_iterate(
    input_size: int, input_iterable: list[str], expected_output: list[list[str]]
) -> None:
    """Test batching function."""

    async def _to_async_iterable(iterable: list[str]) -> AsyncIterator[str]:
        for item in iterable:
            yield item

    iterator_ = abatch_iterate(input_size, _to_async_iterable(input_iterable))

    assert isinstance(iterator_, AsyncIterator)

    output = [el async for el in iterator_]
    assert output == expected_output
