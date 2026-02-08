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


# ---------------------------------------------------------------------------
# py_anext (deprecated)
# ---------------------------------------------------------------------------
import warnings

from langchain_core.utils.aiter import py_anext


async def test_py_anext_returns_next_item() -> None:
    """py_anext returns the next item from async iterator."""

    async def async_range(n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i

    it = async_range(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = await py_anext(it)
    assert result == 0


async def test_py_anext_with_default_on_exhausted() -> None:
    """py_anext returns default when iterator is exhausted."""

    async def empty_gen() -> AsyncIterator[int]:
        return
        yield

    it = empty_gen()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = await py_anext(it, "default_val")
    assert result == "default_val"


async def test_py_anext_raises_stop_async_iteration() -> None:
    """py_anext raises StopAsyncIteration when exhausted and no default."""

    async def empty_gen() -> AsyncIterator[int]:
        return
        yield

    it = empty_gen()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(StopAsyncIteration):
            await py_anext(it)


async def test_py_anext_raises_type_error_for_non_iterator() -> None:
    """py_anext raises TypeError for non-async-iterator."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(TypeError, match="is not an async iterator"):
            await py_anext(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# abatch_iterate edge cases
# ---------------------------------------------------------------------------
async def test_abatch_iterate_exact_multiple() -> None:
    """abatch_iterate with size that evenly divides the iterable."""

    async def async_range(n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i

    result = [batch async for batch in abatch_iterate(3, async_range(6))]
    assert result == [[0, 1, 2], [3, 4, 5]]


async def test_abatch_iterate_size_larger_than_iterable() -> None:
    """abatch_iterate with size larger than iterable returns single batch."""

    async def async_range(n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i

    result = [batch async for batch in abatch_iterate(100, async_range(3))]
    assert result == [[0, 1, 2]]


async def test_abatch_iterate_size_one() -> None:
    """abatch_iterate with size=1 yields individual items as lists."""

    async def async_gen() -> AsyncIterator[str]:
        for s in ["a", "b", "c"]:
            yield s

    result = [batch async for batch in abatch_iterate(1, async_gen())]
    assert result == [["a"], ["b"], ["c"]]


# ---------------------------------------------------------------------------
# Tee with slice access
# ---------------------------------------------------------------------------
async def test_tee_async_slice() -> None:
    """Test async Tee slicing."""

    async def async_range(n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i

    tee_obj = Tee(async_range(3), n=3)
    sliced = tee_obj[0:2]
    assert len(sliced) == 2


# ---------------------------------------------------------------------------
# aclosing with non-generator async iterator
# ---------------------------------------------------------------------------
async def test_aclosing_with_async_iterator_no_aclose() -> None:
    """aclosing works with async iterators that don't have aclose."""

    class SimpleAsyncIter:
        def __init__(self) -> None:
            self.items = [1, 2, 3]
            self.idx = 0

        def __aiter__(self) -> "SimpleAsyncIter":
            return self

        async def __anext__(self) -> int:
            if self.idx >= len(self.items):
                raise StopAsyncIteration
            val = self.items[self.idx]
            self.idx += 1
            return val

    it = SimpleAsyncIter()
    async with aclosing(it) as ait:
        result = [x async for x in ait]
    assert result == [1, 2, 3]
