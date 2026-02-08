import pytest

from langchain_core.utils.iter import NoLock, Tee, batch_iterate, safetee, tee_peer


def test_tee_basic() -> None:
    """Test basic tee functionality."""
    data = [1, 2, 3, 4, 5]
    tee_obj = Tee(iter(data), n=2)

    iter1, iter2 = tee_obj

    # Both iterators should produce same results
    assert list(iter1) == [1, 2, 3, 4, 5]
    assert list(iter2) == [1, 2, 3, 4, 5]


def test_tee_len() -> None:
    """Test Tee __len__ method."""
    data = [1, 2, 3]
    tee_obj = Tee(iter(data), n=3)
    assert len(tee_obj) == 3


def test_tee_getitem() -> None:
    """Test Tee indexing."""
    data = [1, 2, 3]
    tee_obj = Tee(iter(data), n=3)

    # Access by index
    iter0 = tee_obj[0]
    assert list(iter0) == [1, 2, 3]


def test_tee_slice() -> None:
    """Test Tee slicing."""
    data = [1, 2, 3]
    tee_obj = Tee(iter(data), n=3)

    # Access by slice
    iters = tee_obj[0:2]
    assert len(iters) == 2


def test_tee_iter() -> None:
    """Test Tee iteration."""
    data = [1, 2, 3]
    tee_obj = Tee(iter(data), n=2)

    # Can iterate over the tee object itself
    iters = list(tee_obj)
    assert len(iters) == 2


def test_tee_context_manager() -> None:
    """Test Tee as context manager."""
    data = [1, 2, 3]
    with Tee(iter(data), n=2) as tee_obj:
        iter1, iter2 = tee_obj
        # Both should work within context
        assert next(iter1) == 1
        assert next(iter2) == 1


def test_tee_close() -> None:
    """Test Tee close method."""
    data = [1, 2, 3]
    tee_obj = Tee(iter(data), n=2)
    iter1, iter2 = tee_obj

    # Get one item from each
    next(iter1)
    next(iter2)

    # Close should work without error
    tee_obj.close()


def test_tee_different_speeds() -> None:
    """Test that tee handles iterators advancing at different speeds."""
    data = [1, 2, 3, 4, 5]
    tee_obj = Tee(iter(data), n=2)
    iter1, iter2 = tee_obj

    # Advance iter1 ahead
    assert next(iter1) == 1
    assert next(iter1) == 2
    assert next(iter1) == 3

    # iter2 should still have all items
    assert list(iter2) == [1, 2, 3, 4, 5]

    # iter1 should have remaining items
    assert list(iter1) == [4, 5]


def test_safetee() -> None:
    """Test that safetee is an alias for Tee."""
    assert safetee is Tee


def test_no_lock_enter_exit() -> None:
    """Test NoLock context manager."""
    lock = NoLock()

    # Should enter and exit without error
    with lock:
        pass

    # __exit__ should return False (not suppressing exceptions)
    result = lock.__exit__(None, None, None)
    assert result is False


def test_tee_empty_iterator() -> None:
    """Test tee with empty iterator."""
    data: list[int] = []
    tee_obj = Tee(iter(data), n=2)
    iter1, iter2 = tee_obj

    assert list(iter1) == []
    assert list(iter2) == []


def test_tee_single_iterator() -> None:
    """Test tee with n=1."""
    data = [1, 2, 3]
    tee_obj = Tee(iter(data), n=1)

    assert len(tee_obj) == 1
    iter1 = tee_obj[0]
    assert list(iter1) == [1, 2, 3]


@pytest.mark.parametrize(
    ("input_size", "input_iterable", "expected_output"),
    [
        (2, [1, 2, 3, 4, 5], [[1, 2], [3, 4], [5]]),
        (3, [10, 20, 30, 40, 50], [[10, 20, 30], [40, 50]]),
        (1, [100, 200, 300], [[100], [200], [300]]),
        (4, [], []),
    ],
)
def test_batch_iterate(
    input_size: int, input_iterable: list[str], expected_output: list[list[str]]
) -> None:
    """Test batching function."""
    assert list(batch_iterate(input_size, input_iterable)) == expected_output


# ---------------------------------------------------------------------------
# batch_iterate with None size (returns single batch)
# ---------------------------------------------------------------------------
def test_batch_iterate_none_size() -> None:
    """batch_iterate with size=None returns all items in a single batch."""
    data = [1, 2, 3, 4, 5]
    result = list(batch_iterate(None, data))
    assert result == [[1, 2, 3, 4, 5]]


def test_batch_iterate_none_size_empty() -> None:
    """batch_iterate with size=None and empty iterable returns empty."""
    result = list(batch_iterate(None, []))
    assert result == []


# ---------------------------------------------------------------------------
# batch_iterate edge cases
# ---------------------------------------------------------------------------
def test_batch_iterate_exact_multiple() -> None:
    """batch_iterate with size that evenly divides the iterable."""
    result = list(batch_iterate(3, [1, 2, 3, 4, 5, 6]))
    assert result == [[1, 2, 3], [4, 5, 6]]


def test_batch_iterate_size_larger_than_iterable() -> None:
    """batch_iterate with size larger than iterable returns single batch."""
    result = list(batch_iterate(100, [1, 2, 3]))
    assert result == [[1, 2, 3]]


def test_batch_iterate_with_generator() -> None:
    """batch_iterate works with generators, not just lists."""

    def gen() -> "Iterator[int]":
        yield from range(5)

    from collections.abc import Iterator

    result = list(batch_iterate(2, gen()))
    assert result == [[0, 1], [2, 3], [4]]


# ---------------------------------------------------------------------------
# tee_peer cleanup behavior
# ---------------------------------------------------------------------------
def test_tee_peer_closes_iterator_when_last_peer() -> None:
    """tee_peer closes the iterator when the last peer is done."""
    closed = False

    class CloseableIterator:
        def __init__(self) -> None:
            self.items = iter([1, 2, 3])

        def __iter__(self) -> "CloseableIterator":
            return self

        def __next__(self) -> int:
            return next(self.items)

        def close(self) -> None:
            nonlocal closed
            closed = True

    it = CloseableIterator()
    tee_obj = Tee(it, n=1)  # type: ignore[arg-type]
    result = list(tee_obj[0])
    assert result == [1, 2, 3]
    assert closed is True


def test_tee_large_n() -> None:
    """Tee works with many child iterators."""
    data = [10, 20, 30]
    tee_obj = Tee(iter(data), n=5)
    assert len(tee_obj) == 5

    for i in range(5):
        assert list(tee_obj[i]) == [10, 20, 30]
