import operator

import pytest

from langchain_core.utils.usage import _dict_int_op


def test_dict_int_op_add() -> None:
    left = {"a": 1, "b": 2}
    right = {"b": 3, "c": 4}
    result = _dict_int_op(left, right, operator.add)
    assert result == {"a": 1, "b": 5, "c": 4}


def test_dict_int_op_subtract() -> None:
    left = {"a": 5, "b": 10}
    right = {"a": 2, "b": 3, "c": 1}
    result = _dict_int_op(left, right, lambda x, y: max(x - y, 0))
    assert result == {"a": 3, "b": 7, "c": 0}


def test_dict_int_op_nested() -> None:
    left = {"a": 1, "b": {"c": 2, "d": 3}}
    right = {"a": 2, "b": {"c": 1, "e": 4}}
    result = _dict_int_op(left, right, operator.add)
    assert result == {"a": 3, "b": {"c": 3, "d": 3, "e": 4}}


def test_dict_int_op_max_depth_exceeded() -> None:
    left = {"a": {"b": {"c": 1}}}
    right = {"a": {"b": {"c": 2}}}
    with pytest.raises(
        ValueError, match="max_depth=2 exceeded, unable to combine dicts"
    ):
        _dict_int_op(left, right, operator.add, max_depth=2)


def test_dict_int_op_invalid_types() -> None:
    left = {"a": 1, "b": "string"}
    right = {"a": 2, "b": 3}
    with pytest.raises(
        ValueError,
        match="Only dict and int values are supported",
    ):
        _dict_int_op(left, right, operator.add)


def test_dict_int_op_empty_dicts() -> None:
    """Combining two empty dicts returns an empty dict."""
    result = _dict_int_op({}, {}, operator.add)
    assert result == {}


def test_dict_int_op_one_empty_dict() -> None:
    """Combining with one empty dict uses default for missing keys."""
    result = _dict_int_op({"a": 5}, {}, operator.add)
    assert result == {"a": 5}

    result = _dict_int_op({}, {"b": 3}, operator.add)
    assert result == {"b": 3}


def test_dict_int_op_custom_default() -> None:
    """Custom default value is used for missing keys."""
    left = {"a": 10}
    right = {"a": 3, "b": 7}
    result = _dict_int_op(left, right, operator.add, default=5)
    assert result == {"a": 13, "b": 12}  # b: 5 (default) + 7


def test_dict_int_op_multiply() -> None:
    """Works with arbitrary integer operations like multiplication."""
    left = {"a": 3, "b": 4}
    right = {"a": 2, "b": 5}
    result = _dict_int_op(left, right, operator.mul)
    assert result == {"a": 6, "b": 20}


def test_dict_int_op_single_key() -> None:
    """Works correctly with a single key in both dicts."""
    result = _dict_int_op({"x": 10}, {"x": 20}, operator.add)
    assert result == {"x": 30}


def test_dict_int_op_deeply_nested() -> None:
    """Handles deeply nested dicts within max_depth."""
    left = {"a": {"b": {"c": {"d": 1}}}}
    right = {"a": {"b": {"c": {"d": 2}}}}
    result = _dict_int_op(left, right, operator.add, max_depth=10)
    assert result == {"a": {"b": {"c": {"d": 3}}}}


def test_dict_int_op_mixed_nested_and_flat() -> None:
    """Handles mix of nested and flat keys."""
    left = {"a": 1, "b": {"c": 2}}
    right = {"a": 3, "b": {"c": 4, "d": 5}}
    result = _dict_int_op(left, right, operator.add)
    assert result == {"a": 4, "b": {"c": 6, "d": 5}}


def test_dict_int_op_nested_empty_dict() -> None:
    """Handles nested empty dicts gracefully."""
    left = {"a": {}}
    right = {"a": {"b": 1}}
    result = _dict_int_op(left, right, operator.add)
    assert result == {"a": {"b": 1}}


def test_dict_int_op_max_depth_exact_boundary() -> None:
    """max_depth=1 allows one level of nesting but not two."""
    # depth=0 for top level, depth=1 for first nesting level
    left = {"a": {"b": 1}}
    right = {"a": {"b": 2}}
    # max_depth=2 means depth must be < 2, so depth=0 and depth=1 are OK
    result = _dict_int_op(left, right, operator.add, max_depth=2)
    assert result == {"a": {"b": 3}}

    # But double nesting should fail at max_depth=2
    left2 = {"a": {"b": {"c": 1}}}
    right2 = {"a": {"b": {"c": 2}}}
    with pytest.raises(ValueError, match="max_depth=2 exceeded"):
        _dict_int_op(left2, right2, operator.add, max_depth=2)
