"""Tests for merge utilities."""

import pytest

from langchain_core.utils._merge import merge_dicts, merge_lists, merge_obj


def test_merge_lists_basic() -> None:
    """Test basic list merging."""
    result = merge_lists([1, 2], [3, 4])
    assert result == [1, 2, 3, 4]


def test_merge_lists_with_none() -> None:
    """Test merging lists with None values."""
    result = merge_lists(None, [1, 2])
    assert result == [1, 2]

    result = merge_lists([1, 2], None)
    assert result == [1, 2]

    result = merge_lists(None, None)
    assert result is None


def test_merge_lists_with_index() -> None:
    """Test merging lists with indexed elements."""
    left = [{"index": 0, "value": "a"}]
    right = [{"index": 0, "value": "b"}]
    result = merge_lists(left, right)
    # Should merge elements with same index
    assert len(result) == 1
    assert result[0]["index"] == 0
    assert result[0]["value"] == "ab"


def test_merge_lists_with_different_indices() -> None:
    """Test merging lists with different indices."""
    left = [{"index": 0, "value": "a"}]
    right = [{"index": 1, "value": "b"}]
    result = merge_lists(left, right)
    # Should have both elements
    assert len(result) == 2


def test_merge_lists_multiple() -> None:
    """Test merging multiple lists."""
    result = merge_lists([1, 2], [3, 4], [5, 6])
    assert result == [1, 2, 3, 4, 5, 6]


def test_merge_obj_strings() -> None:
    """Test merging string objects."""
    result = merge_obj("hello", " world")
    assert result == "hello world"


def test_merge_obj_dicts() -> None:
    """Test merging dictionary objects."""
    left = {"a": 1, "b": 2}
    right = {"c": 3, "d": 4}
    result = merge_obj(left, right)
    assert result == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_merge_obj_lists() -> None:
    """Test merging list objects."""
    result = merge_obj([1, 2], [3, 4])
    assert result == [1, 2, 3, 4]


def test_merge_obj_with_none_left() -> None:
    """Test merging with None on left."""
    result = merge_obj(None, "value")
    assert result == "value"


def test_merge_obj_with_none_right() -> None:
    """Test merging with None on right."""
    result = merge_obj("value", None)
    assert result == "value"


def test_merge_obj_both_none() -> None:
    """Test merging two None values."""
    result = merge_obj(None, None)
    assert result is None


def test_merge_obj_equal_values() -> None:
    """Test merging equal values."""
    result = merge_obj(42, 42)
    assert result == 42


def test_merge_obj_type_mismatch() -> None:
    """Test merging objects of different types raises TypeError."""
    with pytest.raises(TypeError, match="left and right are of different types"):
        merge_obj("string", 42)


def test_merge_obj_incompatible_values() -> None:
    """Test merging incompatible values raises ValueError."""
    with pytest.raises(ValueError, match="Unable to merge"):
        merge_obj(42, 43)


def test_merge_dicts_overlapping_keys_str() -> None:
    """Test merging dicts with overlapping string keys."""
    left = {"key": "hello"}
    right = {"key": " world"}
    result = merge_dicts(left, right)
    assert result == {"key": "hello world"}


def test_merge_dicts_special_keys() -> None:
    """Test merging dicts with special keys like 'index'."""
    left = {"index": "lc_123"}
    right = {"index": "lc_456"}
    result = merge_dicts(left, right)
    # index starting with "lc_" should not be concatenated
    assert result["index"] == "lc_123"


def test_merge_dicts_id_key() -> None:
    """Test merging dicts with 'id' key."""
    left = {"id": "same"}
    right = {"id": "same"}
    result = merge_dicts(left, right)
    # Same id should not be concatenated
    assert result["id"] == "same"


def test_merge_dicts_nested_dicts() -> None:
    """Test merging nested dictionaries."""
    left = {"outer": {"inner": "a"}}
    right = {"outer": {"inner": "b"}}
    result = merge_dicts(left, right)
    assert result == {"outer": {"inner": "ab"}}


def test_merge_dicts_nested_lists() -> None:
    """Test merging dicts containing lists."""
    left = {"list": [1, 2]}
    right = {"list": [3, 4]}
    result = merge_dicts(left, right)
    assert result == {"list": [1, 2, 3, 4]}


def test_merge_dicts_integers() -> None:
    """Test merging dicts with integer values."""
    left = {"count": 5}
    right = {"count": 3}
    result = merge_dicts(left, right)
    assert result == {"count": 8}


def test_merge_dicts_none_handling() -> None:
    """Test merging dicts with None values."""
    left = {"key": None}
    right = {"key": "value"}
    result = merge_dicts(left, right)
    assert result == {"key": "value"}

    left = {"key": "value"}
    right = {"key": None}
    result = merge_dicts(left, right)
    assert result == {"key": "value"}


def test_merge_dicts_multiple() -> None:
    """Test merging multiple dictionaries."""
    result = merge_dicts(
        {"a": 1},
        {"b": 2},
        {"c": 3}
    )
    assert result == {"a": 1, "b": 2, "c": 3}


def test_merge_dicts_type_error() -> None:
    """Test merging dicts with type mismatch."""
    left = {"key": "string"}
    right = {"key": 42}
    with pytest.raises(TypeError, match="already exists.*different type"):
        merge_dicts(left, right)


def test_merge_dicts_unsupported_type() -> None:
    """Test merging dicts with unsupported type."""
    left = {"key": (1, 2)}
    right = {"key": (3, 4)}
    with pytest.raises(TypeError, match="unsupported type"):
        merge_dicts(left, right)


def test_merge_lists_with_lc_index() -> None:
    """Test merging lists with lc_ prefixed indices."""
    left = [{"index": "lc_0", "value": "a"}]
    right = [{"index": "lc_0", "value": "b"}]
    result = merge_lists(left, right)
    # Should merge elements with same lc_ prefixed index
    assert len(result) == 1
    assert result[0]["value"] == "ab"
