import os

import pytest

from langchain_core.utils.env import env_var_is_set, get_from_dict_or_env, get_from_env


def test_env_var_is_set_true() -> None:
    """Test env_var_is_set returns True for set variables."""
    os.environ["TEST_VAR_SET"] = "value"
    try:
        assert env_var_is_set("TEST_VAR_SET") is True
    finally:
        del os.environ["TEST_VAR_SET"]


def test_env_var_is_set_false_empty() -> None:
    """Test env_var_is_set returns False for empty string."""
    os.environ["TEST_VAR_EMPTY"] = ""
    try:
        assert env_var_is_set("TEST_VAR_EMPTY") is False
    finally:
        del os.environ["TEST_VAR_EMPTY"]


def test_env_var_is_set_false_zero() -> None:
    """Test env_var_is_set returns False for '0'."""
    os.environ["TEST_VAR_ZERO"] = "0"
    try:
        assert env_var_is_set("TEST_VAR_ZERO") is False
    finally:
        del os.environ["TEST_VAR_ZERO"]


def test_env_var_is_set_false_false_string() -> None:
    """Test env_var_is_set returns False for 'false' and 'False'."""
    os.environ["TEST_VAR_FALSE"] = "false"
    try:
        assert env_var_is_set("TEST_VAR_FALSE") is False
    finally:
        del os.environ["TEST_VAR_FALSE"]

    os.environ["TEST_VAR_FALSE_CAPS"] = "False"
    try:
        assert env_var_is_set("TEST_VAR_FALSE_CAPS") is False
    finally:
        del os.environ["TEST_VAR_FALSE_CAPS"]


def test_env_var_is_set_false_not_set() -> None:
    """Test env_var_is_set returns False for unset variable."""
    # Make sure it's not set
    if "TEST_VAR_NOT_SET" in os.environ:
        del os.environ["TEST_VAR_NOT_SET"]
    assert env_var_is_set("TEST_VAR_NOT_SET") is False


def test_get_from_env_with_env_variable() -> None:
    """Test get_from_env retrieves value from environment."""
    os.environ["TEST_GET_ENV"] = "test_value"
    try:
        result = get_from_env("test_key", "TEST_GET_ENV")
        assert result == "test_value"
    finally:
        del os.environ["TEST_GET_ENV"]


def test_get_from_env_with_default() -> None:
    """Test get_from_env returns default when env var not set."""
    # Make sure it's not set
    if "TEST_GET_ENV_NOT_SET" in os.environ:
        del os.environ["TEST_GET_ENV_NOT_SET"]

    result = get_from_env("test_key", "TEST_GET_ENV_NOT_SET", default="default_value")
    assert result == "default_value"


def test_get_from_env_raises_without_default() -> None:
    """Test get_from_env raises ValueError when no default and var not set."""
    # Make sure it's not set
    if "TEST_GET_ENV_MISSING" in os.environ:
        del os.environ["TEST_GET_ENV_MISSING"]

    with pytest.raises(ValueError, match="Did not find test_key"):
        get_from_env("test_key", "TEST_GET_ENV_MISSING")


def test_get_from_dict_or_env() -> None:
    assert (
        get_from_dict_or_env(
            {
                "a": "foo",
            },
            ["a"],
            "__SOME_KEY_IN_ENV",
        )
        == "foo"
    )

    assert (
        get_from_dict_or_env(
            {
                "a": "foo",
            },
            ["b", "a"],
            "__SOME_KEY_IN_ENV",
        )
        == "foo"
    )

    assert (
        get_from_dict_or_env(
            {
                "a": "foo",
            },
            "a",
            "__SOME_KEY_IN_ENV",
        )
        == "foo"
    )

    assert (
        get_from_dict_or_env(
            {
                "a": "foo",
            },
            "not exists",
            "__SOME_KEY_IN_ENV",
            default="default",
        )
        == "default"
    )

    # Not the most obvious behavior, but
    # this is how it works right now
    with pytest.raises(
        ValueError,
        match="Did not find not exists, "
        "please add an environment variable `__SOME_KEY_IN_ENV` which contains it, "
        "or pass `not exists` as a named parameter",
    ):
        assert (
            get_from_dict_or_env(
                {
                    "a": "foo",
                },
                "not exists",
                "__SOME_KEY_IN_ENV",
            )
            is None
        )
