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


# ---------------------------------------------------------------------------
# get_from_dict_or_env edge cases
# ---------------------------------------------------------------------------
def test_get_from_dict_or_env_with_list_keys_first_match() -> None:
    """get_from_dict_or_env with list of keys returns first matching key."""
    data = {"first": "val1", "second": "val2"}
    result = get_from_dict_or_env(data, ["first", "second"], "UNUSED_ENV")
    assert result == "val1"


def test_get_from_dict_or_env_with_list_keys_second_match() -> None:
    """get_from_dict_or_env with list of keys falls back to second key."""
    data = {"second": "val2"}
    result = get_from_dict_or_env(data, ["first", "second"], "UNUSED_ENV")
    assert result == "val2"


def test_get_from_dict_or_env_falls_through_to_env() -> None:
    """get_from_dict_or_env falls through to env var when key not in dict."""
    os.environ["FALLBACK_ENV_KEY"] = "from_env"
    try:
        result = get_from_dict_or_env({}, "missing_key", "FALLBACK_ENV_KEY")
        assert result == "from_env"
    finally:
        del os.environ["FALLBACK_ENV_KEY"]


def test_get_from_dict_or_env_list_keys_falls_through_to_env() -> None:
    """get_from_dict_or_env with list keys falls through to env when no match."""
    os.environ["FALLBACK_ENV_KEY2"] = "env_value"
    try:
        result = get_from_dict_or_env({}, ["missing1", "missing2"], "FALLBACK_ENV_KEY2")
        assert result == "env_value"
    finally:
        del os.environ["FALLBACK_ENV_KEY2"]


def test_get_from_dict_or_env_with_falsy_dict_value() -> None:
    """get_from_dict_or_env skips falsy dict values and falls to env/default."""
    # Empty string is falsy in Python, so get_from_dict_or_env skips it
    data = {"key": ""}
    result = get_from_dict_or_env(data, "key", "UNUSED_ENV", default="default_val")
    assert result == "default_val"


def test_get_from_env_empty_env_value() -> None:
    """get_from_env returns default when env var is empty string."""
    os.environ["EMPTY_VAR"] = ""
    try:
        # os.getenv returns "", which is falsy
        result = get_from_env("key", "EMPTY_VAR", default="fallback")
        assert result == "fallback"
    finally:
        del os.environ["EMPTY_VAR"]


def test_env_var_is_set_with_truthy_values() -> None:
    """env_var_is_set returns True for various truthy values."""
    for val in ["1", "true", "True", "yes", "anything"]:
        os.environ["TEST_TRUTHY"] = val
        try:
            assert env_var_is_set("TEST_TRUTHY") is True, f"Failed for value: {val}"
        finally:
            del os.environ["TEST_TRUTHY"]
