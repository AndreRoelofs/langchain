import os
import re
import sys
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field, SecretStr
from pydantic.v1 import BaseModel as PydanticV1BaseModel
from pydantic.v1 import Field as PydanticV1Field

from langchain_core import utils
from langchain_core.outputs import GenerationChunk
from langchain_core.utils import (
    check_package_version,
    from_env,
    get_pydantic_field_names,
    guard_import,
)
from langchain_core.utils._merge import merge_dicts
from langchain_core.utils.utils import secret_from_env

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("package", "check_kwargs", "actual_version", "expected"),
    [
        ("stub", {"gt_version": "0.1"}, "0.1.2", None),
        ("stub", {"gt_version": "0.1.2"}, "0.1.12", None),
        ("stub", {"gt_version": "0.1.2"}, "0.1.2", (ValueError, "> 0.1.2")),
        ("stub", {"gte_version": "0.1"}, "0.1.2", None),
        ("stub", {"gte_version": "0.1.2"}, "0.1.2", None),
    ],
)
def test_check_package_version(
    package: str,
    check_kwargs: dict[str, str | None],
    actual_version: str,
    expected: tuple[type[Exception], str] | None,
) -> None:
    with patch("langchain_core.utils.utils.version", return_value=actual_version):
        if expected is None:
            check_package_version(package, **check_kwargs)
        else:
            with pytest.raises(expected[0], match=expected[1]):
                check_package_version(package, **check_kwargs)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        # Merge `None` and `1`.
        ({"a": None}, {"a": 1}, {"a": 1}),
        # Merge `1` and `None`.
        ({"a": 1}, {"a": None}, {"a": 1}),
        # Merge `None` and a value.
        ({"a": None}, {"a": 0}, {"a": 0}),
        ({"a": None}, {"a": "txt"}, {"a": "txt"}),
        # Merge equal values.
        ({"a": 1}, {"a": 1}, {"a": 1}),
        ({"a": 1.5}, {"a": 1.5}, {"a": 1.5}),
        ({"a": True}, {"a": True}, {"a": True}),
        ({"a": False}, {"a": False}, {"a": False}),
        ({"a": "txt"}, {"a": "txt"}, {"a": "txttxt"}),
        ({"a": [1, 2]}, {"a": [1, 2]}, {"a": [1, 2, 1, 2]}),
        ({"a": {"b": "txt"}}, {"a": {"b": "txt"}}, {"a": {"b": "txttxt"}}),
        # Merge strings.
        ({"a": "one"}, {"a": "two"}, {"a": "onetwo"}),
        # Merge dicts.
        ({"a": {"b": 1}}, {"a": {"c": 2}}, {"a": {"b": 1, "c": 2}}),
        (
            {"function_call": {"arguments": None}},
            {"function_call": {"arguments": "{\n"}},
            {"function_call": {"arguments": "{\n"}},
        ),
        # Merge lists.
        ({"a": [1, 2]}, {"a": [3]}, {"a": [1, 2, 3]}),
        ({"a": 1, "b": 2}, {"a": 1}, {"a": 1, "b": 2}),
        ({"a": 1, "b": 2}, {"c": None}, {"a": 1, "b": 2, "c": None}),
        #
        # Invalid inputs.
        #
        (
            {"a": 1},
            {"a": "1"},
            pytest.raises(
                TypeError,
                match=re.escape(
                    'additional_kwargs["a"] already exists in this message, '
                    "but with a different type."
                ),
            ),
        ),
        (
            {"a": (1, 2)},
            {"a": (3,)},
            pytest.raises(
                TypeError,
                match=(
                    "Additional kwargs key a already exists in left dict and value "
                    r"has unsupported type .+tuple.+."
                ),
            ),
        ),
        # 'index' keyword has special handling
        (
            {"a": [{"index": 0, "b": "{"}]},
            {"a": [{"index": 0, "b": "f"}]},
            {"a": [{"index": 0, "b": "{f"}]},
        ),
        (
            {"a": [{"idx": 0, "b": "{"}]},
            {"a": [{"idx": 0, "b": "f"}]},
            {"a": [{"idx": 0, "b": "{"}, {"idx": 0, "b": "f"}]},
        ),
    ],
)
def test_merge_dicts(
    left: dict, right: dict, expected: dict | AbstractContextManager
) -> None:
    err = expected if isinstance(expected, AbstractContextManager) else nullcontext()

    left_copy = deepcopy(left)
    right_copy = deepcopy(right)
    with err:
        actual = merge_dicts(left, right)
        assert actual == expected
        # no mutation
        assert left == left_copy
        assert right == right_copy


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        # 'type' special key handling
        ({"type": "foo"}, {"type": "foo"}, {"type": "foo"}),
        (
            {"type": "foo"},
            {"type": "bar"},
            pytest.raises(ValueError, match="Unable to merge"),
        ),
    ],
)
@pytest.mark.xfail(reason="Refactors to make in 0.3")
def test_merge_dicts_0_3(
    left: dict, right: dict, expected: dict | AbstractContextManager
) -> None:
    err = expected if isinstance(expected, AbstractContextManager) else nullcontext()

    left_copy = deepcopy(left)
    right_copy = deepcopy(right)
    with err:
        actual = merge_dicts(left, right)
        assert actual == expected
        # no mutation
        assert left == left_copy
        assert right == right_copy


@pytest.mark.parametrize(
    ("module_name", "pip_name", "package", "expected"),
    [
        ("langchain_core.utils", None, None, utils),
        ("langchain_core.utils", "langchain-core", None, utils),
        ("langchain_core.utils", None, "langchain-core", utils),
        ("langchain_core.utils", "langchain-core", "langchain-core", utils),
    ],
)
def test_guard_import(
    module_name: str, pip_name: str | None, package: str | None, expected: Any
) -> None:
    if package is None and pip_name is None:
        ret = guard_import(module_name)
    elif package is None and pip_name is not None:
        ret = guard_import(module_name, pip_name=pip_name)
    elif package is not None and pip_name is None:
        ret = guard_import(module_name, package=package)
    elif package is not None and pip_name is not None:
        ret = guard_import(module_name, pip_name=pip_name, package=package)
    else:
        msg = "Invalid test case"
        raise ValueError(msg)
    assert ret == expected


@pytest.mark.parametrize(
    ("module_name", "pip_name", "package", "expected_pip_name"),
    [
        ("langchain_core.utilsW", None, None, "langchain-core"),
        ("langchain_core.utilsW", "langchain-core-2", None, "langchain-core-2"),
        ("langchain_core.utilsW", None, "langchain-coreWX", "langchain-core"),
        (
            "langchain_core.utilsW",
            "langchain-core-2",
            "langchain-coreWX",
            "langchain-core-2",
        ),
        ("langchain_coreW", None, None, "langchain-coreW"),  # ModuleNotFoundError
    ],
)
def test_guard_import_failure(
    module_name: str,
    pip_name: str | None,
    package: str | None,
    expected_pip_name: str,
) -> None:
    with pytest.raises(
        ImportError,
        match=f"Could not import {module_name} python package. "
        f"Please install it with `pip install {expected_pip_name}`.",
    ):
        guard_import(module_name, pip_name=pip_name, package=package)


@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="pydantic.v1 namespace not supported with Python 3.14+",
)
def test_get_pydantic_field_names_v1_in_2() -> None:
    class PydanticV1Model(PydanticV1BaseModel):
        field1: str
        field2: int
        alias_field: int = PydanticV1Field(alias="aliased_field")

    result = get_pydantic_field_names(PydanticV1Model)
    expected = {"field1", "field2", "aliased_field", "alias_field"}
    assert result == expected


def test_get_pydantic_field_names_v2_in_2() -> None:
    class PydanticModel(BaseModel):
        field1: str
        field2: int
        alias_field: int = Field(alias="aliased_field")

    result = get_pydantic_field_names(PydanticModel)
    expected = {"field1", "field2", "aliased_field", "alias_field"}
    assert result == expected


def test_from_env_with_env_variable() -> None:
    key = "TEST_KEY"
    value = "test_value"
    with patch.dict(os.environ, {key: value}):
        get_value = from_env(key)
        assert get_value() == value


def test_from_env_with_default_value() -> None:
    key = "TEST_KEY"
    default_value = "default_value"
    with patch.dict(os.environ, {}, clear=True):
        get_value = from_env(key, default=default_value)
        assert get_value() == default_value


def test_from_env_with_error_message() -> None:
    key = "TEST_KEY"
    error_message = "Custom error message"
    with patch.dict(os.environ, {}, clear=True):
        get_value = from_env(key, error_message=error_message)
        with pytest.raises(ValueError, match=error_message):
            get_value()


def test_from_env_with_default_error_message() -> None:
    key = "TEST_KEY"
    with patch.dict(os.environ, {}, clear=True):
        get_value = from_env(key)
        with pytest.raises(ValueError, match=f"Did not find {key}"):
            get_value()


def test_secret_from_env_with_env_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set the environment variable
    monkeypatch.setenv("TEST_KEY", "secret_value")

    # Get the function
    get_secret: Callable[[], SecretStr | None] = secret_from_env("TEST_KEY")

    # Assert that it returns the correct value
    assert get_secret() == SecretStr("secret_value")


def test_secret_from_env_with_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unset the environment variable
    monkeypatch.delenv("TEST_KEY", raising=False)

    # Get the function with a default value
    get_secret: Callable[[], SecretStr] = secret_from_env(
        "TEST_KEY", default="default_value"
    )

    # Assert that it returns the default value
    assert get_secret() == SecretStr("default_value")


def test_secret_from_env_with_none_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unset the environment variable
    monkeypatch.delenv("TEST_KEY", raising=False)

    # Get the function with a default value of None
    get_secret: Callable[[], SecretStr | None] = secret_from_env(
        "TEST_KEY", default=None
    )

    # Assert that it returns None
    assert get_secret() is None


def test_secret_from_env_without_default_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unset the environment variable
    monkeypatch.delenv("TEST_KEY", raising=False)

    # Get the function without a default value
    get_secret: Callable[[], SecretStr] = secret_from_env("TEST_KEY")

    # Assert that it raises a ValueError with the correct message
    with pytest.raises(ValueError, match="Did not find TEST_KEY"):
        get_secret()


def test_secret_from_env_with_custom_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unset the environment variable
    monkeypatch.delenv("TEST_KEY", raising=False)

    # Get the function without a default value but with a custom error message
    get_secret: Callable[[], SecretStr] = secret_from_env(
        "TEST_KEY", error_message="Custom error message"
    )

    # Assert that it raises a ValueError with the custom message
    with pytest.raises(ValueError, match="Custom error message"):
        get_secret()


def test_using_secret_from_env_as_default_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Foo(BaseModel):
        secret: SecretStr = Field(default_factory=secret_from_env("TEST_KEY"))

    # Pass the secret as a parameter
    foo = Foo(secret="super_secret")
    assert foo.secret.get_secret_value() == "super_secret"

    # Set the environment variable
    monkeypatch.setenv("TEST_KEY", "secret_value")
    assert Foo().secret.get_secret_value() == "secret_value"

    class Bar(BaseModel):
        secret: SecretStr | None = Field(
            default_factory=secret_from_env("TEST_KEY_2", default=None)
        )

    assert Bar().secret is None

    class Buzz(BaseModel):
        secret: SecretStr | None = Field(
            default_factory=secret_from_env("TEST_KEY_2", default="hello")
        )

    # We know it will be SecretStr rather than SecretStr | None
    assert Buzz().secret.get_secret_value() == "hello"  # type: ignore[union-attr]

    class OhMy(BaseModel):
        secret: SecretStr | None = Field(
            default_factory=secret_from_env("FOOFOOFOOBAR")
        )

    with pytest.raises(ValueError, match="Did not find FOOFOOFOOBAR"):
        OhMy()


def test_generation_chunk_addition_type_error() -> None:
    chunk1 = GenerationChunk(text="", generation_info={"len": 0})
    chunk2 = GenerationChunk(text="Non-empty text", generation_info={"len": 14})
    result = chunk1 + chunk2
    assert result == GenerationChunk(text="Non-empty text", generation_info={"len": 14})


# ---------------------------------------------------------------------------
# xor_args
# ---------------------------------------------------------------------------
from langchain_core.utils.utils import xor_args


def test_xor_args_exactly_one_defined() -> None:
    """xor_args passes when exactly one arg per group is provided."""

    @xor_args(("a", "b"))
    def fn(*, a: str | None = None, b: str | None = None) -> str:
        return a or b or ""

    assert fn(a="hello") == "hello"
    assert fn(b="world") == "world"


def test_xor_args_none_defined_raises() -> None:
    """xor_args raises when no arg in the group is provided."""

    @xor_args(("a", "b"))
    def fn(*, a: str | None = None, b: str | None = None) -> str:
        return a or b or ""

    with pytest.raises(ValueError, match="Exactly one argument"):
        fn()


def test_xor_args_both_defined_raises() -> None:
    """xor_args raises when more than one arg in the group is provided."""

    @xor_args(("a", "b"))
    def fn(*, a: str | None = None, b: str | None = None) -> str:
        return a or b or ""

    with pytest.raises(ValueError, match="Exactly one argument"):
        fn(a="hello", b="world")


def test_xor_args_multiple_groups() -> None:
    """xor_args works with multiple independent groups."""

    @xor_args(("a", "b"), ("c", "d"))
    def fn(
        *,
        a: str | None = None,
        b: str | None = None,
        c: str | None = None,
        d: str | None = None,
    ) -> str:
        return f"{a or b}-{c or d}"

    assert fn(a="1", c="2") == "1-2"
    assert fn(b="1", d="2") == "1-2"

    with pytest.raises(ValueError, match="Exactly one argument"):
        fn(a="1", b="2", c="3")


# ---------------------------------------------------------------------------
# raise_for_status_with_text
# ---------------------------------------------------------------------------
from langchain_core.utils.utils import raise_for_status_with_text


def test_raise_for_status_with_text_success() -> None:
    """No error raised for successful response."""
    from unittest.mock import MagicMock

    response = MagicMock()
    response.raise_for_status.return_value = None
    raise_for_status_with_text(response)


def test_raise_for_status_with_text_error() -> None:
    """ValueError raised with response text for failed response."""
    from unittest.mock import MagicMock

    from requests import HTTPError

    response = MagicMock()
    response.raise_for_status.side_effect = HTTPError("404 Not Found")
    response.text = "Page not found"

    with pytest.raises(ValueError, match="Page not found"):
        raise_for_status_with_text(response)


# ---------------------------------------------------------------------------
# mock_now
# ---------------------------------------------------------------------------
import datetime

from langchain_core.utils.utils import mock_now


def test_mock_now_overrides_datetime_now() -> None:
    """mock_now should override datetime.datetime.now()."""
    fake_time = datetime.datetime(2020, 1, 15, 12, 30, 45)
    with mock_now(fake_time):
        assert datetime.datetime.now() == fake_time

    # After context, datetime.now() should return real time
    assert datetime.datetime.now() != fake_time


def test_mock_now_preserves_microseconds() -> None:
    """mock_now should preserve microseconds from the provided datetime."""
    fake_time = datetime.datetime(2023, 6, 15, 10, 20, 30, 123456)
    with mock_now(fake_time):
        now = datetime.datetime.now()
        assert now.microsecond == 123456
        assert now.year == 2023
        assert now.month == 6


def test_mock_now_yields_mocked_class() -> None:
    """mock_now should yield the mocked datetime class."""
    fake_time = datetime.datetime(2020, 1, 1)
    with mock_now(fake_time) as mocked_dt:
        assert mocked_dt.now() == fake_time


# ---------------------------------------------------------------------------
# convert_to_secret_str
# ---------------------------------------------------------------------------
from langchain_core.utils.utils import convert_to_secret_str


def test_convert_to_secret_str_from_string() -> None:
    """Converts a plain string to SecretStr."""
    result = convert_to_secret_str("my_secret")
    assert isinstance(result, SecretStr)
    assert result.get_secret_value() == "my_secret"


def test_convert_to_secret_str_from_secret_str() -> None:
    """Returns SecretStr unchanged."""
    original = SecretStr("already_secret")
    result = convert_to_secret_str(original)
    assert result is original


def test_convert_to_secret_str_empty_string() -> None:
    """Converts an empty string to SecretStr."""
    result = convert_to_secret_str("")
    assert isinstance(result, SecretStr)
    assert result.get_secret_value() == ""


# ---------------------------------------------------------------------------
# build_extra_kwargs (deprecated but still public)
# ---------------------------------------------------------------------------
from langchain_core.utils.utils import build_extra_kwargs


def test_build_extra_kwargs_moves_unknown_fields() -> None:
    """Unknown fields are moved to extra_kwargs."""
    extra = {}
    values = {"known_field": "v1", "unknown_field": "v2"}
    required = {"known_field"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = build_extra_kwargs(extra, values, required)

    assert result == {"unknown_field": "v2"}
    assert "unknown_field" not in values


def test_build_extra_kwargs_raises_on_duplicate() -> None:
    """Raises ValueError if a field is in both values and extra_kwargs."""
    extra = {"dup": "from_extra"}
    values = {"dup": "from_values"}
    required = set()

    with pytest.raises(ValueError, match="Found dup supplied twice"):
        build_extra_kwargs(extra, values, required)


def test_build_extra_kwargs_raises_on_overlap_with_required() -> None:
    """Raises ValueError if extra_kwargs contains a required field name."""
    extra = {"required_field": "in_extra"}
    values = {}
    required = {"required_field"}

    with pytest.raises(ValueError, match="should be specified explicitly"):
        build_extra_kwargs(extra, values, required)


# ---------------------------------------------------------------------------
# _build_model_kwargs
# ---------------------------------------------------------------------------
import warnings

from langchain_core.utils.utils import _build_model_kwargs


def test_build_model_kwargs_moves_unknown_fields() -> None:
    """Unknown fields are moved to model_kwargs."""
    values = {"known": "v1", "extra_param": "v2", "model_kwargs": {}}
    required = {"known", "model_kwargs"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _build_model_kwargs(values, required)

    assert result["model_kwargs"] == {"extra_param": "v2"}
    assert "extra_param" not in result


def test_build_model_kwargs_raises_on_duplicate() -> None:
    """Raises ValueError if a field is in both values and model_kwargs."""
    values = {"field1": "v1", "model_kwargs": {"field1": "v2"}}
    required = {"field1", "model_kwargs"}

    with pytest.raises(ValueError, match="Found field1 supplied twice"):
        _build_model_kwargs(values, required)


def test_build_model_kwargs_warns_on_overlap_with_required() -> None:
    """Warns and moves required fields out of model_kwargs."""
    values = {"model_kwargs": {"known": "v1"}}
    required = {"known", "model_kwargs"}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _build_model_kwargs(values, required)

    assert result["known"] == "v1"
    assert result["model_kwargs"] == {}
    assert len(w) > 0
    assert "should be specified explicitly" in str(w[0].message)


# ---------------------------------------------------------------------------
# ensure_id
# ---------------------------------------------------------------------------
from langchain_core.utils.utils import LC_AUTO_PREFIX, ensure_id


def test_ensure_id_with_value() -> None:
    """Returns provided ID unchanged."""
    assert ensure_id("my-id") == "my-id"


def test_ensure_id_with_none() -> None:
    """Generates a new lc_ prefixed UUID when None."""
    result = ensure_id(None)
    assert result.startswith(LC_AUTO_PREFIX)
    # The rest should be a valid UUID4 string
    import uuid

    uuid_part = result[len(LC_AUTO_PREFIX) :]
    uuid.UUID(uuid_part, version=4)  # should not raise


def test_ensure_id_with_empty_string() -> None:
    """Generates a new ID when empty string (falsy)."""
    result = ensure_id("")
    assert result.startswith(LC_AUTO_PREFIX)


# ---------------------------------------------------------------------------
# from_env with list of keys
# ---------------------------------------------------------------------------
def test_from_env_with_list_keys_first_found() -> None:
    """from_env with list of keys returns first matching env var."""
    with patch.dict(os.environ, {"KEY_A": "value_a", "KEY_B": "value_b"}, clear=True):
        get_value = from_env(["KEY_A", "KEY_B"])
        assert get_value() == "value_a"


def test_from_env_with_list_keys_second_found() -> None:
    """from_env with list of keys falls back to second key."""
    with patch.dict(os.environ, {"KEY_B": "value_b"}, clear=True):
        get_value = from_env(["KEY_A", "KEY_B"])
        assert get_value() == "value_b"


def test_from_env_with_list_keys_none_found_default() -> None:
    """from_env with list of keys returns default when no keys found."""
    with patch.dict(os.environ, {}, clear=True):
        get_value = from_env(["KEY_A", "KEY_B"], default="fallback")
        assert get_value() == "fallback"


def test_from_env_with_list_keys_none_found_raises() -> None:
    """from_env with list of keys raises when no keys found and no default."""
    with patch.dict(os.environ, {}, clear=True):
        get_value = from_env(["KEY_A", "KEY_B"])
        with pytest.raises(ValueError, match="Did not find"):
            get_value()


def test_from_env_with_none_default() -> None:
    """from_env with default=None returns None."""
    with patch.dict(os.environ, {}, clear=True):
        get_value = from_env("MISSING_KEY", default=None)
        assert get_value() is None


# ---------------------------------------------------------------------------
# secret_from_env with list of keys
# ---------------------------------------------------------------------------
def test_secret_from_env_with_list_keys_first_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """secret_from_env with list of keys returns first matching env var."""
    monkeypatch.setenv("SEC_A", "secret_a")
    monkeypatch.setenv("SEC_B", "secret_b")
    get_secret = secret_from_env(["SEC_A", "SEC_B"])
    result = get_secret()
    assert isinstance(result, SecretStr)
    assert result.get_secret_value() == "secret_a"


def test_secret_from_env_with_list_keys_second_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """secret_from_env with list of keys falls back to second key."""
    monkeypatch.delenv("SEC_A", raising=False)
    monkeypatch.setenv("SEC_B", "secret_b")
    get_secret = secret_from_env(["SEC_A", "SEC_B"])
    result = get_secret()
    assert isinstance(result, SecretStr)
    assert result.get_secret_value() == "secret_b"


def test_secret_from_env_with_list_keys_none_found_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """secret_from_env with list of keys returns default when no keys found."""
    monkeypatch.delenv("SEC_A", raising=False)
    monkeypatch.delenv("SEC_B", raising=False)
    get_secret = secret_from_env(["SEC_A", "SEC_B"], default=None)
    assert get_secret() is None


# ---------------------------------------------------------------------------
# check_package_version - more cases for lt_version and lte_version
# ---------------------------------------------------------------------------
def test_check_package_version_lt_version_pass() -> None:
    """lt_version passes when actual version is less."""
    with patch("langchain_core.utils.utils.version", return_value="0.1.0"):
        check_package_version("stub", lt_version="0.2.0")


def test_check_package_version_lt_version_fail() -> None:
    """lt_version fails when actual version is equal."""
    with patch("langchain_core.utils.utils.version", return_value="0.2.0"):
        with pytest.raises(ValueError, match="< 0.2.0"):
            check_package_version("stub", lt_version="0.2.0")


def test_check_package_version_lte_version_pass() -> None:
    """lte_version passes when actual version is equal."""
    with patch("langchain_core.utils.utils.version", return_value="0.2.0"):
        check_package_version("stub", lte_version="0.2.0")


def test_check_package_version_lte_version_fail() -> None:
    """lte_version fails when actual version is greater."""
    with patch("langchain_core.utils.utils.version", return_value="0.3.0"):
        with pytest.raises(ValueError, match="<= 0.2.0"):
            check_package_version("stub", lte_version="0.2.0")


def test_check_package_version_gte_version_fail() -> None:
    """gte_version fails when actual version is less."""
    with patch("langchain_core.utils.utils.version", return_value="0.0.9"):
        with pytest.raises(ValueError, match=">= 0.1.0"):
            check_package_version("stub", gte_version="0.1.0")
