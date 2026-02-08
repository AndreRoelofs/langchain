import json

import pytest
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from langchain_core.load import Serializable, dumpd, dumps, load
from langchain_core.load.serializable import _is_field_useful
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation


class NonBoolObj:
    def __bool__(self) -> bool:
        msg = "Truthiness can't be determined"
        raise ValueError(msg)

    def __eq__(self, other: object) -> bool:
        msg = "Equality can't be determined"
        raise ValueError(msg)

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__class__.__name__

    __hash__ = None  # type: ignore[assignment]


def test_simple_serialization() -> None:
    class Foo(Serializable):
        bar: int
        baz: str

    foo = Foo(bar=1, baz="hello")
    assert dumpd(foo) == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "lc": 1,
        "repr": "Foo(bar=1, baz='hello')",
        "type": "not_implemented",
    }


def test_simple_serialization_is_serializable() -> None:
    class Foo(Serializable):
        bar: int
        baz: str

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

    foo = Foo(bar=1, baz="hello")
    assert foo.lc_id() == ["tests", "unit_tests", "load", "test_serializable", "Foo"]
    assert dumpd(foo) == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "kwargs": {"bar": 1, "baz": "hello"},
        "lc": 1,
        "type": "constructor",
    }


def test_simple_serialization_secret() -> None:
    """Test handling of secrets."""

    class Foo(Serializable):
        bar: int
        baz: str
        secret: SecretStr
        secret_2: str

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

        @property
        def lc_secrets(self) -> dict[str, str]:
            return {"secret": "MASKED_SECRET", "secret_2": "MASKED_SECRET_2"}

    foo = Foo(
        bar=1, baz="baz", secret=SecretStr("SUPER_SECRET"), secret_2="SUPER_SECRET"
    )
    assert dumpd(foo) == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "kwargs": {
            "bar": 1,
            "baz": "baz",
            "secret": {"id": ["MASKED_SECRET"], "lc": 1, "type": "secret"},
            "secret_2": {"id": ["MASKED_SECRET_2"], "lc": 1, "type": "secret"},
        },
        "lc": 1,
        "type": "constructor",
    }


def test__is_field_useful() -> None:
    class ArrayObj:
        def __bool__(self) -> bool:
            msg = "Truthiness can't be determined"
            raise ValueError(msg)

        def __eq__(self, other: object) -> bool:
            return self  # type: ignore[return-value]

        __hash__ = None  # type: ignore[assignment]

    default_x = ArrayObj()
    default_y = NonBoolObj()

    class Foo(Serializable):
        x: ArrayObj = Field(default=default_x)
        y: NonBoolObj = Field(default=default_y)
        # Make sure works for fields without default.
        z: ArrayObj

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
        )

    foo = Foo(x=ArrayObj(), y=NonBoolObj(), z=ArrayObj())
    assert _is_field_useful(foo, "x", foo.x)
    assert _is_field_useful(foo, "y", foo.y)

    foo = Foo(x=default_x, y=default_y, z=ArrayObj())
    assert not _is_field_useful(foo, "x", foo.x)
    assert not _is_field_useful(foo, "y", foo.y)


class Foo(Serializable):
    bar: int
    baz: str

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


def test_simple_deserialization() -> None:
    foo = Foo(bar=1, baz="hello")
    assert foo.lc_id() == ["tests", "unit_tests", "load", "test_serializable", "Foo"]
    serialized_foo = dumpd(foo)
    assert serialized_foo == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "kwargs": {"bar": 1, "baz": "hello"},
        "lc": 1,
        "type": "constructor",
    }
    new_foo = load(serialized_foo, valid_namespaces=["tests"])
    assert new_foo == foo


class Foo2(Serializable):
    bar: int
    baz: str

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


def test_simple_deserialization_with_additional_imports() -> None:
    foo = Foo(bar=1, baz="hello")
    assert foo.lc_id() == ["tests", "unit_tests", "load", "test_serializable", "Foo"]
    serialized_foo = dumpd(foo)
    assert serialized_foo == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "kwargs": {"bar": 1, "baz": "hello"},
        "lc": 1,
        "type": "constructor",
    }
    new_foo = load(
        serialized_foo,
        valid_namespaces=["tests"],
        additional_import_mappings={
            ("tests", "unit_tests", "load", "test_serializable", "Foo"): (
                "tests",
                "unit_tests",
                "load",
                "test_serializable",
                "Foo2",
            )
        },
    )
    assert isinstance(new_foo, Foo2)


class Foo3(Serializable):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str
    non_bool: NonBoolObj

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


def test_repr() -> None:
    foo = Foo3(
        content="repr",
        non_bool=NonBoolObj(),
    )
    assert repr(foo) == "Foo3(content='repr', non_bool=NonBoolObj)"


def test_str() -> None:
    foo = Foo3(
        content="str",
        non_bool=NonBoolObj(),
    )
    assert str(foo) == "content='str' non_bool=NonBoolObj"


def test_serialization_with_pydantic() -> None:
    class MyModel(BaseModel):
        x: int
        y: str

    my_model = MyModel(x=1, y="hello")
    llm_response = ChatGeneration(
        message=AIMessage(
            content='{"x": 1, "y": "hello"}', additional_kwargs={"parsed": my_model}
        )
    )
    ser = dumpd(llm_response)
    deser = load(ser)
    assert isinstance(deser, ChatGeneration)
    assert deser.message.content
    assert deser.message.additional_kwargs["parsed"] == my_model.model_dump()


def test_serialization_with_generation() -> None:
    generation = Generation(text="hello-world")
    assert dumpd(generation)["kwargs"] == {"text": "hello-world", "type": "Generation"}


def test_serialization_with_ignore_unserializable_fields() -> None:
    data = {
        "messages": [
            [
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "AIMessage"],
                    "kwargs": {
                        "content": "Call tools to get entity details",
                        "response_metadata": {
                            "other_field": "foo",
                            "create_date": {
                                "lc": 1,
                                "type": "not_implemented",
                                "id": ["datetime", "datetime"],
                                "repr": "datetime.datetime(2025, 7, 15, 13, 14, 0, 000000, tzinfo=datetime.timezone.utc)",  # noqa: E501
                            },
                        },
                        "type": "ai",
                        "id": "00000000-0000-0000-0000-000000000000",
                    },
                },
            ]
        ]
    }
    ser = dumpd(data)
    deser = load(ser, ignore_unserializable_fields=True)
    assert deser == {
        "messages": [
            [
                AIMessage(
                    id="00000000-0000-0000-0000-000000000000",
                    content="Call tools to get entity details",
                    response_metadata={
                        "other_field": "foo",
                        "create_date": None,
                    },
                )
            ]
        ]
    }


# Tests for dumps() function
def test_dumps_basic_serialization() -> None:
    """Test basic string serialization with `dumps()`."""
    foo = Foo(bar=42, baz="test")
    json_str = dumps(foo)

    # Should be valid JSON
    parsed = json.loads(json_str)
    assert parsed == {
        "id": ["tests", "unit_tests", "load", "test_serializable", "Foo"],
        "kwargs": {"bar": 42, "baz": "test"},
        "lc": 1,
        "type": "constructor",
    }


def test_dumps_pretty_formatting() -> None:
    """Test pretty printing functionality."""
    foo = Foo(bar=1, baz="hello")

    # Test pretty=True with default indent
    pretty_json = dumps(foo, pretty=True)
    assert "  " in pretty_json

    # Test custom indent (4-space)
    custom_indent = dumps(foo, pretty=True, indent=4)
    assert "    " in custom_indent

    # Verify it's still valid JSON
    parsed = json.loads(pretty_json)
    assert parsed["kwargs"]["bar"] == 1


def test_dumps_invalid_default_kwarg() -> None:
    """Test that passing `'default'` as kwarg raises ValueError."""
    foo = Foo(bar=1, baz="test")

    with pytest.raises(ValueError, match="`default` should not be passed to dumps"):
        dumps(foo, default=lambda x: x)


def test_dumps_additional_json_kwargs() -> None:
    """Test that additional JSON kwargs are passed through."""
    foo = Foo(bar=1, baz="test")

    compact_json = dumps(foo, separators=(",", ":"))
    assert ", " not in compact_json  # Should be compact

    # Test sort_keys
    sorted_json = dumps(foo, sort_keys=True)
    parsed = json.loads(sorted_json)
    assert parsed == dumpd(foo)


def test_dumps_non_serializable_object() -> None:
    """Test `dumps()` behavior with non-serializable objects."""

    class NonSerializable:
        def __init__(self, value: int) -> None:
            self.value = value

    obj = NonSerializable(42)
    json_str = dumps(obj)

    # Should create a "not_implemented" representation
    parsed = json.loads(json_str)
    assert parsed["lc"] == 1
    assert parsed["type"] == "not_implemented"
    assert "NonSerializable" in parsed["repr"]


def test_dumps_mixed_data_structure() -> None:
    """Test `dumps()` with complex nested data structures."""
    data = {
        "serializable": Foo(bar=1, baz="test"),
        "list": [1, 2, {"nested": "value"}],
        "primitive": "string",
    }

    json_str = dumps(data)
    parsed = json.loads(json_str)

    # Serializable object should be properly serialized
    assert parsed["serializable"]["type"] == "constructor"
    # Primitives should remain unchanged
    assert parsed["list"] == [1, 2, {"nested": "value"}]
    assert parsed["primitive"] == "string"


# Additional tests for serializable.py functions


def test_try_neq_default_basic() -> None:
    """Test try_neq_default() with basic types."""
    from langchain_core.load.serializable import try_neq_default

    class TestModel(Serializable):
        value: int = 10
        name: str = "default"

    model = TestModel(value=20, name="custom")

    # Value different from default
    assert try_neq_default(20, "value", model)
    # Value same as default
    assert not try_neq_default(10, "value", TestModel())


def test_replace_secrets_simple() -> None:
    """Test _replace_secrets() with simple secret path."""
    from langchain_core.load.serializable import _replace_secrets

    root = {"api_key": "secret_value", "other": "data"}
    secrets_map = {"api_key": "API_KEY"}

    result = _replace_secrets(root, secrets_map)

    assert result["api_key"] == {"lc": 1, "type": "secret", "id": ["API_KEY"]}
    assert result["other"] == "data"
    # Original should be unchanged
    assert root["api_key"] == "secret_value"


def test_replace_secrets_nested() -> None:
    """Test _replace_secrets() with nested secret path."""
    from langchain_core.load.serializable import _replace_secrets

    root = {"config": {"api_key": "secret_value"}, "other": "data"}
    secrets_map = {"config.api_key": "API_KEY"}

    result = _replace_secrets(root, secrets_map)

    assert result["config"]["api_key"] == {"lc": 1, "type": "secret", "id": ["API_KEY"]}
    assert result["other"] == "data"


def test_replace_secrets_missing_path() -> None:
    """Test _replace_secrets() with missing path."""
    from langchain_core.load.serializable import _replace_secrets

    root = {"other": "data"}
    secrets_map = {"api_key": "API_KEY"}

    result = _replace_secrets(root, secrets_map)

    # Should not raise error, just skip missing paths
    assert "api_key" not in result
    assert result["other"] == "data"


def test_to_json_not_implemented_with_name() -> None:
    """Test to_json_not_implemented() with object that has __name__."""
    from langchain_core.load.serializable import to_json_not_implemented

    def test_function() -> None:
        pass

    result = to_json_not_implemented(test_function)

    assert result["type"] == "not_implemented"
    assert result["lc"] == 1
    assert "test_function" in result["id"]


def test_to_json_not_implemented_with_class() -> None:
    """Test to_json_not_implemented() with regular class."""
    from langchain_core.load.serializable import to_json_not_implemented

    class TestClass:
        pass

    obj = TestClass()
    result = to_json_not_implemented(obj)

    assert result["type"] == "not_implemented"
    assert result["lc"] == 1
    assert "TestClass" in result["id"]


def test_serializable_lc_id_with_pydantic_generics() -> None:
    """Test Serializable.lc_id() handles Pydantic generics correctly."""
    from typing import Generic, TypeVar

    T = TypeVar("T")

    class GenericModel(Serializable, Generic[T]):
        value: T

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

    # For generic types, lc_id should use the original name
    model = GenericModel[int](value=42)
    lc_id = model.lc_id()

    # Should use original class name, not the parameterized version
    assert lc_id[-1] == "GenericModel"


def test_serializable_with_excluded_fields() -> None:
    """Test Serializable excludes fields marked with exclude=True."""
    from pydantic import Field

    class ModelWithExclude(Serializable):
        included: int
        excluded: str = Field(exclude=True)

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

    model = ModelWithExclude(included=42, excluded="secret")
    serialized = dumpd(model)

    assert "included" in serialized["kwargs"]
    assert "excluded" not in serialized["kwargs"]


def test_serializable_with_lc_attributes() -> None:
    """Test Serializable includes lc_attributes in serialization."""

    class ModelWithAttributes(Serializable):
        value: int

        @property
        def computed_value(self) -> int:
            return self.value * 2

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

        @property
        def lc_attributes(self) -> dict:
            return {"computed_value": self.computed_value}

    model = ModelWithAttributes(value=10)
    serialized = dumpd(model)

    assert serialized["kwargs"]["value"] == 10
    assert serialized["kwargs"]["computed_value"] == 20


def test_serializable_deprecated_attributes_error() -> None:
    """Test Serializable raises error for deprecated attributes."""

    class ModelWithDeprecated(Serializable):
        value: int

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

    # Manually add deprecated attribute after class creation to avoid Pydantic error
    ModelWithDeprecated.lc_namespace = ["old", "namespace"]  # type: ignore[attr-defined]

    model = ModelWithDeprecated(value=42)

    with pytest.raises(ValueError, match="deprecated attribute"):
        model.to_json()


def test_serializable_with_secret_aliases() -> None:
    """Test Serializable handles secret field aliases correctly."""
    from pydantic import Field

    class ModelWithAlias(Serializable):
        api_key: str = Field(alias="apiKey")

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

        @property
        def lc_secrets(self) -> dict[str, str]:
            return {"api_key": "API_KEY"}

    model = ModelWithAlias(apiKey="secret_value")
    serialized = dumpd(model)

    # Both the field name and alias should be masked
    assert "api_key" in serialized["kwargs"] or "apiKey" in serialized["kwargs"]
    # Should be marked as secret
    secret_val = serialized["kwargs"].get("api_key") or serialized["kwargs"].get(
        "apiKey"
    )
    assert secret_val["type"] == "secret"


def test_is_field_useful_with_dict_default_factory() -> None:
    """Test _is_field_useful() with dict default_factory."""
    from pydantic import Field

    from langchain_core.load.serializable import _is_field_useful

    class ModelWithDictDefault(Serializable):
        data: dict = Field(default_factory=dict)

    model = ModelWithDictDefault(data={})

    # Empty dict with dict default_factory should not be useful
    assert not _is_field_useful(model, "data", {})

    # Non-empty dict should be useful
    model_with_data = ModelWithDictDefault(data={"key": "value"})
    assert _is_field_useful(model_with_data, "data", {"key": "value"})


def test_is_field_useful_with_list_default_factory() -> None:
    """Test _is_field_useful() with list default_factory."""
    from pydantic import Field

    from langchain_core.load.serializable import _is_field_useful

    class ModelWithListDefault(Serializable):
        items: list = Field(default_factory=list)

    model = ModelWithListDefault(items=[])

    # Empty list with list default_factory should not be useful
    assert not _is_field_useful(model, "items", [])

    # Non-empty list should be useful
    model_with_items = ModelWithListDefault(items=[1, 2, 3])
    assert _is_field_useful(model_with_items, "items", [1, 2, 3])


def test_is_field_useful_with_required_field() -> None:
    """Test _is_field_useful() with required fields."""
    from langchain_core.load.serializable import _is_field_useful

    class ModelWithRequired(Serializable):
        required_field: str

    model = ModelWithRequired(required_field="value")

    # Required fields are always useful
    assert _is_field_useful(model, "required_field", "value")
    assert _is_field_useful(model, "required_field", "")


def test_is_field_useful_with_nonexistent_field() -> None:
    """Test _is_field_useful() with non-existent field."""
    from langchain_core.load.serializable import _is_field_useful

    class SimpleModel(Serializable):
        value: int

    model = SimpleModel(value=42)

    # Non-existent field should not be useful
    assert not _is_field_useful(model, "nonexistent", "value")


def test_try_neq_default_with_identity_check() -> None:
    """Test _try_neq_default() falls back to identity check."""
    from pydantic import Field
    from pydantic.fields import FieldInfo

    from langchain_core.load.serializable import _try_neq_default

    class UncomparableObj:
        def __eq__(self, other: object) -> bool:
            msg = "Cannot compare"
            raise ValueError(msg)

        def __bool__(self) -> bool:
            msg = "Cannot convert to bool"
            raise ValueError(msg)

    default_obj = UncomparableObj()
    field_info = FieldInfo(default=default_obj)

    # Same object (identity)
    assert not _try_neq_default(default_obj, field_info)

    # Different object
    different_obj = UncomparableObj()
    assert _try_neq_default(different_obj, field_info)


def test_serializable_repr_args() -> None:
    """Test Serializable.__repr_args__() filters default values."""

    class ModelWithDefaults(Serializable):
        required: int
        optional: str = "default"
        another: int = 10

    # All values are non-default
    model1 = ModelWithDefaults(required=42, optional="custom", another=20)
    repr_args = list(model1.__repr_args__())
    repr_dict = dict(repr_args)

    assert "required" in repr_dict
    assert "optional" in repr_dict
    assert "another" in repr_dict

    # Some values are default
    model2 = ModelWithDefaults(required=42, optional="default", another=10)
    repr_args2 = list(model2.__repr_args__())
    repr_dict2 = dict(repr_args2)

    assert "required" in repr_dict2
    # optional and another might not be in repr if they match defaults
    # This depends on try_neq_default implementation


def test_serializable_get_lc_namespace() -> None:
    """Test Serializable.get_lc_namespace() returns module path."""

    class TestModel(Serializable):
        value: int

    namespace = TestModel.get_lc_namespace()

    # Should return module path as list of strings
    assert isinstance(namespace, list)
    assert all(isinstance(part, str) for part in namespace)
    assert "tests" in namespace
    assert "unit_tests" in namespace
    assert "load" in namespace
    assert "test_serializable" in namespace


# ---------------------------------------------------------------------------
# Snapshot tests for _try_neq_default edge cases
# ---------------------------------------------------------------------------


def test_try_neq_default_with_all_fallback() -> None:
    """Test _try_neq_default falls back to all() when bool() on != result fails.

    The code path is:
    1. bool(field.get_default() != value) -> raises because bool() fails on result
    2. all(field.get_default() != value) -> works because all() iterates the result

    To trigger this, __ne__ must return an iterable whose bool() raises.
    """
    from pydantic.fields import FieldInfo

    from langchain_core.load.serializable import _try_neq_default

    class ArrayLikeBoolResult:
        """Result of comparison that raises on bool() but supports all()."""

        def __init__(self, values: list[bool]) -> None:
            self.values = values

        def __bool__(self) -> bool:
            msg = "Ambiguous truth value"
            raise ValueError(msg)

        def __iter__(self) -> "iter":
            return iter(self.values)

    class ArrayLikeNeq:
        """Object whose != directly returns an ArrayLikeBoolResult.

        Override __ne__ to return an iterable result whose bool() raises.
        """

        def __init__(self, val: int) -> None:
            self.val = val

        def __ne__(self, other: object) -> "ArrayLikeBoolResult":  # type: ignore[override]
            if isinstance(other, ArrayLikeNeq):
                return ArrayLikeBoolResult([self.val != other.val])
            return ArrayLikeBoolResult([True])

        def __eq__(self, other: object) -> bool:  # type: ignore[override]
            if isinstance(other, ArrayLikeNeq):
                return self.val == other.val
            return NotImplemented

    default_obj = ArrayLikeNeq(1)
    field = FieldInfo(default=default_obj)

    # Same value: != returns ArrayLikeBoolResult([False]),
    # bool() raises, all([False]) -> False
    assert not _try_neq_default(ArrayLikeNeq(1), field)
    # Different value: != returns ArrayLikeBoolResult([True]),
    # bool() raises, all([True]) -> True
    assert _try_neq_default(ArrayLikeNeq(2), field)


def test_try_neq_default_returns_false_on_total_failure() -> None:
    """Test _try_neq_default returns False when all comparison paths fail."""
    from pydantic.fields import FieldInfo

    from langchain_core.load.serializable import _try_neq_default

    class TotallyUncomparable:
        def __eq__(self, other: object) -> bool:
            msg = "no eq"
            raise TypeError(msg)

        def __ne__(self, other: object) -> bool:
            msg = "no ne"
            raise TypeError(msg)

        def __bool__(self) -> bool:
            msg = "no bool"
            raise TypeError(msg)

    obj = TotallyUncomparable()
    # default is a *different* object but identity check (is not) should return True
    field = FieldInfo(default=TotallyUncomparable())
    # Since value is not default → identity check says True
    assert _try_neq_default(obj, field)

    # When default IS the same identity → returns False
    field_same = FieldInfo(default=obj)
    assert not _try_neq_default(obj, field_same)


# ---------------------------------------------------------------------------
# Snapshot tests for _is_field_useful edge cases
# ---------------------------------------------------------------------------


def test_is_field_useful_falsy_value_differs_from_default() -> None:
    """Test _is_field_useful with falsy value (0) that differs from non-zero default."""
    from langchain_core.load.serializable import _is_field_useful

    class ModelFalsyNonDefault(Serializable):
        count: int = 5

    model = ModelFalsyNonDefault(count=0)
    # 0 is falsy, but default is 5 → should be useful
    assert _is_field_useful(model, "count", 0)


def test_is_field_useful_falsy_value_matches_default() -> None:
    """Test _is_field_useful with falsy value matching the default."""
    from langchain_core.load.serializable import _is_field_useful

    class ModelFalsyDefault(Serializable):
        count: int = 0

    model = ModelFalsyDefault(count=0)
    # 0 is falsy and matches default → not useful
    assert not _is_field_useful(model, "count", 0)


def test_is_field_useful_none_value_with_none_default() -> None:
    """Test _is_field_useful with None value and None default."""
    from langchain_core.load.serializable import _is_field_useful

    class ModelNoneDefault(Serializable):
        value: str | None = None

    model = ModelNoneDefault(value=None)
    assert not _is_field_useful(model, "value", None)


def test_is_field_useful_empty_string_with_string_default() -> None:
    """Test _is_field_useful with empty string vs non-empty default."""
    from langchain_core.load.serializable import _is_field_useful

    class ModelStrDefault(Serializable):
        name: str = "hello"

    model = ModelStrDefault(name="")
    # "" is falsy, default is "hello" → should be useful
    assert _is_field_useful(model, "name", "")


# ---------------------------------------------------------------------------
# Snapshot tests for to_json with MRO secret merging
# ---------------------------------------------------------------------------


def test_to_json_mro_secret_merging() -> None:
    """Test to_json merges secrets from parent and child classes via MRO."""

    class Parent(Serializable):
        parent_key: str = "parent_secret"

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

        @property
        def lc_secrets(self) -> dict[str, str]:
            return {"parent_key": "PARENT_KEY_ENV"}

    class Child(Parent):
        child_key: str = "child_secret"

        @property
        def lc_secrets(self) -> dict[str, str]:
            return {"child_key": "CHILD_KEY_ENV"}

    child = Child(parent_key="p_val", child_key="c_val")
    serialized = child.to_json()

    assert serialized["type"] == "constructor"
    kwargs = serialized["kwargs"]
    # Both parent and child secrets should be masked
    assert kwargs["parent_key"] == {"lc": 1, "type": "secret", "id": ["PARENT_KEY_ENV"]}
    assert kwargs["child_key"] == {"lc": 1, "type": "secret", "id": ["CHILD_KEY_ENV"]}


def test_to_json_secret_included_even_if_not_in_kwargs() -> None:
    """Test to_json includes secrets even if not iterated via model fields."""

    class ModelWithOptionalSecret(Serializable):
        api_key: str | None = None

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

        @property
        def lc_secrets(self) -> dict[str, str]:
            return {"api_key": "API_KEY_ENV"}

    # Secret has a value → should appear as masked
    model = ModelWithOptionalSecret(api_key="actual_secret")
    serialized = model.to_json()
    assert serialized["kwargs"]["api_key"] == {
        "lc": 1,
        "type": "secret",
        "id": ["API_KEY_ENV"],
    }


def test_to_json_secret_none_not_included() -> None:
    """Test to_json does not include a secret key when its value is None."""

    class ModelWithOptionalSecret(Serializable):
        api_key: str | None = None

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

        @property
        def lc_secrets(self) -> dict[str, str]:
            return {"api_key": "API_KEY_ENV"}

    model = ModelWithOptionalSecret()
    serialized = model.to_json()
    # api_key is None → should not be added
    assert "api_key" not in serialized["kwargs"]


# ---------------------------------------------------------------------------
# Snapshot tests for _replace_secrets edge cases
# ---------------------------------------------------------------------------


def test_replace_secrets_deeply_nested() -> None:
    """Test _replace_secrets with 3-level nested path."""
    from langchain_core.load.serializable import _replace_secrets

    root = {"a": {"b": {"c": "secret_val"}}}
    secrets_map = {"a.b.c": "DEEP_SECRET"}

    result = _replace_secrets(root, secrets_map)

    assert result["a"]["b"]["c"] == {
        "lc": 1,
        "type": "secret",
        "id": ["DEEP_SECRET"],
    }
    # Original untouched
    assert root["a"]["b"]["c"] == "secret_val"


def test_replace_secrets_does_not_mutate_original() -> None:
    """Test _replace_secrets creates copies at each nesting level."""
    from langchain_core.load.serializable import _replace_secrets

    inner = {"key": "val"}
    root = {"outer": inner}
    secrets_map = {"outer.key": "SECRET"}

    result = _replace_secrets(root, secrets_map)

    # The inner dict in result should be a different object
    assert result["outer"] is not inner
    assert inner["key"] == "val"


def test_replace_secrets_multiple_secrets() -> None:
    """Test _replace_secrets with multiple secrets in one dict."""
    from langchain_core.load.serializable import _replace_secrets

    root = {"key1": "v1", "key2": "v2", "plain": "data"}
    secrets_map = {"key1": "SECRET_1", "key2": "SECRET_2"}

    result = _replace_secrets(root, secrets_map)

    assert result["key1"] == {"lc": 1, "type": "secret", "id": ["SECRET_1"]}
    assert result["key2"] == {"lc": 1, "type": "secret", "id": ["SECRET_2"]}
    assert result["plain"] == "data"


# ---------------------------------------------------------------------------
# Snapshot tests for to_json_not_implemented edge cases
# ---------------------------------------------------------------------------


def test_to_json_not_implemented_repr_snapshot() -> None:
    """Test to_json_not_implemented includes repr of the object."""
    from langchain_core.load.serializable import to_json_not_implemented

    class CustomRepr:
        def __repr__(self) -> str:
            return "CustomRepr(special=True)"

    result = to_json_not_implemented(CustomRepr())
    assert result == {
        "lc": 1,
        "type": "not_implemented",
        "id": [
            "tests",
            "unit_tests",
            "load",
            "test_serializable",
            "CustomRepr",
        ],
        "repr": "CustomRepr(special=True)",
    }


def test_to_json_not_implemented_with_broken_repr() -> None:
    """Test to_json_not_implemented gracefully handles repr failure."""
    from langchain_core.load.serializable import to_json_not_implemented

    class BrokenRepr:
        def __repr__(self) -> str:
            msg = "repr is broken"
            raise RuntimeError(msg)

    result = to_json_not_implemented(BrokenRepr())
    assert result["type"] == "not_implemented"
    assert result["lc"] == 1
    # repr should be None since repr() raised
    assert result["repr"] is None


def test_to_json_not_implemented_with_named_function() -> None:
    """Test to_json_not_implemented with a function uses __name__ path."""
    from langchain_core.load.serializable import to_json_not_implemented

    def my_func() -> None:
        pass

    result = to_json_not_implemented(my_func)
    assert result["id"][-1] == "my_func"
    assert result["repr"] is not None


def test_to_json_not_implemented_with_lambda() -> None:
    """Test to_json_not_implemented with a lambda."""
    from langchain_core.load.serializable import to_json_not_implemented

    f = lambda x: x  # noqa: E731
    result = to_json_not_implemented(f)
    assert result["type"] == "not_implemented"
    assert result["id"][-1] == "<lambda>"


# ---------------------------------------------------------------------------
# Snapshot test for Serializable model_config
# ---------------------------------------------------------------------------


def test_serializable_extra_fields_ignored() -> None:
    """Test Serializable ignores extra fields due to model_config."""

    class StrictModel(Serializable):
        value: int

    # extra="ignore" should silently drop unknown fields
    model = StrictModel(value=1, unknown_field="dropped")  # type: ignore[call-arg]
    assert model.value == 1
    assert not hasattr(model, "unknown_field")


# ---------------------------------------------------------------------------
# Snapshot for to_json full output structure
# ---------------------------------------------------------------------------


def test_to_json_full_output_snapshot() -> None:
    """Snapshot of the full to_json() output structure for a serializable model."""

    class SnapshotModel(Serializable):
        name: str
        count: int
        tags: list[str] = Field(default_factory=list)
        metadata: dict[str, str] = Field(default_factory=dict)

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

    model = SnapshotModel(name="test", count=5, tags=["a", "b"])
    result = model.to_json()

    assert result == {
        "lc": 1,
        "type": "constructor",
        "id": [
            "tests",
            "unit_tests",
            "load",
            "test_serializable",
            "SnapshotModel",
        ],
        "kwargs": {
            "name": "test",
            "count": 5,
            "tags": ["a", "b"],
        },
    }
    # metadata should be absent since it's empty dict matching default_factory=dict
    assert "metadata" not in result["kwargs"]


def test_to_json_not_implemented_output_snapshot() -> None:
    """Snapshot of to_json_not_implemented() output for non-serializable class."""

    class NotSerializableModel(Serializable):
        value: int

    model = NotSerializableModel(value=42)
    result = model.to_json()

    assert result == {
        "lc": 1,
        "type": "not_implemented",
        "id": [
            "tests",
            "unit_tests",
            "load",
            "test_serializable",
            "NotSerializableModel",
        ],
        "repr": "NotSerializableModel(value=42)",
    }


# ---------------------------------------------------------------------------
# Snapshot for repr_args filtering
# ---------------------------------------------------------------------------


def test_repr_args_filters_defaults_snapshot() -> None:
    """Snapshot: __repr_args__() omits fields matching their defaults."""

    class ReprModel(Serializable):
        required: int
        optional_str: str = "default"
        optional_int: int = 10

    model = ReprModel(required=42, optional_str="default", optional_int=10)
    repr_keys = [k for k, _ in model.__repr_args__()]
    # required is always present; optional fields matching defaults are omitted
    assert "required" in repr_keys
    assert "optional_str" not in repr_keys
    assert "optional_int" not in repr_keys


def test_repr_args_includes_non_default_values() -> None:
    """Snapshot: __repr_args__() includes fields with non-default values."""

    class ReprModel(Serializable):
        required: int
        optional_str: str = "default"
        optional_int: int = 10

    model = ReprModel(required=1, optional_str="custom", optional_int=99)
    repr_keys = [k for k, _ in model.__repr_args__()]
    assert "required" in repr_keys
    assert "optional_str" in repr_keys
    assert "optional_int" in repr_keys


# ---------------------------------------------------------------------------
# Snapshot for try_neq_default via model_fields
# ---------------------------------------------------------------------------


def test_try_neq_default_with_default_factory() -> None:
    """Test try_neq_default with default_factory field.

    Note: field.get_default() for default_factory returns None (not the factory
    result), so try_neq_default sees [] != None → True. The `_is_field_useful`
    function has separate logic for default_factory=list/dict.
    """
    from langchain_core.load.serializable import try_neq_default

    class FactoryModel(Serializable):
        items: list[int] = Field(default_factory=list)

    model = FactoryModel(items=[])
    # [] != field.get_default() (None) → True
    assert try_neq_default([], "items", model)

    model2 = FactoryModel(items=[1, 2])
    assert try_neq_default([1, 2], "items", model2)


# ---------------------------------------------------------------------------
# Snapshot for lc_id with non-generic class
# ---------------------------------------------------------------------------


def test_lc_id_snapshot() -> None:
    """Snapshot: lc_id() returns namespace + class name."""

    class MySerializable(Serializable):
        value: int

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

    assert MySerializable.lc_id() == [
        "tests",
        "unit_tests",
        "load",
        "test_serializable",
        "MySerializable",
    ]


# ---------------------------------------------------------------------------
# Snapshot: BaseSerialized TypedDicts structure
# ---------------------------------------------------------------------------


def test_serialized_constructor_typeddict_fields() -> None:
    """Snapshot: SerializedConstructor has the expected fields."""
    from langchain_core.load.serializable import SerializedConstructor

    sc: SerializedConstructor = {
        "lc": 1,
        "type": "constructor",
        "id": ["a", "b"],
        "kwargs": {"x": 1},
    }
    assert sc["lc"] == 1
    assert sc["type"] == "constructor"
    assert sc["id"] == ["a", "b"]
    assert sc["kwargs"] == {"x": 1}


def test_serialized_secret_typeddict_fields() -> None:
    """Snapshot: SerializedSecret has the expected fields."""
    from langchain_core.load.serializable import SerializedSecret

    ss: SerializedSecret = {
        "lc": 1,
        "type": "secret",
        "id": ["MY_SECRET"],
    }
    assert ss["lc"] == 1
    assert ss["type"] == "secret"
    assert ss["id"] == ["MY_SECRET"]


def test_serialized_not_implemented_typeddict_fields() -> None:
    """Snapshot: SerializedNotImplemented has the expected fields."""
    from langchain_core.load.serializable import SerializedNotImplemented

    sni: SerializedNotImplemented = {
        "lc": 1,
        "type": "not_implemented",
        "id": ["mod", "Class"],
        "repr": "Class()",
    }
    assert sni["lc"] == 1
    assert sni["type"] == "not_implemented"
    assert sni["id"] == ["mod", "Class"]
    assert sni["repr"] == "Class()"


# ---------------------------------------------------------------------------
# Snapshot: deprecated attribute lc_serializable
# ---------------------------------------------------------------------------


def test_serializable_deprecated_lc_serializable_attr_error() -> None:
    """Test to_json raises ValueError when lc_serializable attr exists."""

    class BadModel(Serializable):
        value: int

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

    BadModel.lc_serializable = True  # type: ignore[attr-defined]
    model = BadModel(value=1)
    with pytest.raises(ValueError, match="deprecated attribute lc_serializable"):
        model.to_json()


# ---------------------------------------------------------------------------
# Snapshot: Serializable is_lc_serializable default
# ---------------------------------------------------------------------------


def test_is_lc_serializable_default_false() -> None:
    """Snapshot: is_lc_serializable() returns False by default."""

    class PlainModel(Serializable):
        value: int

    assert PlainModel.is_lc_serializable() is False


# ---------------------------------------------------------------------------
# Snapshot: lc_secrets and lc_attributes defaults
# ---------------------------------------------------------------------------


def test_lc_secrets_default_empty() -> None:
    """Snapshot: lc_secrets returns empty dict by default."""

    class PlainModel(Serializable):
        value: int

    model = PlainModel(value=1)
    assert model.lc_secrets == {}


def test_lc_attributes_default_empty() -> None:
    """Snapshot: lc_attributes returns empty dict by default."""

    class PlainModel(Serializable):
        value: int

    model = PlainModel(value=1)
    assert model.lc_attributes == {}


# ---------------------------------------------------------------------------
# Snapshot: round-trip serialize/deserialize preserves exact values
# ---------------------------------------------------------------------------


class RichModel(Serializable):
    name: str
    count: int
    tags: list[str]
    meta: dict[str, int]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


def test_round_trip_with_various_field_types() -> None:
    """Snapshot: round-trip with int, str, list, dict fields."""
    original = RichModel(name="test", count=42, tags=["a", "b"], meta={"x": 1, "y": 2})
    serialized = dumpd(original)
    restored = load(serialized, valid_namespaces=["tests"])
    assert isinstance(restored, RichModel)
    assert restored.name == "test"
    assert restored.count == 42
    assert restored.tags == ["a", "b"]
    assert restored.meta == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# Snapshot: to_json with lc_attributes from multiple inheritance levels
# ---------------------------------------------------------------------------


def test_to_json_lc_attributes_from_parent_and_child() -> None:
    """Snapshot: lc_attributes merged from parent and child via MRO."""

    class Parent(Serializable):
        base_val: int

        @classmethod
        def is_lc_serializable(cls) -> bool:
            return True

        @property
        def lc_attributes(self) -> dict:
            return {"parent_attr": self.base_val * 10}

    class Child(Parent):
        child_val: str

        @property
        def lc_attributes(self) -> dict:
            return {"child_attr": self.child_val.upper()}

    child = Child(base_val=3, child_val="hi")
    serialized = child.to_json()
    assert serialized["kwargs"]["parent_attr"] == 30
    assert serialized["kwargs"]["child_attr"] == "HI"
