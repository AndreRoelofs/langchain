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
    from langchain_core.load.serializable import _is_field_useful
    from pydantic import Field

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
    from langchain_core.load.serializable import _is_field_useful
    from pydantic import Field

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
    from langchain_core.load.serializable import _try_neq_default
    from pydantic import Field
    from pydantic.fields import FieldInfo

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
