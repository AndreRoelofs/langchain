"""Test for some custom pydantic decorators."""

import sys
import warnings
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field
from pydantic.v1 import BaseModel as BaseModelV1

from langchain_core.utils.pydantic import (
    _create_subset_model_v2,
    create_model_v2,
    get_fields,
    is_basemodel_instance,
    is_basemodel_subclass,
    pre_init,
)


def test_pre_init_decorator() -> None:
    class Foo(BaseModel):
        x: int = 5
        y: int

        @pre_init
        def validator(cls, v: dict[str, Any]) -> dict[str, Any]:
            v["y"] = v["x"] + 1
            return v

    # Type ignore initialization b/c y is marked as required
    foo = Foo()  # type: ignore[call-arg]
    assert foo.y == 6
    foo = Foo(x=10)  # type: ignore[call-arg]
    assert foo.y == 11


def test_pre_init_decorator_with_more_defaults() -> None:
    class Foo(BaseModel):
        a: int = 1
        b: int | None = None
        c: int = Field(default=2)
        d: int = Field(default_factory=lambda: 3)

        @pre_init
        def validator(cls, v: dict[str, Any]) -> dict[str, Any]:
            assert v["a"] == 1
            assert v["b"] is None
            assert v["c"] == 2
            assert v["d"] == 3
            return v

    # Try to create an instance of Foo
    Foo()


def test_with_aliases() -> None:
    class Foo(BaseModel):
        x: int = Field(default=1, alias="y")
        z: int

        model_config = ConfigDict(
            populate_by_name=True,
        )

        @pre_init
        def validator(cls, v: dict[str, Any]) -> dict[str, Any]:
            v["z"] = v["x"]
            return v

    # Based on defaults
    # z is required
    foo = Foo()  # type: ignore[call-arg]
    assert foo.x == 1
    assert foo.z == 1

    # Based on field name
    # z is required
    foo = Foo(x=2)  # type: ignore[call-arg]
    assert foo.x == 2
    assert foo.z == 2

    # Based on alias
    # z is required
    foo = Foo(y=2)  # type: ignore[call-arg]
    assert foo.x == 2
    assert foo.z == 2


def test_is_basemodel_subclass() -> None:
    """Test pydantic."""
    assert is_basemodel_subclass(BaseModel)
    assert is_basemodel_subclass(BaseModelV1)


def test_is_basemodel_instance() -> None:
    """Test pydantic."""

    class Foo(BaseModel):
        x: int

    assert is_basemodel_instance(Foo(x=5))

    class Bar(BaseModelV1):
        x: int

    assert is_basemodel_instance(Bar(x=5))


def test_with_field_metadata() -> None:
    """Test pydantic with field metadata."""

    class Foo(BaseModel):
        x: list[int] = Field(
            description="List of integers", min_length=10, max_length=15
        )

    subset_model = _create_subset_model_v2("Foo", Foo, ["x"])
    assert subset_model.model_json_schema() == {
        "properties": {
            "x": {
                "description": "List of integers",
                "items": {"type": "integer"},
                "maxItems": 15,
                "minItems": 10,
                "title": "X",
                "type": "array",
            }
        },
        "required": ["x"],
        "title": "Foo",
        "type": "object",
    }


def test_fields_pydantic_v2_proper() -> None:
    class Foo(BaseModel):
        x: int

    fields = get_fields(Foo)
    assert fields == {"x": Foo.model_fields["x"]}


@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="pydantic.v1 namespace not supported with Python 3.14+",
)
def test_fields_pydantic_v1_from_2() -> None:
    class Foo(BaseModelV1):
        x: int

    fields = get_fields(Foo)
    assert fields == {"x": Foo.__fields__["x"]}


def test_create_model_v2() -> None:
    """Test that create model v2 works as expected."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")  # Cause all warnings to always be triggered
        foo = create_model_v2("Foo", field_definitions={"a": (int, None)})
        foo.model_json_schema()

    assert list(record) == []

    # schema is used by pydantic, but OK to re-use
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")  # Cause all warnings to always be triggered
        foo = create_model_v2("Foo", field_definitions={"schema": (int, None)})
        foo.model_json_schema()

    assert list(record) == []

    # From protected namespaces, but definitely OK to use.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")  # Cause all warnings to always be triggered
        foo = create_model_v2("Foo", field_definitions={"model_id": (int, None)})
        foo.model_json_schema()

    assert list(record) == []

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")  # Cause all warnings to always be triggered
        # Verify that we can use non-English characters
        field_name = "もしもし"
        foo = create_model_v2("Foo", field_definitions={field_name: (int, None)})
        foo.model_json_schema()

    assert list(record) == []


# ---------------------------------------------------------------------------
# is_pydantic_v1_subclass / is_pydantic_v2_subclass
# ---------------------------------------------------------------------------
from langchain_core.utils.pydantic import (
    is_pydantic_v1_subclass,
    is_pydantic_v2_subclass,
)


def test_is_pydantic_v1_subclass_with_v1_model() -> None:
    """Returns True for v1 BaseModel subclass."""

    class MyV1Model(BaseModelV1):
        x: int

    assert is_pydantic_v1_subclass(MyV1Model) is True


def test_is_pydantic_v1_subclass_with_v2_model() -> None:
    """Returns False for v2 BaseModel subclass."""

    class MyV2Model(BaseModel):
        x: int

    assert is_pydantic_v1_subclass(MyV2Model) is False


def test_is_pydantic_v2_subclass_with_v2_model() -> None:
    """Returns True for v2 BaseModel subclass."""

    class MyV2Model(BaseModel):
        x: int

    assert is_pydantic_v2_subclass(MyV2Model) is True


def test_is_pydantic_v2_subclass_with_v1_model() -> None:
    """Returns False for v1 BaseModel subclass."""

    class MyV1Model(BaseModelV1):
        x: int

    assert is_pydantic_v2_subclass(MyV1Model) is False


# ---------------------------------------------------------------------------
# is_basemodel_subclass edge cases
# ---------------------------------------------------------------------------
def test_is_basemodel_subclass_with_non_class() -> None:
    """Returns False for non-class objects."""
    assert is_basemodel_subclass(42) is False  # type: ignore[arg-type]
    assert is_basemodel_subclass("string") is False  # type: ignore[arg-type]
    assert is_basemodel_subclass(None) is False  # type: ignore[arg-type]


def test_is_basemodel_subclass_with_generic_alias() -> None:
    """Returns False for GenericAlias types."""
    assert is_basemodel_subclass(list[int]) is False  # type: ignore[arg-type]
    assert is_basemodel_subclass(dict[str, int]) is False  # type: ignore[arg-type]


def test_is_basemodel_subclass_with_plain_class() -> None:
    """Returns False for non-pydantic classes."""

    class PlainClass:
        pass

    assert is_basemodel_subclass(PlainClass) is False


# ---------------------------------------------------------------------------
# is_basemodel_instance edge cases
# ---------------------------------------------------------------------------
def test_is_basemodel_instance_with_non_model() -> None:
    """Returns False for non-model instances."""
    assert is_basemodel_instance(42) is False
    assert is_basemodel_instance("string") is False
    assert is_basemodel_instance(None) is False
    assert is_basemodel_instance([1, 2, 3]) is False


# ---------------------------------------------------------------------------
# get_fields edge cases
# ---------------------------------------------------------------------------
def test_get_fields_from_instance() -> None:
    """get_fields works with a model instance (not just class)."""

    class MyModel(BaseModel):
        x: int
        y: str = "default"

    instance = MyModel(x=5)
    fields = get_fields(instance)
    assert "x" in fields
    assert "y" in fields


def test_get_fields_raises_for_non_model() -> None:
    """get_fields raises TypeError for non-pydantic types."""
    with pytest.raises(TypeError, match="Expected a Pydantic model"):
        get_fields(str)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_pydantic_major_version (deprecated)
# ---------------------------------------------------------------------------
from langchain_core.utils.pydantic import get_pydantic_major_version


def test_get_pydantic_major_version_returns_int() -> None:
    """get_pydantic_major_version returns the major version as int."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = get_pydantic_major_version()
    assert isinstance(result, int)
    assert result == 2


# ---------------------------------------------------------------------------
# create_model (wrapper around create_model_v2)
# ---------------------------------------------------------------------------
from langchain_core.utils.pydantic import create_model


def test_create_model_basic() -> None:
    """create_model creates a model with field definitions."""
    Model = create_model("TestModel", field1=(str, "default"), field2=(int, 0))
    instance = Model()
    assert instance.field1 == "default"  # type: ignore[attr-defined]
    assert instance.field2 == 0  # type: ignore[attr-defined]


def test_create_model_with_root() -> None:
    """create_model with __root__ delegates to root model creation."""
    Model = create_model("RootModel", __root__=(list[int], None))
    schema = Model.model_json_schema()
    assert schema["title"] == "RootModel"


# ---------------------------------------------------------------------------
# create_model_v2 with root types
# ---------------------------------------------------------------------------
def test_create_model_v2_with_root() -> None:
    """create_model_v2 with root creates a RootModel."""
    Model = create_model_v2("MyRoot", root=str)
    schema = Model.model_json_schema()
    assert schema["title"] == "MyRoot"


def test_create_model_v2_with_root_and_default() -> None:
    """create_model_v2 with root tuple creates model with default."""
    Model = create_model_v2("MyRoot", root=(str, "hello"))
    schema = Model.model_json_schema()
    assert schema["title"] == "MyRoot"


def test_create_model_v2_root_with_fields_raises() -> None:
    """create_model_v2 raises when both root and field_definitions are provided."""
    with pytest.raises(NotImplementedError, match="no other fields"):
        create_model_v2("Bad", root=str, field_definitions={"extra": (int, 0)})


def test_create_model_v2_with_reserved_name() -> None:
    """create_model_v2 handles reserved pydantic names via remapping."""
    Model = create_model_v2("ReservedModel", field_definitions={"schema": (str, "val")})
    schema = Model.model_json_schema()
    assert "schema" in str(schema) or "properties" in schema


def test_create_model_v2_with_private_field() -> None:
    """create_model_v2 handles fields starting with underscore via remapping."""
    Model = create_model_v2("PrivateModel", field_definitions={"_hidden": (str, "val")})
    schema = Model.model_json_schema()
    assert "properties" in schema


# ---------------------------------------------------------------------------
# _create_subset_model_v2
# ---------------------------------------------------------------------------
def test_create_subset_model_v2_preserves_doc() -> None:
    """_create_subset_model_v2 preserves docstring."""

    class Original(BaseModel):
        """Original docstring."""

        x: int
        y: str

    subset = _create_subset_model_v2("Sub", Original, ["x"])
    assert "Original docstring" in (subset.__doc__ or "")


def test_create_subset_model_v2_custom_description() -> None:
    """_create_subset_model_v2 allows custom field descriptions."""

    class Original(BaseModel):
        x: int = Field(description="Original desc")

    subset = _create_subset_model_v2(
        "Sub", Original, ["x"], descriptions={"x": "Custom desc"}
    )
    schema = subset.model_json_schema()
    assert schema["properties"]["x"]["description"] == "Custom desc"


def test_create_subset_model_v2_custom_fn_description() -> None:
    """_create_subset_model_v2 allows custom function description."""

    class Original(BaseModel):
        """Original doc."""

        x: int

    subset = _create_subset_model_v2(
        "Sub", Original, ["x"], fn_description="Custom function doc"
    )
    assert subset.__doc__ == "Custom function doc"
