from typing import Any

import pytest
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self, override

from langchain_core.runnables import (
    ConfigurableField,
    RunnableConfig,
    RunnableSerializable,
)


class MyRunnable(RunnableSerializable[str, str]):
    my_property: str = Field(alias="my_property_alias")
    _my_hidden_property: str = ""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def my_error(cls, values: dict[str, Any]) -> Any:
        if "_my_hidden_property" in values:
            msg = "Cannot set _my_hidden_property"
            raise ValueError(msg)
        return values

    @model_validator(mode="after")
    def build_extra(self) -> Self:
        self._my_hidden_property = self.my_property
        return self

    @override
    def invoke(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        return input + self._my_hidden_property

    def my_custom_function(self) -> str:
        return self.my_property

    def my_custom_function_w_config(
        self,
        config: RunnableConfig | None = None,  # noqa: ARG002
    ) -> str:
        return self.my_property

    def my_custom_function_w_kw_config(
        self,
        *,
        config: RunnableConfig | None = None,  # noqa: ARG002
    ) -> str:
        return self.my_property


class MyOtherRunnable(RunnableSerializable[str, str]):
    my_other_property: str

    @override
    def invoke(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        return input + self.my_other_property

    def my_other_custom_function(self) -> str:
        return self.my_other_property

    def my_other_custom_function_w_config(self, config: RunnableConfig) -> str:  # noqa: ARG002
        return self.my_other_property


def test_doubly_set_configurable() -> None:
    """Test that setting a configurable field with a default value works."""
    runnable = MyRunnable(my_property="a")
    configurable_runnable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="The property to test",
        )
    )

    assert configurable_runnable.invoke("d", config={"my_property": "c"}) == "dc"  # type: ignore[arg-type]


def test_alias_set_configurable() -> None:
    runnable = MyRunnable(my_property="a")
    configurable_runnable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property_alias",
            name="My property alias",
            description="The property to test alias",
        )
    )

    assert (
        configurable_runnable.invoke(
            "d", config=RunnableConfig(configurable={"my_property_alias": "c"})
        )
        == "dc"
    )


def test_field_alias_set_configurable() -> None:
    runnable = MyRunnable(my_property_alias="a")  # type: ignore[call-arg]
    configurable_runnable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property alias",
            description="The property to test alias",
        )
    )

    assert (
        configurable_runnable.invoke(
            "d", config=RunnableConfig(configurable={"my_property": "c"})
        )
        == "dc"
    )


def test_config_passthrough() -> None:
    runnable = MyRunnable(my_property="a")
    configurable_runnable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="The property to test",
        )
    )
    # first one
    with pytest.raises(AttributeError):
        configurable_runnable.not_my_custom_function()  # type: ignore[attr-defined]

    assert configurable_runnable.my_custom_function() == "a"  # type: ignore[attr-defined]
    assert (
        configurable_runnable.my_custom_function_w_config(  # type: ignore[attr-defined]
            {"configurable": {"my_property": "b"}}
        )
        == "b"
    )
    assert (
        configurable_runnable.my_custom_function_w_config(  # type: ignore[attr-defined]
            config={"configurable": {"my_property": "b"}}
        )
        == "b"
    )

    # second one
    assert (
        configurable_runnable.with_config(
            configurable={"my_property": "b"}
        ).my_custom_function()  # type: ignore[attr-defined]
        == "b"
    )


def test_config_passthrough_nested() -> None:
    runnable = MyRunnable(my_property="a")
    configurable_runnable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="The property to test",
        )
    ).configurable_alternatives(
        ConfigurableField(id="which", description="Which runnable to use"),
        other=MyOtherRunnable(my_other_property="c"),
    )
    # first one
    with pytest.raises(AttributeError):
        configurable_runnable.not_my_custom_function()  # type: ignore[attr-defined]
    assert configurable_runnable.my_custom_function() == "a"  # type: ignore[attr-defined]
    assert (
        configurable_runnable.my_custom_function_w_config(  # type: ignore[attr-defined]
            {"configurable": {"my_property": "b"}}
        )
        == "b"
    )
    assert (
        configurable_runnable.my_custom_function_w_config(  # type: ignore[attr-defined]
            config={"configurable": {"my_property": "b"}}
        )
        == "b"
    )
    assert (
        configurable_runnable.with_config(
            configurable={"my_property": "b"}
        ).my_custom_function()  # type: ignore[attr-defined]
        == "b"
    ), "function without config can be called w bound config"
    assert (
        configurable_runnable.with_config(
            configurable={"my_property": "b"}
        ).my_custom_function_w_config(  # type: ignore[attr-defined]
        )
        == "b"
    ), "func with config arg can be called w bound config without config"
    assert (
        configurable_runnable.with_config(
            configurable={"my_property": "b"}
        ).my_custom_function_w_config(  # type: ignore[attr-defined]
            config={"configurable": {"my_property": "c"}}
        )
        == "c"
    ), "func with config arg can be called w bound config with config as kwarg"
    assert (
        configurable_runnable.with_config(
            configurable={"my_property": "b"}
        ).my_custom_function_w_kw_config(  # type: ignore[attr-defined]
        )
        == "b"
    ), "function with config kwarg can be called w bound config w/out config"
    assert (
        configurable_runnable.with_config(
            configurable={"my_property": "b"}
        ).my_custom_function_w_kw_config(  # type: ignore[attr-defined]
            config={"configurable": {"my_property": "c"}}
        )
        == "c"
    ), "function with config kwarg can be called w bound config with config"
    assert (
        configurable_runnable.with_config(configurable={"my_property": "b"})
        .with_types()
        .my_custom_function()  # type: ignore[attr-defined]
        == "b"
    ), "function without config can be called w bound config"
    assert (
        configurable_runnable.with_config(configurable={"my_property": "b"})
        .with_types()
        .my_custom_function_w_config(  # type: ignore[attr-defined]
        )
        == "b"
    ), "func with config arg can be called w bound config without config"
    assert (
        configurable_runnable.with_config(configurable={"my_property": "b"})
        .with_types()
        .my_custom_function_w_config(  # type: ignore[attr-defined]
            config={"configurable": {"my_property": "c"}}
        )
        == "c"
    ), "func with config arg can be called w bound config with config as kwarg"
    assert (
        configurable_runnable.with_config(configurable={"my_property": "b"})
        .with_types()
        .my_custom_function_w_kw_config(  # type: ignore[attr-defined]
        )
        == "b"
    ), "function with config kwarg can be called w bound config w/out config"
    assert (
        configurable_runnable.with_config(configurable={"my_property": "b"})
        .with_types()
        .my_custom_function_w_kw_config(  # type: ignore[attr-defined]
            config={"configurable": {"my_property": "c"}}
        )
        == "c"
    ), "function with config kwarg can be called w bound config with config"
    # second one
    with pytest.raises(AttributeError):
        configurable_runnable.my_other_custom_function()  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        configurable_runnable.my_other_custom_function_w_config(  # type: ignore[attr-defined]
            {"configurable": {"my_other_property": "b"}}
        )
    with pytest.raises(AttributeError):
        configurable_runnable.with_config(
            configurable={"my_other_property": "c", "which": "other"}
        ).my_other_custom_function()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests for _strremoveprefix
# ---------------------------------------------------------------------------
from langchain_core.runnables.configurable import _strremoveprefix


def test_strremoveprefix_with_prefix() -> None:
    assert _strremoveprefix("model==gpt4/temperature", "model==gpt4/") == "temperature"


def test_strremoveprefix_without_prefix() -> None:
    assert _strremoveprefix("temperature", "model==gpt4/") == "temperature"


def test_strremoveprefix_empty_prefix() -> None:
    assert _strremoveprefix("hello", "") == "hello"


def test_strremoveprefix_empty_string() -> None:
    assert _strremoveprefix("", "prefix") == ""


def test_strremoveprefix_exact_match() -> None:
    assert _strremoveprefix("prefix", "prefix") == ""


# ---------------------------------------------------------------------------
# Tests for prefix_config_spec
# ---------------------------------------------------------------------------
from langchain_core.runnables.configurable import prefix_config_spec
from langchain_core.runnables.utils import ConfigurableFieldSpec


def test_prefix_config_spec_non_shared() -> None:
    spec = ConfigurableFieldSpec(
        id="temperature",
        annotation=float,
        name="Temperature",
        description="LLM temp",
        default=0.7,
        is_shared=False,
    )
    result = prefix_config_spec(spec, "model==gpt4")
    assert result.id == "model==gpt4/temperature"
    assert result.name == "Temperature"
    assert result.description == "LLM temp"
    assert result.default == 0.7
    assert result.is_shared is False


def test_prefix_config_spec_shared_unchanged() -> None:
    """Shared specs should NOT be prefixed."""
    spec = ConfigurableFieldSpec(
        id="global_setting",
        annotation=str,
        is_shared=True,
    )
    result = prefix_config_spec(spec, "model==gpt4")
    assert result.id == "global_setting"
    assert result is spec


# ---------------------------------------------------------------------------
# Tests for make_options_spec
# ---------------------------------------------------------------------------
from langchain_core.runnables.configurable import make_options_spec
from langchain_core.runnables.utils import (
    ConfigurableFieldMultiOption,
    ConfigurableFieldSingleOption,
)


def test_make_options_spec_single_option() -> None:
    spec = ConfigurableFieldSingleOption(
        id="model",
        options={"gpt4": "gpt-4", "gpt3": "gpt-3.5"},
        default="gpt4",
        name="Model",
        description="Which model",
    )
    result = make_options_spec(spec, "fallback desc")
    assert result.id == "model"
    assert result.default == "gpt4"
    assert result.description == "Which model"
    # annotation should be an enum
    assert result.annotation is not None


def test_make_options_spec_single_option_uses_fallback_description() -> None:
    spec = ConfigurableFieldSingleOption(
        id="model",
        options={"gpt4": "gpt-4"},
        default="gpt4",
    )
    result = make_options_spec(spec, "fallback desc")
    assert result.description == "fallback desc"


def test_make_options_spec_multi_option() -> None:
    spec = ConfigurableFieldMultiOption(
        id="tools",
        options={"search": "web_search", "calc": "calculator"},
        default=["search"],
        name="Tools",
    )
    result = make_options_spec(spec, "fallback desc")
    assert result.id == "tools"
    assert result.default == ["search"]


# ---------------------------------------------------------------------------
# Tests for RunnableConfigurableAlternatives
# ---------------------------------------------------------------------------
from langchain_core.runnables.configurable import RunnableConfigurableAlternatives


def test_configurable_alternatives_invoke_default() -> None:
    """When no alternative is specified, the default runnable is used."""
    default = MyRunnable(my_property="default_val")
    alt = MyOtherRunnable(my_other_property="alt_val")
    configurable = RunnableConfigurableAlternatives(
        which=ConfigurableField(id="which"),
        default=default,
        alternatives={"other": alt},
        default_key="default",
        prefix_keys=False,
    )
    result = configurable.invoke("input_")
    assert result == "input_default_val"


def test_configurable_alternatives_invoke_alternative() -> None:
    """When alternative is specified via configurable, it's used."""
    default = MyRunnable(my_property="default_val")
    alt = MyOtherRunnable(my_other_property="alt_val")
    configurable = RunnableConfigurableAlternatives(
        which=ConfigurableField(id="which"),
        default=default,
        alternatives={"other": alt},
        default_key="default",
        prefix_keys=False,
    )
    result = configurable.invoke(
        "input_",
        config=RunnableConfig(configurable={"which": "other"}),
    )
    assert result == "input_alt_val"


def test_configurable_alternatives_unknown_raises() -> None:
    """Unknown alternative key raises ValueError."""
    default = MyRunnable(my_property="default_val")
    configurable = RunnableConfigurableAlternatives(
        which=ConfigurableField(id="which"),
        default=default,
        alternatives={},
        default_key="default",
        prefix_keys=False,
    )
    with pytest.raises(ValueError, match="Unknown alternative"):
        configurable.invoke(
            "input_",
            config=RunnableConfig(configurable={"which": "nonexistent"}),
        )


def test_configurable_alternatives_with_callable_factory() -> None:
    """Alternatives can be callables (factories) instead of Runnable instances."""
    default = MyRunnable(my_property="default_val")

    def make_alt() -> MyOtherRunnable:
        return MyOtherRunnable(my_other_property="factory_val")

    configurable = RunnableConfigurableAlternatives(
        which=ConfigurableField(id="which"),
        default=default,
        alternatives={"other": make_alt},
        default_key="default",
        prefix_keys=False,
    )
    result = configurable.invoke(
        "input_",
        config=RunnableConfig(configurable={"which": "other"}),
    )
    assert result == "input_factory_val"


def test_configurable_alternatives_config_specs() -> None:
    """config_specs should include the 'which' field spec."""
    default = MyRunnable(my_property="a")
    alt = MyOtherRunnable(my_other_property="b")
    configurable = RunnableConfigurableAlternatives(
        which=ConfigurableField(id="which"),
        default=default,
        alternatives={"other": alt},
        default_key="default",
        prefix_keys=False,
    )
    specs = configurable.config_specs
    spec_ids = [s.id for s in specs]
    assert "which" in spec_ids


def test_configurable_alternatives_with_prefix_keys() -> None:
    """prefix_keys renames configurable keys per alternative."""
    default = MyRunnable(my_property="a")
    default_configurable = default.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="test",
        )
    )
    alt = MyOtherRunnable(my_other_property="b")
    configurable = RunnableConfigurableAlternatives(
        which=ConfigurableField(id="which"),
        default=default_configurable,
        alternatives={"other": alt},
        default_key="default",
        prefix_keys=True,
    )
    specs = configurable.config_specs
    spec_ids = [s.id for s in specs]
    assert "which" in spec_ids
    # The default's config spec should be prefixed
    assert any("which==default" in sid for sid in spec_ids)


# ---------------------------------------------------------------------------
# Tests for DynamicRunnable
# ---------------------------------------------------------------------------


def test_dynamic_runnable_with_config() -> None:
    """DynamicRunnable.with_config produces a new DynamicRunnable with merged config."""
    runnable = MyRunnable(my_property="a")
    configurable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="test",
        )
    )
    new = configurable.with_config(tags=["test_tag"])
    assert new.config is not None
    assert "test_tag" in new.config.get("tags", [])


def test_dynamic_runnable_input_output_types() -> None:
    """InputType and OutputType are delegated to the default runnable."""
    runnable = MyRunnable(my_property="a")
    configurable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="test",
        )
    )
    assert configurable.InputType == runnable.InputType
    assert configurable.OutputType == runnable.OutputType


# ---------------------------------------------------------------------------
# Tests for RunnableConfigurableFields
# ---------------------------------------------------------------------------


def test_configurable_fields_config_specs() -> None:
    """config_specs includes spec for each configurable field."""
    runnable = MyRunnable(my_property="a")
    configurable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="test desc",
        )
    )
    specs = configurable.config_specs
    spec_ids = [s.id for s in specs]
    assert "my_property" in spec_ids
    spec = next(s for s in specs if s.id == "my_property")
    assert spec.name == "My property"
    assert spec.description == "test desc"


def test_configurable_fields_prepare_no_config() -> None:
    """prepare() with no configurable overrides returns the default."""
    runnable = MyRunnable(my_property="a")
    configurable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="test",
        )
    )
    prepared, config = configurable.prepare()
    assert prepared.invoke("x") == "xa"


def test_configurable_fields_prepare_with_override() -> None:
    """prepare() with configurable overrides replaces the field."""
    runnable = MyRunnable(my_property="a")
    configurable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="test",
        )
    )
    prepared, config = configurable.prepare(
        RunnableConfig(configurable={"my_property": "b"})
    )
    assert prepared.invoke("x") == "xb"


def test_configurable_fields_batch() -> None:
    """batch with configurable fields works."""
    runnable = MyRunnable(my_property="a")
    configurable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="test",
        )
    )
    results = configurable.batch(
        ["x", "y"],
        config=[
            RunnableConfig(configurable={"my_property": "1"}),
            RunnableConfig(configurable={"my_property": "2"}),
        ],
    )
    assert results == ["x1", "y2"]


def test_configurable_fields_stream() -> None:
    """stream with configurable fields works."""
    runnable = MyRunnable(my_property="a")
    configurable = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="test",
        )
    )
    chunks = list(
        configurable.stream(
            "x", config=RunnableConfig(configurable={"my_property": "b"})
        )
    )
    assert chunks == ["xb"]


def test_configurable_fields_chained_configurable_fields() -> None:
    """Calling configurable_fields on a RunnableConfigurableFields merges fields."""
    runnable = MyRunnable(my_property="a")
    c1 = runnable.configurable_fields(
        my_property=ConfigurableField(
            id="my_property",
            name="My property",
            description="test",
        )
    )
    # Calling configurable_fields again should still work
    c2 = c1.configurable_fields()
    result = c2.invoke("x", config=RunnableConfig(configurable={"my_property": "c"}))
    assert result == "xc"
