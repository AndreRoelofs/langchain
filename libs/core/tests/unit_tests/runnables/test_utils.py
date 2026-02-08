import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

import pytest

from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables.utils import (
    AddableDict,
    ConfigurableField,
    ConfigurableFieldMultiOption,
    ConfigurableFieldSingleOption,
    ConfigurableFieldSpec,
    _RootEventFilter,
    accepts_config,
    accepts_context,
    accepts_run_manager,
    add,
    gather_with_concurrency,
    get_function_first_arg_dict_keys,
    get_function_nonlocals,
    get_lambda_source,
    get_unique_config_specs,
    indent_lines_after_first,
    is_async_callable,
    is_async_generator,
)


@pytest.mark.parametrize(
    ("func", "expected_source"),
    [
        (lambda x: x * 2, "lambda x: x * 2"),
        (lambda a, b: a + b, "lambda a, b: a + b"),
        (lambda x: x if x > 0 else 0, "lambda x: x if x > 0 else 0"),  # noqa: FURB136
    ],
)
def test_get_lambda_source(func: Callable, expected_source: str) -> None:
    """Test get_lambda_source function."""
    source = get_lambda_source(func)
    assert source == expected_source


@pytest.mark.parametrize(
    ("text", "prefix", "expected_output"),
    [
        ("line 1\nline 2\nline 3", "1", "line 1\n line 2\n line 3"),
        ("line 1\nline 2\nline 3", "ax", "line 1\n  line 2\n  line 3"),
    ],
)
def test_indent_lines_after_first(text: str, prefix: str, expected_output: str) -> None:
    """Test indent_lines_after_first function."""
    indented_text = indent_lines_after_first(text, prefix)
    assert indented_text == expected_output


global_agent = RunnableLambda(lambda x: x * 3)


def test_nonlocals() -> None:
    agent = RunnableLambda(lambda x: x * 2)

    def my_func(value: str, agent: dict[str, str]) -> str:
        return agent.get("agent_name", value)

    def my_func2(value: str) -> str:
        return agent.get("agent_name", value)  # type: ignore[attr-defined]

    def my_func3(value: str) -> str:
        return agent.invoke(value)

    def my_func4(value: str) -> str:
        return global_agent.invoke(value)

    def my_func5() -> tuple[Callable[[str], str], RunnableLambda]:
        global_agent = RunnableLambda(lambda x: x * 3)

        def my_func6(value: str) -> str:
            return global_agent.invoke(value)

        return my_func6, global_agent

    assert get_function_nonlocals(my_func) == []
    assert get_function_nonlocals(my_func2) == []
    assert get_function_nonlocals(my_func3) == [agent.invoke]
    assert get_function_nonlocals(my_func4) == [global_agent.invoke]
    func, nl = my_func5()
    assert get_function_nonlocals(func) == [nl.invoke]
    assert RunnableLambda(my_func3).deps == [agent]
    assert RunnableLambda(my_func4).deps == [global_agent]
    assert RunnableLambda(func).deps == [nl]


# ---------------------------------------------------------------------------
# Tests for accepts_run_manager / accepts_config / accepts_context
# ---------------------------------------------------------------------------


def test_accepts_run_manager_true() -> None:
    def func(x: str, run_manager: Any = None) -> str:
        return x

    assert accepts_run_manager(func) is True


def test_accepts_run_manager_false() -> None:
    def func(x: str) -> str:
        return x

    assert accepts_run_manager(func) is False


def test_accepts_config_true() -> None:
    def func(x: str, config: Any = None) -> str:
        return x

    assert accepts_config(func) is True


def test_accepts_config_false() -> None:
    def func(x: str) -> str:
        return x

    assert accepts_config(func) is False


def test_accepts_context_true() -> None:
    def func(x: str, context: Any = None) -> str:
        return x

    assert accepts_context(func) is True


def test_accepts_context_false() -> None:
    def func(x: str) -> str:
        return x

    assert accepts_context(func) is False


def test_accepts_run_manager_builtin_returns_false() -> None:
    """Builtins (e.g. len) raise ValueError on signature(); should return False."""
    assert accepts_run_manager(len) is False


def test_accepts_config_builtin_returns_false() -> None:
    assert accepts_config(len) is False


def test_accepts_context_builtin_returns_false() -> None:
    assert accepts_context(len) is False


# ---------------------------------------------------------------------------
# Tests for is_async_generator
# ---------------------------------------------------------------------------


def test_is_async_generator_true() -> None:
    async def gen() -> AsyncIterator[int]:
        yield 1

    assert is_async_generator(gen) is True


def test_is_async_generator_false_sync() -> None:
    def gen():  # type: ignore[no-untyped-def]
        yield 1

    assert is_async_generator(gen) is False


def test_is_async_generator_false_regular_function() -> None:
    def func() -> int:
        return 1

    assert is_async_generator(func) is False


def test_is_async_generator_callable_object() -> None:
    class MyGen:
        async def __call__(self) -> AsyncIterator[int]:
            yield 1

    assert is_async_generator(MyGen()) is True


# ---------------------------------------------------------------------------
# Tests for is_async_callable
# ---------------------------------------------------------------------------


def test_is_async_callable_true() -> None:
    async def func() -> int:
        return 1

    assert is_async_callable(func) is True


def test_is_async_callable_false_sync() -> None:
    def func() -> int:
        return 1

    assert is_async_callable(func) is False


def test_is_async_callable_callable_object() -> None:
    class MyCallable:
        async def __call__(self) -> int:
            return 1

    assert is_async_callable(MyCallable()) is True


def test_is_async_callable_sync_callable_object() -> None:
    class MyCallable:
        def __call__(self) -> int:
            return 1

    assert is_async_callable(MyCallable()) is False


# ---------------------------------------------------------------------------
# Tests for AddableDict
# ---------------------------------------------------------------------------


def test_addable_dict_add_basic() -> None:
    a = AddableDict({"x": 1, "y": "hello"})
    b = AddableDict({"x": 2, "y": " world"})
    result = a + b
    assert result == {"x": 3, "y": "hello world"}
    assert isinstance(result, AddableDict)


def test_addable_dict_add_new_keys() -> None:
    a = AddableDict({"x": 1})
    b = AddableDict({"y": 2})
    result = a + b
    assert result == {"x": 1, "y": 2}


def test_addable_dict_add_none_values() -> None:
    """When one side is None, the other side's value is used."""
    a = AddableDict({"x": None})
    b = AddableDict({"x": 5})
    assert (a + b) == {"x": 5}

    a2 = AddableDict({"x": 5})
    b2 = AddableDict({"x": None})
    assert (a2 + b2) == {"x": 5}


def test_addable_dict_add_type_error_fallback() -> None:
    """When addition raises TypeError, the right side value wins."""
    a = AddableDict({"x": 1})
    b = AddableDict({"x": "string"})
    result = a + b
    assert result["x"] == "string"


def test_addable_dict_radd() -> None:
    """__radd__ supports adding a plain dict on the left."""
    a = {"x": 1}
    b = AddableDict({"x": 2, "y": 3})
    result = b.__radd__(a)
    assert result == {"x": 3, "y": 3}
    assert isinstance(result, AddableDict)


def test_addable_dict_radd_new_keys() -> None:
    a = {"a": 1}
    b = AddableDict({"b": 2})
    result = b.__radd__(a)
    assert result == {"a": 1, "b": 2}


def test_addable_dict_preserves_dict_behavior() -> None:
    """AddableDict still behaves like a regular dict for lookups."""
    d = AddableDict({"key": "value"})
    assert d["key"] == "value"
    assert list(d.keys()) == ["key"]
    assert len(d) == 1


# ---------------------------------------------------------------------------
# Tests for add()
# ---------------------------------------------------------------------------


def test_add_strings() -> None:
    result = add(["hello", " ", "world"])
    assert result == "hello world"


def test_add_lists() -> None:
    result = add([[1, 2], [3, 4], [5]])
    assert result == [1, 2, 3, 4, 5]


def test_add_integers() -> None:
    result = add([1, 2, 3])
    assert result == 6


def test_add_empty_iterable() -> None:
    result = add([])
    assert result is None


def test_add_single_element() -> None:
    result = add(["only"])
    assert result == "only"


def test_add_addable_dicts() -> None:
    result = add([AddableDict({"a": 1}), AddableDict({"a": 2, "b": 3})])
    assert result == {"a": 3, "b": 3}


# ---------------------------------------------------------------------------
# Tests for gather_with_concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gather_with_concurrency_none() -> None:
    """When n is None, all coroutines run without a semaphore."""

    async def identity(x: int) -> int:
        return x

    results = await gather_with_concurrency(None, identity(1), identity(2), identity(3))
    assert results == [1, 2, 3]


@pytest.mark.asyncio
async def test_gather_with_concurrency_limited() -> None:
    """When n is set, concurrency is limited."""
    running = 0
    max_running = 0

    async def track(x: int) -> int:
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        await asyncio.sleep(0.05)
        running -= 1
        return x

    results = await gather_with_concurrency(2, track(1), track(2), track(3), track(4))
    assert sorted(results) == [1, 2, 3, 4]
    assert max_running <= 2


@pytest.mark.asyncio
async def test_gather_with_concurrency_empty() -> None:
    """No coroutines returns an empty list."""
    results = await gather_with_concurrency(5)
    assert results == []


# ---------------------------------------------------------------------------
# Tests for get_function_first_arg_dict_keys
# ---------------------------------------------------------------------------


def test_get_function_first_arg_dict_keys_subscript() -> None:
    def func(data: dict) -> str:
        return data["name"] + data["age"]

    keys = get_function_first_arg_dict_keys(func)
    assert keys is not None
    assert sorted(keys) == ["age", "name"]


def test_get_function_first_arg_dict_keys_get() -> None:
    def func(data: dict) -> str:
        return data.get("name", "") + data.get("city", "")

    keys = get_function_first_arg_dict_keys(func)
    assert keys is not None
    assert sorted(keys) == ["city", "name"]


def test_get_function_first_arg_dict_keys_no_dict_access() -> None:
    def func(x: int) -> int:
        return x + 1

    keys = get_function_first_arg_dict_keys(func)
    assert keys is None


def test_get_function_first_arg_dict_keys_lambda() -> None:
    func = lambda data: data["key"]  # noqa: E731
    keys = get_function_first_arg_dict_keys(func)
    assert keys is not None
    assert keys == ["key"]


def test_get_function_first_arg_dict_keys_no_args() -> None:
    def func() -> None:
        pass

    keys = get_function_first_arg_dict_keys(func)
    assert keys is None


def test_get_function_first_arg_dict_keys_builtin_returns_none() -> None:
    """Builtins can't be inspected; should return None."""
    keys = get_function_first_arg_dict_keys(len)
    assert keys is None


# ---------------------------------------------------------------------------
# Tests for get_lambda_source (extended)
# ---------------------------------------------------------------------------


def test_get_lambda_source_named_function() -> None:
    """Named functions should return their name, not source."""

    def my_named_func(x: int) -> int:
        return x

    result = get_lambda_source(my_named_func)
    assert result == "my_named_func"


def test_get_lambda_source_no_name_attr() -> None:
    """Object without __name__ should return None."""

    class NoName:
        pass

    result = get_lambda_source(NoName())  # type: ignore[arg-type]
    assert result is None


# ---------------------------------------------------------------------------
# Tests for ConfigurableField and related NamedTuples
# ---------------------------------------------------------------------------


def test_configurable_field_defaults() -> None:
    field = ConfigurableField(id="test_id")
    assert field.id == "test_id"
    assert field.name is None
    assert field.description is None
    assert field.annotation is None
    assert field.is_shared is False


def test_configurable_field_with_values() -> None:
    field = ConfigurableField(
        id="temp",
        name="Temperature",
        description="The LLM temperature",
        annotation=float,
        is_shared=True,
    )
    assert field.id == "temp"
    assert field.name == "Temperature"
    assert field.annotation is float
    assert field.is_shared is True


def test_configurable_field_hash() -> None:
    f1 = ConfigurableField(id="a", annotation=int)
    f2 = ConfigurableField(id="a", annotation=int)
    f3 = ConfigurableField(id="b", annotation=int)
    assert hash(f1) == hash(f2)
    assert hash(f1) != hash(f3)


def test_configurable_field_single_option() -> None:
    field = ConfigurableFieldSingleOption(
        id="model",
        options={"gpt4": "gpt-4", "gpt3": "gpt-3.5"},
        default="gpt4",
    )
    assert field.id == "model"
    assert field.default == "gpt4"
    assert field.options["gpt4"] == "gpt-4"


def test_configurable_field_single_option_hash() -> None:
    f1 = ConfigurableFieldSingleOption(id="m", options={"a": 1, "b": 2}, default="a")
    f2 = ConfigurableFieldSingleOption(id="m", options={"a": 1, "b": 2}, default="a")
    assert hash(f1) == hash(f2)


def test_configurable_field_multi_option() -> None:
    field = ConfigurableFieldMultiOption(
        id="tools",
        options={"search": "web_search", "calc": "calculator"},
        default=["search"],
    )
    assert field.id == "tools"
    assert field.default == ["search"]
    assert len(field.options) == 2


def test_configurable_field_multi_option_hash() -> None:
    f1 = ConfigurableFieldMultiOption(id="t", options={"a": 1}, default=["a"])
    f2 = ConfigurableFieldMultiOption(id="t", options={"a": 1}, default=["a"])
    assert hash(f1) == hash(f2)


def test_configurable_field_spec_defaults() -> None:
    spec = ConfigurableFieldSpec(id="s", annotation=str)
    assert spec.id == "s"
    assert spec.annotation is str
    assert spec.name is None
    assert spec.description is None
    assert spec.default is None
    assert spec.is_shared is False
    assert spec.dependencies is None


def test_configurable_field_spec_with_dependencies() -> None:
    spec = ConfigurableFieldSpec(
        id="s",
        annotation=str,
        dependencies=["dep1", "dep2"],
    )
    assert spec.dependencies == ["dep1", "dep2"]


# ---------------------------------------------------------------------------
# Tests for get_unique_config_specs
# ---------------------------------------------------------------------------


def test_get_unique_config_specs_no_duplicates() -> None:
    specs = [
        ConfigurableFieldSpec(id="a", annotation=str),
        ConfigurableFieldSpec(id="b", annotation=int),
    ]
    result = get_unique_config_specs(specs)
    assert len(result) == 2


def test_get_unique_config_specs_identical_duplicates() -> None:
    spec = ConfigurableFieldSpec(id="a", annotation=str, default="x")
    result = get_unique_config_specs([spec, spec])
    assert len(result) == 1
    assert result[0].id == "a"


def test_get_unique_config_specs_conflicting_raises() -> None:
    s1 = ConfigurableFieldSpec(id="a", annotation=str, default="x")
    s2 = ConfigurableFieldSpec(id="a", annotation=int, default="y")
    with pytest.raises(ValueError, match="conflicting"):
        get_unique_config_specs([s1, s2])


def test_get_unique_config_specs_empty() -> None:
    result = get_unique_config_specs([])
    assert result == []


# ---------------------------------------------------------------------------
# Tests for _RootEventFilter
# ---------------------------------------------------------------------------


def test_root_event_filter_include_all_by_default() -> None:
    """When no include filters are set, all events are included."""
    f = _RootEventFilter()
    event: dict[str, Any] = {"name": "test", "tags": ["a"]}
    assert f.include_event(event, "chain") is True  # type: ignore[arg-type]


def test_root_event_filter_include_names() -> None:
    f = _RootEventFilter(include_names=["foo"])
    assert f.include_event({"name": "foo", "tags": []}, "chain") is True  # type: ignore[arg-type]
    assert f.include_event({"name": "bar", "tags": []}, "chain") is False  # type: ignore[arg-type]


def test_root_event_filter_include_types() -> None:
    f = _RootEventFilter(include_types=["llm"])
    assert f.include_event({"name": "x", "tags": []}, "llm") is True  # type: ignore[arg-type]
    assert f.include_event({"name": "x", "tags": []}, "chain") is False  # type: ignore[arg-type]


def test_root_event_filter_include_tags() -> None:
    f = _RootEventFilter(include_tags=["my_tag"])
    assert f.include_event({"name": "x", "tags": ["my_tag"]}, "chain") is True  # type: ignore[arg-type]
    assert f.include_event({"name": "x", "tags": ["other"]}, "chain") is False  # type: ignore[arg-type]


def test_root_event_filter_exclude_names() -> None:
    f = _RootEventFilter(exclude_names=["bad"])
    assert f.include_event({"name": "good", "tags": []}, "chain") is True  # type: ignore[arg-type]
    assert f.include_event({"name": "bad", "tags": []}, "chain") is False  # type: ignore[arg-type]


def test_root_event_filter_exclude_types() -> None:
    f = _RootEventFilter(exclude_types=["llm"])
    assert f.include_event({"name": "x", "tags": []}, "chain") is True  # type: ignore[arg-type]
    assert f.include_event({"name": "x", "tags": []}, "llm") is False  # type: ignore[arg-type]


def test_root_event_filter_exclude_tags() -> None:
    f = _RootEventFilter(exclude_tags=["secret"])
    assert f.include_event({"name": "x", "tags": ["public"]}, "chain") is True  # type: ignore[arg-type]
    assert f.include_event({"name": "x", "tags": ["secret"]}, "chain") is False  # type: ignore[arg-type]


def test_root_event_filter_include_and_exclude_combined() -> None:
    """Include by name but exclude by tag."""
    f = _RootEventFilter(include_names=["foo"], exclude_tags=["no"])
    assert f.include_event({"name": "foo", "tags": []}, "chain") is True  # type: ignore[arg-type]
    assert f.include_event({"name": "foo", "tags": ["no"]}, "chain") is False  # type: ignore[arg-type]
    assert f.include_event({"name": "bar", "tags": []}, "chain") is False  # type: ignore[arg-type]


def test_root_event_filter_no_tags_in_event() -> None:
    """Events without tags should still work."""
    f = _RootEventFilter(include_tags=["needed"])
    assert f.include_event({"name": "x"}, "chain") is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests for indent_lines_after_first (extended)
# ---------------------------------------------------------------------------


def test_indent_lines_after_first_single_line() -> None:
    """Single-line text should be returned as-is."""
    result = indent_lines_after_first("single line", "xx")
    assert result == "single line"


def test_indent_lines_after_first_empty_prefix() -> None:
    result = indent_lines_after_first("a\nb\nc", "")
    assert result == "a\nb\nc"


# ---------------------------------------------------------------------------
# Tests for get_lambda_source with multiple lambdas
# ---------------------------------------------------------------------------


def test_get_lambda_source_separate_lambdas() -> None:
    """Each lambda on its own line returns its own source."""
    a = lambda x: x + 1  # noqa: E731
    b = lambda x: x + 2  # noqa: E731
    # Each line has exactly one lambda, so source is returned for each
    assert get_lambda_source(a) == "lambda x: x + 1"
    assert get_lambda_source(b) == "lambda x: x + 2"
