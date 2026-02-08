import json
import uuid
from contextvars import copy_context
from typing import Any, cast

import pytest

from langchain_core.callbacks.manager import (
    AsyncCallbackManager,
    CallbackManager,
    atrace_as_chain_group,
    trace_as_chain_group,
)
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnableBinding, RunnablePassthrough
from langchain_core.runnables.config import (
    RunnableConfig,
    _set_config_context,
    ensure_config,
    merge_configs,
    run_in_executor,
)
from langchain_core.tracers.stdout import ConsoleCallbackHandler


def test_ensure_config() -> None:
    run_id = str(uuid.uuid4())
    arg: dict = {
        "something": "else",
        "metadata": {"foo": "bar"},
        "configurable": {"baz": "qux"},
        "callbacks": [StdOutCallbackHandler()],
        "tags": ["tag1", "tag2"],
        "max_concurrency": 1,
        "recursion_limit": 100,
        "run_id": run_id,
        "run_name": "test",
    }
    arg_str = json.dumps({**arg, "callbacks": []})
    ctx = copy_context()
    ctx.run(
        _set_config_context,
        {
            "callbacks": [ConsoleCallbackHandler()],
            "metadata": {"a": "b"},
            "configurable": {"c": "d"},
            "tags": ["tag3", "tag4"],
        },
    )
    config = ctx.run(ensure_config, cast("RunnableConfig", arg))
    assert len(arg["callbacks"]) == 1, (
        "ensure_config should not modify the original config"
    )
    assert json.dumps({**arg, "callbacks": []}) == arg_str, (
        "ensure_config should not modify the original config"
    )
    assert config is not arg
    assert config["callbacks"] is not arg["callbacks"]
    assert config["metadata"] is not arg["metadata"]
    assert config["configurable"] is not arg["configurable"]
    assert config == {
        "tags": ["tag1", "tag2"],
        "metadata": {"foo": "bar", "baz": "qux", "something": "else"},
        "callbacks": [arg["callbacks"][0]],
        "recursion_limit": 100,
        "configurable": {"baz": "qux", "something": "else"},
        "max_concurrency": 1,
        "run_id": run_id,
        "run_name": "test",
    }


async def test_merge_config_callbacks() -> None:
    manager: RunnableConfig = {
        "callbacks": CallbackManager(handlers=[StdOutCallbackHandler()])
    }
    handlers: RunnableConfig = {"callbacks": [ConsoleCallbackHandler()]}
    other_handlers: RunnableConfig = {"callbacks": [StreamingStdOutCallbackHandler()]}

    merged = merge_configs(manager, handlers)["callbacks"]

    assert isinstance(merged, CallbackManager)
    assert len(merged.handlers) == 2
    assert isinstance(merged.handlers[0], StdOutCallbackHandler)
    assert isinstance(merged.handlers[1], ConsoleCallbackHandler)

    merged = merge_configs(handlers, manager)["callbacks"]

    assert isinstance(merged, CallbackManager)
    assert len(merged.handlers) == 2
    assert isinstance(merged.handlers[0], StdOutCallbackHandler)
    assert isinstance(merged.handlers[1], ConsoleCallbackHandler)

    merged = merge_configs(handlers, other_handlers)["callbacks"]

    assert isinstance(merged, list)
    assert len(merged) == 2
    assert isinstance(merged[0], ConsoleCallbackHandler)
    assert isinstance(merged[1], StreamingStdOutCallbackHandler)

    # Check that the original object wasn't mutated
    merged = merge_configs(manager, handlers)["callbacks"]

    assert isinstance(merged, CallbackManager)
    assert len(merged.handlers) == 2
    assert isinstance(merged.handlers[0], StdOutCallbackHandler)
    assert isinstance(merged.handlers[1], ConsoleCallbackHandler)

    with trace_as_chain_group("test") as gm:
        group_manager: RunnableConfig = {
            "callbacks": gm,
        }
        merged = merge_configs(group_manager, handlers)["callbacks"]
        assert isinstance(merged, CallbackManager)
        assert len(merged.handlers) == 1
        assert isinstance(merged.handlers[0], ConsoleCallbackHandler)

        merged = merge_configs(handlers, group_manager)["callbacks"]
        assert isinstance(merged, CallbackManager)
        assert len(merged.handlers) == 1
        assert isinstance(merged.handlers[0], ConsoleCallbackHandler)
        merged = merge_configs(group_manager, manager)["callbacks"]
        assert isinstance(merged, CallbackManager)
        assert len(merged.handlers) == 1
        assert isinstance(merged.handlers[0], StdOutCallbackHandler)

    async with atrace_as_chain_group("test_async") as gm:
        group_manager = {
            "callbacks": gm,
        }
        merged = merge_configs(group_manager, handlers)["callbacks"]
        assert isinstance(merged, AsyncCallbackManager)
        assert len(merged.handlers) == 1
        assert isinstance(merged.handlers[0], ConsoleCallbackHandler)

        merged = merge_configs(handlers, group_manager)["callbacks"]
        assert isinstance(merged, AsyncCallbackManager)
        assert len(merged.handlers) == 1
        assert isinstance(merged.handlers[0], ConsoleCallbackHandler)
        merged = merge_configs(group_manager, manager)["callbacks"]
        assert isinstance(merged, AsyncCallbackManager)
        assert len(merged.handlers) == 1
        assert isinstance(merged.handlers[0], StdOutCallbackHandler)


def test_config_arbitrary_keys() -> None:
    base: RunnablePassthrough[Any] = RunnablePassthrough()
    bound = base.with_config(my_custom_key="my custom value")
    config = cast("RunnableBinding", bound).config

    assert config.get("my_custom_key") == "my custom value"


async def test_run_in_executor() -> None:
    def raises_stop_iter() -> Any:
        return next(iter([]))

    with pytest.raises(StopIteration):
        raises_stop_iter()

    with pytest.raises(RuntimeError):
        await run_in_executor(None, raises_stop_iter)


# ---------------------------------------------------------------------------
# Tests for ensure_config
# ---------------------------------------------------------------------------


def test_ensure_config_none_returns_defaults() -> None:
    """ensure_config(None) returns a config with all default keys."""
    config = ensure_config(None)
    assert config["tags"] == []
    assert config["metadata"] == {}
    assert config["callbacks"] is None
    assert config["recursion_limit"] == 25
    assert config["configurable"] == {}


def test_ensure_config_empty_dict_returns_defaults() -> None:
    """ensure_config({}) behaves like ensure_config(None)."""
    config = ensure_config(cast("RunnableConfig", {}))
    assert config["tags"] == []
    assert config["metadata"] == {}
    assert config["callbacks"] is None
    assert config["recursion_limit"] == 25
    assert config["configurable"] == {}


def test_ensure_config_non_config_keys_go_to_configurable() -> None:
    """Non-standard keys are moved into the 'configurable' sub-dict."""
    config = ensure_config(cast("RunnableConfig", {"my_key": "my_val"}))
    assert config["configurable"]["my_key"] == "my_val"


def test_ensure_config_none_values_are_ignored() -> None:
    """Keys with None values in the input are not propagated."""
    config = ensure_config(cast("RunnableConfig", {"tags": None, "metadata": None}))
    assert config["tags"] == []
    assert config["metadata"] == {}


def test_ensure_config_copies_tags_metadata_configurable() -> None:
    """Copiable keys are shallow-copied so mutations don't affect the original."""
    original_tags = ["a", "b"]
    original_metadata = {"k": "v"}
    original_configurable = {"x": "y"}
    arg: RunnableConfig = {
        "tags": original_tags,
        "metadata": original_metadata,
        "configurable": original_configurable,
    }
    config = ensure_config(arg)
    assert config["tags"] is not original_tags
    assert config["metadata"] is not original_metadata
    assert config["configurable"] is not original_configurable


def test_ensure_config_configurable_values_added_to_metadata() -> None:
    """Simple configurable values (str, int, float, bool) are mirrored to metadata."""
    config = ensure_config(
        cast(
            "RunnableConfig",
            {"configurable": {"model_name": "gpt-4", "temperature": 0.5}},
        )
    )
    assert config["metadata"]["model_name"] == "gpt-4"
    assert config["metadata"]["temperature"] == 0.5


def test_ensure_config_api_key_not_added_to_metadata() -> None:
    """The special key 'api_key' should NOT be mirrored to metadata."""
    config = ensure_config(
        cast("RunnableConfig", {"configurable": {"api_key": "secret123"}})
    )
    assert "api_key" not in config["metadata"]


def test_ensure_config_dunder_keys_not_added_to_metadata() -> None:
    """Keys starting with '__' should NOT be mirrored to metadata."""
    config = ensure_config(
        cast("RunnableConfig", {"configurable": {"__internal": "hidden"}})
    )
    assert "__internal" not in config["metadata"]


# ---------------------------------------------------------------------------
# Tests for get_config_list
# ---------------------------------------------------------------------------
from langchain_core.runnables.config import get_config_list


def test_get_config_list_single_config_replicated() -> None:
    """A single config should be replicated `length` times."""
    configs = get_config_list({"tags": ["a"]}, 3)
    assert len(configs) == 3
    for c in configs:
        assert "tags" in c
        assert "recursion_limit" in c


def test_get_config_list_none_config() -> None:
    """None config produces `length` default configs."""
    configs = get_config_list(None, 2)
    assert len(configs) == 2
    for c in configs:
        assert c["tags"] == []


def test_get_config_list_sequence_of_configs() -> None:
    """A sequence of configs is returned (each ensured) when lengths match."""
    configs = get_config_list(
        [{"tags": ["a"]}, {"tags": ["b"]}],
        2,
    )
    assert len(configs) == 2
    assert configs[0]["tags"] == ["a"]
    assert configs[1]["tags"] == ["b"]


def test_get_config_list_sequence_length_mismatch_raises() -> None:
    """ValueError when the config list length doesn't match the input length."""
    with pytest.raises(ValueError, match="same length"):
        get_config_list([{"tags": ["a"]}], 3)


def test_get_config_list_negative_length_raises() -> None:
    """ValueError when length is negative."""
    with pytest.raises(ValueError, match="length must be >= 0"):
        get_config_list(None, -1)


def test_get_config_list_zero_length() -> None:
    """A length of 0 returns an empty list."""
    configs = get_config_list(None, 0)
    assert configs == []


def test_get_config_list_run_id_warning() -> None:
    """When a single config with run_id is replicated, a warning is issued."""
    run_id = uuid.uuid4()
    with pytest.warns(RuntimeWarning, match="run_id"):
        configs = get_config_list({"run_id": run_id}, 3)
    assert configs[0].get("run_id") == run_id
    assert configs[1].get("run_id") is None
    assert configs[2].get("run_id") is None


def test_get_config_list_run_id_single_no_warning() -> None:
    """When length is 1, no warning is issued even with run_id."""
    run_id = uuid.uuid4()
    configs = get_config_list({"run_id": run_id}, 1)
    assert len(configs) == 1
    assert configs[0].get("run_id") == run_id


# ---------------------------------------------------------------------------
# Tests for patch_config
# ---------------------------------------------------------------------------
from langchain_core.runnables.config import patch_config


def test_patch_config_none_input() -> None:
    """patch_config(None) returns a default config."""
    config = patch_config(None)
    assert config["tags"] == []
    assert config["recursion_limit"] == 25


def test_patch_config_sets_recursion_limit() -> None:
    config = patch_config({"recursion_limit": 10}, recursion_limit=50)
    assert config["recursion_limit"] == 50


def test_patch_config_sets_max_concurrency() -> None:
    config = patch_config(None, max_concurrency=5)
    assert config["max_concurrency"] == 5


def test_patch_config_sets_run_name() -> None:
    config = patch_config(None, run_name="my_run")
    assert config["run_name"] == "my_run"


def test_patch_config_sets_configurable_merges() -> None:
    """Configurable dict is merged, not replaced."""
    config = patch_config(
        cast("RunnableConfig", {"configurable": {"a": 1}}),
        configurable={"b": 2},
    )
    assert config["configurable"]["a"] == 1
    assert config["configurable"]["b"] == 2


def test_patch_config_callbacks_clears_run_name_and_run_id() -> None:
    """Setting callbacks should remove run_name and run_id from the config."""
    callback_mgr = CallbackManager(handlers=[])
    config = patch_config(
        {"run_name": "old_name", "run_id": uuid.uuid4()},
        callbacks=callback_mgr,
    )
    assert "run_name" not in config
    assert "run_id" not in config
    assert config["callbacks"] is callback_mgr


# ---------------------------------------------------------------------------
# Tests for merge_configs (non-callback scenarios)
# ---------------------------------------------------------------------------


def test_merge_configs_tags_are_deduplicated_and_sorted() -> None:
    merged = merge_configs({"tags": ["b", "a"]}, {"tags": ["a", "c"]})
    assert merged["tags"] == ["a", "b", "c"]


def test_merge_configs_metadata_is_merged() -> None:
    merged = merge_configs({"metadata": {"a": 1}}, {"metadata": {"b": 2}})
    assert merged["metadata"] == {"a": 1, "b": 2}


def test_merge_configs_metadata_later_overrides() -> None:
    merged = merge_configs({"metadata": {"a": 1}}, {"metadata": {"a": 2}})
    assert merged["metadata"]["a"] == 2


def test_merge_configs_configurable_is_merged() -> None:
    merged = merge_configs(
        {"configurable": {"a": 1}},
        {"configurable": {"b": 2}},
    )
    assert merged["configurable"] == {"a": 1, "b": 2}


def test_merge_configs_recursion_limit_non_default_wins() -> None:
    """A non-default recursion_limit from a later config overrides the earlier one."""
    merged = merge_configs(
        {"recursion_limit": 10},
        {"recursion_limit": 50},
    )
    assert merged["recursion_limit"] == 50


def test_merge_configs_recursion_limit_default_does_not_override() -> None:
    """The default recursion_limit (25) does NOT override a custom one."""
    merged = merge_configs(
        {"recursion_limit": 50},
        {"recursion_limit": 25},
    )
    assert merged["recursion_limit"] == 50


def test_merge_configs_none_configs_skipped() -> None:
    """None entries in merge_configs are safely skipped."""
    merged = merge_configs(None, {"tags": ["a"]}, None, {"tags": ["b"]})
    assert merged["tags"] == ["a", "b"]


def test_merge_configs_run_name_and_run_id() -> None:
    run_id = uuid.uuid4()
    merged = merge_configs(
        {"run_name": "first"},
        {"run_name": "second", "run_id": run_id},
    )
    assert merged["run_name"] == "second"
    assert merged["run_id"] == run_id


def test_merge_configs_empty() -> None:
    """merge_configs with no arguments returns base defaults."""
    merged = merge_configs()
    assert merged == {}


# ---------------------------------------------------------------------------
# Tests for call_func_with_variable_args / acall_func_with_variable_args
# ---------------------------------------------------------------------------
from langchain_core.runnables.config import (
    acall_func_with_variable_args,
    call_func_with_variable_args,
)


def test_call_func_with_variable_args_simple() -> None:
    """Function that only takes input."""

    def func(x: str) -> str:
        return x.upper()

    result = call_func_with_variable_args(func, "hello", ensure_config(None))
    assert result == "HELLO"


def test_call_func_with_variable_args_with_config() -> None:
    """Function that takes input and config."""

    def func(x: str, config: RunnableConfig) -> str:
        return x + str(config.get("recursion_limit"))

    result = call_func_with_variable_args(func, "val", ensure_config(None))
    assert result == "val25"


async def test_acall_func_with_variable_args_simple() -> None:
    """Async function that only takes input."""

    async def func(x: str) -> str:
        return x.upper()

    result = await acall_func_with_variable_args(func, "hello", ensure_config(None))
    assert result == "HELLO"


async def test_acall_func_with_variable_args_with_config() -> None:
    """Async function that takes input and config."""

    async def func(x: str, config: RunnableConfig) -> str:
        return x + str(config.get("recursion_limit"))

    result = await acall_func_with_variable_args(func, "val", ensure_config(None))
    assert result == "val25"


# ---------------------------------------------------------------------------
# Tests for get_callback_manager_for_config / get_async_callback_manager_for_config
# ---------------------------------------------------------------------------
from langchain_core.runnables.config import (
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
)


def test_get_callback_manager_for_config_basic() -> None:
    config = ensure_config(None)
    mgr = get_callback_manager_for_config(config)
    assert isinstance(mgr, CallbackManager)


def test_get_callback_manager_for_config_with_tags_and_metadata() -> None:
    config = ensure_config({"tags": ["a"], "metadata": {"k": "v"}})
    mgr = get_callback_manager_for_config(config)
    assert isinstance(mgr, CallbackManager)
    assert "a" in mgr.inheritable_tags
    assert mgr.inheritable_metadata.get("k") == "v"


def test_get_async_callback_manager_for_config_basic() -> None:
    config = ensure_config(None)
    mgr = get_async_callback_manager_for_config(config)
    assert isinstance(mgr, AsyncCallbackManager)


def test_get_async_callback_manager_for_config_with_tags_and_metadata() -> None:
    config = ensure_config({"tags": ["a"], "metadata": {"k": "v"}})
    mgr = get_async_callback_manager_for_config(config)
    assert isinstance(mgr, AsyncCallbackManager)
    assert "a" in mgr.inheritable_tags
    assert mgr.inheritable_metadata.get("k") == "v"


# ---------------------------------------------------------------------------
# Tests for ContextThreadPoolExecutor
# ---------------------------------------------------------------------------
from concurrent.futures import Future
from contextvars import ContextVar

from langchain_core.runnables.config import ContextThreadPoolExecutor


def test_context_thread_pool_executor_submit_copies_context() -> None:
    """submit() should copy the current context into the child thread."""
    var: ContextVar[str] = ContextVar("test_var", default="unset")
    var.set("parent_value")

    with ContextThreadPoolExecutor(max_workers=1) as executor:
        future: Future[str] = executor.submit(var.get)
        result = future.result(timeout=5)

    assert result == "parent_value"


def test_context_thread_pool_executor_map_copies_context() -> None:
    """map() should copy context for each item."""
    var: ContextVar[str] = ContextVar("test_var2", default="unset")
    var.set("mapped_value")

    def read_var(_: int) -> str:
        return var.get()

    with ContextThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(read_var, [1, 2, 3]))

    assert results == ["mapped_value", "mapped_value", "mapped_value"]


# ---------------------------------------------------------------------------
# Tests for get_executor_for_config
# ---------------------------------------------------------------------------
from langchain_core.runnables.config import get_executor_for_config


def test_get_executor_for_config_none() -> None:
    """get_executor_for_config(None) yields a ContextThreadPoolExecutor."""
    with get_executor_for_config(None) as executor:
        assert isinstance(executor, ContextThreadPoolExecutor)
        future = executor.submit(lambda: 42)
        assert future.result(timeout=5) == 42


def test_get_executor_for_config_with_max_concurrency() -> None:
    """max_concurrency from config is respected."""
    with get_executor_for_config({"max_concurrency": 2}) as executor:
        assert isinstance(executor, ContextThreadPoolExecutor)
        assert executor._max_workers == 2


# ---------------------------------------------------------------------------
# Tests for run_in_executor
# ---------------------------------------------------------------------------


async def test_run_in_executor_with_none_executor() -> None:
    """Normal function runs to completion when executor is None."""

    def add(a: int, b: int) -> int:
        return a + b

    result = await run_in_executor(None, add, 3, 7)
    assert result == 10


async def test_run_in_executor_with_config_dict() -> None:
    """A config dict (not an executor) is also accepted."""

    def mul(a: int, b: int) -> int:
        return a * b

    result = await run_in_executor(cast("RunnableConfig", {}), mul, 4, 5)
    assert result == 20


async def test_run_in_executor_with_kwargs() -> None:
    """kwargs are forwarded correctly."""

    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

    result = await run_in_executor(None, greet, "World", greeting="Hi")
    assert result == "Hi, World!"


# ---------------------------------------------------------------------------
# Tests for set_config_context
# ---------------------------------------------------------------------------
from langchain_core.runnables.config import (
    set_config_context,
    var_child_runnable_config,
)


def test_set_config_context_sets_and_resets() -> None:
    """set_config_context sets the var inside the context and resets on exit."""
    config: RunnableConfig = {"tags": ["ctx_test"]}
    with set_config_context(config) as ctx:
        val = ctx.run(var_child_runnable_config.get)
        assert val is not None
        assert val["tags"] == ["ctx_test"]
    # After the context manager exits, the original context is cleaned up
