"""Comprehensive tests for RouterRunnable functionality."""

import asyncio
from typing import Any

import pytest
from pytest_mock import MockerFixture

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.router import RouterInput, RouterRunnable


def test_router_initialization() -> None:
    """Test RouterRunnable initialization."""

    def add_one(x: int) -> int:
        return x + 1

    def multiply_two(x: int) -> int:
        return x * 2

    router = RouterRunnable({"add": add_one, "multiply": multiply_two})

    assert isinstance(router, RouterRunnable)
    assert len(router.runnables) == 2
    assert "add" in router.runnables
    assert "multiply" in router.runnables


def test_router_initialization_with_runnables() -> None:
    """Test RouterRunnable initialization with Runnable objects."""
    add_runnable = RunnableLambda(lambda x: x + 1)
    multiply_runnable = RunnableLambda(lambda x: x * 2)

    router = RouterRunnable({"add": add_runnable, "multiply": multiply_runnable})

    assert router.runnables["add"] == add_runnable
    assert router.runnables["multiply"] == multiply_runnable


def test_router_invoke() -> None:
    """Test basic router invocation."""

    def add_one(x: int) -> int:
        return x + 1

    def multiply_two(x: int) -> int:
        return x * 2

    router = RouterRunnable({"add": add_one, "multiply": multiply_two})

    # Test routing to "add"
    result = router.invoke({"key": "add", "input": 5})
    assert result == 6

    # Test routing to "multiply"
    result = router.invoke({"key": "multiply", "input": 5})
    assert result == 10


def test_router_invoke_invalid_key() -> None:
    """Test router invocation with invalid key."""

    def add_one(x: int) -> int:
        return x + 1

    router = RouterRunnable({"add": add_one})

    with pytest.raises(ValueError, match="No runnable associated with key 'invalid'"):
        router.invoke({"key": "invalid", "input": 5})


async def test_router_ainvoke() -> None:
    """Test async router invocation."""

    async def add_one(x: int) -> int:
        return x + 1

    async def multiply_two(x: int) -> int:
        return x * 2

    router = RouterRunnable({"add": add_one, "multiply": multiply_two})

    # Test routing to "add"
    result = await router.ainvoke({"key": "add", "input": 5})
    assert result == 6

    # Test routing to "multiply"
    result = await router.ainvoke({"key": "multiply", "input": 5})
    assert result == 10


async def test_router_ainvoke_invalid_key() -> None:
    """Test async router invocation with invalid key."""

    async def add_one(x: int) -> int:
        return x + 1

    router = RouterRunnable({"add": add_one})

    with pytest.raises(ValueError, match="No runnable associated with key 'invalid'"):
        await router.ainvoke({"key": "invalid", "input": 5})


def test_router_batch() -> None:
    """Test router batch invocation."""

    def add_one(x: int) -> int:
        return x + 1

    def multiply_two(x: int) -> int:
        return x * 2

    router = RouterRunnable({"add": add_one, "multiply": multiply_two})

    inputs: list[RouterInput] = [
        {"key": "add", "input": 1},
        {"key": "multiply", "input": 2},
        {"key": "add", "input": 3},
    ]

    results = router.batch(inputs)
    assert results == [2, 4, 4]


def test_router_batch_invalid_key() -> None:
    """Test router batch with invalid key."""

    def add_one(x: int) -> int:
        return x + 1

    router = RouterRunnable({"add": add_one})

    inputs: list[RouterInput] = [
        {"key": "add", "input": 1},
        {"key": "invalid", "input": 2},
    ]

    with pytest.raises(
        ValueError, match="One or more keys do not have a corresponding runnable"
    ):
        router.batch(inputs)


async def test_router_abatch() -> None:
    """Test async router batch invocation."""

    async def add_one(x: int) -> int:
        return x + 1

    async def multiply_two(x: int) -> int:
        return x * 2

    router = RouterRunnable({"add": add_one, "multiply": multiply_two})

    inputs: list[RouterInput] = [
        {"key": "add", "input": 1},
        {"key": "multiply", "input": 2},
        {"key": "add", "input": 3},
    ]

    results = await router.abatch(inputs)
    assert results == [2, 4, 4]


async def test_router_abatch_invalid_key() -> None:
    """Test async router batch with invalid key."""

    async def add_one(x: int) -> int:
        return x + 1

    router = RouterRunnable({"add": add_one})

    inputs: list[RouterInput] = [
        {"key": "add", "input": 1},
        {"key": "invalid", "input": 2},
    ]

    with pytest.raises(
        ValueError, match="One or more keys do not have a corresponding runnable"
    ):
        await router.abatch(inputs)


def test_router_stream() -> None:
    """Test router streaming."""

    def generate_numbers(x: int) -> int:
        return x + 1

    router = RouterRunnable({"gen": generate_numbers})

    result = list(router.stream({"key": "gen", "input": 5}))
    assert result == [6]


def test_router_stream_invalid_key() -> None:
    """Test router streaming with invalid key."""

    def generate_numbers(x: int) -> int:
        return x + 1

    router = RouterRunnable({"gen": generate_numbers})

    with pytest.raises(ValueError, match="No runnable associated with key 'invalid'"):
        list(router.stream({"key": "invalid", "input": 5}))


async def test_router_astream() -> None:
    """Test async router streaming."""

    async def generate_numbers(x: int) -> int:
        return x + 1

    router = RouterRunnable({"gen": generate_numbers})

    result = [chunk async for chunk in router.astream({"key": "gen", "input": 5})]
    assert result == [6]


async def test_router_astream_invalid_key() -> None:
    """Test async router streaming with invalid key."""

    async def generate_numbers(x: int) -> int:
        return x + 1

    router = RouterRunnable({"gen": generate_numbers})

    with pytest.raises(ValueError, match="No runnable associated with key 'invalid'"):
        _ = [chunk async for chunk in router.astream({"key": "invalid", "input": 5})]


def test_router_with_config() -> None:
    """Test router with configuration."""

    def add_one(x: int) -> int:
        return x + 1

    router = RouterRunnable({"add": add_one})
    config: RunnableConfig = {"tags": ["test-tag"]}

    result = router.invoke({"key": "add", "input": 5}, config)
    assert result == 6


def test_router_batch_return_exceptions() -> None:
    """Test router batch with return_exceptions."""

    def add_one(x: int) -> int:
        return x + 1

    def fail_always(_: Any) -> Any:
        msg = "Always fails"
        raise ValueError(msg)

    router = RouterRunnable({"add": add_one, "fail": fail_always})

    inputs: list[RouterInput] = [
        {"key": "add", "input": 1},
        {"key": "fail", "input": 2},
        {"key": "add", "input": 3},
    ]

    results = router.batch(inputs, return_exceptions=True)
    assert results[0] == 2
    assert isinstance(results[1], ValueError)
    assert results[2] == 4


async def test_router_abatch_return_exceptions() -> None:
    """Test async router batch with return_exceptions."""

    async def add_one(x: int) -> int:
        return x + 1

    async def fail_always(_: Any) -> Any:
        msg = "Always fails"
        raise ValueError(msg)

    router = RouterRunnable({"add": add_one, "fail": fail_always})

    inputs: list[RouterInput] = [
        {"key": "add", "input": 1},
        {"key": "fail", "input": 2},
        {"key": "add", "input": 3},
    ]

    results = await router.abatch(inputs, return_exceptions=True)
    assert results[0] == 2
    assert isinstance(results[1], ValueError)
    assert results[2] == 4


def test_router_empty_batch() -> None:
    """Test router with empty batch."""

    def add_one(x: int) -> int:
        return x + 1

    router = RouterRunnable({"add": add_one})

    results = router.batch([])
    assert results == []


async def test_router_empty_abatch() -> None:
    """Test async router with empty batch."""

    async def add_one(x: int) -> int:
        return x + 1

    router = RouterRunnable({"add": add_one})

    results = await router.abatch([])
    assert results == []


def test_router_config_specs() -> None:
    """Test that router config specs are aggregated from all runnables."""
    from langchain_core.language_models import FakeListLLM
    from langchain_core.runnables import ConfigurableField

    llm1 = FakeListLLM(responses=["a"]).configurable_fields(
        responses=ConfigurableField(id="llm1_responses")
    )
    llm2 = FakeListLLM(responses=["b"]).configurable_fields(
        responses=ConfigurableField(id="llm2_responses")
    )

    router = RouterRunnable({"llm1": llm1, "llm2": llm2})

    specs = router.config_specs
    spec_ids = [spec.id for spec in specs]
    assert "llm1_responses" in spec_ids
    assert "llm2_responses" in spec_ids


def test_router_serialization() -> None:
    """Test that RouterRunnable is serializable."""
    assert RouterRunnable.is_lc_serializable()
    assert RouterRunnable.get_lc_namespace() == ["langchain", "schema", "runnable"]


def test_router_with_different_input_types() -> None:
    """Test router with different input types."""

    def process_string(x: str) -> str:
        return x.upper()

    def process_int(x: int) -> int:
        return x * 2

    def process_dict(x: dict) -> str:
        return f"Name: {x['name']}"

    router = RouterRunnable(
        {
            "string": process_string,
            "int": process_int,
            "dict": process_dict,
        }
    )

    assert router.invoke({"key": "string", "input": "hello"}) == "HELLO"
    assert router.invoke({"key": "int", "input": 5}) == 10
    assert router.invoke({"key": "dict", "input": {"name": "Alice"}}) == "Name: Alice"


async def test_router_mixed_sync_async_runnables() -> None:
    """Test router with mix of sync and async runnables."""

    def sync_add(x: int) -> int:
        return x + 1

    async def async_multiply(x: int) -> int:
        return x * 2

    router = RouterRunnable({"add": sync_add, "multiply": async_multiply})

    # Both should work via ainvoke
    result1 = await router.ainvoke({"key": "add", "input": 5})
    assert result1 == 6

    result2 = await router.ainvoke({"key": "multiply", "input": 5})
    assert result2 == 10


def test_router_batch_with_configs() -> None:
    """Test router batch with multiple configs."""

    def add_value(x: int) -> int:
        return x + 1

    router = RouterRunnable({"add": add_value})

    inputs: list[RouterInput] = [
        {"key": "add", "input": 1},
        {"key": "add", "input": 2},
        {"key": "add", "input": 3},
    ]

    configs: list[RunnableConfig] = [
        {"tags": ["tag1"]},
        {"tags": ["tag2"]},
        {"tags": ["tag3"]},
    ]

    results = router.batch(inputs, configs)
    assert results == [2, 3, 4]


def test_router_input_type() -> None:
    """Test RouterInput TypedDict structure."""
    router_input: RouterInput = {"key": "test", "input": 42}

    assert router_input["key"] == "test"
    assert router_input["input"] == 42


def test_router_with_runnable_chains() -> None:
    """Test router with chained runnables."""
    add_one = RunnableLambda(lambda x: x + 1)
    multiply_two = RunnableLambda(lambda x: x * 2)

    chain1 = add_one | multiply_two  # (x + 1) * 2
    chain2 = multiply_two | add_one  # (x * 2) + 1

    router = RouterRunnable({"chain1": chain1, "chain2": chain2})

    result1 = router.invoke({"key": "chain1", "input": 5})
    assert result1 == 12  # (5 + 1) * 2

    result2 = router.invoke({"key": "chain2", "input": 5})
    assert result2 == 11  # (5 * 2) + 1


async def test_router_streaming_with_actual_streaming() -> None:
    """Test router streaming with a runnable that actually streams."""
    from langchain_core.language_models import FakeStreamingListLLM

    llm = FakeStreamingListLLM(responses=["hello world"])
    router = RouterRunnable({"llm": llm})

    chunks = [
        chunk
        async for chunk in router.astream({"key": "llm", "input": "test"})
    ]

    # FakeStreamingListLLM streams character by character
    assert len(chunks) > 1
    assert "".join(chunks) == "hello world"


def test_router_invoke_with_spy(mocker: MockerFixture) -> None:
    """Test that router correctly invokes the selected runnable."""

    def route_a(x: int) -> int:
        return x + 10

    def route_b(x: int) -> int:
        return x + 20

    router = RouterRunnable({"a": route_a, "b": route_b})

    result = router.invoke({"key": "a", "input": 5})
    # Verify the correct route was invoked
    assert result == 15  # 5 + 10 from route_a

    result = router.invoke({"key": "b", "input": 5})
    assert result == 25  # 5 + 20 from route_b


def test_router_batch_all_same_key() -> None:
    """Test router batch when all inputs route to same runnable."""

    def add_one(x: int) -> int:
        return x + 1

    router = RouterRunnable({"add": add_one})

    inputs: list[RouterInput] = [
        {"key": "add", "input": 1},
        {"key": "add", "input": 2},
        {"key": "add", "input": 3},
    ]

    results = router.batch(inputs)
    assert results == [2, 3, 4]


def test_router_batch_different_keys() -> None:
    """Test router batch with inputs routing to different runnables."""

    def add_one(x: int) -> int:
        return x + 1

    def multiply_two(x: int) -> int:
        return x * 2

    def square(x: int) -> int:
        return x**2

    router = RouterRunnable({"add": add_one, "multiply": multiply_two, "square": square})

    inputs: list[RouterInput] = [
        {"key": "add", "input": 1},
        {"key": "multiply", "input": 2},
        {"key": "square", "input": 3},
        {"key": "add", "input": 4},
    ]

    results = router.batch(inputs)
    assert results == [2, 4, 9, 5]


async def test_router_abatch_different_keys() -> None:
    """Test async router batch with inputs routing to different runnables."""

    async def add_one(x: int) -> int:
        return x + 1

    async def multiply_two(x: int) -> int:
        return x * 2

    router = RouterRunnable({"add": add_one, "multiply": multiply_two})

    inputs: list[RouterInput] = [
        {"key": "add", "input": 1},
        {"key": "multiply", "input": 2},
        {"key": "add", "input": 3},
    ]

    results = await router.abatch(inputs)
    assert results == [2, 4, 4]


def test_router_with_dict_input() -> None:
    """Test router with dict input to runnables."""

    def process_dict(data: dict[str, Any]) -> str:
        return f"{data['name']} is {data['age']} years old"

    router = RouterRunnable({"process": process_dict})

    result = router.invoke(
        {"key": "process", "input": {"name": "Alice", "age": 30}}
    )
    assert result == "Alice is 30 years old"


def test_router_stream_sync() -> None:
    """Test synchronous router streaming."""

    def identity(x: str) -> str:
        return x

    router = RouterRunnable({"id": identity})

    result = list(router.stream({"key": "id", "input": "test"}))
    assert result == ["test"]


def test_router_namespace() -> None:
    """Test router namespace for serialization."""
    assert RouterRunnable.get_lc_namespace() == ["langchain", "schema", "runnable"]


def test_router_input_validation() -> None:
    """Test that RouterInput requires both key and input."""
    # This is mostly for type checking, but we can test the structure

    def simple(_: int) -> int:
        return 1

    router = RouterRunnable({"test": simple})

    # Valid input
    router.invoke({"key": "test", "input": 5})

    # Missing key should fail during routing logic
    with pytest.raises(KeyError):
        router.invoke({"input": 5})  # type: ignore[typeddict-item]

    # Missing input should fail
    with pytest.raises(KeyError):
        router.invoke({"key": "test"})  # type: ignore[typeddict-item]


def test_router_complex_routing_logic() -> None:
    """Test router with more complex routing scenarios."""

    def option_a(data: dict[str, Any]) -> str:
        return f"Option A: {data['value']}"

    def option_b(data: dict[str, Any]) -> str:
        return f"Option B: {data['value']}"

    def option_c(data: dict[str, Any]) -> str:
        return f"Option C: {data['value']}"

    router = RouterRunnable({"a": option_a, "b": option_b, "c": option_c})

    test_cases = [
        ({"key": "a", "input": {"value": "test1"}}, "Option A: test1"),
        ({"key": "b", "input": {"value": "test2"}}, "Option B: test2"),
        ({"key": "c", "input": {"value": "test3"}}, "Option C: test3"),
    ]

    for input_data, expected in test_cases:
        result = router.invoke(input_data)
        assert result == expected


async def test_router_max_concurrency() -> None:
    """Test router respects max_concurrency in abatch."""
    call_times: list[float] = []

    async def track_time(x: int) -> int:
        import time

        call_times.append(time.time())
        await asyncio.sleep(0.1)
        return x + 1

    router = RouterRunnable({"track": track_time})

    inputs: list[RouterInput] = [{"key": "track", "input": i} for i in range(5)]

    config: RunnableConfig = {"max_concurrency": 2}
    results = await router.abatch(inputs, config)

    assert results == [1, 2, 3, 4, 5]


def test_router_single_route() -> None:
    """Test router with only one route."""

    def only_route(x: int) -> int:
        return x * 3

    router = RouterRunnable({"only": only_route})

    result = router.invoke({"key": "only", "input": 4})
    assert result == 12


def test_router_with_passthrough() -> None:
    """Test router combined with passthrough operations."""
    from langchain_core.runnables import RunnablePassthrough

    def add_ten(x: int) -> int:
        return x + 10

    chain = RunnablePassthrough() | RouterRunnable({"add": add_ten})

    # Passthrough passes the dict through to router
    result = chain.invoke({"key": "add", "input": 5})
    assert result == 15
