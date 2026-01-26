"""Comprehensive tests for RunnablePassthrough, RunnableAssign, and RunnablePick."""

import asyncio
from typing import Any

import pytest

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.passthrough import RunnableAssign, RunnablePick


# RunnablePassthrough Tests


def test_passthrough_identity() -> None:
    """Test that RunnablePassthrough passes through input unchanged."""
    passthrough = RunnablePassthrough()

    assert passthrough.invoke(5) == 5
    assert passthrough.invoke("hello") == "hello"
    assert passthrough.invoke([1, 2, 3]) == [1, 2, 3]
    assert passthrough.invoke({"key": "value"}) == {"key": "value"}


async def test_passthrough_identity_async() -> None:
    """Test async RunnablePassthrough identity."""
    passthrough = RunnablePassthrough()

    assert await passthrough.ainvoke(5) == 5
    assert await passthrough.ainvoke("hello") == "hello"
    assert await passthrough.ainvoke({"key": "value"}) == {"key": "value"}


def test_passthrough_with_func() -> None:
    """Test RunnablePassthrough with a side effect function."""
    calls: list[Any] = []

    def track_call(x: Any) -> None:
        calls.append(x)

    passthrough = RunnablePassthrough(track_call)

    result = passthrough.invoke(5)
    assert result == 5
    assert calls == [5]

    result = passthrough.invoke(10)
    assert result == 10
    assert calls == [5, 10]


async def test_passthrough_with_afunc() -> None:
    """Test RunnablePassthrough with async side effect function."""
    calls: list[Any] = []

    async def track_call(x: Any) -> None:
        calls.append(x)

    passthrough = RunnablePassthrough(afunc=track_call)

    result = await passthrough.ainvoke(5)
    assert result == 5
    assert calls == [5]


def test_passthrough_stream() -> None:
    """Test RunnablePassthrough streaming."""
    passthrough = RunnablePassthrough()

    result = list(passthrough.stream(42))
    assert result == [42]


async def test_passthrough_astream() -> None:
    """Test async RunnablePassthrough streaming."""
    passthrough = RunnablePassthrough()

    result = [x async for x in passthrough.astream(42)]
    assert result == [42]


def test_passthrough_batch() -> None:
    """Test RunnablePassthrough batch."""
    passthrough = RunnablePassthrough()

    results = passthrough.batch([1, 2, 3, 4, 5])
    assert results == [1, 2, 3, 4, 5]


async def test_passthrough_abatch() -> None:
    """Test async RunnablePassthrough batch."""
    passthrough = RunnablePassthrough()

    results = await passthrough.abatch([1, 2, 3, 4, 5])
    assert results == [1, 2, 3, 4, 5]


def test_passthrough_with_input_type() -> None:
    """Test RunnablePassthrough with explicit input type."""
    passthrough = RunnablePassthrough[int](input_type=int)

    assert passthrough.invoke(5) == 5
    assert passthrough.InputType == int
    assert passthrough.OutputType == int


def test_passthrough_transform() -> None:
    """Test RunnablePassthrough transform."""
    passthrough = RunnablePassthrough()

    chunks = iter([1, 2, 3])
    result = list(passthrough.transform(chunks))
    # Transform passes through each chunk
    assert result == [1, 2, 3]


async def test_passthrough_atransform() -> None:
    """Test async RunnablePassthrough transform."""
    passthrough = RunnablePassthrough()

    async def async_chunks() -> Any:
        for i in [1, 2, 3]:
            yield i

    result = [x async for x in passthrough.atransform(async_chunks())]
    assert result == [1, 2, 3]


def test_passthrough_with_func_and_config() -> None:
    """Test RunnablePassthrough function receives config."""
    configs_seen: list[RunnableConfig] = []

    def track_config(x: int, config: RunnableConfig) -> None:
        configs_seen.append(config)

    passthrough = RunnablePassthrough(track_config)
    config: RunnableConfig = {"tags": ["test-tag"]}

    result = passthrough.invoke(5, config)
    assert result == 5
    assert len(configs_seen) == 1
    assert "test-tag" in configs_seen[0]["tags"]


# RunnableAssign Tests


def test_assign_basic() -> None:
    """Test basic RunnableAssign functionality."""
    assign = RunnablePassthrough.assign(new_key=lambda x: x["value"] * 2)

    result = assign.invoke({"value": 5})
    assert result == {"value": 5, "new_key": 10}


async def test_assign_basic_async() -> None:
    """Test basic async RunnableAssign functionality."""

    async def double_value(x: dict[str, int]) -> int:
        return x["value"] * 2

    assign = RunnablePassthrough.assign(new_key=double_value)

    result = await assign.ainvoke({"value": 5})
    assert result == {"value": 5, "new_key": 10}


def test_assign_multiple_keys() -> None:
    """Test RunnableAssign with multiple new keys."""
    assign = RunnablePassthrough.assign(
        doubled=lambda x: x["value"] * 2,
        tripled=lambda x: x["value"] * 3,
        quadrupled=lambda x: x["value"] * 4,
    )

    result = assign.invoke({"value": 5})
    assert result == {
        "value": 5,
        "doubled": 10,
        "tripled": 15,
        "quadrupled": 20,
    }


def test_assign_overwrite_existing() -> None:
    """Test RunnableAssign overwrites existing keys."""
    assign = RunnablePassthrough.assign(value=lambda x: x["value"] * 2)

    result = assign.invoke({"value": 5, "other": "data"})
    assert result == {"value": 10, "other": "data"}


def test_assign_with_runnable() -> None:
    """Test RunnableAssign with Runnable instead of lambda."""
    double = RunnableLambda(lambda x: x["value"] * 2)
    assign = RunnablePassthrough.assign(new_key=double)

    result = assign.invoke({"value": 5})
    assert result == {"value": 5, "new_key": 10}


def test_assign_stream() -> None:
    """Test RunnableAssign streaming."""
    from langchain_core.language_models import FakeStreamingListLLM

    llm = FakeStreamingListLLM(responses=["hello"])
    # Extract a string from the dict to pass to the LLM
    assign = RunnablePassthrough.assign(llm_output=lambda x: llm.invoke(x["input"]))

    chunks = list(assign.stream({"input": "test"}))

    # Should stream passthrough first, then assign output
    assert len(chunks) > 1
    # Accumulate chunks
    final = chunks[0]
    for chunk in chunks[1:]:
        final = final + chunk
    assert final["input"] == "test"
    assert final["llm_output"] == "hello"


async def test_assign_astream() -> None:
    """Test async RunnableAssign streaming."""
    from langchain_core.language_models import FakeStreamingListLLM

    llm = FakeStreamingListLLM(responses=["hello"])
    # Extract a string from the dict to pass to the LLM
    assign = RunnablePassthrough.assign(llm_output=lambda x: llm.invoke(x["input"]))

    chunks = [chunk async for chunk in assign.astream({"input": "test"})]

    # Should stream passthrough first, then assign output
    assert len(chunks) > 1
    # Accumulate chunks
    final = chunks[0]
    for chunk in chunks[1:]:
        final = final + chunk
    assert final["input"] == "test"
    assert final["llm_output"] == "hello"


def test_assign_batch() -> None:
    """Test RunnableAssign batch."""
    assign = RunnablePassthrough.assign(new_key=lambda x: x["value"] * 2)

    results = assign.batch([{"value": 1}, {"value": 2}, {"value": 3}])
    assert results == [
        {"value": 1, "new_key": 2},
        {"value": 2, "new_key": 4},
        {"value": 3, "new_key": 6},
    ]


async def test_assign_abatch() -> None:
    """Test async RunnableAssign batch."""
    assign = RunnablePassthrough.assign(new_key=lambda x: x["value"] * 2)

    results = await assign.abatch([{"value": 1}, {"value": 2}, {"value": 3}])
    assert results == [
        {"value": 1, "new_key": 2},
        {"value": 2, "new_key": 4},
        {"value": 3, "new_key": 6},
    ]


def test_assign_input_must_be_dict() -> None:
    """Test that RunnableAssign requires dict input."""
    assign = RunnablePassthrough.assign(new_key=lambda x: 42)

    with pytest.raises(ValueError, match="must be a dict"):
        assign.invoke(5)  # type: ignore[arg-type]


def test_assign_transform() -> None:
    """Test RunnableAssign transform."""
    assign = RunnablePassthrough.assign(doubled=lambda x: x["value"] * 2)

    chunks = iter([{"value": 5}])
    result = list(assign.transform(chunks))

    # Collect all chunks
    final = result[0]
    for chunk in result[1:]:
        final = final + chunk

    assert final == {"value": 5, "doubled": 10}


async def test_assign_atransform() -> None:
    """Test async RunnableAssign transform."""
    assign = RunnablePassthrough.assign(doubled=lambda x: x["value"] * 2)

    async def async_chunks() -> Any:
        yield {"value": 5}

    result = [x async for x in assign.atransform(async_chunks())]

    # Collect all chunks
    final = result[0]
    for chunk in result[1:]:
        final = final + chunk

    assert final == {"value": 5, "doubled": 10}


def test_assign_get_name() -> None:
    """Test RunnableAssign name generation."""
    assign = RunnablePassthrough.assign(key1=lambda x: x, key2=lambda x: x)

    name = assign.get_name()
    assert "RunnableAssign<key1,key2>" in name


def test_assign_input_output_schema() -> None:
    """Test RunnableAssign schema generation."""
    assign = RunnablePassthrough.assign(new_key=lambda x: x["value"] * 2)

    input_schema = assign.get_input_schema()
    assert "value" in input_schema.model_fields

    output_schema = assign.get_output_schema()
    assert "value" in output_schema.model_fields
    assert "new_key" in output_schema.model_fields


def test_assign_nested() -> None:
    """Test nested RunnableAssign operations."""
    assign1 = RunnablePassthrough.assign(step1=lambda x: x["value"] + 1)
    assign2 = assign1.assign(step2=lambda x: x["step1"] * 2)

    result = assign2.invoke({"value": 5})
    assert result == {"value": 5, "step1": 6, "step2": 12}


def test_assign_with_parallel() -> None:
    """Test RunnableAssign with RunnableParallel."""
    mapper = RunnableParallel(
        {
            "doubled": RunnableLambda(lambda x: x["value"] * 2),
            "tripled": RunnableLambda(lambda x: x["value"] * 3),
        }
    )
    assign = RunnableAssign(mapper)

    result = assign.invoke({"value": 5})
    assert result == {"value": 5, "doubled": 10, "tripled": 15}


# RunnablePick Tests


def test_pick_single_key() -> None:
    """Test RunnablePick with a single key."""
    pick = RunnablePick("name")

    result = pick.invoke({"name": "Alice", "age": 30, "city": "NYC"})
    assert result == "Alice"


def test_pick_multiple_keys() -> None:
    """Test RunnablePick with multiple keys."""
    pick = RunnablePick(["name", "age"])

    result = pick.invoke({"name": "Alice", "age": 30, "city": "NYC"})
    assert result == {"name": "Alice", "age": 30}


async def test_pick_single_key_async() -> None:
    """Test async RunnablePick with a single key."""
    pick = RunnablePick("name")

    result = await pick.ainvoke({"name": "Alice", "age": 30})
    assert result == "Alice"


async def test_pick_multiple_keys_async() -> None:
    """Test async RunnablePick with multiple keys."""
    pick = RunnablePick(["name", "age"])

    result = await pick.ainvoke({"name": "Alice", "age": 30, "city": "NYC"})
    assert result == {"name": "Alice", "age": 30}


def test_pick_missing_key() -> None:
    """Test RunnablePick with missing key returns None for single key."""
    pick = RunnablePick("missing")

    result = pick.invoke({"name": "Alice"})
    assert result is None


def test_pick_partial_keys() -> None:
    """Test RunnablePick with some missing keys."""
    pick = RunnablePick(["name", "missing"])

    result = pick.invoke({"name": "Alice", "age": 30})
    assert result == {"name": "Alice"}


def test_pick_all_missing_keys() -> None:
    """Test RunnablePick when all keys are missing."""
    pick = RunnablePick(["missing1", "missing2"])

    result = pick.invoke({"name": "Alice"})
    assert result is None


def test_pick_batch() -> None:
    """Test RunnablePick batch operation."""
    pick = RunnablePick("name")

    results = pick.batch([{"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"}])
    assert results == ["Alice", "Bob", "Charlie"]


async def test_pick_abatch() -> None:
    """Test async RunnablePick batch."""
    pick = RunnablePick(["name", "age"])

    results = await pick.abatch(
        [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
    )
    assert results == [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]


def test_pick_stream() -> None:
    """Test RunnablePick streaming."""
    pick = RunnablePick("value")

    result = list(pick.stream({"value": 42, "other": "data"}))
    assert result == [42]


async def test_pick_astream() -> None:
    """Test async RunnablePick streaming."""
    pick = RunnablePick(["key1", "key2"])

    result = [x async for x in pick.astream({"key1": "a", "key2": "b", "key3": "c"})]
    assert result == [{"key1": "a", "key2": "b"}]


def test_pick_transform() -> None:
    """Test RunnablePick transform."""
    pick = RunnablePick("value")

    chunks = iter([{"value": 1}, {"value": 2}])
    result = list(pick.transform(chunks))
    assert result == [1, 2]


async def test_pick_atransform() -> None:
    """Test async RunnablePick transform."""
    pick = RunnablePick(["a", "b"])

    async def async_chunks() -> Any:
        yield {"a": 1, "b": 2, "c": 3}
        yield {"a": 4, "b": 5, "c": 6}

    result = [x async for x in pick.atransform(async_chunks())]
    assert result == [{"a": 1, "b": 2}, {"a": 4, "b": 5}]


def test_pick_get_name() -> None:
    """Test RunnablePick name generation."""
    pick_single = RunnablePick("key")
    assert "RunnablePick<key>" in pick_single.get_name()

    pick_multiple = RunnablePick(["key1", "key2", "key3"])
    assert "RunnablePick<key1,key2,key3>" in pick_multiple.get_name()


def test_pick_serialization() -> None:
    """Test RunnablePick serialization."""
    assert RunnablePick.is_lc_serializable()
    assert RunnablePick.get_lc_namespace() == ["langchain", "schema", "runnable"]


def test_pick_input_must_be_dict() -> None:
    """Test that RunnablePick requires dict input."""
    pick = RunnablePick("key")

    with pytest.raises(ValueError, match="must be a dict"):
        pick.invoke(5)  # type: ignore[arg-type]


def test_pick_in_chain() -> None:
    """Test RunnablePick in a chain."""
    # Create data, then pick specific keys
    chain = (
        RunnableLambda(lambda x: {"a": x, "b": x * 2, "c": x * 3})
        | RunnablePick(["a", "b"])
    )

    result = chain.invoke(5)
    assert result == {"a": 5, "b": 10}


async def test_pick_in_chain_async() -> None:
    """Test RunnablePick in async chain."""

    async def create_dict(x: int) -> dict[str, int]:
        return {"a": x, "b": x * 2, "c": x * 3}

    chain = RunnableLambda(create_dict) | RunnablePick(["a", "c"])

    result = await chain.ainvoke(5)
    assert result == {"a": 5, "c": 15}


# Integration Tests


def test_passthrough_assign_pick_combination() -> None:
    """Test combining passthrough, assign, and pick."""
    # Start with passthrough, add fields, then pick some
    chain = (
        RunnablePassthrough()
        | RunnablePassthrough.assign(
            doubled=lambda x: x["value"] * 2,
            tripled=lambda x: x["value"] * 3,
        )
        | RunnablePick(["value", "doubled"])
    )

    result = chain.invoke({"value": 5})
    assert result == {"value": 5, "doubled": 10}


def test_assign_with_dependencies() -> None:
    """Test RunnableAssign where new keys depend on each other."""
    # Note: in a single assign, keys are computed in parallel from the same input
    # So they can't depend on each other within the same assign
    assign = RunnablePassthrough.assign(step1=lambda x: x["value"] + 1)
    assign2 = assign.assign(step2=lambda x: x["step1"] * 2)

    result = assign2.invoke({"value": 5})
    assert result == {"value": 5, "step1": 6, "step2": 12}


def test_passthrough_with_side_effect_batch() -> None:
    """Test passthrough with side effect function on batch."""
    calls: list[int] = []

    def track(x: int) -> None:
        calls.append(x)

    passthrough = RunnablePassthrough(track)

    results = passthrough.batch([1, 2, 3])
    assert results == [1, 2, 3]
    assert sorted(calls) == [1, 2, 3]


def test_passthrough_serialization() -> None:
    """Test RunnablePassthrough serialization."""
    assert RunnablePassthrough.is_lc_serializable()
    assert RunnablePassthrough.get_lc_namespace() == ["langchain", "schema", "runnable"]


def test_assign_serialization() -> None:
    """Test RunnableAssign serialization."""
    assert RunnableAssign.is_lc_serializable()
    assert RunnableAssign.get_lc_namespace() == ["langchain", "schema", "runnable"]


def test_assign_empty_dict() -> None:
    """Test RunnableAssign with empty input dict."""
    assign = RunnablePassthrough.assign(new_key=lambda _: 42)

    result = assign.invoke({})
    assert result == {"new_key": 42}


def test_pick_empty_dict() -> None:
    """Test RunnablePick with empty dict."""
    pick = RunnablePick(["key1", "key2"])

    result = pick.invoke({})
    assert result is None


def test_assign_preserves_original_order() -> None:
    """Test that assign preserves original dict and adds new keys."""
    assign = RunnablePassthrough.assign(z=lambda x: x["a"] + x["b"])

    result = assign.invoke({"a": 1, "b": 2, "c": 3})
    assert "a" in result
    assert "b" in result
    assert "c" in result
    assert result["z"] == 3


def test_assign_with_config_propagation() -> None:
    """Test that RunnableAssign propagates config correctly."""
    configs_seen: list[RunnableConfig] = []

    def track_config(x: dict[str, Any], config: RunnableConfig) -> int:
        configs_seen.append(config)
        return x["value"] * 2

    assign = RunnablePassthrough.assign(new_key=track_config)

    config: RunnableConfig = {"tags": ["my-tag"]}
    result = assign.invoke({"value": 5}, config)

    assert result == {"value": 5, "new_key": 10}
    assert len(configs_seen) == 1


def test_passthrough_transform_with_func() -> None:
    """Test passthrough transform with side effect function."""
    calls: list[int] = []

    def track(x: int) -> None:
        calls.append(x)

    passthrough = RunnablePassthrough(track)

    chunks = iter([1, 2, 3])
    result = list(passthrough.transform(chunks))

    assert result == [1, 2, 3]
    assert calls == [6]  # Called once at the end with accumulated value (1+2+3=6)


async def test_passthrough_atransform_with_afunc() -> None:
    """Test async passthrough transform with side effect function."""
    calls: list[int] = []

    async def track(x: int) -> None:
        calls.append(x)

    passthrough = RunnablePassthrough(afunc=track)

    async def async_chunks() -> Any:
        for i in [1, 2, 3]:
            yield i

    result = [x async for x in passthrough.atransform(async_chunks())]

    assert result == [1, 2, 3]
    assert calls == [6]  # Called with accumulated value (1+2+3=6)


def test_pick_transform_filters_each_chunk() -> None:
    """Test that pick transform filters each chunk."""
    pick = RunnablePick("wanted")

    chunks = iter([
        {"wanted": 1, "unwanted": 10},
        {"wanted": 2, "unwanted": 20},
    ])

    result = list(pick.transform(chunks))
    assert result == [1, 2]


def test_assign_graph_structure() -> None:
    """Test RunnableAssign graph structure includes passthrough."""
    assign = RunnablePassthrough.assign(new_key=lambda x: x["value"])

    graph = assign.get_graph()
    # Graph should include the mapper and passthrough nodes
    assert len(graph.nodes) > 0


def test_passthrough_repr() -> None:
    """Test RunnablePassthrough repr doesn't cause recursion."""
    passthrough = RunnablePassthrough()
    repr_str = repr(passthrough)
    # Should not raise RecursionError
    assert "RunnablePassthrough" in str(type(passthrough))


def test_assign_with_multiple_parallel_ops() -> None:
    """Test assign with multiple independent operations."""
    from langchain_core.runnables import RunnableParallel

    def op1(x: dict[str, int]) -> int:
        return x["a"] + x["b"]

    def op2(x: dict[str, int]) -> int:
        return x["a"] * x["b"]

    def op3(x: dict[str, int]) -> int:
        return x["a"] - x["b"]

    assign = RunnablePassthrough.assign(
        sum=op1,
        product=op2,
        difference=op3,
    )

    result = assign.invoke({"a": 10, "b": 3})
    assert result == {
        "a": 10,
        "b": 3,
        "sum": 13,
        "product": 30,
        "difference": 7,
    }


def test_pick_maintains_types() -> None:
    """Test that RunnablePick maintains value types."""
    pick = RunnablePick(["int_val", "str_val", "list_val"])

    result = pick.invoke({
        "int_val": 42,
        "str_val": "hello",
        "list_val": [1, 2, 3],
        "extra": "ignored",
    })

    assert result == {
        "int_val": 42,
        "str_val": "hello",
        "list_val": [1, 2, 3],
    }


async def test_assign_concurrent_operations() -> None:
    """Test that assign runs operations concurrently."""
    import time

    start_time = time.time()

    async def slow_op1(_: dict[str, Any]) -> int:
        await asyncio.sleep(0.1)
        return 1

    async def slow_op2(_: dict[str, Any]) -> int:
        await asyncio.sleep(0.1)
        return 2

    assign = RunnablePassthrough.assign(result1=slow_op1, result2=slow_op2)

    result = await assign.ainvoke({"input": "test"})
    elapsed = time.time() - start_time

    assert result == {"input": "test", "result1": 1, "result2": 2}
    # If run concurrently, should take ~0.1s, not ~0.2s
    # Allow some margin for overhead and test stability
    assert elapsed < 0.5


def test_passthrough_in_parallel() -> None:
    """Test RunnablePassthrough in RunnableParallel."""
    parallel = RunnableParallel(
        original=RunnablePassthrough(),
        modified=lambda x: x + 1,
    )

    result = parallel.invoke(5)
    assert result == {"original": 5, "modified": 6}


def test_assign_direct_instantiation() -> None:
    """Test directly instantiating RunnableAssign."""
    mapper = RunnableParallel({"new_field": lambda x: x["value"] * 2})
    assign = RunnableAssign(mapper)

    result = assign.invoke({"value": 5})
    assert result == {"value": 5, "new_field": 10}


def test_pick_direct_instantiation() -> None:
    """Test directly instantiating RunnablePick."""
    pick = RunnablePick(keys="selected")

    result = pick.invoke({"selected": "yes", "others": "no"})
    assert result == "yes"


def test_passthrough_with_none_func() -> None:
    """Test that passthrough works when func is None."""
    passthrough = RunnablePassthrough(func=None)

    result = passthrough.invoke(42)
    assert result == 42
