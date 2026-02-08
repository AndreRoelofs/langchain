from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from typing_extensions import override

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    LLM,
    BaseLLM,
    FakeListLLM,
)
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.tracers.context import collect_runs
from tests.unit_tests.fake.callbacks import (
    BaseFakeCallbackHandler,
    FakeAsyncCallbackHandler,
    FakeCallbackHandler,
)


def test_batch() -> None:
    llm = FakeListLLM(responses=["foo"] * 3)
    output = llm.batch(["foo", "bar", "foo"])
    assert output == ["foo"] * 3

    output = llm.batch(["foo", "bar", "foo"], config={"max_concurrency": 2})
    assert output == ["foo"] * 3


async def test_abatch() -> None:
    llm = FakeListLLM(responses=["foo"] * 3)
    output = await llm.abatch(["foo", "bar", "foo"])
    assert output == ["foo"] * 3

    output = await llm.abatch(["foo", "bar", "foo"], config={"max_concurrency": 2})
    assert output == ["foo"] * 3


def test_batch_size() -> None:
    llm = FakeListLLM(responses=["foo"] * 3)
    with collect_runs() as cb:
        llm.batch(["foo", "bar", "foo"], {"callbacks": [cb]})
        assert all((r.extra or {}).get("batch_size") == 3 for r in cb.traced_runs)
        assert len(cb.traced_runs) == 3
    llm = FakeListLLM(responses=["foo"])
    with collect_runs() as cb:
        llm.batch(["foo"], {"callbacks": [cb]})
        assert all((r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs)
        assert len(cb.traced_runs) == 1

    llm = FakeListLLM(responses=["foo"])
    with collect_runs() as cb:
        llm.invoke("foo")
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1

    llm = FakeListLLM(responses=["foo"])
    with collect_runs() as cb:
        list(llm.stream("foo"))
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1

    llm = FakeListLLM(responses=["foo"] * 1)
    with collect_runs() as cb:
        llm.invoke("foo")
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1


async def test_async_batch_size() -> None:
    llm = FakeListLLM(responses=["foo"] * 3)
    with collect_runs() as cb:
        await llm.abatch(["foo", "bar", "foo"], {"callbacks": [cb]})
        assert all((r.extra or {}).get("batch_size") == 3 for r in cb.traced_runs)
        assert len(cb.traced_runs) == 3
    llm = FakeListLLM(responses=["foo"])
    with collect_runs() as cb:
        await llm.abatch(["foo"], {"callbacks": [cb]})
        assert all((r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs)
        assert len(cb.traced_runs) == 1

    llm = FakeListLLM(responses=["foo"])
    with collect_runs() as cb:
        await llm.ainvoke("foo")
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1

    llm = FakeListLLM(responses=["foo"])
    with collect_runs() as cb:
        async for _ in llm.astream("foo"):
            pass
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1


async def test_error_callback() -> None:
    class FailingLLMError(Exception):
        """FailingLLMError."""

    class FailingLLM(LLM):
        @property
        def _llm_type(self) -> str:
            """Return type of llm."""
            return "failing-llm"

        @override
        def _call(
            self,
            prompt: str,
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> str:
            raise FailingLLMError

    def eval_response(callback: BaseFakeCallbackHandler) -> None:
        assert callback.errors == 1
        assert len(callback.errors_args) == 1
        assert isinstance(callback.errors_args[0]["args"][0], FailingLLMError)

    llm = FailingLLM()
    cb_async = FakeAsyncCallbackHandler()
    with pytest.raises(FailingLLMError):
        await llm.ainvoke("Dummy message", config={"callbacks": [cb_async]})
    eval_response(cb_async)

    cb_sync = FakeCallbackHandler()
    with pytest.raises(FailingLLMError):
        llm.invoke("Dummy message", config={"callbacks": [cb_sync]})
    eval_response(cb_sync)


async def test_astream_fallback_to_ainvoke() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithGenerate(BaseLLM):
        @override
        def _generate(
            self,
            prompts: list[str],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> LLMResult:
            generations = [Generation(text="hello")]
            return LLMResult(generations=[generations])

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    model = ModelWithGenerate()
    chunks = list(model.stream("anything"))
    assert chunks == ["hello"]

    chunks = [chunk async for chunk in model.astream("anything")]
    assert chunks == ["hello"]


async def test_astream_implementation_fallback_to_stream() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithSyncStream(BaseLLM):
        def _generate(
            self,
            prompts: list[str],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> LLMResult:
            """Top Level call."""
            raise NotImplementedError

        @override
        def _stream(
            self,
            prompt: str,
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> Iterator[GenerationChunk]:
            """Stream the output of the model."""
            yield GenerationChunk(text="a")
            yield GenerationChunk(text="b")

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    model = ModelWithSyncStream()
    chunks = list(model.stream("anything"))
    assert chunks == ["a", "b"]
    assert type(model)._astream == BaseLLM._astream
    astream_chunks = [chunk async for chunk in model.astream("anything")]
    assert astream_chunks == ["a", "b"]


async def test_astream_implementation_uses_astream() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithAsyncStream(BaseLLM):
        def _generate(
            self,
            prompts: list[str],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> LLMResult:
            """Top Level call."""
            raise NotImplementedError

        @override
        async def _astream(
            self,
            prompt: str,
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> AsyncIterator[GenerationChunk]:
            """Stream the output of the model."""
            yield GenerationChunk(text="a")
            yield GenerationChunk(text="b")

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    model = ModelWithAsyncStream()
    chunks = [chunk async for chunk in model.astream("anything")]
    assert chunks == ["a", "b"]


def test_get_ls_params() -> None:
    class LSParamsModel(BaseLLM):
        model: str = "foo"
        temperature: float = 0.1
        max_tokens: int = 1024

        @override
        def _generate(
            self,
            prompts: list[str],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> LLMResult:
            raise NotImplementedError

        @property
        def _llm_type(self) -> str:
            return "fake-model"

    llm = LSParamsModel()

    # Test standard tracing params
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "lsparamsmodel",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
        "ls_temperature": 0.1,
        "ls_max_tokens": 1024,
    }

    ls_params = llm._get_ls_params(model="bar")
    assert ls_params["ls_model_name"] == "bar"

    ls_params = llm._get_ls_params(temperature=0.2)
    assert ls_params["ls_temperature"] == 0.2

    ls_params = llm._get_ls_params(max_tokens=2048)
    assert ls_params["ls_max_tokens"] == 2048

    ls_params = llm._get_ls_params(stop=["stop"])
    assert ls_params["ls_stop"] == ["stop"]


# ---------------------------------------------------------------------------
# Additional test coverage
# ---------------------------------------------------------------------------

import json
import uuid
from pathlib import Path
from typing import Union
from unittest.mock import MagicMock, patch

from langchain_core.caches import BaseCache

# Reuse helper imports already at top of file where possible.
from langchain_core.language_models.llms import (
    _resolve_cache,
    create_base_retry_decorator,
    get_prompts,
    update_cache,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue

# ---- Minimal concrete LLM used across multiple tests ----


class _SimpleLLM(LLM):
    """Minimal concrete LLM for testing."""

    response: str = "hello"

    @property
    def _llm_type(self) -> str:
        return "simple-test-llm"

    @override
    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        return self.response


# ===================================================================
# 1. _convert_input
# ===================================================================


class TestConvertInput:
    """Test BaseLLM._convert_input with various input types."""

    def test_convert_string_input(self) -> None:
        llm = _SimpleLLM()
        result = llm._convert_input("hello world")
        assert isinstance(result, StringPromptValue)
        assert result.to_string() == "hello world"

    def test_convert_prompt_value_input(self) -> None:
        llm = _SimpleLLM()
        pv = StringPromptValue(text="already a prompt value")
        result = llm._convert_input(pv)
        assert result is pv

    def test_convert_message_sequence_input(self) -> None:
        llm = _SimpleLLM()
        messages = [HumanMessage(content="hi")]
        result = llm._convert_input(messages)
        assert isinstance(result, ChatPromptValue)
        assert len(result.messages) == 1
        assert result.messages[0].content == "hi"

    def test_convert_invalid_input_raises(self) -> None:
        llm = _SimpleLLM()
        with pytest.raises(ValueError, match="Invalid input type"):
            llm._convert_input(12345)  # type: ignore[arg-type]


# ===================================================================
# 2. BaseLLM.save()
# ===================================================================


class TestSave:
    """Test BaseLLM.save() to JSON and YAML."""

    def test_save_json(self, tmp_path: Path) -> None:
        llm = FakeListLLM(responses=["a", "b"])
        file_path = tmp_path / "llm.json"
        llm.save(file_path)
        assert file_path.exists()
        with file_path.open() as f:
            data = json.load(f)
        assert data["_type"] == "fake-list"
        assert data["responses"] == ["a", "b"]

    def test_save_yaml(self, tmp_path: Path) -> None:
        import yaml

        llm = FakeListLLM(responses=["x"])
        file_path = tmp_path / "llm.yaml"
        llm.save(file_path)
        assert file_path.exists()
        with file_path.open() as f:
            data = yaml.safe_load(f)
        assert data["_type"] == "fake-list"

    def test_save_yml(self, tmp_path: Path) -> None:
        import yaml

        llm = FakeListLLM(responses=["x"])
        file_path = tmp_path / "llm.yml"
        llm.save(file_path)
        assert file_path.exists()
        with file_path.open() as f:
            data = yaml.safe_load(f)
        assert data["_type"] == "fake-list"

    def test_save_invalid_extension_raises(self, tmp_path: Path) -> None:
        llm = FakeListLLM(responses=["a"])
        file_path = tmp_path / "llm.txt"
        with pytest.raises(ValueError, match="must be json or yaml"):
            llm.save(file_path)


# ===================================================================
# 3. BaseLLM.dict()
# ===================================================================


def test_dict_contains_type_and_identifying_params() -> None:
    llm = FakeListLLM(responses=["a", "b"])
    d = llm.dict()
    assert "_type" in d
    assert d["_type"] == "fake-list"
    # FakeListLLM._identifying_params returns {"responses": ...}
    assert "responses" in d
    assert d["responses"] == ["a", "b"]


# ===================================================================
# 4. BaseLLM.__str__
# ===================================================================


def test_str_representation() -> None:
    llm = FakeListLLM(responses=["foo"])
    result = str(llm)
    assert "FakeListLLM" in result
    assert "Params:" in result


# ===================================================================
# 5. BaseLLM.OutputType
# ===================================================================


def test_output_type_is_str() -> None:
    llm = FakeListLLM(responses=["a"])
    assert llm.OutputType is str


# ===================================================================
# 6. LLM._generate
# ===================================================================


class TestLLMGenerate:
    """Test the LLM subclass _generate method which wraps _call."""

    def test_generate_single_prompt(self) -> None:
        llm = _SimpleLLM(response="result")
        result = llm._generate(["prompt1"])
        assert len(result.generations) == 1
        assert result.generations[0][0].text == "result"

    def test_generate_multiple_prompts(self) -> None:
        llm = _SimpleLLM(response="out")
        result = llm._generate(["p1", "p2", "p3"])
        assert len(result.generations) == 3
        for gen_list in result.generations:
            assert len(gen_list) == 1
            assert gen_list[0].text == "out"

    def test_generate_empty_prompts(self) -> None:
        llm = _SimpleLLM(response="out")
        result = llm._generate([])
        assert len(result.generations) == 0


# ===================================================================
# 7. LLM._agenerate
# ===================================================================


class TestLLMAgenerate:
    """Test the LLM subclass _agenerate method which wraps _acall."""

    async def test_agenerate_single_prompt(self) -> None:
        llm = _SimpleLLM(response="async_result")
        result = await llm._agenerate(["prompt1"])
        assert len(result.generations) == 1
        assert result.generations[0][0].text == "async_result"

    async def test_agenerate_multiple_prompts(self) -> None:
        llm = _SimpleLLM(response="out")
        result = await llm._agenerate(["p1", "p2"])
        assert len(result.generations) == 2
        for gen_list in result.generations:
            assert gen_list[0].text == "out"


# ===================================================================
# 8. _resolve_cache
# ===================================================================


class _FakeCache(BaseCache):
    """Minimal in-memory cache for testing."""

    def __init__(self) -> None:
        self.store: dict[str, list] = {}

    def lookup(self, prompt: str, llm_string: str) -> list | None:
        return self.store.get(f"{prompt}|{llm_string}")

    def update(self, prompt: str, llm_string: str, return_val: list) -> None:
        self.store[f"{prompt}|{llm_string}"] = return_val

    def clear(self, **kwargs: Any) -> None:
        self.store.clear()

    async def alookup(self, prompt: str, llm_string: str) -> list | None:
        return self.lookup(prompt, llm_string)

    async def aupdate(self, prompt: str, llm_string: str, return_val: list) -> None:
        self.update(prompt, llm_string, return_val)

    async def aclear(self, **kwargs: Any) -> None:
        self.clear()


class TestResolveCache:
    """Test _resolve_cache with various cache argument values."""

    def test_cache_is_base_cache_instance(self) -> None:
        cache = _FakeCache()
        assert _resolve_cache(cache=cache) is cache

    def test_cache_is_none_returns_global_cache(self) -> None:
        fake_global = _FakeCache()
        with patch(
            "langchain_core.language_models.llms.get_llm_cache",
            return_value=fake_global,
        ):
            assert _resolve_cache(cache=None) is fake_global

    def test_cache_is_none_no_global_returns_none(self) -> None:
        with patch(
            "langchain_core.language_models.llms.get_llm_cache", return_value=None
        ):
            assert _resolve_cache(cache=None) is None

    def test_cache_is_true_with_global_cache(self) -> None:
        fake_global = _FakeCache()
        with patch(
            "langchain_core.language_models.llms.get_llm_cache",
            return_value=fake_global,
        ):
            assert _resolve_cache(cache=True) is fake_global

    def test_cache_is_true_without_global_cache_raises(self) -> None:
        with patch(
            "langchain_core.language_models.llms.get_llm_cache", return_value=None
        ):
            with pytest.raises(ValueError, match="No global cache was configured"):
                _resolve_cache(cache=True)

    def test_cache_is_false(self) -> None:
        assert _resolve_cache(cache=False) is None

    def test_cache_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported cache value"):
            _resolve_cache(cache="invalid")  # type: ignore[arg-type]


# ===================================================================
# 9. get_prompts
# ===================================================================


class TestGetPrompts:
    """Test get_prompts function with and without cache."""

    def test_no_cache_returns_all_missing(self) -> None:
        with patch(
            "langchain_core.language_models.llms.get_llm_cache", return_value=None
        ):
            existing, llm_string, missing_idxs, missing = get_prompts(
                {"model": "test"}, ["p1", "p2"], cache=None
            )
        assert existing == {}
        assert missing_idxs == []
        assert missing == []

    def test_with_cache_all_miss(self) -> None:
        cache = _FakeCache()
        existing, llm_string, missing_idxs, missing = get_prompts(
            {"model": "test"}, ["p1", "p2"], cache=cache
        )
        assert existing == {}
        assert missing_idxs == [0, 1]
        assert missing == ["p1", "p2"]

    def test_with_cache_partial_hit(self) -> None:
        cache = _FakeCache()
        llm_string = str(sorted({"model": "test"}.items()))
        cached_gen = [Generation(text="cached")]
        cache.update("p1", llm_string, cached_gen)

        existing, _, missing_idxs, missing = get_prompts(
            {"model": "test"}, ["p1", "p2"], cache=cache
        )
        assert 0 in existing
        assert existing[0] == cached_gen
        assert missing_idxs == [1]
        assert missing == ["p2"]

    def test_with_cache_all_hit(self) -> None:
        cache = _FakeCache()
        llm_string = str(sorted({"model": "test"}.items()))
        gen1 = [Generation(text="c1")]
        gen2 = [Generation(text="c2")]
        cache.update("p1", llm_string, gen1)
        cache.update("p2", llm_string, gen2)

        existing, _, missing_idxs, missing = get_prompts(
            {"model": "test"}, ["p1", "p2"], cache=cache
        )
        assert len(existing) == 2
        assert missing_idxs == []
        assert missing == []


# ===================================================================
# 10. update_cache
# ===================================================================


class TestUpdateCache:
    """Test update_cache function."""

    def test_update_cache_stores_results(self) -> None:
        cache = _FakeCache()
        llm_string = "test_llm"
        new_results = LLMResult(
            generations=[[Generation(text="r1")], [Generation(text="r2")]]
        )
        existing: dict[int, list] = {}
        llm_output = update_cache(
            cache=cache,
            existing_prompts=existing,
            llm_string=llm_string,
            missing_prompt_idxs=[0, 1],
            new_results=new_results,
            prompts=["p1", "p2"],
        )
        # Verify the cache was updated
        assert cache.lookup("p1", llm_string) is not None
        assert cache.lookup("p2", llm_string) is not None
        # Verify existing_prompts dict was updated
        assert 0 in existing
        assert 1 in existing
        assert existing[0][0].text == "r1"

    def test_update_cache_with_false_does_not_store(self) -> None:
        new_results = LLMResult(generations=[[Generation(text="r1")]])
        existing: dict[int, list] = {}
        llm_output = update_cache(
            cache=False,
            existing_prompts=existing,
            llm_string="llm",
            missing_prompt_idxs=[0],
            new_results=new_results,
            prompts=["p1"],
        )
        # existing_prompts should still be populated
        assert 0 in existing


# ===================================================================
# 11. create_base_retry_decorator
# ===================================================================


class TestCreateBaseRetryDecorator:
    """Test create_base_retry_decorator."""

    def test_retries_on_specified_error(self) -> None:
        call_count = 0

        decorator = create_base_retry_decorator(error_types=[ValueError], max_retries=3)

        @decorator
        def failing_fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient error")
            return "success"

        result = failing_fn()
        assert result == "success"
        assert call_count == 3

    def test_does_not_retry_on_unspecified_error(self) -> None:
        decorator = create_base_retry_decorator(error_types=[ValueError], max_retries=3)

        @decorator
        def failing_fn() -> str:
            raise TypeError("unrecoverable")

        with pytest.raises(TypeError, match="unrecoverable"):
            failing_fn()

    def test_max_retries_one_means_no_retry(self) -> None:
        call_count = 0
        decorator = create_base_retry_decorator(error_types=[ValueError], max_retries=1)

        @decorator
        def failing_fn() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            failing_fn()
        assert call_count == 1


# ===================================================================
# 12. BaseLLM.generate input validation
# ===================================================================


def test_generate_non_list_raises_value_error() -> None:
    llm = FakeListLLM(responses=["a"])
    with pytest.raises(ValueError, match="expected to be of type list"):
        llm.generate("not a list")  # type: ignore[arg-type]


# ===================================================================
# 13. BaseLLM._get_run_ids_list
# ===================================================================


class TestGetRunIdsList:
    """Test BaseLLM._get_run_ids_list static method."""

    def test_none_run_id(self) -> None:
        result = BaseLLM._get_run_ids_list(None, ["a", "b", "c"])
        assert result == [None, None, None]

    def test_single_uuid(self) -> None:
        uid = uuid.uuid4()
        result = BaseLLM._get_run_ids_list(uid, ["a", "b", "c"])
        assert result[0] == uid
        assert result[1] is None
        assert result[2] is None

    def test_list_of_uuids(self) -> None:
        uid1, uid2 = uuid.uuid4(), uuid.uuid4()
        result = BaseLLM._get_run_ids_list([uid1, uid2], ["a", "b"])
        assert result == [uid1, uid2]

    def test_mismatched_list_length_raises(self) -> None:
        uid1 = uuid.uuid4()
        with pytest.raises(ValueError, match="does not match batch length"):
            BaseLLM._get_run_ids_list([uid1], ["a", "b", "c"])

    def test_single_prompt_with_uuid(self) -> None:
        uid = uuid.uuid4()
        result = BaseLLM._get_run_ids_list(uid, ["a"])
        assert result == [uid]


# ===================================================================
# 14. BaseLLM.batch with empty inputs
# ===================================================================


def test_batch_empty_inputs_returns_empty_list() -> None:
    llm = FakeListLLM(responses=["a"])
    assert llm.batch([]) == []


# ===================================================================
# 15. BaseLLM.abatch with empty inputs
# ===================================================================


async def test_abatch_empty_inputs_returns_empty_list() -> None:
    llm = FakeListLLM(responses=["a"])
    assert await llm.abatch([]) == []


# ===================================================================
# 16. BaseLLM.batch with return_exceptions=True
# ===================================================================


def test_batch_return_exceptions_true() -> None:
    class _FailingLLM(LLM):
        @property
        def _llm_type(self) -> str:
            return "failing"

        @override
        def _call(
            self,
            prompt: str,
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> str:
            raise RuntimeError("boom")

    llm = _FailingLLM()
    results = llm.batch(["a", "b"], return_exceptions=True)
    assert len(results) == 2
    assert all(isinstance(r, RuntimeError) for r in results)


async def test_abatch_return_exceptions_true() -> None:
    class _AsyncFailingLLM(LLM):
        @property
        def _llm_type(self) -> str:
            return "async-failing"

        @override
        def _call(
            self,
            prompt: str,
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> str:
            raise RuntimeError("async boom")

    llm = _AsyncFailingLLM()
    results = await llm.abatch(["a", "b"], return_exceptions=True)
    assert len(results) == 2
    assert all(isinstance(r, RuntimeError) for r in results)


# ===================================================================
# 17. generate_prompt / agenerate_prompt
# ===================================================================


class TestGeneratePrompt:
    """Test generate_prompt and agenerate_prompt convert PromptValues."""

    def test_generate_prompt_converts_prompt_values(self) -> None:
        llm = FakeListLLM(responses=["resp1", "resp2"])
        prompts = [
            StringPromptValue(text="hello"),
            StringPromptValue(text="world"),
        ]
        result = llm.generate_prompt(prompts)
        assert len(result.generations) == 2
        assert result.generations[0][0].text == "resp1"
        assert result.generations[1][0].text == "resp2"

    async def test_agenerate_prompt_converts_prompt_values(self) -> None:
        llm = FakeListLLM(responses=["async_resp"])
        prompts = [StringPromptValue(text="hello")]
        result = await llm.agenerate_prompt(prompts)
        assert len(result.generations) == 1
        assert result.generations[0][0].text == "async_resp"

    def test_generate_prompt_with_chat_prompt_value(self) -> None:
        llm = FakeListLLM(responses=["chat_resp"])
        prompts = [ChatPromptValue(messages=[HumanMessage(content="hi there")])]
        result = llm.generate_prompt(prompts)
        assert len(result.generations) == 1
        assert result.generations[0][0].text == "chat_resp"
