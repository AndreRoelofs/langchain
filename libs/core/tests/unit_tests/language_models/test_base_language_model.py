"""Tests for langchain_core.language_models.base module."""

import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from langchain_core.callbacks import Callbacks
from langchain_core.language_models.base import (
    BaseLanguageModel,
    LangSmithParams,
    LanguageModelInput,
    LanguageModelLike,
    LanguageModelOutput,
    _get_token_ids_default_method,
    _get_verbosity,
    get_tokenizer,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import PromptValue, StringPromptValue


class ConcreteLanguageModel(BaseLanguageModel[str]):
    """Concrete implementation of BaseLanguageModel for testing."""

    def invoke(
        self,
        input: LanguageModelInput,
        config: Any = None,
        **kwargs: Any,
    ) -> str:
        """Invoke the model."""
        return "test response"

    def generate_prompt(
        self,
        prompts: list[PromptValue],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate from prompts."""
        from langchain_core.outputs import Generation

        generations = [[Generation(text="test response")] for _ in prompts]
        return LLMResult(generations=generations)

    async def agenerate_prompt(
        self,
        prompts: list[PromptValue],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async generate from prompts."""
        return self.generate_prompt(prompts, stop, callbacks, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "test-llm"


class TestLangSmithParams:
    """Tests for LangSmithParams TypedDict."""

    def test_langsmith_params_all_fields(self) -> None:
        """Test LangSmithParams with all fields."""
        params: LangSmithParams = {
            "ls_provider": "openai",
            "ls_model_name": "gpt-4",
            "ls_model_type": "chat",
            "ls_temperature": 0.7,
            "ls_max_tokens": 1000,
            "ls_stop": ["stop1", "stop2"],
        }
        assert params["ls_provider"] == "openai"
        assert params["ls_model_name"] == "gpt-4"
        assert params["ls_model_type"] == "chat"
        assert params["ls_temperature"] == 0.7
        assert params["ls_max_tokens"] == 1000
        assert params["ls_stop"] == ["stop1", "stop2"]

    def test_langsmith_params_partial(self) -> None:
        """Test LangSmithParams with partial fields."""
        params: LangSmithParams = {
            "ls_provider": "anthropic",
            "ls_model_type": "chat",
        }
        assert params["ls_provider"] == "anthropic"
        assert params["ls_model_type"] == "chat"
        assert "ls_model_name" not in params

    def test_langsmith_params_empty(self) -> None:
        """Test LangSmithParams with no fields."""
        params: LangSmithParams = {}
        assert len(params) == 0

    def test_langsmith_params_model_type_values(self) -> None:
        """Test LangSmithParams model_type accepts valid values."""
        chat_params: LangSmithParams = {"ls_model_type": "chat"}
        llm_params: LangSmithParams = {"ls_model_type": "llm"}
        assert chat_params["ls_model_type"] == "chat"
        assert llm_params["ls_model_type"] == "llm"


class TestGetTokenizer:
    """Tests for get_tokenizer function."""

    def test_get_tokenizer_without_transformers(self) -> None:
        """Test get_tokenizer raises ImportError when transformers not installed."""
        with patch("langchain_core.language_models.base._HAS_TRANSFORMERS", False):
            # Clear the cache to force re-evaluation
            get_tokenizer.cache_clear()
            with pytest.raises(ImportError) as exc_info:
                get_tokenizer()
            assert "transformers" in str(exc_info.value)
            assert "pip install transformers" in str(exc_info.value)

    def test_get_tokenizer_with_transformers(self) -> None:
        """Test get_tokenizer returns tokenizer when transformers installed."""
        # Skip this test if transformers is not installed
        try:
            from transformers import GPT2TokenizerFast  # noqa: F401
        except ImportError:
            pytest.skip("transformers not installed")

        get_tokenizer.cache_clear()
        result = get_tokenizer()
        # Should return a tokenizer instance
        assert result is not None
        assert hasattr(result, "encode")

    def test_get_tokenizer_is_cached(self) -> None:
        """Test get_tokenizer caches the result."""
        # Skip this test if transformers is not installed
        try:
            from transformers import GPT2TokenizerFast  # noqa: F401
        except ImportError:
            pytest.skip("transformers not installed")

        get_tokenizer.cache_clear()
        result1 = get_tokenizer()
        result2 = get_tokenizer()
        # Should return the same cached instance
        assert result1 is result2


class TestGetTokenIdsDefaultMethod:
    """Tests for _get_token_ids_default_method function."""

    def test_get_token_ids_default_method(self) -> None:
        """Test _get_token_ids_default_method encodes text."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        with patch(
            "langchain_core.language_models.base.get_tokenizer",
            return_value=mock_tokenizer,
        ):
            result = _get_token_ids_default_method("hello world")
            mock_tokenizer.encode.assert_called_once_with("hello world")
            assert result == [1, 2, 3, 4, 5]


class TestGetVerbosity:
    """Tests for _get_verbosity function."""

    def test_get_verbosity_returns_global_verbose(self) -> None:
        """Test _get_verbosity returns global verbose setting."""
        with patch(
            "langchain_core.language_models.base.get_verbose", return_value=True
        ):
            assert _get_verbosity() is True

        with patch(
            "langchain_core.language_models.base.get_verbose", return_value=False
        ):
            assert _get_verbosity() is False


class TestBaseLanguageModel:
    """Tests for BaseLanguageModel class."""

    def test_initialization_defaults(self) -> None:
        """Test BaseLanguageModel initializes with defaults."""
        model = ConcreteLanguageModel()
        assert model.cache is None
        assert model.callbacks is None
        assert model.tags is None
        assert model.metadata is None
        assert model.custom_get_token_ids is None

    def test_initialization_with_cache_true(self) -> None:
        """Test BaseLanguageModel with cache=True."""
        model = ConcreteLanguageModel(cache=True)
        assert model.cache is True

    def test_initialization_with_cache_false(self) -> None:
        """Test BaseLanguageModel with cache=False."""
        model = ConcreteLanguageModel(cache=False)
        assert model.cache is False

    def test_initialization_with_verbose(self) -> None:
        """Test BaseLanguageModel with verbose setting."""
        model = ConcreteLanguageModel(verbose=True)
        assert model.verbose is True

        model = ConcreteLanguageModel(verbose=False)
        assert model.verbose is False

    def test_verbose_validator_with_none(self) -> None:
        """Test verbose validator converts None to global setting."""
        with patch(
            "langchain_core.language_models.base._get_verbosity", return_value=True
        ):
            model = ConcreteLanguageModel(verbose=None)
            assert model.verbose is True

    def test_initialization_with_tags(self) -> None:
        """Test BaseLanguageModel with tags."""
        model = ConcreteLanguageModel(tags=["tag1", "tag2"])
        assert model.tags == ["tag1", "tag2"]

    def test_initialization_with_metadata(self) -> None:
        """Test BaseLanguageModel with metadata."""
        model = ConcreteLanguageModel(metadata={"key": "value"})
        assert model.metadata == {"key": "value"}

    def test_initialization_with_callbacks(self) -> None:
        """Test BaseLanguageModel with callbacks."""
        from langchain_core.callbacks import BaseCallbackHandler

        class TestHandler(BaseCallbackHandler):
            pass

        callbacks = [TestHandler()]
        model = ConcreteLanguageModel(callbacks=callbacks)
        assert model.callbacks == callbacks

    def test_custom_get_token_ids(self) -> None:
        """Test BaseLanguageModel with custom_get_token_ids."""

        def custom_tokenizer(text: str) -> list[int]:
            return [ord(c) for c in text]

        model = ConcreteLanguageModel(custom_get_token_ids=custom_tokenizer)
        assert model.custom_get_token_ids is custom_tokenizer

    def test_get_token_ids_with_custom_tokenizer(self) -> None:
        """Test get_token_ids uses custom tokenizer when provided."""

        def custom_tokenizer(text: str) -> list[int]:
            return [1, 2, 3]

        model = ConcreteLanguageModel(custom_get_token_ids=custom_tokenizer)
        result = model.get_token_ids("any text")
        assert result == [1, 2, 3]

    def test_get_token_ids_with_default_tokenizer(self) -> None:
        """Test get_token_ids uses default tokenizer when no custom provided."""
        model = ConcreteLanguageModel()
        with patch(
            "langchain_core.language_models.base._get_token_ids_default_method",
            return_value=[10, 20, 30],
        ):
            result = model.get_token_ids("test")
            assert result == [10, 20, 30]

    def test_get_num_tokens(self) -> None:
        """Test get_num_tokens returns length of token ids."""

        def custom_tokenizer(text: str) -> list[int]:
            return [1, 2, 3, 4, 5]

        model = ConcreteLanguageModel(custom_get_token_ids=custom_tokenizer)
        result = model.get_num_tokens("test text")
        assert result == 5

    def test_get_num_tokens_from_messages(self) -> None:
        """Test get_num_tokens_from_messages sums tokens from all messages."""

        def custom_tokenizer(text: str) -> list[int]:
            # Return 2 tokens per character for testing
            return list(range(len(text) * 2))

        model = ConcreteLanguageModel(custom_get_token_ids=custom_tokenizer)
        messages = [
            HumanMessage(content="Hi"),
            AIMessage(content="Hello"),
        ]
        result = model.get_num_tokens_from_messages(messages)
        # Result depends on get_buffer_string formatting
        assert result > 0

    def test_get_num_tokens_from_messages_with_tools_warning(self) -> None:
        """Test get_num_tokens_from_messages warns when tools provided."""

        def custom_tokenizer(text: str) -> list[int]:
            return [1]

        model = ConcreteLanguageModel(custom_get_token_ids=custom_tokenizer)
        messages = [HumanMessage(content="Hi")]
        tools = [{"name": "test_tool"}]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.get_num_tokens_from_messages(messages, tools=tools)
            assert len(w) == 1
            assert "tool schemas" in str(w[0].message).lower()

    def test_identifying_params(self) -> None:
        """Test _identifying_params returns lc_attributes."""
        model = ConcreteLanguageModel()
        params = model._identifying_params
        assert isinstance(params, dict)

    def test_input_type(self) -> None:
        """Test InputType property returns correct type."""
        model = ConcreteLanguageModel()
        input_type = model.InputType
        # Should be a union type including str, PromptValue, and message sequences
        assert input_type is not None

    def test_with_structured_output_not_implemented(self) -> None:
        """Test with_structured_output raises NotImplementedError by default."""
        model = ConcreteLanguageModel()

        class TestSchema(BaseModel):
            name: str

        with pytest.raises(NotImplementedError):
            model.with_structured_output(TestSchema)


class TestLanguageModelTypeAliases:
    """Tests for type aliases."""

    def test_language_model_input_accepts_string(self) -> None:
        """Test LanguageModelInput accepts string."""
        input_val: LanguageModelInput = "test string"
        assert input_val == "test string"

    def test_language_model_input_accepts_prompt_value(self) -> None:
        """Test LanguageModelInput accepts PromptValue."""
        prompt = StringPromptValue(text="test")
        input_val: LanguageModelInput = prompt
        assert input_val == prompt

    def test_language_model_input_accepts_message_sequence(self) -> None:
        """Test LanguageModelInput accepts message sequence."""
        messages = [HumanMessage(content="Hello")]
        input_val: LanguageModelInput = messages
        assert input_val == messages

    def test_language_model_output_accepts_string(self) -> None:
        """Test LanguageModelOutput accepts string."""
        output: LanguageModelOutput = "test output"
        assert output == "test output"

    def test_language_model_output_accepts_base_message(self) -> None:
        """Test LanguageModelOutput accepts BaseMessage."""
        message = AIMessage(content="test")
        output: LanguageModelOutput = message
        assert output == message


class TestBaseLanguageModelSerialization:
    """Tests for BaseLanguageModel serialization."""

    def test_model_config_allows_arbitrary_types(self) -> None:
        """Test model_config allows arbitrary types."""
        # This should not raise an error
        model = ConcreteLanguageModel()
        assert model.model_config.get("arbitrary_types_allowed") is True

    def test_cache_excluded_from_serialization(self) -> None:
        """Test cache field is excluded from serialization."""
        model = ConcreteLanguageModel(cache=True)
        model_dict = model.model_dump()
        assert "cache" not in model_dict

    def test_verbose_excluded_from_serialization(self) -> None:
        """Test verbose field is excluded from serialization."""
        model = ConcreteLanguageModel(verbose=True)
        model_dict = model.model_dump()
        assert "verbose" not in model_dict

    def test_callbacks_excluded_from_serialization(self) -> None:
        """Test callbacks field is excluded from serialization."""
        from langchain_core.callbacks import BaseCallbackHandler

        class TestHandler(BaseCallbackHandler):
            pass

        model = ConcreteLanguageModel(callbacks=[TestHandler()])
        model_dict = model.model_dump()
        assert "callbacks" not in model_dict

    def test_tags_excluded_from_serialization(self) -> None:
        """Test tags field is excluded from serialization."""
        model = ConcreteLanguageModel(tags=["tag1"])
        model_dict = model.model_dump()
        assert "tags" not in model_dict

    def test_metadata_excluded_from_serialization(self) -> None:
        """Test metadata field is excluded from serialization."""
        model = ConcreteLanguageModel(metadata={"key": "value"})
        model_dict = model.model_dump()
        assert "metadata" not in model_dict

    def test_custom_get_token_ids_excluded_from_serialization(self) -> None:
        """Test custom_get_token_ids field is excluded from serialization."""
        model = ConcreteLanguageModel(custom_get_token_ids=lambda x: [1])
        model_dict = model.model_dump()
        assert "custom_get_token_ids" not in model_dict


class TestGetTokenIdsDefaultFallback:
    """Tests for get_token_ids falling back to _get_token_ids_default_method."""

    def test_get_token_ids_calls_default_method_when_no_custom(self) -> None:
        """Test get_token_ids calls _get_token_ids_default_method when no custom
        tokenizer is set."""
        model = ConcreteLanguageModel()
        assert model.custom_get_token_ids is None

        with patch(
            "langchain_core.language_models.base._get_token_ids_default_method",
            return_value=[100, 200, 300],
        ) as mock_default:
            result = model.get_token_ids("hello world")
            mock_default.assert_called_once_with("hello world")
            assert result == [100, 200, 300]

    def test_get_token_ids_does_not_call_default_when_custom_set(self) -> None:
        """Test get_token_ids does not call _get_token_ids_default_method when
        custom tokenizer is set."""

        def custom_fn(text: str) -> list[int]:
            return [42]

        model = ConcreteLanguageModel(custom_get_token_ids=custom_fn)

        with patch(
            "langchain_core.language_models.base._get_token_ids_default_method",
        ) as mock_default:
            result = model.get_token_ids("test")
            mock_default.assert_not_called()
            assert result == [42]


class TestGetNumTokensEdgeCases:
    """Tests for get_num_tokens edge cases."""

    def test_get_num_tokens_empty_string(self) -> None:
        """Test get_num_tokens returns 0 for empty string."""

        def custom_tokenizer(text: str) -> list[int]:
            if text == "":
                return []
            return [1, 2, 3]

        model = ConcreteLanguageModel(custom_get_token_ids=custom_tokenizer)
        result = model.get_num_tokens("")
        assert result == 0

    def test_get_num_tokens_whitespace_only(self) -> None:
        """Test get_num_tokens with whitespace-only string."""

        def custom_tokenizer(text: str) -> list[int]:
            return [1] if text.strip() == "" and len(text) > 0 else [1, 2]

        model = ConcreteLanguageModel(custom_get_token_ids=custom_tokenizer)
        result = model.get_num_tokens(" ")
        assert result == 1

    def test_get_num_tokens_single_token(self) -> None:
        """Test get_num_tokens with single-token text."""

        def custom_tokenizer(text: str) -> list[int]:
            return [99]

        model = ConcreteLanguageModel(custom_get_token_ids=custom_tokenizer)
        result = model.get_num_tokens("a")
        assert result == 1


class TestGetNumTokensFromMessagesNoTools:
    """Tests for get_num_tokens_from_messages without tools."""

    def test_no_warning_when_tools_not_provided(self) -> None:
        """Test that no warning is emitted when tools parameter is not provided."""

        def custom_tokenizer(text: str) -> list[int]:
            return [1]

        model = ConcreteLanguageModel(custom_get_token_ids=custom_tokenizer)
        messages = [HumanMessage(content="Hello")]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.get_num_tokens_from_messages(messages)
            tool_warnings = [x for x in w if "tool schemas" in str(x.message).lower()]
            assert len(tool_warnings) == 0

    def test_no_warning_when_tools_is_none(self) -> None:
        """Test that no warning is emitted when tools is explicitly None."""

        def custom_tokenizer(text: str) -> list[int]:
            return [1]

        model = ConcreteLanguageModel(custom_get_token_ids=custom_tokenizer)
        messages = [HumanMessage(content="Hello")]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.get_num_tokens_from_messages(messages, tools=None)
            tool_warnings = [x for x in w if "tool schemas" in str(x.message).lower()]
            assert len(tool_warnings) == 0


class TestGetNumTokensFromMessagesEdgeCases:
    """Tests for get_num_tokens_from_messages with edge cases and multiple messages."""

    def test_empty_messages_list(self) -> None:
        """Test get_num_tokens_from_messages returns 0 for empty list."""

        def custom_tokenizer(text: str) -> list[int]:
            return [1, 2, 3]

        model = ConcreteLanguageModel(custom_get_token_ids=custom_tokenizer)
        result = model.get_num_tokens_from_messages([])
        assert result == 0

    def test_multiple_messages_sums_tokens(self) -> None:
        """Test get_num_tokens_from_messages sums tokens from multiple messages."""
        call_args: list[str] = []

        def custom_tokenizer(text: str) -> list[int]:
            call_args.append(text)
            # Return different number of tokens based on text length
            return list(range(len(text)))

        model = ConcreteLanguageModel(custom_get_token_ids=custom_tokenizer)
        messages = [
            HumanMessage(content="Hi"),
            AIMessage(content="Hello there"),
            HumanMessage(content="Bye"),
        ]
        result = model.get_num_tokens_from_messages(messages)
        # Each message is formatted via get_buffer_string individually,
        # so result should be the sum of all individual token counts.
        assert result > 0
        # Verify the tokenizer was called once per message
        assert len(call_args) == 3

    def test_single_message_returns_correct_count(self) -> None:
        """Test get_num_tokens_from_messages with a single message."""

        def custom_tokenizer(text: str) -> list[int]:
            return [1, 2, 3, 4, 5]

        model = ConcreteLanguageModel(custom_get_token_ids=custom_tokenizer)
        messages = [HumanMessage(content="Hello world")]
        result = model.get_num_tokens_from_messages(messages)
        assert result == 5


class TestGeneratePrompt:
    """Tests for generate_prompt via concrete implementation."""

    def test_generate_prompt_single_prompt(self) -> None:
        """Test generate_prompt with a single prompt returns LLMResult."""
        model = ConcreteLanguageModel()
        prompts = [StringPromptValue(text="Hello")]
        result = model.generate_prompt(prompts)

        assert isinstance(result, LLMResult)
        assert len(result.generations) == 1
        assert result.generations[0][0].text == "test response"

    def test_generate_prompt_multiple_prompts(self) -> None:
        """Test generate_prompt with multiple prompts returns matching generations."""
        model = ConcreteLanguageModel()
        prompts = [
            StringPromptValue(text="Prompt 1"),
            StringPromptValue(text="Prompt 2"),
            StringPromptValue(text="Prompt 3"),
        ]
        result = model.generate_prompt(prompts)

        assert isinstance(result, LLMResult)
        assert len(result.generations) == 3
        for gen_list in result.generations:
            assert len(gen_list) == 1
            assert gen_list[0].text == "test response"

    def test_generate_prompt_empty_prompts(self) -> None:
        """Test generate_prompt with empty prompt list."""
        model = ConcreteLanguageModel()
        result = model.generate_prompt([])

        assert isinstance(result, LLMResult)
        assert len(result.generations) == 0


class TestAGeneratePrompt:
    """Tests for agenerate_prompt via concrete implementation."""

    @pytest.mark.asyncio
    async def test_agenerate_prompt_single_prompt(self) -> None:
        """Test agenerate_prompt with a single prompt returns LLMResult."""
        model = ConcreteLanguageModel()
        prompts = [StringPromptValue(text="Hello")]
        result = await model.agenerate_prompt(prompts)

        assert isinstance(result, LLMResult)
        assert len(result.generations) == 1
        assert result.generations[0][0].text == "test response"

    @pytest.mark.asyncio
    async def test_agenerate_prompt_multiple_prompts(self) -> None:
        """Test agenerate_prompt with multiple prompts returns matching
        generations."""
        model = ConcreteLanguageModel()
        prompts = [
            StringPromptValue(text="Prompt 1"),
            StringPromptValue(text="Prompt 2"),
        ]
        result = await model.agenerate_prompt(prompts)

        assert isinstance(result, LLMResult)
        assert len(result.generations) == 2

    @pytest.mark.asyncio
    async def test_agenerate_prompt_empty_prompts(self) -> None:
        """Test agenerate_prompt with empty prompt list."""
        model = ConcreteLanguageModel()
        result = await model.agenerate_prompt([])

        assert isinstance(result, LLMResult)
        assert len(result.generations) == 0


class TestIdentifyingParams:
    """Tests for _identifying_params property."""

    def test_identifying_params_returns_mapping(self) -> None:
        """Test _identifying_params returns a Mapping type."""
        from collections.abc import Mapping

        model = ConcreteLanguageModel()
        params = model._identifying_params
        assert isinstance(params, Mapping)

    def test_identifying_params_returns_lc_attributes(self) -> None:
        """Test _identifying_params returns the same value as lc_attributes."""
        model = ConcreteLanguageModel()
        assert model._identifying_params == model.lc_attributes

    def test_identifying_params_is_dict(self) -> None:
        """Test _identifying_params is a dict for the base implementation."""
        model = ConcreteLanguageModel()
        params = model._identifying_params
        assert isinstance(params, dict)
        # Default lc_attributes is an empty dict
        assert params == {}


class TestInputTypeProperty:
    """Tests for InputType property."""

    def test_input_type_is_not_none(self) -> None:
        """Test InputType property returns a non-None type."""
        model = ConcreteLanguageModel()
        input_type = model.InputType
        assert input_type is not None

    def test_input_type_includes_str(self) -> None:
        """Test InputType includes str as a valid option."""
        from typing import get_args

        model = ConcreteLanguageModel()
        input_type = model.InputType
        type_args = get_args(input_type)
        assert str in type_args

    def test_input_type_includes_prompt_values(self) -> None:
        """Test InputType includes StringPromptValue and ChatPromptValueConcrete."""
        from typing import get_args

        from langchain_core.prompt_values import (
            ChatPromptValueConcrete,
            StringPromptValue,
        )

        model = ConcreteLanguageModel()
        input_type = model.InputType
        type_args = get_args(input_type)
        assert StringPromptValue in type_args
        assert ChatPromptValueConcrete in type_args

    def test_input_type_includes_message_list(self) -> None:
        """Test InputType includes list[AnyMessage]."""
        from typing import get_args, get_origin

        model = ConcreteLanguageModel()
        input_type = model.InputType
        type_args = get_args(input_type)
        # One of the type_args should be list[AnyMessage]
        list_types = [t for t in type_args if get_origin(t) is list]
        assert len(list_types) == 1


class TestSetVerboseValidator:
    """Tests for the set_verbose field validator."""

    def test_verbose_true_stays_true(self) -> None:
        """Test verbose=True remains True regardless of global setting."""
        with patch(
            "langchain_core.language_models.base._get_verbosity", return_value=False
        ):
            model = ConcreteLanguageModel(verbose=True)
            assert model.verbose is True

    def test_verbose_false_stays_false(self) -> None:
        """Test verbose=False remains False regardless of global setting."""
        with patch(
            "langchain_core.language_models.base._get_verbosity", return_value=True
        ):
            model = ConcreteLanguageModel(verbose=False)
            assert model.verbose is False

    def test_verbose_none_uses_global_true(self) -> None:
        """Test verbose=None resolves to True when global is True."""
        with patch(
            "langchain_core.language_models.base._get_verbosity", return_value=True
        ):
            model = ConcreteLanguageModel(verbose=None)
            assert model.verbose is True

    def test_verbose_none_uses_global_false(self) -> None:
        """Test verbose=None resolves to False when global is False."""
        with patch(
            "langchain_core.language_models.base._get_verbosity", return_value=False
        ):
            model = ConcreteLanguageModel(verbose=None)
            assert model.verbose is False

    def test_verbose_default_uses_global_setting(self) -> None:
        """Test verbose default_factory uses global verbosity setting.

        The default_factory holds a direct reference to _get_verbosity, so we
        must patch the function it delegates to (get_verbose) rather than
        _get_verbosity itself.
        """
        with patch(
            "langchain_core.language_models.base.get_verbose", return_value=True
        ):
            model = ConcreteLanguageModel()
            assert model.verbose is True

        with patch(
            "langchain_core.language_models.base.get_verbose", return_value=False
        ):
            model = ConcreteLanguageModel()
            assert model.verbose is False


class TestModelDumpExclusions:
    """Tests for model_dump and model_dump_json excluding expected fields."""

    EXCLUDED_FIELDS = [
        "cache",
        "verbose",
        "callbacks",
        "tags",
        "metadata",
        "custom_get_token_ids",
    ]

    def test_model_dump_excludes_all_private_fields(self) -> None:
        """Test model_dump excludes cache, verbose, callbacks, tags, metadata,
        and custom_get_token_ids."""
        model = ConcreteLanguageModel(
            cache=True,
            verbose=True,
            tags=["tag1"],
            metadata={"key": "value"},
            custom_get_token_ids=lambda x: [1],
        )
        model_dict = model.model_dump()
        for field in self.EXCLUDED_FIELDS:
            assert field not in model_dict, (
                f"{field} should be excluded from model_dump"
            )

    def test_model_dump_json_excludes_all_private_fields(self) -> None:
        """Test model_dump_json excludes the same fields as model_dump."""
        import json

        model = ConcreteLanguageModel(
            cache=True,
            verbose=True,
            tags=["tag1"],
            metadata={"key": "value"},
        )
        json_str = model.model_dump_json()
        parsed = json.loads(json_str)
        for field in self.EXCLUDED_FIELDS:
            assert field not in parsed, (
                f"{field} should be excluded from model_dump_json"
            )


class TestCacheWithBaseCacheInstance:
    """Tests for initialization with a BaseCache instance."""

    def test_cache_with_base_cache_instance(self) -> None:
        """Test initializing with a BaseCache instance."""
        from langchain_core.caches import BaseCache

        mock_cache = MagicMock(spec=BaseCache)
        model = ConcreteLanguageModel(cache=mock_cache)
        assert model.cache is mock_cache

    def test_cache_instance_is_not_bool(self) -> None:
        """Test that a BaseCache instance is distinct from bool cache values."""
        from langchain_core.caches import BaseCache

        mock_cache = MagicMock(spec=BaseCache)
        model = ConcreteLanguageModel(cache=mock_cache)
        assert model.cache is not True
        assert model.cache is not False
        assert model.cache is not None

    def test_cache_instance_excluded_from_serialization(self) -> None:
        """Test that a BaseCache instance is excluded from model_dump."""
        from langchain_core.caches import BaseCache

        mock_cache = MagicMock(spec=BaseCache)
        model = ConcreteLanguageModel(cache=mock_cache)
        model_dict = model.model_dump()
        assert "cache" not in model_dict


class TestWithStructuredOutputDictSchema:
    """Tests for with_structured_output with dict schema."""

    def test_with_structured_output_dict_raises_not_implemented(self) -> None:
        """Test with_structured_output raises NotImplementedError for dict schema."""
        model = ConcreteLanguageModel()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        with pytest.raises(NotImplementedError):
            model.with_structured_output(schema)

    def test_with_structured_output_pydantic_raises_not_implemented(self) -> None:
        """Test with_structured_output raises NotImplementedError for Pydantic model."""
        model = ConcreteLanguageModel()

        class TestSchema(BaseModel):
            name: str
            age: int

        with pytest.raises(NotImplementedError):
            model.with_structured_output(TestSchema)

    def test_with_structured_output_with_kwargs_raises_not_implemented(self) -> None:
        """Test with_structured_output raises NotImplementedError even with kwargs."""
        model = ConcreteLanguageModel()
        schema = {"type": "object"}
        with pytest.raises(NotImplementedError):
            model.with_structured_output(schema, method="json_mode")
