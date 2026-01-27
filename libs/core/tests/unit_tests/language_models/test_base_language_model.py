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
        with patch(
            "langchain_core.language_models.base._HAS_TRANSFORMERS", False
        ):
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
