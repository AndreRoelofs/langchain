"""Tests for langchain_core.language_models.model_profile module."""

from langchain_core.language_models.model_profile import (
    ModelProfile,
    ModelProfileRegistry,
)


class TestModelProfile:
    """Tests for ModelProfile TypedDict."""

    def test_model_profile_all_fields(self) -> None:
        """Test ModelProfile with all fields populated."""
        profile: ModelProfile = {
            # Input constraints
            "max_input_tokens": 128000,
            "image_inputs": True,
            "image_url_inputs": True,
            "pdf_inputs": True,
            "audio_inputs": True,
            "video_inputs": False,
            "image_tool_message": True,
            "pdf_tool_message": False,
            # Output constraints
            "max_output_tokens": 4096,
            "reasoning_output": True,
            "image_outputs": False,
            "audio_outputs": True,
            "video_outputs": False,
            # Tool calling
            "tool_calling": True,
            "tool_choice": True,
            # Structured output
            "structured_output": True,
        }

        assert profile["max_input_tokens"] == 128000
        assert profile["image_inputs"] is True
        assert profile["image_url_inputs"] is True
        assert profile["pdf_inputs"] is True
        assert profile["audio_inputs"] is True
        assert profile["video_inputs"] is False
        assert profile["image_tool_message"] is True
        assert profile["pdf_tool_message"] is False
        assert profile["max_output_tokens"] == 4096
        assert profile["reasoning_output"] is True
        assert profile["image_outputs"] is False
        assert profile["audio_outputs"] is True
        assert profile["video_outputs"] is False
        assert profile["tool_calling"] is True
        assert profile["tool_choice"] is True
        assert profile["structured_output"] is True

    def test_model_profile_partial_input_constraints(self) -> None:
        """Test ModelProfile with only input constraint fields."""
        profile: ModelProfile = {
            "max_input_tokens": 32000,
            "image_inputs": True,
            "audio_inputs": False,
        }

        assert profile["max_input_tokens"] == 32000
        assert profile["image_inputs"] is True
        assert profile["audio_inputs"] is False
        assert "max_output_tokens" not in profile
        assert "tool_calling" not in profile

    def test_model_profile_partial_output_constraints(self) -> None:
        """Test ModelProfile with only output constraint fields."""
        profile: ModelProfile = {
            "max_output_tokens": 8192,
            "reasoning_output": True,
            "image_outputs": True,
        }

        assert profile["max_output_tokens"] == 8192
        assert profile["reasoning_output"] is True
        assert profile["image_outputs"] is True
        assert "max_input_tokens" not in profile

    def test_model_profile_tool_calling_only(self) -> None:
        """Test ModelProfile with only tool calling fields."""
        profile: ModelProfile = {
            "tool_calling": True,
            "tool_choice": False,
        }

        assert profile["tool_calling"] is True
        assert profile["tool_choice"] is False
        assert "structured_output" not in profile

    def test_model_profile_structured_output_only(self) -> None:
        """Test ModelProfile with only structured output field."""
        profile: ModelProfile = {
            "structured_output": True,
        }

        assert profile["structured_output"] is True
        assert "tool_calling" not in profile

    def test_model_profile_empty(self) -> None:
        """Test ModelProfile with no fields (total=False allows this)."""
        profile: ModelProfile = {}
        assert len(profile) == 0

    def test_model_profile_gpt4_like(self) -> None:
        """Test ModelProfile representing a GPT-4 like model."""
        profile: ModelProfile = {
            "max_input_tokens": 128000,
            "max_output_tokens": 4096,
            "image_inputs": True,
            "image_url_inputs": True,
            "pdf_inputs": False,
            "audio_inputs": True,
            "video_inputs": False,
            "tool_calling": True,
            "tool_choice": True,
            "structured_output": True,
            "reasoning_output": False,
        }

        assert profile["max_input_tokens"] == 128000
        assert profile["tool_calling"] is True
        assert profile["structured_output"] is True

    def test_model_profile_claude_like(self) -> None:
        """Test ModelProfile representing a Claude-like model."""
        profile: ModelProfile = {
            "max_input_tokens": 200000,
            "max_output_tokens": 8192,
            "image_inputs": True,
            "image_url_inputs": True,
            "pdf_inputs": True,
            "audio_inputs": False,
            "video_inputs": False,
            "tool_calling": True,
            "tool_choice": True,
            "structured_output": True,
            "reasoning_output": True,
        }

        assert profile["max_input_tokens"] == 200000
        assert profile["pdf_inputs"] is True
        assert profile["reasoning_output"] is True

    def test_model_profile_basic_llm(self) -> None:
        """Test ModelProfile representing a basic text-only LLM."""
        profile: ModelProfile = {
            "max_input_tokens": 4096,
            "max_output_tokens": 2048,
            "image_inputs": False,
            "audio_inputs": False,
            "video_inputs": False,
            "tool_calling": False,
            "structured_output": False,
        }

        assert profile["max_input_tokens"] == 4096
        assert profile["image_inputs"] is False
        assert profile["tool_calling"] is False

    def test_model_profile_multimodal_output(self) -> None:
        """Test ModelProfile for a model with multimodal outputs."""
        profile: ModelProfile = {
            "max_input_tokens": 32000,
            "max_output_tokens": 4096,
            "image_outputs": True,
            "audio_outputs": True,
            "video_outputs": True,
        }

        assert profile["image_outputs"] is True
        assert profile["audio_outputs"] is True
        assert profile["video_outputs"] is True


class TestModelProfileRegistry:
    """Tests for ModelProfileRegistry type alias."""

    def test_model_profile_registry_empty(self) -> None:
        """Test empty ModelProfileRegistry."""
        registry: ModelProfileRegistry = {}
        assert len(registry) == 0

    def test_model_profile_registry_single_model(self) -> None:
        """Test ModelProfileRegistry with single model."""
        registry: ModelProfileRegistry = {
            "gpt-4": {
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
                "tool_calling": True,
            }
        }

        assert "gpt-4" in registry
        assert registry["gpt-4"]["max_input_tokens"] == 128000
        assert registry["gpt-4"]["tool_calling"] is True

    def test_model_profile_registry_multiple_models(self) -> None:
        """Test ModelProfileRegistry with multiple models."""
        registry: ModelProfileRegistry = {
            "gpt-4": {
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
                "tool_calling": True,
            },
            "gpt-3.5-turbo": {
                "max_input_tokens": 16385,
                "max_output_tokens": 4096,
                "tool_calling": True,
            },
            "claude-3-opus": {
                "max_input_tokens": 200000,
                "max_output_tokens": 4096,
                "tool_calling": True,
                "pdf_inputs": True,
            },
        }

        assert len(registry) == 3
        assert registry["gpt-4"]["max_input_tokens"] == 128000
        assert registry["gpt-3.5-turbo"]["max_input_tokens"] == 16385
        assert registry["claude-3-opus"]["pdf_inputs"] is True

    def test_model_profile_registry_lookup(self) -> None:
        """Test looking up models in registry."""
        registry: ModelProfileRegistry = {
            "model-a": {"max_input_tokens": 1000},
            "model-b": {"max_input_tokens": 2000},
        }

        # Test get with default
        profile = registry.get("model-a")
        assert profile is not None
        assert profile["max_input_tokens"] == 1000

        # Test get with missing key
        missing = registry.get("model-c")
        assert missing is None

        # Test get with default value
        default_profile: ModelProfile = {"max_input_tokens": 0}
        result = registry.get("model-c", default_profile)
        assert result["max_input_tokens"] == 0

    def test_model_profile_registry_iteration(self) -> None:
        """Test iterating over registry."""
        registry: ModelProfileRegistry = {
            "model-1": {"max_input_tokens": 1000},
            "model-2": {"max_input_tokens": 2000},
            "model-3": {"max_input_tokens": 3000},
        }

        model_names = list(registry.keys())
        assert len(model_names) == 3
        assert "model-1" in model_names
        assert "model-2" in model_names
        assert "model-3" in model_names

        profiles = list(registry.values())
        assert len(profiles) == 3

        items = list(registry.items())
        assert len(items) == 3

    def test_model_profile_registry_update(self) -> None:
        """Test updating registry."""
        registry: ModelProfileRegistry = {
            "model-a": {"max_input_tokens": 1000},
        }

        # Add new model
        registry["model-b"] = {"max_input_tokens": 2000, "tool_calling": True}
        assert len(registry) == 2
        assert registry["model-b"]["tool_calling"] is True

        # Update existing model
        registry["model-a"] = {"max_input_tokens": 1500, "image_inputs": True}
        assert registry["model-a"]["max_input_tokens"] == 1500
        assert registry["model-a"]["image_inputs"] is True

    def test_model_profile_registry_delete(self) -> None:
        """Test deleting from registry."""
        registry: ModelProfileRegistry = {
            "model-a": {"max_input_tokens": 1000},
            "model-b": {"max_input_tokens": 2000},
        }

        del registry["model-a"]
        assert len(registry) == 1
        assert "model-a" not in registry
        assert "model-b" in registry

    def test_model_profile_registry_with_version_suffixes(self) -> None:
        """Test registry with versioned model names."""
        registry: ModelProfileRegistry = {
            "gpt-4-0613": {"max_input_tokens": 8192},
            "gpt-4-1106-preview": {"max_input_tokens": 128000},
            "gpt-4-turbo-2024-04-09": {"max_input_tokens": 128000},
        }

        assert registry["gpt-4-0613"]["max_input_tokens"] == 8192
        assert registry["gpt-4-1106-preview"]["max_input_tokens"] == 128000
        assert registry["gpt-4-turbo-2024-04-09"]["max_input_tokens"] == 128000

    def test_model_profile_registry_provider_namespaced(self) -> None:
        """Test registry with provider-namespaced model names."""
        registry: ModelProfileRegistry = {
            "openai/gpt-4": {"max_input_tokens": 128000, "tool_calling": True},
            "anthropic/claude-3-opus": {
                "max_input_tokens": 200000,
                "tool_calling": True,
            },
            "google/gemini-pro": {"max_input_tokens": 32000, "tool_calling": True},
        }

        assert "openai/gpt-4" in registry
        assert "anthropic/claude-3-opus" in registry
        assert "google/gemini-pro" in registry
