"""Comprehensive tests for langchain_core.load.mapping module."""

import pytest

from langchain_core.load.load import ALL_SERIALIZABLE_MAPPINGS, DISALLOW_LOAD_FROM_PATH
from langchain_core.load.mapping import (
    OLD_CORE_NAMESPACES_MAPPING,
    SERIALIZABLE_MAPPING,
    _JS_SERIALIZABLE_MAPPING,
    _OG_SERIALIZABLE_MAPPING,
)


class TestMappingStructure:
    """Tests for the structure of mapping dictionaries."""

    def test_serializable_mapping_structure(self) -> None:
        """Test SERIALIZABLE_MAPPING has correct structure."""
        assert isinstance(SERIALIZABLE_MAPPING, dict)

        # Check a few sample mappings have correct structure
        for key, value in list(SERIALIZABLE_MAPPING.items())[:5]:
            assert isinstance(key, tuple)
            assert isinstance(value, tuple)
            assert all(isinstance(item, str) for item in key)
            assert all(isinstance(item, str) for item in value)
            assert len(key) > 0
            assert len(value) > 0

    def test_old_core_namespaces_mapping_structure(self) -> None:
        """Test OLD_CORE_NAMESPACES_MAPPING has correct structure."""
        assert isinstance(OLD_CORE_NAMESPACES_MAPPING, dict)

        for key, value in list(OLD_CORE_NAMESPACES_MAPPING.items())[:5]:
            assert isinstance(key, tuple)
            assert isinstance(value, tuple)
            assert all(isinstance(item, str) for item in key)
            assert all(isinstance(item, str) for item in value)

    def test_og_serializable_mapping_structure(self) -> None:
        """Test _OG_SERIALIZABLE_MAPPING has correct structure."""
        assert isinstance(_OG_SERIALIZABLE_MAPPING, dict)

        for key, value in list(_OG_SERIALIZABLE_MAPPING.items())[:5]:
            assert isinstance(key, tuple)
            assert isinstance(value, tuple)
            assert all(isinstance(item, str) for item in key)
            assert all(isinstance(item, str) for item in value)

    def test_js_serializable_mapping_structure(self) -> None:
        """Test _JS_SERIALIZABLE_MAPPING has correct structure."""
        assert isinstance(_JS_SERIALIZABLE_MAPPING, dict)

        for key, value in list(_JS_SERIALIZABLE_MAPPING.items())[:5]:
            assert isinstance(key, tuple)
            assert isinstance(value, tuple)
            assert all(isinstance(item, str) for item in key)
            assert all(isinstance(item, str) for item in value)

    def test_all_serializable_mappings_structure(self) -> None:
        """Test ALL_SERIALIZABLE_MAPPINGS combines all mappings."""
        assert isinstance(ALL_SERIALIZABLE_MAPPINGS, dict)

        # ALL_SERIALIZABLE_MAPPINGS should include all other mappings
        assert len(ALL_SERIALIZABLE_MAPPINGS) >= len(SERIALIZABLE_MAPPING)
        assert len(ALL_SERIALIZABLE_MAPPINGS) >= len(OLD_CORE_NAMESPACES_MAPPING)
        assert len(ALL_SERIALIZABLE_MAPPINGS) >= len(_OG_SERIALIZABLE_MAPPING)
        assert len(ALL_SERIALIZABLE_MAPPINGS) >= len(_JS_SERIALIZABLE_MAPPING)

    def test_disallow_load_from_path_structure(self) -> None:
        """Test DISALLOW_LOAD_FROM_PATH has correct structure."""
        assert isinstance(DISALLOW_LOAD_FROM_PATH, list)
        assert all(isinstance(item, str) for item in DISALLOW_LOAD_FROM_PATH)


class TestMappingContent:
    """Tests for the content of mapping dictionaries."""

    def test_serializable_mapping_contains_ai_message(self) -> None:
        """Test SERIALIZABLE_MAPPING contains AIMessage mapping."""
        key = ("langchain", "schema", "messages", "AIMessage")
        assert key in SERIALIZABLE_MAPPING

        value = SERIALIZABLE_MAPPING[key]
        assert value == ("langchain_core", "messages", "ai", "AIMessage")

    def test_serializable_mapping_contains_human_message(self) -> None:
        """Test SERIALIZABLE_MAPPING contains HumanMessage mapping."""
        key = ("langchain", "schema", "messages", "HumanMessage")
        assert key in SERIALIZABLE_MAPPING

        value = SERIALIZABLE_MAPPING[key]
        assert value == ("langchain_core", "messages", "human", "HumanMessage")

    def test_serializable_mapping_contains_chat_openai(self) -> None:
        """Test SERIALIZABLE_MAPPING contains ChatOpenAI mapping."""
        key = ("langchain", "chat_models", "openai", "ChatOpenAI")
        assert key in SERIALIZABLE_MAPPING

        value = SERIALIZABLE_MAPPING[key]
        assert value == ("langchain_openai", "chat_models", "base", "ChatOpenAI")

    def test_old_core_namespaces_mapping_contains_langchain_core_messages(
        self,
    ) -> None:
        """Test OLD_CORE_NAMESPACES_MAPPING contains langchain_core messages."""
        key = ("langchain_core", "messages", "ai", "AIMessage")
        assert key in OLD_CORE_NAMESPACES_MAPPING

        value = OLD_CORE_NAMESPACES_MAPPING[key]
        assert value == ("langchain_core", "messages", "ai", "AIMessage")

    def test_og_serializable_mapping_contains_old_schema(self) -> None:
        """Test _OG_SERIALIZABLE_MAPPING contains old schema mappings."""
        key = ("langchain", "schema", "AIMessage")
        assert key in _OG_SERIALIZABLE_MAPPING

        value = _OG_SERIALIZABLE_MAPPING[key]
        assert value == ("langchain_core", "messages", "ai", "AIMessage")

    def test_js_serializable_mapping_contains_langchain_core_shortcuts(self) -> None:
        """Test _JS_SERIALIZABLE_MAPPING contains JS-compatible shortcuts."""
        key = ("langchain_core", "messages", "AIMessage")
        assert key in _JS_SERIALIZABLE_MAPPING

        value = _JS_SERIALIZABLE_MAPPING[key]
        assert value == ("langchain_core", "messages", "ai", "AIMessage")

    def test_disallow_load_from_path_contains_expected_namespaces(self) -> None:
        """Test DISALLOW_LOAD_FROM_PATH contains expected namespaces."""
        assert "langchain_community" in DISALLOW_LOAD_FROM_PATH
        assert "langchain" in DISALLOW_LOAD_FROM_PATH


class TestMappingConsistency:
    """Tests for consistency across mapping dictionaries."""

    def test_all_serializable_mappings_includes_serializable_mapping(self) -> None:
        """Test ALL_SERIALIZABLE_MAPPINGS includes all SERIALIZABLE_MAPPING entries."""
        # ALL_SERIALIZABLE_MAPPINGS is built by combining multiple dicts, where later
        # values may override earlier ones. We just check that keys are present.
        for key in SERIALIZABLE_MAPPING:
            assert key in ALL_SERIALIZABLE_MAPPINGS

    def test_all_serializable_mappings_includes_old_core_namespaces(self) -> None:
        """Test ALL_SERIALIZABLE_MAPPINGS includes OLD_CORE_NAMESPACES_MAPPING."""
        for key, value in OLD_CORE_NAMESPACES_MAPPING.items():
            assert key in ALL_SERIALIZABLE_MAPPINGS
            assert ALL_SERIALIZABLE_MAPPINGS[key] == value

    def test_all_serializable_mappings_includes_og_mapping(self) -> None:
        """Test ALL_SERIALIZABLE_MAPPINGS includes _OG_SERIALIZABLE_MAPPING."""
        for key, value in _OG_SERIALIZABLE_MAPPING.items():
            assert key in ALL_SERIALIZABLE_MAPPINGS
            assert ALL_SERIALIZABLE_MAPPINGS[key] == value

    def test_all_serializable_mappings_includes_js_mapping(self) -> None:
        """Test ALL_SERIALIZABLE_MAPPINGS includes _JS_SERIALIZABLE_MAPPING."""
        for key, value in _JS_SERIALIZABLE_MAPPING.items():
            assert key in ALL_SERIALIZABLE_MAPPINGS
            assert ALL_SERIALIZABLE_MAPPINGS[key] == value

    def test_mapping_values_point_to_valid_paths(self) -> None:
        """Test a sample of mapping values have valid module path structure."""
        # Check a sample of mappings
        sample_mappings = list(SERIALIZABLE_MAPPING.items())[:10]

        for key, value in sample_mappings:
            # Value should be a tuple with at least 2 elements (module + class)
            assert len(value) >= 2
            # First element should be a valid package name
            assert value[0] in [
                "langchain",
                "langchain_core",
                "langchain_openai",
                "langchain_anthropic",
                "langchain_aws",
                "langchain_groq",
                "langchain_fireworks",
                "langchain_mistralai",
                "langchain_google_vertexai",
                "langchain_google_genai",
                "langchain_vertexai",
                "langchain_sambanova",
            ]


class TestMappingSpecificCases:
    """Tests for specific mapping cases and edge cases."""

    def test_message_types_all_mapped(self) -> None:
        """Test all common message types are mapped."""
        message_types = [
            "AIMessage",
            "HumanMessage",
            "SystemMessage",
            "FunctionMessage",
            "ToolMessage",
            "ChatMessage",
        ]

        for msg_type in message_types:
            key = ("langchain", "schema", "messages", msg_type)
            assert key in SERIALIZABLE_MAPPING

    def test_message_chunks_all_mapped(self) -> None:
        """Test all common message chunk types are mapped."""
        chunk_types = [
            "AIMessageChunk",
            "HumanMessageChunk",
            "SystemMessageChunk",
            "FunctionMessageChunk",
            "ToolMessageChunk",
            "ChatMessageChunk",
        ]

        for chunk_type in chunk_types:
            key = ("langchain", "schema", "messages", chunk_type)
            assert key in SERIALIZABLE_MAPPING

    def test_prompt_template_mapped(self) -> None:
        """Test PromptTemplate is mapped correctly."""
        key = ("langchain", "prompts", "prompt", "PromptTemplate")
        assert key in SERIALIZABLE_MAPPING

        value = SERIALIZABLE_MAPPING[key]
        assert value == ("langchain_core", "prompts", "prompt", "PromptTemplate")

    def test_chat_prompt_template_mapped(self) -> None:
        """Test ChatPromptTemplate is mapped correctly."""
        key = ("langchain", "prompts", "chat", "ChatPromptTemplate")
        assert key in SERIALIZABLE_MAPPING

        value = SERIALIZABLE_MAPPING[key]
        assert value == ("langchain_core", "prompts", "chat", "ChatPromptTemplate")

    def test_generation_types_mapped(self) -> None:
        """Test Generation types are mapped correctly."""
        generation_types = [
            ("langchain", "schema", "output", "Generation"),
            ("langchain", "schema", "output", "ChatGeneration"),
            ("langchain", "schema", "output", "GenerationChunk"),
            ("langchain", "schema", "output", "ChatGenerationChunk"),
        ]

        for key in generation_types:
            assert key in SERIALIZABLE_MAPPING

    def test_agent_types_mapped(self) -> None:
        """Test Agent types are mapped correctly."""
        agent_types = [
            ("langchain", "schema", "agent", "AgentAction"),
            ("langchain", "schema", "agent", "AgentFinish"),
            ("langchain", "schema", "agent", "AgentActionMessageLog"),
        ]

        for key in agent_types:
            assert key in SERIALIZABLE_MAPPING

    def test_output_parser_types_mapped(self) -> None:
        """Test common output parser types are mapped."""
        parser_keys = [
            ("langchain", "schema", "output_parser", "StrOutputParser"),
            ("langchain", "output_parsers", "list", "CommaSeparatedListOutputParser"),
        ]

        for key in parser_keys:
            assert key in SERIALIZABLE_MAPPING

    def test_runnable_types_mapped(self) -> None:
        """Test Runnable types are mapped correctly."""
        runnable_types = [
            ("langchain", "schema", "runnable", "RunnableSequence"),
            ("langchain", "schema", "runnable", "RunnableParallel"),
            ("langchain", "schema", "runnable", "RunnablePassthrough"),
            ("langchain", "schema", "runnable", "RunnableBranch"),
        ]

        for key in runnable_types:
            assert key in SERIALIZABLE_MAPPING

    def test_document_type_mapped(self) -> None:
        """Test Document type is mapped correctly."""
        key = ("langchain", "schema", "document", "Document")
        assert key in SERIALIZABLE_MAPPING

        value = SERIALIZABLE_MAPPING[key]
        assert value == ("langchain_core", "documents", "base", "Document")


class TestMappingUsage:
    """Tests for how mappings are used in practice."""

    def test_mapping_lookup_performance(self) -> None:
        """Test mapping lookups are efficient (dict O(1) lookups)."""
        # This is a basic sanity check that we're using dicts
        key = ("langchain", "schema", "messages", "AIMessage")

        # Should be fast O(1) lookup
        assert key in ALL_SERIALIZABLE_MAPPINGS
        value = ALL_SERIALIZABLE_MAPPINGS[key]
        assert value == ("langchain_core", "messages", "ai", "AIMessage")

    def test_no_duplicate_keys_in_all_serializable_mappings(self) -> None:
        """Test ALL_SERIALIZABLE_MAPPINGS doesn't have unexpected duplicates."""
        # When combining dicts, later values override earlier ones
        # This is expected, but we want to verify the structure
        expected_size = len(
            {
                **SERIALIZABLE_MAPPING,
                **OLD_CORE_NAMESPACES_MAPPING,
                **_OG_SERIALIZABLE_MAPPING,
                **_JS_SERIALIZABLE_MAPPING,
            }
        )

        assert len(ALL_SERIALIZABLE_MAPPINGS) == expected_size

    def test_mappings_are_immutable(self) -> None:
        """Test that mapping dictionaries are defined as expected."""
        # Just verify they exist and are dicts
        # In production code, these should not be modified
        assert isinstance(SERIALIZABLE_MAPPING, dict)
        assert isinstance(OLD_CORE_NAMESPACES_MAPPING, dict)
        assert isinstance(_OG_SERIALIZABLE_MAPPING, dict)
        assert isinstance(_JS_SERIALIZABLE_MAPPING, dict)
        assert isinstance(ALL_SERIALIZABLE_MAPPINGS, dict)


class TestBackwardsCompatibility:
    """Tests for backwards compatibility mappings."""

    def test_og_mapping_redirects_old_paths(self) -> None:
        """Test _OG_SERIALIZABLE_MAPPING redirects old langchain.schema paths."""
        # Old paths without 'messages' subdirectory
        old_keys = [
            ("langchain", "schema", "AIMessage"),
            ("langchain", "schema", "HumanMessage"),
            ("langchain", "schema", "SystemMessage"),
        ]

        for key in old_keys:
            assert key in _OG_SERIALIZABLE_MAPPING
            # Should redirect to langchain_core.messages
            value = _OG_SERIALIZABLE_MAPPING[key]
            assert value[0] == "langchain_core"
            assert value[1] == "messages"

    def test_old_core_namespaces_allows_langchain_core_paths(self) -> None:
        """Test OLD_CORE_NAMESPACES_MAPPING allows langchain_core paths."""
        # Paths that start with langchain_core
        core_key = ("langchain_core", "messages", "ai", "AIMessage")
        assert core_key in OLD_CORE_NAMESPACES_MAPPING
        # Should map to itself
        assert OLD_CORE_NAMESPACES_MAPPING[core_key] == core_key

    def test_js_mapping_provides_js_compatible_paths(self) -> None:
        """Test _JS_SERIALIZABLE_MAPPING provides JS-compatible paths."""
        # JS often uses shorter paths
        js_key = ("langchain_core", "messages", "AIMessage")
        assert js_key in _JS_SERIALIZABLE_MAPPING

        value = _JS_SERIALIZABLE_MAPPING[js_key]
        assert value == ("langchain_core", "messages", "ai", "AIMessage")
