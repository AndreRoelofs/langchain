"""Comprehensive tests for langchain_core.load.load module."""

import json
import os

import pytest
from pydantic import SecretStr

from langchain_core.load import load, loads
from langchain_core.load.load import Reviver
from langchain_core.load.serializable import Serializable


class TestSerializable(Serializable):
    """Test serializable class."""

    value: int
    name: str

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


class DefaultModel(Serializable):
    """Model with default values for testing empty kwargs deserialization."""

    value: int = 0
    name: str = "default"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


class SecretModel(Serializable):
    """Model with secrets for round-trip testing."""

    api_key: SecretStr
    name: str

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "MY_API_KEY"}


class TestReviver:
    """Tests for the Reviver class."""

    def test_reviver_init_default(self) -> None:
        """Test Reviver initialization with default parameters."""
        reviver = Reviver()

        assert reviver.secrets_map == {}
        assert reviver.secrets_from_env is True
        assert "langchain" in reviver.valid_namespaces
        assert "langchain_core" in reviver.valid_namespaces
        assert reviver.ignore_unserializable_fields is False

    def test_reviver_init_custom_namespaces(self) -> None:
        """Test Reviver initialization with custom namespaces."""
        reviver = Reviver(valid_namespaces=["tests", "custom"])

        assert "langchain" in reviver.valid_namespaces
        assert "tests" in reviver.valid_namespaces
        assert "custom" in reviver.valid_namespaces

    def test_reviver_init_with_secrets_map(self) -> None:
        """Test Reviver initialization with secrets_map."""
        secrets = {"API_KEY": "secret_value"}
        reviver = Reviver(secrets_map=secrets)

        assert reviver.secrets_map == secrets

    def test_reviver_init_with_additional_import_mappings(self) -> None:
        """Test Reviver initialization with additional_import_mappings."""
        mappings = {("custom", "module", "Class"): ("actual", "module", "Class")}
        reviver = Reviver(additional_import_mappings=mappings)

        assert mappings.items() <= reviver.import_mappings.items()

    def test_reviver_secret_from_map(self) -> None:
        """Test Reviver loads secret from secrets_map."""
        secrets = {"API_KEY": "secret_value"}
        reviver = Reviver(secrets_map=secrets)

        value = {
            "lc": 1,
            "type": "secret",
            "id": ["API_KEY"],
        }

        result = reviver(value)
        assert result == "secret_value"

    def test_reviver_secret_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Reviver loads secret from environment."""
        monkeypatch.setenv("TEST_API_KEY", "env_secret_value")
        reviver = Reviver(secrets_from_env=True)

        value = {
            "lc": 1,
            "type": "secret",
            "id": ["TEST_API_KEY"],
        }

        result = reviver(value)
        assert result == "env_secret_value"

    def test_reviver_secret_not_in_env_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test Reviver returns None when secret not in environment."""
        monkeypatch.delenv("MISSING_KEY", raising=False)
        reviver = Reviver(secrets_from_env=True)

        value = {
            "lc": 1,
            "type": "secret",
            "id": ["MISSING_KEY"],
        }

        result = reviver(value)
        assert result is None

    def test_reviver_secret_from_env_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test Reviver does not load from env when secrets_from_env=False."""
        monkeypatch.setenv("TEST_KEY", "env_value")
        reviver = Reviver(secrets_from_env=False)

        value = {
            "lc": 1,
            "type": "secret",
            "id": ["TEST_KEY"],
        }

        result = reviver(value)
        assert result is None

    def test_reviver_not_implemented_raises_error(self) -> None:
        """Test Reviver raises NotImplementedError for not_implemented type."""
        reviver = Reviver()

        value = {
            "lc": 1,
            "type": "not_implemented",
            "id": ["some", "module", "Class"],
        }

        with pytest.raises(
            NotImplementedError, match="doesn't implement serialization"
        ):
            reviver(value)

    def test_reviver_not_implemented_with_ignore_flag(self) -> None:
        """Test Reviver returns None for not_implemented when ignore flag is True."""
        reviver = Reviver(ignore_unserializable_fields=True)

        value = {
            "lc": 1,
            "type": "not_implemented",
            "id": ["some", "module", "Class"],
        }

        result = reviver(value)
        assert result is None

    def test_reviver_constructor_deserialization(self) -> None:
        """Test Reviver deserializes constructor type."""
        reviver = Reviver(valid_namespaces=["tests"])

        value = {
            "lc": 1,
            "type": "constructor",
            "id": ["tests", "unit_tests", "load", "test_load", "TestSerializable"],
            "kwargs": {"value": 42, "name": "test"},
        }

        result = reviver(value)
        assert isinstance(result, TestSerializable)
        assert result.value == 42
        assert result.name == "test"

    def test_reviver_invalid_namespace_raises_error(self) -> None:
        """Test Reviver raises ValueError for invalid namespace."""
        reviver = Reviver(valid_namespaces=["tests"])

        value = {
            "lc": 1,
            "type": "constructor",
            "id": ["invalid_namespace", "module", "Class"],
            "kwargs": {},
        }

        with pytest.raises(ValueError, match="Invalid namespace"):
            reviver(value)

    def test_reviver_root_langchain_namespace_raises_error(self) -> None:
        """Test Reviver raises ValueError for root langchain namespace."""
        reviver = Reviver()

        value = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "SomeClass"],
            "kwargs": {},
        }

        with pytest.raises(ValueError, match="Invalid namespace"):
            reviver(value)

    def test_reviver_with_import_mapping(self) -> None:
        """Test Reviver uses import_mappings to find class."""
        reviver = Reviver(
            valid_namespaces=["tests", "old"],
            additional_import_mappings={
                ("old", "namespace", "Class"): (
                    "tests",
                    "unit_tests",
                    "load",
                    "test_load",
                    "TestSerializable",
                )
            },
        )

        value = {
            "lc": 1,
            "type": "constructor",
            "id": ["old", "namespace", "Class"],
            "kwargs": {"value": 42, "name": "test"},
        }

        result = reviver(value)
        assert isinstance(result, TestSerializable)

    def test_reviver_disallow_load_from_path(self) -> None:
        """Test Reviver blocks loading from path for disallowed namespaces."""
        reviver = Reviver()

        # langchain_community is in DISALLOW_LOAD_FROM_PATH
        value = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain_community", "some", "module", "Class"],
            "kwargs": {},
        }

        with pytest.raises(ValueError, match="cannot be deserialized"):
            reviver(value)

    def test_reviver_non_serializable_class_raises_error(self) -> None:
        """Test Reviver raises ValueError if class is not Serializable."""
        reviver = Reviver(valid_namespaces=["tests"])

        # Point to a non-Serializable class (using pytest itself as example)
        value = {
            "lc": 1,
            "type": "constructor",
            "id": ["pytest", "MonkeyPatch"],
            "kwargs": {},
        }

        with pytest.raises(ValueError, match="Invalid namespace"):
            reviver(value)

    def test_reviver_passthrough_non_lc_dict(self) -> None:
        """Test Reviver passes through non-LangChain dicts."""
        reviver = Reviver()

        value = {"key": "value", "number": 42}

        result = reviver(value)
        assert result == value


class TestLoads:
    """Tests for the loads() function."""

    def test_loads_basic(self) -> None:
        """Test basic loads() functionality."""
        json_str = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["tests", "unit_tests", "load", "test_load", "TestSerializable"],
                "kwargs": {"value": 42, "name": "test"},
            }
        )

        result = loads(json_str, valid_namespaces=["tests"])
        assert isinstance(result, TestSerializable)
        assert result.value == 42
        assert result.name == "test"

    def test_loads_with_secrets_map(self) -> None:
        """Test loads() with secrets_map."""
        json_str = json.dumps(
            {
                "lc": 1,
                "type": "secret",
                "id": ["API_KEY"],
            }
        )

        result = loads(json_str, secrets_map={"API_KEY": "secret_value"})
        assert result == "secret_value"

    def test_loads_with_secrets_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loads() with secrets from environment."""
        monkeypatch.setenv("TEST_KEY", "env_value")

        json_str = json.dumps(
            {
                "lc": 1,
                "type": "secret",
                "id": ["TEST_KEY"],
            }
        )

        result = loads(json_str, secrets_from_env=True)
        assert result == "env_value"

    def test_loads_with_additional_import_mappings(self) -> None:
        """Test loads() with additional_import_mappings."""
        json_str = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["old", "namespace", "Class"],
                "kwargs": {"value": 42, "name": "test"},
            }
        )

        result = loads(
            json_str,
            valid_namespaces=["tests", "old"],
            additional_import_mappings={
                ("old", "namespace", "Class"): (
                    "tests",
                    "unit_tests",
                    "load",
                    "test_load",
                    "TestSerializable",
                )
            },
        )
        assert isinstance(result, TestSerializable)

    def test_loads_with_ignore_unserializable_fields(self) -> None:
        """Test loads() with ignore_unserializable_fields=True."""
        json_str = json.dumps(
            {
                "lc": 1,
                "type": "not_implemented",
                "id": ["some", "module", "Class"],
            }
        )

        result = loads(json_str, ignore_unserializable_fields=True)
        assert result is None

    def test_loads_nested_structure(self) -> None:
        """Test loads() with nested structures."""
        json_str = json.dumps(
            {
                "data": {
                    "lc": 1,
                    "type": "constructor",
                    "id": [
                        "tests",
                        "unit_tests",
                        "load",
                        "test_load",
                        "TestSerializable",
                    ],
                    "kwargs": {"value": 42, "name": "test"},
                },
                "list": [1, 2, 3],
            }
        )

        result = loads(json_str, valid_namespaces=["tests"])
        assert isinstance(result["data"], TestSerializable)
        assert result["list"] == [1, 2, 3]


class TestLoad:
    """Tests for the load() function."""

    def test_load_basic(self) -> None:
        """Test basic load() functionality."""
        obj = {
            "lc": 1,
            "type": "constructor",
            "id": ["tests", "unit_tests", "load", "test_load", "TestSerializable"],
            "kwargs": {"value": 42, "name": "test"},
        }

        result = load(obj, valid_namespaces=["tests"])
        assert isinstance(result, TestSerializable)
        assert result.value == 42
        assert result.name == "test"

    def test_load_with_secrets_map(self) -> None:
        """Test load() with secrets_map."""
        obj = {
            "lc": 1,
            "type": "secret",
            "id": ["API_KEY"],
        }

        result = load(obj, secrets_map={"API_KEY": "secret_value"})
        assert result == "secret_value"

    def test_load_with_secrets_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test load() with secrets from environment."""
        monkeypatch.setenv("TEST_KEY", "env_value")

        obj = {
            "lc": 1,
            "type": "secret",
            "id": ["TEST_KEY"],
        }

        result = load(obj, secrets_from_env=True)
        assert result == "env_value"

    def test_load_with_additional_import_mappings(self) -> None:
        """Test load() with additional_import_mappings."""
        obj = {
            "lc": 1,
            "type": "constructor",
            "id": ["old", "namespace", "Class"],
            "kwargs": {"value": 42, "name": "test"},
        }

        result = load(
            obj,
            valid_namespaces=["tests", "old"],
            additional_import_mappings={
                ("old", "namespace", "Class"): (
                    "tests",
                    "unit_tests",
                    "load",
                    "test_load",
                    "TestSerializable",
                )
            },
        )
        assert isinstance(result, TestSerializable)

    def test_load_with_ignore_unserializable_fields(self) -> None:
        """Test load() with ignore_unserializable_fields=True."""
        obj = {
            "lc": 1,
            "type": "not_implemented",
            "id": ["some", "module", "Class"],
        }

        result = load(obj, ignore_unserializable_fields=True)
        assert result is None

    def test_load_nested_dict_structure(self) -> None:
        """Test load() with nested dict structures."""
        obj = {
            "outer": {
                "inner": {
                    "lc": 1,
                    "type": "constructor",
                    "id": [
                        "tests",
                        "unit_tests",
                        "load",
                        "test_load",
                        "TestSerializable",
                    ],
                    "kwargs": {"value": 42, "name": "test"},
                }
            }
        }

        result = load(obj, valid_namespaces=["tests"])
        assert isinstance(result["outer"]["inner"], TestSerializable)

    def test_load_nested_list_structure(self) -> None:
        """Test load() with nested list structures."""
        obj = [
            {
                "lc": 1,
                "type": "constructor",
                "id": ["tests", "unit_tests", "load", "test_load", "TestSerializable"],
                "kwargs": {"value": 1, "name": "first"},
            },
            {
                "lc": 1,
                "type": "constructor",
                "id": ["tests", "unit_tests", "load", "test_load", "TestSerializable"],
                "kwargs": {"value": 2, "name": "second"},
            },
        ]

        result = load(obj, valid_namespaces=["tests"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], TestSerializable)
        assert result[0].value == 1
        assert isinstance(result[1], TestSerializable)
        assert result[1].value == 2

    def test_load_primitive_types(self) -> None:
        """Test load() with primitive types."""
        # String
        assert load("test") == "test"

        # Number
        assert load(42) == 42

        # Boolean
        assert load(True) is True

        # None
        assert load(None) is None

    def test_load_complex_nested_structure(self) -> None:
        """Test load() with complex nested structures."""
        obj = {
            "serializable": {
                "lc": 1,
                "type": "constructor",
                "id": ["tests", "unit_tests", "load", "test_load", "TestSerializable"],
                "kwargs": {"value": 42, "name": "test"},
            },
            "list_of_serializables": [
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": [
                        "tests",
                        "unit_tests",
                        "load",
                        "test_load",
                        "TestSerializable",
                    ],
                    "kwargs": {"value": 1, "name": "first"},
                },
            ],
            "primitive": "string",
            "nested": {"data": [1, 2, 3]},
        }

        result = load(obj, valid_namespaces=["tests"])
        assert isinstance(result["serializable"], TestSerializable)
        assert isinstance(result["list_of_serializables"][0], TestSerializable)
        assert result["primitive"] == "string"
        assert result["nested"]["data"] == [1, 2, 3]

    def test_load_with_empty_env_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test load() returns None when env var is empty string."""
        monkeypatch.setenv("EMPTY_KEY", "")

        obj = {
            "lc": 1,
            "type": "secret",
            "id": ["EMPTY_KEY"],
        }

        result = load(obj, secrets_from_env=True)
        assert result is None


class TestRoundTrip:
    """Tests for serialization/deserialization round trips."""

    def test_round_trip_basic(self) -> None:
        """Test round trip serialization and deserialization."""
        from langchain_core.load import dumpd

        original = TestSerializable(value=42, name="test")
        serialized = dumpd(original)
        deserialized = load(serialized, valid_namespaces=["tests"])

        assert isinstance(deserialized, TestSerializable)
        assert deserialized.value == original.value
        assert deserialized.name == original.name

    def test_round_trip_with_loads_dumps(self) -> None:
        """Test round trip with loads() and dumps()."""
        from langchain_core.load import dumps

        original = TestSerializable(value=42, name="test")
        json_str = dumps(original)
        deserialized = loads(json_str, valid_namespaces=["tests"])

        assert isinstance(deserialized, TestSerializable)
        assert deserialized.value == original.value
        assert deserialized.name == original.name


# ---------------------------------------------------------------------------
# Additional snapshot tests for load.py
# ---------------------------------------------------------------------------


class TestDefaultNamespacesSnapshot:
    """Snapshot tests for DEFAULT_NAMESPACES constant."""

    def test_default_namespaces_exact_snapshot(self) -> None:
        """Snapshot: DEFAULT_NAMESPACES contains the expected namespaces."""
        from langchain_core.load.load import DEFAULT_NAMESPACES

        assert DEFAULT_NAMESPACES == [
            "langchain",
            "langchain_core",
            "langchain_community",
            "langchain_anthropic",
            "langchain_groq",
            "langchain_google_genai",
            "langchain_aws",
            "langchain_openai",
            "langchain_google_vertexai",
            "langchain_mistralai",
            "langchain_fireworks",
            "langchain_xai",
            "langchain_sambanova",
            "langchain_perplexity",
        ]

    def test_disallow_load_from_path_exact_snapshot(self) -> None:
        """Snapshot: DISALLOW_LOAD_FROM_PATH contains the expected namespaces."""
        from langchain_core.load.load import DISALLOW_LOAD_FROM_PATH

        assert DISALLOW_LOAD_FROM_PATH == [
            "langchain_community",
            "langchain",
        ]


class TestReviverSnapshot:
    """Snapshot tests for Reviver edge cases."""

    def test_reviver_non_lc_versioned_dict_passthrough(self) -> None:
        """Snapshot: dict with lc != 1 passes through unchanged."""
        reviver = Reviver()

        value = {"lc": 2, "type": "constructor", "id": ["a"], "kwargs": {}}
        result = reviver(value)
        assert result == value

    def test_reviver_dict_without_lc_passthrough(self) -> None:
        """Snapshot: dict without 'lc' key passes through unchanged."""
        reviver = Reviver()

        value = {"type": "constructor", "id": ["a"], "kwargs": {}}
        result = reviver(value)
        assert result == value

    def test_reviver_secret_without_id_passthrough(self) -> None:
        """Snapshot: secret-like dict without 'id' passes through."""
        reviver = Reviver()

        value = {"lc": 1, "type": "secret"}
        result = reviver(value)
        assert result == value

    def test_reviver_constructor_without_id_passthrough(self) -> None:
        """Snapshot: constructor-like dict without 'id' passes through."""
        reviver = Reviver()

        value = {"lc": 1, "type": "constructor", "kwargs": {}}
        result = reviver(value)
        assert result == value

    def test_reviver_constructor_empty_kwargs(self) -> None:
        """Snapshot: constructor with empty kwargs creates object with defaults."""
        reviver = Reviver(valid_namespaces=["tests"])
        value = {
            "lc": 1,
            "type": "constructor",
            "id": [
                "tests",
                "unit_tests",
                "load",
                "test_load",
                "DefaultModel",
            ],
            "kwargs": {},
        }
        result = reviver(value)
        assert isinstance(result, DefaultModel)
        assert result.value == 0
        assert result.name == "default"

    def test_reviver_constructor_missing_kwargs_key(self) -> None:
        """Snapshot: constructor without 'kwargs' key uses empty dict."""
        reviver = Reviver(valid_namespaces=["tests"])
        value = {
            "lc": 1,
            "type": "constructor",
            "id": [
                "tests",
                "unit_tests",
                "load",
                "test_load",
                "TestSerializable",
            ],
        }
        # value.get("kwargs", {}) → empty dict → requires defaults or will fail
        # TestSerializable has required fields, so this should raise
        with pytest.raises(Exception):
            reviver(value)

    def test_reviver_secret_map_takes_priority_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snapshot: secrets_map takes priority over environment variables."""
        monkeypatch.setenv("MY_KEY", "from_env")
        reviver = Reviver(
            secrets_map={"MY_KEY": "from_map"},
            secrets_from_env=True,
        )
        value = {"lc": 1, "type": "secret", "id": ["MY_KEY"]}
        result = reviver(value)
        assert result == "from_map"

    def test_reviver_valid_namespaces_merged_with_defaults(self) -> None:
        """Snapshot: valid_namespaces are appended to DEFAULT_NAMESPACES."""
        from langchain_core.load.load import DEFAULT_NAMESPACES

        reviver = Reviver(valid_namespaces=["my_custom_ns"])
        assert reviver.valid_namespaces == [*DEFAULT_NAMESPACES, "my_custom_ns"]

    def test_reviver_no_custom_namespaces_uses_defaults(self) -> None:
        """Snapshot: without custom namespaces, uses DEFAULT_NAMESPACES."""
        from langchain_core.load.load import DEFAULT_NAMESPACES

        reviver = Reviver()
        assert reviver.valid_namespaces == DEFAULT_NAMESPACES

    def test_reviver_additional_import_mappings_override(self) -> None:
        """Snapshot: additional_import_mappings override existing mappings."""
        custom_mapping = {
            ("langchain", "schema", "messages", "AIMessage"): (
                "tests",
                "unit_tests",
                "load",
                "test_load",
                "TestSerializable",
            )
        }
        reviver = Reviver(
            valid_namespaces=["tests"],
            additional_import_mappings=custom_mapping,
        )
        # The custom mapping should override the default AIMessage mapping
        key = ("langchain", "schema", "messages", "AIMessage")
        assert reviver.import_mappings[key] == (
            "tests",
            "unit_tests",
            "load",
            "test_load",
            "TestSerializable",
        )

    def test_reviver_import_mappings_unchanged_without_additional(self) -> None:
        """Snapshot: without additional_import_mappings, uses ALL_SERIALIZABLE_MAPPINGS."""
        from langchain_core.load.load import ALL_SERIALIZABLE_MAPPINGS

        reviver = Reviver()
        assert reviver.import_mappings is ALL_SERIALIZABLE_MAPPINGS

    def test_reviver_not_implemented_with_repr(self) -> None:
        """Snapshot: not_implemented includes repr in error message."""
        reviver = Reviver()

        value = {
            "lc": 1,
            "type": "not_implemented",
            "id": ["some", "Class"],
            "repr": "Class(x=1)",
        }
        with pytest.raises(NotImplementedError, match="doesn't implement"):
            reviver(value)

    def test_reviver_unknown_type_passthrough(self) -> None:
        """Snapshot: dict with lc=1 but unknown type passes through."""
        reviver = Reviver()

        value = {"lc": 1, "type": "unknown_type", "id": ["a"]}
        result = reviver(value)
        assert result == value

    def test_reviver_langchain_core_direct_namespace(self) -> None:
        """Snapshot: langchain_core namespace is valid for direct import."""
        reviver = Reviver()

        # langchain_core is in DEFAULT_NAMESPACES and NOT in DISALLOW_LOAD_FROM_PATH
        # Constructing an AIMessage via langchain_core namespace
        value = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain_core", "messages", "ai", "AIMessage"],
            "kwargs": {"content": "hello"},
        }
        result = reviver(value)
        from langchain_core.messages import AIMessage

        assert isinstance(result, AIMessage)
        assert result.content == "hello"


class TestLoadsSnapshot:
    """Snapshot tests for loads() edge cases."""

    def test_loads_invalid_json_raises_error(self) -> None:
        """Snapshot: loads() with invalid JSON raises json.JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            loads("not valid json{{{")

    def test_loads_plain_json_string(self) -> None:
        """Snapshot: loads() with plain JSON string returns the string."""
        result = loads('"hello"')
        assert result == "hello"

    def test_loads_plain_json_number(self) -> None:
        """Snapshot: loads() with plain JSON number returns the number."""
        result = loads("42")
        assert result == 42

    def test_loads_plain_json_null(self) -> None:
        """Snapshot: loads() with JSON null returns None."""
        result = loads("null")
        assert result is None

    def test_loads_plain_json_array(self) -> None:
        """Snapshot: loads() with plain JSON array returns the list."""
        result = loads("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_loads_secret_not_in_map_or_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snapshot: loads() returns None for secret not in map or env."""
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        json_str = json.dumps(
            {
                "lc": 1,
                "type": "secret",
                "id": ["NONEXISTENT_KEY"],
            }
        )
        result = loads(json_str, secrets_map={})
        assert result is None

    def test_loads_with_secrets_from_env_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snapshot: loads() with secrets_from_env=False ignores env vars."""
        monkeypatch.setenv("MY_SECRET", "from_env")
        json_str = json.dumps(
            {
                "lc": 1,
                "type": "secret",
                "id": ["MY_SECRET"],
            }
        )
        result = loads(json_str, secrets_from_env=False)
        assert result is None


class TestLoadSnapshot:
    """Snapshot tests for load() edge cases."""

    def test_load_deeply_nested_mixed(self) -> None:
        """Snapshot: load() with deeply nested mix of dicts, lists, and objects."""
        obj = {
            "level1": {
                "level2": [
                    {
                        "level3": {
                            "lc": 1,
                            "type": "constructor",
                            "id": [
                                "tests",
                                "unit_tests",
                                "load",
                                "test_load",
                                "TestSerializable",
                            ],
                            "kwargs": {"value": 99, "name": "deep"},
                        }
                    }
                ]
            }
        }
        result = load(obj, valid_namespaces=["tests"])
        inner = result["level1"]["level2"][0]["level3"]
        assert isinstance(inner, TestSerializable)
        assert inner.value == 99
        assert inner.name == "deep"

    def test_load_empty_dict(self) -> None:
        """Snapshot: load() with empty dict returns empty dict."""
        assert load({}) == {}

    def test_load_empty_list(self) -> None:
        """Snapshot: load() with empty list returns empty list."""
        assert load([]) == []

    def test_load_float(self) -> None:
        """Snapshot: load() with float returns float."""
        assert load(3.14) == 3.14

    def test_load_nested_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Snapshot: load() resolves secrets inside nested kwargs."""
        monkeypatch.setenv("NESTED_KEY", "nested_secret_value")
        obj = {
            "lc": 1,
            "type": "constructor",
            "id": [
                "tests",
                "unit_tests",
                "load",
                "test_load",
                "TestSerializable",
            ],
            "kwargs": {
                "value": 42,
                "name": {
                    "lc": 1,
                    "type": "secret",
                    "id": ["NESTED_KEY"],
                },
            },
        }
        result = load(obj, valid_namespaces=["tests"])
        assert isinstance(result, TestSerializable)
        assert result.name == "nested_secret_value"

    def test_load_list_with_mixed_types(self) -> None:
        """Snapshot: load() with a list containing primitives and objects."""
        obj = [
            "string",
            42,
            True,
            None,
            {
                "lc": 1,
                "type": "constructor",
                "id": [
                    "tests",
                    "unit_tests",
                    "load",
                    "test_load",
                    "TestSerializable",
                ],
                "kwargs": {"value": 1, "name": "in_list"},
            },
        ]
        result = load(obj, valid_namespaces=["tests"])
        assert result[0] == "string"
        assert result[1] == 42
        assert result[2] is True
        assert result[3] is None
        assert isinstance(result[4], TestSerializable)


class TestRoundTripSnapshot:
    """Snapshot tests for round-trip serialization/deserialization."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Snapshot: round trip preserves value and name exactly."""
        from langchain_core.load import dumpd

        original = TestSerializable(value=99, name="round_trip_test")
        serialized = dumpd(original)

        assert serialized == {
            "lc": 1,
            "type": "constructor",
            "id": [
                "tests",
                "unit_tests",
                "load",
                "test_load",
                "TestSerializable",
            ],
            "kwargs": {"value": 99, "name": "round_trip_test"},
        }

        deserialized = load(serialized, valid_namespaces=["tests"])
        assert isinstance(deserialized, TestSerializable)
        assert deserialized.value == 99
        assert deserialized.name == "round_trip_test"

    def test_round_trip_with_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Snapshot: round-trip with secrets via map."""
        from langchain_core.load import dumpd

        original = SecretModel(api_key=SecretStr("secret123"), name="test")
        serialized = dumpd(original)

        # The secret should be masked
        assert serialized["kwargs"]["api_key"]["type"] == "secret"
        assert serialized["kwargs"]["api_key"]["id"] == ["MY_API_KEY"]

        # Deserialize with secrets_map
        deserialized = load(
            serialized,
            valid_namespaces=["tests"],
            secrets_map={"MY_API_KEY": "secret123"},
        )
        assert isinstance(deserialized, SecretModel)
        assert deserialized.api_key.get_secret_value() == "secret123"
        assert deserialized.name == "test"
