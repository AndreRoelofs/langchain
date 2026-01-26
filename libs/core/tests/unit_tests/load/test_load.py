"""Comprehensive tests for langchain_core.load.load module."""

import json
import os

import pytest

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
        mappings = {
            ("custom", "module", "Class"): ("actual", "module", "Class")
        }
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

        with pytest.raises(NotImplementedError, match="doesn't implement serialization"):
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
        json_str = json.dumps({
            "lc": 1,
            "type": "constructor",
            "id": ["tests", "unit_tests", "load", "test_load", "TestSerializable"],
            "kwargs": {"value": 42, "name": "test"},
        })

        result = loads(json_str, valid_namespaces=["tests"])
        assert isinstance(result, TestSerializable)
        assert result.value == 42
        assert result.name == "test"

    def test_loads_with_secrets_map(self) -> None:
        """Test loads() with secrets_map."""
        json_str = json.dumps({
            "lc": 1,
            "type": "secret",
            "id": ["API_KEY"],
        })

        result = loads(json_str, secrets_map={"API_KEY": "secret_value"})
        assert result == "secret_value"

    def test_loads_with_secrets_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loads() with secrets from environment."""
        monkeypatch.setenv("TEST_KEY", "env_value")

        json_str = json.dumps({
            "lc": 1,
            "type": "secret",
            "id": ["TEST_KEY"],
        })

        result = loads(json_str, secrets_from_env=True)
        assert result == "env_value"

    def test_loads_with_additional_import_mappings(self) -> None:
        """Test loads() with additional_import_mappings."""
        json_str = json.dumps({
            "lc": 1,
            "type": "constructor",
            "id": ["old", "namespace", "Class"],
            "kwargs": {"value": 42, "name": "test"},
        })

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
        json_str = json.dumps({
            "lc": 1,
            "type": "not_implemented",
            "id": ["some", "module", "Class"],
        })

        result = loads(json_str, ignore_unserializable_fields=True)
        assert result is None

    def test_loads_nested_structure(self) -> None:
        """Test loads() with nested structures."""
        json_str = json.dumps({
            "data": {
                "lc": 1,
                "type": "constructor",
                "id": ["tests", "unit_tests", "load", "test_load", "TestSerializable"],
                "kwargs": {"value": 42, "name": "test"},
            },
            "list": [1, 2, 3],
        })

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
