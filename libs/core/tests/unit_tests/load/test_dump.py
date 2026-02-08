"""Comprehensive tests for langchain_core.load.dump module."""

import json

import pytest
from pydantic import BaseModel

from langchain_core.load import dumpd, dumps
from langchain_core.load.dump import _dump_pydantic_models, default
from langchain_core.load.serializable import Serializable
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration


class SerializableTest(Serializable):
    """Test serializable class."""

    value: int

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


class NonSerializableTest:
    """Test non-serializable class."""

    def __init__(self, value: int) -> None:
        self.value = value


class TestDefault:
    """Tests for the default() function."""

    def test_default_with_serializable(self) -> None:
        """Test default() handles Serializable objects."""
        obj = SerializableTest(value=42)
        result = default(obj)
        assert isinstance(result, dict)
        assert result["type"] == "constructor"
        assert result["lc"] == 1

    def test_default_with_non_serializable(self) -> None:
        """Test default() handles non-Serializable objects."""
        obj = NonSerializableTest(value=42)
        result = default(obj)
        assert isinstance(result, dict)
        assert result["type"] == "not_implemented"
        assert result["lc"] == 1
        assert "NonSerializableTest" in result["repr"]

    def test_default_with_primitive(self) -> None:
        """Test default() handles primitive types."""
        result = default("test_string")
        assert isinstance(result, dict)
        assert result["type"] == "not_implemented"
        assert result["lc"] == 1


class TestDumpPydanticModels:
    """Tests for the _dump_pydantic_models() function."""

    def test_dump_pydantic_models_with_chat_generation(self) -> None:
        """Test _dump_pydantic_models() with ChatGeneration containing parsed BaseModel."""

        class MyModel(BaseModel):
            x: int
            y: str

        my_model = MyModel(x=1, y="hello")
        chat_gen = ChatGeneration(
            message=AIMessage(
                content='{"x": 1, "y": "hello"}', additional_kwargs={"parsed": my_model}
            )
        )

        result = _dump_pydantic_models(chat_gen)

        # Should return a copy with parsed as dict
        assert isinstance(result, ChatGeneration)
        assert result.message.additional_kwargs["parsed"] == {"x": 1, "y": "hello"}
        # Original should be unchanged
        assert isinstance(chat_gen.message.additional_kwargs["parsed"], MyModel)

    def test_dump_pydantic_models_without_parsed(self) -> None:
        """Test _dump_pydantic_models() with ChatGeneration without parsed field."""
        chat_gen = ChatGeneration(message=AIMessage(content="test"))

        result = _dump_pydantic_models(chat_gen)

        # Should return the same object
        assert result is chat_gen

    def test_dump_pydantic_models_with_non_chat_generation(self) -> None:
        """Test _dump_pydantic_models() with non-ChatGeneration objects."""

        class MyModel(BaseModel):
            x: int

        my_model = MyModel(x=1)

        result = _dump_pydantic_models(my_model)

        # Should return the same object
        assert result is my_model

    def test_dump_pydantic_models_with_non_pydantic_parsed(self) -> None:
        """Test _dump_pydantic_models() when parsed is not a BaseModel."""
        chat_gen = ChatGeneration(
            message=AIMessage(
                content="test", additional_kwargs={"parsed": "not_a_model"}
            )
        )

        result = _dump_pydantic_models(chat_gen)

        # Should return the same object
        assert result is chat_gen


class TestDumps:
    """Tests for the dumps() function."""

    def test_dumps_basic(self) -> None:
        """Test basic dumps() functionality."""
        obj = SerializableTest(value=42)
        json_str = dumps(obj)

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["type"] == "constructor"
        assert parsed["kwargs"]["value"] == 42

    def test_dumps_with_pretty_flag(self) -> None:
        """Test dumps() with pretty=True."""
        obj = SerializableTest(value=42)
        json_str = dumps(obj, pretty=True)

        # Should have indentation
        assert "  " in json_str
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["kwargs"]["value"] == 42

    def test_dumps_with_custom_indent(self) -> None:
        """Test dumps() with custom indent."""
        obj = SerializableTest(value=42)
        json_str = dumps(obj, pretty=True, indent=4)

        # Should have 4-space indentation
        assert "    " in json_str
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["kwargs"]["value"] == 42

    def test_dumps_with_default_kwarg_raises_error(self) -> None:
        """Test dumps() raises ValueError when default kwarg is passed."""
        obj = SerializableTest(value=42)

        with pytest.raises(ValueError, match="`default` should not be passed to dumps"):
            dumps(obj, default=lambda x: x)

    def test_dumps_with_json_kwargs(self) -> None:
        """Test dumps() passes through additional JSON kwargs."""
        obj = SerializableTest(value=42)

        # Test compact separators
        compact = dumps(obj, separators=(",", ":"))
        assert ", " not in compact

        # Test sort_keys
        sorted_json = dumps(obj, sort_keys=True)
        parsed = json.loads(sorted_json)
        assert "kwargs" in parsed

    def test_dumps_with_non_serializable_fallback(self) -> None:
        """Test dumps() fallback for TypeError."""
        # Create an object that will cause TypeError during normal serialization
        obj = NonSerializableTest(value=42)

        json_str = dumps(obj)
        parsed = json.loads(json_str)

        assert parsed["type"] == "not_implemented"
        assert "NonSerializableTest" in parsed["repr"]

    def test_dumps_with_pretty_and_typeerror_fallback(self) -> None:
        """Test dumps() fallback with pretty=True when TypeError occurs."""
        obj = NonSerializableTest(value=42)

        json_str = dumps(obj, pretty=True)
        parsed = json.loads(json_str)

        # Should have indentation
        assert "  " in json_str
        assert parsed["type"] == "not_implemented"

    def test_dumps_nested_structures(self) -> None:
        """Test dumps() with nested data structures."""
        data = {
            "serializable": SerializableTest(value=1),
            "list": [1, 2, {"nested": "value"}],
            "primitive": "string",
        }

        json_str = dumps(data)
        parsed = json.loads(json_str)

        assert parsed["serializable"]["type"] == "constructor"
        assert parsed["list"] == [1, 2, {"nested": "value"}]
        assert parsed["primitive"] == "string"

    def test_dumps_with_pydantic_model_in_chat_generation(self) -> None:
        """Test dumps() handles Pydantic models in ChatGeneration."""

        class MyModel(BaseModel):
            x: int
            y: str

        my_model = MyModel(x=1, y="hello")
        chat_gen = ChatGeneration(
            message=AIMessage(
                content='{"x": 1, "y": "hello"}', additional_kwargs={"parsed": my_model}
            )
        )

        json_str = dumps(chat_gen)
        parsed = json.loads(json_str)

        # parsed should be dumped as dict
        assert parsed["kwargs"]["message"]["kwargs"]["additional_kwargs"]["parsed"] == {
            "x": 1,
            "y": "hello",
        }


class TestDumpd:
    """Tests for the dumpd() function."""

    def test_dumpd_basic(self) -> None:
        """Test basic dumpd() functionality."""
        obj = SerializableTest(value=42)
        result = dumpd(obj)

        assert isinstance(result, dict)
        assert result["type"] == "constructor"
        assert result["kwargs"]["value"] == 42

    def test_dumpd_with_non_serializable(self) -> None:
        """Test dumpd() with non-serializable object."""
        obj = NonSerializableTest(value=42)
        result = dumpd(obj)

        assert isinstance(result, dict)
        assert result["type"] == "not_implemented"
        assert "NonSerializableTest" in result["repr"]

    def test_dumpd_with_nested_structures(self) -> None:
        """Test dumpd() with nested data structures."""
        data = {
            "serializable": SerializableTest(value=1),
            "list": [1, 2, {"nested": "value"}],
            "primitive": "string",
        }

        result = dumpd(data)

        assert result["serializable"]["type"] == "constructor"
        assert result["list"] == [1, 2, {"nested": "value"}]
        assert result["primitive"] == "string"

    def test_dumpd_equivalence_with_dumps(self) -> None:
        """Test dumpd() produces same result as json.loads(dumps())."""
        obj = SerializableTest(value=42)

        dumpd_result = dumpd(obj)
        dumps_result = json.loads(dumps(obj))

        assert dumpd_result == dumps_result

    def test_dumpd_with_primitive_types(self) -> None:
        """Test dumpd() with primitive types."""
        # List
        result = dumpd([1, 2, 3])
        assert result == [1, 2, 3]

        # Dict
        result = dumpd({"key": "value"})
        assert result == {"key": "value"}

        # Serializable object in dict
        data = {"obj": SerializableTest(value=42)}
        result = dumpd(data)
        assert result["obj"]["type"] == "constructor"

    def test_dumpd_with_pydantic_model_in_chat_generation(self) -> None:
        """Test dumpd() handles Pydantic models in ChatGeneration."""

        class MyModel(BaseModel):
            x: int
            y: str

        my_model = MyModel(x=1, y="hello")
        chat_gen = ChatGeneration(
            message=AIMessage(
                content='{"x": 1, "y": "hello"}', additional_kwargs={"parsed": my_model}
            )
        )

        result = dumpd(chat_gen)

        # parsed should be dumped as dict
        assert result["kwargs"]["message"]["kwargs"]["additional_kwargs"]["parsed"] == {
            "x": 1,
            "y": "hello",
        }

    def test_dumpd_list_of_serializable(self) -> None:
        """Test dumpd() with a list of Serializable objects."""
        objs = [SerializableTest(value=1), SerializableTest(value=2)]

        result = dumpd(objs)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["kwargs"]["value"] == 1
        assert result[1]["kwargs"]["value"] == 2


# ---------------------------------------------------------------------------
# Additional snapshot tests for dump.py
# ---------------------------------------------------------------------------


class TestDefaultSnapshot:
    """Snapshot tests for the default() function output structure."""

    def test_default_serializable_full_snapshot(self) -> None:
        """Snapshot: default() output for a Serializable object."""
        obj = SerializableTest(value=99)
        result = default(obj)
        assert result == {
            "lc": 1,
            "type": "constructor",
            "id": [
                "tests",
                "unit_tests",
                "load",
                "test_dump",
                "SerializableTest",
            ],
            "kwargs": {"value": 99},
        }

    def test_default_non_serializable_full_snapshot(self) -> None:
        """Snapshot: default() output for a non-Serializable object."""
        obj = NonSerializableTest(value=7)
        result = default(obj)
        assert result["lc"] == 1
        assert result["type"] == "not_implemented"
        assert result["id"] == [
            "tests",
            "unit_tests",
            "load",
            "test_dump",
            "NonSerializableTest",
        ]
        assert "NonSerializableTest" in result["repr"]

    def test_default_with_builtin_type(self) -> None:
        """Snapshot: default() on a set (non-JSON-serializable builtin)."""
        result = default({1, 2, 3})
        assert result["lc"] == 1
        assert result["type"] == "not_implemented"
        assert "set" in result["id"][-1].lower()

    def test_default_with_none(self) -> None:
        """Snapshot: default() on None."""
        result = default(None)
        assert result["lc"] == 1
        assert result["type"] == "not_implemented"


class TestDumpPydanticModelsSnapshot:
    """Snapshot tests for _dump_pydantic_models edge cases."""

    def test_dump_pydantic_models_parsed_is_none(self) -> None:
        """Test _dump_pydantic_models when parsed key exists but is None."""
        chat_gen = ChatGeneration(
            message=AIMessage(content="test", additional_kwargs={"parsed": None})
        )
        result = _dump_pydantic_models(chat_gen)
        # None is not a BaseModel, so it should return the same object
        assert result is chat_gen

    def test_dump_pydantic_models_parsed_is_dict(self) -> None:
        """Test _dump_pydantic_models when parsed is already a dict."""
        chat_gen = ChatGeneration(
            message=AIMessage(content="test", additional_kwargs={"parsed": {"x": 1}})
        )
        result = _dump_pydantic_models(chat_gen)
        # Dict is not a BaseModel, so it should return the same object
        assert result is chat_gen

    def test_dump_pydantic_models_non_ai_message(self) -> None:
        """Test _dump_pydantic_models with a non-AIMessage in ChatGeneration."""
        from langchain_core.messages import HumanMessage

        chat_gen = ChatGeneration(message=HumanMessage(content="test"))
        result = _dump_pydantic_models(chat_gen)
        assert result is chat_gen

    def test_dump_pydantic_models_preserves_original(self) -> None:
        """Snapshot: _dump_pydantic_models deep-copies and does not mutate original."""

        class Inner(BaseModel):
            val: int

        model = Inner(val=42)
        chat_gen = ChatGeneration(
            message=AIMessage(content="x", additional_kwargs={"parsed": model})
        )
        result = _dump_pydantic_models(chat_gen)

        # Result should have the dict form
        assert result.message.additional_kwargs["parsed"] == {"val": 42}
        # Original should still have the BaseModel
        assert isinstance(chat_gen.message.additional_kwargs["parsed"], Inner)
        assert chat_gen.message.additional_kwargs["parsed"].val == 42

    def test_dump_pydantic_models_with_string(self) -> None:
        """Test _dump_pydantic_models with a plain string."""
        result = _dump_pydantic_models("hello")
        assert result == "hello"

    def test_dump_pydantic_models_with_list(self) -> None:
        """Test _dump_pydantic_models with a list."""
        result = _dump_pydantic_models([1, 2, 3])
        assert result == [1, 2, 3]


class TestDumpsSnapshot:
    """Snapshot tests for dumps() output structure."""

    def test_dumps_serializable_full_snapshot(self) -> None:
        """Snapshot: full JSON output of dumps() for a Serializable."""
        obj = SerializableTest(value=42)
        json_str = dumps(obj)
        parsed = json.loads(json_str)
        assert parsed == {
            "lc": 1,
            "type": "constructor",
            "id": [
                "tests",
                "unit_tests",
                "load",
                "test_dump",
                "SerializableTest",
            ],
            "kwargs": {"value": 42},
        }

    def test_dumps_non_serializable_full_snapshot(self) -> None:
        """Snapshot: full JSON output of dumps() for a non-Serializable object."""
        obj = NonSerializableTest(value=7)
        json_str = dumps(obj)
        parsed = json.loads(json_str)
        assert parsed["lc"] == 1
        assert parsed["type"] == "not_implemented"
        assert parsed["id"] == [
            "tests",
            "unit_tests",
            "load",
            "test_dump",
            "NonSerializableTest",
        ]

    def test_dumps_pretty_default_indent_is_2(self) -> None:
        """Snapshot: pretty=True uses 2-space indent by default."""
        obj = SerializableTest(value=1)
        pretty = dumps(obj, pretty=True)
        lines = pretty.split("\n")
        # Second line should start with 2 spaces (standard JSON indent)
        assert lines[1].startswith("  ")

    def test_dumps_with_none(self) -> None:
        """Snapshot: dumps() on None returns JSON 'null'."""
        result = dumps(None)
        assert result == "null"

    def test_dumps_with_plain_string(self) -> None:
        """Snapshot: dumps() on a string returns JSON string."""
        result = dumps("hello")
        assert result == '"hello"'

    def test_dumps_with_plain_int(self) -> None:
        """Snapshot: dumps() on an int returns JSON number."""
        result = dumps(42)
        assert result == "42"

    def test_dumps_with_plain_dict(self) -> None:
        """Snapshot: dumps() on a dict returns JSON object."""
        result = dumps({"a": 1, "b": 2})
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": 2}

    def test_dumps_with_plain_list(self) -> None:
        """Snapshot: dumps() on a list returns JSON array."""
        result = dumps([1, "two", 3])
        parsed = json.loads(result)
        assert parsed == [1, "two", 3]

    def test_dumps_bool_values(self) -> None:
        """Snapshot: dumps() with bool values."""
        assert dumps(True) == "true"
        assert dumps(False) == "false"


class TestDumpdSnapshot:
    """Snapshot tests for dumpd() output structure."""

    def test_dumpd_none(self) -> None:
        """Snapshot: dumpd() with None returns None."""
        assert dumpd(None) is None

    def test_dumpd_string(self) -> None:
        """Snapshot: dumpd() with plain string returns the string."""
        assert dumpd("hello") == "hello"

    def test_dumpd_int(self) -> None:
        """Snapshot: dumpd() with int returns the int."""
        assert dumpd(42) == 42

    def test_dumpd_bool(self) -> None:
        """Snapshot: dumpd() with booleans returns booleans."""
        assert dumpd(True) is True
        assert dumpd(False) is False

    def test_dumpd_float(self) -> None:
        """Snapshot: dumpd() with float returns float."""
        assert dumpd(3.14) == 3.14

    def test_dumpd_nested_serializable_in_list(self) -> None:
        """Snapshot: dumpd() with Serializable inside a list inside a dict."""
        data = {"items": [SerializableTest(value=10), "plain", 42]}
        result = dumpd(data)
        assert result["items"][0]["type"] == "constructor"
        assert result["items"][0]["kwargs"]["value"] == 10
        assert result["items"][1] == "plain"
        assert result["items"][2] == 42

    def test_dumpd_empty_structures(self) -> None:
        """Snapshot: dumpd() with empty dict and list."""
        assert dumpd({}) == {}
        assert dumpd([]) == []

    def test_dumpd_serializable_full_snapshot(self) -> None:
        """Snapshot: exact dumpd output for SerializableTest."""
        obj = SerializableTest(value=42)
        result = dumpd(obj)
        assert result == {
            "lc": 1,
            "type": "constructor",
            "id": [
                "tests",
                "unit_tests",
                "load",
                "test_dump",
                "SerializableTest",
            ],
            "kwargs": {"value": 42},
        }
