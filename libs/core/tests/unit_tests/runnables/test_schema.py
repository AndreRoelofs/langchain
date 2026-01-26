"""Tests for runnable schema types and structures."""

from typing import Any

import pytest
from pydantic import BaseModel

from langchain_core.runnables.schema import (
    BaseStreamEvent,
    CustomStreamEvent,
    EventData,
    StandardStreamEvent,
    StreamEvent,
)


def test_event_data_structure() -> None:
    """Test EventData TypedDict structure."""
    # Test with all fields
    event_data: EventData = {
        "input": {"question": "test"},
        "output": {"answer": "response"},
        "chunk": {"partial": "data"},
    }

    assert event_data["input"] == {"question": "test"}
    assert event_data["output"] == {"answer": "response"}
    assert event_data["chunk"] == {"partial": "data"}

    # Test with minimal fields (all are optional except as noted)
    minimal_data: EventData = {}
    assert isinstance(minimal_data, dict)


def test_event_data_with_error() -> None:
    """Test EventData with error field."""
    error = ValueError("Test error")
    event_data: EventData = {
        "input": "test",
        "error": error,
    }

    assert event_data["error"] == error
    assert isinstance(event_data["error"], BaseException)


def test_base_stream_event_structure() -> None:
    """Test BaseStreamEvent TypedDict structure."""
    event: BaseStreamEvent = {
        "event": "on_chain_start",
        "run_id": "test-run-id",
        "parent_ids": [],
    }

    assert event["event"] == "on_chain_start"
    assert event["run_id"] == "test-run-id"
    assert event["parent_ids"] == []


def test_base_stream_event_with_optional_fields() -> None:
    """Test BaseStreamEvent with optional fields."""
    event: BaseStreamEvent = {
        "event": "on_chain_end",
        "run_id": "test-run-id",
        "parent_ids": ["parent-1", "parent-2"],
        "tags": ["tag1", "tag2"],
        "metadata": {"key": "value"},
    }

    assert event["tags"] == ["tag1", "tag2"]
    assert event["metadata"] == {"key": "value"}
    assert len(event["parent_ids"]) == 2


def test_standard_stream_event_structure() -> None:
    """Test StandardStreamEvent TypedDict structure."""
    event: StandardStreamEvent = {
        "event": "on_llm_start",
        "run_id": "test-run-id",
        "parent_ids": [],
        "name": "TestLLM",
        "data": {"input": "test input"},
    }

    assert event["name"] == "TestLLM"
    assert event["data"]["input"] == "test input"


def test_standard_stream_event_types() -> None:
    """Test various standard stream event types."""
    # Test different event types
    event_types = [
        "on_llm_start",
        "on_llm_stream",
        "on_llm_end",
        "on_chat_model_start",
        "on_chat_model_stream",
        "on_chat_model_end",
        "on_chain_start",
        "on_chain_stream",
        "on_chain_end",
        "on_tool_start",
        "on_tool_stream",
        "on_tool_end",
        "on_retriever_start",
        "on_retriever_stream",
        "on_retriever_end",
        "on_prompt_start",
        "on_prompt_end",
    ]

    for event_type in event_types:
        event: StandardStreamEvent = {
            "event": event_type,
            "run_id": "test-id",
            "parent_ids": [],
            "name": "test",
            "data": {},
        }
        assert event["event"] == event_type


def test_custom_stream_event_structure() -> None:
    """Test CustomStreamEvent TypedDict structure."""
    event: CustomStreamEvent = {
        "event": "on_custom_event",
        "run_id": "test-run-id",
        "parent_ids": [],
        "name": "my_custom_event",
        "data": {"custom_field": "custom_value"},
    }

    assert event["event"] == "on_custom_event"
    assert event["name"] == "my_custom_event"
    assert event["data"]["custom_field"] == "custom_value"


def test_custom_stream_event_with_any_data() -> None:
    """Test CustomStreamEvent can have any type of data."""
    # String data
    event1: CustomStreamEvent = {
        "event": "on_custom_event",
        "run_id": "id1",
        "parent_ids": [],
        "name": "event1",
        "data": "string data",
    }
    assert event1["data"] == "string data"

    # List data
    event2: CustomStreamEvent = {
        "event": "on_custom_event",
        "run_id": "id2",
        "parent_ids": [],
        "name": "event2",
        "data": [1, 2, 3],
    }
    assert event2["data"] == [1, 2, 3]

    # Complex nested data
    event3: CustomStreamEvent = {
        "event": "on_custom_event",
        "run_id": "id3",
        "parent_ids": [],
        "name": "event3",
        "data": {"nested": {"deeply": {"value": 42}}},
    }
    assert event3["data"]["nested"]["deeply"]["value"] == 42


def test_stream_event_union_type() -> None:
    """Test that StreamEvent accepts both Standard and Custom events."""
    # Standard event
    standard: StreamEvent = {
        "event": "on_chain_start",
        "run_id": "id",
        "parent_ids": [],
        "name": "chain",
        "data": {"input": "test"},
    }
    assert standard["event"] == "on_chain_start"

    # Custom event
    custom: StreamEvent = {
        "event": "on_custom_event",
        "run_id": "id",
        "parent_ids": [],
        "name": "custom",
        "data": "anything",
    }
    assert custom["event"] == "on_custom_event"


def test_event_data_chunk_field() -> None:
    """Test EventData chunk field for streaming."""
    event_data: EventData = {
        "chunk": "partial output",
    }

    assert event_data["chunk"] == "partial output"

    # Chunks can be any type
    chunk_list: EventData = {"chunk": [1, 2, 3]}
    assert chunk_list["chunk"] == [1, 2, 3]


def test_parent_ids_hierarchy() -> None:
    """Test parent_ids represents hierarchy from root to immediate parent."""
    # Root event (no parents)
    root_event: BaseStreamEvent = {
        "event": "on_chain_start",
        "run_id": "root-id",
        "parent_ids": [],
    }
    assert len(root_event["parent_ids"]) == 0

    # Child event
    child_event: BaseStreamEvent = {
        "event": "on_chain_start",
        "run_id": "child-id",
        "parent_ids": ["root-id"],
    }
    assert child_event["parent_ids"] == ["root-id"]

    # Grandchild event
    grandchild_event: BaseStreamEvent = {
        "event": "on_chain_start",
        "run_id": "grandchild-id",
        "parent_ids": ["root-id", "child-id"],
    }
    assert grandchild_event["parent_ids"] == ["root-id", "child-id"]


def test_metadata_serializable() -> None:
    """Test that metadata should be JSON-serializable."""
    event: StandardStreamEvent = {
        "event": "on_chain_start",
        "run_id": "id",
        "parent_ids": [],
        "name": "test",
        "data": {},
        "metadata": {
            "string": "value",
            "number": 42,
            "boolean": True,
            "null": None,
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        },
    }

    # All should be JSON-serializable types
    import json

    serialized = json.dumps(event["metadata"])
    deserialized = json.loads(serialized)
    assert deserialized == event["metadata"]


def test_tags_list_of_strings() -> None:
    """Test that tags are list of strings."""
    event: StandardStreamEvent = {
        "event": "on_chain_start",
        "run_id": "id",
        "parent_ids": [],
        "name": "test",
        "data": {},
        "tags": ["tag1", "tag2", "tag3"],
    }

    assert isinstance(event["tags"], list)
    assert all(isinstance(tag, str) for tag in event["tags"])


def test_event_with_pydantic_model_in_data() -> None:
    """Test that event data can contain Pydantic models."""

    class OutputModel(BaseModel):
        result: str
        confidence: float

    event_data: EventData = {
        "output": OutputModel(result="test", confidence=0.95),
    }

    assert isinstance(event_data["output"], OutputModel)
    assert event_data["output"].result == "test"
    assert event_data["output"].confidence == 0.95


def test_standard_event_naming_convention() -> None:
    """Test event naming convention follows pattern."""
    # Format: on_[runnable_type]_(start|stream|end)
    valid_patterns = [
        "on_llm_start",
        "on_llm_stream",
        "on_llm_end",
        "on_chat_model_start",
        "on_chat_model_stream",
        "on_chat_model_end",
        "on_chain_start",
        "on_chain_stream",
        "on_chain_end",
        "on_tool_start",
        "on_tool_end",
        "on_retriever_start",
        "on_retriever_end",
        "on_prompt_start",
        "on_prompt_end",
    ]

    for pattern in valid_patterns:
        event: StandardStreamEvent = {
            "event": pattern,
            "run_id": "id",
            "parent_ids": [],
            "name": "test",
            "data": {},
        }
        # Verify it follows the pattern
        assert event["event"].startswith("on_")
        parts = event["event"].split("_")
        assert parts[0] == "on"
        assert parts[-1] in ["start", "stream", "end"]


def test_custom_event_must_be_on_custom_event() -> None:
    """Test that CustomStreamEvent event field must be 'on_custom_event'."""
    event: CustomStreamEvent = {
        "event": "on_custom_event",
        "run_id": "id",
        "parent_ids": [],
        "name": "my_event",
        "data": {},
    }

    # This is enforced by the Literal type
    assert event["event"] == "on_custom_event"


def test_event_data_supports_various_input_types() -> None:
    """Test EventData input field supports various types."""
    # String input
    event1: EventData = {"input": "simple string"}
    assert event1["input"] == "simple string"

    # Dict input
    event2: EventData = {"input": {"key": "value"}}
    assert event2["input"]["key"] == "value"

    # List input
    event3: EventData = {"input": [1, 2, 3]}
    assert event3["input"] == [1, 2, 3]

    # Complex object input
    class ComplexInput(BaseModel):
        field1: str
        field2: int

    event4: EventData = {"input": ComplexInput(field1="test", field2=42)}
    assert event4["input"].field1 == "test"


def test_event_data_supports_various_output_types() -> None:
    """Test EventData output field supports various types."""
    # String output
    event1: EventData = {"output": "result"}
    assert event1["output"] == "result"

    # Dict output
    event2: EventData = {"output": {"result": "value"}}
    assert event2["output"]["result"] == "value"

    # List output
    event3: EventData = {"output": [1, 2, 3]}
    assert event3["output"] == [1, 2, 3]


def test_event_minimal_required_fields() -> None:
    """Test creating events with only required fields."""
    # BaseStreamEvent minimal
    base: BaseStreamEvent = {
        "event": "on_chain_start",
        "run_id": "id",
        "parent_ids": [],
    }
    assert "event" in base
    assert "run_id" in base

    # StandardStreamEvent minimal
    standard: StandardStreamEvent = {
        "event": "on_chain_start",
        "run_id": "id",
        "parent_ids": [],
        "name": "test",
        "data": {},
    }
    assert "name" in standard
    assert "data" in standard

    # CustomStreamEvent minimal
    custom: CustomStreamEvent = {
        "event": "on_custom_event",
        "run_id": "id",
        "parent_ids": [],
        "name": "custom",
        "data": "any",
    }
    assert custom["event"] == "on_custom_event"


def test_event_all_optional_fields() -> None:
    """Test event with all optional fields populated."""
    event: StandardStreamEvent = {
        "event": "on_chain_start",
        "run_id": "test-run-id",
        "parent_ids": ["parent-1", "parent-2"],
        "name": "TestChain",
        "data": {
            "input": {"query": "test"},
            "output": {"response": "result"},
            "chunk": {"partial": "data"},
        },
        "tags": ["tag1", "tag2"],
        "metadata": {"version": "1.0", "environment": "test"},
    }

    assert len(event["parent_ids"]) == 2
    assert len(event["tags"]) == 2
    assert len(event["metadata"]) == 2


def test_stream_event_can_be_either_type() -> None:
    """Test that StreamEvent union accepts both types."""

    def process_event(event: StreamEvent) -> str:
        """Process any type of stream event."""
        return event["event"]

    standard: StandardStreamEvent = {
        "event": "on_llm_start",
        "run_id": "id",
        "parent_ids": [],
        "name": "llm",
        "data": {},
    }

    custom: CustomStreamEvent = {
        "event": "on_custom_event",
        "run_id": "id",
        "parent_ids": [],
        "name": "custom",
        "data": {},
    }

    assert process_event(standard) == "on_llm_start"
    assert process_event(custom) == "on_custom_event"


def test_event_data_empty() -> None:
    """Test EventData can be completely empty."""
    empty_data: EventData = {}
    assert len(empty_data) == 0


def test_custom_event_data_can_be_none() -> None:
    """Test that custom event data can be None."""
    event: CustomStreamEvent = {
        "event": "on_custom_event",
        "run_id": "id",
        "parent_ids": [],
        "name": "event",
        "data": None,
    }

    assert event["data"] is None


def test_event_metadata_empty_dict() -> None:
    """Test event with empty metadata dict."""
    event: StandardStreamEvent = {
        "event": "on_chain_start",
        "run_id": "id",
        "parent_ids": [],
        "name": "test",
        "data": {},
        "metadata": {},
    }

    assert event["metadata"] == {}


def test_event_tags_empty_list() -> None:
    """Test event with empty tags list."""
    event: StandardStreamEvent = {
        "event": "on_chain_start",
        "run_id": "id",
        "parent_ids": [],
        "name": "test",
        "data": {},
        "tags": [],
    }

    assert event["tags"] == []


def test_event_run_id_format() -> None:
    """Test that run_id is a string (typically UUID)."""
    import uuid

    run_id = str(uuid.uuid4())
    event: BaseStreamEvent = {
        "event": "on_chain_start",
        "run_id": run_id,
        "parent_ids": [],
    }

    assert isinstance(event["run_id"], str)
    # Verify it's a valid UUID string
    uuid.UUID(event["run_id"])


def test_standard_event_with_all_data_fields() -> None:
    """Test standard event with all possible EventData fields."""
    event: StandardStreamEvent = {
        "event": "on_chain_end",
        "run_id": "id",
        "parent_ids": [],
        "name": "chain",
        "data": {
            "input": {"query": "test"},
            "output": {"result": "answer"},
            "chunk": {"partial": "data"},
            "error": ValueError("test error"),
        },
    }

    assert "input" in event["data"]
    assert "output" in event["data"]
    assert "chunk" in event["data"]
    assert "error" in event["data"]


def test_custom_event_name_can_be_any_string() -> None:
    """Test that custom event names can be any user-defined string."""
    event_names = [
        "my_custom_event",
        "progress_update",
        "step_completed",
        "intermediate_result",
        "debug_info",
    ]

    for name in event_names:
        event: CustomStreamEvent = {
            "event": "on_custom_event",
            "run_id": "id",
            "parent_ids": [],
            "name": name,
            "data": {},
        }
        assert event["name"] == name


def test_event_parent_ids_can_be_nested() -> None:
    """Test deeply nested parent_ids hierarchy."""
    # Simulate a deep call stack
    parent_chain = [f"parent-{i}" for i in range(10)]

    event: BaseStreamEvent = {
        "event": "on_chain_start",
        "run_id": "leaf-id",
        "parent_ids": parent_chain,
    }

    assert len(event["parent_ids"]) == 10
    assert event["parent_ids"][0] == "parent-0"
    assert event["parent_ids"][-1] == "parent-9"


def test_event_data_with_base_message() -> None:
    """Test event data with LangChain message types."""
    from langchain_core.messages import AIMessage, HumanMessage

    event_data: EventData = {
        "input": [HumanMessage(content="hello")],
        "output": AIMessage(content="hi there"),
        "chunk": AIMessage(content="hi"),
    }

    assert isinstance(event_data["input"][0], HumanMessage)
    assert isinstance(event_data["output"], AIMessage)
    assert isinstance(event_data["chunk"], AIMessage)


def test_standard_event_data_field_required() -> None:
    """Test that StandardStreamEvent requires data field."""
    # This should be valid
    event: StandardStreamEvent = {
        "event": "on_chain_start",
        "run_id": "id",
        "parent_ids": [],
        "name": "test",
        "data": {},
    }
    assert "data" in event


def test_custom_event_data_field_required() -> None:
    """Test that CustomStreamEvent requires data field."""
    event: CustomStreamEvent = {
        "event": "on_custom_event",
        "run_id": "id",
        "parent_ids": [],
        "name": "test",
        "data": {"info": "required"},
    }
    assert "data" in event


def test_event_with_multiple_chunks() -> None:
    """Test events can represent multiple streaming chunks."""
    chunks_sequence: list[EventData] = [
        {"chunk": "Hello"},
        {"chunk": " "},
        {"chunk": "World"},
        {"chunk": "!"},
    ]

    accumulated = "".join(chunk["chunk"] for chunk in chunks_sequence)
    assert accumulated == "Hello World!"


def test_event_metadata_nested_structure() -> None:
    """Test event metadata with nested structures."""
    event: StandardStreamEvent = {
        "event": "on_llm_start",
        "run_id": "id",
        "parent_ids": [],
        "name": "llm",
        "data": {},
        "metadata": {
            "model_info": {
                "provider": "openai",
                "model": "gpt-4",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 100,
                },
            },
            "user_info": {
                "user_id": "123",
                "session_id": "456",
            },
        },
    }

    assert event["metadata"]["model_info"]["provider"] == "openai"
    assert event["metadata"]["user_info"]["user_id"] == "123"


def test_event_tags_inherited_from_parent() -> None:
    """Test documentation that tags are inherited from parent runnables."""
    # This is a documentation test - tags should be accumulated
    parent_event: StandardStreamEvent = {
        "event": "on_chain_start",
        "run_id": "parent",
        "parent_ids": [],
        "name": "parent",
        "data": {},
        "tags": ["parent-tag"],
    }

    child_event: StandardStreamEvent = {
        "event": "on_chain_start",
        "run_id": "child",
        "parent_ids": ["parent"],
        "name": "child",
        "data": {},
        "tags": ["parent-tag", "child-tag"],  # Inherited + own
    }

    assert "parent-tag" in child_event["tags"]
    assert "child-tag" in child_event["tags"]
