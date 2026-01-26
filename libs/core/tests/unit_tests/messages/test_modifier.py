"""Tests for langchain_core.messages.modifier module."""

import pytest

from langchain_core.load import dumpd, load
from langchain_core.messages.modifier import RemoveMessage


class TestRemoveMessage:
    """Tests for the RemoveMessage class."""

    def test_init_basic(self) -> None:
        """Test basic RemoveMessage initialization."""
        msg = RemoveMessage(id="msg-to-remove")
        assert msg.id == "msg-to-remove"
        assert msg.type == "remove"
        assert msg.content == ""

    def test_type_is_remove(self) -> None:
        """Test that RemoveMessage type is 'remove'."""
        msg = RemoveMessage(id="msg-123")
        assert msg.type == "remove"

    def test_content_is_empty_string(self) -> None:
        """Test that RemoveMessage content is always empty string."""
        msg = RemoveMessage(id="msg-123")
        assert msg.content == ""

    def test_content_kwarg_raises_error(self) -> None:
        """Test that passing content raises ValueError."""
        with pytest.raises(ValueError, match="does not support 'content' field"):
            RemoveMessage(id="msg-123", content="should fail")  # type: ignore[call-arg]

    def test_id_is_required(self) -> None:
        """Test that id is required for RemoveMessage."""
        with pytest.raises(Exception):  # Pydantic validation error
            RemoveMessage()  # type: ignore[call-arg]

    def test_serialization_roundtrip(self) -> None:
        """Test RemoveMessage serialization and deserialization."""
        msg = RemoveMessage(id="msg-to-remove")
        dumped = dumpd(msg)
        assert dumped["type"] == "constructor"
        assert dumped["id"] == ["langchain", "schema", "messages", "RemoveMessage"]

        loaded = load(dumped)
        assert isinstance(loaded, RemoveMessage)
        assert loaded.id == "msg-to-remove"
        assert loaded.content == ""

    def test_with_name(self) -> None:
        """Test RemoveMessage with name."""
        msg = RemoveMessage(id="msg-123", name="delete-marker")
        assert msg.name == "delete-marker"
        assert msg.id == "msg-123"

    def test_with_additional_kwargs(self) -> None:
        """Test RemoveMessage with additional_kwargs."""
        msg = RemoveMessage(id="msg-123", additional_kwargs={"reason": "outdated"})
        assert msg.additional_kwargs["reason"] == "outdated"

    def test_with_response_metadata(self) -> None:
        """Test RemoveMessage with response_metadata."""
        msg = RemoveMessage(id="msg-123", response_metadata={"deleted_at": "2024-01-01"})
        assert msg.response_metadata["deleted_at"] == "2024-01-01"

    def test_text_property_is_empty(self) -> None:
        """Test that .text property returns empty string."""
        msg = RemoveMessage(id="msg-123")
        assert msg.text == ""

    def test_content_blocks_property_is_empty(self) -> None:
        """Test that content_blocks property returns empty list."""
        msg = RemoveMessage(id="msg-123")
        blocks = msg.content_blocks
        assert blocks == []

    def test_pretty_repr(self) -> None:
        """Test pretty_repr output."""
        msg = RemoveMessage(id="msg-123")
        result = msg.pretty_repr()
        assert "Remove Message" in result

    def test_multiple_remove_messages(self) -> None:
        """Test creating multiple RemoveMessages for different IDs."""
        msg1 = RemoveMessage(id="msg-1")
        msg2 = RemoveMessage(id="msg-2")
        msg3 = RemoveMessage(id="msg-3")

        assert msg1.id == "msg-1"
        assert msg2.id == "msg-2"
        assert msg3.id == "msg-3"

        # All should have empty content
        assert msg1.content == ""
        assert msg2.content == ""
        assert msg3.content == ""


class TestRemoveMessageUseCases:
    """Tests for RemoveMessage use cases."""

    def test_remove_message_in_list(self) -> None:
        """Test using RemoveMessage in a list with other messages."""
        from langchain_core.messages import HumanMessage, AIMessage

        messages = [
            HumanMessage(content="Hello", id="human-1"),
            AIMessage(content="Hi there!", id="ai-1"),
            RemoveMessage(id="human-1"),  # Mark human-1 for removal
        ]

        assert len(messages) == 3
        assert isinstance(messages[2], RemoveMessage)
        assert messages[2].id == "human-1"

    def test_remove_message_serialization_in_list(self) -> None:
        """Test serialization of a list containing RemoveMessage."""
        from langchain_core.messages import HumanMessage

        messages = [
            HumanMessage(content="Hello", id="human-1"),
            RemoveMessage(id="human-1"),
        ]

        # Serialize both messages
        dumped = [dumpd(m) for m in messages]
        assert len(dumped) == 2
        assert dumped[0]["id"] == ["langchain", "schema", "messages", "HumanMessage"]
        assert dumped[1]["id"] == ["langchain", "schema", "messages", "RemoveMessage"]

        # Deserialize both messages
        loaded = [load(d) for d in dumped]
        assert isinstance(loaded[0], HumanMessage)
        assert isinstance(loaded[1], RemoveMessage)
        assert loaded[1].id == "human-1"

    def test_remove_message_does_not_modify_content(self) -> None:
        """Test that RemoveMessage doesn't allow content modification."""
        msg = RemoveMessage(id="msg-123")

        # Content should always be empty string
        assert msg.content == ""

        # Even if we try to access content_blocks, it should be empty
        assert msg.content_blocks == []
