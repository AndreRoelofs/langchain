from langchain_core.messages import __all__

EXPECTED_ALL = [
    "MessageLikeRepresentation",
    "_message_from_dict",
    "AIMessage",
    "AIMessageChunk",
    "Annotation",
    "AnyMessage",
    "AudioContentBlock",
    "BaseMessage",
    "BaseMessageChunk",
    "ContentBlock",
    "ChatMessage",
    "ChatMessageChunk",
    "Citation",
    "DataContentBlock",
    "FileContentBlock",
    "FunctionMessage",
    "FunctionMessageChunk",
    "HumanMessage",
    "HumanMessageChunk",
    "ImageContentBlock",
    "InvalidToolCall",
    "LC_AUTO_PREFIX",
    "LC_ID_PREFIX",
    "NonStandardAnnotation",
    "NonStandardContentBlock",
    "PlainTextContentBlock",
    "ServerToolCall",
    "ServerToolCallChunk",
    "ServerToolResult",
    "SystemMessage",
    "SystemMessageChunk",
    "TextContentBlock",
    "ToolCall",
    "ToolCallChunk",
    "ToolMessage",
    "ToolMessageChunk",
    "VideoContentBlock",
    "ReasoningContentBlock",
    "RemoveMessage",
    "convert_to_messages",
    "ensure_id",
    "get_buffer_string",
    "is_data_content_block",
    "merge_content",
    "message_chunk_to_message",
    "message_to_dict",
    "messages_from_dict",
    "messages_to_dict",
    "filter_messages",
    "merge_message_runs",
    "trim_messages",
    "convert_to_openai_data_block",
    "convert_to_openai_image_block",
    "convert_to_openai_messages",
    "UsageMetadata",
    "InputTokenDetails",
    "OutputTokenDetails",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)


def test_each_name_in_all_can_be_imported() -> None:
    """Test that each name in __all__ can actually be imported dynamically."""
    import importlib

    messages_module = importlib.import_module("langchain_core.messages")
    for name in __all__:
        obj = getattr(messages_module, name)
        assert obj is not None, f"Failed to import {name} from langchain_core.messages"


def test_dir_matches_all() -> None:
    """Test that __dir__() returns the same items as __all__."""
    import langchain_core.messages as messages_mod

    dir_items = dir(messages_mod)
    assert set(dir_items) == set(__all__)


def test_lazy_loading_specific_items() -> None:
    """Test that lazy loading works by importing specific items and checking types."""
    import langchain_core.messages as messages_mod

    # Check that classes are actually class types
    assert isinstance(messages_mod.AIMessage, type)
    assert isinstance(messages_mod.HumanMessage, type)
    assert isinstance(messages_mod.SystemMessage, type)
    assert isinstance(messages_mod.ToolMessage, type)
    assert isinstance(messages_mod.FunctionMessage, type)
    assert isinstance(messages_mod.RemoveMessage, type)
    assert isinstance(messages_mod.BaseMessage, type)
    assert isinstance(messages_mod.BaseMessageChunk, type)

    # Check that callable items are callable
    assert callable(messages_mod.convert_to_messages)
    assert callable(messages_mod.get_buffer_string)
    assert callable(messages_mod.filter_messages)
    assert callable(messages_mod.merge_message_runs)
    assert callable(messages_mod.trim_messages)
