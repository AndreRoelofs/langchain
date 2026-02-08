"""Unit tests for retriever tool functionality."""

import pytest
from typing_extensions import override

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import create_retriever_tool
from langchain_core.tools.retriever import RetrieverInput


class MockRetriever(BaseRetriever):
    """Mock retriever for testing."""

    @override
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Return mock documents."""
        return [
            Document(page_content=f"Document 1 for {query}"),
            Document(page_content=f"Document 2 for {query}"),
        ]


class EmptyRetriever(BaseRetriever):
    """Retriever that returns no documents."""

    @override
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Return empty list."""
        return []


class ContextRetriever(BaseRetriever):
    """Retriever with metadata."""

    @override
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Return documents with metadata."""
        return [
            Document(
                page_content=f"Result for {query}",
                metadata={"source": "test", "score": 0.95},
            ),
        ]


def test_retriever_input_schema() -> None:
    """Test RetrieverInput schema structure."""
    # Test instantiation
    input_obj = RetrieverInput(query="test query")
    assert input_obj.query == "test query"

    # Test schema has description
    schema = RetrieverInput.model_json_schema()
    assert "properties" in schema
    assert "query" in schema["properties"]
    assert "description" in schema["properties"]["query"]


def test_create_retriever_tool_basic() -> None:
    """Test basic retriever tool creation."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="test_retriever",
        description="A test retriever tool",
    )

    assert tool.name == "test_retriever"
    assert tool.description == "A test retriever tool"
    assert tool.args_schema == RetrieverInput


def test_create_retriever_tool_invoke_string() -> None:
    """Test retriever tool with string input."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="test_retriever",
        description="A test retriever",
    )

    result = tool.invoke("test query")

    assert isinstance(result, str)
    assert "Document 1 for test query" in result
    assert "Document 2 for test query" in result
    assert "\n\n" in result  # Default separator


def test_create_retriever_tool_invoke_dict() -> None:
    """Test retriever tool with dict input."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="test_retriever",
        description="A test retriever",
    )

    result = tool.invoke({"query": "test query"})

    assert isinstance(result, str)
    assert "Document 1 for test query" in result
    assert "Document 2 for test query" in result


def test_create_retriever_tool_with_tool_call() -> None:
    """Test retriever tool with ToolCall input."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="test_retriever",
        description="A test retriever",
    )

    tool_call: ToolCall = {
        "name": "test_retriever",
        "args": {"query": "test query"},
        "id": "call_123",
        "type": "tool_call",
    }

    result = tool.invoke(tool_call)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call_123"
    assert result.name == "test_retriever"
    assert "Document 1 for test query" in result.content


def test_create_retriever_tool_custom_separator() -> None:
    """Test retriever tool with custom document separator."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="test_retriever",
        description="A test retriever",
        document_separator="\n---\n",
    )

    result = tool.invoke("test query")

    assert "\n---\n" in result
    assert "Document 1 for test query\n---\nDocument 2 for test query" in result


def test_create_retriever_tool_custom_prompt() -> None:
    """Test retriever tool with custom document prompt."""
    retriever = MockRetriever()
    custom_prompt = PromptTemplate.from_template("Content: {page_content}")

    tool = create_retriever_tool(
        retriever=retriever,
        name="test_retriever",
        description="A test retriever",
        document_prompt=custom_prompt,
    )

    result = tool.invoke("test query")

    assert "Content: Document 1 for test query" in result
    assert "Content: Document 2 for test query" in result


def test_create_retriever_tool_empty_results() -> None:
    """Test retriever tool when no documents are found."""
    retriever = EmptyRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="empty_retriever",
        description="Returns no results",
    )

    result = tool.invoke("test query")

    assert result == ""


def test_create_retriever_tool_content_format() -> None:
    """Test retriever tool with content response format."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="test_retriever",
        description="A test retriever",
        response_format="content",
    )

    result = tool.invoke("test query")

    # Content format returns just the string
    assert isinstance(result, str)
    assert "Document 1 for test query" in result


def test_create_retriever_tool_content_and_artifact_format() -> None:
    """Test retriever tool with content_and_artifact response format."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="test_retriever",
        description="A test retriever",
        response_format="content_and_artifact",
    )

    # When invoked without ToolCall, returns string content directly
    result = tool.invoke("test query")
    assert isinstance(result, str)
    assert "Document 1 for test query" in result


def test_create_retriever_tool_artifact_with_tool_call() -> None:
    """Test retriever tool with artifact format and ToolCall input."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="test_retriever",
        description="A test retriever",
        response_format="content_and_artifact",
    )

    tool_call: ToolCall = {
        "name": "test_retriever",
        "args": {"query": "test query"},
        "id": "call_456",
        "type": "tool_call",
    }

    result = tool.invoke(tool_call)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call_456"
    assert result.name == "test_retriever"
    assert "Document 1 for test query" in result.content
    assert result.artifact is not None
    assert len(result.artifact) == 2
    assert all(isinstance(doc, Document) for doc in result.artifact)


def test_retriever_tool_with_metadata() -> None:
    """Test retriever that returns documents with metadata."""
    retriever = ContextRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="context_retriever",
        description="Retriever with metadata",
    )

    result = tool.invoke("test")

    assert "Result for test" in result


def test_retriever_tool_artifact_preserves_metadata() -> None:
    """Test that artifact format preserves document metadata with ToolCall."""
    retriever = ContextRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="context_retriever",
        description="Retriever with metadata",
        response_format="content_and_artifact",
    )

    tool_call: ToolCall = {
        "name": "context_retriever",
        "args": {"query": "test"},
        "id": "call_789",
        "type": "tool_call",
    }

    result = tool.invoke(tool_call)
    assert isinstance(result, ToolMessage)
    assert result.artifact is not None
    assert len(result.artifact) == 1
    doc = result.artifact[0]
    assert doc.metadata["source"] == "test"
    assert doc.metadata["score"] == 0.95


async def test_retriever_tool_async_invoke() -> None:
    """Test retriever tool async invocation."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="async_retriever",
        description="Async retriever tool",
    )

    result = await tool.ainvoke("test query")

    assert isinstance(result, str)
    assert "Document 1 for test query" in result
    assert "Document 2 for test query" in result


async def test_retriever_tool_async_with_artifact() -> None:
    """Test retriever tool async invocation with artifact format."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="async_retriever",
        description="Async retriever tool",
        response_format="content_and_artifact",
    )

    # Without ToolCall, returns string content
    result = await tool.ainvoke("test query")

    assert isinstance(result, str)
    assert "Document 1 for test query" in result


def test_retriever_tool_preserves_document_order() -> None:
    """Test that retriever tool preserves document order."""

    class OrderedRetriever(BaseRetriever):
        @override
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> list[Document]:
            return [
                Document(page_content="First"),
                Document(page_content="Second"),
                Document(page_content="Third"),
            ]

    retriever = OrderedRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="ordered_retriever",
        description="Retriever that returns ordered documents",
    )

    result = tool.invoke("test")

    # Check order is preserved
    first_pos = result.find("First")
    second_pos = result.find("Second")
    third_pos = result.find("Third")

    assert first_pos < second_pos < third_pos


def test_retriever_tool_custom_prompt_without_metadata() -> None:
    """Test custom prompt template with simple formatting."""
    retriever = ContextRetriever()
    # Use a simple template that doesn't require metadata access
    custom_prompt = PromptTemplate.from_template("Content: {page_content}")

    tool = create_retriever_tool(
        retriever=retriever,
        name="custom_retriever",
        description="Retriever with custom prompt",
        document_prompt=custom_prompt,
    )

    result = tool.invoke("test")

    assert "Content: Result for test" in result


# ---------------------------------------------------------------------------
# Retriever tool schema and properties
# ---------------------------------------------------------------------------


def test_retriever_tool_has_correct_schema() -> None:
    """Test that retriever tool has RetrieverInput as args_schema."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="schema_check",
        description="Schema check tool",
    )
    assert tool.args_schema is RetrieverInput
    schema = tool.args_schema.model_json_schema()
    assert "query" in schema["properties"]
    assert schema["properties"]["query"]["type"] == "string"


def test_retriever_tool_response_format_default_is_content() -> None:
    """Test that default response_format is 'content'."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="default_format",
        description="Default format tool",
    )
    assert tool.response_format == "content"


def test_retriever_tool_name_and_description_set() -> None:
    """Test retriever tool name and description are correctly set."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="my_retriever",
        description="My retriever description",
    )
    assert tool.name == "my_retriever"
    assert tool.description == "My retriever description"


# ---------------------------------------------------------------------------
# Retriever tool with custom separator edge cases
# ---------------------------------------------------------------------------


def test_retriever_tool_empty_separator() -> None:
    """Test retriever tool with empty string separator."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="no_sep",
        description="No separator",
        document_separator="",
    )
    result = tool.invoke("test")
    # Documents should be concatenated with no separator
    assert "Document 1 for test" in result
    assert "Document 2 for test" in result
    # No separator between docs
    assert "Document 1 for testDocument 2 for test" in result


def test_retriever_tool_single_char_separator() -> None:
    """Test retriever tool with single character separator."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="pipe_sep",
        description="Pipe separator",
        document_separator="|",
    )
    result = tool.invoke("test")
    assert "|" in result


# ---------------------------------------------------------------------------
# Retriever tool custom prompt with metadata access
# ---------------------------------------------------------------------------


def test_retriever_tool_prompt_with_metadata() -> None:
    """Test retriever tool with prompt template that accesses metadata."""
    retriever = ContextRetriever()
    custom_prompt = PromptTemplate.from_template("[{source}] {page_content}")
    tool = create_retriever_tool(
        retriever=retriever,
        name="meta_retriever",
        description="Metadata retriever",
        document_prompt=custom_prompt,
    )
    result = tool.invoke("test")
    assert "[test]" in result
    assert "Result for test" in result


# ---------------------------------------------------------------------------
# Retriever tool async with ToolCall
# ---------------------------------------------------------------------------


async def test_retriever_tool_async_with_tool_call() -> None:
    """Test retriever tool async invocation with ToolCall."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="async_tc_retriever",
        description="Async ToolCall retriever",
    )

    tool_call: ToolCall = {
        "name": "async_tc_retriever",
        "args": {"query": "async test"},
        "id": "call_async_tc",
        "type": "tool_call",
    }
    result = await tool.ainvoke(tool_call)
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call_async_tc"
    assert "Document 1 for async test" in result.content


async def test_retriever_tool_async_artifact_with_tool_call() -> None:
    """Test retriever tool async with artifact format and ToolCall."""
    retriever = MockRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="async_art_retriever",
        description="Async artifact retriever",
        response_format="content_and_artifact",
    )

    tool_call: ToolCall = {
        "name": "async_art_retriever",
        "args": {"query": "async art"},
        "id": "call_async_art",
        "type": "tool_call",
    }
    result = await tool.ainvoke(tool_call)
    assert isinstance(result, ToolMessage)
    assert result.artifact is not None
    assert len(result.artifact) == 2
    assert all(isinstance(doc, Document) for doc in result.artifact)


# ---------------------------------------------------------------------------
# Retriever tool empty results with artifact format
# ---------------------------------------------------------------------------


def test_retriever_tool_empty_results_artifact_format() -> None:
    """Test retriever tool with no results in content_and_artifact format."""
    retriever = EmptyRetriever()
    tool = create_retriever_tool(
        retriever=retriever,
        name="empty_art_retriever",
        description="Empty artifact retriever",
        response_format="content_and_artifact",
    )

    tc: ToolCall = {
        "name": "empty_art_retriever",
        "args": {"query": "nothing"},
        "id": "call_empty",
        "type": "tool_call",
    }
    result = tool.invoke(tc)
    assert isinstance(result, ToolMessage)
    assert result.content == ""
    assert result.artifact == []


# ---------------------------------------------------------------------------
# RetrieverInput validation
# ---------------------------------------------------------------------------


def test_retriever_input_empty_query() -> None:
    """Test RetrieverInput accepts empty string query."""
    ri = RetrieverInput(query="")
    assert ri.query == ""


def test_retriever_input_rejects_missing_query() -> None:
    """Test RetrieverInput requires query field."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RetrieverInput()  # type: ignore[call-arg]
