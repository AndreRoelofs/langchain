"""Expanded tests for StructuredPrompt functionality."""

from functools import partial
from inspect import isclass
from typing import Any, cast

import pytest
from pydantic import BaseModel
from typing_extensions import override

from langchain_core.language_models import FakeListChatModel
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.messages import HumanMessage
from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.runnables.base import Runnable, RunnableLambda
from langchain_core.utils.pydantic import is_basemodel_subclass


def _fake_runnable(
    _: Any, *, schema: dict | type[BaseModel], value: Any = 42, **_kwargs: Any
) -> BaseModel | dict:
    if isclass(schema) and is_basemodel_subclass(schema):
        return schema(name="yo", value=value)
    params = cast("dict", schema)["parameters"]
    return {k: 1 if k != "value" else value for k, v in params.items()}


class FakeStructuredChatModel(FakeListChatModel):
    """Fake chat model for testing purposes."""

    @override
    def with_structured_output(
        self, schema: dict | type[BaseModel], **kwargs: Any
    ) -> Runnable:
        return RunnableLambda(partial(_fake_runnable, schema=schema, **kwargs))

    @property
    def _llm_type(self) -> str:
        return "fake-messages-list-chat-model"


# --- Initialization ---


def test_empty_schema_raises() -> None:
    """Test that empty schema raises ValueError."""
    with pytest.raises(ValueError, match="Must pass in a non-empty"):
        StructuredPrompt(
            [("human", "Hello")],
            schema_=None,
        )


def test_empty_dict_schema_raises() -> None:
    """Test that empty dict schema raises ValueError."""
    with pytest.raises(ValueError, match="Must pass in a non-empty"):
        StructuredPrompt(
            [("human", "Hello")],
            schema_={},
        )


def test_schema_from_kwarg() -> None:
    """Test that schema can be passed via 'schema' kwarg (legacy)."""
    prompt = StructuredPrompt(
        [("human", "Hello")],
        schema={"type": "object", "properties": {}, "title": "foo"},
    )
    assert prompt.schema_ is not None


def test_pydantic_schema() -> None:
    """Test StructuredPrompt with Pydantic model schema."""

    class OutputSchema(BaseModel):
        name: str
        value: int

    prompt = StructuredPrompt(
        [("human", "Hello {name}")],
        OutputSchema,
    )
    assert prompt.schema_ == OutputSchema
    assert "name" in prompt.input_variables


def test_dict_schema() -> None:
    """Test StructuredPrompt with dict schema."""
    prompt = StructuredPrompt(
        [("human", "Hello")],
        {
            "name": "test",
            "description": "a test",
            "parameters": {"name": {"type": "string"}},
        },
    )
    assert isinstance(prompt.schema_, dict)


def test_structured_output_kwargs() -> None:
    """Test that extra kwargs are passed to structured_output_kwargs."""
    prompt = StructuredPrompt(
        [("human", "Hello")],
        {"name": "test", "description": "a test", "parameters": {}},
        include_raw=True,
    )
    assert prompt.structured_output_kwargs.get("include_raw") is True


# --- get_lc_namespace ---


def test_get_lc_namespace() -> None:
    """Test get_lc_namespace returns module path."""
    ns = StructuredPrompt.get_lc_namespace()
    assert "prompts" in ".".join(ns)
    assert "structured" in ".".join(ns)


# --- pipe / __or__ ---


def test_pipe_with_llm() -> None:
    """Test piping StructuredPrompt to a language model."""

    class OutputSchema(BaseModel):
        name: str
        value: int

    prompt = StructuredPrompt(
        [("human", "Hello")],
        OutputSchema,
    )
    model = FakeStructuredChatModel(responses=[])
    chain = prompt | model
    result = chain.invoke({})
    assert isinstance(result, OutputSchema)
    assert result.name == "yo"


def test_pipe_with_non_llm_raises() -> None:
    """Test piping to non-LLM raises NotImplementedError."""
    prompt = StructuredPrompt(
        [("human", "Hello")],
        {"name": "test", "description": "a test", "parameters": {}},
    )
    with pytest.raises(
        NotImplementedError, match="Structured prompts need to be piped"
    ):
        prompt | RunnableLambda(lambda x: x)


def test_pipe_explicit_method() -> None:
    """Test using the explicit pipe() method."""

    class OutputSchema(BaseModel):
        name: str
        value: int

    prompt = StructuredPrompt(
        [("human", "Hello")],
        OutputSchema,
    )
    model = FakeStructuredChatModel(responses=[])
    chain = prompt.pipe(model)
    result = chain.invoke({})
    assert isinstance(result, OutputSchema)


# --- from_messages_and_schema ---


def test_from_messages_and_schema() -> None:
    """Test from_messages_and_schema class method."""

    class OutputSchema(BaseModel):
        name: str
        value: int

    prompt = StructuredPrompt.from_messages_and_schema(
        [("human", "Hello {name}")],
        OutputSchema,
    )
    assert isinstance(prompt, StructuredPrompt)
    assert prompt.schema_ == OutputSchema


# --- Serialization ---


def test_serialization_with_dict_schema() -> None:
    """Test serialization round-trip with dict schema."""
    prompt = StructuredPrompt(
        [("human", "Hello")],
        {
            "name": "test",
            "description": "a test",
            "parameters": {"name": {"type": "string"}},
        },
    )
    serialized = dumps(prompt)
    loaded = loads(serialized)
    assert loaded.model_dump() == prompt.model_dump()


def test_serialization_with_kwargs() -> None:
    """Test serialization preserves structured_output_kwargs."""
    prompt = StructuredPrompt(
        [("human", "Hello")],
        {
            "name": "test",
            "description": "a test",
            "parameters": {"name": {"type": "string"}, "value": {"type": "integer"}},
        },
        value=7,
    )
    serialized = dumps(prompt)
    loaded = loads(serialized)
    assert loaded.structured_output_kwargs == prompt.structured_output_kwargs


# --- Integration with ChatPromptTemplate features ---


def test_inherits_chat_template_features() -> None:
    """Test that StructuredPrompt inherits ChatPromptTemplate features."""

    class OutputSchema(BaseModel):
        name: str
        value: int

    prompt = StructuredPrompt(
        [
            ("system", "You are helpful"),
            ("human", "{input}"),
        ],
        OutputSchema,
    )
    assert "input" in prompt.input_variables
    messages = prompt.format_messages(input="Hello")
    assert len(messages) == 2


def test_template_format_mustache() -> None:
    """Test StructuredPrompt with mustache template format."""
    prompt = StructuredPrompt(
        [("human", "hi {{name}}")],
        schema={"type": "object", "properties": {}, "title": "foo"},
        template_format="mustache",
    )
    assert prompt.input_variables == ["name"]
    result = prompt.invoke({"name": "World"})
    messages = result.to_messages()
    assert messages[0].content == "hi World"
