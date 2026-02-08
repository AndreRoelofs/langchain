"""Expanded tests for message prompt template functionality."""

from pathlib import Path
from unittest import mock

import pytest

from langchain_core.messages import AIMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    BaseStringMessagePromptTemplate,
    ChatMessagePromptTemplate,
)
from langchain_core.prompts.message import BaseMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

# --- BaseMessagePromptTemplate ---


def test_pretty_repr_raises_not_implemented() -> None:
    """Test that pretty_repr raises NotImplementedError on base class."""
    template = HumanMessagePromptTemplate.from_template("Hello {name}")
    # HumanMessagePromptTemplate overrides pretty_repr, so test the base
    # through an approach that hits the base class

    class MinimalTemplate(BaseMessagePromptTemplate):
        @property
        def input_variables(self) -> list[str]:
            return []

        def format_messages(self, **kwargs: object) -> list:
            return []

    t = MinimalTemplate()
    with pytest.raises(NotImplementedError):
        t.pretty_repr()


# --- ChatMessagePromptTemplate ---


def test_chat_message_prompt_template_format() -> None:
    """Test ChatMessagePromptTemplate formats with role."""
    template = ChatMessagePromptTemplate.from_template("Hello {name}", role="assistant")
    result = template.format(name="World")
    assert isinstance(result, ChatMessage)
    assert result.content == "Hello World"
    assert result.role == "assistant"


async def test_chat_message_prompt_template_aformat() -> None:
    """Test ChatMessagePromptTemplate async format."""
    template = ChatMessagePromptTemplate.from_template(
        "Hello {name}", role="custom_role"
    )
    result = await template.aformat(name="World")
    assert isinstance(result, ChatMessage)
    assert result.content == "Hello World"
    assert result.role == "custom_role"


def test_chat_message_prompt_template_with_additional_kwargs() -> None:
    """Test ChatMessagePromptTemplate with additional_kwargs."""
    prompt = PromptTemplate.from_template("Hello {name}")
    template = ChatMessagePromptTemplate(
        prompt=prompt,
        role="custom",
        additional_kwargs={"foo": "bar"},
    )
    result = template.format(name="World")
    assert result.additional_kwargs == {"foo": "bar"}


def test_chat_message_prompt_template_format_messages() -> None:
    """Test ChatMessagePromptTemplate format_messages returns list."""
    template = ChatMessagePromptTemplate.from_template("Hello {name}", role="assistant")
    messages = template.format_messages(name="World")
    assert len(messages) == 1
    assert messages[0].content == "Hello World"


async def test_chat_message_prompt_template_aformat_messages() -> None:
    """Test ChatMessagePromptTemplate aformat_messages."""
    template = ChatMessagePromptTemplate.from_template("Hello {name}", role="assistant")
    messages = await template.aformat_messages(name="World")
    assert len(messages) == 1
    assert messages[0].content == "Hello World"


# --- BaseStringMessagePromptTemplate ---


def test_base_string_message_from_template() -> None:
    """Test from_template creates correct prompt."""
    template = HumanMessagePromptTemplate.from_template("Hello {name}")
    assert template.prompt.template == "Hello {name}"
    assert template.input_variables == ["name"]


def test_base_string_message_from_template_with_partial() -> None:
    """Test from_template with partial_variables."""
    template = HumanMessagePromptTemplate.from_template(
        "Hello {name} {greeting}",
        partial_variables={"greeting": "Hi"},
    )
    assert template.input_variables == ["name"]
    msg = template.format(name="World")
    assert msg.content == "Hello World Hi"


def test_base_string_message_from_template_file(tmp_path: Path) -> None:
    """Test from_template_file loads from file."""
    file = tmp_path / "template.txt"
    file.write_text("Hello {name}", encoding="utf-8")
    template = HumanMessagePromptTemplate.from_template_file(
        str(file), input_variables=["name"]
    )
    assert template.input_variables == ["name"]
    msg = template.format(name="World")
    assert msg.content == "Hello World"


def test_base_string_message_pretty_repr() -> None:
    """Test pretty_repr output."""
    template = HumanMessagePromptTemplate.from_template("Hello {name}")
    result = template.pretty_repr()
    assert "Human" in result or "human" in result.lower()
    assert "{name}" in result


def test_base_string_message_pretty_repr_html() -> None:
    """Test pretty_repr with HTML formatting."""
    template = HumanMessagePromptTemplate.from_template("Hello {name}")
    result = template.pretty_repr(html=True)
    assert "{name}" in result


# --- HumanMessagePromptTemplate ---


def test_human_message_prompt_template_format() -> None:
    """Test HumanMessagePromptTemplate produces HumanMessage."""
    template = HumanMessagePromptTemplate.from_template("Hello {name}")
    msg = template.format(name="World")
    assert isinstance(msg, HumanMessage)
    assert msg.content == "Hello World"


async def test_human_message_prompt_template_aformat() -> None:
    """Test HumanMessagePromptTemplate async format."""
    template = HumanMessagePromptTemplate.from_template("Hello {name}")
    msg = await template.aformat(name="World")
    assert isinstance(msg, HumanMessage)
    assert msg.content == "Hello World"


# --- AIMessagePromptTemplate ---


def test_ai_message_prompt_template_format() -> None:
    """Test AIMessagePromptTemplate produces AIMessage."""
    template = AIMessagePromptTemplate.from_template("Response: {text}")
    msg = template.format(text="Hello")
    assert isinstance(msg, AIMessage)
    assert msg.content == "Response: Hello"


async def test_ai_message_prompt_template_aformat() -> None:
    """Test AIMessagePromptTemplate async format."""
    template = AIMessagePromptTemplate.from_template("Response: {text}")
    msg = await template.aformat(text="Hello")
    assert isinstance(msg, AIMessage)
    assert msg.content == "Response: Hello"


# --- SystemMessagePromptTemplate ---


def test_system_message_prompt_template_format() -> None:
    """Test SystemMessagePromptTemplate produces SystemMessage."""
    template = SystemMessagePromptTemplate.from_template("System: {instruction}")
    msg = template.format(instruction="Be helpful")
    assert isinstance(msg, SystemMessage)
    assert msg.content == "System: Be helpful"


async def test_system_message_prompt_template_aformat() -> None:
    """Test SystemMessagePromptTemplate async format."""
    template = SystemMessagePromptTemplate.from_template("System: {instruction}")
    msg = await template.aformat(instruction="Be helpful")
    assert isinstance(msg, SystemMessage)
    assert msg.content == "System: Be helpful"


# --- Multipart message templates ---


def test_human_message_prompt_template_multipart_text() -> None:
    """Test HumanMessagePromptTemplate with list of text templates."""
    template = HumanMessagePromptTemplate.from_template(
        [
            "Part 1: {a}",
            {"type": "text", "text": "Part 2: {b}"},
        ]
    )
    msg = template.format(a="hello", b="world")
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2
    assert msg.content[0]["text"] == "Part 1: hello"
    assert msg.content[1]["text"] == "Part 2: world"


def test_human_message_prompt_template_multipart_image() -> None:
    """Test HumanMessagePromptTemplate with image template."""
    template = HumanMessagePromptTemplate.from_template(
        [
            {"type": "text", "text": "Look at this:"},
            {
                "type": "image_url",
                "image_url": "https://example.com/{img}",
            },
        ]
    )
    msg = template.format(img="photo.jpg")
    assert isinstance(msg.content, list)
    assert msg.content[1]["type"] == "image_url"
    assert msg.content[1]["image_url"]["url"] == "https://example.com/photo.jpg"


def test_human_message_prompt_template_multipart_with_dict_image() -> None:
    """Test HumanMessagePromptTemplate with dict image_url template."""
    template = HumanMessagePromptTemplate.from_template(
        [
            {"type": "text", "text": "Look:"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/{img}", "detail": "low"},
            },
        ]
    )
    msg = template.format(img="photo.jpg")
    assert isinstance(msg.content, list)
    assert msg.content[1]["image_url"]["url"] == "https://example.com/photo.jpg"
    assert msg.content[1]["image_url"]["detail"] == "low"


def test_multipart_partial_variables_raises() -> None:
    """Test that partial_variables with list template raises."""
    with pytest.raises(
        ValueError, match="Partial variables are not supported for list"
    ):
        HumanMessagePromptTemplate.from_template(
            ["Hello {name}"],
            partial_variables={"name": "World"},
        )


def test_multipart_invalid_template_raises() -> None:
    """Test that invalid template type raises."""
    with pytest.raises(ValueError, match="Invalid template"):
        HumanMessagePromptTemplate.from_template(
            [42]  # type: ignore[list-item]
        )


def test_from_template_invalid_type_raises() -> None:
    """Test that non-str/non-list template raises."""
    with pytest.raises(ValueError, match="Invalid template"):
        HumanMessagePromptTemplate.from_template(42)  # type: ignore[arg-type]


# --- __add__ creates ChatPromptTemplate ---


def test_message_template_add_creates_chat() -> None:
    """Test that adding message templates creates ChatPromptTemplate."""
    t1 = SystemMessagePromptTemplate.from_template("System: {sys}")
    t2 = HumanMessagePromptTemplate.from_template("Human: {msg}")
    result = t1 + t2
    assert isinstance(result, ChatPromptTemplate)
    messages = result.format_messages(sys="Be helpful", msg="Hello")
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)


# --- Multipart async ---


async def test_multipart_aformat() -> None:
    """Test async format for multipart message template."""
    template = HumanMessagePromptTemplate.from_template(
        [
            "Part 1: {a}",
            "Part 2: {b}",
        ]
    )
    msg = await template.aformat(a="hello", b="world")
    assert isinstance(msg.content, list)
    assert msg.content[0]["text"] == "Part 1: hello"
    assert msg.content[1]["text"] == "Part 2: world"


# --- Multipart empty text filtered ---


def test_multipart_empty_text_filtered() -> None:
    """Test that empty text parts are filtered out in multipart messages."""
    template = HumanMessagePromptTemplate.from_template(
        [
            "{maybe_empty}",
            "Always present",
        ]
    )
    msg = template.format(maybe_empty="")
    # Empty text parts should be filtered
    assert isinstance(msg.content, list)
    text_parts = [p for p in msg.content if p.get("type") == "text"]
    assert all(p["text"] != "" for p in text_parts)
