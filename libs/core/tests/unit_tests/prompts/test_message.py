"""Test message prompt template functionality."""

import pytest

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts.message import BaseMessagePromptTemplate


def test_base_message_prompt_template_is_serializable() -> None:
    """Test that BaseMessagePromptTemplate is serializable."""
    assert BaseMessagePromptTemplate.is_lc_serializable()


def test_base_message_prompt_template_lc_namespace() -> None:
    """Test that BaseMessagePromptTemplate has correct namespace."""
    assert BaseMessagePromptTemplate.get_lc_namespace() == [
        "langchain",
        "prompts",
        "chat",
    ]


async def test_base_message_prompt_template_aformat_messages_default() -> None:
    """Test default async format_messages calls sync version."""
    template = HumanMessagePromptTemplate.from_template("Hello {name}")

    sync_messages = template.format_messages(name="World")
    async_messages = await template.aformat_messages(name="World")

    assert sync_messages == async_messages
    assert len(async_messages) == 1
    assert async_messages[0].content == "Hello World"


def test_base_message_prompt_template_add_with_message() -> None:
    """Test adding a message template with another message."""
    template1 = HumanMessagePromptTemplate.from_template("Hello {name}")
    result = template1 + HumanMessage(content="How are you?")

    assert isinstance(result, ChatPromptTemplate)
    assert len(result.messages) == 2


def test_base_message_prompt_template_add_with_template() -> None:
    """Test adding a message template with another template."""
    template1 = HumanMessagePromptTemplate.from_template("Hello {name}")
    template2 = HumanMessagePromptTemplate.from_template("Goodbye {name}")

    result = template1 + template2

    assert isinstance(result, ChatPromptTemplate)
    assert len(result.messages) == 2


def test_base_message_prompt_template_add_with_string() -> None:
    """Test adding a message template with a string."""
    template1 = HumanMessagePromptTemplate.from_template("Hello {name}")
    result = template1 + "Goodbye {name}"

    assert isinstance(result, ChatPromptTemplate)
    assert len(result.messages) == 2

    messages = result.format_messages(name="World")
    assert messages[0].content == "Hello World"
    assert messages[1].content == "Goodbye World"


def test_base_message_prompt_template_add_with_tuple() -> None:
    """Test adding a message template with a tuple."""
    template1 = HumanMessagePromptTemplate.from_template("Hello {name}")
    result = template1 + ("human", "How are you?")

    assert isinstance(result, ChatPromptTemplate)
    # When adding tuple, it creates a ChatPromptTemplate with the tuple converted
    # The tuple gets converted to a message template
    assert len(result.messages) >= 2


def test_base_message_prompt_template_add_with_list() -> None:
    """Test adding a message template with a list of messages."""
    template1 = HumanMessagePromptTemplate.from_template("Hello {name}")
    result = template1 + [("human", "How are you?"), ("ai", "I'm good!")]

    assert isinstance(result, ChatPromptTemplate)
    assert len(result.messages) == 3


def test_base_message_prompt_template_pretty_print(capsys: pytest.CaptureFixture) -> None:
    """Test pretty_print method outputs something."""
    template = HumanMessagePromptTemplate.from_template("Hello {name}")

    # Just ensure it doesn't crash - output depends on environment
    template.pretty_print()
    captured = capsys.readouterr()
    assert captured.out  # Should have printed something


def test_base_message_prompt_template_abstract_methods() -> None:
    """Test that abstract methods raise errors if not implemented."""

    # Create a minimal implementation that doesn't implement abstract methods
    class IncompleteTemplate(BaseMessagePromptTemplate):
        pass

    # Should not be able to instantiate without implementing abstract methods
    with pytest.raises(TypeError):
        IncompleteTemplate()  # type: ignore[abstract]


def test_base_message_prompt_template_input_variables_property() -> None:
    """Test that input_variables property works correctly."""
    template = HumanMessagePromptTemplate.from_template("Hello {name} and {friend}")

    assert set(template.input_variables) == {"name", "friend"}


def test_base_message_prompt_template_format_messages_result() -> None:
    """Test that format_messages returns list of BaseMessage."""
    template = HumanMessagePromptTemplate.from_template("Hello {name}")
    messages = template.format_messages(name="World")

    assert isinstance(messages, list)
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
