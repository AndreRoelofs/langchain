"""Expanded tests for ChatPromptTemplate functionality."""

import pytest

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    BaseChatPromptTemplate,
    ChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    _convert_to_message_template,
    _create_template_from_message_type,
)

# --- ChatPromptTemplate init ---


def test_chat_prompt_template_from_string_shorthand() -> None:
    """Test that a string is interpreted as a human message template."""
    template = ChatPromptTemplate.from_messages(["{input}"])
    messages = template.format_messages(input="Hello")
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Hello"


def test_chat_prompt_template_from_base_message() -> None:
    """Test creating from BaseMessage instances."""
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="Be helpful"),
            HumanMessage(content="Hello"),
        ]
    )
    messages = template.format_messages()
    assert len(messages) == 2
    assert messages[0].content == "Be helpful"
    assert messages[1].content == "Hello"


def test_chat_prompt_template_from_dict_message() -> None:
    """Test creating from dict with role and content keys."""
    template = ChatPromptTemplate.from_messages(
        [{"role": "human", "content": "Hello {name}"}]
    )
    messages = template.format_messages(name="World")
    assert messages[0].content == "Hello World"


def test_chat_prompt_template_from_dict_message_invalid_keys() -> None:
    """Test that dict message with invalid keys raises."""
    with pytest.raises(ValueError, match="Expected dict to have exact keys"):
        ChatPromptTemplate.from_messages(
            [{"role": "human", "content": "Hello", "extra": "bad"}]
        )


# --- from_template ---


def test_chat_prompt_template_from_template() -> None:
    """Test ChatPromptTemplate.from_template creates human message."""
    template = ChatPromptTemplate.from_template("Hello {name}")
    messages = template.format_messages(name="World")
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Hello World"


# --- format_messages / aformat_messages ---


def test_format_messages_with_mixed_types() -> None:
    """Test format_messages handles mixed message and template types."""
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="System"),
            ("human", "Hello {name}"),
            AIMessage(content="Hi!"),
        ]
    )
    messages = template.format_messages(name="World")
    assert len(messages) == 3
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)
    assert messages[1].content == "Hello World"


async def test_aformat_messages() -> None:
    """Test async format_messages."""
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are {role}"),
            ("human", "{input}"),
        ]
    )
    messages = await template.aformat_messages(role="helpful", input="Hi")
    assert len(messages) == 2
    assert messages[0].content == "You are helpful"
    assert messages[1].content == "Hi"


# --- format / aformat (string output) ---


def test_format_returns_string() -> None:
    """Test that format returns a string representation."""
    template = ChatPromptTemplate.from_messages([("human", "Hello {name}")])
    result = template.format(name="World")
    assert isinstance(result, str)
    assert "Hello World" in result


async def test_aformat_returns_string() -> None:
    """Test that async format returns a string representation."""
    template = ChatPromptTemplate.from_messages([("human", "Hello {name}")])
    result = await template.aformat(name="World")
    assert isinstance(result, str)
    assert "Hello World" in result


# --- format_prompt / aformat_prompt ---


def test_format_prompt_returns_chat_prompt_value() -> None:
    """Test that format_prompt returns ChatPromptValue."""
    from langchain_core.prompt_values import ChatPromptValue

    template = ChatPromptTemplate.from_messages([("human", "Hello {name}")])
    result = template.format_prompt(name="World")
    assert isinstance(result, ChatPromptValue)
    assert len(result.messages) == 1


async def test_aformat_prompt_returns_chat_prompt_value() -> None:
    """Test that aformat_prompt returns ChatPromptValue."""
    from langchain_core.prompt_values import ChatPromptValue

    template = ChatPromptTemplate.from_messages([("human", "Hello {name}")])
    result = await template.aformat_prompt(name="World")
    assert isinstance(result, ChatPromptValue)


# --- __add__ ---


def test_add_chat_prompt_templates() -> None:
    """Test adding two ChatPromptTemplates."""
    t1 = ChatPromptTemplate.from_messages([("system", "System")])
    t2 = ChatPromptTemplate.from_messages([("human", "Hello {name}")])
    combined = t1 + t2
    assert isinstance(combined, ChatPromptTemplate)
    messages = combined.format_messages(name="World")
    assert len(messages) == 2


def test_add_with_message_prompt_template() -> None:
    """Test adding ChatPromptTemplate with MessagePromptTemplate."""
    t1 = ChatPromptTemplate.from_messages([("system", "System")])
    t2 = HumanMessagePromptTemplate.from_template("Hello {name}")
    combined = t1 + t2
    assert isinstance(combined, ChatPromptTemplate)
    assert len(combined.messages) == 2


def test_add_with_base_message() -> None:
    """Test adding ChatPromptTemplate with a BaseMessage."""
    t1 = ChatPromptTemplate.from_messages([("system", "System")])
    combined = t1 + HumanMessage(content="Hello")
    assert isinstance(combined, ChatPromptTemplate)
    assert len(combined.messages) == 2


def test_add_with_string() -> None:
    """Test adding ChatPromptTemplate with a string."""
    t1 = ChatPromptTemplate.from_messages([("system", "System")])
    combined = t1 + "Hello {name}"
    assert isinstance(combined, ChatPromptTemplate)
    messages = combined.format_messages(name="World")
    assert len(messages) == 2
    assert messages[1].content == "Hello World"


def test_add_with_list_of_tuples() -> None:
    """Test adding ChatPromptTemplate with a list of tuples."""
    t1 = ChatPromptTemplate.from_messages([("system", "System")])
    combined = t1 + [("human", "Hello"), ("ai", "Hi")]
    assert isinstance(combined, ChatPromptTemplate)
    assert len(combined.messages) == 3


def test_add_with_unsupported_type_raises() -> None:
    """Test adding with unsupported type raises NotImplementedError."""
    t1 = ChatPromptTemplate.from_messages([("system", "System")])
    with pytest.raises(NotImplementedError, match="Unsupported operand type"):
        t1 + 42  # type: ignore[operator]


def test_add_merges_partial_variables() -> None:
    """Test that adding merges partial variables from both templates."""
    t1 = ChatPromptTemplate.from_messages([("placeholder", "{history}")])
    t2 = ChatPromptTemplate.from_messages([("placeholder", "{other}")])
    combined = t1 + t2
    # Both placeholders are optional, so partials should be merged
    assert "history" in combined.partial_variables
    assert "other" in combined.partial_variables


# --- __getitem__ / __len__ ---


def test_getitem_int() -> None:
    """Test indexing with int returns a message."""
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "System"),
            ("human", "Hello"),
            ("ai", "Hi"),
        ]
    )
    msg = template[1]
    assert isinstance(msg, HumanMessagePromptTemplate)


def test_getitem_slice() -> None:
    """Test slicing returns a new ChatPromptTemplate."""
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "System"),
            ("human", "Hello"),
            ("ai", "Hi"),
        ]
    )
    sliced = template[1:]
    assert isinstance(sliced, ChatPromptTemplate)
    assert len(sliced) == 2


def test_len() -> None:
    """Test __len__ returns number of messages."""
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "System"),
            ("human", "Hello"),
            ("ai", "Hi"),
        ]
    )
    assert len(template) == 3


# --- append / extend ---


def test_append_message() -> None:
    """Test appending a message to chat template."""
    template = ChatPromptTemplate.from_messages([("system", "System")])
    template.append(("human", "Hello {name}"))
    assert len(template) == 2
    messages = template.format_messages(name="World")
    assert messages[1].content == "Hello World"


def test_extend_messages() -> None:
    """Test extending chat template with multiple messages."""
    template = ChatPromptTemplate.from_messages([("system", "System")])
    template.extend([("human", "Hello"), ("ai", "Hi")])
    assert len(template) == 3


# --- partial ---


def test_partial_chat_prompt() -> None:
    """Test partial fills in variables."""
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are {role}"),
            ("human", "{input}"),
        ]
    )
    partial = template.partial(role="helpful")
    assert "role" not in partial.input_variables
    messages = partial.format_messages(input="Hello")
    assert messages[0].content == "You are helpful"


# --- MessagesPlaceholder ---


def test_messages_placeholder_basic() -> None:
    """Test basic MessagesPlaceholder functionality."""
    placeholder = MessagesPlaceholder("history")
    assert placeholder.input_variables == ["history"]
    messages = placeholder.format_messages(history=[HumanMessage(content="Hello")])
    assert len(messages) == 1
    assert messages[0].content == "Hello"


def test_messages_placeholder_optional() -> None:
    """Test optional MessagesPlaceholder returns empty when no input."""
    placeholder = MessagesPlaceholder("history", optional=True)
    assert placeholder.input_variables == []
    messages = placeholder.format_messages()
    assert messages == []


def test_messages_placeholder_required_raises() -> None:
    """Test required MessagesPlaceholder raises when missing."""
    placeholder = MessagesPlaceholder("history")
    with pytest.raises(KeyError):
        placeholder.format_messages()


def test_messages_placeholder_non_list_raises() -> None:
    """Test MessagesPlaceholder raises when value is not a list."""
    placeholder = MessagesPlaceholder("history")
    with pytest.raises(ValueError, match="should be a list"):
        placeholder.format_messages(history="not a list")


def test_messages_placeholder_n_messages() -> None:
    """Test MessagesPlaceholder with n_messages limit."""
    placeholder = MessagesPlaceholder("history", n_messages=2)
    messages = placeholder.format_messages(
        history=[
            ("human", "First"),
            ("ai", "Second"),
            ("human", "Third"),
        ]
    )
    assert len(messages) == 2
    assert messages[0].content == "Second"
    assert messages[1].content == "Third"


def test_messages_placeholder_converts_tuples() -> None:
    """Test MessagesPlaceholder converts tuple messages."""
    placeholder = MessagesPlaceholder("history")
    messages = placeholder.format_messages(
        history=[
            ("system", "System msg"),
            ("human", "Human msg"),
            ("ai", "AI msg"),
        ]
    )
    assert len(messages) == 3
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)


def test_messages_placeholder_pretty_repr() -> None:
    """Test MessagesPlaceholder pretty_repr."""
    placeholder = MessagesPlaceholder("history")
    result = placeholder.pretty_repr()
    assert "Messages Placeholder" in result
    assert "{history}" in result


def test_messages_placeholder_pretty_repr_html() -> None:
    """Test MessagesPlaceholder pretty_repr with HTML."""
    placeholder = MessagesPlaceholder("history")
    result = placeholder.pretty_repr(html=True)
    assert "history" in result


# --- _create_template_from_message_type ---


def test_create_template_human() -> None:
    """Test creating human message template."""
    msg = _create_template_from_message_type("human", "Hello {name}")
    assert isinstance(msg, HumanMessagePromptTemplate)


def test_create_template_user_alias() -> None:
    """Test 'user' alias works for human."""
    msg = _create_template_from_message_type("user", "Hello {name}")
    assert isinstance(msg, HumanMessagePromptTemplate)


def test_create_template_ai() -> None:
    """Test creating AI message template."""
    msg = _create_template_from_message_type("ai", "Response: {text}")
    assert isinstance(msg, AIMessagePromptTemplate)


def test_create_template_assistant_alias() -> None:
    """Test 'assistant' alias works for AI."""
    msg = _create_template_from_message_type("assistant", "Response: {text}")
    assert isinstance(msg, AIMessagePromptTemplate)


def test_create_template_system() -> None:
    """Test creating system message template."""
    msg = _create_template_from_message_type("system", "Be {role}")
    assert isinstance(msg, SystemMessagePromptTemplate)


def test_create_template_placeholder_string() -> None:
    """Test creating placeholder from string."""
    msg = _create_template_from_message_type("placeholder", "{history}")
    assert isinstance(msg, MessagesPlaceholder)
    assert msg.variable_name == "history"
    assert msg.optional is True


def test_create_template_placeholder_with_options() -> None:
    """Test creating placeholder from list with optional flag."""
    msg = _create_template_from_message_type("placeholder", ["{history}", False])
    assert isinstance(msg, MessagesPlaceholder)
    assert msg.variable_name == "history"
    assert msg.optional is False


def test_create_template_placeholder_invalid_string() -> None:
    """Test placeholder with invalid string raises."""
    with pytest.raises(ValueError, match="Expected a variable name"):
        _create_template_from_message_type("placeholder", "no_braces")


def test_create_template_placeholder_invalid_optional_type() -> None:
    """Test placeholder with non-bool optional raises."""
    with pytest.raises(ValueError, match="Expected is_optional to be a boolean"):
        _create_template_from_message_type("placeholder", ["{history}", "not_bool"])


def test_create_template_placeholder_invalid_var_name_type() -> None:
    """Test placeholder with non-string variable name raises."""
    with pytest.raises(ValueError, match="Expected variable name to be a string"):
        _create_template_from_message_type(
            "placeholder",
            [123, True],  # type: ignore[list-item]
        )


def test_create_template_placeholder_list_invalid_braces() -> None:
    """Test placeholder list with invalid braces raises."""
    with pytest.raises(ValueError, match="Expected a variable name"):
        _create_template_from_message_type("placeholder", ["no_braces", True])


def test_create_template_unknown_type_raises() -> None:
    """Test unknown message type raises."""
    with pytest.raises(ValueError, match="Unexpected message type"):
        _create_template_from_message_type("unknown", "Hello")


# --- _convert_to_message_template ---


def test_convert_to_message_template_from_string() -> None:
    """Test converting string to human message template."""
    result = _convert_to_message_template("Hello {name}")
    assert isinstance(result, HumanMessagePromptTemplate)


def test_convert_to_message_template_from_base_message() -> None:
    """Test converting BaseMessage passes through."""
    msg = HumanMessage(content="Hello")
    result = _convert_to_message_template(msg)
    assert result is msg


def test_convert_to_message_template_from_template() -> None:
    """Test converting message template passes through."""
    tmpl = HumanMessagePromptTemplate.from_template("Hello")
    result = _convert_to_message_template(tmpl)
    assert result is tmpl


def test_convert_to_message_template_from_tuple() -> None:
    """Test converting 2-tuple creates message template."""
    result = _convert_to_message_template(("human", "Hello {name}"))
    assert isinstance(result, HumanMessagePromptTemplate)


def test_convert_to_message_template_invalid_tuple_raises() -> None:
    """Test that 3-tuple raises ValueError."""
    with pytest.raises(ValueError, match="Expected 2-tuple"):
        _convert_to_message_template(("human", "Hello", "extra"))  # type: ignore[arg-type]


def test_convert_to_message_template_unsupported_type_raises() -> None:
    """Test that unsupported type raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Unsupported message type"):
        _convert_to_message_template(42)  # type: ignore[arg-type]


# --- ChatPromptTemplate pretty_repr ---


def test_pretty_repr() -> None:
    """Test ChatPromptTemplate pretty_repr."""
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are {role}"),
            ("human", "{input}"),
        ]
    )
    result = template.pretty_repr()
    assert "System" in result or "system" in result.lower()
    assert "Human" in result or "human" in result.lower()


# --- ChatPromptTemplate save raises ---


def test_save_raises_not_implemented() -> None:
    """Test that save raises NotImplementedError."""
    template = ChatPromptTemplate.from_messages([("human", "Hello")])
    with pytest.raises(NotImplementedError):
        template.save("test.json")


# --- ChatPromptTemplate _prompt_type ---


def test_prompt_type() -> None:
    """Test _prompt_type returns 'chat'."""
    template = ChatPromptTemplate.from_messages([("human", "Hello")])
    assert template._prompt_type == "chat"


# --- lc_namespace ---


def test_lc_namespace() -> None:
    """Test get_lc_namespace returns correct value."""
    assert ChatPromptTemplate.get_lc_namespace() == [
        "langchain",
        "prompts",
        "chat",
    ]


# --- validate_template=True ---


def test_validate_template_true_mismatch_raises() -> None:
    """Test validate_template=True raises on variable mismatch."""
    with pytest.raises(ValueError, match="mismatched input_variables"):
        ChatPromptTemplate(
            messages=[("human", "Hello {name}")],
            input_variables=["wrong"],
            validate_template=True,
        )


def test_validate_template_true_correct() -> None:
    """Test validate_template=True passes with correct variables."""
    template = ChatPromptTemplate(
        messages=[("human", "Hello {name}")],
        input_variables=["name"],
        validate_template=True,
    )
    assert template.input_variables == ["name"]


# --- BaseChatPromptTemplate ---


def test_base_chat_prompt_template_pretty_repr_raises() -> None:
    """Test BaseChatPromptTemplate.pretty_repr raises NotImplementedError."""

    class MinimalChat(BaseChatPromptTemplate):
        def format_messages(self, **kwargs: object) -> list[BaseMessage]:
            return []

    t = MinimalChat(input_variables=[])
    with pytest.raises(NotImplementedError):
        t.pretty_repr()


# --- input_types for MessagesPlaceholder ---


def test_messages_placeholder_sets_input_types() -> None:
    """Test that MessagesPlaceholder sets input_types for its variable."""
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "System"),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    assert "history" in template.input_types


# --- jinja2 dict template raises ---


@pytest.mark.requires("jinja2")
def test_dict_message_prompt_template_jinja2_raises() -> None:
    """Test that jinja2 format with dict template raises."""
    with pytest.raises(ValueError, match="jinja2 is unsafe"):
        ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    [{"custom_key": "value"}],
                ),
            ],
            template_format="jinja2",
        )
