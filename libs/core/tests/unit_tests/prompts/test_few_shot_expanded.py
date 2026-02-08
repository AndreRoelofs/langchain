"""Expanded tests for FewShotPromptTemplate and FewShotChatMessagePromptTemplate."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.prompts.few_shot import (
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    _FewShotPromptTemplateMixin,
)
from langchain_core.prompts.prompt import PromptTemplate

EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)


# --- _FewShotPromptTemplateMixin validation ---


def test_mixin_both_examples_and_selector_raises() -> None:
    """Test that providing both examples and example_selector raises."""
    selector = MagicMock(spec=BaseExampleSelector)
    with pytest.raises(
        ValueError, match="Only one of 'examples' and 'example_selector'"
    ):
        FewShotPromptTemplate(
            examples=[{"input": "a", "output": "b"}],
            example_selector=selector,
            example_prompt=EXAMPLE_PROMPT,
            suffix="Question: {input}",
        )


def test_mixin_neither_examples_nor_selector_raises() -> None:
    """Test that providing neither examples nor example_selector raises."""
    with pytest.raises(ValueError, match="One of 'examples' and 'example_selector'"):
        FewShotPromptTemplate(
            example_prompt=EXAMPLE_PROMPT,
            suffix="Question: {input}",
        )


# --- FewShotPromptTemplate ---


def test_few_shot_prompt_type() -> None:
    """Test _prompt_type returns 'few_shot'."""
    prompt = FewShotPromptTemplate(
        examples=[{"input": "a", "output": "b"}],
        example_prompt=EXAMPLE_PROMPT,
        suffix="Question: {q}",
    )
    assert prompt._prompt_type == "few_shot"


def test_few_shot_is_not_serializable() -> None:
    """Test is_lc_serializable returns False."""
    assert FewShotPromptTemplate.is_lc_serializable() is False


def test_few_shot_basic_format() -> None:
    """Test basic FewShotPromptTemplate formatting."""
    prompt = FewShotPromptTemplate(
        examples=[
            {"input": "happy", "output": "sad"},
            {"input": "tall", "output": "short"},
        ],
        example_prompt=EXAMPLE_PROMPT,
        suffix="Input: {adjective}\nOutput:",
        prefix="Write antonyms:",
    )
    result = prompt.format(adjective="big")
    assert "Write antonyms:" in result
    assert "Input: happy\nOutput: sad" in result
    assert "Input: tall\nOutput: short" in result
    assert "Input: big\nOutput:" in result


async def test_few_shot_aformat() -> None:
    """Test async FewShotPromptTemplate formatting."""
    prompt = FewShotPromptTemplate(
        examples=[{"input": "happy", "output": "sad"}],
        example_prompt=EXAMPLE_PROMPT,
        suffix="Input: {adjective}\nOutput:",
    )
    result = await prompt.aformat(adjective="big")
    assert "Input: happy\nOutput: sad" in result
    assert "Input: big\nOutput:" in result


def test_few_shot_with_custom_separator() -> None:
    """Test FewShotPromptTemplate with custom example_separator."""
    prompt = FewShotPromptTemplate(
        examples=[
            {"input": "a", "output": "b"},
            {"input": "c", "output": "d"},
        ],
        example_prompt=EXAMPLE_PROMPT,
        suffix="End",
        example_separator=" | ",
    )
    result = prompt.format()
    assert " | " in result


def test_few_shot_with_partial_variables() -> None:
    """Test FewShotPromptTemplate with partial variables."""
    prompt = FewShotPromptTemplate(
        examples=[{"input": "a", "output": "b"}],
        example_prompt=EXAMPLE_PROMPT,
        suffix="Input: {adjective}\nOutput:",
        prefix="Context: {ctx}",
        partial_variables={"ctx": "test"},
    )
    result = prompt.format(adjective="big")
    assert "Context: test" in result


def test_few_shot_with_example_selector() -> None:
    """Test FewShotPromptTemplate with example selector."""

    class SimpleSelector(BaseExampleSelector):
        def __init__(self, examples: list[dict]) -> None:
            self.examples = examples

        def add_example(self, example: dict) -> Any:
            self.examples.append(example)

        def select_examples(self, input_variables: dict) -> list[dict]:
            return self.examples[:1]

    selector = SimpleSelector(
        [{"input": "a", "output": "b"}, {"input": "c", "output": "d"}]
    )
    prompt = FewShotPromptTemplate(
        example_selector=selector,
        example_prompt=EXAMPLE_PROMPT,
        suffix="Input: {q}\nOutput:",
        input_variables=["q"],
    )
    result = prompt.format(q="test")
    assert "Input: a\nOutput: b" in result
    # Only first example should be selected
    assert "Input: c" not in result


async def test_few_shot_aformat_with_selector() -> None:
    """Test async FewShotPromptTemplate with example selector."""

    class AsyncSelector(BaseExampleSelector):
        def __init__(self, examples: list[dict]) -> None:
            self.examples = examples

        def add_example(self, example: dict) -> Any:
            self.examples.append(example)

        def select_examples(self, input_variables: dict) -> list[dict]:
            return self.examples

        async def aselect_examples(self, input_variables: dict) -> list[dict]:
            return self.examples[:1]

    selector = AsyncSelector(
        [
            {"input": "a", "output": "b"},
            {"input": "c", "output": "d"},
        ]
    )
    prompt = FewShotPromptTemplate(
        example_selector=selector,
        example_prompt=EXAMPLE_PROMPT,
        suffix="End",
        input_variables=[],
    )
    result = await prompt.aformat()
    assert "Input: a\nOutput: b" in result
    assert "Input: c" not in result


def test_few_shot_save_with_selector_raises() -> None:
    """Test save raises when example_selector is provided."""
    selector = MagicMock(spec=BaseExampleSelector)
    selector.select_examples = MagicMock(return_value=[])
    prompt = FewShotPromptTemplate(
        example_selector=selector,
        example_prompt=EXAMPLE_PROMPT,
        suffix="End",
        input_variables=[],
    )
    with pytest.raises(ValueError, match="Saving an example selector"):
        prompt.save("test.yaml")


def test_few_shot_validate_template() -> None:
    """Test validate_template=True validates prefix+suffix."""
    prompt = FewShotPromptTemplate(
        examples=[{"input": "a", "output": "b"}],
        example_prompt=EXAMPLE_PROMPT,
        suffix="Input: {adjective}\nOutput:",
        input_variables=["adjective"],
        validate_template=True,
    )
    assert prompt.input_variables == ["adjective"]


def test_few_shot_auto_infer_variables() -> None:
    """Test input_variables are auto-inferred from prefix+suffix."""
    prompt = FewShotPromptTemplate(
        examples=[{"input": "a", "output": "b"}],
        example_prompt=EXAMPLE_PROMPT,
        suffix="Input: {adjective}\nOutput:",
        prefix="Context: {ctx}",
    )
    assert sorted(prompt.input_variables) == ["adjective", "ctx"]


def test_few_shot_empty_prefix() -> None:
    """Test FewShotPromptTemplate with empty prefix."""
    prompt = FewShotPromptTemplate(
        examples=[{"input": "a", "output": "b"}],
        example_prompt=EXAMPLE_PROMPT,
        suffix="Question: {q}",
    )
    result = prompt.format(q="test")
    assert result.startswith("Input: a")


def test_few_shot_no_examples() -> None:
    """Test FewShotPromptTemplate with empty examples list."""
    prompt = FewShotPromptTemplate(
        examples=[],
        example_prompt=EXAMPLE_PROMPT,
        suffix="End",
    )
    result = prompt.format()
    assert result == "End"


# --- FewShotChatMessagePromptTemplate ---


def test_few_shot_chat_basic() -> None:
    """Test basic FewShotChatMessagePromptTemplate."""
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    prompt = FewShotChatMessagePromptTemplate(
        examples=[
            {"input": "2+2", "output": "4"},
            {"input": "2+3", "output": "5"},
        ],
        example_prompt=example_prompt,
    )
    messages = prompt.format_messages()
    assert len(messages) == 4
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "2+2"
    assert messages[1].content == "4"


async def test_few_shot_chat_aformat_messages() -> None:
    """Test async FewShotChatMessagePromptTemplate format_messages."""
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    prompt = FewShotChatMessagePromptTemplate(
        examples=[{"input": "2+2", "output": "4"}],
        example_prompt=example_prompt,
    )
    messages = await prompt.aformat_messages()
    assert len(messages) == 2


def test_few_shot_chat_format_string() -> None:
    """Test FewShotChatMessagePromptTemplate.format returns string."""
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    prompt = FewShotChatMessagePromptTemplate(
        examples=[{"input": "2+2", "output": "4"}],
        example_prompt=example_prompt,
    )
    result = prompt.format()
    assert isinstance(result, str)
    assert "2+2" in result
    assert "4" in result


async def test_few_shot_chat_aformat_string() -> None:
    """Test async FewShotChatMessagePromptTemplate.aformat returns string."""
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    prompt = FewShotChatMessagePromptTemplate(
        examples=[{"input": "2+2", "output": "4"}],
        example_prompt=example_prompt,
    )
    result = await prompt.aformat()
    assert isinstance(result, str)
    assert "2+2" in result


def test_few_shot_chat_is_not_serializable() -> None:
    """Test FewShotChatMessagePromptTemplate is not serializable."""
    assert FewShotChatMessagePromptTemplate.is_lc_serializable() is False


def test_few_shot_chat_pretty_repr_raises() -> None:
    """Test pretty_repr raises NotImplementedError."""
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    prompt = FewShotChatMessagePromptTemplate(
        examples=[{"input": "2+2", "output": "4"}],
        example_prompt=example_prompt,
    )
    with pytest.raises(NotImplementedError):
        prompt.pretty_repr()


def test_few_shot_chat_with_selector() -> None:
    """Test FewShotChatMessagePromptTemplate with example selector."""

    class SimpleSelector(BaseExampleSelector):
        def __init__(self, examples: list[dict]) -> None:
            self.examples = examples

        def add_example(self, example: dict) -> Any:
            self.examples.append(example)

        def select_examples(self, input_variables: dict) -> list[dict]:
            return self.examples[:1]

    selector = SimpleSelector(
        [
            {"input": "2+2", "output": "4"},
            {"input": "2+3", "output": "5"},
        ]
    )
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    prompt = FewShotChatMessagePromptTemplate(
        example_selector=selector,
        example_prompt=example_prompt,
    )
    messages = prompt.format_messages()
    assert len(messages) == 2  # Only first example selected


async def test_few_shot_chat_aformat_messages_with_selector() -> None:
    """Test async FewShotChatMessagePromptTemplate with selector."""

    class AsyncSelector(BaseExampleSelector):
        def __init__(self, examples: list[dict]) -> None:
            self.examples = examples

        def add_example(self, example: dict) -> Any:
            self.examples.append(example)

        def select_examples(self, input_variables: dict) -> list[dict]:
            return self.examples

        async def aselect_examples(self, input_variables: dict) -> list[dict]:
            return self.examples[:1]

    selector = AsyncSelector(
        [
            {"input": "2+2", "output": "4"},
            {"input": "2+3", "output": "5"},
        ]
    )
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    prompt = FewShotChatMessagePromptTemplate(
        example_selector=selector,
        example_prompt=example_prompt,
    )
    messages = await prompt.aformat_messages()
    assert len(messages) == 2  # Only first example from async selector
