"""Expanded tests for FewShotPromptWithTemplates."""

from typing import Any

import pytest

from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.prompts.few_shot_with_templates import FewShotPromptWithTemplates
from langchain_core.prompts.prompt import PromptTemplate

EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["question", "answer"], template="{question}: {answer}"
)


# --- Validation ---


def test_both_examples_and_selector_raises() -> None:
    """Test that providing both examples and example_selector raises."""

    class DummySelector(BaseExampleSelector):
        def add_example(self, example: dict) -> Any:
            pass

        def select_examples(self, input_variables: dict) -> list[dict]:
            return []

    selector = DummySelector()
    suffix = PromptTemplate(input_variables=["q"], template="Q: {q}")
    with pytest.raises(
        ValueError, match="Only one of 'examples' and 'example_selector'"
    ):
        FewShotPromptWithTemplates(
            examples=[{"question": "a", "answer": "b"}],
            example_selector=selector,
            example_prompt=EXAMPLE_PROMPT,
            suffix=suffix,
            input_variables=[],
        )


def test_neither_examples_nor_selector_raises() -> None:
    """Test that providing neither raises."""
    suffix = PromptTemplate(input_variables=["q"], template="Q: {q}")
    with pytest.raises(ValueError, match="One of 'examples' and 'example_selector'"):
        FewShotPromptWithTemplates(
            example_prompt=EXAMPLE_PROMPT,
            suffix=suffix,
            input_variables=[],
        )


# --- Properties ---


def test_prompt_type() -> None:
    """Test _prompt_type returns 'few_shot_with_templates'."""
    suffix = PromptTemplate(input_variables=[], template="End")
    prompt = FewShotPromptWithTemplates(
        examples=[{"question": "a", "answer": "b"}],
        example_prompt=EXAMPLE_PROMPT,
        suffix=suffix,
        input_variables=[],
    )
    assert prompt._prompt_type == "few_shot_with_templates"


def test_lc_namespace() -> None:
    """Test get_lc_namespace."""
    assert FewShotPromptWithTemplates.get_lc_namespace() == [
        "langchain",
        "prompts",
        "few_shot_with_templates",
    ]


# --- format ---


def test_format_with_prefix_and_suffix() -> None:
    """Test formatting with both prefix and suffix PromptTemplates."""
    prefix = PromptTemplate(input_variables=["topic"], template="Topic: {topic}")
    suffix = PromptTemplate(input_variables=["question"], template="Q: {question}")
    prompt = FewShotPromptWithTemplates(
        examples=[
            {"question": "foo", "answer": "bar"},
        ],
        example_prompt=EXAMPLE_PROMPT,
        prefix=prefix,
        suffix=suffix,
        input_variables=["topic", "question"],
    )
    result = prompt.format(topic="test", question="what?")
    assert "Topic: test" in result
    assert "foo: bar" in result
    assert "Q: what?" in result


def test_format_without_prefix() -> None:
    """Test formatting without prefix."""
    suffix = PromptTemplate(input_variables=["question"], template="Q: {question}")
    prompt = FewShotPromptWithTemplates(
        examples=[{"question": "foo", "answer": "bar"}],
        example_prompt=EXAMPLE_PROMPT,
        suffix=suffix,
        input_variables=["question"],
    )
    result = prompt.format(question="what?")
    assert "foo: bar" in result
    assert "Q: what?" in result


def test_format_with_custom_separator() -> None:
    """Test formatting with custom example_separator."""
    suffix = PromptTemplate(input_variables=[], template="End")
    prompt = FewShotPromptWithTemplates(
        examples=[
            {"question": "a", "answer": "1"},
            {"question": "b", "answer": "2"},
        ],
        example_prompt=EXAMPLE_PROMPT,
        suffix=suffix,
        example_separator=" | ",
        input_variables=[],
    )
    result = prompt.format()
    assert " | " in result


# --- aformat ---


async def test_aformat_basic() -> None:
    """Test async formatting."""
    prefix = PromptTemplate(input_variables=["topic"], template="Topic: {topic}")
    suffix = PromptTemplate(input_variables=["question"], template="Q: {question}")
    prompt = FewShotPromptWithTemplates(
        examples=[{"question": "foo", "answer": "bar"}],
        example_prompt=EXAMPLE_PROMPT,
        prefix=prefix,
        suffix=suffix,
        input_variables=["topic", "question"],
    )
    result = await prompt.aformat(topic="test", question="what?")
    assert "Topic: test" in result
    assert "foo: bar" in result
    assert "Q: what?" in result


async def test_aformat_without_prefix() -> None:
    """Test async formatting without prefix."""
    suffix = PromptTemplate(input_variables=["q"], template="Q: {q}")
    prompt = FewShotPromptWithTemplates(
        examples=[{"question": "foo", "answer": "bar"}],
        example_prompt=EXAMPLE_PROMPT,
        suffix=suffix,
        input_variables=["q"],
    )
    result = await prompt.aformat(q="what?")
    assert "foo: bar" in result


# --- with example_selector ---


def test_format_with_selector() -> None:
    """Test formatting with example selector."""

    class SimpleSelector(BaseExampleSelector):
        def __init__(self, examples: list[dict]) -> None:
            self.examples = examples

        def add_example(self, example: dict) -> Any:
            self.examples.append(example)

        def select_examples(self, input_variables: dict) -> list[dict]:
            return self.examples[:1]

    selector = SimpleSelector(
        [
            {"question": "a", "answer": "1"},
            {"question": "b", "answer": "2"},
        ]
    )
    suffix = PromptTemplate(input_variables=[], template="End")
    prompt = FewShotPromptWithTemplates(
        example_selector=selector,
        example_prompt=EXAMPLE_PROMPT,
        suffix=suffix,
        input_variables=[],
    )
    result = prompt.format()
    assert "a: 1" in result
    assert "b: 2" not in result


async def test_aformat_with_selector() -> None:
    """Test async formatting with example selector."""

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
            {"question": "a", "answer": "1"},
            {"question": "b", "answer": "2"},
        ]
    )
    suffix = PromptTemplate(input_variables=[], template="End")
    prompt = FewShotPromptWithTemplates(
        example_selector=selector,
        example_prompt=EXAMPLE_PROMPT,
        suffix=suffix,
        input_variables=[],
    )
    result = await prompt.aformat()
    assert "a: 1" in result
    assert "b: 2" not in result


# --- save ---


def test_save_with_selector_raises() -> None:
    """Test save raises when example_selector is provided."""

    class DummySelector(BaseExampleSelector):
        def add_example(self, example: dict) -> Any:
            pass

        def select_examples(self, input_variables: dict) -> list[dict]:
            return []

    suffix = PromptTemplate(input_variables=[], template="End")
    prompt = FewShotPromptWithTemplates(
        example_selector=DummySelector(),
        example_prompt=EXAMPLE_PROMPT,
        suffix=suffix,
        input_variables=[],
    )
    with pytest.raises(ValueError, match="Saving an example selector"):
        prompt.save("test.yaml")


# --- auto-inferred input_variables ---


def test_auto_infer_input_variables() -> None:
    """Test that input variables are auto-inferred when validate_template=False."""
    prefix = PromptTemplate(input_variables=["topic"], template="Topic: {topic}")
    suffix = PromptTemplate(input_variables=["question"], template="Q: {question}")
    prompt = FewShotPromptWithTemplates(
        examples=[{"question": "a", "answer": "b"}],
        example_prompt=EXAMPLE_PROMPT,
        prefix=prefix,
        suffix=suffix,
        input_variables=[],
    )
    assert sorted(prompt.input_variables) == ["question", "topic"]


# --- partial variables ---


def test_format_with_partial_variables() -> None:
    """Test formatting with partial variables."""
    prefix = PromptTemplate(input_variables=["topic"], template="Topic: {topic}")
    suffix = PromptTemplate(input_variables=["question"], template="Q: {question}")
    prompt = FewShotPromptWithTemplates(
        examples=[{"question": "a", "answer": "b"}],
        example_prompt=EXAMPLE_PROMPT,
        prefix=prefix,
        suffix=suffix,
        input_variables=["question"],
        partial_variables={"topic": "math"},
    )
    result = prompt.format(question="what?")
    assert "Topic: math" in result
    assert "Q: what?" in result
