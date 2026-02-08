"""Expanded tests for base prompt template functionality."""

import json
from pathlib import Path

import pytest
import yaml

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import (
    BasePromptTemplate,
    _get_document_info,
    aformat_document,
    format_document,
)

# --- format_document / aformat_document ---


def test_format_document_empty_page_content() -> None:
    """Test formatting a document with empty page_content."""
    doc = Document(page_content="", metadata={"page": "1"})
    prompt = PromptTemplate.from_template("Page {page}: {page_content}")
    result = format_document(doc, prompt)
    assert result == "Page 1: "


def test_format_document_only_metadata_vars() -> None:
    """Test formatting a document that uses only metadata vars (no page_content)."""
    doc = Document(page_content="ignored", metadata={"page": "1", "chapter": "2"})
    prompt = PromptTemplate.from_template("Page {page}, Chapter {chapter}")
    result = format_document(doc, prompt)
    assert result == "Page 1, Chapter 2"


def test_format_document_metadata_overrides_page_content_key() -> None:
    """Test that metadata with page_content key uses the actual page_content."""
    doc = Document(
        page_content="real content",
        metadata={"page_content": "metadata content"},
    )
    prompt = PromptTemplate.from_template("{page_content}")
    # page_content from doc.page_content should be used since base_info
    # starts with page_content=doc.page_content then merges metadata
    result = format_document(doc, prompt)
    # The metadata page_content overrides the document page_content in base_info
    assert result == "metadata content"


async def test_aformat_document_multiple_metadata() -> None:
    """Test async document formatting with multiple metadata fields."""
    doc = Document(
        page_content="Test content",
        metadata={"page": "5", "chapter": "Intro"},
    )
    prompt = PromptTemplate.from_template("Ch {chapter}, Page {page}: {page_content}")
    result = await aformat_document(doc, prompt)
    assert result == "Ch Intro, Page 5: Test content"


def test_get_document_info_missing_metadata_error_message() -> None:
    """Test that _get_document_info error message lists required metadata."""
    doc = Document(page_content="content", metadata={})
    prompt = PromptTemplate.from_template("{page_content} {author} {date}")
    with pytest.raises(ValueError, match="author"):
        _get_document_info(doc, prompt)


# --- BasePromptTemplate validation ---


def test_base_prompt_template_overlapping_input_and_partial_auto_resolves() -> None:
    """Test that PromptTemplate auto-resolves overlapping variables.

    PromptTemplate's pre_init_validation auto-infers input_variables from
    the template, excluding partial_variables. So explicitly providing
    overlapping input_variables and partial_variables does NOT raise because
    the auto-inference step removes the overlap before the validator runs.
    """
    template = PromptTemplate(
        template="Hello {name}",
        input_variables=["name"],
        partial_variables={"name": "World"},
    )
    # Auto-inference removes 'name' from input_variables since it's partial
    assert template.input_variables == []
    assert template.format() == "Hello World"


# --- dict() method ---


def test_base_prompt_template_dict_includes_type() -> None:
    """Test that dict() includes _type key."""
    template = PromptTemplate.from_template("Hello {name}")
    d = template.dict()
    assert d["_type"] == "prompt"
    assert d["template"] == "Hello {name}"


def test_base_prompt_template_dict_no_type_when_not_implemented() -> None:
    """Test dict() without _type when _prompt_type not implemented."""
    from langchain_core.prompts.string import StringPromptTemplate

    class CustomPrompt(StringPromptTemplate):
        def format(self, **kwargs: str) -> str:
            return "test"

    p = CustomPrompt(input_variables=[])
    d = p.dict()
    # _prompt_type raises NotImplementedError on StringPromptTemplate base,
    # but the dict method catches it with contextlib.suppress
    assert "_type" not in d


# --- save() ---


def test_save_prompt_to_json(tmp_path: Path) -> None:
    """Test saving a prompt to JSON file."""
    template = PromptTemplate.from_template("Hello {name}")
    file_path = tmp_path / "prompt.json"
    template.save(file_path)

    with file_path.open(encoding="utf-8") as f:
        data = json.load(f)

    assert data["template"] == "Hello {name}"
    assert data["_type"] == "prompt"


def test_save_prompt_to_yaml(tmp_path: Path) -> None:
    """Test saving a prompt to YAML file."""
    template = PromptTemplate.from_template("Hello {name}")
    file_path = tmp_path / "prompt.yaml"
    template.save(file_path)

    with file_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["template"] == "Hello {name}"
    assert data["_type"] == "prompt"


def test_save_prompt_to_yml(tmp_path: Path) -> None:
    """Test saving a prompt to .yml file."""
    template = PromptTemplate.from_template("Hello {name}")
    file_path = tmp_path / "prompt.yml"
    template.save(file_path)

    with file_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["template"] == "Hello {name}"


def test_save_prompt_unsupported_format(tmp_path: Path) -> None:
    """Test saving a prompt to unsupported file format raises error."""
    template = PromptTemplate.from_template("Hello {name}")
    with pytest.raises(ValueError, match="must be json or yaml"):
        template.save(tmp_path / "prompt.txt")


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    """Test that save creates parent directories."""
    template = PromptTemplate.from_template("Hello {name}")
    file_path = tmp_path / "subdir" / "nested" / "prompt.json"
    template.save(file_path)
    assert file_path.exists()


def test_save_with_partial_variables_raises() -> None:
    """Test that saving with partial variables raises error."""
    template = PromptTemplate(
        template="Hello {name} {greeting}",
        input_variables=["name"],
        partial_variables={"greeting": "Hi"},
    )
    with pytest.raises(ValueError, match="Cannot save prompt with partial variables"):
        template.save("test.json")


# --- OutputType property ---


def test_output_type_property() -> None:
    """Test that OutputType returns expected type."""
    template = PromptTemplate.from_template("Hello {name}")
    assert template.OutputType is not None


# --- aformat_prompt / aformat defaults ---


async def test_base_prompt_template_aformat_prompt_default() -> None:
    """Test that aformat_prompt calls sync format_prompt by default."""
    template = PromptTemplate.from_template("Hello {name}")
    result = await template.aformat_prompt(name="World")
    assert result.to_string() == "Hello World"


async def test_base_prompt_template_aformat_default() -> None:
    """Test that aformat calls sync format by default."""
    template = PromptTemplate.from_template("Hello {name}")
    result = await template.aformat(name="World")
    assert result == "Hello World"


# --- get_input_schema ---


def test_get_input_schema_optional_variables() -> None:
    """Test that optional variables are included in schema with default None."""
    from langchain_core.prompts.chat import ChatPromptTemplate

    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are helpful"),
            ("placeholder", "{history}"),
            ("human", "{input}"),
        ]
    )
    schema = template.get_input_schema()
    assert "input" in schema.model_fields
    assert "history" in schema.model_fields


# --- is_lc_serializable ---


def test_is_lc_serializable() -> None:
    """Test that BasePromptTemplate reports serializable."""
    assert BasePromptTemplate.is_lc_serializable() is True


# --- invoke with metadata and tags ---


def test_invoke_merges_metadata() -> None:
    """Test that invoke merges template metadata with config metadata."""
    from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

    template = PromptTemplate(
        template="Hello {name}",
        input_variables=["name"],
        metadata={"template_key": "v1"},
    )
    tracer = RunCollectorCallbackHandler()
    template.invoke(
        {"name": "World"},
        {"metadata": {"config_key": "v2"}, "callbacks": [tracer]},
    )
    assert len(tracer.traced_runs) == 1
    run_metadata = tracer.traced_runs[0].extra["metadata"]
    assert run_metadata["template_key"] == "v1"
    assert run_metadata["config_key"] == "v2"


def test_invoke_merges_tags() -> None:
    """Test that invoke merges template tags with config tags."""
    from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

    template = PromptTemplate(
        template="Hello {name}",
        input_variables=["name"],
        tags=["template_tag"],
    )
    tracer = RunCollectorCallbackHandler()
    template.invoke(
        {"name": "World"},
        {"tags": ["config_tag"], "callbacks": [tracer]},
    )
    assert len(tracer.traced_runs) == 1
    assert "template_tag" in tracer.traced_runs[0].tags
    assert "config_tag" in tracer.traced_runs[0].tags


# --- _validate_input edge cases ---


def test_validate_input_missing_variable_hint() -> None:
    """Test that missing variable error includes escape hint."""
    template = PromptTemplate.from_template("Hello {name} {greeting}")
    with pytest.raises(KeyError, match="double curly braces"):
        template.invoke({"name": "World"})


def test_validate_input_single_variable_non_dict() -> None:
    """Test that single-variable template accepts non-dict input."""
    template = PromptTemplate.from_template("Say {text}")
    result = template.invoke("hello")
    assert result.to_string() == "Say hello"


def test_validate_input_multiple_variables_non_dict_raises() -> None:
    """Test that multi-variable template raises on non-dict input."""
    template = PromptTemplate.from_template("Say {text} and {more}")
    with pytest.raises(TypeError, match="Expected mapping type"):
        template.invoke("hello")
