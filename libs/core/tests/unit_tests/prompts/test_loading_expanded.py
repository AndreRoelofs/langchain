"""Expanded tests for prompt loading functionality."""

import json
import logging
from pathlib import Path

import pytest
import yaml

from langchain_core.prompts.loading import (
    _load_chat_prompt,
    _load_examples,
    _load_output_parser,
    _load_prompt,
    _load_template,
    load_prompt,
    load_prompt_from_config,
)
from langchain_core.prompts.prompt import PromptTemplate

# --- load_prompt ---


def test_load_prompt_lc_hub_path_raises() -> None:
    """Test that loading from deprecated lc:// path raises RuntimeError."""
    with pytest.raises(RuntimeError, match="deprecated github-based Hub"):
        load_prompt("lc://some/path")


def test_load_prompt_unsupported_file_type_raises(tmp_path: Path) -> None:
    """Test that unsupported file type raises ValueError."""
    file = tmp_path / "prompt.txt"
    file.write_text("Hello {name}", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported file type"):
        load_prompt(file)


def test_load_prompt_from_json(tmp_path: Path) -> None:
    """Test loading prompt from JSON file."""
    config = {
        "_type": "prompt",
        "template": "Hello {name}",
        "input_variables": ["name"],
    }
    file = tmp_path / "prompt.json"
    with file.open("w", encoding="utf-8") as f:
        json.dump(config, f)

    prompt = load_prompt(file)
    assert prompt.format(name="World") == "Hello World"


def test_load_prompt_from_yaml(tmp_path: Path) -> None:
    """Test loading prompt from YAML file."""
    config = {
        "_type": "prompt",
        "template": "Hello {name}",
        "input_variables": ["name"],
    }
    file = tmp_path / "prompt.yaml"
    with file.open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    prompt = load_prompt(file)
    assert prompt.format(name="World") == "Hello World"


def test_load_prompt_from_yml(tmp_path: Path) -> None:
    """Test loading prompt from .yml file."""
    config = {
        "_type": "prompt",
        "template": "Hello {name}",
        "input_variables": ["name"],
    }
    file = tmp_path / "prompt.yml"
    with file.open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    prompt = load_prompt(file)
    assert prompt.format(name="World") == "Hello World"


def test_load_prompt_with_encoding(tmp_path: Path) -> None:
    """Test loading prompt with specific encoding."""
    config = {
        "_type": "prompt",
        "template": "Hello {name}",
        "input_variables": ["name"],
    }
    file = tmp_path / "prompt.json"
    with file.open("w", encoding="utf-8") as f:
        json.dump(config, f)

    prompt = load_prompt(file, encoding="utf-8")
    assert prompt.format(name="World") == "Hello World"


def test_load_prompt_path_as_string(tmp_path: Path) -> None:
    """Test loading prompt with path as string."""
    config = {
        "_type": "prompt",
        "template": "Hello {name}",
        "input_variables": ["name"],
    }
    file = tmp_path / "prompt.json"
    with file.open("w", encoding="utf-8") as f:
        json.dump(config, f)

    prompt = load_prompt(str(file))
    assert prompt.format(name="World") == "Hello World"


# --- load_prompt_from_config ---


def test_load_prompt_from_config_defaults_to_prompt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that missing _type defaults to 'prompt' with warning."""
    config = {
        "template": "Hello {name}",
        "input_variables": ["name"],
    }
    with caplog.at_level(logging.WARNING):
        prompt = load_prompt_from_config(config)

    assert prompt.format(name="World") == "Hello World"
    assert "No `_type` key found" in caplog.text


def test_load_prompt_from_config_unsupported_type_raises() -> None:
    """Test that unsupported _type raises ValueError."""
    config = {
        "_type": "unsupported_type",
        "template": "Hello",
    }
    with pytest.raises(
        ValueError, match="Loading unsupported_type prompt not supported"
    ):
        load_prompt_from_config(config)


# --- _load_template ---


def test_load_template_from_path(tmp_path: Path) -> None:
    """Test loading template from file path."""
    template_file = tmp_path / "template.txt"
    template_file.write_text("Hello {name}", encoding="utf-8")

    config = {"template_path": str(template_file)}
    result = _load_template("template", config)
    assert result["template"] == "Hello {name}"
    assert "template_path" not in result


def test_load_template_both_template_and_path_raises(tmp_path: Path) -> None:
    """Test that providing both template and template_path raises."""
    template_file = tmp_path / "template.txt"
    template_file.write_text("Hello", encoding="utf-8")

    config = {
        "template": "Direct",
        "template_path": str(template_file),
    }
    with pytest.raises(
        ValueError, match="Both `template_path` and `template` cannot be provided"
    ):
        _load_template("template", config)


def test_load_template_non_txt_file_raises(tmp_path: Path) -> None:
    """Test that non-.txt template file raises ValueError."""
    template_file = tmp_path / "template.json"
    template_file.write_text("{}", encoding="utf-8")

    config = {"template_path": str(template_file)}
    with pytest.raises(ValueError):
        _load_template("template", config)


def test_load_template_no_path_returns_unchanged() -> None:
    """Test that config without path is returned unchanged."""
    config = {"template": "Hello {name}"}
    result = _load_template("template", config)
    assert result == {"template": "Hello {name}"}


# --- _load_examples ---


def test_load_examples_from_list() -> None:
    """Test loading examples when they're already a list."""
    config = {"examples": [{"input": "a", "output": "b"}]}
    result = _load_examples(config)
    assert result["examples"] == [{"input": "a", "output": "b"}]


def test_load_examples_from_json_file(tmp_path: Path) -> None:
    """Test loading examples from JSON file."""
    examples = [{"input": "a", "output": "b"}]
    file = tmp_path / "examples.json"
    with file.open("w", encoding="utf-8") as f:
        json.dump(examples, f)

    config = {"examples": str(file)}
    result = _load_examples(config)
    assert result["examples"] == examples


def test_load_examples_from_yaml_file(tmp_path: Path) -> None:
    """Test loading examples from YAML file."""
    examples = [{"input": "a", "output": "b"}]
    file = tmp_path / "examples.yaml"
    with file.open("w", encoding="utf-8") as f:
        yaml.dump(examples, f)

    config = {"examples": str(file)}
    result = _load_examples(config)
    assert result["examples"] == examples


def test_load_examples_from_yml_file(tmp_path: Path) -> None:
    """Test loading examples from .yml file."""
    examples = [{"input": "a", "output": "b"}]
    file = tmp_path / "examples.yml"
    with file.open("w", encoding="utf-8") as f:
        yaml.dump(examples, f)

    config = {"examples": str(file)}
    result = _load_examples(config)
    assert result["examples"] == examples


def test_load_examples_unsupported_file_raises(tmp_path: Path) -> None:
    """Test that unsupported file format for examples raises."""
    file = tmp_path / "examples.txt"
    file.write_text("some text", encoding="utf-8")

    config = {"examples": str(file)}
    with pytest.raises(ValueError, match="Invalid file format"):
        _load_examples(config)


def test_load_examples_invalid_type_raises() -> None:
    """Test that invalid examples type raises."""
    config = {"examples": 42}
    with pytest.raises(ValueError, match="Invalid examples format"):
        _load_examples(config)


# --- _load_output_parser ---


def test_load_output_parser_none() -> None:
    """Test loading with no output_parser returns unchanged config."""
    config = {"template": "Hello"}
    result = _load_output_parser(config)
    assert result == {"template": "Hello"}


# --- _load_prompt ---


def test_load_prompt_basic() -> None:
    """Test _load_prompt with basic config."""
    config = {
        "template": "Hello {name}",
        "input_variables": ["name"],
    }
    prompt = _load_prompt(config)
    assert isinstance(prompt, PromptTemplate)
    assert prompt.format(name="World") == "Hello World"


def test_load_prompt_jinja2_raises() -> None:
    """Test _load_prompt rejects jinja2 format."""
    config = {
        "template": "Hello {{ name }}",
        "input_variables": ["name"],
        "template_format": "jinja2",
    }
    with pytest.raises(ValueError, match="can lead to arbitrary code execution"):
        _load_prompt(config)


# --- _load_chat_prompt ---


def test_load_chat_prompt_basic() -> None:
    """Test _load_chat_prompt with basic config."""
    config = {
        "messages": [
            {
                "prompt": {"template": "Hello {name}"},
            }
        ],
        "input_variables": ["name"],
    }
    prompt = _load_chat_prompt(config)
    messages = prompt.format_messages(name="World")
    assert len(messages) == 1
    assert messages[0].content == "Hello World"


def test_load_chat_prompt_no_template_raises() -> None:
    """Test _load_chat_prompt with empty messages raises."""
    config = {
        "messages": [],
        "input_variables": [],
    }
    with pytest.raises(ValueError, match="Can't load chat prompt without template"):
        _load_chat_prompt(config)


# --- Round-trip save/load ---


def test_round_trip_json(tmp_path: Path) -> None:
    """Test save to JSON and load back."""
    prompt = PromptTemplate(
        template="Hello {name}",
        input_variables=["name"],
    )
    file = tmp_path / "prompt.json"
    prompt.save(file)
    loaded = load_prompt(file)
    assert loaded == prompt


def test_round_trip_yaml(tmp_path: Path) -> None:
    """Test save to YAML and load back."""
    prompt = PromptTemplate(
        template="Hello {name}",
        input_variables=["name"],
    )
    file = tmp_path / "prompt.yaml"
    prompt.save(file)
    loaded = load_prompt(file)
    assert loaded == prompt
