"""Expanded test suite for image prompt template."""

import pytest

from langchain_core.prompt_values import ImagePromptValue
from langchain_core.prompts.image import ImagePromptTemplate


def test_image_prompt_template_basic_url() -> None:
    """Test basic image prompt template with URL."""
    template = ImagePromptTemplate(
        template={"url": "https://example.com/image.png"},
        input_variables=[],
    )
    result = template.format()
    assert result == {"url": "https://example.com/image.png"}


def test_image_prompt_template_url_with_variable() -> None:
    """Test image prompt template with URL containing variable."""
    template = ImagePromptTemplate(
        template={"url": "https://example.com/{image_id}.png"},
        input_variables=["image_id"],
    )
    result = template.format(image_id="12345")
    assert result == {"url": "https://example.com/12345.png"}


def test_image_prompt_template_with_detail() -> None:
    """Test image prompt template with detail parameter."""
    template = ImagePromptTemplate(
        template={"url": "https://example.com/image.png", "detail": "high"},
        input_variables=[],
    )
    result = template.format()
    assert result == {"url": "https://example.com/image.png", "detail": "high"}


def test_image_prompt_template_with_detail_variable() -> None:
    """Test image prompt template with detail as variable."""
    template = ImagePromptTemplate(
        template={"url": "https://example.com/image.png"},
        input_variables=[],
    )
    result = template.format(detail="low")
    assert result == {"url": "https://example.com/image.png", "detail": "low"}


def test_image_prompt_template_mustache_format() -> None:
    """Test image prompt template with mustache format."""
    template = ImagePromptTemplate(
        template={"url": "data:image/png;base64,{{image_data}}"},
        input_variables=["image_data"],
        template_format="mustache",
    )
    result = template.format(image_data="abc123")
    assert result == {"url": "data:image/png;base64,abc123"}


def test_image_prompt_template_path_raises_error() -> None:
    """Test that using path raises ValueError for security."""
    template = ImagePromptTemplate(
        template={"path": "/some/path/image.png"},
        input_variables=[],
    )

    with pytest.raises(ValueError, match="Loading images from 'path' has been removed"):
        template.format()


def test_image_prompt_template_path_in_kwargs_raises_error() -> None:
    """Test that passing path in kwargs raises ValueError."""
    template = ImagePromptTemplate(
        template={"url": "https://example.com/image.png"},
        input_variables=[],
    )

    with pytest.raises(ValueError, match="Loading images from 'path' has been removed"):
        template.format(path="/some/path")


def test_image_prompt_template_missing_url_raises_error() -> None:
    """Test that missing URL raises ValueError."""
    template = ImagePromptTemplate(
        template={},
        input_variables=[],
    )

    with pytest.raises(ValueError, match="Must provide url"):
        template.format()


def test_image_prompt_template_non_string_url_raises_error() -> None:
    """Test that non-string URL raises ValueError."""
    template = ImagePromptTemplate(
        template={"url": 123},  # type: ignore[dict-item]
        input_variables=[],
    )

    with pytest.raises(ValueError, match="url must be a string"):
        template.format()


async def test_image_prompt_template_aformat() -> None:
    """Test async formatting of image prompt template."""
    template = ImagePromptTemplate(
        template={"url": "https://example.com/{image_id}.png"},
        input_variables=["image_id"],
    )
    result = await template.aformat(image_id="async123")
    assert result == {"url": "https://example.com/async123.png"}


def test_image_prompt_template_format_prompt() -> None:
    """Test format_prompt returns ImagePromptValue."""
    template = ImagePromptTemplate(
        template={"url": "https://example.com/image.png"},
        input_variables=[],
    )
    result = template.format_prompt()

    assert isinstance(result, ImagePromptValue)
    assert result.image_url == {"url": "https://example.com/image.png"}


async def test_image_prompt_template_aformat_prompt() -> None:
    """Test async format_prompt returns ImagePromptValue."""
    template = ImagePromptTemplate(
        template={"url": "https://example.com/image.png"},
        input_variables=[],
    )
    result = await template.aformat_prompt()

    assert isinstance(result, ImagePromptValue)
    assert result.image_url == {"url": "https://example.com/image.png"}


def test_image_prompt_template_init_with_reserved_variables() -> None:
    """Test that initializing with reserved variable names raises error."""
    with pytest.raises(ValueError, match="input_variables.*cannot contain.*'url'"):
        ImagePromptTemplate(
            template={"url": "{url}"},
            input_variables=["url"],
        )

    with pytest.raises(ValueError, match="input_variables.*cannot contain.*'path'"):
        ImagePromptTemplate(
            template={"url": "https://example.com/image.png"},
            input_variables=["path"],
        )

    with pytest.raises(ValueError, match="input_variables.*cannot contain.*'detail'"):
        ImagePromptTemplate(
            template={"url": "https://example.com/image.png"},
            input_variables=["detail"],
        )


def test_image_prompt_template_prompt_type() -> None:
    """Test that _prompt_type returns correct value."""
    template = ImagePromptTemplate(
        template={"url": "https://example.com/image.png"},
        input_variables=[],
    )
    assert template._prompt_type == "image-prompt"


def test_image_prompt_template_lc_namespace() -> None:
    """Test that get_lc_namespace returns correct value."""
    assert ImagePromptTemplate.get_lc_namespace() == [
        "langchain",
        "prompts",
        "image",
    ]


def test_image_prompt_template_with_base64_data() -> None:
    """Test image prompt template with base64 encoded data."""
    template = ImagePromptTemplate(
        template={"url": "data:image/png;base64,{data}"},
        input_variables=["data"],
    )
    result = template.format(data="iVBORw0KGgoAAAANSUhEUgA")
    assert result == {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA"}


def test_image_prompt_template_pretty_repr_not_implemented() -> None:
    """Test that pretty_repr raises NotImplementedError."""
    template = ImagePromptTemplate(
        template={"url": "https://example.com/image.png"},
        input_variables=[],
    )

    with pytest.raises(NotImplementedError):
        template.pretty_repr()


def test_image_prompt_template_url_from_kwargs() -> None:
    """Test that URL can be provided via kwargs."""
    template = ImagePromptTemplate(
        template={},
        input_variables=[],
    )
    result = template.format(url="https://example.com/image.png")
    assert result == {"url": "https://example.com/image.png"}


def test_image_prompt_template_detail_from_kwargs() -> None:
    """Test that detail can be provided via kwargs."""
    template = ImagePromptTemplate(
        template={"url": "https://example.com/image.png"},
        input_variables=[],
    )
    result = template.format(detail="auto")
    assert result == {"url": "https://example.com/image.png", "detail": "auto"}


def test_image_prompt_template_multiple_variables() -> None:
    """Test image prompt template with multiple variables."""
    template = ImagePromptTemplate(
        template={"url": "https://{domain}/{folder}/{filename}"},
        input_variables=["domain", "folder", "filename"],
    )
    result = template.format(
        domain="example.com",
        folder="images",
        filename="photo.png",
    )
    assert result == {"url": "https://example.com/images/photo.png"}


def test_image_prompt_template_with_static_detail_and_variable_url() -> None:
    """Test template with static detail and variable URL."""
    template = ImagePromptTemplate(
        template={
            "url": "https://example.com/{image_id}.png",
            "detail": "high",
        },
        input_variables=["image_id"],
    )
    result = template.format(image_id="photo123")
    assert result == {
        "url": "https://example.com/photo123.png",
        "detail": "high",
    }
