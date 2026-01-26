"""Tests for image module (deprecated/removed functionality)."""

import pytest

from langchain_core.utils import image


def test_encode_image_raises_error() -> None:
    """Test that encode_image raises an appropriate error."""
    with pytest.raises(ValueError, match="has been removed for security reasons"):
        image.encode_image


def test_image_to_data_url_raises_error() -> None:
    """Test that image_to_data_url raises an appropriate error."""
    with pytest.raises(ValueError, match="has been removed for security reasons"):
        image.image_to_data_url


def test_invalid_attribute_raises_attribute_error() -> None:
    """Test that accessing non-existent attributes raises AttributeError."""
    with pytest.raises(AttributeError):
        image.some_random_attribute
