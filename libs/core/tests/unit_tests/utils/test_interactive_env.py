"""Tests for interactive environment detection."""

import sys

from langchain_core.utils.interactive_env import is_interactive_env


def test_is_interactive_env_not_interactive() -> None:
    """Test that is_interactive_env returns False in normal test environment."""
    # In a normal test environment, sys.ps2 shouldn't be set
    result = is_interactive_env()
    assert isinstance(result, bool)
    # In most test environments, this should be False
    assert result is False


def test_is_interactive_env_with_ps2() -> None:
    """Test that is_interactive_env returns True when ps2 is set."""
    # Simulate interactive environment by setting ps2
    original_ps2 = getattr(sys, "ps2", None)
    try:
        sys.ps2 = "... "
        result = is_interactive_env()
        assert result is True
    finally:
        # Clean up
        if original_ps2 is None:
            if hasattr(sys, "ps2"):
                delattr(sys, "ps2")
        else:
            sys.ps2 = original_ps2


def test_is_interactive_env_without_ps2() -> None:
    """Test that is_interactive_env returns False when ps2 is not set."""
    # Ensure ps2 is not set
    original_ps2 = getattr(sys, "ps2", None)
    try:
        if hasattr(sys, "ps2"):
            delattr(sys, "ps2")
        result = is_interactive_env()
        assert result is False
    finally:
        # Restore original state
        if original_ps2 is not None:
            sys.ps2 = original_ps2
