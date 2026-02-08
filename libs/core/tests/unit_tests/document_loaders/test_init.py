"""Tests for langchain_core.document_loaders.__init__ dynamic imports."""

import pytest


def test_all_contains_expected_names() -> None:
    """Verify __all__ lists the expected public names."""
    from langchain_core import document_loaders

    assert set(document_loaders.__all__) == {
        "BaseBlobParser",
        "BaseLoader",
        "Blob",
        "BlobLoader",
        "LangSmithLoader",
        "PathLike",
    }


def test_dir_matches_all() -> None:
    """Verify __dir__() returns the same names as __all__."""
    from langchain_core import document_loaders

    assert set(dir(document_loaders)) == set(document_loaders.__all__)


def test_dynamic_import_base_loader() -> None:
    """Test dynamic import of BaseLoader via __getattr__."""
    from langchain_core.document_loaders import BaseLoader
    from langchain_core.document_loaders.base import BaseLoader as DirectImport

    assert BaseLoader is DirectImport


def test_dynamic_import_base_blob_parser() -> None:
    """Test dynamic import of BaseBlobParser via __getattr__."""
    from langchain_core.document_loaders import BaseBlobParser
    from langchain_core.document_loaders.base import BaseBlobParser as DirectImport

    assert BaseBlobParser is DirectImport


def test_dynamic_import_blob() -> None:
    """Test dynamic import of Blob via __getattr__."""
    from langchain_core.document_loaders import Blob
    from langchain_core.document_loaders.blob_loaders import Blob as DirectImport

    assert Blob is DirectImport


def test_dynamic_import_blob_loader() -> None:
    """Test dynamic import of BlobLoader via __getattr__."""
    from langchain_core.document_loaders import BlobLoader
    from langchain_core.document_loaders.blob_loaders import BlobLoader as DirectImport

    assert BlobLoader is DirectImport


def test_dynamic_import_pathlike() -> None:
    """Test dynamic import of PathLike via __getattr__."""
    from langchain_core.document_loaders import PathLike
    from langchain_core.document_loaders.blob_loaders import PathLike as DirectImport

    assert PathLike is DirectImport


def test_dynamic_import_langsmith_loader() -> None:
    """Test dynamic import of LangSmithLoader via __getattr__."""
    from langchain_core.document_loaders import LangSmithLoader
    from langchain_core.document_loaders.langsmith import (
        LangSmithLoader as DirectImport,
    )

    assert LangSmithLoader is DirectImport


def test_invalid_attribute_raises_attribute_error() -> None:
    """Accessing an undefined attribute should raise AttributeError."""
    from langchain_core import document_loaders

    with pytest.raises(AttributeError):
        document_loaders.NonExistentClass  # noqa: B018


def test_dynamic_import_caches_in_globals() -> None:
    """After first access, the imported class should be cached in module globals."""
    from langchain_core import document_loaders

    # Access triggers __getattr__ which sets globals()
    _ = document_loaders.BaseLoader

    # Second access should hit the cached global, not __getattr__
    loader_cls = document_loaders.BaseLoader
    assert loader_cls is not None

    # Verify it's actually in the module dict
    assert "BaseLoader" in document_loaders.__dict__


def test_all_exports_are_importable() -> None:
    """Every name in __all__ should be importable without error."""
    from langchain_core import document_loaders

    for name in document_loaders.__all__:
        obj = getattr(document_loaders, name)
        assert obj is not None, f"{name} resolved to None"
