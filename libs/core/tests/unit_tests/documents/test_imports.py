import importlib

import pytest

import langchain_core.documents as documents_module
from langchain_core.documents import __all__

EXPECTED_ALL = ["Document", "BaseDocumentTransformer", "BaseDocumentCompressor"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)


def test_dir_returns_all_exports() -> None:
    """Test that dir() on the module returns the exported names."""
    module_dir = dir(documents_module)
    for name in EXPECTED_ALL:
        assert name in module_dir


def test_dynamic_import_document() -> None:
    """Test that Document can be dynamically imported via __getattr__."""
    # Force a fresh module state by reimporting
    mod = importlib.import_module("langchain_core.documents")
    cls = getattr(mod, "Document")
    from langchain_core.documents.base import Document

    assert cls is Document


def test_dynamic_import_base_document_compressor() -> None:
    """Test that BaseDocumentCompressor can be dynamically imported."""
    mod = importlib.import_module("langchain_core.documents")
    cls = getattr(mod, "BaseDocumentCompressor")
    from langchain_core.documents.compressor import BaseDocumentCompressor

    assert cls is BaseDocumentCompressor


def test_dynamic_import_base_document_transformer() -> None:
    """Test that BaseDocumentTransformer can be dynamically imported."""
    mod = importlib.import_module("langchain_core.documents")
    cls = getattr(mod, "BaseDocumentTransformer")
    from langchain_core.documents.transformers import BaseDocumentTransformer

    assert cls is BaseDocumentTransformer


def test_invalid_attribute_raises_attribute_error() -> None:
    """Test that accessing a non-existent attribute raises AttributeError."""
    with pytest.raises(AttributeError):
        _ = documents_module.NonExistentClass  # type: ignore[attr-defined]


def test_all_exports_are_importable() -> None:
    """Test that all exports in __all__ can actually be imported."""
    for name in EXPECTED_ALL:
        obj = getattr(documents_module, name)
        assert obj is not None
