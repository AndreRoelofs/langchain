from langchain_core.load import __all__

EXPECTED_ALL = ["dumpd", "dumps", "load", "loads", "Serializable"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)


# ---------------------------------------------------------------------------
# Additional snapshot tests for __init__.py
# ---------------------------------------------------------------------------


def test_all_imports_exact_tuple() -> None:
    """Snapshot: __all__ is a tuple with exact contents."""
    assert isinstance(__all__, tuple)
    assert __all__ == ("Serializable", "dumpd", "dumps", "load", "loads")


def test_dynamic_import_dumpd() -> None:
    """Snapshot: dumpd is importable from langchain_core.load."""
    from langchain_core.load import dumpd

    assert callable(dumpd)


def test_dynamic_import_dumps() -> None:
    """Snapshot: dumps is importable from langchain_core.load."""
    from langchain_core.load import dumps

    assert callable(dumps)


def test_dynamic_import_loads() -> None:
    """Snapshot: loads is importable from langchain_core.load."""
    from langchain_core.load import loads

    assert callable(loads)


def test_dynamic_import_serializable() -> None:
    """Snapshot: Serializable is importable from langchain_core.load."""
    from langchain_core.load import Serializable

    assert isinstance(Serializable, type)


def test_eager_import_load() -> None:
    """Snapshot: load is eagerly imported from langchain_core.load."""
    from langchain_core.load import load

    assert callable(load)


def test_dir_returns_all_exports() -> None:
    """Snapshot: dir(langchain_core.load) returns all exported names."""
    import langchain_core.load as load_module

    dir_result = dir(load_module)
    for name in EXPECTED_ALL:
        assert name in dir_result


def test_dynamic_imports_resolve_to_correct_modules() -> None:
    """Snapshot: dynamic imports resolve to the correct source modules."""
    from langchain_core.load import Serializable, dumpd, dumps, load, loads

    assert dumpd.__module__ == "langchain_core.load.dump"
    assert dumps.__module__ == "langchain_core.load.dump"
    assert loads.__module__ == "langchain_core.load.load"
    assert load.__module__ == "langchain_core.load.load"
    assert Serializable.__module__ == "langchain_core.load.serializable"


def test_load_submodule_import_also_works() -> None:
    """Snapshot: absolute imports from submodules still work."""
    from langchain_core.load.dump import dumpd, dumps
    from langchain_core.load.load import load, loads
    from langchain_core.load.serializable import Serializable

    assert callable(dumpd)
    assert callable(dumps)
    assert callable(load)
    assert callable(loads)
    assert isinstance(Serializable, type)
