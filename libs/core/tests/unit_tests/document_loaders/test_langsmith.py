import datetime
import json
import uuid
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langsmith import Client as LangSmithClient
from langsmith.schemas import Example

from langchain_core.document_loaders import LangSmithLoader
from langchain_core.documents import Document


def test_init() -> None:
    LangSmithLoader(api_key="secret")


EXAMPLES = [
    Example(
        inputs={"first": {"second": "foo"}},
        outputs={"res": "a"},
        dataset_id=uuid.uuid4(),
        id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    ),
    Example(
        inputs={"first": {"second": "bar"}},
        outputs={"res": "b"},
        dataset_id=uuid.uuid4(),
        id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    ),
    Example(
        inputs={"first": {"second": "baz"}},
        outputs={"res": "c"},
        dataset_id=uuid.uuid4(),
        id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    ),
]


@patch("langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES)))
def test_lazy_load() -> None:
    loader = LangSmithLoader(
        api_key="dummy",
        dataset_id="mock",
        content_key="first.second",
        format_content=(lambda x: x.upper()),
    )
    expected = []
    for example in EXAMPLES:
        metadata = {
            k: v if not v or isinstance(v, dict) else str(v)
            for k, v in example.dict().items()
        }
        expected.append(
            Document(example.inputs["first"]["second"].upper(), metadata=metadata)
            if example.inputs
            else None
        )
    actual = list(loader.lazy_load())
    assert expected == actual


def test_init_with_both_client_and_kwargs() -> None:
    """Test that ValueError is raised when both client and client_kwargs provided."""
    from langsmith import Client

    client = Client(api_key="test")

    with pytest.raises(ValueError):
        LangSmithLoader(client=client, api_key="also_test")


def test_init_with_dataset_name() -> None:
    """Test initialization with dataset_name instead of dataset_id."""
    loader = LangSmithLoader(dataset_name="my-dataset", api_key="test-key")
    assert loader.dataset_name == "my-dataset"
    assert loader.dataset_id is None


def test_init_with_all_parameters() -> None:
    """Test initialization with all available parameters."""
    metadata = {"type": "test", "version": "1.0"}

    loader = LangSmithLoader(
        dataset_id=str(uuid.uuid4()),
        dataset_name="test-dataset",
        example_ids=[uuid.uuid4(), uuid.uuid4()],
        as_of="2024-01-01T00:00:00Z",
        splits=["train", "test"],
        inline_s3_urls=False,
        offset=10,
        limit=50,
        metadata=metadata,
        filter="key eq 'value'",
        content_key="nested.key.path",
        format_content=str.upper,
        api_key="test-key",
    )

    assert loader.dataset_name == "test-dataset"
    assert loader.offset == 10
    assert loader.limit == 50
    assert loader.metadata == metadata
    assert loader.filter == "key eq 'value'"
    assert loader.content_key == ["nested", "key", "path"]
    assert loader.inline_s3_urls is False
    assert loader.splits == ["train", "test"]


@patch("langsmith.Client.list_examples", MagicMock(return_value=iter([])))
def test_lazy_load_empty_dataset() -> None:
    """Test lazy_load with empty dataset."""
    loader = LangSmithLoader(dataset_id="empty", api_key="test")
    docs = list(loader.lazy_load())
    assert docs == []


@patch("langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES)))
def test_lazy_load_without_content_key() -> None:
    """Test lazy_load without specifying content_key (uses entire inputs)."""
    loader = LangSmithLoader(dataset_id="test", api_key="dummy")
    docs = list(loader.lazy_load())

    assert len(docs) == 3
    # When no content_key, entire inputs should be stringified
    for i, doc in enumerate(docs):
        assert "first" in doc.page_content or '"first"' in doc.page_content
        assert doc.metadata["id"] == str(EXAMPLES[i].id)


@patch("langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES)))
def test_lazy_load_with_single_level_key() -> None:
    """Test lazy_load with single-level content_key."""
    loader = LangSmithLoader(
        dataset_id="test",
        content_key="first",
        api_key="dummy",
    )
    docs = list(loader.lazy_load())

    assert len(docs) == 3
    for doc in docs:
        # The content should be the "first" dict stringified
        assert "second" in doc.page_content


@patch("langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES)))
def test_lazy_load_default_format_content() -> None:
    """Test that default format_content function handles dict and str correctly."""
    loader = LangSmithLoader(
        dataset_id="test",
        content_key="first.second",
        api_key="dummy",
    )
    docs = list(loader.lazy_load())

    assert len(docs) == 3
    # Default format_content should return strings as-is
    assert docs[0].page_content == "foo"
    assert docs[1].page_content == "bar"
    assert docs[2].page_content == "baz"


def test_lazy_load_with_dict_content() -> None:
    """Test handling of dict content with default formatter."""
    example_with_dict = Example(
        inputs={"data": {"nested": {"key": "value"}, "list": [1, 2, 3]}},
        outputs={"res": "x"},
        dataset_id=uuid.uuid4(),
        id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )

    with patch(
        "langsmith.Client.list_examples",
        MagicMock(return_value=iter([example_with_dict])),
    ):
        loader = LangSmithLoader(
            dataset_id="test",
            content_key="data",
            api_key="dummy",
        )
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        # Should be JSON-formatted
        assert "nested" in docs[0].page_content
        assert "key" in docs[0].page_content
        assert "value" in docs[0].page_content


def test_metadata_stringification() -> None:
    """Test that datetime and UUID metadata fields are properly stringified."""
    now = datetime.datetime.now(datetime.timezone.utc)
    dataset_uuid = uuid.uuid4()
    example_uuid = uuid.uuid4()
    source_run_uuid = uuid.uuid4()

    example = Example(
        inputs={"content": "test"},
        outputs={"result": "output"},
        dataset_id=dataset_uuid,
        id=example_uuid,
        created_at=now,
        modified_at=now,
        source_run_id=source_run_uuid,
    )

    with patch(
        "langsmith.Client.list_examples", MagicMock(return_value=iter([example]))
    ):
        loader = LangSmithLoader(
            dataset_id="test",
            content_key="content",
            api_key="dummy",
        )
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        metadata = docs[0].metadata

        # Check that UUIDs are stringified
        assert metadata["dataset_id"] == str(dataset_uuid)
        assert metadata["id"] == str(example_uuid)
        assert metadata["source_run_id"] == str(source_run_uuid)

        # Check that datetimes are stringified
        assert isinstance(metadata["created_at"], str)
        assert isinstance(metadata["modified_at"], str)


def test_metadata_with_none_values() -> None:
    """Test handling of None values in metadata."""
    example = Example(
        inputs={"content": "test"},
        outputs={"result": "output"},
        dataset_id=uuid.uuid4(),
        id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
        modified_at=None,
        source_run_id=None,
    )

    with patch(
        "langsmith.Client.list_examples", MagicMock(return_value=iter([example]))
    ):
        loader = LangSmithLoader(
            dataset_id="test",
            content_key="content",
            api_key="dummy",
        )
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        metadata = docs[0].metadata

        # None values should remain None (not stringified)
        assert metadata["modified_at"] is None
        assert metadata["source_run_id"] is None


def test_content_key_with_empty_string() -> None:
    """Test that empty content_key uses entire inputs."""
    example = Example(
        inputs={"key": "value"},
        outputs=None,
        dataset_id=uuid.uuid4(),
        id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )

    with patch(
        "langsmith.Client.list_examples", MagicMock(return_value=iter([example]))
    ):
        loader = LangSmithLoader(
            dataset_id="test",
            content_key="",
            api_key="dummy",
        )
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        # With empty content_key, should get stringified inputs dict
        assert "key" in docs[0].page_content


def test_custom_format_content_function() -> None:
    """Test using a custom format_content function."""

    def custom_formatter(content: str) -> str:
        return f"[CUSTOM] {content.upper()} [/CUSTOM]"

    with patch(
        "langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES))
    ):
        loader = LangSmithLoader(
            dataset_id="test",
            content_key="first.second",
            format_content=custom_formatter,
            api_key="dummy",
        )
        docs = list(loader.lazy_load())

        assert len(docs) == 3
        assert docs[0].page_content == "[CUSTOM] FOO [/CUSTOM]"
        assert docs[1].page_content == "[CUSTOM] BAR [/CUSTOM]"
        assert docs[2].page_content == "[CUSTOM] BAZ [/CUSTOM]"


@patch("langsmith.Client.list_examples")
def test_lazy_load_passes_all_parameters(mock_list_examples: MagicMock) -> None:
    """Test that lazy_load passes all parameters to client.list_examples."""
    mock_list_examples.return_value = iter([])

    dataset_uuid = uuid.uuid4()
    example_ids = [uuid.uuid4(), uuid.uuid4()]
    as_of = "2024-01-01T00:00:00Z"
    splits = ["train", "test"]
    metadata = {"key": "value"}
    filter_str = "tag eq 'test'"

    loader = LangSmithLoader(
        dataset_id=dataset_uuid,
        dataset_name="test-dataset",
        example_ids=example_ids,
        as_of=as_of,
        splits=splits,
        inline_s3_urls=False,
        offset=5,
        limit=20,
        metadata=metadata,
        filter=filter_str,
        api_key="test-key",
    )

    list(loader.lazy_load())

    mock_list_examples.assert_called_once_with(
        dataset_id=dataset_uuid,
        dataset_name="test-dataset",
        example_ids=example_ids,
        as_of=as_of,
        splits=splits,
        inline_s3_urls=False,
        offset=5,
        limit=20,
        metadata=metadata,
        filter=filter_str,
    )


async def test_aload() -> None:
    """Test async loading via inherited aload method."""
    with patch(
        "langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES))
    ):
        loader = LangSmithLoader(
            dataset_id="test",
            content_key="first.second",
            api_key="dummy",
        )
        docs = await loader.aload()

        assert len(docs) == 3
        assert docs[0].page_content == "foo"
        assert docs[1].page_content == "bar"
        assert docs[2].page_content == "baz"


async def test_alazy_load() -> None:
    """Test async lazy loading via inherited alazy_load method."""
    with patch(
        "langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES))
    ):
        loader = LangSmithLoader(
            dataset_id="test",
            content_key="first.second",
            api_key="dummy",
        )
        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)

        assert len(docs) == 3
        assert docs[0].page_content == "foo"
        assert docs[1].page_content == "bar"
        assert docs[2].page_content == "baz"


def test_load_method() -> None:
    """Test synchronous load method (convenience method)."""
    with patch(
        "langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES))
    ):
        loader = LangSmithLoader(
            dataset_id="test",
            content_key="first.second",
            api_key="dummy",
        )
        docs = loader.load()

        assert len(docs) == 3
        assert isinstance(docs, list)
        assert all(isinstance(doc, Document) for doc in docs)


def test_stringify_with_string_input() -> None:
    """Test _stringify returns strings as-is."""
    from langchain_core.document_loaders.langsmith import _stringify

    assert _stringify("hello") == "hello"
    assert _stringify("") == ""
    assert _stringify("multi\nline") == "multi\nline"


def test_stringify_with_dict_input() -> None:
    """Test _stringify JSON-encodes dicts with indent=2."""
    from langchain_core.document_loaders.langsmith import _stringify

    result = _stringify({"key": "value"})
    assert '"key": "value"' in result
    # Should be indented JSON
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_stringify_with_nested_dict() -> None:
    """Test _stringify with nested dict structure."""
    from langchain_core.document_loaders.langsmith import _stringify

    data = {"outer": {"inner": [1, 2, 3]}}
    result = _stringify(data)
    parsed = json.loads(result)
    assert parsed == data


def test_stringify_with_non_serializable_dict() -> None:
    """Test _stringify falls back to str() for non-JSON-serializable dicts."""
    from langchain_core.document_loaders.langsmith import _stringify

    class NotSerializable:
        def __repr__(self) -> str:
            return "NotSerializable()"

    data = {"key": NotSerializable()}
    result = _stringify(data)
    # Should fall back to str()
    assert isinstance(result, str)
    assert "key" in result


def test_content_key_splitting() -> None:
    """Test that content_key is properly split on dots."""
    loader = LangSmithLoader(
        content_key="a.b.c",
        api_key="dummy",
    )
    assert loader.content_key == ["a", "b", "c"]


def test_content_key_single_key_no_dots() -> None:
    """Test content_key with a single key (no dots)."""
    loader = LangSmithLoader(
        content_key="simple",
        api_key="dummy",
    )
    assert loader.content_key == ["simple"]


def test_content_key_empty_string_results_in_empty_list() -> None:
    """Test that empty string content_key results in empty list."""
    loader = LangSmithLoader(
        content_key="",
        api_key="dummy",
    )
    assert loader.content_key == []


def test_deeply_nested_content_key() -> None:
    """Test content_key with 3+ levels of nesting."""
    example = Example(
        inputs={"a": {"b": {"c": {"d": "deeply_nested_value"}}}},
        outputs=None,
        dataset_id=uuid.uuid4(),
        id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )

    with patch(
        "langsmith.Client.list_examples",
        MagicMock(return_value=iter([example])),
    ):
        loader = LangSmithLoader(
            dataset_id="test",
            content_key="a.b.c.d",
            api_key="dummy",
        )
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        assert docs[0].page_content == "deeply_nested_value"


def test_missing_content_key_raises_key_error() -> None:
    """Test that a missing key in the content_key path raises KeyError."""
    example = Example(
        inputs={"a": {"b": "value"}},
        outputs=None,
        dataset_id=uuid.uuid4(),
        id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )

    with patch(
        "langsmith.Client.list_examples",
        MagicMock(return_value=iter([example])),
    ):
        loader = LangSmithLoader(
            dataset_id="test",
            content_key="a.nonexistent",
            api_key="dummy",
        )
        with pytest.raises(KeyError):
            list(loader.lazy_load())


def test_format_content_default_is_stringify() -> None:
    """Test that the default format_content is _stringify."""
    from langchain_core.document_loaders.langsmith import _stringify

    loader = LangSmithLoader(api_key="dummy")
    assert loader.format_content is _stringify


def test_langsmith_loader_inherits_from_base_loader() -> None:
    """Verify LangSmithLoader is a subclass of BaseLoader."""
    from langchain_core.document_loaders.base import BaseLoader

    assert issubclass(LangSmithLoader, BaseLoader)


def test_init_with_explicit_client() -> None:
    """Test initialization with an explicit client and no kwargs."""
    client = MagicMock(spec=LangSmithClient)

    loader = LangSmithLoader(client=client, dataset_id="test")

    assert loader._client is client
    assert loader.dataset_id == "test"


def test_init_default_values() -> None:
    """Test that all default values are set correctly on init."""
    loader = LangSmithLoader(api_key="dummy")

    assert loader.dataset_id is None
    assert loader.dataset_name is None
    assert loader.example_ids is None
    assert loader.as_of is None
    assert loader.splits is None
    assert loader.inline_s3_urls is True
    assert loader.offset == 0
    assert loader.limit is None
    assert loader.metadata is None
    assert loader.filter is None
    assert loader.content_key == []


def test_lazy_load_metadata_contains_example_fields() -> None:
    """Test that metadata from lazy_load contains expected example fields."""
    dataset_uuid = uuid.uuid4()
    example_uuid = uuid.uuid4()
    now = datetime.datetime.now(datetime.timezone.utc)

    example = Example(
        inputs={"text": "hello"},
        outputs={"result": "world"},
        dataset_id=dataset_uuid,
        id=example_uuid,
        created_at=now,
    )

    with patch(
        "langsmith.Client.list_examples",
        MagicMock(return_value=iter([example])),
    ):
        loader = LangSmithLoader(
            dataset_id="test",
            content_key="text",
            api_key="dummy",
        )
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        metadata = docs[0].metadata

        # These fields should be present in the metadata
        assert "inputs" in metadata
        assert "outputs" in metadata
        assert "dataset_id" in metadata
        assert "id" in metadata
        assert "created_at" in metadata

        # Stringified fields
        assert metadata["dataset_id"] == str(dataset_uuid)
        assert metadata["id"] == str(example_uuid)


def test_lazy_load_with_multiple_examples_produces_correct_count() -> None:
    """Verify that lazy_load yields exactly one Document per Example."""
    examples = [
        Example(
            inputs={"content": f"item-{i}"},
            outputs=None,
            dataset_id=uuid.uuid4(),
            id=uuid.uuid4(),
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        for i in range(7)
    ]

    with patch(
        "langsmith.Client.list_examples",
        MagicMock(return_value=iter(examples)),
    ):
        loader = LangSmithLoader(
            dataset_id="test",
            content_key="content",
            api_key="dummy",
        )
        docs = list(loader.lazy_load())

        assert len(docs) == 7
        for i, doc in enumerate(docs):
            assert doc.page_content == f"item-{i}"


def test_lazy_load_is_lazy() -> None:
    """Verify lazy_load is actually lazy — documents produced on demand."""
    call_count = 0

    def counting_examples(**kwargs: Any) -> Iterator:
        nonlocal call_count
        for ex in EXAMPLES:
            call_count += 1
            yield ex

    with patch("langsmith.Client.list_examples", side_effect=counting_examples):
        loader = LangSmithLoader(
            dataset_id="test",
            content_key="first.second",
            api_key="dummy",
        )
        gen = loader.lazy_load()

        assert call_count == 0
        next(gen)
        assert call_count == 1


def test_init_with_client_kwargs_only() -> None:
    """Test that client_kwargs are forwarded to the LangSmith Client."""
    with patch("langchain_core.document_loaders.langsmith.LangSmithClient") as mock_cls:
        LangSmithLoader(api_key="my-key", api_url="https://custom.api.com")
        mock_cls.assert_called_once_with(
            api_key="my-key", api_url="https://custom.api.com"
        )


def test_format_content_receives_extracted_content() -> None:
    """Verify format_content receives the content after key extraction,
    not the raw inputs."""
    received_values: list[Any] = []

    def capturing_formatter(x: Any) -> str:
        received_values.append(x)
        return str(x)

    example = Example(
        inputs={"outer": {"inner": "target_value"}},
        outputs=None,
        dataset_id=uuid.uuid4(),
        id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )

    with patch(
        "langsmith.Client.list_examples",
        MagicMock(return_value=iter([example])),
    ):
        loader = LangSmithLoader(
            dataset_id="test",
            content_key="outer.inner",
            format_content=capturing_formatter,
            api_key="dummy",
        )
        list(loader.lazy_load())

    assert len(received_values) == 1
    assert received_values[0] == "target_value"
