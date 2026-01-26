import datetime
import uuid
from unittest.mock import MagicMock, patch

import pytest
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

    with patch("langsmith.Client.list_examples", MagicMock(return_value=iter([example_with_dict]))):
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

    with patch("langsmith.Client.list_examples", MagicMock(return_value=iter([example]))):
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

    with patch("langsmith.Client.list_examples", MagicMock(return_value=iter([example]))):
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

    with patch("langsmith.Client.list_examples", MagicMock(return_value=iter([example]))):
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

    with patch("langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES))):
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
    with patch("langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES))):
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
    with patch("langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES))):
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
    with patch("langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES))):
        loader = LangSmithLoader(
            dataset_id="test",
            content_key="first.second",
            api_key="dummy",
        )
        docs = loader.load()

        assert len(docs) == 3
        assert isinstance(docs, list)
        assert all(isinstance(doc, Document) for doc in docs)
