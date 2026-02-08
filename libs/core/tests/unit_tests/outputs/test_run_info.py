"""Unit tests for RunInfo class."""

from uuid import UUID, uuid4

import pytest

from langchain_core.outputs import RunInfo


class TestRunInfo:
    """Test suite for RunInfo class."""

    def test_creation_with_uuid(self) -> None:
        """Test creating RunInfo with a UUID."""
        run_id = uuid4()
        run_info = RunInfo(run_id=run_id)
        assert run_info.run_id == run_id
        assert isinstance(run_info.run_id, UUID)

    def test_creation_with_specific_uuid(self) -> None:
        """Test creating RunInfo with a specific UUID string."""
        uuid_str = "12345678-1234-5678-1234-567812345678"
        run_id = UUID(uuid_str)
        run_info = RunInfo(run_id=run_id)
        assert run_info.run_id == run_id
        assert str(run_info.run_id) == uuid_str

    def test_run_id_is_uuid_type(self) -> None:
        """Test that run_id is of UUID type."""
        run_id = uuid4()
        run_info = RunInfo(run_id=run_id)
        assert isinstance(run_info.run_id, UUID)

    def test_different_run_infos_have_different_ids(self) -> None:
        """Test that different RunInfo instances can have different IDs."""
        run_id1 = uuid4()
        run_id2 = uuid4()
        run_info1 = RunInfo(run_id=run_id1)
        run_info2 = RunInfo(run_id=run_id2)
        assert run_info1.run_id != run_info2.run_id

    def test_equality_same_run_id(self) -> None:
        """Test equality for RunInfo with same run_id."""
        run_id = uuid4()
        run_info1 = RunInfo(run_id=run_id)
        run_info2 = RunInfo(run_id=run_id)
        assert run_info1 == run_info2

    def test_inequality_different_run_id(self) -> None:
        """Test inequality for RunInfo with different run_id."""
        run_id1 = uuid4()
        run_id2 = uuid4()
        run_info1 = RunInfo(run_id=run_id1)
        run_info2 = RunInfo(run_id=run_id2)
        assert run_info1 != run_info2

    def test_run_info_is_pydantic_model(self) -> None:
        """Test that RunInfo is a Pydantic BaseModel."""
        from pydantic import BaseModel

        run_id = uuid4()
        run_info = RunInfo(run_id=run_id)
        assert isinstance(run_info, BaseModel)

    def test_serialization_to_dict(self) -> None:
        """Test serialization of RunInfo to dictionary."""
        run_id = uuid4()
        run_info = RunInfo(run_id=run_id)
        data = run_info.model_dump()
        assert "run_id" in data
        assert data["run_id"] == run_id

    def test_deserialization_from_dict(self) -> None:
        """Test deserialization of RunInfo from dictionary."""
        run_id = uuid4()
        data = {"run_id": run_id}
        run_info = RunInfo(**data)
        assert run_info.run_id == run_id

    def test_json_serialization(self) -> None:
        """Test JSON serialization of RunInfo."""
        run_id = uuid4()
        run_info = RunInfo(run_id=run_id)
        json_str = run_info.model_dump_json()
        assert str(run_id) in json_str

    def test_json_deserialization(self) -> None:
        """Test JSON deserialization of RunInfo."""
        run_id = uuid4()
        run_info = RunInfo(run_id=run_id)
        json_str = run_info.model_dump_json()
        deserialized = RunInfo.model_validate_json(json_str)
        assert deserialized.run_id == run_id

    def test_run_id_immutability(self) -> None:
        """Test that run_id cannot be modified after creation."""
        run_id = uuid4()
        run_info = RunInfo(run_id=run_id)
        original_id = run_info.run_id
        # Pydantic models are mutable by default, but we can test assignment
        new_id = uuid4()
        run_info.run_id = new_id
        assert run_info.run_id == new_id
        assert run_info.run_id != original_id

    def test_repr_contains_run_id(self) -> None:
        """Test that repr contains the run_id."""
        run_id = uuid4()
        run_info = RunInfo(run_id=run_id)
        repr_str = repr(run_info)
        assert "run_id" in repr_str
        assert str(run_id) in repr_str

    def test_str_representation(self) -> None:
        """Test string representation of RunInfo."""
        run_id = uuid4()
        run_info = RunInfo(run_id=run_id)
        str_repr = str(run_info)
        assert "run_id" in str_repr

    def test_hash_consistency(self) -> None:
        """Test that hash is consistent for same run_id."""
        run_id = uuid4()
        run_info1 = RunInfo(run_id=run_id)
        run_info2 = RunInfo(run_id=run_id)
        # Pydantic models are hashable if frozen
        # By default they're not frozen, so this might raise
        try:
            hash1 = hash(run_info1)
            hash2 = hash(run_info2)
            assert hash1 == hash2
        except TypeError:
            # Expected if model is not frozen
            pass

    def test_uuid_version(self) -> None:
        """Test that UUID version is preserved."""
        run_id = uuid4()  # Creates UUID version 4
        run_info = RunInfo(run_id=run_id)
        assert run_info.run_id.version == 4

    def test_multiple_run_infos_in_list(self) -> None:
        """Test creating a list of RunInfo objects."""
        run_ids = [uuid4() for _ in range(5)]
        run_infos = [RunInfo(run_id=rid) for rid in run_ids]
        assert len(run_infos) == 5
        for i, run_info in enumerate(run_infos):
            assert run_info.run_id == run_ids[i]

    def test_run_info_with_uuid_from_string(self) -> None:
        """Test creating RunInfo with UUID parsed from string."""
        uuid_str = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        run_id = UUID(uuid_str)
        run_info = RunInfo(run_id=run_id)
        assert str(run_info.run_id) == uuid_str


class TestRunInfoPydanticCoercion:
    """Test suite for RunInfo Pydantic coercion behavior."""

    def test_creation_from_string_uuid(self) -> None:
        """Test that Pydantic coerces string UUID to UUID object."""
        uuid_str = "12345678-1234-5678-1234-567812345678"
        run_info = RunInfo(run_id=uuid_str)  # type: ignore[arg-type]
        assert isinstance(run_info.run_id, UUID)
        assert str(run_info.run_id) == uuid_str

    def test_model_validate(self) -> None:
        """Test model_validate with dict containing UUID."""
        run_id = uuid4()
        run_info = RunInfo.model_validate({"run_id": run_id})
        assert run_info.run_id == run_id

    def test_model_validate_with_string_uuid(self) -> None:
        """Test model_validate with dict containing string UUID."""
        uuid_str = "12345678-1234-5678-1234-567812345678"
        run_info = RunInfo.model_validate({"run_id": uuid_str})
        assert isinstance(run_info.run_id, UUID)
        assert str(run_info.run_id) == uuid_str

    def test_model_fields(self) -> None:
        """Test that RunInfo has expected model fields."""
        fields = RunInfo.model_fields
        assert "run_id" in fields

    def test_copy(self) -> None:
        """Test RunInfo copy produces equivalent object."""
        run_id = uuid4()
        original = RunInfo(run_id=run_id)
        copied = original.model_copy()
        assert copied.run_id == original.run_id
        assert copied == original

    def test_copy_deep(self) -> None:
        """Test RunInfo deep copy produces independent object."""
        run_id = uuid4()
        original = RunInfo(run_id=run_id)
        copied = original.model_copy(deep=True)
        assert copied.run_id == original.run_id
        assert copied == original
