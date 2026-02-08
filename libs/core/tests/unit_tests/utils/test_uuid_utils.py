import time
from uuid import UUID

from langchain_core.utils.uuid import uuid7


def _uuid_v7_ms(uuid_obj: UUID | str) -> int:
    """Extract milliseconds since epoch from a UUIDv7 using string layout.

    UUIDv7 stores Unix time in ms in the first 12 hex chars of the canonical
    string representation (48 msb bits).
    """
    s = str(uuid_obj).replace("-", "")
    return int(s[:12], 16)


def test_uuid7() -> None:
    """Some simple tests."""
    # Note the sequence value increments by 1 between each of these uuid7(...) calls
    ns = time.time_ns()
    ms = ns // 1_000_000
    out1 = str(uuid7(ns))

    # Verify that the timestamp part matches
    out1_ms = _uuid_v7_ms(out1)
    assert out1_ms == ms


def test_monotonicity() -> None:
    """Test that UUIDs are monotonically increasing."""
    last = ""
    for n in range(100_000):
        i = str(uuid7())
        if n > 0 and i <= last:
            msg = f"UUIDs are not monotonic: {last} versus {i}"
            raise RuntimeError(msg)
        last = i


def test_uuid7_returns_uuid_type() -> None:
    """uuid7 returns a UUID object."""
    result = uuid7()
    assert isinstance(result, UUID)


def test_uuid7_version_is_7() -> None:
    """UUID7 should have version 7."""
    result = uuid7()
    assert result.version == 7


def test_uuid7_variant_is_rfc4122() -> None:
    """UUID7 should use the RFC 4122 variant."""
    result = uuid7()
    # RFC 4122 variant has the two most significant bits of clock_seq_hi_and_low
    # set to 10, which means variant is one of the RFC4122 constants
    assert result.variant == uuid_module.RFC_4122


def test_uuid7_with_custom_nanoseconds() -> None:
    """uuid7 with custom nanoseconds encodes the correct timestamp."""
    # 1_000_000_000 ns = 1 second after epoch = 1000 ms
    ns = 1_000_000_000_000  # 1000 seconds = 1_000_000 ms
    result = uuid7(nanoseconds=ns)
    ms = _uuid_v7_ms(result)
    assert ms == ns // 1_000_000


def test_uuid7_with_zero_nanoseconds() -> None:
    """uuid7 at epoch (nanoseconds=0) encodes timestamp 0."""
    result = uuid7(nanoseconds=0)
    ms = _uuid_v7_ms(result)
    assert ms == 0


def test_uuid7_uniqueness() -> None:
    """Multiple uuid7 calls produce unique values."""
    uuids = {uuid7() for _ in range(1000)}
    assert len(uuids) == 1000


def test_uuid7_string_format() -> None:
    """uuid7 string representation has standard UUID format."""
    result = str(uuid7())
    # UUID format: 8-4-4-4-12 hex characters
    parts = result.split("-")
    assert len(parts) == 5
    assert [len(p) for p in parts] == [8, 4, 4, 4, 12]


# Import uuid module for variant check
import uuid as uuid_module
