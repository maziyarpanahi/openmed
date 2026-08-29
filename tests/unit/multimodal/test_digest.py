from __future__ import annotations

import hashlib
import io

import pytest

from openmed.multimodal.digest import (
    DigestLimitExceededError,
    DigestStreamError,
    digest_asset,
)


class NonSeekableStream:
    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)
        self.closed = False
        self.read_sizes: list[int] = []

    def seekable(self) -> bool:
        return False

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self._stream.read(size)


@pytest.mark.parametrize("payload", [b"", b"openmed"])
def test_digest_bytes_matches_hashlib(payload: bytes) -> None:
    result = digest_asset(payload)

    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert result.byte_count == len(payload)


def test_digest_seekable_stream_restores_position_without_closing() -> None:
    payload = b"prefix-payload"
    stream = io.BytesIO(payload)
    stream.seek(len(b"prefix-"))

    result = digest_asset(stream)

    assert result.sha256 == hashlib.sha256(b"payload").hexdigest()
    assert result.byte_count == len(b"payload")
    assert stream.tell() == len(b"prefix-")
    assert not stream.closed


def test_digest_non_seekable_multichunk_stream_uses_bounded_reads() -> None:
    payload = b"x" * (2 * 1024 * 1024 + 7)
    stream = NonSeekableStream(payload)

    result = digest_asset(stream)

    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert result.byte_count == len(payload)
    assert max(stream.read_sizes) == 1024 * 1024
    assert len(stream.read_sizes) == 4
    assert not stream.closed


def test_digest_limit_reads_only_one_byte_past_limit_and_restores_position() -> None:
    sensitive = b"patient-name-at-/private/scan.dcm"
    stream = io.BytesIO(sensitive)
    stream.seek(3)

    with pytest.raises(DigestLimitExceededError) as raised:
        digest_asset(stream, max_bytes=5)

    assert raised.value.maximum_bytes == 5
    assert raised.value.bytes_read == 6
    assert str(raised.value) == ("digest_size_limit: maximum_bytes=5, bytes_read=6")
    assert "patient" not in str(raised.value)
    assert "scan.dcm" not in str(raised.value)
    assert stream.tell() == 3
    assert not stream.closed


def test_digest_stream_error_does_not_expose_underlying_details() -> None:
    class BrokenStream(NonSeekableStream):
        def read(self, size: int) -> bytes:
            raise OSError("/private/patient-name.dcm contains secret bytes")

    with pytest.raises(DigestStreamError) as raised:
        digest_asset(BrokenStream(b""))

    assert str(raised.value) == "digest_stream_read_error"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.parametrize("max_bytes", [-1, True, 1.5])
def test_digest_rejects_invalid_limits(max_bytes: object) -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        digest_asset(b"", max_bytes=max_bytes)  # type: ignore[arg-type]
