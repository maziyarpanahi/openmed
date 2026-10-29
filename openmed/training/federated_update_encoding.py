"""Canonical byte encoding and digest binding for dense federated updates.

A federated update's declared ``update_digest`` identifies exact payload bytes
only when every submitter serializes tensors the same way. This module defines
that single plaintext layout, a strict decoder that rejects every other layout,
and an admission check that binds the declared digest and the validated schema
fingerprint to the submitted bytes before aggregation.

The encoding is versioned and contains no free-form metadata. All integers are
little-endian and unsigned::

    header  := magic "OMFU" | u16 version (1) | u16 parameter_count
    entry   := u16 name_length | name (ASCII) | u8 dtype_code | u8 rank
               | rank * u32 dimension | u64 payload_length | payload
    encoded := header | parameter_count * entry

Entries are strictly ascending by name, payloads are contiguous little-endian
element bytes with no padding, and nothing may follow the final entry.
Tensor bytes are treated as opaque; no tensor library is required.

Diagnostics contain only stable reason codes. They never include tensor bytes,
parameter names, declared digests, or paths. The module performs no logging,
filesystem access, or network calls.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import struct
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Any, Final, NoReturn

from .federated_schema_fingerprint import fingerprint_update_schema
from .federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdateMetadataError,
)

FEDERATED_UPDATE_ENCODING_VERSION = "openmed.training.federated_update_encoding.v1"
DEFAULT_MAX_ENCODED_BYTES = 1 << 30

_DOMAIN: Final = FEDERATED_UPDATE_ENCODING_VERSION.encode("ascii") + b"\0"
_MAGIC: Final = b"OMFU"
_FORMAT_VERSION: Final = 1
_HEADER: Final = struct.Struct("<4sHH")
_NAME_LENGTH: Final = struct.Struct("<H")
_DTYPE_RANK: Final = struct.Struct("<BB")
_DIMENSION: Final = struct.Struct("<I")
_PAYLOAD_LENGTH: Final = struct.Struct("<Q")

_MAX_PARAMETERS: Final = 1024
_MAX_NAME_LENGTH: Final = 256
_MAX_RANK: Final = 8
_MAX_DIMENSION: Final = (1 << 31) - 1
_MAX_ELEMENTS: Final = (1 << 63) - 1
_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")

# Stable on-wire codes; never renumber. Values are (code, item size in bytes).
_DTYPE_CODES: Final = {
    "float16": (1, 2),
    "bfloat16": (2, 2),
    "float32": (3, 4),
    "float64": (4, 8),
}
_DTYPES_BY_CODE: Final = {
    code: (name, size) for name, (code, size) in _DTYPE_CODES.items()
}
_ADMISSION_TOKEN: Final = object()

_BytesLike = bytes | bytearray | memoryview


class FederatedUpdateEncodingReason(str, Enum):
    """Stable, value-free reasons for rejecting an update encoding."""

    INVALID_ARGUMENT = "invalid_argument"
    ENCODED_SIZE_EXCEEDED = "encoded_size_exceeded"
    TRUNCATED_ENCODING = "truncated_encoding"
    INVALID_MAGIC = "invalid_magic"
    UNSUPPORTED_VERSION = "unsupported_version"
    INVALID_PARAMETER_COUNT = "invalid_parameter_count"
    INVALID_PARAMETER_NAME = "invalid_parameter_name"
    UNSORTED_PARAMETERS = "unsorted_parameters"
    DUPLICATE_PARAMETER = "duplicate_parameter"
    UNSUPPORTED_DTYPE = "unsupported_dtype"
    INVALID_SHAPE = "invalid_shape"
    PAYLOAD_LENGTH_MISMATCH = "payload_length_mismatch"
    TRAILING_BYTES = "trailing_bytes"
    SCHEMA_FINGERPRINT_MISMATCH = "schema_fingerprint_mismatch"
    PARAMETER_MISMATCH = "parameter_mismatch"
    DTYPE_MISMATCH = "dtype_mismatch"
    SHAPE_MISMATCH = "shape_mismatch"
    UPDATE_DIGEST_MISMATCH = "update_digest_mismatch"

    def __str__(self) -> str:
        return self.value


class FederatedUpdateEncodingError(ValueError):
    """Raised for a rejected update encoding with only a stable reason code.

    Args:
        reason_code: The categorical rejection reason. The message is derived
            from it alone and never includes submitted bytes, names, or digests.
    """

    def __init__(self, reason_code: FederatedUpdateEncodingReason) -> None:
        self.reason_code = FederatedUpdateEncodingReason(reason_code)
        super().__init__(f"federated update encoding rejected: {self.reason_code}")


@dataclass(frozen=True, slots=True)
class FederatedTensorEntry:
    """One decoded parameter with read-only access to its payload bytes.

    Args:
        name: Validated dotted ASCII parameter name.
        dtype: One of float16, bfloat16, float32, or float64.
        shape: Positive dimensions in declared order.
        data: Read-only little-endian element bytes; excluded from ``repr``.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    data: memoryview = field(repr=False)

    @property
    def metadata(self) -> FederatedParameterMetadata:
        """Return the parameter metadata described by this entry."""
        return FederatedParameterMetadata(self.name, self.shape, self.dtype)


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedUpdateAdmission:
    """Value-free receipt for an update whose bytes match its declarations.

    Instances are produced only by :func:`admit_federated_update`.

    Args:
        encoding_version: Canonical encoding identifier that was verified.
        update_digest: Verified domain-separated digest of the encoded bytes.
        schema_fingerprint: Verified schema fingerprint of the metadata.
        parameter_count: Number of encoded parameters.
        total_elements: Total element count across parameters.
        encoded_bytes: Exact encoded size in bytes.
    """

    encoding_version: str
    update_digest: str
    schema_fingerprint: str
    parameter_count: int
    total_elements: int
    encoded_bytes: int
    _token: InitVar[object]

    def __post_init__(self, _token: object) -> None:
        if _token is not _ADMISSION_TOKEN:
            raise TypeError("admission receipts are issued by admit_federated_update")

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mapping containing only codes, counts, and digests."""
        return {
            "encoding_version": self.encoding_version,
            "update_digest": self.update_digest,
            "schema_fingerprint": self.schema_fingerprint,
            "parameter_count": self.parameter_count,
            "total_elements": self.total_elements,
            "encoded_bytes": self.encoded_bytes,
        }


def encode_federated_update(
    tensors: Mapping[str, tuple[str, tuple[int, ...] | list[int], _BytesLike]],
    *,
    max_encoded_bytes: int = DEFAULT_MAX_ENCODED_BYTES,
) -> bytes:
    """Encode named dense tensors into the canonical version 1 layout.

    Args:
        tensors: Mapping from parameter name to a ``(dtype, shape, data)``
            tuple. ``data`` holds exactly ``prod(shape) * itemsize``
            little-endian element bytes. Mapping order does not matter.
        max_encoded_bytes: Positive upper bound for the encoded size.

    Returns:
        Canonical bytes; identical tensors always produce identical bytes.

    Raises:
        FederatedUpdateEncodingError: If any name, dtype, shape, payload, or
            bound is invalid. The error carries only a reason code.
    """
    limit = _require_limit(max_encoded_bytes)
    if not isinstance(tensors, Mapping):
        _reject(FederatedUpdateEncodingReason.INVALID_ARGUMENT)
    if not 1 <= len(tensors) <= _MAX_PARAMETERS:
        _reject(FederatedUpdateEncodingReason.INVALID_PARAMETER_COUNT)
    records = []
    for name, value in tensors.items():
        _require_name(name)
        if type(value) is not tuple or len(value) != 3:
            _reject(FederatedUpdateEncodingReason.INVALID_ARGUMENT)
        dtype, shape, data = value
        if type(dtype) is not str or dtype not in _DTYPE_CODES:
            _reject(FederatedUpdateEncodingReason.UNSUPPORTED_DTYPE)
        if type(shape) not in (tuple, list):
            _reject(FederatedUpdateEncodingReason.INVALID_SHAPE)
        shape = tuple(shape)
        if not 1 <= len(shape) <= _MAX_RANK:
            _reject(FederatedUpdateEncodingReason.INVALID_SHAPE)
        elements = _elements(shape)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            _reject(FederatedUpdateEncodingReason.INVALID_ARGUMENT)
        payload = data if type(data) is bytes else bytes(data)
        code, size = _DTYPE_CODES[dtype]
        if len(payload) != elements * size:
            _reject(FederatedUpdateEncodingReason.PAYLOAD_LENGTH_MISMATCH)
        records.append((name, code, shape, payload))
    records.sort(key=lambda record: record[0])
    parts = [_HEADER.pack(_MAGIC, _FORMAT_VERSION, len(records))]
    total = _HEADER.size
    for name, code, shape, payload in records:
        raw = name.encode("ascii")
        total += (
            _NAME_LENGTH.size
            + len(raw)
            + _DTYPE_RANK.size
            + _DIMENSION.size * len(shape)
            + _PAYLOAD_LENGTH.size
            + len(payload)
        )
        if total > limit:
            _reject(FederatedUpdateEncodingReason.ENCODED_SIZE_EXCEEDED)
        parts.append(_NAME_LENGTH.pack(len(raw)))
        parts.append(raw)
        parts.append(_DTYPE_RANK.pack(code, len(shape)))
        parts.extend(_DIMENSION.pack(dimension) for dimension in shape)
        parts.append(_PAYLOAD_LENGTH.pack(len(payload)))
        parts.append(payload)
    return b"".join(parts)


def decode_federated_update(
    encoded: _BytesLike, *, max_encoded_bytes: int = DEFAULT_MAX_ENCODED_BYTES
) -> tuple[FederatedTensorEntry, ...]:
    """Strictly decode canonical bytes, rejecting every non-canonical layout.

    Declared lengths are checked against the remaining buffer before any
    payload is sliced, so oversized headers cannot trigger large allocations.
    Mutable buffers are copied once so later caller writes cannot change the
    decoded entries.

    Args:
        encoded: Encoded update as ``bytes``, ``bytearray``, or ``memoryview``.
        max_encoded_bytes: Positive upper bound for the encoded size.

    Returns:
        Entries in canonical name order with read-only payload views.

    Raises:
        FederatedUpdateEncodingError: On wrong magic or version, invalid
            counts, names, dtypes, or shapes, unsorted or duplicate entries,
            payload length mismatches, truncation, or trailing bytes.
    """
    view = memoryview(_snapshot(encoded, _require_limit(max_encoded_bytes)))
    if len(view) < _HEADER.size:
        _reject(FederatedUpdateEncodingReason.TRUNCATED_ENCODING)
    magic, version, count = _HEADER.unpack_from(view, 0)
    if magic != _MAGIC:
        _reject(FederatedUpdateEncodingReason.INVALID_MAGIC)
    if version != _FORMAT_VERSION:
        _reject(FederatedUpdateEncodingReason.UNSUPPORTED_VERSION)
    if not 1 <= count <= _MAX_PARAMETERS:
        _reject(FederatedUpdateEncodingReason.INVALID_PARAMETER_COUNT)
    offset = _HEADER.size
    entries: list[FederatedTensorEntry] = []
    previous = None
    for _ in range(count):
        (name_length,) = _read(_NAME_LENGTH, view, offset)
        offset += _NAME_LENGTH.size
        if not 1 <= name_length <= _MAX_NAME_LENGTH:
            _reject(FederatedUpdateEncodingReason.INVALID_PARAMETER_NAME)
        raw_name = _take(view, offset, name_length)
        offset += name_length
        try:
            name = raw_name.decode("ascii")
        except UnicodeDecodeError:
            _reject(FederatedUpdateEncodingReason.INVALID_PARAMETER_NAME)
        _require_name(name)
        if previous is not None:
            if name == previous:
                _reject(FederatedUpdateEncodingReason.DUPLICATE_PARAMETER)
            if name < previous:
                _reject(FederatedUpdateEncodingReason.UNSORTED_PARAMETERS)
        previous = name
        code, rank = _read(_DTYPE_RANK, view, offset)
        offset += _DTYPE_RANK.size
        if code not in _DTYPES_BY_CODE:
            _reject(FederatedUpdateEncodingReason.UNSUPPORTED_DTYPE)
        if not 1 <= rank <= _MAX_RANK:
            _reject(FederatedUpdateEncodingReason.INVALID_SHAPE)
        shape = []
        for _dimension in range(rank):
            shape.append(_read(_DIMENSION, view, offset)[0])
            offset += _DIMENSION.size
        shape_tuple = tuple(shape)
        elements = _elements(shape_tuple)
        (payload_length,) = _read(_PAYLOAD_LENGTH, view, offset)
        offset += _PAYLOAD_LENGTH.size
        dtype, size = _DTYPES_BY_CODE[code]
        if payload_length != elements * size:
            _reject(FederatedUpdateEncodingReason.PAYLOAD_LENGTH_MISMATCH)
        if payload_length > len(view) - offset:
            _reject(FederatedUpdateEncodingReason.TRUNCATED_ENCODING)
        data = view[offset : offset + payload_length]
        offset += payload_length
        entries.append(FederatedTensorEntry(name, dtype, shape_tuple, data))
    if offset != len(view):
        _reject(FederatedUpdateEncodingReason.TRAILING_BYTES)
    return tuple(entries)


def digest_federated_update(
    encoded: _BytesLike, *, max_encoded_bytes: int = DEFAULT_MAX_ENCODED_BYTES
) -> str:
    """Return the domain-separated SHA-256 digest of a canonical encoding.

    The digest is ``sha256(domain || encoded)`` where ``domain`` is the ASCII
    encoding version followed by a NUL byte. Non-canonical input is rejected
    instead of hashed.

    Args:
        encoded: Encoded update bytes.
        max_encoded_bytes: Positive upper bound for the encoded size.

    Returns:
        ``sha256:`` followed by 64 lowercase hexadecimal digits.

    Raises:
        FederatedUpdateEncodingError: If the encoding is not canonical.
    """
    snapshot = _snapshot(encoded, _require_limit(max_encoded_bytes))
    decode_federated_update(snapshot, max_encoded_bytes=max_encoded_bytes)
    return _digest(snapshot)


def admit_federated_update(
    metadata: FederatedUpdateMetadata,
    encoded: _BytesLike,
    *,
    expected_schema_fingerprint: str,
    max_encoded_bytes: int = DEFAULT_MAX_ENCODED_BYTES,
) -> FederatedUpdateAdmission:
    """Bind validated metadata to its canonical payload bytes before aggregation.

    Checks run in a fixed order so reason codes are stable: argument types,
    schema fingerprint, canonical decoding, parameter names, dtypes, shapes,
    and finally the declared ``update_digest``. Digest and fingerprint
    comparisons use a constant-time primitive.

    Args:
        metadata: Policy-validated update metadata for this submission.
        encoded: Submitted canonical update bytes.
        expected_schema_fingerprint: Coordinator-owned schema fingerprint from
            :func:`fingerprint_update_schema`; never taken from the submission.
        max_encoded_bytes: Positive upper bound for the encoded size.

    Returns:
        A value-free admission receipt with digests and counts only.

    Raises:
        FederatedUpdateEncodingError: If any binding check fails.
    """
    if type(metadata) is not FederatedUpdateMetadata:
        _reject(FederatedUpdateEncodingReason.INVALID_ARGUMENT)
    if (
        type(expected_schema_fingerprint) is not str
        or _DIGEST.fullmatch(expected_schema_fingerprint) is None
    ):
        _reject(FederatedUpdateEncodingReason.INVALID_ARGUMENT)
    fingerprint = fingerprint_update_schema(metadata)
    if not hmac.compare_digest(
        fingerprint.encode("ascii"), expected_schema_fingerprint.encode("ascii")
    ):
        _reject(FederatedUpdateEncodingReason.SCHEMA_FINGERPRINT_MISMATCH)
    snapshot = _snapshot(encoded, _require_limit(max_encoded_bytes))
    entries = decode_federated_update(snapshot, max_encoded_bytes=max_encoded_bytes)
    declared = metadata.parameters
    if len(entries) != len(declared) or any(
        entry.name != parameter.name
        for entry, parameter in zip(entries, declared, strict=True)
    ):
        _reject(FederatedUpdateEncodingReason.PARAMETER_MISMATCH)
    for entry, parameter in zip(entries, declared, strict=True):
        if entry.dtype != parameter.dtype:
            _reject(FederatedUpdateEncodingReason.DTYPE_MISMATCH)
        if entry.shape != parameter.shape:
            _reject(FederatedUpdateEncodingReason.SHAPE_MISMATCH)
    digest = _digest(snapshot)
    if not hmac.compare_digest(
        digest.encode("ascii"), metadata.update_digest.encode("ascii")
    ):
        _reject(FederatedUpdateEncodingReason.UPDATE_DIGEST_MISMATCH)
    return FederatedUpdateAdmission(
        encoding_version=FEDERATED_UPDATE_ENCODING_VERSION,
        update_digest=digest,
        schema_fingerprint=fingerprint,
        parameter_count=len(entries),
        total_elements=metadata.total_elements,
        encoded_bytes=len(snapshot),
        _token=_ADMISSION_TOKEN,
    )


def _reject(reason: FederatedUpdateEncodingReason) -> NoReturn:
    raise FederatedUpdateEncodingError(reason) from None


def _require_limit(value: object) -> int:
    if type(value) is not int or value < 1:
        _reject(FederatedUpdateEncodingReason.INVALID_ARGUMENT)
    return value


def _snapshot(encoded: object, limit: int) -> bytes:
    if not isinstance(encoded, (bytes, bytearray, memoryview)):
        _reject(FederatedUpdateEncodingReason.INVALID_ARGUMENT)
    size = encoded.nbytes if isinstance(encoded, memoryview) else len(encoded)
    if size > limit:
        _reject(FederatedUpdateEncodingReason.ENCODED_SIZE_EXCEEDED)
    return encoded if type(encoded) is bytes else bytes(encoded)


def _require_name(name: object) -> None:
    if type(name) is not str or not 1 <= len(name) <= _MAX_NAME_LENGTH:
        _reject(FederatedUpdateEncodingReason.INVALID_PARAMETER_NAME)
    try:
        FederatedParameterMetadata(name, (1,), "float32")
    except FederatedUpdateMetadataError:
        _reject(FederatedUpdateEncodingReason.INVALID_PARAMETER_NAME)


def _elements(shape: tuple[object, ...]) -> int:
    product = 1
    for dimension in shape:
        if type(dimension) is not int or not 1 <= dimension <= _MAX_DIMENSION:
            _reject(FederatedUpdateEncodingReason.INVALID_SHAPE)
        if product > _MAX_ELEMENTS // dimension:
            _reject(FederatedUpdateEncodingReason.INVALID_SHAPE)
        product *= dimension
    return product


def _read(layout: struct.Struct, view: memoryview, offset: int) -> tuple[int, ...]:
    if layout.size > len(view) - offset:
        _reject(FederatedUpdateEncodingReason.TRUNCATED_ENCODING)
    return layout.unpack_from(view, offset)


def _take(view: memoryview, offset: int, length: int) -> bytes:
    if length > len(view) - offset:
        _reject(FederatedUpdateEncodingReason.TRUNCATED_ENCODING)
    return bytes(view[offset : offset + length])


def _digest(encoded: bytes) -> str:
    hasher = hashlib.sha256(_DOMAIN)
    hasher.update(encoded)
    return "sha256:" + hasher.hexdigest()


__all__ = [
    "DEFAULT_MAX_ENCODED_BYTES",
    "FEDERATED_UPDATE_ENCODING_VERSION",
    "FederatedTensorEntry",
    "FederatedUpdateAdmission",
    "FederatedUpdateEncodingError",
    "FederatedUpdateEncodingReason",
    "admit_federated_update",
    "decode_federated_update",
    "digest_federated_update",
    "encode_federated_update",
]
