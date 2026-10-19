"""Synthetic, offline contracts for canonical federated update encodings."""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
import traceback
from itertools import permutations
from pathlib import Path

import pytest

from openmed.training import federated_update_encoding as encoding_module
from openmed.training import federated_update_metadata as metadata_module
from openmed.training.federated_schema_fingerprint import fingerprint_update_schema
from openmed.training.federated_update_encoding import (
    DEFAULT_MAX_ENCODED_BYTES,
    FEDERATED_UPDATE_ENCODING_VERSION,
    FederatedTensorEntry,
    FederatedUpdateAdmission,
    FederatedUpdateEncodingError,
    FederatedUpdateEncodingReason,
    admit_federated_update,
    decode_federated_update,
    digest_federated_update,
    encode_federated_update,
)
from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)

Reason = FederatedUpdateEncodingReason
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL = "sha256:" + "a" * 64
DOMAIN = b"openmed.training.federated_update_encoding.v1\0"

# Synthetic little-endian payloads; values are arbitrary test constants.
LORA_A = struct.pack("<6f", 0.5, -1.0, 0.25, 2.0, -0.125, 1.5)
LORA_B = bytes.fromhex("003c00c0")  # float16 1.0, -2.0
SCALE = bytes.fromhex("803f")  # bfloat16 1.0

GOLDEN_HEX = (
    "4f4d4655010003001500616461707465722e6c6f72615f412e7765696768740302020000"
    "000300000018000000000000000000003f000080bf0000803e00000040000000be0000c0"
    "3f1500616461707465722e6c6f72615f422e7765696768740101020000000400000000"
    "000000003c00c00d00616461707465722e7363616c650201010000000200000000000000"
    "803f"
)
GOLDEN = bytes.fromhex(GOLDEN_HEX)
GOLDEN_DIGEST = (
    "sha256:2bbe770880dc583ae9a0afcacda0865ac9fc34055316fab303977311906b714a"
)

# Byte ranges holding tensor payload bytes inside GOLDEN.
PAYLOAD_RANGES = ((49, 73), (110, 114), (143, 145))

SENTINEL = b"SENTINEL-TENSOR-VALUE"
SENTINEL_NAME = "adapter.sentinel_secret_name"


def _tensors() -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    return {
        "adapter.scale": ("bfloat16", (1,), SCALE),
        "adapter.lora_B.weight": ("float16", (2,), LORA_B),
        "adapter.lora_A.weight": ("float32", (2, 3), LORA_A),
    }


def _entry(
    name: bytes | str,
    code: int,
    dims: tuple[int, ...],
    data: bytes,
    *,
    length: int | None = None,
    name_length: int | None = None,
    rank: int | None = None,
) -> bytes:
    raw = name.encode("ascii") if isinstance(name, str) else name
    return (
        struct.pack("<H", len(raw) if name_length is None else name_length)
        + raw
        + struct.pack("<BB", code, len(dims) if rank is None else rank)
        + b"".join(struct.pack("<I", dim) for dim in dims)
        + struct.pack("<Q", len(data) if length is None else length)
        + data
    )


def _container(
    *entries: bytes, count: int | None = None, version: int = 1, magic=b"OMFU"
) -> bytes:
    return (
        magic
        + struct.pack("<HH", version, len(entries) if count is None else count)
        + b"".join(entries)
    )


A = _entry("adapter.lora_A.weight", 3, (2, 3), LORA_A)
B = _entry("adapter.lora_B.weight", 1, (2,), LORA_B)
S = _entry("adapter.scale", 2, (1,), SCALE)


def _policy(
    *parameters: FederatedParameterMetadata,
) -> FederatedUpdatePolicy:
    if not parameters:
        parameters = (
            FederatedParameterMetadata("adapter.lora_A.weight", (2, 3), "float32"),
            FederatedParameterMetadata("adapter.lora_B.weight", (2,), "float16"),
            FederatedParameterMetadata("adapter.scale", (1,), "bfloat16"),
        )
    return FederatedUpdatePolicy(model_digest=MODEL, parameters=parameters)


def _metadata(
    update_digest: str = GOLDEN_DIGEST, policy: FederatedUpdatePolicy | None = None
) -> FederatedUpdateMetadata:
    policy = policy or _policy()
    return FederatedUpdateMetadata(
        model_digest=MODEL,
        adapter_format="dense",
        parameters=policy.parameters,
        total_elements=sum(p.element_count for p in policy.parameters),
        update_digest=update_digest,
        clipped=True,
        policy=policy,
    )


def _reason(excinfo: pytest.ExceptionInfo[FederatedUpdateEncodingError]) -> Reason:
    return excinfo.value.reason_code


# --- Golden vectors and determinism -------------------------------------------


def test_golden_vector_bytes_and_digest_are_hard_coded():
    encoded = encode_federated_update(_tensors())

    assert encoded == GOLDEN
    assert len(encoded) == 145
    assert digest_federated_update(encoded) == GOLDEN_DIGEST
    # Independent reference: domain-separated SHA-256 over the exact bytes.
    expected = "sha256:" + hashlib.sha256(DOMAIN + GOLDEN).hexdigest()
    assert expected == GOLDEN_DIGEST


def test_hand_built_layout_matches_encoder_output():
    assert _container(A, B, S) == GOLDEN


def test_digest_is_domain_separated_and_versioned():
    assert FEDERATED_UPDATE_ENCODING_VERSION == (
        "openmed.training.federated_update_encoding.v1"
    )
    assert GOLDEN_DIGEST != "sha256:" + hashlib.sha256(GOLDEN).hexdigest()
    schema_domain = b"openmed.training.federated_schema_fingerprint.v1\0"
    assert GOLDEN_DIGEST != (
        "sha256:" + hashlib.sha256(schema_domain + GOLDEN).hexdigest()
    )


def test_identical_tensors_encode_identically_regardless_of_input_form():
    reference = encode_federated_update(_tensors())
    items = list(_tensors().items())
    for order in permutations(items):
        tensors = {}
        for name, (dtype, shape, data) in order:
            tensors[name] = (dtype, list(shape), bytearray(data))
        assert encode_federated_update(tensors) == reference
    as_views = {
        name: (dtype, shape, memoryview(data)) for name, (dtype, shape, data) in items
    }
    assert encode_federated_update(as_views) == reference
    assert encode_federated_update(_tensors()) == encode_federated_update(_tensors())


@pytest.mark.parametrize("hash_seed", ["0", "1", "4242"])
def test_golden_digest_is_stable_across_processes(hash_seed):
    script = (
        "import struct\n"
        "from openmed.training.federated_update_encoding import (\n"
        "    digest_federated_update, encode_federated_update)\n"
        "t = {\n"
        "    'adapter.scale': ('bfloat16', (1,), bytes.fromhex('803f')),\n"
        "    'adapter.lora_B.weight': ('float16', (2,), bytes.fromhex('003c00c0')),\n"
        "    'adapter.lora_A.weight': ('float32', (2, 3),\n"
        "        struct.pack('<6f', 0.5, -1.0, 0.25, 2.0, -0.125, 1.5)),\n"
        "}\n"
        "e = encode_federated_update(t)\n"
        "print(e.hex())\n"
        "print(digest_federated_update(e))\n"
    )
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        env=env,
        text=True,
        timeout=60,
    )
    hex_line, digest_line = completed.stdout.split()
    assert hex_line == GOLDEN_HEX
    assert digest_line == GOLDEN_DIGEST


def test_every_single_byte_change_alters_digest_or_is_rejected():
    payload_offsets = {
        offset for start, end in PAYLOAD_RANGES for offset in range(start, end)
    }
    assert GOLDEN[49:73] == LORA_A
    assert GOLDEN[110:114] == LORA_B
    assert GOLDEN[143:145] == SCALE
    accepted = rejected = 0
    for offset in range(len(GOLDEN)):
        for mask in (0x01, 0x80, 0xFF):
            mutated = bytearray(GOLDEN)
            mutated[offset] ^= mask
            try:
                digest = digest_federated_update(bytes(mutated))
            except FederatedUpdateEncodingError:
                assert offset not in payload_offsets
                rejected += 1
            else:
                assert digest != GOLDEN_DIGEST
                accepted += 1
        if offset in payload_offsets:
            mutated = bytearray(GOLDEN)
            mutated[offset] ^= 0x01
            assert digest_federated_update(bytes(mutated)) != GOLDEN_DIGEST
    assert accepted >= 3 * len(payload_offsets)
    assert rejected > 0


def test_decode_round_trips_golden_vector():
    entries = decode_federated_update(GOLDEN)

    assert [entry.name for entry in entries] == [
        "adapter.lora_A.weight",
        "adapter.lora_B.weight",
        "adapter.scale",
    ]
    assert [entry.dtype for entry in entries] == ["float32", "float16", "bfloat16"]
    assert [entry.shape for entry in entries] == [(2, 3), (2,), (1,)]
    assert [bytes(entry.data) for entry in entries] == [LORA_A, LORA_B, SCALE]
    assert entries[0].metadata == FederatedParameterMetadata(
        "adapter.lora_A.weight", (2, 3), "float32"
    )
    rebuilt = {entry.name: (entry.dtype, entry.shape, entry.data) for entry in entries}
    assert encode_federated_update(rebuilt) == GOLDEN
    assert all(type(entry) is FederatedTensorEntry for entry in entries)


def test_decoded_entries_are_read_only_and_hide_values_in_repr():
    entries = decode_federated_update(bytearray(GOLDEN))
    for entry in entries:
        assert entry.data.readonly
        assert "data" not in repr(entry)
        assert entry.data.hex() not in repr(entry)
    with pytest.raises(TypeError):
        entries[0].data[0] = 0


# --- Admission -----------------------------------------------------------------


def test_admission_accepts_matching_digest_and_schema_fingerprint():
    metadata = _metadata()
    fingerprint = fingerprint_update_schema(metadata)

    admission = admit_federated_update(
        metadata, GOLDEN, expected_schema_fingerprint=fingerprint
    )

    assert type(admission) is FederatedUpdateAdmission
    assert admission.update_digest == GOLDEN_DIGEST
    assert admission.schema_fingerprint == fingerprint
    assert admission.to_dict() == {
        "encoding_version": FEDERATED_UPDATE_ENCODING_VERSION,
        "update_digest": GOLDEN_DIGEST,
        "schema_fingerprint": fingerprint,
        "parameter_count": 3,
        "total_elements": 9,
        "encoded_bytes": 145,
    }
    for name in ("adapter", "lora", "scale"):
        assert name not in repr(admission)
        assert name not in str(admission.to_dict())


def test_admission_receipts_cannot_be_forged_by_direct_construction():
    with pytest.raises(TypeError):
        FederatedUpdateAdmission(  # type: ignore[call-arg]
            encoding_version=FEDERATED_UPDATE_ENCODING_VERSION,
            update_digest=GOLDEN_DIGEST,
            schema_fingerprint=GOLDEN_DIGEST,
            parameter_count=3,
            total_elements=9,
            encoded_bytes=145,
        )


def test_admission_rejects_declared_digest_that_does_not_match_bytes():
    metadata = _metadata(update_digest="sha256:" + "b" * 64)
    fingerprint = fingerprint_update_schema(metadata)

    with pytest.raises(FederatedUpdateEncodingError) as excinfo:
        admit_federated_update(
            metadata, GOLDEN, expected_schema_fingerprint=fingerprint
        )
    assert _reason(excinfo) is Reason.UPDATE_DIGEST_MISMATCH


def test_admission_rejects_payload_changed_after_digest_declaration():
    metadata = _metadata()
    fingerprint = fingerprint_update_schema(metadata)
    mutated = bytearray(GOLDEN)
    mutated[PAYLOAD_RANGES[0][0]] ^= 0x01

    with pytest.raises(FederatedUpdateEncodingError) as excinfo:
        admit_federated_update(
            metadata, bytes(mutated), expected_schema_fingerprint=fingerprint
        )
    assert _reason(excinfo) is Reason.UPDATE_DIGEST_MISMATCH


def test_admission_requires_matching_schema_fingerprint():
    metadata = _metadata()
    other = fingerprint_update_schema(
        _metadata(
            policy=_policy(
                FederatedParameterMetadata("adapter.lora_A.weight", (3, 2), "float32")
            ),
            update_digest=GOLDEN_DIGEST,
        )
    )
    with pytest.raises(FederatedUpdateEncodingError) as excinfo:
        admit_federated_update(metadata, GOLDEN, expected_schema_fingerprint=other)
    assert _reason(excinfo) is Reason.SCHEMA_FINGERPRINT_MISMATCH

    for invalid in ("", "sha256:" + "A" * 64, None, b"sha256:" + b"a" * 64):
        with pytest.raises(FederatedUpdateEncodingError) as excinfo:
            admit_federated_update(
                metadata,
                GOLDEN,
                expected_schema_fingerprint=invalid,  # type: ignore[arg-type]
            )
        assert _reason(excinfo) is Reason.INVALID_ARGUMENT


def test_admission_rejects_non_metadata_and_non_bytes_arguments():
    metadata = _metadata()
    fingerprint = fingerprint_update_schema(metadata)
    with pytest.raises(FederatedUpdateEncodingError) as excinfo:
        admit_federated_update(
            metadata.to_dict(),  # type: ignore[arg-type]
            GOLDEN,
            expected_schema_fingerprint=fingerprint,
        )
    assert _reason(excinfo) is Reason.INVALID_ARGUMENT
    with pytest.raises(FederatedUpdateEncodingError) as excinfo:
        admit_federated_update(
            metadata,
            GOLDEN.hex(),  # type: ignore[arg-type]
            expected_schema_fingerprint=fingerprint,
        )
    assert _reason(excinfo) is Reason.INVALID_ARGUMENT


def test_non_canonical_payload_with_self_consistent_digest_is_still_rejected():
    # A submitter can hash its own non-canonical bytes; the digest alone must
    # never be enough for admission.
    trailing = GOLDEN + b"covert"
    declared = "sha256:" + hashlib.sha256(DOMAIN + trailing).hexdigest()
    metadata = _metadata(update_digest=declared)
    fingerprint = fingerprint_update_schema(metadata)

    with pytest.raises(FederatedUpdateEncodingError) as excinfo:
        admit_federated_update(
            metadata, trailing, expected_schema_fingerprint=fingerprint
        )
    assert _reason(excinfo) is Reason.TRAILING_BYTES


@pytest.mark.parametrize(
    ("encoded", "reason"),
    [
        (
            _container(_entry("adapter.lora_A.weight", 4, (2, 3), LORA_A * 2), B, S),
            Reason.DTYPE_MISMATCH,
        ),
        (
            _container(_entry("adapter.lora_A.weight", 3, (3, 2), LORA_A), B, S),
            Reason.SHAPE_MISMATCH,
        ),
        (_container(A, B), Reason.PARAMETER_MISMATCH),
        (
            _container(A, B, _entry("adapter.zzz_extra", 2, (1,), SCALE), count=3),
            Reason.PARAMETER_MISMATCH,
        ),
        (
            _container(A, B, S, _entry("adapter.zzz_extra", 2, (1,), SCALE)),
            Reason.PARAMETER_MISMATCH,
        ),
        (
            _container(A, B, _entry("adapter.scale", 1, (1,), SCALE)),
            Reason.DTYPE_MISMATCH,
        ),
    ],
)
def test_admission_rejects_canonical_payload_that_differs_from_metadata(
    encoded, reason
):
    declared = "sha256:" + hashlib.sha256(DOMAIN + encoded).hexdigest()
    metadata = _metadata(update_digest=declared)
    fingerprint = fingerprint_update_schema(metadata)
    digest_federated_update(encoded)  # the payload itself is canonical

    with pytest.raises(FederatedUpdateEncodingError) as excinfo:
        admit_federated_update(
            metadata, encoded, expected_schema_fingerprint=fingerprint
        )
    assert _reason(excinfo) is reason


def test_admission_negative_control_accepts_a_different_canonical_update():
    other_values = {
        "adapter.lora_A.weight": ("float32", (2, 3), struct.pack("<6f", *[0.0] * 6)),
        "adapter.lora_B.weight": ("float16", (2,), bytes(4)),
        "adapter.scale": ("bfloat16", (1,), bytes(2)),
    }
    encoded = encode_federated_update(other_values)
    digest = digest_federated_update(encoded)
    assert digest != GOLDEN_DIGEST
    metadata = _metadata(update_digest=digest)

    admission = admit_federated_update(
        metadata,
        encoded,
        expected_schema_fingerprint=fingerprint_update_schema(metadata),
    )
    assert admission.update_digest == digest


# --- Strict decoding: covert channels and non-canonical layouts ----------------

SAFETENSORS_STYLE = struct.pack("<Q", 27) + b'{"__metadata__":{"k":"v"}} ' + GOLDEN


@pytest.mark.parametrize(
    ("encoded", "reason"),
    [
        (b"", Reason.TRUNCATED_ENCODING),
        (b"OMF", Reason.TRUNCATED_ENCODING),
        (SAFETENSORS_STYLE, Reason.INVALID_MAGIC),
        (b"omfu" + GOLDEN[4:], Reason.INVALID_MAGIC),
        (_container(A, B, S, magic=b"OMFX"), Reason.INVALID_MAGIC),
        (_container(A, B, S, version=0), Reason.UNSUPPORTED_VERSION),
        (_container(A, B, S, version=2), Reason.UNSUPPORTED_VERSION),
        (_container(count=0), Reason.INVALID_PARAMETER_COUNT),
        (_container(A, B, S, count=1025), Reason.INVALID_PARAMETER_COUNT),
        (_container(A, B, S, count=4), Reason.TRUNCATED_ENCODING),
        (_container(A, B, S, count=2), Reason.TRAILING_BYTES),
        (GOLDEN + b"\x00", Reason.TRAILING_BYTES),
        (GOLDEN + b'{"note":"x"}', Reason.TRAILING_BYTES),
        (_container(B, A, S), Reason.UNSORTED_PARAMETERS),
        (_container(A, S, B), Reason.UNSORTED_PARAMETERS),
        (_container(A, A, S), Reason.DUPLICATE_PARAMETER),
        (
            _container(A + b"\x00\x00\x00\x00", B, S, count=3),
            Reason.INVALID_PARAMETER_NAME,
        ),
        (
            _container(
                _entry("adapter.lora_A.weight", 3, (2, 3), LORA_A + bytes(4)), B, S
            ),
            Reason.PAYLOAD_LENGTH_MISMATCH,
        ),
        (
            _container(_entry("adapter.lora_A.weight", 3, (2, 3), LORA_A[:-1]), B, S),
            Reason.PAYLOAD_LENGTH_MISMATCH,
        ),
        (
            _container(
                _entry(
                    "adapter.lora_A.weight", 3, (2, 3), LORA_A, length=(1 << 64) - 1
                ),
                B,
                S,
            ),
            Reason.PAYLOAD_LENGTH_MISMATCH,
        ),
        (
            _container(_entry("adapter.lora_A.weight", 0, (2, 3), LORA_A), B, S),
            Reason.UNSUPPORTED_DTYPE,
        ),
        (
            _container(_entry("adapter.lora_A.weight", 5, (2, 3), LORA_A), B, S),
            Reason.UNSUPPORTED_DTYPE,
        ),
        (
            _container(_entry("adapter.lora_A.weight", 255, (2, 3), LORA_A), B, S),
            Reason.UNSUPPORTED_DTYPE,
        ),
        (
            _container(_entry("adapter.lora_A.weight", 3, (), b""), B, S),
            Reason.INVALID_SHAPE,
        ),
        (
            _container(_entry("adapter.lora_A.weight", 3, (1,) * 9, LORA_A[:4]), B, S),
            Reason.INVALID_SHAPE,
        ),
        (
            _container(_entry("adapter.lora_A.weight", 3, (0, 3), b""), B, S),
            Reason.INVALID_SHAPE,
        ),
        (
            _container(_entry("adapter.lora_A.weight", 3, (1 << 31,), b""), B, S),
            Reason.INVALID_SHAPE,
        ),
        (
            _container(
                _entry("adapter.lora_A.weight", 3, ((1 << 31) - 1,) * 8, b""), B, S
            ),
            Reason.INVALID_SHAPE,
        ),
        (
            _container(
                _entry("adapter.lora_A.weight", 3, (1 << 30, 4), b"", length=(1 << 34)),
                B,
                S,
            ),
            Reason.TRUNCATED_ENCODING,
        ),
        (
            _container(_entry(b"", 3, (2, 3), LORA_A), B, S),
            Reason.INVALID_PARAMETER_NAME,
        ),
        (
            _container(_entry("a" * 257, 3, (2, 3), LORA_A)),
            Reason.INVALID_PARAMETER_NAME,
        ),
        (
            _container(_entry("adapter/lora_A", 3, (2, 3), LORA_A)),
            Reason.INVALID_PARAMETER_NAME,
        ),
        (
            _container(_entry("adapter lora", 3, (2, 3), LORA_A)),
            Reason.INVALID_PARAMETER_NAME,
        ),
        (
            _container(_entry(b"adapter.x\xc3\xa9", 3, (2, 3), LORA_A)),
            Reason.INVALID_PARAMETER_NAME,
        ),
        (
            _container(
                _entry("adapter.lora_A.weight", 3, (2, 3), LORA_A, name_length=22), B, S
            ),
            Reason.INVALID_PARAMETER_NAME,
        ),
    ],
)
def test_strict_decoder_rejects_non_canonical_encodings(encoded, reason):
    with pytest.raises(FederatedUpdateEncodingError) as excinfo:
        decode_federated_update(encoded)
    assert _reason(excinfo) is reason
    with pytest.raises(FederatedUpdateEncodingError) as excinfo:
        digest_federated_update(encoded)
    assert _reason(excinfo) is reason


def test_every_strict_prefix_of_golden_is_rejected_as_truncated():
    for length in range(len(GOLDEN)):
        with pytest.raises(FederatedUpdateEncodingError) as excinfo:
            decode_federated_update(GOLDEN[:length])
        assert _reason(excinfo) is Reason.TRUNCATED_ENCODING, length


def test_decoder_bounds_total_encoded_size():
    with pytest.raises(FederatedUpdateEncodingError) as excinfo:
        decode_federated_update(GOLDEN, max_encoded_bytes=len(GOLDEN) - 1)
    assert _reason(excinfo) is Reason.ENCODED_SIZE_EXCEEDED
    assert len(decode_federated_update(GOLDEN, max_encoded_bytes=len(GOLDEN))) == 3
    for invalid in (0, -1, True, 1.5, None):
        with pytest.raises(FederatedUpdateEncodingError) as excinfo:
            decode_federated_update(GOLDEN, max_encoded_bytes=invalid)
        assert _reason(excinfo) is Reason.INVALID_ARGUMENT


@pytest.mark.parametrize("encoded", [None, "OMFU", 145, [0x4F], GOLDEN.hex()])
def test_decoder_rejects_non_bytes_input(encoded):
    with pytest.raises(FederatedUpdateEncodingError) as excinfo:
        decode_federated_update(encoded)
    assert _reason(excinfo) is Reason.INVALID_ARGUMENT


def test_decoder_snapshots_mutable_buffers():
    buffer = bytearray(GOLDEN)
    entries = decode_federated_update(buffer)
    buffer[PAYLOAD_RANGES[0][0]] ^= 0xFF
    assert bytes(entries[0].data) == LORA_A


# --- Encoder validation ---------------------------------------------------------


@pytest.mark.parametrize(
    ("tensors", "reason"),
    [
        ({}, Reason.INVALID_PARAMETER_COUNT),
        ([("adapter.a", ("float32", (1,), bytes(4)))], Reason.INVALID_ARGUMENT),
        ({1: ("float32", (1,), bytes(4))}, Reason.INVALID_PARAMETER_NAME),
        ({"adapter/a": ("float32", (1,), bytes(4))}, Reason.INVALID_PARAMETER_NAME),
        ({"a" * 257: ("float32", (1,), bytes(4))}, Reason.INVALID_PARAMETER_NAME),
        ({"adapter.a": ("int8", (1,), bytes(1))}, Reason.UNSUPPORTED_DTYPE),
        ({"adapter.a": ("float32", (), b"")}, Reason.INVALID_SHAPE),
        ({"adapter.a": ("float32", (True,), bytes(4))}, Reason.INVALID_SHAPE),
        ({"adapter.a": ("float32", (1.0,), bytes(4))}, Reason.INVALID_SHAPE),
        ({"adapter.a": ("float32", "1", bytes(4))}, Reason.INVALID_SHAPE),
        ({"adapter.a": ("float32", (0,), b"")}, Reason.INVALID_SHAPE),
        ({"adapter.a": ("float32", (1,) * 9, bytes(4))}, Reason.INVALID_SHAPE),
        ({"adapter.a": ("float32", (2,), bytes(4))}, Reason.PAYLOAD_LENGTH_MISMATCH),
        ({"adapter.a": ("float32", (1,), bytes(5))}, Reason.PAYLOAD_LENGTH_MISMATCH),
        ({"adapter.a": ("float32", (1,), "abcd")}, Reason.INVALID_ARGUMENT),
        ({"adapter.a": ("float32", (1,), [0, 0, 0, 0])}, Reason.INVALID_ARGUMENT),
        ({"adapter.a": ("float32", (1,))}, Reason.INVALID_ARGUMENT),
        ({"adapter.a": ["float32", (1,), bytes(4)]}, Reason.INVALID_ARGUMENT),
    ],
)
def test_encoder_rejects_invalid_tensors(tensors, reason):
    with pytest.raises(FederatedUpdateEncodingError) as excinfo:
        encode_federated_update(tensors)
    assert _reason(excinfo) is reason


def test_encoder_bounds_parameter_count_and_encoded_size():
    too_many = {f"adapter.p{i:04d}": ("float16", (1,), bytes(2)) for i in range(1025)}
    with pytest.raises(FederatedUpdateEncodingError) as excinfo:
        encode_federated_update(too_many)
    assert _reason(excinfo) is Reason.INVALID_PARAMETER_COUNT

    limit = dict(list(too_many.items())[:1024])
    encoded = encode_federated_update(limit)
    assert len(decode_federated_update(encoded)) == 1024

    with pytest.raises(FederatedUpdateEncodingError) as excinfo:
        encode_federated_update(_tensors(), max_encoded_bytes=len(GOLDEN) - 1)
    assert _reason(excinfo) is Reason.ENCODED_SIZE_EXCEEDED
    assert encode_federated_update(_tensors(), max_encoded_bytes=len(GOLDEN)) == GOLDEN


def test_bounds_match_federated_update_metadata_limits():
    assert encoding_module._MAX_PARAMETERS == metadata_module._MAX_PARAMETERS
    assert encoding_module._MAX_RANK == metadata_module._MAX_RANK
    assert encoding_module._MAX_DIMENSION == metadata_module._MAX_DIMENSION
    assert encoding_module._MAX_ELEMENTS == metadata_module._MAX_ELEMENTS
    assert encoding_module._MAX_NAME_LENGTH == 256
    assert set(encoding_module._DTYPE_CODES) == set(metadata_module._DTYPES)
    assert DEFAULT_MAX_ENCODED_BYTES == 1 << 30


def test_reason_codes_are_stable():
    assert [reason.value for reason in Reason] == [
        "invalid_argument",
        "encoded_size_exceeded",
        "truncated_encoding",
        "invalid_magic",
        "unsupported_version",
        "invalid_parameter_count",
        "invalid_parameter_name",
        "unsorted_parameters",
        "duplicate_parameter",
        "unsupported_dtype",
        "invalid_shape",
        "payload_length_mismatch",
        "trailing_bytes",
        "schema_fingerprint_mismatch",
        "parameter_mismatch",
        "dtype_mismatch",
        "shape_mismatch",
        "update_digest_mismatch",
    ]
    assert str(Reason.TRAILING_BYTES) == "trailing_bytes"


# --- Privacy of diagnostics -----------------------------------------------------


def _rendered(error: BaseException) -> str:
    return (
        "".join(traceback.format_exception_only(type(error), error))
        + repr(error)
        + repr(error.args)
    )


def _sentinel_failures():
    sentinel_payload = SENTINEL + bytes(4 * 8 - len(SENTINEL))
    good = _entry(SENTINEL_NAME, 3, (8,), sentinel_payload)
    sentinel_policy = _policy(
        FederatedParameterMetadata(SENTINEL_NAME, (8,), "float32")
    )
    canonical = _container(good)
    declared = digest_federated_update(canonical)
    yield lambda: decode_federated_update(canonical + SENTINEL)
    yield lambda: decode_federated_update(_container(good, good))
    yield lambda: decode_federated_update(
        _container(_entry(SENTINEL_NAME, 3, (8,), sentinel_payload + SENTINEL))
    )
    yield lambda: decode_federated_update(
        _container(_entry(SENTINEL_NAME, 9, (8,), sentinel_payload))
    )
    yield lambda: decode_federated_update(
        _container(_entry(SENTINEL_NAME + "/x", 3, (8,), sentinel_payload))
    )
    yield lambda: decode_federated_update(SENTINEL + canonical)
    yield lambda: encode_federated_update(
        {SENTINEL_NAME: ("float32", (9,), sentinel_payload)}
    )
    yield lambda: encode_federated_update(
        {SENTINEL_NAME + "/x": ("float32", (8,), sentinel_payload)}
    )
    yield lambda: encode_federated_update(
        {SENTINEL_NAME: ("sentinel_dtype", (8,), sentinel_payload)}
    )
    wrong_digest = _metadata(update_digest="sha256:" + "c" * 64, policy=sentinel_policy)
    yield lambda: admit_federated_update(
        wrong_digest,
        canonical,
        expected_schema_fingerprint=fingerprint_update_schema(wrong_digest),
    )
    declared_metadata = _metadata(update_digest=declared, policy=sentinel_policy)
    yield lambda: admit_federated_update(
        declared_metadata,
        canonical,
        expected_schema_fingerprint="sha256:" + "d" * 64,
    )
    yield lambda: admit_federated_update(
        declared_metadata,
        _container(_entry(SENTINEL_NAME, 4, (8,), sentinel_payload * 2)),
        expected_schema_fingerprint=fingerprint_update_schema(declared_metadata),
    )


def test_errors_never_contain_tensor_values_names_digests_or_paths():
    failures = list(_sentinel_failures())
    assert len(failures) == 12
    forbidden = (
        SENTINEL.decode("ascii"),
        SENTINEL.hex(),
        "sentinel",
        SENTINEL_NAME,
        "sha256:",
        str(REPO_ROOT),
        "/x",
    )
    for failure in failures:
        with pytest.raises(FederatedUpdateEncodingError) as excinfo:
            failure()
        error = excinfo.value
        assert error.__cause__ is None
        assert error.__suppress_context__ is True
        message = str(error)
        assert message == f"federated update encoding rejected: {error.reason_code}"
        rendered = _rendered(error)
        for token in forbidden:
            assert token not in rendered


def test_successful_admission_report_contains_no_names_or_values():
    sentinel_payload = SENTINEL + bytes(4 * 8 - len(SENTINEL))
    canonical = encode_federated_update(
        {SENTINEL_NAME: ("float32", (8,), sentinel_payload)}
    )
    policy = _policy(FederatedParameterMetadata(SENTINEL_NAME, (8,), "float32"))
    metadata = _metadata(
        update_digest=digest_federated_update(canonical), policy=policy
    )
    admission = admit_federated_update(
        metadata,
        canonical,
        expected_schema_fingerprint=fingerprint_update_schema(metadata),
    )
    rendered = repr(admission) + repr(admission.to_dict())
    assert "sentinel" not in rendered.lower()
    assert SENTINEL.hex() not in rendered
    assert SENTINEL_NAME not in rendered


def test_module_performs_no_io_or_logging():
    source = Path(encoding_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("import logging", "open(", "socket", "urllib", "print("):
        assert forbidden not in source


def test_end_to_end_client_submission_is_admitted_from_json_boundary():
    # Client: encode tensors, declare the digest in the metadata envelope.
    encoded = encode_federated_update(_tensors())
    policy = _policy()
    envelope = _metadata(update_digest=digest_federated_update(encoded)).to_json()

    # Coordinator: parse with its own policy and its own expected fingerprint.
    metadata = FederatedUpdateMetadata.from_json(envelope, policy=policy)
    expected = fingerprint_update_schema(_metadata(policy=policy))
    admission = admit_federated_update(
        metadata, encoded, expected_schema_fingerprint=expected
    )
    assert admission.update_digest == metadata.update_digest == GOLDEN_DIGEST

    # The same envelope cannot be replayed with different bytes.
    swapped = encode_federated_update(
        {**_tensors(), "adapter.scale": ("bfloat16", (1,), bytes(2))}
    )
    with pytest.raises(FederatedUpdateEncodingError) as excinfo:
        admit_federated_update(metadata, swapped, expected_schema_fingerprint=expected)
    assert _reason(excinfo) is Reason.UPDATE_DIGEST_MISMATCH
