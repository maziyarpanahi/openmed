"""Tests for content-free multimodal provider result envelopes."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError, replace

import pytest

from openmed.multimodal.provider_result import (
    MAX_PROVIDER_RESULT_JSON_BYTES,
    PROVIDER_RESULT_SCHEMA_VERSION,
    ProviderAbstentionCode,
    ProviderResultEnvelope,
    ProviderResultError,
    ProviderResultOutcome,
)

INPUT_DIGEST = "1" * 64
OUTPUT_DIGEST = "2" * 64


def _result(**changes: object) -> ProviderResultEnvelope:
    values: dict[str, object] = {
        "provider_id": "doctr-1",
        "model_id": "ocr-base-v2",
        "input_digest": INPUT_DIGEST,
        "output_digest": OUTPUT_DIGEST,
        "outcome": ProviderResultOutcome.SUCCESS,
        "duration_ms": 12.5,
        "count_metadata": {"page_count": 2, "token_count": 37},
    }
    values.update(changes)
    return ProviderResultEnvelope(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("outcome", "output_digest", "abstention_code"),
    [
        (ProviderResultOutcome.SUCCESS, OUTPUT_DIGEST, None),
        (
            ProviderResultOutcome.ABSTENTION,
            None,
            ProviderAbstentionCode.LOW_QUALITY,
        ),
        (ProviderResultOutcome.PROVIDER_UNAVAILABLE, None, None),
        (ProviderResultOutcome.VALIDATION_FAILURE, None, None),
    ],
)
def test_all_outcomes_round_trip_deterministically(
    outcome: ProviderResultOutcome,
    output_digest: str | None,
    abstention_code: ProviderAbstentionCode | None,
) -> None:
    result = _result(
        outcome=outcome,
        output_digest=output_digest,
        abstention_code=abstention_code,
    )

    encoded = result.to_json()
    restored = ProviderResultEnvelope.from_json(encoded)

    assert restored == result
    assert restored.to_json() == encoded
    assert encoded == json.dumps(
        json.loads(encoded),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def test_mapping_order_does_not_change_canonical_json() -> None:
    first = _result(count_metadata={"page_count": 2, "token_count": 37})
    second = _result(count_metadata={"token_count": 37, "page_count": 2})

    assert first == second
    assert first.to_json() == second.to_json()


def test_result_and_count_metadata_are_immutable() -> None:
    result = _result()

    with pytest.raises(FrozenInstanceError):
        result.provider_id = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.count_metadata["page_count"] = 3  # type: ignore[index]


def test_to_dict_returns_fresh_nested_data() -> None:
    result = _result()
    payload = result.to_dict()
    payload["provider_id"] = "changed"
    payload["count_metadata"]["page_count"] = 99

    assert result.provider_id == "doctr-1"
    assert result.count_metadata["page_count"] == 2


@pytest.mark.parametrize(
    ("outcome", "output_digest", "abstention_code", "message"),
    [
        (ProviderResultOutcome.SUCCESS, None, None, "requires output digest"),
        (
            ProviderResultOutcome.SUCCESS,
            OUTPUT_DIGEST,
            ProviderAbstentionCode.LOW_QUALITY,
            "abstention code is invalid",
        ),
        (
            ProviderResultOutcome.ABSTENTION,
            None,
            None,
            "requires abstention code",
        ),
        (
            ProviderResultOutcome.ABSTENTION,
            OUTPUT_DIGEST,
            ProviderAbstentionCode.LOW_QUALITY,
            "output digest is invalid",
        ),
        (
            ProviderResultOutcome.PROVIDER_UNAVAILABLE,
            OUTPUT_DIGEST,
            None,
            "output digest is invalid",
        ),
        (
            ProviderResultOutcome.VALIDATION_FAILURE,
            None,
            ProviderAbstentionCode.RESOURCE_LIMIT,
            "abstention code is invalid",
        ),
    ],
)
def test_outcome_cross_field_invariants(
    outcome: ProviderResultOutcome,
    output_digest: str | None,
    abstention_code: ProviderAbstentionCode | None,
    message: str,
) -> None:
    with pytest.raises(ProviderResultError, match=message):
        _result(
            outcome=outcome,
            output_digest=output_digest,
            abstention_code=abstention_code,
        )


@pytest.mark.parametrize("outcome", ["complete", "error", "", 1, True, None])
def test_unknown_outcomes_are_rejected_without_echo(outcome: object) -> None:
    with pytest.raises(ProviderResultError) as error:
        _result(outcome=outcome)
    assert (
        not isinstance(outcome, str) or not outcome or outcome not in str(error.value)
    )


@pytest.mark.parametrize("code", ["provider_unavailable", "other", "", 1, True])
def test_unknown_abstention_codes_are_rejected_without_echo(code: object) -> None:
    with pytest.raises(ProviderResultError) as error:
        _result(
            outcome=ProviderResultOutcome.ABSTENTION,
            output_digest=None,
            abstention_code=code,
        )
    assert not isinstance(code, str) or not code or code not in str(error.value)


@pytest.mark.parametrize(
    "digest",
    ["A" * 64, "a" * 63, "g" * 64, "sha256:" + "a" * 64, b"raw bytes"],
)
@pytest.mark.parametrize("field", ["input_digest", "output_digest"])
def test_invalid_digests_are_rejected_without_echo(field: str, digest: object) -> None:
    with pytest.raises(ProviderResultError) as error:
        _result(**{field: digest})
    assert str(digest) not in str(error.value)


@pytest.mark.parametrize(
    "duration",
    [-1, math.inf, -math.inf, math.nan, 86_400_001, True, "12.5", None],
)
def test_invalid_timings_are_rejected_without_echo(duration: object) -> None:
    with pytest.raises(ProviderResultError) as error:
        _result(duration_ms=duration)
    assert str(duration) not in str(error.value)


@pytest.mark.parametrize("duration", [0, 0.0, 1, 12.5, 86_400_000])
def test_finite_bounded_timings_round_trip(duration: float) -> None:
    result = _result(duration_ms=duration)
    assert ProviderResultEnvelope.from_json(result.to_json()) == result


@pytest.mark.parametrize(
    "counts",
    [
        {"unknown": 1},
        {"page_count": -1},
        {"page_count": 1 << 63},
        {"page_count": True},
        {"page_count": 1.5},
        {"page_count": "1"},
        {1: 1},
        [1],
    ],
)
def test_count_metadata_is_closed_and_bounded(counts: object) -> None:
    with pytest.raises(ProviderResultError):
        _result(count_metadata=counts)


@pytest.mark.parametrize(
    "name",
    [
        "input_bytes",
        "input_items",
        "output_items",
        "page_count",
        "frame_count",
        "sample_count",
        "segment_count",
        "token_count",
        "detection_count",
    ],
)
def test_documented_count_fields_accept_exact_integer_bounds(name: str) -> None:
    for value in (0, (1 << 63) - 1):
        result = _result(count_metadata={name: value})
        assert result.count_metadata[name] == value


@pytest.mark.parametrize(
    "unsafe",
    [
        "Patient Jane synthetic chart text",
        "/srv/charts/patient.dcm",
        "C:\\clinical\\patient.dcm",
        "https://internal.invalid/model",
        "prompt: transcribe the recording",
        "Bearer synthetic-credential",
        "sk-proj-secret-token",
        b"raw identifier bytes",
        "a" * 129,
        "UPPERCASE",
    ],
)
@pytest.mark.parametrize("field", ["provider_id", "model_id"])
def test_unsafe_identifiers_are_rejected_without_echo(
    field: str, unsafe: object
) -> None:
    with pytest.raises(ProviderResultError) as error:
        _result(**{field: unsafe})
    assert str(unsafe) not in str(error.value)


@pytest.mark.parametrize(
    ("field", "sentinel"),
    [
        ("output", "SENTINEL_OCR_TEXT"),
        ("transcript", "SENTINEL_TRANSCRIPT_TEXT"),
        ("dicom_value", "SENTINEL_DICOM_VALUE"),
        ("message", "SENTINEL_PROVIDER_MESSAGE"),
        ("prompt", "SENTINEL_PROMPT_TEXT"),
        ("credential", "SENTINEL_CREDENTIAL"),
        ("path", "/synthetic/private/patient.dcm"),
        ("url", "https://private.invalid/result"),
        ("binary_data", "SENTINEL_PIXEL_OR_WAVEFORM_DATA"),
    ],
)
def test_inline_sensitive_fields_never_appear_in_output_or_errors(
    field: str, sentinel: str
) -> None:
    payload = _result().to_dict()
    payload[field] = sentinel

    with pytest.raises(ProviderResultError) as error:
        ProviderResultEnvelope.from_dict(payload)

    assert sentinel not in str(error.value)
    assert sentinel not in _result().to_json()


def test_unknown_fields_are_rejected_even_when_value_is_binary() -> None:
    payload = _result().to_dict()
    payload["pixels"] = b"synthetic pixels"
    with pytest.raises(ProviderResultError, match="fields are invalid"):
        ProviderResultEnvelope.from_dict(payload)


def test_duplicate_json_keys_are_rejected_without_echo() -> None:
    sentinel = "SENTINEL_DUPLICATE_PROVIDER"
    payload = (
        _result()
        .to_json()
        .replace(
            '"provider_id":"doctr-1"',
            f'"provider_id":"doctr-1","provider_id":"{sentinel}"',
        )
    )
    with pytest.raises(ProviderResultError) as error:
        ProviderResultEnvelope.from_json(payload)
    assert sentinel not in str(error.value)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_numbers_are_rejected(constant: str) -> None:
    payload = (
        _result().to_json().replace('"duration_ms":12.5', f'"duration_ms":{constant}')
    )
    with pytest.raises(ProviderResultError, match="JSON is invalid"):
        ProviderResultEnvelope.from_json(payload)


def test_json_size_limit_is_measured_in_utf8_bytes() -> None:
    oversized_ascii = " " * (MAX_PROVIDER_RESULT_JSON_BYTES + 1)
    oversized_utf8 = "é" * (MAX_PROVIDER_RESULT_JSON_BYTES // 2 + 1)
    for payload in (oversized_ascii, oversized_utf8):
        with pytest.raises(ProviderResultError, match="JSON is invalid"):
            ProviderResultEnvelope.from_json(payload)


@pytest.mark.parametrize("payload", [None, b"{}", [], {}, "[]", "null", "{"])
def test_malformed_or_non_object_json_is_rejected(payload: object) -> None:
    with pytest.raises(ProviderResultError):
        ProviderResultEnvelope.from_json(payload)  # type: ignore[arg-type]


def test_missing_optional_fields_receive_canonical_defaults() -> None:
    payload = _result(
        outcome=ProviderResultOutcome.PROVIDER_UNAVAILABLE,
        output_digest=None,
        count_metadata={},
    ).to_dict()
    del payload["output_digest"]
    del payload["abstention_code"]
    del payload["count_metadata"]

    restored = ProviderResultEnvelope.from_dict(payload)

    assert restored.output_digest is None
    assert restored.abstention_code is None
    assert dict(restored.count_metadata) == {}


def test_schema_version_is_exact_and_errors_do_not_echo() -> None:
    sentinel = "openmed.multimodal.provider_result.v999-SENTINEL"
    with pytest.raises(ProviderResultError) as error:
        _result(schema_version=sentinel)
    assert sentinel not in str(error.value)
    assert _result().schema_version == PROVIDER_RESULT_SCHEMA_VERSION


def test_input_mapping_and_original_counts_are_not_mutated() -> None:
    counts = {"page_count": 2}
    payload = _result(count_metadata=counts).to_dict()
    original = json.loads(json.dumps(payload))

    result = ProviderResultEnvelope.from_dict(payload)
    counts["page_count"] = 99
    payload["count_metadata"]["page_count"] = 100

    assert original["count_metadata"] == {"page_count": 2}
    assert result.count_metadata["page_count"] == 2


def test_unreadable_count_mapping_does_not_leak_upstream_error() -> None:
    sentinel = "SENTINEL_FROM_BROKEN_MAPPING"

    class BrokenMapping(Mapping[str, int]):
        def __getitem__(self, key: str) -> int:
            raise RuntimeError(sentinel)

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError(sentinel)

        def __len__(self) -> int:
            return 1

    with pytest.raises(ProviderResultError) as error:
        _result(count_metadata=BrokenMapping())
    assert sentinel not in str(error.value)


def test_dataclass_replace_revalidates_cross_field_invariants() -> None:
    with pytest.raises(ProviderResultError, match="output digest is invalid"):
        replace(_result(), outcome=ProviderResultOutcome.VALIDATION_FAILURE)


@pytest.mark.parametrize("value", [10**1000, -(10**1000)])
def test_huge_duration_is_a_value_free_validation_error(value):
    with pytest.raises(ProviderResultError, match="^provider duration is invalid$"):
        _result(duration_ms=value)


@pytest.mark.parametrize("field", ["outcome", "abstention_code"])
def test_unknown_enum_does_not_retain_submitted_value(field):
    with pytest.raises(ProviderResultError) as caught:
        _result(**{field: "synthetic-private-note"})
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


def test_malformed_json_does_not_retain_source_content():
    with pytest.raises(ProviderResultError) as caught:
        ProviderResultEnvelope.from_json('{"synthetic-private-note":')
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
