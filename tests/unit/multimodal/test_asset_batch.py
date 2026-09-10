"""Tests for privacy-safe multimodal asset batches."""

from __future__ import annotations

import json

import pytest

from openmed.multimodal.asset_batch import (
    BATCH_VERSION,
    MAX_BATCH_ASSETS,
    AssetBatch,
    AssetBatchError,
    BatchFinding,
    validate_asset_batch,
)
from openmed.multimodal.asset_manifest import (
    MAX_MANIFEST_BYTE_SIZE,
    MAX_MANIFEST_COUNT,
    MAX_MANIFEST_DURATION_SECONDS,
    AssetManifest,
)


def _digest(seed: int) -> str:
    return f"{seed:064x}"


def _image(asset_id: str, seed: int, **fields: object) -> AssetManifest:
    payload: dict[str, object] = {
        "asset_id": asset_id,
        "media_type": "image/png",
        "sha256": _digest(seed),
        "byte_size": 1024,
        "width": 640,
        "height": 480,
    }
    payload.update(fields)
    return AssetManifest.from_dict(payload)


IMAGE = _image("img-001", 1)
PDF = AssetManifest.from_dict(
    {
        "asset_id": "pdf-001",
        "media_type": "application/pdf",
        "sha256": _digest(2),
        "byte_size": 2048,
        "pages": 4,
    }
)
DICOM = AssetManifest.from_dict(
    {
        "asset_id": "dicom-001",
        "media_type": "application/dicom",
        "sha256": _digest(3),
        "byte_size": 4096,
        "frames": 12,
        "width": 512,
        "height": 512,
    }
)
AUDIO = AssetManifest.from_dict(
    {
        "asset_id": "audio-001",
        "media_type": "audio/wav",
        "sha256": _digest(4),
        "byte_size": 8192,
        "duration_seconds": 3.5,
    }
)
AUDIO_2 = AssetManifest.from_dict(
    {
        "asset_id": "audio-002",
        "media_type": "audio/wav",
        "sha256": _digest(5),
        "byte_size": 16,
        "duration_seconds": 2.25,
    }
)
ORDERED_FIELDS = [
    "version",
    "batch_id",
    "asset_count",
    "total_bytes",
    "total_pages",
    "total_frames",
    "total_duration_seconds",
    "assets",
]


def test_mixed_modality_batch_round_trips_with_stable_json() -> None:
    batch = AssetBatch.build("packet-001", [AUDIO, PDF, IMAGE, DICOM])

    assert [manifest.asset_id for manifest in batch.assets] == [
        "audio-001",
        "dicom-001",
        "img-001",
        "pdf-001",
    ]
    assert batch.version == BATCH_VERSION
    assert batch.asset_count == 4
    assert batch.total_bytes == 1024 + 2048 + 4096 + 8192
    assert batch.total_pages == 4
    assert batch.total_frames == 12
    assert batch.total_duration_seconds == 3.5
    assert list(batch.to_dict()) == ORDERED_FIELDS
    assert AssetBatch.from_json(batch.to_json()) == batch
    assert AssetBatch.from_dict(batch.to_dict()) == batch
    assert json.loads(batch.to_json()) == batch.to_dict()
    assert batch.to_json() == batch.to_json()
    assert validate_asset_batch(batch) == []
    assert validate_asset_batch(batch.to_dict()) == []


def test_single_asset_batch_round_trips() -> None:
    batch = AssetBatch.build("packet-single", [PDF])

    assert batch.to_dict() == {
        "version": 1,
        "batch_id": "packet-single",
        "asset_count": 1,
        "total_bytes": 2048,
        "total_pages": 4,
        "total_frames": 0,
        "total_duration_seconds": 0.0,
        "assets": [PDF.to_dict()],
    }
    assert AssetBatch.from_json(batch.to_json()) == batch
    assert validate_asset_batch(batch.to_dict()) == []


def test_empty_batch_is_only_accepted_when_allowed() -> None:
    batch = AssetBatch.build("packet-empty", [])
    expected = [BatchFinding("empty_batch", field_name="assets")]

    assert batch.to_dict()["assets"] == []
    assert batch.total_bytes == 0
    assert batch.total_duration_seconds == 0.0
    assert validate_asset_batch(batch) == expected
    assert validate_asset_batch(batch.to_dict()) == expected
    assert validate_asset_batch(batch, allow_empty=True) == []
    assert AssetBatch.from_json(batch.to_json(), allow_empty=True) == batch

    with pytest.raises(AssetBatchError, match="empty_batch"):
        AssetBatch.from_json(batch.to_json())


def test_duration_totals_use_exact_summation() -> None:
    batch = AssetBatch.build("packet-audio", [AUDIO_2, AUDIO])

    assert batch.total_duration_seconds == 5.75
    assert json.loads(batch.to_json())["total_duration_seconds"] == 5.75
    assert AssetBatch.from_json(batch.to_json()) == batch


def test_build_orders_assets_canonically_and_rejects_other_orders() -> None:
    batch = AssetBatch.build("packet-order", [PDF, IMAGE])
    payload = batch.to_dict()
    payload["assets"] = list(reversed(payload["assets"]))

    assert [entry["asset_id"] for entry in batch.to_dict()["assets"]] == [
        "img-001",
        "pdf-001",
    ]
    assert validate_asset_batch(payload) == [
        BatchFinding("order_not_canonical", position=1)
    ]
    with pytest.raises(AssetBatchError, match="order_not_canonical"):
        AssetBatch(batch_id="packet-order", assets=(PDF, IMAGE))


@pytest.mark.parametrize(
    ("assets", "expected"),
    [
        (
            (IMAGE, _image("img-001", 9)),
            [BatchFinding("duplicate_asset_id", position=1)],
        ),
        (
            (IMAGE, _image("img-002", 1)),
            [BatchFinding("duplicate_sha256", position=1)],
        ),
        (
            (IMAGE, IMAGE),
            [
                BatchFinding("duplicate_asset_id", position=1),
                BatchFinding("duplicate_sha256", position=1),
            ],
        ),
    ],
)
def test_duplicate_identifiers_and_digests_produce_stable_findings(
    assets, expected
) -> None:
    payload = {
        "batch_id": "packet-dup",
        "assets": [manifest.to_dict() for manifest in assets],
    }

    assert validate_asset_batch(payload) == expected
    with pytest.raises(AssetBatchError) as exc_info:
        AssetBatch(batch_id="packet-dup", assets=assets)
    for finding in expected:
        assert finding.reason_code in str(exc_info.value)
    with pytest.raises(AssetBatchError):
        AssetBatch.build("packet-dup", assets)


def test_batches_above_the_policy_limit_fail_closed() -> None:
    batch = AssetBatch.build("packet-limit", [IMAGE, PDF])
    expected = [BatchFinding("batch_too_large", field_name="assets")]

    assert validate_asset_batch(batch, max_assets=2) == []
    assert validate_asset_batch(batch, max_assets=1) == expected
    assert validate_asset_batch(batch.to_dict(), max_assets=1) == expected
    with pytest.raises(AssetBatchError, match="batch_too_large"):
        AssetBatch.from_dict(batch.to_dict(), max_assets=1)


def test_batches_above_the_hard_cap_fail_closed_with_one_finding() -> None:
    assets = tuple(
        AssetManifest(
            asset_id=f"asset-{index:05d}",
            media_type="image/png",
            sha256=_digest(index + 1),
            byte_size=1,
        )
        for index in range(MAX_BATCH_ASSETS + 1)
    )
    payload = {"batch_id": "packet-cap", "assets": [m.to_dict() for m in assets]}

    with pytest.raises(AssetBatchError, match="batch_too_large"):
        AssetBatch(batch_id="packet-cap", assets=assets)
    assert validate_asset_batch(payload) == [
        BatchFinding("batch_too_large", field_name="assets")
    ]


@pytest.mark.parametrize(
    ("field", "assets"),
    [
        (
            "total_bytes",
            (
                _image("img-001", 1, byte_size=MAX_MANIFEST_BYTE_SIZE),
                _image("img-002", 2, byte_size=MAX_MANIFEST_BYTE_SIZE),
            ),
        ),
        (
            "total_pages",
            (
                _image("pdf-001", 1, pages=MAX_MANIFEST_COUNT),
                _image("pdf-002", 2, pages=MAX_MANIFEST_COUNT),
            ),
        ),
        (
            "total_frames",
            (
                _image("dicom-001", 1, frames=MAX_MANIFEST_COUNT),
                _image("dicom-002", 2, frames=MAX_MANIFEST_COUNT),
            ),
        ),
        (
            "total_duration_seconds",
            (
                _image("audio-001", 1, duration_seconds=MAX_MANIFEST_DURATION_SECONDS),
                _image("audio-002", 2, duration_seconds=MAX_MANIFEST_DURATION_SECONDS),
            ),
        ),
    ],
)
def test_aggregate_overflow_fails_closed(field, assets) -> None:
    payload = {
        "batch_id": "packet-overflow",
        "assets": [manifest.to_dict() for manifest in assets],
    }

    assert validate_asset_batch(payload) == [
        BatchFinding("aggregate_overflow", field_name=field)
    ]
    with pytest.raises(AssetBatchError, match="aggregate_overflow"):
        AssetBatch(batch_id="packet-overflow", assets=assets)


@pytest.mark.parametrize(
    ("field", "value", "reason_code"),
    [
        ("asset_count", 3, "aggregate_mismatch"),
        ("total_bytes", 1, "aggregate_mismatch"),
        ("total_pages", 0, "aggregate_mismatch"),
        ("total_frames", 13, "aggregate_mismatch"),
        ("total_duration_seconds", 3.6, "aggregate_mismatch"),
        ("total_duration_seconds", -3.5, "aggregate_mismatch"),
        ("asset_count", True, "aggregate_invalid"),
        ("total_bytes", "15360", "aggregate_invalid"),
        ("total_bytes", 15360.0, "aggregate_invalid"),
        ("total_frames", None, "aggregate_invalid"),
        ("total_duration_seconds", float("nan"), "aggregate_invalid"),
        ("total_duration_seconds", float("inf"), "aggregate_invalid"),
        ("total_duration_seconds", True, "aggregate_invalid"),
        ("total_duration_seconds", "3.5", "aggregate_invalid"),
    ],
)
def test_declared_aggregates_must_match_manifests(field, value, reason_code) -> None:
    batch = AssetBatch.build("packet-agg", [AUDIO, PDF, IMAGE, DICOM])
    payload = batch.to_dict()
    payload[field] = value

    assert validate_asset_batch(payload) == [
        BatchFinding(reason_code, field_name=field)
    ]
    with pytest.raises(AssetBatchError, match=reason_code):
        AssetBatch.from_dict(payload)


def test_duration_tolerance_is_absolute_not_relative() -> None:
    duration = float(MAX_MANIFEST_DURATION_SECONDS)
    audio = _image("audio-large", 9, duration_seconds=duration)
    batch = AssetBatch.build("packet-duration", [audio])
    payload = batch.to_dict()
    payload["total_duration_seconds"] = duration - 1.0

    assert validate_asset_batch(payload) == [
        BatchFinding("aggregate_mismatch", field_name="total_duration_seconds")
    ]
    with pytest.raises(AssetBatchError, match="aggregate_mismatch"):
        AssetBatch.from_dict(payload)


def test_declared_aggregates_are_optional_and_accept_integer_durations() -> None:
    batch = AssetBatch.build("packet-optional", [PDF, IMAGE])
    payload = batch.to_dict()
    for field in ORDERED_FIELDS[2:-1]:
        payload.pop(field)

    assert AssetBatch.from_dict(payload) == batch
    assert AssetBatch.from_dict({**payload, "total_duration_seconds": 0}) == batch


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("not-a-dict", [BatchFinding("invalid_batch")]),
        (
            {},
            [
                BatchFinding("missing_required", field_name="assets"),
                BatchFinding("missing_required", field_name="batch_id"),
            ],
        ),
        (
            {"batch_id": "packet-001"},
            [BatchFinding("missing_required", field_name="assets")],
        ),
        (
            {"batch_id": "packet-001", "assets": [], "patient_name": "x"},
            [BatchFinding("unknown_field")],
        ),
        (
            {"batch_id": "packet-001", "assets": [IMAGE.to_dict()], "version": 2},
            [BatchFinding("invalid_version", field_name="version")],
        ),
        (
            {"batch_id": "packet-001", "assets": [IMAGE.to_dict()], "version": True},
            [BatchFinding("invalid_version", field_name="version")],
        ),
        (
            {"batch_id": "packet-001", "assets": [IMAGE.to_dict()], "version": "1"},
            [BatchFinding("invalid_version", field_name="version")],
        ),
        (
            {"batch_id": "", "assets": [IMAGE.to_dict()]},
            [BatchFinding("invalid_batch_id", field_name="batch_id")],
        ),
        (
            {"batch_id": 42, "assets": [IMAGE.to_dict()]},
            [BatchFinding("invalid_batch_id", field_name="batch_id")],
        ),
        (
            {"batch_id": "a" * 129, "assets": [IMAGE.to_dict()]},
            [BatchFinding("invalid_batch_id", field_name="batch_id")],
        ),
        (
            {"batch_id": "/tmp/packet", "assets": [IMAGE.to_dict()]},
            [BatchFinding("invalid_batch_id", field_name="batch_id")],
        ),
        (
            {"batch_id": "s3://bucket/packet", "assets": [IMAGE.to_dict()]},
            [BatchFinding("invalid_batch_id", field_name="batch_id")],
        ),
        (
            {"batch_id": "packet-001", "assets": "img-001"},
            [BatchFinding("invalid_assets", field_name="assets")],
        ),
        (
            {"batch_id": "packet-001", "assets": IMAGE.to_dict()},
            [BatchFinding("invalid_assets", field_name="assets")],
        ),
        (
            {"batch_id": "packet-001", "assets": 5},
            [BatchFinding("invalid_assets", field_name="assets")],
        ),
        (
            {"batch_id": "packet-001", "assets": ["img-001"]},
            [BatchFinding("invalid_asset", position=0)],
        ),
        (
            {"batch_id": "packet-001", "assets": [IMAGE.to_dict(), 42]},
            [BatchFinding("invalid_asset", position=1)],
        ),
        (
            {
                "batch_id": "packet-001",
                "assets": [IMAGE.to_dict(), {**PDF.to_dict(), "description": "x"}],
            },
            [BatchFinding("invalid_asset", position=1)],
        ),
        (
            {"batch_id": "/tmp/packet", "assets": [42], "version": 2},
            [
                BatchFinding("invalid_asset", position=0),
                BatchFinding("invalid_batch_id", field_name="batch_id"),
                BatchFinding("invalid_version", field_name="version"),
            ],
        ),
    ],
)
def test_schema_failures_produce_stable_findings_and_fail_closed(
    payload, expected
) -> None:
    assert validate_asset_batch(payload) == expected
    with pytest.raises(AssetBatchError) as exc_info:
        AssetBatch.from_dict(payload)
    for finding in expected:
        assert finding.reason_code in str(exc_info.value)


def test_findings_are_sorted_and_deduplicated() -> None:
    payload = {
        "batch_id": "packet-mixed",
        "assets": [_image("img-002", 1).to_dict(), IMAGE.to_dict()],
    }

    assert validate_asset_batch(payload, max_assets=1) == [
        BatchFinding("batch_too_large", field_name="assets"),
        BatchFinding("duplicate_sha256", position=1),
        BatchFinding("order_not_canonical", position=1),
    ]


def test_sentinel_values_never_appear_in_findings_or_errors() -> None:
    sentinels = (
        "C:\\patients\\JANE-DOE-19700101",
        "synthetic sentinel Jane Doe",
        "synthetic sentinel chart text",
        "patient_name",
    )
    payload = {
        "batch_id": sentinels[0],
        "assets": [
            IMAGE.to_dict(),
            {**PDF.to_dict(), "description": sentinels[2]},
        ],
        sentinels[3]: sentinels[1],
    }

    findings = validate_asset_batch(payload)
    with pytest.raises(AssetBatchError) as exc_info:
        AssetBatch.from_dict(payload)
    rendered = json.dumps([finding.to_dict() for finding in findings])
    assert findings
    for sentinel in sentinels:
        assert sentinel not in rendered
        assert sentinel not in str(exc_info.value)

    with pytest.raises(AssetBatchError) as direct_error:
        AssetBatch(batch_id=sentinels[0], assets=(IMAGE,))
    assert sentinels[0] not in str(direct_error.value)


@pytest.mark.parametrize("payload", ["{", b"\xff", 42])
def test_malformed_json_fails_closed(payload) -> None:
    with pytest.raises(AssetBatchError):
        AssetBatch.from_json(payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "payload",
    [
        '{"batch_id":"packet-001","batch_id":"synthetic-secret","assets":[]}',
        (
            '{"batch_id":"packet-001","assets":[{"asset_id":"img-001",'
            '"asset_id":"synthetic-secret","media_type":"image/png",'
            f'"sha256":"{_digest(1)}","byte_size":1}}]}}'
        ),
    ],
)
def test_duplicate_json_fields_fail_closed_without_echoing_values(payload) -> None:
    with pytest.raises(AssetBatchError) as exc_info:
        AssetBatch.from_json(payload)

    assert "synthetic-secret" not in str(exc_info.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_assets": 0},
        {"max_assets": True},
        {"max_assets": MAX_BATCH_ASSETS + 1},
        {"max_assets": "5"},
        {"allow_empty": "yes"},
        {"allow_empty": 1},
    ],
)
def test_invalid_policy_arguments_fail_closed(kwargs) -> None:
    batch = AssetBatch.build("packet-policy", [IMAGE])

    with pytest.raises(AssetBatchError):
        validate_asset_batch(batch, **kwargs)
    with pytest.raises(AssetBatchError):
        AssetBatch.from_dict(batch.to_dict(), **kwargs)


@pytest.mark.parametrize("manifests", [42, [IMAGE, "pdf-001"], [IMAGE.to_dict()]])
def test_build_rejects_non_manifest_inputs(manifests) -> None:
    with pytest.raises(AssetBatchError):
        AssetBatch.build("packet-build", manifests)


@pytest.mark.parametrize(
    "assets",
    [[IMAGE], (IMAGE, "pdf-001"), (IMAGE, PDF.to_dict())],
)
def test_direct_construction_requires_a_tuple_of_manifests(assets) -> None:
    with pytest.raises(AssetBatchError):
        AssetBatch(batch_id="packet-direct", assets=assets)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reason_code": "patient_leaked"},
        {"reason_code": 1},
        {"reason_code": "unknown_field", "field_name": "patient_name"},
        {"reason_code": "invalid_asset", "position": -1},
        {"reason_code": "invalid_asset", "position": True},
        {"reason_code": "invalid_asset", "position": "0"},
    ],
)
def test_findings_reject_unsupported_codes_fields_and_positions(kwargs) -> None:
    with pytest.raises(AssetBatchError):
        BatchFinding(**kwargs)


def test_findings_serialize_without_unset_fields() -> None:
    assert BatchFinding("invalid_batch").to_dict() == {"reason_code": "invalid_batch"}
    assert BatchFinding("invalid_asset", position=2).to_dict() == {
        "reason_code": "invalid_asset",
        "position": 2,
    }
    assert BatchFinding("empty_batch", field_name="assets").to_dict() == {
        "reason_code": "empty_batch",
        "field_name": "assets",
    }


def test_batch_contract_is_available_from_public_multimodal_api() -> None:
    import openmed.multimodal as multimodal

    assert multimodal.AssetBatch is AssetBatch
    assert multimodal.AssetBatchError is AssetBatchError
    assert multimodal.BatchFinding is BatchFinding
    assert multimodal.validate_asset_batch is validate_asset_batch
    assert multimodal.BATCH_VERSION == 1
    assert multimodal.MAX_BATCH_ASSETS == 10_000
