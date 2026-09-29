"""Contract tests for the exported multimodal preflight JSON Schemas."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from typing import Any, NoReturn

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry

from openmed.multimodal import schemas
from openmed.multimodal.abstention import (
    _ALLOWED_REASONS,
    ABSTENTION_SCHEMA_VERSION,
    AbstentionReason,
    AbstentionRecord,
    AbstentionStage,
    AbstentionValidationError,
)
from openmed.multimodal.asset_batch import (
    _AGGREGATE_LIMITS,
    BATCH_VERSION,
    MAX_BATCH_ASSETS,
    AssetBatch,
    AssetBatchError,
)
from openmed.multimodal.asset_manifest import (
    _SUPPORTED_EXACT_MEDIA_TYPES,
    _SUPPORTED_MEDIA_PREFIXES,
    MANIFEST_VERSION,
    MAX_MANIFEST_BYTE_SIZE,
    MAX_MANIFEST_COUNT,
    MAX_MANIFEST_DURATION_SECONDS,
    AssetManifest,
    AssetManifestError,
)
from openmed.multimodal.digest import AssetDigest
from openmed.multimodal.processing_summary import (
    _SUMMARY_ORDERED_FIELDS,
    PROCESSING_SUMMARY_SCHEMA_VERSION,
    AssetProcessingResult,
    ProcessingOutcome,
    summarize_processing_run,
)
from openmed.multimodal.provider_result import (
    _COUNT_FIELDS,
    _MAX_COUNT,
    _MAX_DURATION_MS,
    _REQUIRED_FIELDS,
    PROVIDER_RESULT_SCHEMA_VERSION,
    ProviderAbstentionCode,
    ProviderResultEnvelope,
    ProviderResultError,
    ProviderResultOutcome,
)

_SCHEMA_ID_PREFIX = "https://openmed.ai/schemas/multimodal/"
_RENDERED_SCHEMA_DIGESTS = {
    "asset_manifest": (
        "ee7e6dd106af9be4bd293b746b9e0cfe9122e33827889ae4e7a4885ec63a7797"
    ),
    "asset_batch": ("bbb4c7c8c72c1ef9ceb542ab965e753d2675b99e716c138599ed4b822b982684"),
    "abstention_record": (
        "eabe10c1f72b14283d97455ef3d8ad4205168b5711026a31a92495308f941175"
    ),
    "processing_summary": (
        "b825bd84dbb6e0c6b91a8908f78df7fa0b9e0e451879cb46f86b1f3dbc60efb9"
    ),
    "provider_result": (
        "e5565619893dac402564612cb31301d9116a3243ae8da09a488f7df0a9184337"
    ),
}
_CATALOG_DIGEST = "20eb4c8cb74858b42d82014825bcaca5a8420b8b2ce760cd8fa8f78553deaaae"


def _validator(name: str) -> Draft202012Validator:
    def reject_remote_resolution(uri: str) -> NoReturn:
        raise AssertionError(f"unexpected remote schema resolution: {uri}")

    document = schemas.export_multimodal_schema(name)
    registry = Registry(retrieve=reject_remote_resolution)  # type: ignore[call-arg]
    return Draft202012Validator(document, registry=registry)


def _refs(node: Any) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        if "$ref" in node:
            found.append(node["$ref"])
        for value in node.values():
            found.extend(_refs(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_refs(item))
    return found


def _object_schemas(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            found.extend(_object_schemas(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_object_schemas(item))
    return found


def _without(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in keys}


def _manifest_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "asset_id": "asset-1",
        "media_type": "image/png",
        "sha256": "a" * 64,
        "byte_size": 1024,
    }
    payload.update(overrides)
    return payload


def _batch_payload() -> dict[str, Any]:
    manifest = AssetManifest.from_dict(_manifest_payload())
    return AssetBatch.build("batch-1", [manifest]).to_dict()


def _abstention_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": ABSTENTION_SCHEMA_VERSION,
        "stage": AbstentionStage.DECODE.value,
        "reason": AbstentionReason.MALFORMED_MEDIA.value,
    }
    payload.update(overrides)
    return payload


def _processing_summary_payload() -> dict[str, Any]:
    manifest = AssetManifest.from_dict(_manifest_payload())
    success = AssetProcessingResult(
        manifest=manifest,
        outcome_code=ProcessingOutcome.SUCCESS,
        duration_seconds=1.5,
        input_digest=AssetDigest(sha256=manifest.sha256, byte_count=manifest.byte_size),
        output_digest=AssetDigest(sha256="c" * 64, byte_count=512),
    )
    abstained_manifest = AssetManifest.from_dict(
        _manifest_payload(asset_id="asset-2", sha256="b" * 64)
    )
    abstained = AssetProcessingResult(
        manifest=abstained_manifest,
        outcome_code=ProcessingOutcome.ABSTAINED,
        duration_seconds=0.25,
        input_digest=AssetDigest(
            sha256=abstained_manifest.sha256,
            byte_count=abstained_manifest.byte_size,
        ),
        abstention=AbstentionRecord.from_dict(_abstention_payload()),
    )
    return summarize_processing_run([success, abstained]).to_dict()


def _provider_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": PROVIDER_RESULT_SCHEMA_VERSION,
        "provider_id": "local-ocr",
        "model_id": "doctr-base",
        "input_digest": "b" * 64,
        "output_digest": "c" * 64,
        "outcome": ProviderResultOutcome.SUCCESS.value,
        "abstention_code": None,
        "duration_ms": 12.5,
        "count_metadata": {"page_count": 2},
    }
    payload.update(overrides)
    return payload


def _provider_payloads() -> list[dict[str, Any]]:
    return [
        ProviderResultEnvelope(
            provider_id="local-ocr",
            model_id="doctr-base",
            input_digest="b" * 64,
            output_digest="c" * 64,
            outcome=ProviderResultOutcome.SUCCESS,
            duration_ms=12.5,
            count_metadata={"page_count": 2},
        ).to_dict(),
        ProviderResultEnvelope(
            provider_id="local-ocr",
            model_id="doctr-base",
            input_digest="b" * 64,
            outcome=ProviderResultOutcome.ABSTENTION,
            abstention_code=ProviderAbstentionCode.LOW_QUALITY,
            duration_ms=1.0,
        ).to_dict(),
        ProviderResultEnvelope(
            provider_id="local-ocr",
            model_id="doctr-base",
            input_digest="b" * 64,
            outcome=ProviderResultOutcome.PROVIDER_UNAVAILABLE,
            duration_ms=0.0,
        ).to_dict(),
        ProviderResultEnvelope(
            provider_id="local-ocr",
            model_id="doctr-base",
            input_digest="b" * 64,
            outcome=ProviderResultOutcome.VALIDATION_FAILURE,
            duration_ms=3.25,
        ).to_dict(),
    ]


def test_schema_catalog_matches_the_exported_names() -> None:
    catalog = schemas.build_multimodal_schemas()

    assert tuple(catalog) == schemas.MULTIMODAL_SCHEMA_NAMES
    assert len(catalog) == len(set(catalog)) == 5


def test_schemas_are_valid_draft_2020_12_and_resolve_locally() -> None:
    payloads = {
        "asset_manifest": _manifest_payload(),
        "asset_batch": _batch_payload(),
        "abstention_record": _abstention_payload(),
        "processing_summary": _processing_summary_payload(),
        "provider_result": _provider_payload(),
    }

    for name in schemas.MULTIMODAL_SCHEMA_NAMES:
        document = schemas.export_multimodal_schema(name)
        Draft202012Validator.check_schema(document)
        assert document["$schema"] == schemas.MULTIMODAL_SCHEMA_DIALECT
        assert document["$id"].startswith(_SCHEMA_ID_PREFIX)
        assert document["$ref"].startswith("#/$defs/")
        _validator(name).validate(payloads[name])


def test_schema_identifiers_are_unique_and_versioned() -> None:
    identifiers = {
        name: schemas.export_multimodal_schema(name)["$id"]
        for name in schemas.MULTIMODAL_SCHEMA_NAMES
    }

    assert identifiers == {
        "asset_manifest": schemas.MANIFEST_SCHEMA_ID,
        "asset_batch": schemas.BATCH_SCHEMA_ID,
        "abstention_record": schemas.ABSTENTION_SCHEMA_ID,
        "processing_summary": schemas.PROCESSING_SUMMARY_SCHEMA_ID,
        "provider_result": schemas.PROVIDER_RESULT_SCHEMA_ID,
    }
    assert len(set(identifiers.values())) == 5
    assert all(value.endswith("-v1.schema.json") for value in identifiers.values())


def test_every_reference_targets_a_local_definition() -> None:
    for name in schemas.MULTIMODAL_SCHEMA_NAMES:
        document = schemas.export_multimodal_schema(name)
        references = _refs(document)

        assert references
        assert all(reference.startswith("#/$defs/") for reference in references)
        for reference in references:
            assert reference.removeprefix("#/$defs/") in document["$defs"]


def test_every_declared_object_is_closed() -> None:
    for name in schemas.MULTIMODAL_SCHEMA_NAMES:
        objects = _object_schemas(schemas.export_multimodal_schema(name))

        assert objects
        for node in objects:
            if "propertyNames" in node:
                assert isinstance(node["additionalProperties"], dict)
            else:
                assert node["additionalProperties"] is False


def test_build_returns_fresh_documents() -> None:
    first = schemas.export_multimodal_schema("asset_manifest")
    first["$defs"]["manifest"]["required"].append("asset_id")
    first["$defs"]["manifest"]["properties"].pop("byte_size")

    second = schemas.export_multimodal_schema("asset_manifest")

    assert second["$defs"]["manifest"]["required"] == [
        "asset_id",
        "byte_size",
        "media_type",
        "sha256",
    ]
    assert "byte_size" in second["$defs"]["manifest"]["properties"]


def test_export_multimodal_schema_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown multimodal schema name"):
        schemas.export_multimodal_schema("asset_manifests")


def test_rendered_schemas_are_byte_stable() -> None:
    for name in schemas.MULTIMODAL_SCHEMA_NAMES:
        first = schemas.export_multimodal_schema_json(name)

        assert first == schemas.export_multimodal_schema_json(name)
        assert not first.endswith("\n")
        assert json.loads(first) == schemas.export_multimodal_schema(name)

    catalog = schemas.export_multimodal_schemas_json()

    assert catalog == schemas.export_multimodal_schemas_json()
    assert not catalog.endswith("\n")
    assert json.loads(catalog) == schemas.build_multimodal_schemas()


def test_rendered_schema_digests_are_pinned() -> None:
    for name, digest in _RENDERED_SCHEMA_DIGESTS.items():
        rendered = schemas.export_multimodal_schema_json(name)

        assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == digest

    catalog = schemas.export_multimodal_schemas_json()

    assert hashlib.sha256(catalog.encode("utf-8")).hexdigest() == _CATALOG_DIGEST


def test_manifest_schema_matches_the_python_fields() -> None:
    document = schemas.export_multimodal_schema("asset_manifest")
    manifest = document["$defs"]["manifest"]
    optional = {"version", "pages", "width", "height", "frames", "duration_seconds"}

    assert set(manifest["properties"]) == {
        field.name for field in fields(AssetManifest)
    }
    assert (
        set(manifest["required"])
        == {field.name for field in fields(AssetManifest)} - optional
    )
    assert manifest["properties"]["version"] == {"const": MANIFEST_VERSION}
    assert manifest["properties"]["asset_id"] == {"$ref": "#/$defs/opaqueIdentifier"}
    assert manifest["properties"]["sha256"] == {"$ref": "#/$defs/sha256Digest"}


def test_manifest_schema_bounds_match_the_python_constants() -> None:
    document = schemas.export_multimodal_schema("asset_manifest")
    definitions = document["$defs"]
    manifest = definitions["manifest"]

    assert definitions["positiveByteSize"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": MAX_MANIFEST_BYTE_SIZE,
    }
    assert definitions["positiveCount"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": MAX_MANIFEST_COUNT,
    }
    assert definitions["positiveDuration"] == {
        "type": "number",
        "exclusiveMinimum": 0,
        "maximum": MAX_MANIFEST_DURATION_SECONDS,
    }
    for field_name in ("pages", "width", "height", "frames"):
        assert manifest["properties"][field_name] == {
            "anyOf": [{"$ref": "#/$defs/positiveCount"}, {"type": "null"}]
        }
    assert manifest["properties"]["duration_seconds"] == {
        "anyOf": [{"$ref": "#/$defs/positiveDuration"}, {"type": "null"}]
    }


def test_manifest_media_type_families_match_the_python_constants() -> None:
    document = schemas.export_multimodal_schema("asset_manifest")
    media_type = document["$defs"]["mediaType"]
    validator = _validator("asset_manifest")

    assert media_type["anyOf"][0] == {"enum": sorted(_SUPPORTED_EXACT_MEDIA_TYPES)}
    assert media_type["anyOf"][1:] == [
        {"pattern": f"^{prefix}"} for prefix in sorted(_SUPPORTED_MEDIA_PREFIXES)
    ]
    assert schemas._SUPPORTED_EXACT_MEDIA_TYPES == _SUPPORTED_EXACT_MEDIA_TYPES
    assert schemas._SUPPORTED_MEDIA_PREFIXES == tuple(sorted(_SUPPORTED_MEDIA_PREFIXES))
    for exact in sorted(_SUPPORTED_EXACT_MEDIA_TYPES):
        assert validator.is_valid(_manifest_payload(media_type=exact))
    for prefix in _SUPPORTED_MEDIA_PREFIXES:
        assert validator.is_valid(_manifest_payload(media_type=f"{prefix}custom"))


def test_manifest_schema_accepts_serialized_manifests() -> None:
    validator = _validator("asset_manifest")
    payloads = [
        _manifest_payload(),
        _manifest_payload(
            version=MANIFEST_VERSION,
            pages=2,
            width=640,
            height=480,
            frames=1,
            duration_seconds=2.5,
        ),
    ]

    for payload in payloads:
        manifest = AssetManifest.from_dict(payload)

        assert validator.is_valid(payload)
        assert validator.is_valid(json.loads(manifest.to_json()))


def test_manifest_schema_accepts_explicit_nulls_the_loader_treats_as_absent() -> None:
    payload = _manifest_payload(pages=None, width=None, duration_seconds=None)
    manifest = AssetManifest.from_dict(payload)

    assert _validator("asset_manifest").is_valid(payload)
    assert manifest.to_dict() == _manifest_payload(version=MANIFEST_VERSION)
    assert manifest.pages is None and manifest.duration_seconds is None


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(_manifest_payload(unknown=1), id="unknown-field"),
        pytest.param(_without(_manifest_payload(), "sha256"), id="missing-field"),
        pytest.param(_manifest_payload(version=2), id="unsupported-version"),
        pytest.param(_manifest_payload(version="1"), id="version-not-integer"),
        pytest.param(_manifest_payload(asset_id="asset/1"), id="path-in-asset-id"),
        pytest.param(
            _manifest_payload(asset_id="~/notes.txt"), id="home-path-asset-id"
        ),
        pytest.param(_manifest_payload(asset_id="a" * 129), id="oversized-asset-id"),
        pytest.param(_manifest_payload(sha256="A" * 64), id="uppercase-digest"),
        pytest.param(_manifest_payload(sha256="a" * 63), id="short-digest"),
        pytest.param(_manifest_payload(byte_size=0), id="zero-byte-size"),
        pytest.param(
            _manifest_payload(byte_size=MAX_MANIFEST_BYTE_SIZE + 1),
            id="oversized-byte-size",
        ),
        pytest.param(_manifest_payload(byte_size=True), id="boolean-byte-size"),
        pytest.param(_manifest_payload(pages=0), id="zero-pages"),
        pytest.param(
            _manifest_payload(width=MAX_MANIFEST_COUNT + 1), id="oversized-width"
        ),
        pytest.param(_manifest_payload(duration_seconds=0), id="zero-duration"),
        pytest.param(
            _manifest_payload(duration_seconds=MAX_MANIFEST_DURATION_SECONDS + 1),
            id="oversized-duration",
        ),
        pytest.param(
            _manifest_payload(duration_seconds=float("inf")), id="infinite-duration"
        ),
        pytest.param(
            _manifest_payload(media_type="text/plain"), id="unsupported-media"
        ),
        pytest.param(_manifest_payload(media_type="IMAGE/PNG"), id="uppercase-media"),
        pytest.param(_manifest_payload(media_type="image/"), id="incomplete-media"),
    ],
)
def test_manifest_schema_rejects_malformed_payloads(payload: dict[str, Any]) -> None:
    assert not _validator("asset_manifest").is_valid(payload)
    with pytest.raises(AssetManifestError):
        AssetManifest.from_dict(payload)


def test_batch_schema_matches_the_python_contract() -> None:
    document = schemas.export_multimodal_schema("asset_batch")
    batch = document["$defs"]["batch"]
    properties = batch["properties"]

    assert set(properties) == {
        "version",
        "batch_id",
        "asset_count",
        "total_bytes",
        "total_pages",
        "total_frames",
        "total_duration_seconds",
        "assets",
    }
    assert set(batch["required"]) == {"batch_id", "assets"}
    assert properties["version"] == {"const": BATCH_VERSION}
    assert properties["batch_id"] == {"$ref": "#/$defs/opaqueIdentifier"}
    assert properties["assets"]["minItems"] == 1
    assert properties["assets"]["maxItems"] == MAX_BATCH_ASSETS
    assert properties["assets"]["items"] == {"$ref": "#/$defs/manifest"}
    assert properties["asset_count"]["maximum"] == MAX_BATCH_ASSETS
    assert {
        field_name: properties[field_name]["maximum"]
        for field_name in _AGGREGATE_LIMITS
    } == _AGGREGATE_LIMITS


def test_batch_schema_accepts_serialized_batches() -> None:
    validator = _validator("asset_batch")
    batch = AssetBatch.build(
        "batch-1",
        [
            AssetManifest.from_dict(_manifest_payload()),
            AssetManifest.from_dict(
                _manifest_payload(asset_id="asset-2", sha256="b" * 64, byte_size=2048)
            ),
        ],
    )
    payload = batch.to_dict()

    assert validator.is_valid(payload)
    assert AssetBatch.from_dict(payload).to_dict() == payload
    assert AssetBatch.from_dict(_without(payload, "asset_count")).to_dict() == payload


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"batch_id": "batch-1", "assets": [dict(_manifest_payload())], "extra": 1},
            id="unknown-field",
        ),
        pytest.param(
            {
                "batch_id": "batch-1",
                "assets": [dict(_manifest_payload())],
                "version": 2,
            },
            id="unsupported-version",
        ),
        pytest.param(
            {"batch_id": "batch/1", "assets": [dict(_manifest_payload())]},
            id="path-in-batch-id",
        ),
        pytest.param({"batch_id": "batch-1", "assets": []}, id="empty-assets"),
        pytest.param({"batch_id": "batch-1", "assets": "batch"}, id="assets-not-list"),
        pytest.param({"batch_id": "batch-1"}, id="missing-assets"),
        pytest.param(
            {
                "batch_id": "batch-1",
                "assets": [dict(_manifest_payload(sha256="A" * 64))],
            },
            id="invalid-nested-manifest",
        ),
        pytest.param(
            {
                "batch_id": "batch-1",
                "assets": [dict(_manifest_payload())],
                "asset_count": -1,
            },
            id="negative-asset-count",
        ),
        pytest.param(
            {
                "batch_id": "batch-1",
                "assets": [dict(_manifest_payload())],
                "total_bytes": MAX_MANIFEST_BYTE_SIZE + 1,
            },
            id="aggregate-overflow",
        ),
        pytest.param(
            {
                "batch_id": "batch-1",
                "assets": [dict(_manifest_payload()), dict(_manifest_payload())],
            },
            id="duplicate-asset-records",
        ),
    ],
)
def test_batch_schema_rejects_malformed_payloads(payload: dict[str, Any]) -> None:
    assert not _validator("asset_batch").is_valid(payload)
    with pytest.raises(AssetBatchError):
        AssetBatch.from_dict(payload)


def test_abstention_schema_matches_the_python_contract() -> None:
    document = schemas.export_multimodal_schema("abstention_record")
    record = document["$defs"]["abstentionRecord"]

    assert set(record["required"]) == {"schema_version", "stage", "reason"}
    assert set(record["properties"]) == {"schema_version", "stage", "reason"}
    assert record["properties"]["schema_version"] == {
        "const": ABSTENTION_SCHEMA_VERSION
    }
    assert record["properties"]["stage"]["enum"] == [
        stage.value for stage in AbstentionStage
    ]
    assert record["properties"]["reason"]["enum"] == [
        reason.value for reason in AbstentionReason
    ]
    assert schemas._REASONS_BY_STAGE == {
        stage.value: tuple(sorted(reason.value for reason in reasons))
        for stage, reasons in _ALLOWED_REASONS.items()
    }


def test_abstention_schema_accepts_every_allowed_stage_reason_pair() -> None:
    validator = _validator("abstention_record")

    for stage, reasons in _ALLOWED_REASONS.items():
        for reason in sorted(reasons, key=lambda item: item.value):
            payload = _abstention_payload(stage=stage.value, reason=reason.value)

            assert validator.is_valid(payload)
            assert AbstentionRecord.from_dict(payload).to_dict() == payload


def test_abstention_schema_rejects_every_disallowed_stage_reason_pair() -> None:
    validator = _validator("abstention_record")

    for stage, reasons in _ALLOWED_REASONS.items():
        for reason in AbstentionReason:
            if reason in reasons:
                continue
            payload = _abstention_payload(stage=stage.value, reason=reason.value)

            assert not validator.is_valid(payload)
            with pytest.raises(AbstentionValidationError):
                AbstentionRecord.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(_abstention_payload(extra="value"), id="unknown-field"),
        pytest.param(_abstention_payload(schema_version=2), id="unsupported-version"),
        pytest.param(_abstention_payload(stage="transport"), id="unknown-stage"),
        pytest.param(_abstention_payload(reason="timeout"), id="unknown-reason"),
        pytest.param(_abstention_payload(stage=1), id="stage-not-string"),
        pytest.param(_abstention_payload(reason=None), id="reason-not-string"),
        pytest.param(
            _without(_abstention_payload(), "schema_version"), id="missing-version"
        ),
    ],
)
def test_abstention_schema_rejects_malformed_payloads(payload: dict[str, Any]) -> None:
    assert not _validator("abstention_record").is_valid(payload)
    with pytest.raises(AbstentionValidationError):
        AbstentionRecord.from_dict(payload)


def test_processing_summary_schema_matches_the_python_contract() -> None:
    document = schemas.export_multimodal_schema("processing_summary")
    definitions = document["$defs"]
    summary = definitions["processingSummary"]

    assert tuple(summary["required"]) == _SUMMARY_ORDERED_FIELDS
    assert set(summary["properties"]) == set(_SUMMARY_ORDERED_FIELDS)
    assert summary["properties"]["schema_version"] == {
        "const": PROCESSING_SUMMARY_SCHEMA_VERSION
    }
    assert summary["properties"]["by_media_type"]["items"] == {
        "$ref": "#/$defs/mediaTypeTotals"
    }
    assert summary["properties"]["outcome_counts"]["items"] == {
        "$ref": "#/$defs/outcomeCount"
    }
    assert summary["properties"]["abstention_counts"]["items"] == {
        "$ref": "#/$defs/abstentionCount"
    }
    assert summary["properties"]["asset_digests"]["items"] == {
        "$ref": "#/$defs/assetDigestEntry"
    }
    assert definitions["mediaTypeTotals"]["properties"]["media_type"] == {
        "$ref": "#/$defs/mediaType"
    }
    assert definitions["outcomeCount"]["properties"]["outcome"]["enum"] == [
        outcome.value for outcome in ProcessingOutcome
    ]
    assert definitions["abstentionCount"]["properties"]["stage"]["enum"] == [
        stage.value for stage in AbstentionStage
    ]
    assert definitions["assetDigestEntry"]["properties"]["output_sha256"] == {
        "anyOf": [{"$ref": "#/$defs/sha256Digest"}, {"type": "null"}]
    }


def test_processing_summary_schema_accepts_serialized_summaries() -> None:
    validator = _validator("processing_summary")
    payloads = [_processing_summary_payload(), summarize_processing_run([]).to_dict()]

    for payload in payloads:
        assert validator.is_valid(payload)
        assert validator.is_valid(json.loads(json.dumps(payload)))


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({**_processing_summary_payload(), "extra": 1}, id="unknown-field"),
        pytest.param(
            _without(_processing_summary_payload(), "outcome_counts"),
            id="missing-field",
        ),
        pytest.param(
            {**_processing_summary_payload(), "total_bytes": -1}, id="negative-bytes"
        ),
        pytest.param(
            {**_processing_summary_payload(), "total_assets": "2"},
            id="assets-not-integer",
        ),
        pytest.param(
            {**_processing_summary_payload(), "schema_version": 2},
            id="unsupported-version",
        ),
        pytest.param(
            {
                **_processing_summary_payload(),
                "outcome_counts": [{"outcome": "skipped", "count": 1}],
            },
            id="unknown-outcome",
        ),
        pytest.param(
            {
                **_processing_summary_payload(),
                "abstention_counts": [
                    {"stage": "preflight", "reason": "malformed_media", "count": 1}
                ],
            },
            id="disallowed-stage-reason-pair",
        ),
        pytest.param(
            {
                **_processing_summary_payload(),
                "by_media_type": [
                    {
                        "media_type": "text/plain",
                        "count": 1,
                        "total_bytes": 1,
                        "total_pages": 0,
                        "total_frames": 0,
                    }
                ],
            },
            id="unsupported-media-type",
        ),
        pytest.param(
            {
                **_processing_summary_payload(),
                "asset_digests": [
                    {
                        "asset_id": "asset-1",
                        "input_sha256": "a" * 64,
                        "output_sha256": "C" * 64,
                    }
                ],
            },
            id="invalid-output-digest",
        ),
        pytest.param(
            {
                **_processing_summary_payload(),
                "by_media_type": [
                    {
                        "media_type": "image/png",
                        "count": 0,
                        "total_bytes": 0,
                        "total_pages": 0,
                        "total_frames": 0,
                    }
                ],
            },
            id="zero-group-count",
        ),
    ],
)
def test_processing_summary_schema_rejects_malformed_payloads(
    payload: dict[str, Any],
) -> None:
    assert not _validator("processing_summary").is_valid(payload)


def test_provider_result_schema_matches_the_python_contract() -> None:
    document = schemas.export_multimodal_schema("provider_result")
    definitions = document["$defs"]
    envelope = definitions["providerResultEnvelope"]
    properties = envelope["properties"]

    assert set(envelope["required"]) == _REQUIRED_FIELDS
    assert set(properties) == _REQUIRED_FIELDS | {
        "output_digest",
        "abstention_code",
        "count_metadata",
    }
    assert properties["schema_version"] == {"const": PROVIDER_RESULT_SCHEMA_VERSION}
    assert properties["provider_id"] == {"$ref": "#/$defs/providerIdentifier"}
    assert properties["model_id"] == {"$ref": "#/$defs/providerIdentifier"}
    assert properties["input_digest"] == {"$ref": "#/$defs/sha256Digest"}
    assert properties["outcome"]["enum"] == [
        outcome.value for outcome in ProviderResultOutcome
    ]
    assert definitions["providerAbstentionCode"] == {
        "enum": [code.value for code in ProviderAbstentionCode]
    }
    assert properties["duration_ms"] == {
        "type": "number",
        "minimum": 0,
        "maximum": _MAX_DURATION_MS,
    }
    assert properties["count_metadata"]["propertyNames"] == {
        "enum": sorted(_COUNT_FIELDS)
    }
    assert properties["count_metadata"]["additionalProperties"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": _MAX_COUNT,
    }
    assert schemas._PROVIDER_COUNT_FIELDS == tuple(sorted(_COUNT_FIELDS))
    assert schemas._PROVIDER_MAX_COUNT == _MAX_COUNT
    assert schemas._PROVIDER_MAX_DURATION_MS == _MAX_DURATION_MS
    assert schemas._PROVIDER_REQUIRED_FIELDS == tuple(sorted(_REQUIRED_FIELDS))


def test_provider_result_schema_accepts_serialized_envelopes() -> None:
    validator = _validator("provider_result")

    for payload in _provider_payloads():
        assert validator.is_valid(payload)
        assert ProviderResultEnvelope.from_dict(payload).to_dict() == payload


def test_provider_result_schema_binds_outcome_to_optional_fields() -> None:
    validator = _validator("provider_result")

    assert validator.is_valid(_provider_payload())
    assert validator.is_valid(_provider_payload(abstention_code=None))
    assert validator.is_valid(
        _provider_payload(
            outcome=ProviderResultOutcome.PROVIDER_UNAVAILABLE.value,
            output_digest=None,
        )
    )
    assert not validator.is_valid(_provider_payload(output_digest=None))
    assert not validator.is_valid(
        _provider_payload(
            outcome=ProviderResultOutcome.ABSTENTION.value,
            output_digest=None,
            abstention_code=None,
        )
    )
    assert not validator.is_valid(
        _provider_payload(
            outcome=ProviderResultOutcome.ABSTENTION.value,
            abstention_code="low_quality",
        )
    )
    assert not validator.is_valid(
        _provider_payload(
            outcome=ProviderResultOutcome.PROVIDER_UNAVAILABLE.value,
            output_digest=None,
            abstention_code="low_quality",
        )
    )


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(_provider_payload(extra="value"), id="unknown-field"),
        pytest.param(
            _provider_payload(schema_version="openmed.multimodal.provider_result.v2"),
            id="unsupported-version",
        ),
        pytest.param(_provider_payload(outcome="timeout"), id="unknown-outcome"),
        pytest.param(_provider_payload(provider_id="Local-OCR"), id="uppercase-id"),
        pytest.param(_provider_payload(provider_id="patient_id"), id="forbidden-part"),
        pytest.param(_provider_payload(provider_id="mrn"), id="forbidden-word"),
        pytest.param(_provider_payload(provider_id="a" * 129), id="oversized-id"),
        pytest.param(_provider_payload(model_id="doctr/base"), id="separator-in-id"),
        pytest.param(_provider_payload(input_digest="B" * 64), id="uppercase-digest"),
        pytest.param(_provider_payload(duration_ms=-1), id="negative-duration"),
        pytest.param(
            _provider_payload(duration_ms=_MAX_DURATION_MS + 1), id="oversized-duration"
        ),
        pytest.param(
            _provider_payload(duration_ms=float("inf")), id="infinite-duration"
        ),
        pytest.param(
            _provider_payload(count_metadata={"unknown_count": 1}),
            id="unknown-count-field",
        ),
        pytest.param(
            _provider_payload(count_metadata={"page_count": -1}), id="negative-count"
        ),
        pytest.param(
            _provider_payload(count_metadata={"page_count": _MAX_COUNT + 1}),
            id="oversized-count",
        ),
        pytest.param(
            _provider_payload(count_metadata={"page_count": 1.5}), id="fractional-count"
        ),
        pytest.param(_provider_payload(count_metadata=None), id="null-counts"),
        pytest.param(_provider_payload(count_metadata=[]), id="list-counts"),
    ],
)
def test_provider_result_schema_rejects_malformed_payloads(
    payload: dict[str, Any],
) -> None:
    assert not _validator("provider_result").is_valid(payload)
    with pytest.raises(ProviderResultError):
        ProviderResultEnvelope.from_dict(payload)


def test_schema_cannot_express_cross_record_batch_invariants() -> None:
    validator = _validator("asset_batch")
    duplicate_identifiers = {
        "batch_id": "batch-1",
        "assets": [
            _manifest_payload(byte_size=1024),
            _manifest_payload(byte_size=2048),
        ],
    }
    mismatched_aggregate = dict(_batch_payload(), asset_count=2)

    assert validator.is_valid(duplicate_identifiers)
    assert validator.is_valid(mismatched_aggregate)
    with pytest.raises(AssetBatchError):
        AssetBatch.from_dict(duplicate_identifiers)
    with pytest.raises(AssetBatchError):
        AssetBatch.from_dict(mismatched_aggregate)


def test_non_finite_numbers_stay_outside_the_exported_contract() -> None:
    provider_payload = _provider_payload(duration_ms=float("nan"))
    manifest_payload = _manifest_payload(duration_seconds=float("nan"))

    assert _validator("provider_result").is_valid(provider_payload)
    assert _validator("asset_manifest").is_valid(manifest_payload)
    with pytest.raises(ProviderResultError):
        ProviderResultEnvelope.from_dict(provider_payload)
    with pytest.raises(AssetManifestError):
        AssetManifest.from_dict(manifest_payload)
    with pytest.raises(ValueError):
        json.dumps(provider_payload, allow_nan=False)
    assert not _validator("asset_manifest").is_valid(
        _manifest_payload(duration_seconds=float("inf"))
    )
    assert not _validator("provider_result").is_valid(
        _provider_payload(duration_ms=float("inf"))
    )
