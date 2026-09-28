"""Tests for versioned ingestion control-plane contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema.validators import validator_for

from openmed.clinical.journey_contracts import canonical_digest
from openmed.interop.ingest import (
    INGESTION_SCHEMA_NAMES,
    INGESTION_SCHEMA_VERSION,
    Cancellation,
    Checkpoint,
    IngestionContractError,
    IngestionJob,
    Lease,
    QuarantinePromotion,
    QuarantineResult,
    ReplayAudit,
    Retry,
    SourceManifest,
    load_all_ingestion_schemas,
    load_ingestion_schema,
)

RECORDED = "2026-01-02T03:04:05Z"
LATER = "2026-01-02T04:04:05Z"
DIGEST_A = canonical_digest({"synthetic": "a"})
DIGEST_B = canonical_digest({"synthetic": "b"})


def _manifest(digests: tuple[str, ...] = (DIGEST_A,)) -> SourceManifest:
    return SourceManifest(
        manifest_id="manifest_aaaaaaaaaaaaaaaa",
        source_id="source_aaaaaaaaaaaaaaaa",
        artifact_digests=digests,
        policy_digest=canonical_digest({"policy": "synthetic"}),
        pipeline_digest=canonical_digest({"pipeline": "synthetic"}),
        created_at=RECORDED,
    )


def _records() -> tuple[object, ...]:
    manifest = _manifest()
    job = IngestionJob(
        job_id="job_aaaaaaaaaaaaaaaa",
        manifest_digest=manifest.manifest_digest,
        state="running",
        checkpoint_sequence=1,
        created_at=RECORDED,
        updated_at=LATER,
    )
    return (
        manifest,
        job,
        Checkpoint(
            checkpoint_id="checkpoint_aaaaaaaaaaaaaaaa",
            job_id=job.job_id,
            manifest_digest=manifest.manifest_digest,
            step="parse",
            sequence=1,
            input_digest=DIGEST_A,
            output_digest=DIGEST_B,
            completed_at=LATER,
            committed_revision=2,
        ),
        Lease(
            lease_id="lease_aaaaaaaaaaaaaaaa",
            job_id=job.job_id,
            worker_id="worker_aaaaaaaaaaaaaaaa",
            epoch=1,
            acquired_at=RECORDED,
            expires_at=LATER,
        ),
        Retry(
            retry_id="retry_aaaaaaaaaaaaaaaa",
            job_id=job.job_id,
            classification="transient",
            reason_code="operation_timeout",
            attempt=1,
            recorded_at=RECORDED,
            retry_after=LATER,
        ),
        Cancellation(
            cancellation_id="cancellation_aaaaaaaaaaaaaaaa",
            job_id=job.job_id,
            actor_digest=DIGEST_A,
            reason_code="owner_requested",
            requested_at=LATER,
        ),
        QuarantineResult(
            quarantine_id="quarantine_aaaaaaaaaaaaaaaa",
            job_id=job.job_id,
            manifest_digest=manifest.manifest_digest,
            classification="partial",
            reason_code="partial_parse",
            candidate_count=2,
            failure_count=1,
            created_at=LATER,
            output_digest=DIGEST_B,
        ),
        QuarantinePromotion(
            promotion_id="promotion_aaaaaaaaaaaaaaaa",
            quarantine_id="quarantine_aaaaaaaaaaaaaaaa",
            reviewer_digest=DIGEST_A,
            evidence_digest=DIGEST_B,
            promoted_at=LATER,
        ),
        ReplayAudit(
            replay_id="replay_aaaaaaaaaaaaaaaa",
            manifest_digest=manifest.manifest_digest,
            job_id=job.job_id,
            action="noop",
            recorded_at=LATER,
        ),
    )


@pytest.mark.parametrize("record", _records())
def test_all_contracts_have_canonical_strict_round_trip(record: object) -> None:
    record_type = type(record)
    encoded = record.to_json()  # type: ignore[attr-defined]
    restored = record_type.from_json(encoded)

    assert restored == record
    assert restored.to_json() == encoded
    assert restored.to_dict()["schema_version"] == INGESTION_SCHEMA_VERSION

    payload = restored.to_dict()
    payload["unknown"] = True
    with pytest.raises(IngestionContractError, match="missing or unknown"):
        record_type.from_dict(payload)


def test_public_ingestion_records_validate_against_bundled_schemas() -> None:
    records = _records()
    record_by_schema = {
        "source_manifest": records[0],
        "ingestion_job": records[1],
        "checkpoint": records[2],
        "lease": records[3],
        "retry": records[4],
        "cancellation": records[5],
        "quarantine_result": records[6],
    }
    schemas = load_all_ingestion_schemas()

    assert set(schemas) == set(INGESTION_SCHEMA_NAMES)
    for name, record in record_by_schema.items():
        schema = schemas[name]
        validator = validator_for(schema)
        validator.check_schema(schema)

        assert schema["schema_version"] == 1
        assert not tuple(validator(schema).iter_errors(record.to_dict()))


def test_ingestion_schema_loader_rejects_unknown_names() -> None:
    assert load_ingestion_schema("ingestion_job")["title"] == "OpenMedIngestionJob"
    with pytest.raises(KeyError, match="unknown ingestion schema"):
        load_ingestion_schema("unknown")


def test_json_rejects_duplicate_keys_nonfinite_values_and_arrays() -> None:
    with pytest.raises(IngestionContractError, match="invalid"):
        SourceManifest.from_json('{"manifest_id":"x","manifest_id":"y"}')
    with pytest.raises(IngestionContractError, match="invalid"):
        SourceManifest.from_json('{"manifest_id":NaN}')
    with pytest.raises(IngestionContractError, match="object"):
        SourceManifest.from_json("[]")


def test_manifest_identity_ignores_record_id_time_and_input_order() -> None:
    first = _manifest((DIGEST_B, DIGEST_A))
    second = replace(
        first,
        manifest_id="manifest_bbbbbbbbbbbbbbbb",
        created_at=LATER,
        artifact_digests=(DIGEST_A, DIGEST_B),
    )

    assert first.artifact_digests == (DIGEST_A, DIGEST_B)
    assert first.manifest_digest == second.manifest_digest
    assert first.to_json() != second.to_json()


@given(
    st.lists(st.sampled_from((DIGEST_A, DIGEST_B)), min_size=1, max_size=2, unique=True)
)
def test_manifest_canonicalization_property(digests: list[str]) -> None:
    manifest = _manifest(tuple(digests))
    restored = SourceManifest.from_json(manifest.to_json())

    assert restored == manifest
    assert restored.artifact_digests == tuple(sorted(digests))
    assert restored.manifest_digest == manifest.manifest_digest


def test_contract_errors_and_reprs_do_not_echo_invalid_values() -> None:
    canary = "raw-clinical-canary-must-not-echo"
    with pytest.raises(IngestionContractError) as captured:
        replace(_manifest(), manifest_id=canary)

    assert canary not in str(captured.value)
    assert canary not in repr(captured.value)


def test_invalid_states_and_time_bounds_are_rejected() -> None:
    records = _records()
    with pytest.raises(IngestionContractError, match="job state"):
        replace(records[1], state="maybe")
    with pytest.raises(IngestionContractError, match="expires_at"):
        replace(records[3], expires_at=RECORDED)
    with pytest.raises(IngestionContractError, match="retry classification"):
        replace(records[4], classification="maybe")
    with pytest.raises(IngestionContractError, match="quarantine classification"):
        replace(records[6], classification="maybe")
