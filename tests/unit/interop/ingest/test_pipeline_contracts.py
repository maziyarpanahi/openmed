"""Tests for safe ingestion pipeline lineage contracts and persistence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from openmed.clinical.journey_contracts import canonical_digest
from openmed.interop.ingest import (
    PIPELINE_STAGE_SCHEMA_NAMES,
    PipelineLineageError,
    PipelineStageInvalidation,
    PipelineStageManifest,
    SourceManifest,
    SQLiteIngestionStore,
    build_stage_manifest,
    load_pipeline_lineage_schema,
)
from openmed.structured.store import StoreState

T0 = "2026-01-02T03:04:05Z"
T1 = "2026-01-02T04:04:05Z"
DIGEST_A = canonical_digest({"synthetic": "a"})
DIGEST_B = canonical_digest({"synthetic": "b"})


def _source_manifest(suffix: str = "a") -> SourceManifest:
    return SourceManifest(
        manifest_id=f"manifest_{suffix * 16}",
        source_id="source_aaaaaaaaaaaaaaaa",
        artifact_digests=(DIGEST_A,),
        policy_digest=DIGEST_A,
        pipeline_digest=canonical_digest({"pipeline": suffix}),
        created_at=T0,
    )


def _stage(
    job_id: str,
    stage: str = "source_adaptation",
    *,
    parents: tuple[str, ...] = (),
) -> PipelineStageManifest:
    return build_stage_manifest(
        job_id=job_id,
        stage=stage,
        state="success",
        input_digests=(DIGEST_A,),
        output_digests=(DIGEST_B,),
        input_record_ids=("source_aaaaaaaaaaaaaaaa",),
        output_record_ids=("artifact_aaaaaaaaaaaaaaaa",),
        component="openmed.pipeline.synthetic",
        component_version="1.0.0",
        policy_digest=DIGEST_A,
        parent_stage_manifest_ids=parents,
        recorded_at=T0,
    )


def test_stage_manifest_round_trip_and_bundled_schemas() -> None:
    stage = _stage("job_aaaaaaaaaaaaaaaa")
    invalidation = PipelineStageInvalidation(
        invalidation_id="invalidation_aaaaaaaaaaaaaaaa",
        job_id=stage.job_id,
        stage_manifest_id=stage.stage_manifest_id,
        replacement_job_id="job_bbbbbbbbbbbbbbbb",
        reason_code="stage_reprocessed",
        recorded_at=T1,
    )
    records = {
        "pipeline_stage": stage,
        "stage_invalidation": invalidation,
    }

    assert set(records) == set(PIPELINE_STAGE_SCHEMA_NAMES)
    for name, record in records.items():
        restored = type(record).from_json(record.to_json())
        assert restored == record
        schema = load_pipeline_lineage_schema(name)
        validator = validator_for(schema)
        validator.check_schema(schema)
        assert not tuple(validator(schema).iter_errors(record.to_dict()))


def test_stage_manifest_rejects_raw_or_unknown_metadata() -> None:
    stage = _stage("job_aaaaaaaaaaaaaaaa")
    payload = stage.to_dict() | {"source_text": "synthetic-canary"}

    with pytest.raises(PipelineLineageError, match="fields are invalid"):
        PipelineStageManifest.from_dict(payload)
    with pytest.raises(PipelineLineageError, match="reason code"):
        replace(stage, reason_code="must_not_exist")


def test_store_persists_edges_and_invalidates_only_descendants(
    tmp_path: Path,
) -> None:
    store = SQLiteIngestionStore(tmp_path / "lineage.sqlite3")
    first = store.register_manifest(_source_manifest("a"), recorded_at=T0)
    second = store.register_manifest(_source_manifest("b"), recorded_at=T1)
    assert first.ok and first.value is not None
    assert second.ok and second.value is not None
    first_job = first.value.job.job_id
    second_job = second.value.job.job_id

    adaptation = _stage(first_job)
    privacy = _stage(
        first_job,
        "privacy_policy",
        parents=(adaptation.stage_manifest_id,),
    )
    routing = _stage(
        first_job,
        "document_routing",
        parents=(privacy.stage_manifest_id,),
    )
    for manifest in (adaptation, privacy, routing):
        persisted = store.put_pipeline_stage(manifest)
        assert persisted.ok and persisted.created
        replay = store.put_pipeline_stage(manifest)
        assert replay.ok and not replay.created

    invalidated = store.invalidate_pipeline_descendants(
        first_job,
        from_stage_manifest_id=privacy.stage_manifest_id,
        replacement_job_id=second_job,
        recorded_at=T1,
    )

    assert invalidated.ok and invalidated.value is not None
    assert {item.stage_manifest_id for item in invalidated.value} == {
        privacy.stage_manifest_id,
        routing.stage_manifest_id,
    }
    assert adaptation.stage_manifest_id not in {
        item.stage_manifest_id for item in invalidated.value
    }
    assert store.list_pipeline_stages(first_job).value == (
        adaptation,
        privacy,
        routing,
    )
    assert store.list_pipeline_invalidations(first_job).value == invalidated.value
    assert store.ingestion_integrity_check().state is StoreState.SUCCESS
    store.close()


def test_stage_parent_must_exist(tmp_path: Path) -> None:
    store = SQLiteIngestionStore(tmp_path / "lineage.sqlite3")
    registration = store.register_manifest(_source_manifest(), recorded_at=T0)
    assert registration.ok and registration.value is not None
    stage = _stage(
        registration.value.job.job_id,
        "privacy_policy",
        parents=("stage_missing000000000",),
    )

    result = store.put_pipeline_stage(stage)

    assert result.state is StoreState.PARTIAL
    assert result.code == "stage_parent_missing"
    store.close()
