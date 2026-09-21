"""Governed dataset snapshot, split, license, and export tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema.validators import validator_for

from openmed.structured.datasets import (
    DatasetAnnotation,
    DatasetBuildSpec,
    DatasetExportAuthorization,
    DatasetExportFormat,
    DatasetExportReceipt,
    DatasetExportRequest,
    DatasetLicenseConstraint,
    DatasetPrivacyState,
    DatasetRecord,
    DatasetSelection,
    GovernedDatasetManifest,
    LocalDatasetExporter,
    RedistributionPolicy,
    build_dataset_snapshot,
    load_governed_dataset_schema,
)
from openmed.structured.store import StoreState


def _selection() -> DatasetSelection:
    return DatasetSelection.from_ingestion_job(
        job_id="job_syntheticjob0001",
        job_digest="sha256:" + "1" * 64,
        source_snapshot_id="snapshot_syntheticsource1",
        source_snapshot_digest="sha256:" + "2" * 64,
    )


def _license(
    redistribution: RedistributionPolicy = RedistributionPolicy.PERMITTED,
) -> DatasetLicenseConstraint:
    return DatasetLicenseConstraint(
        source_id="synthetic_fixture",
        license_id="Apache-2.0",
        terms_digest="sha256:" + "3" * 64,
        redistribution=redistribution,
    )


def _spec(
    *,
    selection: DatasetSelection | None = None,
    redistribution: RedistributionPolicy = RedistributionPolicy.PERMITTED,
    include_values: bool = False,
    formats: tuple[DatasetExportFormat, ...] | None = None,
) -> DatasetBuildSpec:
    return DatasetBuildSpec(
        dataset_id="dataset_syntheticdata001",
        created_at="2026-01-02T03:04:05Z",
        selection=selection or _selection(),
        query_digest="sha256:" + "4" * 64,
        policy_digest="sha256:" + "5" * 64,
        schema_digest="sha256:" + "6" * 64,
        vocabulary_digest="sha256:" + "7" * 64,
        component_versions={"dataset_builder": "1.0.0"},
        model_versions={"clinical_encoder": "synthetic-1.0"},
        licenses=(_license(redistribution),),
        formats=(
            (
                DatasetExportFormat.JSONL,
                DatasetExportFormat.PARQUET,
                DatasetExportFormat.ANNOTATION_JSONL,
            )
            if formats is None
            else formats
        ),
        include_deidentified_values=include_values,
    )


def _record(
    suffix: str,
    *,
    split: str = "train",
    patient_suffix: str | None = None,
    privacy_state: DatasetPrivacyState = DatasetPrivacyState.DEIDENTIFIED,
    values: dict[str, object] | None = None,
) -> DatasetRecord:
    token = suffix * 16
    patient_token = (patient_suffix or suffix) * 16
    return DatasetRecord(
        record_id=f"record_{token}",
        patient_key=f"patient_{patient_token}",
        split=split,
        source_artifact_ids=(f"artifact_{token}",),
        source_fact_ids=(f"fact_{token}",),
        source_event_ids=(f"event_{token}",),
        evidence_ids=(f"evidence_{token}",),
        labels=("Condition",),
        annotations=(
            DatasetAnnotation(
                annotation_id=f"annotation_{token}",
                label="Condition",
                start=2,
                end=8,
                source_digest="sha256:" + suffix * 64,
                evidence_id=f"evidence_{token}",
            ),
        ),
        values=values or {"clinical_code": "synthetic-condition"},
        privacy_state=privacy_state,
    )


def _dataset(
    records: tuple[DatasetRecord, ...] | None = None,
    *,
    spec: DatasetBuildSpec | None = None,
):
    result = build_dataset_snapshot(
        spec or _spec(),
        records or (_record("a"), _record("b", split="test")),
    )
    assert result.state is StoreState.SUCCESS
    assert result.value is not None
    return result.value


class _AuditHook:
    def __init__(
        self,
        *,
        allow: bool = True,
        fail_record: bool = False,
    ) -> None:
        self.allow = allow
        self.fail_record = fail_record
        self.requests: list[DatasetExportRequest] = []
        self.receipts: list[DatasetExportReceipt] = []

    def authorize(self, request: DatasetExportRequest) -> bool:
        self.requests.append(request)
        return self.allow

    def record(self, receipt: DatasetExportReceipt) -> None:
        if self.fail_record:
            raise RuntimeError("synthetic audit sink failure")
        self.receipts.append(receipt)


def _authorization(dataset_id: str, policy_digest: str) -> DatasetExportAuthorization:
    return DatasetExportAuthorization(
        authorization_id="authorization_synthetic0000001",
        snapshot_id=dataset_id,
        policy_digest=policy_digest,
        purpose="approved_research",
        approved_by="reviewer_syntheticreview01",
        export_approved=True,
        identified_export_approved=True,
    )


def test_manifest_round_trip_schema_and_file_digests_are_stable() -> None:
    dataset = _dataset()
    restored = GovernedDatasetManifest.from_json(dataset.manifest.to_json())

    assert restored == dataset.manifest
    assert restored.manifest_digest == dataset.manifest.snapshot.manifest_hash
    assert restored.snapshot.record_count == 2
    assert set(restored.snapshot.file_hashes) == {
        "annotations.jsonl",
        "records.jsonl",
        "records.parquet",
    }

    schema = load_governed_dataset_schema()
    validator = validator_for(schema)
    validator.check_schema(schema)
    assert not tuple(validator(schema).iter_errors(dataset.manifest.to_dict()))


@given(st.permutations(("a", "b", "c")))
def test_record_order_cannot_change_manifest_or_artifacts(order: list[str]) -> None:
    records = tuple(
        _record(item, split="train" if item != "c" else "test") for item in order
    )
    first = _dataset(records)
    second = _dataset(tuple(reversed(records)))

    assert first.manifest.to_json() == second.manifest.to_json()
    assert dict(first.files) == dict(second.files)


def test_default_exports_are_value_free_and_parquet_matches_jsonl() -> None:
    dataset = _dataset(
        (
            _record(
                "a",
                values={"clinical_code": "sensitive-placeholder-value"},
            ),
        )
    )
    jsonl = dataset.files["records.jsonl"].decode("utf-8")
    annotations = dataset.files["annotations.jsonl"].decode("utf-8")
    table = pq.read_table(pa.BufferReader(dataset.files["records.parquet"]))

    assert "sensitive-placeholder-value" not in jsonl
    assert '"values"' not in jsonl
    assert '"privacy_state":"deidentified"' in jsonl
    assert "sensitive-placeholder-value" not in annotations
    assert table.num_rows == 1
    parquet_payload = table.column("payload_json")[0].as_py()
    assert json.loads(parquet_payload) == json.loads(jsonl)


def test_deidentified_values_require_safe_keys_and_explicit_build_option() -> None:
    included = _dataset(
        (_record("a", values={"clinical_code": "synthetic-condition"}),),
        spec=_spec(include_values=True),
    )
    assert "synthetic-condition" in included.files["records.jsonl"].decode()

    unsafe = build_dataset_snapshot(
        _spec(include_values=True),
        (_record("a", values={"email": "person@example.test"}),),
    )
    assert unsafe.state is StoreState.FAILURE
    assert unsafe.code == "dataset_build_invalid"


def test_token_vault_material_is_rejected_even_when_values_are_not_exported() -> None:
    with pytest.raises(Exception, match="protected material"):
        _record("a", values={"token_vault": {"entry": "forbidden"}})


@pytest.mark.parametrize("reference", ["patient", "fact", "evidence"])
def test_split_overlap_fails_closed(reference: str) -> None:
    first = _record("a", split="train")
    if reference == "patient":
        second = _record("b", split="test", patient_suffix="a")
    elif reference == "fact":
        second = replace(
            _record("b", split="test"),
            source_fact_ids=first.source_fact_ids,
        )
    else:
        second = replace(
            _record("b", split="test"),
            evidence_ids=first.evidence_ids,
        )

    result = build_dataset_snapshot(_spec(), (first, second))
    assert result.state is StoreState.CONFLICT
    assert result.code == "dataset_split_or_digest_conflict"


def test_changed_policy_or_source_snapshot_changes_manifest_identity() -> None:
    first = _dataset()
    policy_changed = _dataset(spec=replace(_spec(), policy_digest="sha256:" + "8" * 64))
    source_changed = _dataset(
        spec=replace(
            _spec(),
            selection=DatasetSelection.from_ingestion_job(
                job_id="job_syntheticjob0001",
                job_digest="sha256:" + "1" * 64,
                source_snapshot_id="snapshot_syntheticsource2",
                source_snapshot_digest="sha256:" + "9" * 64,
            ),
        )
    )

    assert (
        first.manifest.snapshot.snapshot_id
        != policy_changed.manifest.snapshot.snapshot_id
    )
    assert (
        first.manifest.snapshot.snapshot_id
        != source_changed.manifest.snapshot.snapshot_id
    )


def test_local_export_is_idempotent_and_distribution_is_license_gated(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    exporter = LocalDatasetExporter(tmp_path / "permitted")
    first = exporter.export(dataset, distribution=True)
    second = exporter.export(dataset, distribution=True)

    assert first.state is StoreState.SUCCESS and first.created
    assert second.state is StoreState.SUCCESS and not second.created
    assert first.value is not None
    assert set(first.value.created_files) == {
        "annotations.jsonl",
        "manifest.json",
        "records.jsonl",
        "records.parquet",
    }

    changed = _dataset(
        (_record("a", values={"clinical_code": "changed-synthetic-code"}),)
    )
    conflict = exporter.export(changed)
    assert conflict.state is StoreState.CONFLICT
    assert conflict.code == "immutable_export_conflict"

    restricted = _dataset(spec=_spec(redistribution=RedistributionPolicy.RESTRICTED))
    denied = LocalDatasetExporter(tmp_path / "restricted").export(
        restricted,
        distribution=True,
    )
    local_only = LocalDatasetExporter(tmp_path / "local-only").export(restricted)
    assert denied.state is StoreState.DENIED
    assert denied.code == "license_distribution_denied"
    assert local_only.state is StoreState.SUCCESS


def test_identified_export_requires_bound_approval_and_two_phase_audit(
    tmp_path: Path,
) -> None:
    record = _record(
        "a",
        privacy_state=DatasetPrivacyState.IDENTIFIED,
        values={"name": "Synthetic Person", "clinical_code": "condition"},
    )
    dataset = _dataset((record,))
    default_text = dataset.files["records.jsonl"].decode()
    assert "Synthetic Person" not in default_text

    hook = _AuditHook()
    exporter = LocalDatasetExporter(tmp_path / "identified")
    authorization = _authorization(
        dataset.manifest.snapshot.snapshot_id,
        dataset.manifest.policy_digest,
    )
    result = exporter.export_identified(
        dataset,
        authorization=authorization,
        audit_hook=hook,
    )

    assert result.state is StoreState.SUCCESS
    assert "Synthetic Person" in (tmp_path / "identified" / "records.jsonl").read_text()
    assert len(hook.requests) == 1
    assert len(hook.receipts) == 1
    assert hook.requests[0].namespace is DatasetPrivacyState.IDENTIFIED
    assert hook.requests[0].authorization_id == authorization.authorization_id
    assert hook.requests[0].authorization_digest is not None


def test_identified_export_denial_and_audit_failure_remain_typed(
    tmp_path: Path,
) -> None:
    record = _record(
        "a",
        privacy_state=DatasetPrivacyState.IDENTIFIED,
        values={"name": "Synthetic Person"},
    )
    dataset = _dataset((record,))
    authorization = _authorization(
        dataset.manifest.snapshot.snapshot_id,
        dataset.manifest.policy_digest,
    )

    snapshot_conflict = LocalDatasetExporter(
        tmp_path / "snapshot-conflict"
    ).export_identified(
        dataset,
        authorization=replace(
            authorization,
            snapshot_id="datasetsnapshot_conflicting00001",
        ),
        audit_hook=_AuditHook(),
    )
    policy_conflict = LocalDatasetExporter(
        tmp_path / "policy-conflict"
    ).export_identified(
        dataset,
        authorization=replace(
            authorization,
            policy_digest="sha256:" + "9" * 64,
        ),
        audit_hook=_AuditHook(),
    )
    approval_denied = LocalDatasetExporter(
        tmp_path / "approval-denied"
    ).export_identified(
        dataset,
        authorization=replace(
            authorization,
            identified_export_approved=False,
        ),
        audit_hook=_AuditHook(),
    )
    denied = LocalDatasetExporter(tmp_path / "denied").export_identified(
        dataset,
        authorization=authorization,
        audit_hook=_AuditHook(allow=False),
    )
    partial = LocalDatasetExporter(tmp_path / "partial").export_identified(
        dataset,
        authorization=authorization,
        audit_hook=_AuditHook(fail_record=True),
    )

    assert snapshot_conflict.state is StoreState.CONFLICT
    assert snapshot_conflict.code == "authorization_snapshot_conflict"
    assert policy_conflict.state is StoreState.CONFLICT
    assert policy_conflict.code == "authorization_policy_conflict"
    assert approval_denied.state is StoreState.DENIED
    assert approval_denied.code == "identified_export_not_approved"
    assert denied.state is StoreState.DENIED
    assert denied.code == "audit_authorization_denied"
    assert not (tmp_path / "denied").exists()
    assert partial.state is StoreState.PARTIAL
    assert partial.code == "export_audit_record_failed"
    assert partial.value is not None


def test_manifest_tampering_is_detected() -> None:
    payload = _dataset().manifest.to_dict()
    payload["policy_digest"] = "sha256:" + "f" * 64

    with pytest.raises(Exception, match="manifest hash differs"):
        GovernedDatasetManifest.from_dict(payload)


def test_empty_ingestion_result_still_produces_reproducible_empty_files() -> None:
    result = build_dataset_snapshot(_spec(), ())

    assert result.state is StoreState.SUCCESS
    assert result.value is not None
    assert result.value.manifest.snapshot.record_count == 0
    assert result.value.files["records.jsonl"] == b""
    assert result.value.files["annotations.jsonl"] == b""
