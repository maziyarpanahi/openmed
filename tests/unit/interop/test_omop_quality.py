"""Tests for digest-bound OMOP quality and reconciliation reports."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema import Draft202012Validator

from openmed.clinical.journey_contracts import ClinicalFact
from openmed.eval.suites.omop_quality import load_frozen_omop_quality_fixture
from openmed.interop.omop.fact_projection import (
    OmopConceptMapping,
    OmopFactProjection,
    OmopFactProjectionInput,
    OmopVocabularySnapshot,
    assess_omop_fact_round_trip,
    project_clinical_facts_to_omop,
)
from openmed.interop.omop.quality import (
    OmopProjectionAggregate,
    OmopQualityCheck,
    OmopQualityConflictError,
    OmopQualityDeniedError,
    OmopQualityReport,
    OmopReconciliation,
    build_omop_quality_tool_output,
    load_omop_quality_report_schema,
    normalize_omop_quality_output,
    projection_quality_checks,
    reconcile_omop_aggregates,
    run_omop_quality_remote,
    run_omop_quality_subprocess,
    verify_omop_quality_report,
)
from openmed.structured.store import StoreState

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "fixtures" / "interop" / "omop"
)
PROJECTION_FIXTURE = FIXTURE_DIRECTORY / "fact_projection.json"
QUALITY_FIXTURE = FIXTURE_DIRECTORY / "quality_reconciliation.json"
REFERENCE_SUMMARY = FIXTURE_DIRECTORY / "synthea_reference_omop_54_summary.json"
SIGNING_KEY = b"synthetic-omop-quality-signing-key"


def _projection_inputs() -> tuple[OmopFactProjectionInput, ...]:
    payload = json.loads(PROJECTION_FIXTURE.read_text(encoding="utf-8"))
    result = []
    for raw in payload["facts"]:
        record = dict(raw)
        mapping = OmopConceptMapping.from_dict(record.pop("mapping"))
        result.append(
            OmopFactProjectionInput(
                fact=ClinicalFact.from_dict(record),
                source_key=payload["source_key"],
                source_revision=payload["source_revision"],
                mapping=mapping,
                dataset_split=payload["dataset_split"],
            )
        )
    return tuple(result)


def _projection() -> OmopFactProjection:
    payload = json.loads(PROJECTION_FIXTURE.read_text(encoding="utf-8"))
    result = project_clinical_facts_to_omop(
        _projection_inputs(),
        vocabulary_snapshot=OmopVocabularySnapshot.from_dict(
            payload["vocabulary_snapshot"]
        ),
        etl_version=payload["etl_version"],
        occurred_at=payload["occurred_at"],
    )
    assert result.ok and result.value is not None
    return result.value


def _quality_checks() -> tuple[OmopQualityCheck, ...]:
    return (
        OmopQualityCheck(
            check_id="external.cdm.conformance",
            category="conformance",
            status="pass",
            severity="error",
            code="cdm_conformance_passed",
            remediation_code="repair_cdm_conformance",
            affected_rows=0,
        ),
        OmopQualityCheck(
            check_id="external.cdm.completeness",
            category="completeness",
            status="pass",
            severity="warning",
            code="cdm_completeness_passed",
            remediation_code="review_missing_values",
            affected_rows=0,
        ),
        OmopQualityCheck(
            check_id="external.cdm.plausibility",
            category="plausibility",
            status="pass",
            severity="error",
            code="cdm_plausibility_passed",
            remediation_code="review_implausible_values",
            affected_rows=0,
        ),
    )


def _tool_output(*, mode: str = "local") -> dict[str, object]:
    fixture = load_frozen_omop_quality_fixture(QUALITY_FIXTURE)
    return build_omop_quality_tool_output(
        fixture.quality_input,
        _quality_checks(),
        tool_name="synthetic.quality_adapter",
        tool_version="1.0.0",
        execution_mode=mode,
    )


def test_frozen_cohort_and_reference_snapshot_are_digest_bound() -> None:
    fixture = load_frozen_omop_quality_fixture(QUALITY_FIXTURE)

    cohort_digest = (
        f"sha256:{hashlib.sha256(PROJECTION_FIXTURE.read_bytes()).hexdigest()}"
    )
    reference_digest = (
        f"sha256:{hashlib.sha256(REFERENCE_SUMMARY.read_bytes()).hexdigest()}"
    )
    assert fixture.quality_input.cohort_digest == cohort_digest
    assert fixture.quality_input.reference_snapshot_digest == reference_digest
    assert fixture.openmed.snapshot_digest == _projection().digest
    assert fixture.reconcile().expectations_met
    assert {
        item.table: item.delta for item in fixture.reconcile().row_deltas
    } == fixture.expected_table_deltas


def test_projection_checks_cover_references_round_trip_mapping_and_dates() -> None:
    projection = _projection()
    round_trip = assess_omop_fact_round_trip(_projection_inputs(), projection)

    checks = projection_quality_checks(projection, round_trip=round_trip)

    assert {item.check_id for item in checks} == {
        "openmed.projection.event_dates",
        "openmed.projection.mapping_coverage",
        "openmed.projection.referential_integrity",
        "openmed.projection.round_trip",
    }
    assert all(item.status == "pass" for item in checks)
    aggregate = OmopProjectionAggregate.from_projection(
        projection,
        cohort_digest=(
            f"sha256:{hashlib.sha256(PROJECTION_FIXTURE.read_bytes()).hexdigest()}"
        ),
    )
    assert aggregate.row_counts == projection.summary.row_counts
    assert aggregate.mapped_coverage_ppm == 1_000_000


def test_normalized_report_is_deterministic_signed_and_schema_valid() -> None:
    fixture = load_frozen_omop_quality_fixture(QUALITY_FIXTURE)
    report = normalize_omop_quality_output(
        _tool_output(),
        quality_input=fixture.quality_input,
        reconciliation=fixture.reconcile(),
        signing_key=SIGNING_KEY,
        expected_execution_mode="local",
    )

    assert report.quality_verdict == "pass"
    assert not report.requires_review
    assert verify_omop_quality_report(report, SIGNING_KEY)
    restored = OmopQualityReport.from_json(report.to_json())
    assert restored == report
    assert verify_omop_quality_report(restored, SIGNING_KEY)
    schema = load_omop_quality_report_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report.to_dict())


def test_input_output_and_report_tampering_are_rejected() -> None:
    fixture = load_frozen_omop_quality_fixture(QUALITY_FIXTURE)
    output = _tool_output()
    output["input_digest"] = "sha256:" + "f" * 64

    with pytest.raises(OmopQualityConflictError):
        normalize_omop_quality_output(
            output,
            quality_input=fixture.quality_input,
            reconciliation=fixture.reconcile(),
            signing_key=SIGNING_KEY,
        )

    report = normalize_omop_quality_output(
        _tool_output(),
        quality_input=fixture.quality_input,
        reconciliation=fixture.reconcile(),
        signing_key=SIGNING_KEY,
    )
    tampered = report.to_dict()
    tampered["tool_version"] = "1.0.1"
    with pytest.raises(OmopQualityConflictError):
        OmopQualityReport.from_dict(tampered)
    assert not verify_omop_quality_report(report, b"different-synthetic-signing-key")


def test_reconciliation_rejects_split_leakage_and_non_permissive_reference() -> None:
    fixture = load_frozen_omop_quality_fixture(QUALITY_FIXTURE)

    with pytest.raises(OmopQualityConflictError):
        reconcile_omop_aggregates(
            fixture.openmed,
            replace(fixture.reference, cohort_split="train"),
        )
    with pytest.raises(OmopQualityDeniedError):
        reconcile_omop_aggregates(
            fixture.openmed,
            replace(fixture.reference, license="restricted"),
        )


def test_remote_runner_surfaces_failure_partial_conflict_and_denied_states() -> None:
    fixture = load_frozen_omop_quality_fixture(QUALITY_FIXTURE)

    success = run_omop_quality_remote(
        fixture.quality_input,
        runner=lambda _request: _tool_output(mode="remote"),
        reconciliation=fixture.reconcile(),
        signing_key=SIGNING_KEY,
    )
    assert success.ok and success.value is not None

    failed_check = replace(_quality_checks()[0], status="fail", affected_rows=1)
    failed_output = build_omop_quality_tool_output(
        fixture.quality_input,
        (failed_check, *_quality_checks()[1:]),
        tool_name="synthetic.quality_adapter",
        tool_version="1.0.0",
        execution_mode="remote",
    )
    failed = run_omop_quality_remote(
        fixture.quality_input,
        runner=lambda _request: failed_output,
        reconciliation=fixture.reconcile(),
        signing_key=SIGNING_KEY,
    )
    assert failed.state is StoreState.FAILURE
    assert failed.code == "quality_checks_failed"
    assert failed.value is not None and failed.value.requires_review

    partial_output = build_omop_quality_tool_output(
        fixture.quality_input,
        _quality_checks()[:1],
        tool_name="synthetic.quality_adapter",
        tool_version="1.0.0",
        execution_mode="remote",
    )
    partial = run_omop_quality_remote(
        fixture.quality_input,
        runner=lambda _request: partial_output,
        reconciliation=fixture.reconcile(),
        signing_key=SIGNING_KEY,
    )
    assert partial.state is StoreState.PARTIAL
    assert partial.code == "quality_report_partial"

    conflicted_output = _tool_output(mode="remote")
    conflicted_output["output_digest"] = "sha256:" + "0" * 64
    conflicted = run_omop_quality_remote(
        fixture.quality_input,
        runner=lambda _request: conflicted_output,
        reconciliation=fixture.reconcile(),
        signing_key=SIGNING_KEY,
    )
    assert conflicted.state is StoreState.CONFLICT

    runner_called = False

    def unexpected_runner(_request: object) -> dict[str, object]:
        nonlocal runner_called
        runner_called = True
        return _tool_output(mode="remote")

    preflight_conflict = run_omop_quality_remote(
        replace(
            fixture.quality_input,
            projection_digest="sha256:" + "e" * 64,
        ),
        runner=unexpected_runner,
        reconciliation=fixture.reconcile(),
        signing_key=SIGNING_KEY,
    )
    assert preflight_conflict.state is StoreState.CONFLICT
    assert not runner_called

    denied = run_omop_quality_remote(
        replace(fixture.quality_input, reference_license="restricted"),
        runner=lambda _request: _tool_output(mode="remote"),
        reconciliation=fixture.reconcile(),
        signing_key=SIGNING_KEY,
    )
    assert denied.state is StoreState.DENIED

    unknown_output = build_omop_quality_tool_output(
        fixture.quality_input,
        (),
        tool_name="synthetic.quality_adapter",
        tool_version="1.0.0",
        execution_mode="remote",
    )
    unknown = run_omop_quality_remote(
        fixture.quality_input,
        runner=lambda _request: unknown_output,
        reconciliation=fixture.reconcile(),
        signing_key=SIGNING_KEY,
    )
    assert unknown.state is StoreState.UNKNOWN

    unsupported_output = _tool_output(mode="remote")
    unsupported_output["artifact_type"] = "unsupported.quality_output"
    unsupported = run_omop_quality_remote(
        fixture.quality_input,
        runner=lambda _request: unsupported_output,
        reconciliation=fixture.reconcile(),
        signing_key=SIGNING_KEY,
    )
    assert unsupported.state is StoreState.UNSUPPORTED


def test_local_subprocess_is_optional_bounded_and_does_not_inherit_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = load_frozen_omop_quality_fixture(QUALITY_FIXTURE)
    adapter = tmp_path / "aggregate_adapter.py"
    adapter.write_text(
        "import json, os, sys\n"
        "if os.getenv('OPENMED_TEST_SECRET'):\n"
        "    raise SystemExit(9)\n"
        f"sys.stdout.write({json.dumps(json.dumps(_tool_output()))})\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENMED_TEST_SECRET", "must-not-cross-boundary")

    result = run_omop_quality_subprocess(
        fixture.quality_input,
        command=(sys.executable, os.fspath(adapter)),
        reconciliation=fixture.reconcile(),
        signing_key=SIGNING_KEY,
    )

    assert result.ok and result.value is not None
    assert result.value.execution_mode == "local"


def test_reconciliation_drift_is_a_failed_reviewable_report() -> None:
    fixture = load_frozen_omop_quality_fixture(QUALITY_FIXTURE)
    drift = reconcile_omop_aggregates(
        fixture.openmed,
        replace(
            fixture.reference,
            row_counts=dict(fixture.reference.row_counts) | {"condition_occurrence": 2},
        ),
        expected_table_deltas=fixture.expected_table_deltas,
        expected_mapped_coverage_delta_ppm=0,
    )
    output = _tool_output(mode="remote")

    result = run_omop_quality_remote(
        fixture.quality_input,
        runner=lambda _request: output,
        reconciliation=drift,
        signing_key=SIGNING_KEY,
    )

    assert result.state is StoreState.FAILURE
    assert result.code == "quality_checks_failed"
    assert result.value is not None
    assert not result.value.reconciliation.expectations_met
    assert result.value.requires_review


def test_local_subprocess_rejects_oversized_output_before_timeout() -> None:
    fixture = load_frozen_omop_quality_fixture(QUALITY_FIXTURE)
    result = run_omop_quality_subprocess(
        fixture.quality_input,
        command=(
            sys.executable,
            "-c",
            "import sys,time; sys.stdout.buffer.write(b'x' * 1000001); "
            "sys.stdout.flush(); time.sleep(30)",
        ),
        reconciliation=fixture.reconcile(),
        signing_key=SIGNING_KEY,
        timeout=5,
    )
    assert result.state is StoreState.FAILURE
    assert result.code == "quality_output_invalid"
    assert result.value is None


def test_local_subprocess_timeout_is_typed_without_output() -> None:
    fixture = load_frozen_omop_quality_fixture(QUALITY_FIXTURE)
    result = run_omop_quality_subprocess(
        fixture.quality_input,
        command=(sys.executable, "-c", "import time; time.sleep(30)"),
        reconciliation=fixture.reconcile(),
        signing_key=SIGNING_KEY,
        timeout=0.1,
    )
    assert result.state is StoreState.FAILURE
    assert result.code == "quality_adapter_timeout"
    assert result.value is None


@given(
    openmed_rows=st.lists(
        st.integers(min_value=0, max_value=1_000), min_size=9, max_size=9
    ),
    reference_rows=st.lists(
        st.integers(min_value=0, max_value=1_000), min_size=9, max_size=9
    ),
    mapping_counts=st.lists(
        st.integers(min_value=0, max_value=1_000), min_size=4, max_size=4
    ),
)
def test_reconciliation_property_preserves_every_signed_delta(
    openmed_rows: list[int],
    reference_rows: list[int],
    mapping_counts: list[int],
) -> None:
    fixture = load_frozen_omop_quality_fixture(QUALITY_FIXTURE)
    tables = tuple(fixture.openmed.row_counts)
    mapping_states = tuple(fixture.openmed.mapping_counts)
    mappings = dict(zip(mapping_states, mapping_counts, strict=True))
    fact_count = sum(mapping_counts)
    openmed = replace(
        fixture.openmed,
        row_counts=dict(zip(tables, openmed_rows, strict=True)),
        mapping_counts=mappings,
        current_fact_count=fact_count,
    )
    reference = replace(
        fixture.reference,
        row_counts=dict(zip(tables, reference_rows, strict=True)),
        mapping_counts=mappings,
        current_fact_count=fact_count,
    )
    expected = {
        table: openmed_rows[index] - reference_rows[index]
        for index, table in enumerate(tables)
    }

    reconciliation = reconcile_omop_aggregates(
        openmed,
        reference,
        expected_table_deltas=expected,
        expected_mapped_coverage_delta_ppm=0,
    )

    assert reconciliation.expectations_met
    assert {item.table: item.delta for item in reconciliation.row_deltas} == expected
    assert OmopReconciliation.from_dict(reconciliation.to_dict()) == reconciliation
