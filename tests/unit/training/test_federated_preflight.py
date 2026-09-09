import pytest
from datetime import datetime, timezone
from openmed.core.repro_hash import compute_environment_lock_digest
from openmed.training.federated_preflight import (
    FEDERATED_PREFLIGHT_SCHEMA_VERSION,
    FederatedPreflightFinding,
    FederatedPreflightReport,
    FederatedPreflightStatus,
    check_federated_schedule,
    check_federated_environment,
    check_federated_metric_schema,
    check_federated_update_schema,
    run_federated_preflight,
)
from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
    FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
)
from openmed.training import (
    FederatedMetricKind,
    FederatedPrivacyMechanism,
    build_federated_metric_envelope,
)
from openmed.training.federated_schedule import FederatedRoundSchedule

def test_empty_preflight_is_eligible() -> None:
    report = FederatedPreflightReport.from_findings([])

    assert report.status is FederatedPreflightStatus.ELIGIBLE
    assert report.findings == ()
    assert report.digest_refs == ()
    assert report.schema_version == FEDERATED_PREFLIGHT_SCHEMA_VERSION


def test_blocked_finding_dominates_review_and_eligible() -> None:
    findings = [
        FederatedPreflightFinding(
            check="schedule",
            status=FederatedPreflightStatus.REVIEW_REQUIRED,
            reason_code="SCHEDULE_REVIEW",
        ),
        FederatedPreflightFinding(
            check="privacy",
            status=FederatedPreflightStatus.BLOCKED,
            reason_code="PRIVACY_MECHANISM_MISSING",
        ),
        FederatedPreflightFinding(
            check="manifest",
            status=FederatedPreflightStatus.ELIGIBLE,
            reason_code="MANIFEST_VALID",
        ),
    ]

    report = FederatedPreflightReport.from_findings(findings)

    assert report.status is FederatedPreflightStatus.BLOCKED
    assert len(report.findings) == 3


def test_review_required_dominates_eligible() -> None:
    findings = [
        FederatedPreflightFinding(
            check="manifest",
            status=FederatedPreflightStatus.ELIGIBLE,
            reason_code="MANIFEST_VALID",
        ),
        FederatedPreflightFinding(
            check="charter",
            status=FederatedPreflightStatus.REVIEW_REQUIRED,
            reason_code="CHARTER_REVIEW",
        ),
    ]

    report = FederatedPreflightReport.from_findings(findings)

    assert report.status is FederatedPreflightStatus.REVIEW_REQUIRED


def test_findings_are_ordered_deterministically() -> None:
    finding_a = FederatedPreflightFinding(
        check="privacy",
        status=FederatedPreflightStatus.BLOCKED,
        reason_code="PRIVACY_MECHANISM_MISSING",
    )
    finding_b = FederatedPreflightFinding(
        check="schedule",
        status=FederatedPreflightStatus.REVIEW_REQUIRED,
        reason_code="SCHEDULE_REVIEW",
    )

    report_one = FederatedPreflightReport.from_findings(
        [finding_a, finding_b]
    )
    report_two = FederatedPreflightReport.from_findings(
        [finding_b, finding_a]
    )

    assert report_one.findings == report_two.findings
    assert report_one.to_json() == report_two.to_json()


def test_json_output_is_byte_stable() -> None:
    findings = [
        FederatedPreflightFinding(
            check="privacy",
            status=FederatedPreflightStatus.BLOCKED,
            reason_code="PRIVACY_MECHANISM_MISSING",
            reference="sha256:privacy-contract",
        ),
        FederatedPreflightFinding(
            check="schedule",
            status=FederatedPreflightStatus.ELIGIBLE,
            reason_code="SCHEDULE_VALID",
            reference="sha256:schedule",
        ),
    ]

    report = FederatedPreflightReport.from_findings(
        findings,
        digest_refs=[
            "sha256:" + "f" * 64,
            "sha256:" + "1" * 64,
        ],
    )

    expected = report.to_json()

    assert report.to_json() == expected
    assert report.to_json().endswith("\n")
    assert '"status": "blocked"' in expected
    assert '"digest_refs": [' in expected


def test_digest_references_are_deterministically_ordered() -> None:
    digest_a = "sha256:" + "1" * 64
    digest_m = "sha256:" + "8" * 64
    digest_z = "sha256:" + "f" * 64

    report = FederatedPreflightReport.from_findings(
        [],
        digest_refs=[
            digest_z,
            digest_a,
            digest_m,
        ],
    )

    assert report.digest_refs == (
        digest_a,
        digest_m,
        digest_z,
    )

def test_report_rejects_invalid_digest_reference() -> None:
    with pytest.raises(ValueError):
        FederatedPreflightReport.from_findings(
            (),
            digest_refs=("not-a-digest",),
        )


def test_report_rejects_non_canonical_digest_reference() -> None:
    digest = "sha256:" + "A" * 64

    with pytest.raises(ValueError):
        FederatedPreflightReport.from_findings(
            (),
            digest_refs=(digest,),
        )


def test_report_rejects_duplicate_digest_references() -> None:
    digest = "sha256:" + "a" * 64

    with pytest.raises(ValueError):
        FederatedPreflightReport.from_findings(
            (),
            digest_refs=(digest, digest),
        )


def test_report_accepts_canonical_sha256_digest_reference() -> None:
    digest = "sha256:" + "a" * 64

    report = FederatedPreflightReport.from_findings(
        (),
        digest_refs=(digest,),
    )

    assert report.digest_refs == (digest,)

def test_finding_rejects_empty_check() -> None:
    try:
        FederatedPreflightFinding(
            check="",
            status=FederatedPreflightStatus.BLOCKED,
            reason_code="INVALID",
        )
    except ValueError as exc:
        assert str(exc) == "check must not be empty"
    else:
        raise AssertionError("expected ValueError")


def test_finding_rejects_empty_reason_code() -> None:
    try:
        FederatedPreflightFinding(
            check="privacy",
            status=FederatedPreflightStatus.BLOCKED,
            reason_code="",
        )
    except ValueError as exc:
        assert str(exc) == "reason_code must not be empty"
    else:
        raise AssertionError("expected ValueError")


def test_report_rejects_status_that_does_not_match_findings() -> None:
    finding = FederatedPreflightFinding(
        check="privacy",
        status=FederatedPreflightStatus.BLOCKED,
        reason_code="PRIVACY_MECHANISM_MISSING",
    )

    try:
        FederatedPreflightReport(
            status=FederatedPreflightStatus.ELIGIBLE,
            findings=(finding,),
        )
    except ValueError as exc:
        assert str(exc) == "report status must be derived from finding statuses"
    else:
        raise AssertionError("expected ValueError")

def test_preflight_runner_preserves_independent_check_results() -> None:
    checks = [
        FederatedPreflightFinding(
            check="schedule",
            status=FederatedPreflightStatus.ELIGIBLE,
            reason_code="SCHEDULE_VALID",
            reference="sha256:schedule",
        ),
        FederatedPreflightFinding(
            check="privacy",
            status=FederatedPreflightStatus.BLOCKED,
            reason_code="PRIVACY_MECHANISM_MISSING",
            reference="sha256:privacy",
        ),
    ]

    report = run_federated_preflight(checks)

    assert report.status is FederatedPreflightStatus.BLOCKED
    assert len(report.findings) == 2
    assert {
        finding.check
        for finding in report.findings
    } == {"schedule", "privacy"}


def test_missing_mandatory_check_fails_closed() -> None:
    checks = [
        FederatedPreflightFinding(
            check="schedule",
            status=FederatedPreflightStatus.ELIGIBLE,
            reason_code="SCHEDULE_VALID",
        ),
    ]

    report = run_federated_preflight(
        checks,
        mandatory_checks=[
            "schedule",
            "privacy",
            "environment",
        ],
    )

    assert report.status is FederatedPreflightStatus.BLOCKED

    missing = [
        finding
        for finding in report.findings
        if finding.reason_code == "MANDATORY_CHECK_MISSING"
    ]

    assert {finding.check for finding in missing} == {
        "privacy",
        "environment",
    }


def test_preflight_runner_is_deterministic_for_check_order() -> None:
    check_a = FederatedPreflightFinding(
        check="privacy",
        status=FederatedPreflightStatus.BLOCKED,
        reason_code="PRIVACY_MECHANISM_MISSING",
    )
    check_b = FederatedPreflightFinding(
        check="schedule",
        status=FederatedPreflightStatus.REVIEW_REQUIRED,
        reason_code="SCHEDULE_REVIEW",
    )

    report_one = run_federated_preflight([check_a, check_b])
    report_two = run_federated_preflight([check_b, check_a])

    assert report_one.to_json() == report_two.to_json()


def _schedule() -> FederatedRoundSchedule:
    return FederatedRoundSchedule(
        enrollment_starts_at=datetime(2026, 9, 9, 10, 0, tzinfo=timezone.utc),
        update_submission_starts_at=datetime(
            2026, 9, 9, 12, 0, tzinfo=timezone.utc
        ),
        aggregation_starts_at=datetime(
            2026, 9, 9, 14, 0, tzinfo=timezone.utc
        ),
        evaluation_starts_at=datetime(
            2026, 9, 9, 16, 0, tzinfo=timezone.utc
        ),
        finishes_at=datetime(2026, 9, 9, 18, 0, tzinfo=timezone.utc),
    )


def test_schedule_check_passes_during_enrollment() -> None:
    finding = check_federated_schedule(
        _schedule(),
        timestamp=datetime(2026, 9, 9, 11, 0, tzinfo=timezone.utc),
        reference="sha256:schedule",
    )

    assert finding.check == "schedule"
    assert finding.status is FederatedPreflightStatus.ELIGIBLE
    assert finding.reason_code == "SCHEDULE_VALID"
    assert finding.reference == "sha256:schedule"


def test_schedule_check_blocks_after_enrollment() -> None:
    finding = check_federated_schedule(
        _schedule(),
        timestamp=datetime(2026, 9, 9, 13, 0, tzinfo=timezone.utc),
    )

    assert finding.status is FederatedPreflightStatus.BLOCKED
    assert finding.reason_code == "ENROLLMENT_WINDOW_CLOSED"


def test_schedule_check_blocks_after_round_finishes() -> None:
    finding = check_federated_schedule(
        _schedule(),
        timestamp=datetime(2026, 9, 9, 19, 0, tzinfo=timezone.utc),
    )

    assert finding.status is FederatedPreflightStatus.BLOCKED
    assert finding.reason_code == "SCHEDULE_EXPIRED"

def test_environment_check_passes_for_matching_lock_digest(tmp_path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("environment-lock", encoding="utf-8")

    expected = compute_environment_lock_digest(lock)

    finding = check_federated_environment(
        expected,
        lock_path=str(lock),
        reference=expected,
    )

    assert finding.check == "environment"
    assert finding.status is FederatedPreflightStatus.ELIGIBLE
    assert finding.reason_code == "ENVIRONMENT_LOCK_VALID"
    assert finding.reference == expected

def test_environment_check_blocks_missing_lock_digest(tmp_path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("environment-lock", encoding="utf-8")

    finding = check_federated_environment(
        None,
        lock_path=str(lock),
    )

    assert finding.status is FederatedPreflightStatus.BLOCKED
    assert finding.reason_code == "ENVIRONMENT_LOCK_MISSING"

def test_environment_check_blocks_mismatched_lock_digest(tmp_path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("environment-lock", encoding="utf-8")

    finding = check_federated_environment(
        "sha256:" + "0" * 64,
        lock_path=str(lock),
    )

    assert finding.status is FederatedPreflightStatus.BLOCKED
    assert finding.reason_code == "ENVIRONMENT_LOCK_MISMATCH"

def test_update_schema_passes_with_valid_metadata():
    model_digest = "sha256:" + "a" * 64
    update_digest = "sha256:" + "b" * 64

    policy = FederatedUpdatePolicy(
        model_digest=model_digest,
        parameters=(
            FederatedParameterMetadata(
                "adapter.lora_A.weight",
                (2, 3),
                "float32",
            ),
            FederatedParameterMetadata(
                "adapter.lora_B.weight",
                (4, 2),
                "float32",
            ),
        ),
        max_total_elements=14,
    )

    payload = {
        "schema_version": FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
        "model_digest": model_digest,
        "adapter_format": "dense",
        "parameters": [
            {
                "name": "adapter.lora_A.weight",
                "shape": [2, 3],
                "dtype": "float32",
            },
            {
                "name": "adapter.lora_B.weight",
                "shape": [4, 2],
                "dtype": "float32",
            },
        ],
        "total_elements": 14,
        "update_digest": update_digest,
        "clipped": True,
    }

    finding = check_federated_update_schema(
        payload,
        policy=policy,
        reference=update_digest,
    )

    assert finding.check == "update-schema"
    assert finding.status is FederatedPreflightStatus.ELIGIBLE
    assert finding.reason_code == "UPDATE_SCHEMA_VALID"
    assert finding.reference == update_digest

def test_update_schema_blocks_unsupported_schema():
    model_digest = "sha256:" + "a" * 64

    policy = FederatedUpdatePolicy(
        model_digest=model_digest,
        parameters=(
            FederatedParameterMetadata(
                "adapter.weight",
                (2, 2),
                "float32",
            ),
        ),
    )

    payload = {
        "schema_version": "openmed.training.federated_update_metadata.v999",
        "model_digest": model_digest,
        "adapter_format": "dense",
        "parameters": [
            {
                "name": "adapter.weight",
                "shape": [2, 2],
                "dtype": "float32",
            }
        ],
        "total_elements": 4,
        "update_digest": "sha256:" + "b" * 64,
        "clipped": True,
    }

    finding = check_federated_update_schema(
        payload,
        policy=policy,
    )

    assert finding.status is FederatedPreflightStatus.BLOCKED
    assert finding.reason_code == "UPDATE_SCHEMA_INVALID"

def test_metric_schema_blocks_unsupported_schema() -> None:
    envelope = build_federated_metric_envelope(
        metric_id="documents_processed",
        metric_kind=FederatedMetricKind.COUNT,
        aggregate_value=18,
        clipping_lower_bound=0,
        clipping_upper_bound=100,
        privacy_mechanism=FederatedPrivacyMechanism.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
        participant_count=12,
    )

    payload = envelope.to_dict()
    payload["schema_version"] = "openmed.training.federated_metric.v999"

    finding = check_federated_metric_schema(payload)

    assert finding.status is FederatedPreflightStatus.BLOCKED
    assert finding.reason_code == "METRIC_SCHEMA_INVALID"

def test_update_schema_blocks_policy_mismatch():
    model_digest = "sha256:" + "a" * 64

    policy = FederatedUpdatePolicy(
        model_digest=model_digest,
        parameters=(
            FederatedParameterMetadata(
                "adapter.weight",
                (2, 2),
                "float32",
            ),
        ),
    )

    payload = {
        "schema_version": FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
        "model_digest": model_digest,
        "adapter_format": "dense",
        "parameters": [
            {
                "name": "adapter.other_weight",
                "shape": [2, 2],
                "dtype": "float32",
            }
        ],
        "total_elements": 4,
        "update_digest": "sha256:" + "b" * 64,
        "clipped": True,
    }

    finding = check_federated_update_schema(
        payload,
        policy=policy,
    )

    assert finding.status is FederatedPreflightStatus.BLOCKED
    assert finding.reason_code == "UPDATE_SCHEMA_INVALID"

def test_update_schema_does_not_expose_invalid_payload():
    model_digest = "sha256:" + "a" * 64
    sentinel = "SYNTHETIC_PRIVATE_SENTINEL_3012"

    policy = FederatedUpdatePolicy(
        model_digest=model_digest,
        parameters=(
            FederatedParameterMetadata(
                "adapter.weight",
                (2, 2),
                "float32",
            ),
        ),
    )

    payload = {
        "schema_version": FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
        "model_digest": sentinel,
        "adapter_format": "dense",
        "parameters": [
            {
                "name": "adapter.weight",
                "shape": [2, 2],
                "dtype": "float32",
            }
        ],
        "total_elements": 4,
        "update_digest": "sha256:" + "b" * 64,
        "clipped": True,
    }

    finding = check_federated_update_schema(
        payload,
        policy=policy,
        reference="sha256:" + "c" * 64,
    )

    assert finding.status is FederatedPreflightStatus.BLOCKED
    assert sentinel not in str(finding.to_dict())

def test_metric_schema_accepts_valid_envelope() -> None:
    envelope = build_federated_metric_envelope(
        metric_id="documents_processed",
        metric_kind=FederatedMetricKind.COUNT,
        aggregate_value=18,
        clipping_lower_bound=0,
        clipping_upper_bound=100,
        privacy_mechanism=FederatedPrivacyMechanism.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
        participant_count=12,
    )

    finding = check_federated_metric_schema(
        envelope.to_dict(),
        reference="sha256:metric-fixture",
    )

    assert finding.status is FederatedPreflightStatus.ELIGIBLE
    assert finding.reason_code == "METRIC_SCHEMA_VALID"
    assert finding.reference == "sha256:metric-fixture"
