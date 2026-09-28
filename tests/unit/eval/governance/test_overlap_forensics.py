from __future__ import annotations

import hashlib
import json

import pytest

from openmed.eval.governance.holdout_commitment import (
    HOLDOUT_MANIFEST_KINDS,
    commit_holdout_manifests,
)
from openmed.eval.governance.overlap_forensics import (
    CHECK_CANARY,
    CHECK_EXACT,
    CHECK_FUZZY,
    CHECK_NORMALIZED,
    CHECK_PUBLIC_VS_SHADOW,
    OVERLAP_FORENSICS_CHECKS,
    OVERLAP_FORENSICS_POLICY_VERSION,
    OVERLAP_FORENSICS_SCHEMA_VERSION,
    OverlapForensicsError,
    scan_overlap_forensics,
)
from openmed.eval.workflows.sealed_manifest import (
    SEALED_MANIFEST_COMPONENTS,
    seal_workflow_manifest,
)


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('ascii')).hexdigest()}"


def _submission_manifest():
    return seal_workflow_manifest(
        {component: _digest(component) for component in SEALED_MANIFEST_COMPONENTS}
    )


def _holdout_commitment():
    manifests = {
        kind: tuple(_digest(f"{kind}-{index}") for index in range(2))
        for kind in HOLDOUT_MANIFEST_KINDS
    }
    return commit_holdout_manifests("synthetic-holdout-v1", manifests)


def _scan(**overrides):
    arguments = {
        "submission_items": (
            "Synthetic exact public artifact.",
            "SYNTHETIC normalized artifact!",
            "Synthetic protocol gamma dosage window eight days.",
            "Synthetic response contains canary phrase delta for audit.",
            "Synthetic shadow-only protocol omega remains stable.",
        ),
        "public_items": (
            "Synthetic exact public artifact.",
            "synthetic normalized artifact",
            "Synthetic protocol gamma dosage window seven days.",
            "Unrelated public comparison material.",
        ),
        "shadow_items": (
            "Distinct evaluator shadow material.",
            "Synthetic shadow-only protocol omega remains stable.",
        ),
        "canaries": ("canary phrase delta",),
        "submission_manifest": _submission_manifest(),
        "holdout_commitment": _holdout_commitment(),
        "fuzzy_threshold_basis_points": 7_000,
        "shadow_margin_basis_points": 1_000,
    }
    arguments.update(overrides)
    return scan_overlap_forensics(**arguments)


def test_report_combines_all_checks_and_records_explicit_coverage() -> None:
    report = _scan()

    assert report.schema_version == OVERLAP_FORENSICS_SCHEMA_VERSION
    assert report.policy_version == OVERLAP_FORENSICS_POLICY_VERSION
    assert report.signals_detected is True
    assert set(signal.check for signal in report.signals) == set(
        OVERLAP_FORENSICS_CHECKS
    )
    assert report.to_dict()["coverage"] == {
        "canary_items": 1,
        "checks": list(OVERLAP_FORENSICS_CHECKS),
        "comparison_counts": {
            CHECK_CANARY: 5,
            CHECK_EXACT: 30,
            CHECK_FUZZY: 30,
            CHECK_NORMALIZED: 30,
            CHECK_PUBLIC_VS_SHADOW: 30,
        },
        "public_items": 4,
        "shadow_items": 2,
        "submission_items": 5,
    }
    assert all(check in report.signal_counts for check in OVERLAP_FORENSICS_CHECKS)
    assert "do not by themselves establish contamination" in report.interpretation


def test_report_is_deterministic_digest_bound_and_contains_no_input_text() -> None:
    first = _scan()
    second = _scan(
        submission_manifest=_submission_manifest().to_dict(),
        holdout_commitment=_holdout_commitment().to_dict(),
    )

    assert first == second
    assert first.to_json() == second.to_json()
    assert first.to_json() == json.dumps(
        first.to_dict(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert first.submission_manifest_digest == _submission_manifest().manifest_digest
    assert first.holdout_commitment_digest == (_holdout_commitment().commitment_digest)
    rendered = first.to_json()
    for private_value in (
        "Synthetic exact public artifact.",
        "Synthetic protocol gamma dosage window seven days.",
        "canary phrase delta",
        "Synthetic shadow-only protocol omega remains stable.",
    ):
        assert private_value not in rendered


def test_signal_references_are_ordinal_and_scores_are_calibrated() -> None:
    report = _scan()

    assert all(
        signal.submission_ref.startswith("submission:") for signal in report.signals
    )
    assert all(
        signal.reference_ref.startswith(f"{signal.reference_scope}:")
        for signal in report.signals
    )
    assert all(0 <= signal.score_basis_points <= 10_000 for signal in report.signals)
    differential = next(
        signal for signal in report.signals if signal.check == CHECK_PUBLIC_VS_SHADOW
    )
    assert differential.reference_scope == "shadow"
    assert differential.comparison_score_basis_points is not None
    assert (
        differential.score_basis_points - differential.comparison_score_basis_points
        >= report.shadow_margin_basis_points
    )


def test_no_signal_language_does_not_claim_absence_of_contamination() -> None:
    report = _scan(
        submission_items=("Synthetic submission alpha.",),
        public_items=("Public benchmark beta.",),
        shadow_items=("Shadow benchmark gamma.",),
        canaries=("canary marker delta",),
        fuzzy_threshold_basis_points=9_500,
    )

    assert report.signals_detected is False
    assert report.signals == ()
    assert all(count == 0 for count in report.signal_counts.values())
    assert "does not establish the absence" in report.interpretation


@pytest.mark.parametrize(
    ("argument", "value", "reason"),
    (
        ("submission_items", (), "submission_items: empty"),
        ("public_items", ("***",), "public_items: empty_after_normalization"),
        ("fuzzy_threshold_basis_points", True, "invalid_basis_points"),
        ("shadow_margin_basis_points", 10_001, "invalid_basis_points"),
    ),
)
def test_invalid_configuration_fails_with_closed_error_codes(
    argument: str,
    value: object,
    reason: str,
) -> None:
    with pytest.raises(OverlapForensicsError, match=reason):
        _scan(**{argument: value})


def test_invalid_evidence_and_items_never_echo_caller_values() -> None:
    private_value = "synthetic-private-clinical-value"

    with pytest.raises(OverlapForensicsError) as invalid_item:
        _scan(canaries=(private_value, 7))
    assert private_value not in str(invalid_item.value)

    invalid_manifest = _submission_manifest().to_dict()
    invalid_manifest["manifest_digest"] = private_value
    with pytest.raises(OverlapForensicsError) as invalid_evidence:
        _scan(submission_manifest=invalid_manifest)
    assert private_value not in str(invalid_evidence.value)


def test_canary_scan_uses_normalized_containment() -> None:
    report = _scan(
        submission_items=("Prefix CANARY---PHRASE delta suffix.",),
        public_items=("Public comparison material.",),
        shadow_items=("Shadow comparison material.",),
        canaries=("canary phrase delta",),
        fuzzy_threshold_basis_points=10_000,
    )

    canary_signals = [
        signal for signal in report.signals if signal.check == CHECK_CANARY
    ]
    assert len(canary_signals) == 1
    assert canary_signals[0].reference_ref == "canary:000001"
