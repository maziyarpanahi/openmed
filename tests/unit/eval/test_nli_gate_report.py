"""Synthetic evidence composition must remain content-free and non-releasing."""

from __future__ import annotations

import pytest

from openmed.eval.nli_calibration import calibrate_nli_thresholds
from openmed.eval.nli_error_slices import build_nli_error_slice_report
from openmed.eval.nli_gate import (
    NLIEvaluationCounts,
    NLIFormatParity,
    build_nli_candidate_report,
    build_nli_preparation_report,
)
from openmed.eval.nli_negation_challenge import (
    default_nli_negation_cases,
    run_nli_negation_challenge,
)

REVISION = "a" * 40
DIGEST = "sha256:" + "b" * 64


def _evidence(*, negation_passes: bool = True):
    cases = default_nli_negation_cases()
    predictions = {
        case.case_id: case.gold_label if negation_passes else "entailment"
        for case in cases
    }
    return {
        "model_id": "OpenMed/Synthetic-NLI",
        "model_revision": REVISION,
        "public": NLIEvaluationCounts(3, 2, DIGEST),
        "synthetic": NLIEvaluationCounts(4, 3, DIGEST),
        "error_slices": build_nli_error_slice_report(
            [
                {
                    "fixture_id": "synthetic-case-1",
                    "phenomena": [
                        "negation",
                        "temporality",
                        "experiencer",
                        "numbers",
                        "medication_status",
                    ],
                    "gold_label": "entailment",
                    "predicted_label": "entailment",
                }
            ],
            fixture_set_id="synthetic-cases",
        ),
        "calibration": calibrate_nli_thresholds(
            model_id="OpenMed/Synthetic-NLI", model_revision=REVISION
        ),
        "negation": run_nli_negation_challenge(predictions),
    }


def test_preparation_report_composes_existing_gates_without_shipping() -> None:
    report, checks = build_nli_preparation_report(**_evidence())
    assert report.fixture_count == 7
    assert report.metrics["public"]["accuracy"] == 2 / 3
    assert report.metrics["synthetic"]["accuracy"] == 3 / 4
    assert report.metadata["stage"] == "preparation_only"
    assert checks[-1].gate == "nli_publication"
    assert checks[-1].passed is False
    assert all(check.passed for check in checks[:-1])
    rendered = report.to_json()
    assert "premise" not in rendered
    assert "hypothesis" not in rendered
    assert "synthetic-case-1" not in rendered
    assert "patient" not in rendered


def test_failed_negation_remains_visible_and_publication_stays_closed() -> None:
    _, checks = build_nli_preparation_report(**_evidence(negation_passes=False))
    assert checks[2].gate == "nli_negation"
    assert checks[2].passed is False
    assert checks[3].passed is False


def test_mismatched_calibration_revision_is_rejected() -> None:
    evidence = _evidence()
    evidence["model_revision"] = "c" * 40
    with pytest.raises(ValueError, match="provenance differs"):
        build_nli_preparation_report(**evidence)


@pytest.mark.parametrize(
    "count,correct,digest",
    [(0, 0, DIGEST), (1, 2, DIGEST), (1, 1, "raw text")],
)
def test_invalid_aggregate_evidence_is_rejected(
    count: int, correct: int, digest: str
) -> None:
    with pytest.raises(ValueError, match="bounded aggregates"):
        NLIEvaluationCounts(count, correct, digest)


def _candidate_evidence():
    evidence = _evidence()
    del evidence["model_revision"]
    fixtures = [
        {
            "premise": "Synthetic premise.",
            "hypothesis": "Synthetic supported claim.",
            "gold_label": "entailment",
            "entailment_score": 0.999,
        },
        {
            "premise": "Synthetic premise.",
            "hypothesis": "Synthetic unsupported claim.",
            "gold_label": "not_entailment",
            "entailment_score": 0.001,
        },
    ]
    calibration = calibrate_nli_thresholds(
        fixtures,
        model_id=evidence["model_id"],
        model_revision=DIGEST,
        precision_floor=0.99,
        recall_floor=0.25,
        false_positive_rate_ceiling=0.01,
    )
    evidence.update(
        artifact_digest=DIGEST,
        public=NLIEvaluationCounts(100, 65, DIGEST),
        biomedical=NLIEvaluationCounts(100, 75, DIGEST),
        synthetic=NLIEvaluationCounts(100, 90, DIGEST),
        calibration=calibration,
        contradiction_calibration=calibration,
        synthetic_entailment_support=100,
        synthetic_entailment_accepted=25,
        parity=(
            NLIFormatParity("mlx", 100, 100, 0.001, DIGEST),
            NLIFormatParity("onnx_int8", 100, 98, 0.05, DIGEST),
        ),
    )
    return evidence


def test_measured_candidate_can_qualify_without_claiming_publication():
    report, checks = build_nli_candidate_report(**_candidate_evidence())
    assert all(check.passed for check in checks[:-1])
    assert checks[-1].gate == "nli_publication" and not checks[-1].passed
    assert report.metadata["stage"] == "qualified_unpublished_candidate"
    assert report.fixture_count == 300
    assert "Synthetic premise" not in report.to_json()


@pytest.mark.parametrize(
    "split,correct", [("public", 64), ("biomedical", 74), ("synthetic", 89)]
)
def test_below_frozen_quality_floor_never_qualifies(split, correct):
    evidence = _candidate_evidence()
    evidence[split] = NLIEvaluationCounts(100, correct, DIGEST)
    report, checks = build_nli_candidate_report(**evidence)
    assert report.metadata["stage"] == "failed_candidate"
    assert not all(check.passed for check in checks[:-1])


def test_all_abstention_cannot_pass_selective_coverage():
    evidence = _candidate_evidence()
    evidence["synthetic_entailment_accepted"] = 0
    _, checks = build_nli_candidate_report(**evidence)
    assert not next(
        check for check in checks if check.gate == "nli_selective_coverage"
    ).passed


def test_calibration_fallback_cannot_be_mistaken_for_valid_thresholds():
    evidence = _candidate_evidence()
    evidence["calibration"] = calibrate_nli_thresholds(
        [
            {
                "premise": "Synthetic premise.",
                "hypothesis": "Supported.",
                "gold_label": "entailment",
                "entailment_score": 0.1,
            },
            {
                "premise": "Synthetic premise.",
                "hypothesis": "Unsupported.",
                "gold_label": "not_entailment",
                "entailment_score": 0.9,
            },
        ],
        model_id=evidence["model_id"],
        model_revision=DIGEST,
        precision_floor=0.99,
        recall_floor=0.25,
        false_positive_rate_ceiling=0.01,
    )
    report, checks = build_nli_candidate_report(**evidence)
    assert report.metadata["stage"] == "failed_candidate"
    assert not next(check for check in checks if check.gate == "nli_calibration").passed


@pytest.mark.parametrize(
    "runtime,count,agreement,delta",
    [
        ("mlx", 95, 95, 0.0),
        ("mlx", 100, 99, 0.0),
        ("mlx", 100, 100, 0.0011),
        ("onnx_int8", 100, 97, 0.01),
        ("onnx_int8", 100, 100, 0.051),
    ],
)
def test_export_parity_requires_actual_coverage_and_frozen_tolerances(
    runtime, count, agreement, delta
):
    assert not NLIFormatParity(runtime, count, agreement, delta, DIGEST).passed


@pytest.mark.parametrize("field", ["artifact_digest", "calibration", "parity"])
def test_candidate_provenance_mismatch_is_rejected(field):
    evidence = _candidate_evidence()
    if field == "artifact_digest":
        evidence[field] = "sha256:" + "c" * 64
    elif field == "calibration":
        evidence[field] = calibrate_nli_thresholds(
            model_id=evidence["model_id"], model_revision="c" * 40
        )
    else:
        evidence[field] = (
            NLIFormatParity("mlx", 100, 100, 0.0, "sha256:" + "c" * 64),
            NLIFormatParity("onnx_int8", 100, 100, 0.0, DIGEST),
        )
    with pytest.raises(ValueError, match="provenance differs|source digest"):
        build_nli_candidate_report(**evidence)


@pytest.mark.parametrize("delta", [float("nan"), float("inf"), -0.01, 1.1, True])
def test_format_measurements_reject_unbounded_values(delta):
    with pytest.raises(ValueError, match="bounded measured aggregates"):
        NLIFormatParity("mlx", 100, 100, delta, DIGEST)
