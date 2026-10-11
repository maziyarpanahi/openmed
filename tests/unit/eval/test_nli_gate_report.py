"""Synthetic evidence composition must remain content-free and non-releasing."""

from __future__ import annotations

import pytest

from openmed.eval.nli_calibration import calibrate_nli_thresholds
from openmed.eval.nli_error_slices import build_nli_error_slice_report
from openmed.eval.nli_gate import NLIEvaluationCounts, build_nli_preparation_report
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
