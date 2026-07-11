"""Tests for grounding calibration, coverage, and abstention."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical.exporters.codeable_concept import (
    GroundedSpan,
    to_codeable_concept,
)
from openmed.clinical.grounding import Candidate
from openmed.clinical.grounding.calibration import (
    GroundingCalibrationRecord,
    apply_grounding_abstention,
    calibrate_grounding,
    evaluate_grounding_coverage_gate,
    fit_grounding_calibrator,
    grounding_calibration_report,
)
from openmed.eval.metrics import expected_calibration_error, reliability_bins
from openmed.eval.suites.grounding_calibration import (
    run_grounding_calibration_suite,
)


def _bucket_records(system: str) -> tuple[list[dict[str, object]], list[bool]]:
    scores: list[dict[str, object]] = []
    gold: list[bool] = []
    for probability in (0.1, 0.3, 0.7, 0.9):
        positives = int(probability * 10)
        for index in range(10):
            scores.append(
                {
                    "system": system,
                    "label": "problem",
                    "score": probability,
                }
            )
            gold.append(index < positives)
    return scores, gold


def test_calibrate_grounding_returns_low_ece_on_held_out_synthetic_gold() -> None:
    train_scores, train_gold = _bucket_records("rxnorm")
    heldout_scores, heldout_gold = _bucket_records("rxnorm")
    calibrator = fit_grounding_calibrator(train_scores, train_gold)

    probabilities = calibrator.predict_many(heldout_scores)
    bins = reliability_bins(zip(probabilities, heldout_gold), n_bins=10)

    assert expected_calibration_error(bins) <= 0.05
    assert list(probabilities[:3]) == pytest.approx([0.1, 0.1, 0.1])


def test_grounding_report_emits_per_vocabulary_reliability_and_coverage_gate() -> None:
    records = [
        {
            "system": "rxnorm",
            "label": "drug",
            "score": 0.95,
            "correct": True,
        }
        for _ in range(70)
    ]
    records.extend(
        {
            "system": "rxnorm",
            "label": "drug",
            "score": 0.10,
            "correct": False,
        }
        for _ in range(30)
    )

    report = grounding_calibration_report(
        records,
        min_accuracy=0.85,
        min_coverage=0.70,
        generated_at="2026-07-05T00:00:00Z",
    )

    rxnorm = report["vocabularies"]["RXNORM"]
    assert rxnorm["expected_calibration_error"] == pytest.approx(0.0)
    assert len(rxnorm["reliability_diagram"]) == 10
    assert rxnorm["coverage_accuracy_curve"]
    assert report["coverage_gate"]["passed"] is True
    assert evaluate_grounding_coverage_gate(report)["passed"] is True


def test_abstained_spans_preserve_candidates_and_export_without_code() -> None:
    records = [
        GroundingCalibrationRecord("RXNORM", "DRUG", 0.20, False),
        GroundingCalibrationRecord("RXNORM", "DRUG", 0.90, True),
    ]
    calibrator = fit_grounding_calibrator(records)
    candidates = (
        Candidate("RXNORM", "1191", "aspirin", 0.20),
        Candidate("RXNORM", "161", "acetaminophen", 0.10),
    )
    span = GroundedSpan(text="aspirin", start=4, end=11, candidates=candidates)

    calibrated = apply_grounding_abstention(
        span,
        calibrator,
        {"RXNORM": {"threshold": 0.80}},
        label="drug",
    )
    concept = to_codeable_concept(calibrated)

    assert calibrated.abstained is True
    assert calibrated.candidates == candidates
    assert "coding" not in concept
    assert concept["_grounding"]["abstained"] is True
    assert concept["_grounding"]["candidate_count"] == 2


def test_calibrate_grounding_sequence_behaves_like_probabilities() -> None:
    result = calibrate_grounding(
        [0.1, 0.9],
        [False, True],
        systems=["hpo", "hpo"],
        labels=["phenotype", "phenotype"],
    )

    assert len(result) == 2
    assert list(result) == pytest.approx([0.0, 1.0])
    assert result.expected_calibration_error == pytest.approx(0.0)


def test_grounding_calibration_suite_reads_local_gold_and_writes_report(
    tmp_path: Path,
) -> None:
    gold_path = tmp_path / "grounding.jsonl"
    rows = [
        {"system": "hpo", "label": "phenotype", "score": 0.95, "correct": True},
        {"system": "hpo", "label": "phenotype", "score": 0.05, "correct": False},
    ]
    gold_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = run_grounding_calibration_suite(
        gold_path,
        artifact_dir=tmp_path / "artifacts",
        min_coverage=0.50,
        generated_at="2026-07-05T00:00:00Z",
    )

    report_path = Path(str(report["report_path"]))
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["suite"] == "grounding_calibration"
    assert payload["vocabularies"]["HPO"]["reliability_diagram"]
    assert payload["vocabularies"]["HPO"]["coverage_accuracy_curve"]
