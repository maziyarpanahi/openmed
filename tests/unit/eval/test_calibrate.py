from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from openmed.cli import main_module
from openmed.core.audit import AuditReport
from openmed.core.pii import deidentify
from openmed.eval.calibrate import (
    build_thresholds_payload,
    fit_calibration_thresholds,
    load_calibration_thresholds,
    write_calibration_artifacts,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _samples() -> list[dict[str, object]]:
    return [
        {
            "model_id": "unit-model",
            "label": "NAME",
            "language": "en",
            "score": 0.95,
            "target": True,
        },
        {
            "model_id": "unit-model",
            "label": "NAME",
            "language": "en",
            "score": 0.72,
            "target": True,
        },
        {
            "model_id": "unit-model",
            "label": "NAME",
            "language": "en",
            "score": 0.71,
            "target": False,
        },
        {
            "model_id": "unit-model",
            "label": "NAME",
            "language": "en",
            "score": 0.40,
            "target": False,
        },
    ]


def test_fit_thresholds_are_monotonic_and_target_leakage_first() -> None:
    report = fit_calibration_thresholds(
        _samples(),
        model_id="unit-model",
        suite="golden",
        target_leakage=0.0,
        min_recall=1.0,
        generated_at="2026-06-15T00:00:00+00:00",
    )

    group = report.groups[0]
    assert group.label == "PERSON"
    assert group.language == "en"
    assert group.chosen_threshold == pytest.approx(0.72)
    assert group.resulting_leakage == 0.0
    assert group.over_redaction == 0.0

    leakage = [row["leakage"] for row in group.reliability]
    over_redaction = [row["over_redaction"] for row in group.reliability]
    assert leakage == sorted(leakage)
    assert over_redaction == sorted(over_redaction, reverse=True)


def test_recall_protection_prefers_leakage_over_over_redaction() -> None:
    report = fit_calibration_thresholds(
        [
            {
                "model_id": "unit-model",
                "label": "EMAIL",
                "language": "en",
                "score": 0.80,
                "target": True,
            },
            {
                "model_id": "unit-model",
                "label": "EMAIL",
                "language": "en",
                "score": 0.55,
                "target": True,
            },
            {
                "model_id": "unit-model",
                "label": "EMAIL",
                "language": "en",
                "score": 0.70,
                "target": False,
            },
        ],
        model_id="unit-model",
        suite="golden",
        target_leakage=0.0,
        min_recall=1.0,
    )

    group = report.groups[0]
    assert group.chosen_threshold == pytest.approx(0.55)
    assert group.recall == 1.0
    assert group.resulting_leakage == 0.0


def test_threshold_artifact_round_trip_and_report_fields(tmp_path: Path) -> None:
    paths = write_calibration_artifacts(
        _samples(),
        artifact_dir=tmp_path,
        model_id="unit-model",
        suite="golden",
        target_leakage=0.0,
        min_recall=1.0,
        generated_at="2026-06-15T00:00:00+00:00",
    )

    assert paths.thresholds_path.exists()
    assert paths.report_path.exists()

    thresholds = load_calibration_thresholds(paths.thresholds_path)
    assert thresholds.lookup("PERSON", "en", model_id="unit-model") == pytest.approx(
        0.72
    )

    payload = json.loads(paths.report_path.read_text(encoding="utf-8"))
    group = payload["groups"][0]
    assert {
        "reliability",
        "chosen_threshold",
        "resulting_leakage",
        "over_redaction",
        "label",
        "language",
    }.issubset(group)


def test_membership_defense_round_trips_in_calibration_artifact(
    tmp_path: Path,
) -> None:
    paths = write_calibration_artifacts(
        [
            {
                "model_id": "unit-model",
                "label": "NAME",
                "language": "en",
                "score": 0.95,
                "target": True,
            }
        ],
        artifact_dir=tmp_path,
        model_id="unit-model",
        suite="golden",
        target_leakage=0.0,
        min_recall=1.0,
        membership_defense={
            "enabled": True,
            "clip_min": 0.5,
            "clip_max": 0.5,
            "advantage_ceiling": 0.05,
        },
    )

    thresholds = load_calibration_thresholds(paths.thresholds_path)
    policy = thresholds.membership_defense_policy
    payload = json.loads(paths.thresholds_path.read_text(encoding="utf-8"))
    report_payload = json.loads(paths.report_path.read_text(encoding="utf-8"))

    assert policy.enabled is True
    assert policy.apply_score(0.99) == pytest.approx(0.5)
    assert thresholds.lookup("PERSON", "en", model_id="unit-model") == pytest.approx(
        0.5
    )
    assert payload["membership_defense"]["enabled"] is True
    assert report_payload["groups"][0]["recall"] >= report_payload["min_recall"]


def test_calibrate_cli_writes_default_golden_artifacts(tmp_path: Path) -> None:
    result = main_module.main(
        [
            "calibrate",
            "--model",
            "unit-model",
            "--suite",
            "golden",
            "--artifact-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert (tmp_path / "thresholds.json").is_file()
    assert (tmp_path / "calibration_report.json").is_file()


def test_inference_loads_thresholds_and_records_active_audit_threshold(
    tmp_path: Path,
) -> None:
    threshold_payload = build_thresholds_payload(
        fit_calibration_thresholds(
            [
                {
                    "model_id": "unit-model",
                    "label": "NAME",
                    "language": "en",
                    "score": 0.91,
                    "target": True,
                }
            ],
            model_id="unit-model",
            suite="golden",
            target_leakage=0.0,
            min_recall=1.0,
        )
    )
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text(
        json.dumps(threshold_payload, indent=2),
        encoding="utf-8",
    )

    text = "Patient John Doe and Jane Roe."

    def _prediction(*args: object, **kwargs: object) -> PredictionResult:
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text="John Doe",
                    label="NAME",
                    start=text.index("John Doe"),
                    end=text.index("John Doe") + len("John Doe"),
                    confidence=0.95,
                ),
                EntityPrediction(
                    text="Jane Roe",
                    label="NAME",
                    start=text.index("Jane Roe"),
                    end=text.index("Jane Roe") + len("Jane Roe"),
                    confidence=0.80,
                ),
            ],
            model_name="unit-model",
            timestamp=datetime.now().isoformat(),
        )

    with patch("openmed.core.pii.extract_pii", side_effect=_prediction):
        report = deidentify(
            text,
            model_name="unit-model",
            confidence_threshold=0.1,
            use_safety_sweep=False,
            calibration_thresholds_path=threshold_path,
            audit=True,
        )

    assert isinstance(report, AuditReport)
    assert report.thresholds["PERSON"] == pytest.approx(0.91)
    assert [span.label for span in report.spans] == ["NAME"]
    assert report.spans[0].threshold == pytest.approx(0.91)


def test_membership_defense_applies_at_calibrated_inference_without_recall_drop(
    tmp_path: Path,
) -> None:
    threshold_payload = build_thresholds_payload(
        fit_calibration_thresholds(
            [
                {
                    "model_id": "unit-model",
                    "label": "NAME",
                    "language": "en",
                    "score": 0.95,
                    "target": True,
                }
            ],
            model_id="unit-model",
            suite="golden",
            target_leakage=0.0,
            min_recall=1.0,
            membership_defense={
                "enabled": True,
                "clip_min": 0.5,
                "clip_max": 0.5,
                "advantage_ceiling": 0.05,
            },
        )
    )
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text(
        json.dumps(threshold_payload, indent=2),
        encoding="utf-8",
    )

    text = "Patient John Doe and Jane Roe."

    def _prediction(*args: object, **kwargs: object) -> PredictionResult:
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text="John Doe",
                    label="NAME",
                    start=text.index("John Doe"),
                    end=text.index("John Doe") + len("John Doe"),
                    confidence=0.99,
                ),
                EntityPrediction(
                    text="Jane Roe",
                    label="NAME",
                    start=text.index("Jane Roe"),
                    end=text.index("Jane Roe") + len("Jane Roe"),
                    confidence=0.80,
                ),
            ],
            model_name="unit-model",
            timestamp=datetime.now().isoformat(),
        )

    with patch("openmed.core.pii.extract_pii", side_effect=_prediction):
        report = deidentify(
            text,
            model_name="unit-model",
            confidence_threshold=0.1,
            use_safety_sweep=False,
            calibration_thresholds_path=threshold_path,
            audit=True,
        )

    assert isinstance(report, AuditReport)
    assert report.thresholds["PERSON"] == pytest.approx(0.5)
    assert [span.label for span in report.spans] == ["NAME", "NAME"]
    assert [span.confidence for span in report.spans] == pytest.approx([0.5, 0.5])
