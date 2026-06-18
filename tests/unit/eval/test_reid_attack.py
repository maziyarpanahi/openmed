from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.eval.attacks.reid import (
    generate_reid_leaderboard,
    render_reid_leaderboard,
    run_reid_attack,
    run_reid_benchmark,
)
from openmed.eval.golden import load_golden_fixtures


def _fixture_with_clean_output():
    return {
        "id": "clean-note",
        "language": "en",
        "text": "Patient Alice River visited North Clinic on 2026-01-10.",
        "gold_spans": [
            {
                "start": 8,
                "end": 19,
                "label": "PERSON",
                "text": "Alice River",
            },
            {
                "start": 28,
                "end": 40,
                "label": "ORGANIZATION",
                "text": "North Clinic",
            },
            {
                "start": 44,
                "end": 54,
                "label": "DATE",
                "text": "2026-01-10",
            },
        ],
        "metadata": {
            "synthetic": True,
            "category": "multilingual",
            "expected_output": {
                "method": "mask",
                "text": "Patient [PERSON] visited [ORGANIZATION] on [DATE].",
            },
        },
    }


def test_reid_attack_scores_planted_quasi_identifier_linkage() -> None:
    fixture = _fixture_with_clean_output()
    deidentified = [
        {
            "record_id": "clean-note",
            "age": 94,
            "city": "Smallville",
            "diagnosis": "rare vasculitis",
        }
    ]
    auxiliary = [
        {
            "age": 94,
            "city": "Smallville",
            "diagnosis": "rare vasculitis",
        }
    ]

    result = run_reid_attack(
        [fixture],
        deidentified_records=deidentified,
        auxiliary_records=auxiliary,
    )

    assert result.rate > 0.0
    assert result.to_metric()["aux_linkage_rate"] > 0.0


def test_reid_attack_scores_clean_fixture_near_zero() -> None:
    result = run_reid_attack([_fixture_with_clean_output()])

    assert result.rate == pytest.approx(0.0)
    assert result.to_metric()["leakage_rate"] == pytest.approx(0.0)


def test_reid_attack_flags_surrogate_and_date_shift_leaks() -> None:
    result = run_reid_attack(
        [_fixture_with_clean_output()],
        deidentified_records=[
            {
                "record_id": "leaky-note",
                "text": "Patient [PERSON] had visits on 2026-02-09 and 2026-02-16.",
                "audit_spans": [
                    {
                        "canonical_label": "PERSON",
                        "text_hash": "sha256:aaa",
                        "surrogate": "Robin Lane",
                    },
                    {
                        "canonical_label": "PERSON",
                        "text_hash": "sha256:aaa",
                        "surrogate": "Casey Lake",
                    },
                ],
                "metadata": {
                    "date_chain": {
                        "original_dates": ["2026-01-10", "2026-01-17"],
                        "shifted_dates": ["2026-02-09", "2026-02-16"],
                    }
                },
            }
        ],
    )

    metric = result.to_metric()
    assert metric["surrogate_consistency_rate"] > 0.0
    assert metric["date_shift_inversion_rate"] > 0.0
    assert result.rate > 0.0


def test_reid_benchmark_emits_report_and_leaderboard_score() -> None:
    report = run_reid_benchmark(
        suite="golden",
        model_name="unit-model",
        generated_at="2026-06-15T00:00:00+00:00",
    )

    assert report.suite == "golden"
    assert report.fixture_count == len(load_golden_fixtures())
    assert "reid_leakage" in report.metrics
    assert "reidentification" in report.metrics

    rows = generate_reid_leaderboard([report])
    assert rows == [
        {
            "model_name": "unit-model",
            "suite": "golden",
            "attack": "reid",
            "reid_leakage_rate": report.metrics["reid_leakage"]["rate"],
            "reid_successes": report.metrics["reid_leakage"]["numerator"],
            "fixture_count": report.fixture_count,
        }
    ]
    markdown = render_reid_leaderboard([report])
    assert "reid_leakage_rate" in markdown
    assert "unit-model" in markdown


def test_benchmark_pii_reid_cli_writes_report_and_leaderboard(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.json"
    leaderboard_path = tmp_path / "leaderboard.md"

    result = main_module.main(
        [
            "benchmark",
            "pii",
            "--attack",
            "reid",
            "--suite",
            "golden",
            "--model",
            "unit-model",
            "--output",
            str(report_path),
            "--leaderboard-output",
            str(leaderboard_path),
        ]
    )

    assert result == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    disk_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout_payload["metrics"]["reid_leakage"] == disk_payload["metrics"][
        "reid_leakage"
    ]
    assert "reid_leakage_rate" in leaderboard_path.read_text(encoding="utf-8")
