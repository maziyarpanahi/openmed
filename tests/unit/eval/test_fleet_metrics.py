"""Tests for model fleet freshness metrics."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.eval.fleet_metrics import (
    compute_fleet_freshness,
    compute_fleet_freshness_from_manifest,
    write_fleet_freshness_artifact,
)

AS_OF = date(2026, 6, 14)


def _row(
    repo_id: str,
    released: str | None,
    *,
    languages: list[str] | None = None,
    **extra: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "repo_id": repo_id,
        "released": released,
        "languages": languages or [],
    }
    row.update(extra)
    return row


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_median_age_odd_count_fixture() -> None:
    metrics = compute_fleet_freshness(
        [
            _row("OpenMed/model-a", "2026-06-04", languages=["en"]),
            _row("OpenMed/model-b", "2026-05-25", languages=["en"]),
            _row("OpenMed/model-c", "2026-06-09", languages=["fr"]),
        ],
        as_of=AS_OF,
    )

    assert metrics.median_age_days == 10.0
    assert metrics.days_since_last_released_checkpoint == 5
    assert metrics.days_since_last_released_language == 5
    assert metrics.last_released_checkpoint is not None
    assert metrics.last_released_checkpoint.repo_id == "OpenMed/model-c"


def test_median_age_even_count_fixture() -> None:
    metrics = compute_fleet_freshness(
        [
            _row("OpenMed/model-a", "2026-06-09", languages=["en"]),
            _row("OpenMed/model-b", "2026-05-30", languages=["en"]),
            _row("OpenMed/model-c", "2026-05-20", languages=["fr"]),
            _row("OpenMed/model-d", "2026-05-10", languages=["de"]),
        ],
        as_of=AS_OF,
    )

    assert metrics.median_age_days == 20.0


def test_single_entry_reports_reference_target() -> None:
    metrics = compute_fleet_freshness(
        [_row("OpenMed/model-a", "2026-06-07", languages=["en"])],
        as_of=AS_OF,
        median_age_target_days=30,
    )
    payload = metrics.to_dict()

    assert metrics.median_age_days == 7.0
    assert payload["median_age_target"] == {
        "operator": "<",
        "days": 30,
        "met": True,
        "gating": False,
    }


def test_empty_manifest_reports_none_values() -> None:
    metrics = compute_fleet_freshness([], as_of=AS_OF)

    assert metrics.total_model_count == 0
    assert metrics.dated_model_count == 0
    assert metrics.undated_model_count == 0
    assert metrics.median_age_days is None
    assert metrics.median_age_target_met is None
    assert metrics.days_since_last_released_checkpoint is None
    assert metrics.days_since_last_released_language is None
    assert metrics.days_since_last_released_language_by_code == {}


def test_days_since_last_released_language_fixture() -> None:
    metrics = compute_fleet_freshness(
        [
            _row("OpenMed/model-en", "2026-06-01", languages=["en"]),
            _row("OpenMed/model-fr-es", "2026-06-10", languages=["fr", "es"]),
            _row(
                "OpenMed/model-de",
                None,
                languages=["de"],
                updated="2026-06-12T09:30:00Z",
            ),
            _row("OpenMed/no-language", "2026-06-13", languages=[]),
        ],
        as_of=AS_OF,
    )

    assert metrics.days_since_last_released_checkpoint == 1
    assert metrics.days_since_last_released_language == 2
    assert metrics.days_since_last_released_language_by_code == {
        "de": 2,
        "en": 13,
        "es": 4,
        "fr": 4,
    }
    assert metrics.last_released_language is not None
    assert metrics.last_released_language[0] == "de"
    assert metrics.last_released_language[1].repo_id == "OpenMed/model-de"


def test_writes_json_and_markdown_artifacts(tmp_path: Path) -> None:
    metrics = compute_fleet_freshness(
        [
            _row("OpenMed/model-a", "2026-06-04", languages=["en"]),
            _row("OpenMed/model-b", "2026-06-09", languages=["fr"]),
        ],
        as_of=AS_OF,
    )
    json_path = tmp_path / "fleet-freshness.json"
    markdown_path = tmp_path / "fleet-freshness.md"

    write_fleet_freshness_artifact(metrics, json_path, output_format="json")
    write_fleet_freshness_artifact(
        metrics,
        markdown_path,
        output_format="markdown",
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["median_age_target"]["days"] == 30
    assert payload["median_age_target"]["gating"] is False
    assert "## Language Freshness" in markdown_path.read_text(encoding="utf-8")


def test_reads_manifest_and_cli_writes_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "models.jsonl"
    output = tmp_path / "fleet-freshness.json"
    _write_manifest(
        manifest,
        [
            _row("OpenMed/model-a", "2026-06-04", languages=["en"]),
            _row("OpenMed/model-b", "2026-06-09", languages=["fr"]),
        ],
    )

    metrics = compute_fleet_freshness_from_manifest(manifest, as_of=AS_OF)
    result = main_module.main(
        [
            "models",
            "freshness",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--format",
            "json",
            "--as-of",
            AS_OF.isoformat(),
        ]
    )

    assert metrics.median_age_days == 7.5
    assert result == 0
    assert "Fleet freshness metrics written to:" in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["median_age_days"] == 7.5
