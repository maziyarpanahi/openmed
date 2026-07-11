"""Offline grounding calibration eval suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openmed.clinical.grounding.calibration import (
    GroundingCalibrationRecord,
    coerce_grounding_calibration_records,
    grounding_calibration_report,
)

GROUNDING_CALIBRATION = "grounding_calibration"


def load_grounding_gold(path: str | Path) -> tuple[GroundingCalibrationRecord, ...]:
    """Load a local grounding gold file without network access.

    The file may be JSONL, a JSON list, or a JSON object containing ``samples``.
    Each row must contain a score, vocabulary/system, and correctness marker.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"grounding gold file not found: {source}")
    if source.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("grounding gold must be a JSON list or contain samples")
    return coerce_grounding_calibration_records(rows)


def run_grounding_calibration_suite(
    gold_path: str | Path,
    *,
    artifact_dir: str | Path | None = None,
    min_accuracy: float = 0.85,
    min_coverage: float = 0.70,
    n_bins: int = 10,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Run the offline grounding calibration suite over a local gold file."""

    records = load_grounding_gold(gold_path)
    report = grounding_calibration_report(
        records,
        min_accuracy=min_accuracy,
        min_coverage=min_coverage,
        n_bins=n_bins,
        generated_at=generated_at,
    )
    report["suite"] = GROUNDING_CALIBRATION
    report["gold_path"] = str(Path(gold_path))
    if artifact_dir is not None:
        output_dir = Path(artifact_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "grounding_calibration_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report["report_path"] = str(report_path)
    return report


def grounding_calibration_metadata(**kwargs: Any) -> dict[str, Any]:
    """Return suite metadata for the grounding calibration harness."""

    metadata = {"suite": GROUNDING_CALIBRATION, "offline": True}
    if kwargs.get("gold_path") is not None:
        metadata["gold_path"] = str(kwargs["gold_path"])
    return metadata


__all__ = [
    "GROUNDING_CALIBRATION",
    "grounding_calibration_metadata",
    "load_grounding_gold",
    "run_grounding_calibration_suite",
]
