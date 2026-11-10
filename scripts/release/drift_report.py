#!/usr/bin/env python3
"""Emit a no-PHI production drift record for the retrain trigger to consume."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openmed.eval.drift_monitor import (
    DEFAULT_DRIFT_THRESHOLD,
    DEFAULT_REFERENCE_PATH,
    DEFAULT_WARNING_THRESHOLD,
    DriftInputError,
    DriftPrivacyError,
    compute_drift_report,
    load_drift_window,
)

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """Compute drift against the committed reference and write a drift record."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observation",
        type=Path,
        required=True,
        help="Path to a privacy-safe aggregate window JSON (counts/hashes only).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Committed reference window (defaults to gates/drift_reference.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("drift_report.json"),
        help="Where to write the deterministic drift record.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DRIFT_THRESHOLD,
        help="Divergence at or above which drift is flagged.",
    )
    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=DEFAULT_WARNING_THRESHOLD,
        help="Divergence at or above which a drift warning is raised.",
    )
    args = parser.parse_args(argv)

    try:
        reference = load_drift_window(args.reference)
        observation = load_drift_window(args.observation)
        report = compute_drift_report(
            reference,
            observation,
            threshold=args.threshold,
            warning_threshold=args.warning_threshold,
        )
        report.write_json(args.output)
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        DriftInputError,
        DriftPrivacyError,
    ) as exc:
        print(f"Drift report failed: {exc}", file=sys.stderr)
        return 2

    signal = report.to_trigger_signal().to_dict()
    print(f"Drift verdict: {report.verdict}")
    print(f"- max_divergence: {report.max_divergence:.6f}")
    print(f"- dominant_family: {report.dominant_family}")
    print(f"- dominant_drifting_label: {report.dominant_drifting_label}")
    print(f"- drift record: {args.output}")
    print("- trigger_signal:")
    print(json.dumps(signal, ensure_ascii=False, indent=2, sort_keys=True))

    return 1 if report.drift_detected else 0


if __name__ == "__main__":
    raise SystemExit(main())
