"""Create de-identified DHIS2 import payloads from local JSON snapshots.

This example performs no upload. The output files can be reviewed locally and
handed to facility-controlled DHIS2 tooling after the privacy checks pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openmed.clinical.exporters import DHIS2ExportConfig, export_dhis2


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    """Run the local facility-to-district export workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--org-units", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generalization-level", type=int, default=3)
    parser.add_argument("--small-cell-threshold", type=int, default=5)
    parser.add_argument(
        "--date-mode",
        choices=("shift", "coarsen", "none"),
        default="shift",
    )
    parser.add_argument("--date-shift-days", type=int)
    parser.add_argument(
        "--period-granularity",
        choices=("month", "year"),
        default="month",
    )
    args = parser.parse_args()

    config = DHIS2ExportConfig(
        generalization_level=args.generalization_level,
        small_cell_threshold=args.small_cell_threshold,
        date_mode=args.date_mode,
        date_shift_days=args.date_shift_days,
        period_granularity=args.period_granularity,
    )
    result = export_dhis2(
        _read_json(args.aggregate),
        _read_json(args.tracker),
        args.org_units,
        config=config,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "aggregate.json": result.aggregate_json(),
        "tracker.json": result.tracker_json(),
        "manifest.json": result.manifest_json(),
    }
    for name, content in outputs.items():
        (args.output_dir / name).write_text(content + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
