"""Run the inspectable v3 five-source synthetic Journey offline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from openmed.eval.golden_journey import (
    load_golden_journey_scenario,
    run_golden_journey,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "tests" / "fixtures" / "journey" / "v3" / "scenario.json"


def main() -> None:
    """Execute the real local contracts and print a compact receipt."""

    scenario = load_golden_journey_scenario(SCENARIO)
    with tempfile.TemporaryDirectory(prefix="openmed-golden-journey-") as work_dir:
        report = run_golden_journey(scenario, work_dir=work_dir)
    receipt = {
        "dataset_snapshot_id": report["dataset"]["snapshot"]["snapshot_id"],
        "fact_count": len(report["facts"]),
        "omop_state": report["omop"]["state"],
        "scenario_id": report["scenario_id"],
        "source_formats": [item["format"] for item in report["sources"]],
        "state_matrix": report["state_matrix"],
        "synthetic": report["synthetic"],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
