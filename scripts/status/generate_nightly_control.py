#!/usr/bin/env python3
"""Run a synthetic, deliberately empty detector as nightly harness control."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from openmed.core.repro_hash import compute_reproducibility_hash
from openmed.eval.cache import hash_fixture_set
from openmed.eval.golden import load_benchmark_fixtures
from openmed.eval.harness import BenchmarkFixture, run_benchmark
from openmed.eval.report import BenchmarkReport

CONTROL_MODEL = "synthetic-empty-detector-v1"
CONTROL_SUITE = "nightly-synthetic-control"
_REVISION = re.compile(r"[0-9a-f]{40}")


def build_control_report(
    source_revision: str, *, now: datetime | None = None
) -> BenchmarkReport:
    """Return reproducible harness evidence without a model or external data."""
    if not _REVISION.fullmatch(source_revision):
        raise ValueError("source_revision must be a full Git commit SHA")
    fixtures = load_benchmark_fixtures()
    if not fixtures or any(
        row.metadata.get("synthetic") is not True for row in fixtures
    ):
        raise ValueError("nightly control requires nonempty synthetic golden fixtures")
    fixture_hash = hash_fixture_set(fixtures)
    generated_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    timestamp = generated_at.isoformat(timespec="seconds").replace("+00:00", "Z")
    recipe = {"runner": CONTROL_MODEL, "output": "no predicted spans", "device": "cpu"}
    data_manifest = {
        "suite": "golden",
        "fixture_set_hash": fixture_hash,
        "source_rights": "OpenMed generated synthetic fixtures",
        "license_id": "Apache-2.0",
    }
    reproducibility_hash = compute_reproducibility_hash(
        recipe=recipe,
        data_manifest=data_manifest,
        base_model={"id": CONTROL_MODEL, "revision": "v1"},
        git_sha=source_revision,
    )

    def empty_detector(
        fixture: BenchmarkFixture, model_name: str, device: str
    ) -> tuple[()]:
        """Return no spans so this control makes no detection claim."""
        del fixture, model_name, device
        return ()

    return run_benchmark(
        fixtures,
        suite=CONTROL_SUITE,
        model_name=CONTROL_MODEL,
        device="cpu",
        runner=empty_detector,
        generated_at=timestamp,
        metadata={
            "synthetic": True,
            "source_rights": data_manifest["source_rights"],
            "fixture_set_hash": fixture_hash,
            "fixture_source": "openmed/eval/golden/fixtures",
            "license_id": "Apache-2.0",
            "model_revision": "v1",
            "config_revision": "v1",
            "source_revision": source_revision,
            "reproducibility_hash": reproducibility_hash,
            "limitations": (
                "Harness control only; an empty detector is not a clinical or "
                "de-identification performance baseline."
            ),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write the public synthetic control report for status rendering."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_control_report(args.source_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
