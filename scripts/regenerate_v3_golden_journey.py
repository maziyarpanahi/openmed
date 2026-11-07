#!/usr/bin/env python3
"""Check or explicitly regenerate the v3 five-source golden Journey."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from openmed.clinical.journey_contracts import canonical_json, sha256_digest
from openmed.eval.golden_journey import (
    load_golden_journey_scenario,
    render_semantic_diff,
    run_golden_journey,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "tests" / "fixtures" / "journey" / "v3" / "scenario.json"
GOLDEN = ROOT / "tests" / "fixtures" / "journey" / "v3" / "golden.json"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="explicitly replace the committed golden output",
    )
    return parser.parse_args()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    args = _arguments()
    scenario = load_golden_journey_scenario(SCENARIO)
    with tempfile.TemporaryDirectory(prefix="openmed-golden-journey-") as work_dir:
        actual = run_golden_journey(scenario, work_dir=work_dir)
    rendered = json.dumps(actual, indent=2, sort_keys=True) + "\n"
    if args.write:
        _atomic_write(GOLDEN, rendered)
        print(f"updated {GOLDEN.relative_to(ROOT)} {sha256_digest(rendered)}")
        return 0
    if not GOLDEN.exists():
        print("golden output is missing; rerun with --write after reviewing inputs")
        return 2
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if canonical_json(expected) == canonical_json(actual):
        print(f"golden output matches {sha256_digest(rendered)}")
        return 0
    print("golden output differs:")
    print(render_semantic_diff(expected, actual))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
