#!/usr/bin/env python3
"""Generate fixed-option decision JSON Schemas from the Python contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openmed.structured.decision import (
    decision_request_schema,
    decision_result_schema,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "openmed" / "core" / "schemas" / "json"
OUTPUTS = {
    SCHEMA_DIR / "decision_request.schema.json": decision_request_schema,
    SCHEMA_DIR / "decision_result.schema.json": decision_result_schema,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale: list[Path] = []
    for path, render in OUTPUTS.items():
        expected = json.dumps(render(), indent=2, sort_keys=True) + "\n"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                stale.append(path)
            continue
        path.write_text(expected, encoding="utf-8")
    if stale:
        for path in stale:
            print(f"stale decision schema: {path.relative_to(ROOT)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
