#!/usr/bin/env python3
"""Export the offline capability-code to intended-use declaration table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "openmed/core/data/intended_use.json"
SWIFT_RESOURCE = (
    ROOT / "swift/OpenMedKit/Sources/OpenMedKit/Resources/intended_use.json"
)


def main(argv=None) -> int:
    """Write the portable declarations, or fail if an existing copy differs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--notices",
        action="store_true",
        help="Project runtime notices from shared safety vectors.",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.notices:
        vectors = json.loads(
            (ROOT / "tests/fixtures/parity/safety_primitives_v1.json").read_bytes()
        )
        brief = json.loads(
            (
                ROOT / "openmed/core/schemas/json/clinical_brief_response.schema.json"
            ).read_bytes()
        )
        payload = {
            "tiers": {row["tier"]: row["notice_id"] for row in vectors["tiers"]},
            "notices": {
                row["id"]: {k: v for k, v in row.items() if k != "acknowledgement"}
                for row in vectors["notices"]
            },
            "capabilities": vectors["capability_notices"],
            "overrides": vectors["entry_point_overrides"],
            "clinical_brief_limitation": brief["$defs"]["envelope"]["properties"][
                "disclaimer"
            ]["const"],
        }
        default = SWIFT_RESOURCE.with_name("safety_notices.json")
    else:
        registry = json.loads(SOURCE.read_bytes())
        payload = {code: row["declaration"] for code, row in registry.items()}
        default = SWIFT_RESOURCE
    output = args.output or default
    wire = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if args.check:
        if not output.is_file() or output.read_bytes() != wire:
            print("Bundled safety resource differs from its source.")
            return 1
    else:
        output.write_bytes(wire)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
