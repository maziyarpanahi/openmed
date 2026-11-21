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
    parser.add_argument("--output", type=Path, default=SWIFT_RESOURCE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    registry = json.loads(SOURCE.read_bytes())
    declarations = {code: row["declaration"] for code, row in registry.items()}
    wire = (json.dumps(declarations, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != wire:
            print("Intended-use resource differs from the Python registry.")
            return 1
    else:
        args.output.write_bytes(wire)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
