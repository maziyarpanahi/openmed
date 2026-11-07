#!/usr/bin/env python3
"""Generate typed Journey workflow client surfaces from the canonical registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openmed.mcp.tool_registry import render_tool_registry_document
from openmed.service.journey_workflows import (
    render_python_journey_client,
    render_typescript_journey_client,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = {
    ROOT / "openmed" / "service" / "journey_client_generated.py": (
        render_python_journey_client
    ),
    ROOT / "clients" / "typescript" / "src" / "journey-workflows.generated.ts": (
        render_typescript_journey_client
    ),
    ROOT / "openmed" / "interop" / "tools.json": lambda: (
        json.dumps(
            render_tool_registry_document(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated clients differ from the canonical registry.",
    )
    args = parser.parse_args()

    stale: list[Path] = []
    for path, render in OUTPUTS.items():
        expected = render()
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")

    if stale:
        for path in stale:
            print(f"stale generated Journey client: {path.relative_to(ROOT)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
