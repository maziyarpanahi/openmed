"""Export the REST service OpenAPI document as deterministic JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import openmed
from openmed.service.app import create_app

DEFAULT_OUTPUT_PATH = REPO_ROOT / "docs" / "api" / "openapi.json"


def build_openapi_spec() -> dict[str, Any]:
    """Build the REST service OpenAPI document."""
    app = create_app()
    spec = app.openapi()
    spec.setdefault("info", {})["version"] = openmed.__version__
    return spec


def render_openapi_spec() -> bytes:
    """Render the OpenAPI document with stable byte-for-byte formatting."""
    payload = json.dumps(
        build_openapi_spec(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"{payload}\n".encode("utf-8")


def export_openapi(output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Write the OpenAPI document to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(render_openapi_spec())
    return output_path


def main() -> int:
    """Run the OpenAPI export command."""
    parser = argparse.ArgumentParser(
        description="Export the OpenMed REST service OpenAPI spec."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination JSON file. Defaults to docs/api/openapi.json.",
    )
    args = parser.parse_args()

    output_path = export_openapi(args.output)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
