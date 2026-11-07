"""Export the service GraphQL schema as deterministic SDL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from openmed.service.graphql_schema import schema

DEFAULT_OUTPUT_PATH = REPO_ROOT / "docs" / "api" / "graphql-schema.graphql"


def render_graphql_schema() -> str:
    """Render the live GraphQL schema with stable trailing whitespace."""
    return f"{str(schema).rstrip()}\n"


def export_graphql_schema(output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Write the live GraphQL schema to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_graphql_schema(), encoding="utf-8")
    return output_path


def main() -> int:
    """Run the GraphQL SDL export command."""
    parser = argparse.ArgumentParser(
        description="Export the OpenMed service GraphQL schema."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination SDL file. Defaults to docs/api/graphql-schema.graphql.",
    )
    args = parser.parse_args()

    output_path = export_graphql_schema(args.output)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
