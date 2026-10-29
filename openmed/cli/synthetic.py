"""Local synthetic dataset recipe CLI registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._output import EXIT_USAGE, CliError, add_json_flag, emit


def add_synthetic_command(subparsers: argparse._SubParsersAction) -> None:
    """Register a real top-level synthetic build command with bounded recipes."""
    parser = subparsers.add_parser(
        "synthetic", help="Build local synthetic datasets from pinned recipes."
    )
    children = parser.add_subparsers(dest="synthetic_command")
    build = children.add_parser("build", help="Run a bounded offline synthetic recipe.")
    build.add_argument("recipe", help="Versioned recipe JSON file.")
    build.add_argument(
        "output_directory",
        help="New output directory; existing data is never overwritten.",
    )
    add_json_flag(build)
    build.set_defaults(handler=_build, command_path="synthetic build")


def _build(args: argparse.Namespace) -> int:
    from openmed.training.synthetic.dataset_recipe import run_dataset_recipe

    try:
        path = Path(args.recipe)
        if path.stat().st_size > 1_000_000:
            raise ValueError

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate recipe field")
                result[key] = value
            return result

        recipe = json.loads(path.read_bytes(), object_pairs_hook=unique_object)
        result = run_dataset_recipe(recipe, args.output_directory)
    except Exception:
        raise CliError(
            "Synthetic recipe build failed validation or generation.",
            code="synthetic_recipe_failed",
            exit_code=EXIT_USAGE,
        ) from None
    return emit(
        args,
        result,
        human=f"Completed synthetic recipe: {result['record_count']} records, {result['artifact_count']} artifacts.",
    )
