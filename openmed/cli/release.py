"""Release management commands for OpenMed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from openmed.core.baseline import BASELINE_PATH
from openmed.core.manifest import (
    DEFAULT_CARD_DIR,
    DEFAULT_ROLLBACK_LOG_PATH,
    DEFAULT_STATUS_PATH,
    ManifestRollbackError,
    rollback_manifest_pointer,
)
from openmed.core.model_registry import MANIFEST_PATH


def add_release_command(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``openmed release`` command group."""

    release_parser = subparsers.add_parser(
        "release",
        help="Manage OpenMed release operations.",
    )
    release_subparsers = release_parser.add_subparsers(dest="release_command")
    _add_rollback_parser(release_subparsers)
    release_parser.set_defaults(handler=_help_handler(release_parser))
    return release_parser


def _add_rollback_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``rollback`` release subcommand."""

    rollback_parser = subparsers.add_parser(
        "rollback",
        help="Rollback a model family to its last-green manifest entry.",
    )
    rollback_parser.add_argument(
        "family",
        help="Model family to roll back, for example PII.",
    )
    rollback_parser.add_argument(
        "--tier",
        default=None,
        help="Model tier to target when a family has multiple baselines.",
    )
    rollback_parser.add_argument(
        "--format",
        dest="format_name",
        default=None,
        help="Artifact format to target when a family has multiple baselines.",
    )
    rollback_parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to models.jsonl.",
    )
    rollback_parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_PATH,
        help="Path to gates/baseline.json.",
    )
    rollback_parser.add_argument(
        "--card-dir",
        type=Path,
        default=DEFAULT_CARD_DIR,
        help="Directory for regenerated model cards.",
    )
    rollback_parser.add_argument(
        "--status-path",
        type=Path,
        default=DEFAULT_STATUS_PATH,
        help="Path for the manifest-derived status snapshot.",
    )
    rollback_parser.add_argument(
        "--tracking-log",
        type=Path,
        default=DEFAULT_ROLLBACK_LOG_PATH,
        help="JSONL rollback tracking log path.",
    )
    rollback_parser.add_argument(
        "--reason",
        default=None,
        help="Optional rollback reason to include in the tracking log.",
    )
    rollback_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the rollback without writing files.",
    )
    rollback_parser.set_defaults(handler=handle_rollback)
    return rollback_parser


def handle_rollback(args: argparse.Namespace) -> int:
    """Execute ``openmed release rollback``."""

    try:
        result = rollback_manifest_pointer(
            family=args.family,
            tier=args.tier,
            format_name=args.format_name,
            manifest_path=args.manifest,
            baseline_path=args.baseline,
            card_dir=args.card_dir,
            status_path=args.status_path,
            tracking_log_path=args.tracking_log,
            reason=args.reason,
            dry_run=args.dry_run,
        )
    except (ManifestRollbackError, OSError, ValueError) as exc:
        sys.stderr.write(f"rollback failed: {exc}\n")
        return 1

    target = f"{result.family}/{result.tier or 'none'}/{result.format_name}"
    if result.changed:
        action = "Would roll back" if args.dry_run else "Rolled back"
        sys.stdout.write(
            f"{action} {target}: {result.previous_repo_id} -> {result.active_repo_id}\n"
        )
    else:
        sys.stdout.write(f"{target} already points to {result.active_repo_id}\n")

    verb = "Would regenerate" if args.dry_run else "Regenerated"
    if result.card_path is not None:
        sys.stdout.write(f"{verb} model card: {result.card_path}\n")
    verb = "Would refresh" if args.dry_run else "Refreshed"
    if result.status_path is not None:
        sys.stdout.write(f"{verb} manifest status: {result.status_path}\n")
    verb = "Would record" if args.dry_run else "Recorded"
    if result.tracking_log_path is not None:
        sys.stdout.write(f"{verb} rollback: {result.tracking_log_path}\n")
    return 0


def _help_handler(parser: argparse.ArgumentParser):
    def _handler(_: argparse.Namespace) -> int:
        parser.print_help()
        return 0

    return _handler


def build_parser() -> argparse.ArgumentParser:
    """Build a standalone parser for ``python -m openmed.cli.release``."""

    parser = argparse.ArgumentParser(
        prog="openmed release",
        description="OpenMed release management commands.",
    )
    subparsers = parser.add_subparsers(dest="release_command")
    _add_rollback_parser(subparsers)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Standalone release command entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
