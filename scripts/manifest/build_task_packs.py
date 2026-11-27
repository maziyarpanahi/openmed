#!/usr/bin/env python3
"""Derive the portable task pack manifest from the committed models.jsonl.

The command is offline: it reads ``models.jsonl``, applies the pack
declarations in ``openmed/core/task_packs.py`` and writes
``openmed/core/data/task_packs.json``. ``--check`` writes nothing and exits
non-zero when the committed file differs from a fresh build.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openmed.core.model_registry import load_manifest_rows  # noqa: E402
from openmed.core.task_packs import (  # noqa: E402
    TaskPackError,
    build_task_packs,
    render_task_packs,
)

DEFAULT_MANIFEST = ROOT / "models.jsonl"
DEFAULT_OUTPUT = ROOT / "openmed" / "core" / "data" / "task_packs.json"


def build_text(manifest: Path = DEFAULT_MANIFEST) -> str:
    """Return the canonical task pack manifest text for a models.jsonl file.

    Args:
        manifest: Path to the model manifest.

    Returns:
        Byte-stable JSON text ending with a newline.
    """

    return render_task_packs(build_task_packs(load_manifest_rows(manifest)))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the committed manifest is stale; write nothing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build or check the task pack manifest and return an exit code."""

    args = build_parser().parse_args(argv)
    try:
        text = build_text(args.manifest)
    except (OSError, ValueError) as exc:
        reason = exc.reason if isinstance(exc, TaskPackError) else type(exc).__name__
        print(f"task pack build failed: {reason}", file=sys.stderr)
        return 1
    if args.check:
        try:
            committed = args.output.read_text(encoding="utf-8")
        except OSError:
            committed = None
        if committed != text:
            print(
                "task_packs.json is stale; run "
                "python scripts/manifest/build_task_packs.py",
                file=sys.stderr,
            )
            return 1
        print("task_packs.json matches models.jsonl.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
