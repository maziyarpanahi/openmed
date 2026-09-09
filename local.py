"""Run the Health Universe agent on already-extracted local Markdown."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from health_universe_a2a.local import create_local_context

from main import agent


def parser() -> argparse.ArgumentParser:
    """Build the local runner parser."""

    command = argparse.ArgumentParser()
    command.add_argument(
        "data_dir",
        type=Path,
        help="Folder containing source/ with already-extracted .md files",
    )
    command.add_argument("--output-dir", type=Path)
    return command


async def run() -> int:
    """Run one local Markdown job."""

    args = parser().parse_args()
    source_dir = args.data_dir / "source"
    if not source_dir.is_dir():
        raise SystemExit("data_dir must contain a source/ folder.")
    unsupported = [
        path for path in source_dir.iterdir() if path.is_file() and path.suffix != ".md"
    ]
    if unsupported:
        raise SystemExit(
            "Local mode accepts already-extracted .md files only. Production uses "
            "Health Universe platform OCR."
        )
    context = create_local_context(
        str(args.data_dir),
        str(args.output_dir) if args.output_dir else None,
    )
    result = await agent.process_message("Replace PHI", context)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
