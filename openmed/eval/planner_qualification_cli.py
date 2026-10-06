"""Offline command for a caller-supplied, trusted local planner callable."""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Sequence

from openmed.eval.planner_qualification import _DiscardOutput, qualify_planner


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, "invalid_arguments\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Write counts-only evidence; exit 0 qualified, 1 refused, or 2 unavailable.

    Args:
        argv: Command arguments, or process arguments when omitted.

    Returns:
        Exit code. No module errors, planner output, or private paths are echoed.
    """
    parser = _Parser(
        description="Qualify a trusted local planner offline; no dispatch."
    )
    parser.add_argument("--planner", required=True, help="Local module:callable")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if (
        re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", args.planner)
        is None
    ):
        print("invalid_planner_reference", file=sys.stderr)
        return 2
    module, name = args.planner.split(":")
    try:
        with redirect_stdout(_DiscardOutput()), redirect_stderr(_DiscardOutput()):
            planner = getattr(importlib.import_module(module), name)
        if not callable(planner):
            raise ValueError("invalid_planner")
        report = qualify_planner(planner)
        args.output.write_text(report.to_json() + "\n", encoding="utf-8")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        print("qualification_unavailable", file=sys.stderr)
        return 2
    return 0 if report.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
