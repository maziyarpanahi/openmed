#!/usr/bin/env python3
"""Evaluate GLiNER2.5 language decisions against the deterministic router.

Runs the synthetic language, script, dialect and code-switch questions from
``openmed.eval.language_decisions`` through a pinned GLiNER2.5 checkpoint and
the stdlib language router, then writes an aggregate report with counts,
rates and digests only. No fixture text is printed or written.

Reproduce the committed summary::

    uv pip install -e ".[gliner2]"
    python scripts/benchmarks/gliner2_language_decisions.py \
        --output docs/benchmarks/gliner2-language-decisions.json

Check a fresh run against the committed summary without rewriting it::

    python scripts/benchmarks/gliner2_language_decisions.py \
        --check docs/benchmarks/gliner2-language-decisions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from openmed.eval.language_decisions import (
    GLINER2_LANGUAGE_DECISION_MODEL,
    GLINER2_LANGUAGE_DECISION_REVISION,
    LANGUAGE_IN_SCRIPT,
    format_language_decision_summary,
    format_language_table,
    format_report_json,
    report_differences,
    run_gliner2_language_decision_eval,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default=GLINER2_LANGUAGE_DECISION_MODEL)
    parser.add_argument(
        "--revision",
        default=GLINER2_LANGUAGE_DECISION_REVISION,
        help="Hugging Face commit to pin (default: the evaluated commit).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Never download; fail if the pinned snapshot is not cached.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the JSON report here instead of standard output.",
    )
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        help="Compare the run with a committed report and exit 1 on drift.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=2,
        help="Allowed per-count drift for model-dependent counts in --check.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation and write, print or check the report."""

    args = parse_args(argv)
    report: dict[str, Any] = run_gliner2_language_decision_eval(
        model_id=args.model_id,
        revision=args.revision or None,
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=True if args.local_files_only else None,
    )
    text = format_report_json(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    elif args.check is None:
        sys.stdout.write(text)
    sys.stderr.write(format_language_decision_summary(report))
    sys.stderr.write("\n" + format_language_table(report, LANGUAGE_IN_SCRIPT))

    if args.check is not None:
        expected = json.loads(args.check.read_text(encoding="utf-8"))
        differences = report_differences(expected, report, tolerance=args.tolerance)
        for difference in differences:
            sys.stderr.write(f"drift: {difference}\n")
        if differences:
            return 1
        sys.stderr.write("report matches the committed summary\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
