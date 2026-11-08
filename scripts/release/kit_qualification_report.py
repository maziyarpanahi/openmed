#!/usr/bin/env python3
"""Collect OpenMedKit device-qualification evidence against the support matrix.

The library lives in ``openmed.eval.kit_qualification``; this script reads a
qualification plan and device-run evidence records from disk, writes the
qualification report and, on request, a matrix proposal and a proposed copy
of the matrix for review.

Usage::

    python scripts/release/kit_qualification_report.py --plan-skeleton plan.json
    python scripts/release/kit_qualification_report.py --plan plan.json \\
        --evidence runs/ --output report.json
    python scripts/release/kit_qualification_report.py --plan plan.json \\
        --evidence runs/ --output report.json --proposal proposal.json \\
        --evidence-ref eval/qualification/v4.0/report.json \\
        --proposed-matrix proposed-matrix.json

Output lines carry fixed codes, counts and controlled matrix identifiers only;
input file names and record values are never printed. The script makes no
network calls and never writes to the committed support matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# A direct ``python scripts/release/...`` invocation otherwise puts only
# ``scripts/release`` first on ``sys.path``; pin imports to this checkout.
_root_string = str(ROOT)
if _root_string in sys.path:
    sys.path.remove(_root_string)
sys.path.insert(0, _root_string)

from openmed.eval.kit_qualification import (  # noqa: E402
    MAX_EVIDENCE_RECORDS,
    QUALIFIED,
    TARGET_STATUSES,
    KitQualificationError,
    apply_matrix_proposal,
    collect_qualification,
    plan_skeleton,
    propose_matrix_updates,
)
from openmed.eval.kit_support_matrix import (  # noqa: E402
    DEFAULT_KIT_SUPPORT_MATRIX,
    KitSupportMatrixError,
    load_kit_support_matrix,
)

MAX_INPUT_BYTES = 1024 * 1024
PREFIX = "kit qualification"


class InputError(ValueError):
    """An input file could not be read or decoded."""


def _read_json(path: Path, code: str) -> Any:
    try:
        if path.stat().st_size > MAX_INPUT_BYTES:
            raise InputError(f"{code}_too_large")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise InputError(f"{code}_unreadable") from None


def evidence_paths(sources: Sequence[Path]) -> list[Path]:
    """Return evidence files from files and directories, in a stable order.

    Directories contribute their ``*.json`` files (not recursively), sorted by
    name.

    Args:
        sources: Evidence files or directories.

    Returns:
        Evidence file paths.

    Raises:
        InputError: If a source is missing or there are too many files.
    """
    paths: list[Path] = []
    for source in sources:
        if source.is_dir():
            paths.extend(
                sorted(path for path in source.glob("*.json") if path.is_file())
            )
        elif source.is_file():
            paths.append(source)
        else:
            raise InputError("evidence_source_missing")
        if len(paths) > MAX_EVIDENCE_RECORDS:
            raise InputError("evidence_too_many")
    return paths


def load_evidence(paths: Sequence[Path]) -> list[Any]:
    """Decode evidence files; an unreadable file becomes ``None``.

    Args:
        paths: Evidence file paths.

    Returns:
        One decoded value or ``None`` per path, in order.
    """
    records: list[Any] = []
    for path in paths:
        try:
            records.append(_read_json(path, "evidence"))
        except InputError:
            records.append(None)
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", "utf-8")


def _summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    targets = ", ".join(
        f"{summary['targets_by_status'][status]} {status}"
        for status in TARGET_STATUSES
        if summary["targets_by_status"][status]
    )
    return (
        f"cells {summary['cells_planned']} planned, "
        f"{summary['cells_qualified']} qualified, "
        f"{summary['cells_unplanned']} unplanned; "
        f"targets {targets or 'none'}; "
        f"records {summary['records_accepted']} accepted, "
        f"{summary['records_rejected']} rejected"
    )


def _same_file(first: Path, second: Path) -> bool:
    return first.expanduser().resolve() == second.expanduser().resolve()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the qualification collector.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` when an input is invalid or, with
        ``--require-qualified``, when a planned cell is not qualified.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_KIT_SUPPORT_MATRIX,
        help="Support matrix JSON (default: the packaged matrix).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", type=Path, help="Qualification plan JSON.")
    mode.add_argument(
        "--plan-skeleton",
        type=Path,
        metavar="OUT",
        help="Write an unfilled plan listing every qualification cell.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        action="append",
        default=[],
        help="Evidence record file or directory of *.json records (repeatable).",
    )
    parser.add_argument("--output", type=Path, help="Write the report JSON here.")
    parser.add_argument(
        "--proposal", type=Path, help="Write the matrix proposal JSON here."
    )
    parser.add_argument(
        "--proposed-matrix",
        type=Path,
        help="Write a proposed copy of the matrix here for review.",
    )
    parser.add_argument(
        "--evidence-ref",
        action="append",
        default=[],
        help="Repository path cited by proposed device-run evidence (repeatable).",
    )
    parser.add_argument(
        "--require-qualified",
        action="store_true",
        help="Exit 1 unless every planned cell is qualified.",
    )
    args = parser.parse_args(argv)
    if (args.proposal or args.proposed_matrix) and not args.evidence_ref:
        parser.error("--proposal and --proposed-matrix need --evidence-ref")
    if args.proposed_matrix and (
        _same_file(args.proposed_matrix, args.matrix)
        or _same_file(args.proposed_matrix, DEFAULT_KIT_SUPPORT_MATRIX)
    ):
        print(f"{PREFIX}: proposed_matrix_would_overwrite_matrix")
        return 1

    try:
        matrix = load_kit_support_matrix(args.matrix)
    except KitSupportMatrixError as exc:
        for problem in exc.problems:
            print(f"{PREFIX}: matrix:{problem}")
        return 1

    if args.plan_skeleton:
        _write_json(args.plan_skeleton, plan_skeleton(matrix))
        print(f"{PREFIX}: plan skeleton written")
        return 0

    try:
        plan = _read_json(args.plan, "plan")
        records = load_evidence(evidence_paths(args.evidence))
        report = collect_qualification(matrix, plan, records)
        proposal = None
        if args.proposal or args.proposed_matrix:
            proposal = propose_matrix_updates(
                matrix, report, evidence_refs=args.evidence_ref
            )
    except InputError as exc:
        print(f"{PREFIX}: {exc}")
        return 1
    except KitQualificationError as exc:
        for problem in exc.problems:
            print(f"{PREFIX}: {problem}")
        return 1

    if args.output:
        _write_json(args.output, report)
    if proposal is not None:
        if args.proposal:
            _write_json(args.proposal, proposal)
        if args.proposed_matrix:
            _write_json(args.proposed_matrix, apply_matrix_proposal(matrix, proposal))
        print(
            f"{PREFIX}: proposal {len(proposal['promotions'])} promotions, "
            f"{len(proposal['evidence_additions'])} evidence additions, "
            f"{len(proposal['reviews'])} reviews; not applied"
        )
    for row in report["rejected_records"]:
        print(f"{PREFIX}: rejected {','.join(row['reasons'])}")
    print(f"{PREFIX}: {_summary(report)}")
    if args.require_qualified:
        unqualified = [
            row["cell"]
            for row in report["cells"]
            if row["planned"] and not row["qualified"]
        ]
        for cell in unqualified:
            print(f"{PREFIX}: not_qualified:{cell}")
        if unqualified or not report["summary"]["targets_by_status"][QUALIFIED]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
