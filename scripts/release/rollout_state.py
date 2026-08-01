#!/usr/bin/env python3
"""Inspect and drive the committed shadow/canary/stable rollout state.

This thin CLI wraps :class:`openmed.eval.rollout.RolloutStateMachine` so a
release operator can read and evolve ``gates/rollout_state.json`` from committed
inputs only -- a stored state document and serialized ``GateReport`` files. It
performs no live model-registry or Hugging Face call.

Subcommands
-----------
``show``      Print the current phase of every key (or one key).
``seed``      Register a challenger in ``SHADOW`` keyed by family/tier/format.
``advance``   Promote a key to its next phase, guarded by a ``RELEASABLE``
              gate report.
``rollback``  Roll a live canary/stable key back to its last green target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openmed.eval.release_gates import GateReport
from openmed.eval.rollout import (
    ROLLOUT_STATE_PATH,
    RolloutError,
    RolloutStateMachine,
)


def _load_machine(path: Path) -> RolloutStateMachine:
    if path.is_file():
        return RolloutStateMachine.load(path)
    return RolloutStateMachine()


def _load_gate_report(path: Path) -> GateReport:
    return GateReport.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _print_state(machine: RolloutStateMachine) -> None:
    if not machine.entries:
        print("rollout: no keys under rollout")
        return
    for key, state in sorted(machine.entries.items()):
        target = state.target or "-"
        last_green = state.last_green or "-"
        print(
            f"{key}: {state.phase} "
            f"(target={target}, last_green={last_green}, "
            f"entered_at={state.entered_at.isoformat()})"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the rollout state-machine command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        default=ROLLOUT_STATE_PATH,
        help=f"Committed rollout state document (default: {ROLLOUT_STATE_PATH}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show", help="Print current rollout phases.")
    show.add_argument("--json", action="store_true", help="Emit the raw document.")

    for name, help_text in (
        ("seed", "Register a challenger in SHADOW."),
        ("advance", "Promote a key to its next phase (gate-guarded)."),
        ("rollback", "Roll a live key back to its last green target."),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--family", required=True)
        sub.add_argument("--tier", required=True)
        sub.add_argument("--format", required=True)
        if name in {"seed", "advance"}:
            sub.add_argument(
                "--target",
                help="Opaque rollout-target version pointer (e.g. v3).",
            )
        if name == "advance":
            sub.add_argument(
                "--gate-report",
                required=True,
                type=Path,
                help="Path to a serialized RELEASABLE GateReport JSON file.",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the rollout state-machine CLI over committed inputs only."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    machine = _load_machine(args.state)

    if args.command == "show":
        if args.json:
            print(machine.to_json(), end="")
        else:
            _print_state(machine)
        return 0

    try:
        if args.command == "seed":
            state = machine.seed(
                args.family, args.tier, args.format, target=args.target
            )
        elif args.command == "advance":
            gate_report = _load_gate_report(args.gate_report)
            state = machine.advance(
                args.family,
                args.tier,
                args.format,
                gate_report,
                target=args.target,
            )
        else:  # rollback
            state = machine.rollback(args.family, args.tier, args.format)
    except (RolloutError, OSError, ValueError) as exc:
        print(f"rollout {args.command} failed: {exc}", file=sys.stderr)
        return 1

    machine.save(args.state)
    print(f"{state.key}: -> {state.phase}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
