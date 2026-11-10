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
``shadow``    Run champion and challenger on shared golden + SHIELD fixtures.
``advance``   Promote a key to its next phase, guarded by a ``RELEASABLE``
              gate report.
``apply-gate`` Apply a scored phase gate and atomically flip local pointers.
``rollback``  Roll a live canary/stable key back to its last green target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from openmed.core.manifest import (
    DEFAULT_CARD_DIR,
    DEFAULT_ROLLBACK_LOG_PATH,
    DEFAULT_STATUS_PATH,
    load_manifest_rows,
    rollback_manifest_pointer,
    write_manifest_rows,
)
from openmed.core.manifest_diff import diff_manifests
from openmed.core.model_registry import MANIFEST_PATH
from openmed.eval.release_gates import GateReport
from openmed.eval.rollout import (
    ROLLOUT_STATE_PATH,
    PhaseState,
    RolloutCoordinator,
    RolloutError,
    RolloutStateMachine,
    run_shadow_comparison,
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

    shadow = subparsers.add_parser(
        "shadow",
        help="Compare champion/challenger on shared golden and SHIELD fixtures.",
    )
    shadow.add_argument("--champion", required=True)
    shadow.add_argument("--challenger", required=True)
    shadow.add_argument("--family", required=True)
    shadow.add_argument("--tier", required=True)
    shadow.add_argument("--format", required=True)
    shadow.add_argument("--golden-fixture", required=True, type=Path)
    shadow.add_argument("--shield-fixture", required=True, type=Path)
    shadow.add_argument("--baseline", required=True, type=Path)
    shadow.add_argument("--device", default="cpu")
    shadow.add_argument("--output", required=True, type=Path)

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
                required=name == "seed",
                help="Opaque rollout-target version pointer (e.g. v3).",
            )
        if name == "seed":
            sub.add_argument(
                "--last-green",
                required=True,
                help="Shipping latest pointer retained during shadow/canary.",
            )
        if name == "advance":
            sub.add_argument(
                "--gate-report",
                required=True,
                type=Path,
                help="Path to a serialized RELEASABLE GateReport JSON file.",
            )

    apply_gate = subparsers.add_parser(
        "apply-gate",
        help="Apply a scored phase gate and execute local pointer changes.",
    )
    apply_gate.add_argument("--family", required=True)
    apply_gate.add_argument("--tier", required=True)
    apply_gate.add_argument("--format", required=True)
    apply_gate.add_argument("--gate-report", required=True, type=Path)
    apply_gate.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    apply_gate.add_argument("--staged-manifest", required=True, type=Path)
    apply_gate.add_argument("--baseline", required=True, type=Path)
    apply_gate.add_argument("--card-dir", type=Path, default=DEFAULT_CARD_DIR)
    apply_gate.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    apply_gate.add_argument(
        "--tracking-log",
        type=Path,
        default=DEFAULT_ROLLBACK_LOG_PATH,
    )
    apply_gate.add_argument("--audit-output", required=True, type=Path)
    apply_gate.add_argument("--result-output", required=True, type=Path)
    return parser


def _manifest_target(path: Path, target: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in load_manifest_rows(path)
            if str(row.get("repo_id") or "") == target
        ),
        None,
    )


def _require_matching_manifest_target(
    path: Path,
    target: str | None,
    state: PhaseState,
) -> str:
    if not target:
        raise ValueError("rollout pointer target is missing")
    row = _manifest_target(path, target)
    if row is None:
        raise ValueError(f"manifest does not contain rollout target: {target}")
    raw_formats = row.get("formats")
    if not isinstance(raw_formats, (list, tuple)):
        raise ValueError("manifest rollout target formats must be an array")
    row_formats = {str(value).strip().casefold() for value in raw_formats}
    if (
        str(row.get("family") or "").strip().casefold()
        != state.family.strip().casefold()
        or str(row.get("tier") or "").strip().casefold()
        != state.tier.strip().casefold()
        or state.format.strip().casefold() not in row_formats
    ):
        raise ValueError("manifest rollout target coordinates do not match state")
    return target


def _canary_action(
    *,
    live_manifest: Path,
    staged_manifest: Path,
):
    def apply(state, _gate_report: GateReport) -> str:
        manifest_diff = diff_manifests(live_manifest, staged_manifest)
        if manifest_diff.has_removed:
            raise ValueError("staged canary manifest removes a shipping model")
        _require_matching_manifest_target(staged_manifest, state.target, state)
        _require_matching_manifest_target(live_manifest, state.last_green, state)
        return str(state.target)

    return apply


def _stable_action(*, staged_manifest: Path, live_manifest: Path):
    def apply(state, _gate_report: GateReport) -> str:
        manifest_diff = diff_manifests(live_manifest, staged_manifest)
        if manifest_diff.has_removed:
            raise ValueError("staged stable manifest removes a shipping model")
        _require_matching_manifest_target(staged_manifest, state.target, state)
        write_manifest_rows(load_manifest_rows(staged_manifest), live_manifest)
        return str(state.target)

    return apply


def _rollback_action(
    *,
    manifest: Path,
    baseline: Path,
    card_dir: Path,
    status_path: Path,
    tracking_log: Path,
):
    def apply(state, _gate_report: GateReport) -> str:
        result = rollback_manifest_pointer(
            family=state.family,
            tier=state.tier,
            format_name=state.format,
            manifest_path=manifest,
            baseline_path=baseline,
            card_dir=card_dir,
            status_path=status_path,
            tracking_log_path=tracking_log,
            reason="automatic canary gate rollback",
        )
        return result.active_repo_id

    return apply


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


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

    if args.command == "shadow":
        try:
            comparisons = run_shadow_comparison(
                {
                    "golden": args.golden_fixture,
                    "shield": args.shield_fixture,
                },
                champion_model=args.champion,
                challenger_model=args.challenger,
                family=args.family,
                tier=args.tier,
                format=args.format,
                baseline_path=args.baseline,
                device=args.device,
            )
            _write_json(
                args.output,
                {"comparisons": [item.to_dict() for item in comparisons]},
            )
        except (RolloutError, OSError, TypeError, ValueError) as exc:
            print(f"rollout shadow failed: {exc}", file=sys.stderr)
            return 1
        print(f"rollout shadow evidence: {args.output}")
        return 0

    try:
        if args.command == "seed":
            state = machine.seed(
                args.family,
                args.tier,
                args.format,
                target=args.target,
                last_green=args.last_green,
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
        elif args.command == "rollback":
            state = machine.rollback(args.family, args.tier, args.format)
        else:  # apply-gate
            gate_report = _load_gate_report(args.gate_report)
            coordinator = RolloutCoordinator(
                machine,
                canary_action=_canary_action(
                    live_manifest=args.manifest,
                    staged_manifest=args.staged_manifest,
                ),
                stable_action=_stable_action(
                    staged_manifest=args.staged_manifest,
                    live_manifest=args.manifest,
                ),
                rollback_action=_rollback_action(
                    manifest=args.manifest,
                    baseline=args.baseline,
                    card_dir=args.card_dir,
                    status_path=args.status_path,
                    tracking_log=args.tracking_log,
                ),
            )
            application = coordinator.apply_gate(
                args.family,
                args.tier,
                args.format,
                gate_report,
            )
            state = application.state
            machine.save(args.state)
            machine.save_audit(args.audit_output)
            _write_json(
                args.result_output,
                {
                    "action": application.action,
                    "audit": application.audit_record.to_dict(),
                    "state": state.to_dict(),
                },
            )
            print(f"{state.key}: {application.action} -> {state.phase}")
            return 0 if application.action in {"canary", "stable"} else 2
    except (RolloutError, OSError, RuntimeError, ValueError) as exc:
        print(f"rollout {args.command} failed: {exc}", file=sys.stderr)
        return 1

    machine.save(args.state)
    print(f"{state.key}: -> {state.phase}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
