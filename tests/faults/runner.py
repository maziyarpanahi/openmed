"""Deterministic matrix runner producing a value-free fault result manifest.

Every case runs twice in fresh directories: a control run (seams count passes
but never fire) proves the seam sits on the component's real call path and
records the unfaulted baseline; a fault run fires the scheduled fault and
records the observed outcome. Manifests contain identifiers, enums, counts and
digests only, so the same plan and seed always produce the same bytes.

Usage (``OPENMED_PROFILE=test`` is required)::

    OPENMED_PROFILE=test python -m tests.faults.runner --output manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .plan import (
    AMBIGUOUS,
    MANIFEST_SCHEMA_PATH,
    SCHEMA_VERSION,
    FaultPlan,
    ResolvedCase,
    canonical_json,
    load_json_schema,
    sha256_digest,
)
from .scenarios import Component, Observation, component_for
from .seams import FaultInjector, FaultSession, SimulatedProcessKill


def _execute(
    component: Component,
    resolved: ResolvedCase,
    workdir: Path,
    *,
    mode: str,
) -> tuple[Observation, FaultInjector]:
    injector = FaultInjector(resolved, mode=mode)
    session = FaultSession(injector, workdir)
    with injector.armed():
        try:
            observation = component.run(session)
        except SimulatedProcessKill:
            observation = Observation(AMBIGUOUS, "unhandled_process_kill")
        except asyncio.CancelledError:
            observation = Observation(AMBIGUOUS, "unhandled_cancellation")
        except Exception:
            observation = Observation(AMBIGUOUS, "unhandled_exception")
    return observation, injector


def run_case(resolved: ResolvedCase, workdir: Path) -> dict[str, Any]:
    """Run one resolved case (control and fault) and return its manifest row.

    Args:
        resolved: Case bound to its plan seed.
        workdir: Empty scratch directory owned by this case.

    Returns:
        A value-free manifest row.
    """
    case = resolved.case
    component = component_for(case)
    control, control_injector = _execute(
        component, resolved, workdir / "control", mode="control"
    )
    observed, fault_injector = _execute(
        component, resolved, workdir / "fault", mode="fault"
    )
    baseline = component.baseline(case)
    control_hits = control_injector.hits()
    residue_ok = not case.require_no_residue or observed.residue == 0
    if control_hits == 0:
        reason = "seam_not_reached"
    elif control.outcome != baseline:
        reason = "control_outcome_mismatch"
    elif fault_injector.fired == 0:
        reason = "fault_not_fired"
    elif observed.outcome != case.expected_outcome:
        reason = observed.reason_code
    elif not residue_ok:
        reason = "residue_present"
    else:
        reason = observed.reason_code
    passed = (
        control_hits > 0
        and control.outcome == baseline
        and fault_injector.fired > 0
        and observed.outcome == case.expected_outcome
        and residue_ok
    )
    if passed:
        status = "passed"
    elif case.known_gap is not None:
        status = "known_gap"
    else:
        status = "failed"
    return {
        "case_id": case.case_id,
        "case_digest": resolved.case_digest,
        "component": case.component,
        "commit_point": component.commit_point,
        "category": case.category,
        "fault": case.fault,
        "injection_point": case.injection_point,
        "phase": case.phase,
        "fault_call": resolved.fault_call,
        "persistent": case.trigger.persistent,
        "expected_outcome": case.expected_outcome,
        "baseline_outcome": baseline,
        "control_outcome": control.outcome,
        "observed_outcome": observed.outcome,
        "control_seam_hits": control_hits,
        "fault_seam_hits": fault_injector.hits(),
        "faults_fired": fault_injector.fired,
        "residue_count": observed.residue,
        "require_no_residue": case.require_no_residue,
        "reason_code": reason,
        "status": status,
        "known_gap_issue": None if case.known_gap is None else case.known_gap.issue,
    }


def build_manifest(plan: FaultPlan, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Assemble and digest a manifest from per-case rows."""
    statuses = Counter(row["status"] for row in rows)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "plan_digest": plan.digest(),
        "seed": plan.seed,
        "cases": list(rows),
        "summary": {
            "total": len(rows),
            "passed": statuses.get("passed", 0),
            "failed": statuses.get("failed", 0),
            "known_gap": statuses.get("known_gap", 0),
            "categories": dict(
                sorted(Counter(row["category"] for row in rows).items())
            ),
            "outcomes": dict(
                sorted(Counter(row["observed_outcome"] for row in rows).items())
            ),
        },
    }
    manifest["manifest_digest"] = sha256_digest(canonical_json(manifest))
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate ``manifest`` against the bundled manifest schema.

    Raises:
        jsonschema.ValidationError: When the manifest does not conform.
    """
    Draft202012Validator(load_json_schema(MANIFEST_SCHEMA_PATH)).validate(manifest)


def run_plan(plan: FaultPlan, workdir: Path) -> dict[str, Any]:
    """Run every case of ``plan`` under ``workdir`` and return the manifest."""
    rows = [
        run_case(resolved, workdir / f"case-{index:03d}")
        for index, resolved in enumerate(plan.resolved())
    ]
    manifest = build_manifest(plan, rows)
    validate_manifest(manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run a plan offline and write its value-free manifest.

    Returns:
        ``0`` when no case failed (known gaps allowed), ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = FaultPlan.load(args.plan) if args.plan else FaultPlan.load()
    if args.seed is not None:
        plan = plan.with_seed(args.seed)
    with tempfile.TemporaryDirectory(prefix="openmed-faults-") as scratch:
        manifest = run_plan(plan, Path(scratch))
    args.output.write_bytes(canonical_json(manifest) + b"\n")
    return 0 if manifest["summary"]["failed"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover - manual evidence entry point
    sys.exit(main())
