#!/usr/bin/env python3
"""Score aggregate retraining signals and prepare recipe-only proposals."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from openmed.core.audit import stable_hash
from openmed.eval.drift_monitor import assert_no_raw_text
from openmed.eval.retrain_trigger import (
    RETRAIN_QUEUE_SCHEMA_VERSION,
    RetrainDecision,
    RetrainTriggerPolicy,
    RetrainTriggerResult,
    load_retrain_trigger_inputs,
    score_retrain_candidates,
    write_retrain_trigger_artifacts,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = ROOT / "gates" / "retrain_trigger_inputs.json"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "retrain-trigger"
DEFAULT_RECIPES_DIR = ROOT / "recipes"
RETRAIN_RECIPE_SCHEMA_VERSION = "openmed.retrain_recipe.v1"
RETRAIN_DISPATCH_SUMMARY_SCHEMA_VERSION = "openmed.retrain_dispatch_summary.v1"


@dataclass(frozen=True)
class RetrainDispatchSummary:
    """Stable summary consumed by the scheduled workflow."""

    queue_count: int
    evidence_hash: str
    changed_families: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-ready summary."""

        payload: dict[str, Any] = {
            "changed_families": list(self.changed_families),
            "changed_recipe_count": len(self.changed_families),
            "evidence_hash": self.evidence_hash,
            "queue_count": self.queue_count,
            "schema_version": RETRAIN_DISPATCH_SUMMARY_SCHEMA_VERSION,
        }
        payload["summary_hash"] = stable_hash(payload)
        assert_no_raw_text(payload, where="retrain dispatch summary")
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the summary with stable ordering."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )


def update_recipe_configs(
    result: RetrainTriggerResult,
    *,
    recipes_dir: str | Path,
) -> tuple[str, ...]:
    """Update only recipe files for queued families and return changed families."""

    grouped: dict[str, list[RetrainDecision]] = {}
    for decision in result.queued:
        grouped.setdefault(decision.candidate.family, []).append(decision)
    if not grouped:
        return ()

    evidence_hash = str(result.to_evidence_dict()["artifact_hash"])
    root = Path(recipes_dir)
    pending: list[tuple[str, Path, str]] = []
    for family, decisions in sorted(grouped.items()):
        path = root / f"{family}.yaml"
        existing = _load_recipe(path)
        proposal = dict(existing)
        proposal["retraining_trigger"] = _trigger_recipe_section(
            decisions,
            evidence_hash=evidence_hash,
        )
        rendered = yaml.safe_dump(
            proposal,
            allow_unicode=False,
            default_flow_style=False,
            sort_keys=True,
            width=88,
        )
        if path.is_file() and path.read_text(encoding="utf-8") == rendered:
            continue
        pending.append((family, path, rendered))

    if pending:
        root.mkdir(parents=True, exist_ok=True)
    for _, path, rendered in pending:
        path.write_text(rendered, encoding="utf-8")
    return tuple(family for family, _, _ in pending)


def run_retrain_dispatch(
    *,
    input_path: str | Path,
    queue_path: str | Path,
    evidence_path: str | Path,
    summary_path: str | Path,
    recipes_dir: str | Path,
    policy: RetrainTriggerPolicy | None = None,
) -> RetrainDispatchSummary:
    """Run offline scoring, write evidence, and prepare deterministic recipes."""

    candidates = load_retrain_trigger_inputs(input_path)
    result = score_retrain_candidates(candidates, policy=policy)
    write_retrain_trigger_artifacts(
        result,
        queue_path=queue_path,
        evidence_path=evidence_path,
    )
    changed_families = update_recipe_configs(result, recipes_dir=recipes_dir)
    evidence_hash = str(result.to_evidence_dict()["artifact_hash"])
    summary = RetrainDispatchSummary(
        queue_count=len(result.queued),
        evidence_hash=evidence_hash,
        changed_families=changed_families,
    )
    output = Path(summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summary.to_json() + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the offline retrain-dispatch argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--queue-output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "retrain_queue.jsonl",
    )
    parser.add_argument(
        "--evidence-output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "decision_evidence.json",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "dispatch_summary.json",
    )
    parser.add_argument("--recipes-dir", type=Path, default=DEFAULT_RECIPES_DIR)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--drift-weight", type=float, default=1.0)
    parser.add_argument("--gate-failure-weight", type=float, default=1.0)
    parser.add_argument("--staleness-weight", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the aggregate-only retrain recipe dispatcher."""

    args = build_parser().parse_args(argv)
    try:
        summary = run_retrain_dispatch(
            input_path=args.input,
            queue_path=args.queue_output,
            evidence_path=args.evidence_output,
            summary_path=args.summary_output,
            recipes_dir=args.recipes_dir,
            policy=RetrainTriggerPolicy(
                threshold=args.threshold,
                drift_weight=args.drift_weight,
                gate_failure_weight=args.gate_failure_weight,
                staleness_weight=args.staleness_weight,
            ),
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Retrain dispatch failed: {exc}", file=sys.stderr)
        return 2

    print(summary.to_json())
    return 0


def _load_recipe(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"recipe must contain a YAML object: {path.name}")
    if any(not isinstance(key, str) for key in payload):
        raise ValueError(f"recipe keys must be strings: {path.name}")
    return payload


def _trigger_recipe_section(
    decisions: Sequence[RetrainDecision],
    *,
    evidence_hash: str,
) -> dict[str, Any]:
    candidates = [_recipe_candidate(decision) for decision in decisions]
    section: dict[str, Any] = {
        "candidates": candidates,
        "downstream_release_gates": {
            "required": True,
            "workflow": "release-gates.yml",
        },
        "evidence_hash": evidence_hash,
        "queue_schema_version": RETRAIN_QUEUE_SCHEMA_VERSION,
        "schema_version": RETRAIN_RECIPE_SCHEMA_VERSION,
    }
    section["proposal_hash"] = stable_hash(section)
    assert_no_raw_text(section, where="retrain recipe proposal")
    return section


def _recipe_candidate(decision: RetrainDecision) -> dict[str, Any]:
    queue_row = decision.to_queue_dict()
    return {
        "decision_hash": queue_row["decision_hash"],
        "dominant_label": queue_row["dominant_label"],
        "dominant_signal": queue_row["dominant_signal"],
        "failed_gates": queue_row["failed_gates"],
        "recipe_slices": queue_row["recipe_slices"],
        "score": queue_row["score"],
        "source_hashes": queue_row["source_hashes"],
        "threshold": queue_row["threshold"],
        "tier": queue_row["tier"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
