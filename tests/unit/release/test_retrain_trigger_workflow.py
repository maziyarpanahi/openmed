"""Tests for the aggregate retrain queue and recipe-only workflow."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from openmed.eval.retrain_trigger import (
    RETRAIN_TRIGGER_INPUT_SCHEMA_VERSION,
    RetrainCandidateSignals,
    RetrainRecipeSlice,
)

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "retrain_queue.py"
WORKFLOW = ROOT / ".github" / "workflows" / "retrain-trigger.yml"
INPUTS = ROOT / "gates" / "retrain_trigger_inputs.json"
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64

spec = importlib.util.spec_from_file_location("scripts.release.retrain_queue", SCRIPT)
assert spec is not None and spec.loader is not None
retrain_queue = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = retrain_queue
spec.loader.exec_module(retrain_queue)


def _write_inputs(path: Path, candidates: list[RetrainCandidateSignals]) -> None:
    payload = {
        "candidates": [candidate.to_dict() for candidate in candidates],
        "schema_version": RETRAIN_TRIGGER_INPUT_SCHEMA_VERSION,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run(
    root: Path,
    candidates: list[RetrainCandidateSignals],
) -> tuple[Any, Path, Path, Path, Path]:
    root.mkdir(parents=True)
    input_path = root / "input.json"
    queue_path = root / "artifacts" / "retrain_queue.jsonl"
    evidence_path = root / "artifacts" / "decision_evidence.json"
    summary_path = root / "artifacts" / "dispatch_summary.json"
    recipes_dir = root / "recipes"
    _write_inputs(input_path, candidates)
    summary = retrain_queue.run_retrain_dispatch(
        input_path=input_path,
        queue_path=queue_path,
        evidence_path=evidence_path,
        summary_path=summary_path,
        recipes_dir=recipes_dir,
    )
    return summary, queue_path, evidence_path, summary_path, recipes_dir


def _queued_candidate(*, family: str = "pii") -> RetrainCandidateSignals:
    return RetrainCandidateSignals(
        family=family,
        tier="tiny",
        gate_failure_ratio=1.25,
        dominant_label="SSN",
        failed_gates=("G3",),
        source_hashes={"gate": HASH_A},
        recipe_slices=(
            RetrainRecipeSlice(
                rank=1,
                label="SSN",
                language="en",
                candidate_count=3,
                priority_score=4.5,
                evidence_hash=HASH_B,
                fixture_hashes=(HASH_A,),
            ),
        ),
    )


def test_empty_queue_writes_evidence_without_recipe_change(tmp_path: Path) -> None:
    summary, queue_path, evidence_path, summary_path, recipes_dir = _run(
        tmp_path / "empty",
        [RetrainCandidateSignals(family="pii", tier="tiny")],
    )

    assert summary.queue_count == 0
    assert summary.changed_families == ()
    assert queue_path.read_bytes() == b""
    assert json.loads(evidence_path.read_text())["queue_count"] == 0
    assert json.loads(summary_path.read_text())["changed_recipe_count"] == 0
    assert not recipes_dir.exists()


def test_queued_family_changes_only_its_recipe_config(tmp_path: Path) -> None:
    root = tmp_path / "queued"
    recipes_dir = root / "recipes"
    recipes_dir.mkdir(parents=True)
    pii_path = recipes_dir / "pii.yaml"
    other_path = recipes_dir / "structured.yaml"
    pii_path.write_text("optimizer:\n  learning_rate: 0.0001\n")
    other_path.write_text("owner: structured\n")
    other_before = other_path.read_bytes()
    input_path = root / "input.json"
    _write_inputs(
        input_path,
        [
            RetrainCandidateSignals(family="structured", tier="tiny"),
            _queued_candidate(),
        ],
    )

    summary = retrain_queue.run_retrain_dispatch(
        input_path=input_path,
        queue_path=root / "artifacts" / "queue.jsonl",
        evidence_path=root / "artifacts" / "evidence.json",
        summary_path=root / "artifacts" / "summary.json",
        recipes_dir=recipes_dir,
    )

    assert summary.queue_count == 1
    assert summary.changed_families == ("pii",)
    assert other_path.read_bytes() == other_before
    recipe = yaml.safe_load(pii_path.read_text())
    assert recipe["optimizer"] == {"learning_rate": 0.0001}
    trigger = recipe["retraining_trigger"]
    assert trigger["evidence_hash"] == summary.evidence_hash
    assert trigger["downstream_release_gates"] == {
        "required": True,
        "workflow": "release-gates.yml",
    }
    candidate = trigger["candidates"][0]
    assert candidate["dominant_signal"] == "gate-failure"
    assert candidate["failed_gates"] == ["G3"]
    assert candidate["recipe_slices"][0]["priority_score"] == 4.5
    assert candidate["source_hashes"] == {"gate": HASH_A}
    assert trigger["proposal_hash"].startswith("sha256:")


def test_reversed_inputs_produce_byte_identical_outputs(tmp_path: Path) -> None:
    pii = _queued_candidate()
    clinical = _queued_candidate(family="clinical")
    first = _run(tmp_path / "first", [pii, clinical])
    second = _run(tmp_path / "second", [clinical, pii])

    _, first_queue, first_evidence, _, first_recipes = first
    _, second_queue, second_evidence, _, second_recipes = second
    assert first_queue.read_bytes() == second_queue.read_bytes()
    assert first_evidence.read_bytes() == second_evidence.read_bytes()
    assert (first_recipes / "pii.yaml").read_bytes() == (
        second_recipes / "pii.yaml"
    ).read_bytes()
    assert (first_recipes / "clinical.yaml").read_bytes() == (
        second_recipes / "clinical.yaml"
    ).read_bytes()


def test_identical_recipe_is_not_reported_as_changed(tmp_path: Path) -> None:
    root = tmp_path / "repeat"
    first = _run(root, [_queued_candidate()])
    recipe_path = first[-1] / "pii.yaml"
    before = recipe_path.read_bytes()

    summary = retrain_queue.run_retrain_dispatch(
        input_path=root / "input.json",
        queue_path=root / "artifacts" / "retrain_queue.jsonl",
        evidence_path=root / "artifacts" / "decision_evidence.json",
        summary_path=root / "artifacts" / "dispatch_summary.json",
        recipes_dir=root / "recipes",
    )

    assert summary.queue_count == 1
    assert summary.changed_families == ()
    assert recipe_path.read_bytes() == before


def test_cli_rejects_raw_text_without_writing_a_recipe(
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "unsafe"
    root.mkdir()
    input_path = root / "input.json"
    payload = {
        "schema_version": RETRAIN_TRIGGER_INPUT_SCHEMA_VERSION,
        "candidates": [
            {
                **RetrainCandidateSignals(family="pii", tier="tiny").to_dict(),
                "raw_text": "Synthetic patient context",
            }
        ],
    }
    input_path.write_text(json.dumps(payload))

    status = retrain_queue.main(
        [
            "--input",
            str(input_path),
            "--queue-output",
            str(root / "queue.jsonl"),
            "--evidence-output",
            str(root / "evidence.json"),
            "--summary-output",
            str(root / "summary.json"),
            "--recipes-dir",
            str(root / "recipes"),
        ]
    )

    assert status == 2
    assert "raw-text" in capsys.readouterr().err
    assert not (root / "recipes").exists()


def test_committed_input_is_a_noop_aggregate_contract(tmp_path: Path) -> None:
    payload = json.loads(INPUTS.read_text())
    assert payload == {
        "candidates": [],
        "schema_version": RETRAIN_TRIGGER_INPUT_SCHEMA_VERSION,
    }

    summary = retrain_queue.run_retrain_dispatch(
        input_path=INPUTS,
        queue_path=tmp_path / "queue.jsonl",
        evidence_path=tmp_path / "evidence.json",
        summary_path=tmp_path / "summary.json",
        recipes_dir=tmp_path / "recipes",
    )
    assert summary.queue_count == 0
    assert not (tmp_path / "recipes").exists()


def test_workflow_is_scheduled_and_recipe_only() -> None:
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(workflow_text, Loader=yaml.BaseLoader)

    assert set(workflow["on"]) == {"schedule", "workflow_dispatch"}
    assert workflow["permissions"] == {
        "contents": "write",
        "pull-requests": "write",
    }
    steps = workflow["jobs"]["propose-recipes"]["steps"]
    action_refs = {
        step["uses"] for step in steps if isinstance(step, dict) and "uses" in step
    }
    assert action_refs == {
        "actions/checkout@v7",
        "actions/setup-python@v7",
        "actions/upload-artifact@v7",
        "astral-sh/setup-uv@v8.3.2",
        "peter-evans/create-pull-request@v8",
    }
    create_pr = next(
        step
        for step in steps
        if step.get("uses") == "peter-evans/create-pull-request@v8"
    )
    assert create_pr["if"] == (
        "steps.dispatch.outputs.queue_count != '0' && "
        "steps.dispatch.outputs.changed_recipe_count != '0'"
    )
    assert create_pr["with"]["add-paths"].strip() == "recipes/*.yaml"
    assert "downstream model\nrelease gates" in create_pr["with"]["body"]


def test_workflow_cannot_train_convert_or_publish_models() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    forbidden = (
        "HF_WRITE_TOKEN",
        "secrets.",
        "openmed.core.hf_publish",
        "openmed.mlx.convert",
        "openmed.coreml.convert",
        "openmed.onnx.convert",
        "scripts/release/dispatch_batch.py",
        "scripts/release/orchestrate.py",
        "recipes/queue.yaml",
    )

    assert [marker for marker in forbidden if marker in workflow] == []
    assert "scripts/release/retrain_queue.py" in workflow
    assert "Every resulting candidate must still pass" in workflow
