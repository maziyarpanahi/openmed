"""Training provenance workflow and verifier tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from openmed.core.model_card import render_model_card
from openmed.core.repro_hash import build_training_provenance
from scripts.release import verify_provenance

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "repro-provenance.yml"


def _provenance() -> dict[str, object]:
    return build_training_provenance(
        rng_seeds={"python": 13, "numpy": 21, "torch": 34},
        data_manifest_hash="sha256:" + "a" * 64,
        recipe_config_hash="sha256:" + "b" * 64,
        env_lock_digest="sha256:" + "c" * 64,
        base_model="OpenMed/base-model",
        base_model_revision="7b4f2ca",
        git_sha="abc123",
        repo_id="OpenMed/test-model",
        checkpoint_id="checkpoint-001",
    )


def _manifest_row(provenance: dict[str, object]) -> dict[str, object]:
    return {
        "repo_id": "OpenMed/test-model",
        "family": "PII",
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Tiny",
        "param_count": 65_000_000,
        "architecture": "distilbert",
        "base_model": "OpenMed/base-model",
        "formats": ["pytorch"],
        "canonical_labels": ["PERSON"],
        "benchmark": {"dataset": None, "micro_f1": None, "recall": None},
        "arxiv": None,
        "license": "apache-2.0",
        "reproducibility_hash": provenance["reproducibility_hash"],
        "released": "2026-06-24",
        "training_provenance": {
            "base_model_revision": provenance["base_model_revision"],
            "data_manifest_hash": provenance["data_manifest_hash"],
            "env_lock_digest": provenance["env_lock_digest"],
            "git_sha": provenance["git_sha"],
            "path": "checkpoints/model/training_provenance.json",
            "recipe_config_hash": provenance["recipe_config_hash"],
            "reproducibility_hash": provenance["reproducibility_hash"],
            "rng_seeds": provenance["rng_seeds"],
        },
    }


def test_repro_provenance_workflow_runs_verifier_for_candidates() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    content = WORKFLOW.read_text(encoding="utf-8")

    assert workflow["on"]["workflow_dispatch"]["inputs"]["checkpoint-dir"]
    assert "training_provenance.json" in content
    assert "scripts/release/verify_provenance.py" in content
    assert "--checkpoint-dir" in content
    assert "--manifest" in content


def test_verify_provenance_cli_checks_hash_manifest_and_model_card(
    tmp_path: Path,
    capsys,
) -> None:
    provenance = _provenance()
    row = _manifest_row(provenance)
    provenance_path = tmp_path / "training_provenance.json"
    manifest_path = tmp_path / "models.jsonl"
    model_card_path = tmp_path / "README.md"
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    model_card_path.write_text(render_model_card(row), encoding="utf-8")

    exit_code = verify_provenance.main(
        [
            "--provenance",
            str(provenance_path),
            "--manifest",
            str(manifest_path),
            "--model-card",
            str(model_card_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Training provenance verified" in captured.out


def test_verify_provenance_cli_quarantines_missing_seeds(
    tmp_path: Path,
    capsys,
) -> None:
    provenance = _provenance()
    provenance.pop("rng_seeds")
    path = tmp_path / "training_provenance.json"
    path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")

    exit_code = verify_provenance.main(["--provenance", str(path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Training provenance quarantined" in captured.err
    assert "rng_seeds" in captured.err
