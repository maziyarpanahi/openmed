"""Registry-state extensions to the release manifest coherence gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.eval.release_gates import _manifest_coherence_check

_REPO_ID = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx"
_MISSING_ID = "OpenMed/OpenMed-PII-SuperClinical-Large-434M"
_SLOT = "pii::small::mlx-fp"


@pytest.mark.parametrize("dangling", ["pointer", "lineage"])
def test_manifest_coherence_fails_for_dangling_registry_targets(
    tmp_path: Path,
    dangling: str,
) -> None:
    manifest = _write_manifest(tmp_path)
    state = _state_payload()
    if dangling == "pointer":
        state["slots"][_SLOT]["pointers"]["canary"] = _MISSING_ID
    else:
        state["slots"][_SLOT]["lineage"] = [_edge(_REPO_ID, _MISSING_ID)]
    state_path = tmp_path / "registry_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    check = _manifest_coherence_check(
        _identity(),
        {
            "manifest_path": str(manifest),
            "registry_state_path": str(state_path),
        },
    )

    assert check.passed is False
    errors = check.details["mismatches"]["registry_state"]["errors"]
    assert any("references missing manifest row" in error for error in errors)


def test_manifest_coherence_accepts_resolved_registry_state(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    state_path = tmp_path / "registry_state.json"
    state_path.write_text(json.dumps(_state_payload()), encoding="utf-8")

    check = _manifest_coherence_check(
        _identity(),
        {
            "manifest_path": str(manifest),
            "registry_state_path": str(state_path),
        },
    )

    assert check.passed is True
    assert check.details["registry_state_path"] == str(state_path)


def _identity() -> dict[str, object]:
    return {
        "repo_id": _REPO_ID,
        "family": "PII",
        "tier": "Small",
        "param_count": 44_000_000,
        "format": "mlx-fp",
        "eval_set_hash": "sha256:synthetic-eval",
        "leakage_fixture_hash": "sha256:synthetic-leakage",
    }


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "models.jsonl"
    path.write_text(
        json.dumps(
            {
                "repo_id": _REPO_ID,
                "family": "PII",
                "task": "token-classification",
                "languages": ["en"],
                "tier": "Small",
                "param_count": 44_000_000,
                "formats": ["mlx-fp", "pytorch"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _state_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "slots": {
            _SLOT: {
                "checkpoints": {_REPO_ID: "1.0.0"},
                "pointers": {
                    "latest": _REPO_ID,
                    "canary": None,
                    "last_green": _REPO_ID,
                },
                "lineage": [],
            }
        },
    }


def _edge(from_repo: str, to_repo: str) -> dict[str, str]:
    return {
        "relation": "supersedes",
        "from": from_repo,
        "to": to_repo,
        "reason": "synthetic-test",
        "recorded_at": "2026-08-03T00:00:00+00:00",
        "gate_report_hash": "sha256:synthetic",
    }
