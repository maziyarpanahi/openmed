"""Focused command tests for ``openmed registry``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.cli.main import main

_V1 = "OpenMed/synthetic-pii-v1"
_V2 = "OpenMed/synthetic-pii-v2"


def test_registry_list_emits_named_pointers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, state = _write_inputs(tmp_path)

    result = main(
        [
            "registry",
            "list",
            "--family",
            "PII",
            "--manifest",
            str(manifest),
            "--state",
            str(state),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["command"] == "registry list"
    assert payload["data"]["pointers"] == {
        "latest": _V2,
        "canary": _V2,
        "last_green": _V1,
    }


def test_registry_rollback_updates_only_local_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, state = _write_inputs(tmp_path)
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "decision": "RELEASABLE",
                "repo_id": _V1,
                "family": "PII",
                "tier": "Small",
                "format": "pytorch",
                "repro_hash": "sha256:synthetic-gate-v1",
            }
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "registry",
            "rollback",
            "PII",
            "--gate-report",
            str(gate),
            "--manifest",
            str(manifest),
            "--state",
            str(state),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    persisted = json.loads(state.read_text(encoding="utf-8"))

    assert result == 0
    assert payload["data"]["pointers"]["latest"] == _V1
    assert payload["data"]["pointers"]["canary"] is None
    assert persisted["families"]["PII"]["lineage"][-1]["relation"] == (
        "rolled-back-from"
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "models.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "family": "PII",
                    "task": "token-classification",
                    "languages": ["en"],
                    "tier": "Small",
                    "param_count": 44_000_000,
                    "formats": ["pytorch"],
                }
            )
            + "\n"
            for repo_id in (_V1, _V2)
        ),
        encoding="utf-8",
    )
    state = tmp_path / "registry_state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "families": {
                    "PII": {
                        "versions": {_V1: "1.0.0", _V2: "2.0.0"},
                        "pointers": {
                            "latest": _V2,
                            "canary": _V2,
                            "last_green": _V1,
                        },
                        "lineage": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, state
