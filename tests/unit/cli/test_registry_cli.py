"""Focused command tests for ``openmed registry``.

Fixtures mirror real manifest shapes: distinct untiered checkpoints sharing one
coarse slot, not invented ``-v1``/``-v2`` stems (which occur zero times in
``models.jsonl``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.cli.main import main

_NER_A = "OpenMed/OpenMed-NER-AnatomyDetect-BigMed-278M"
_NER_B = "OpenMed/OpenMed-NER-AnatomyDetect-BigMed-560M"
_NER_SLOT = "ner::none::pytorch"


def test_registry_list_emits_named_pointers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, state = _write_inputs(tmp_path)

    result = main(
        [
            "registry",
            "list",
            "--slot",
            _NER_SLOT,
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
        "latest": _NER_B,
        "canary": _NER_B,
        "last_green": _NER_A,
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
                "repo_id": _NER_A,
                "family": "NER",
                "tier": None,
                "format": "pytorch",
                "repro_hash": "sha256:synthetic-gate-278m",
            }
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "registry",
            "rollback",
            _NER_SLOT,
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
    assert payload["data"]["pointers"]["latest"] == _NER_A
    assert payload["data"]["pointers"]["canary"] is None
    assert persisted["slots"][_NER_SLOT]["lineage"][-1]["relation"] == (
        "rolled-back-from"
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "models.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "family": "NER",
                    "task": "token-classification",
                    "languages": ["en"],
                    "tier": None,
                    "param_count": 278_000_000,
                    "formats": ["pytorch"],
                }
            )
            + "\n"
            for repo_id in (_NER_A, _NER_B)
        ),
        encoding="utf-8",
    )
    state = tmp_path / "registry_state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "slots": {
                    _NER_SLOT: {
                        "checkpoints": {_NER_A: "1.0.0", _NER_B: "1.1.0"},
                        "pointers": {
                            "latest": _NER_B,
                            "canary": _NER_B,
                            "last_green": _NER_A,
                        },
                        "lineage": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, state
