"""Focused tests for the offline model-registry state service."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openmed.core.registry_service import (
    RegistryGateError,
    RegistryService,
    registry_state_errors,
    semantic_version,
)


def test_semantic_version_maps_openmed_version_suffixes() -> None:
    assert semantic_version("OpenMed/checkpoint-v3") == "3.0.0"
    assert semantic_version("OpenMed/checkpoint-v2.4-mlx") == "2.4.0"
    assert semantic_version("OpenMed/checkpoint-v1.2.7-onnx") == "1.2.7"
    assert semantic_version("OpenMed/legacy-checkpoint") == "0.0.0"


def test_promote_and_rollback_are_reproducible_from_local_state(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    state = _write_state(tmp_path)
    service = RegistryService(
        manifest_path=manifest,
        state_path=state,
        clock=lambda: datetime(2026, 8, 3, tzinfo=timezone.utc),
    )

    service.promote(_V2, gate_report=_gate(_V2))
    assert service.pointers("PII") == {
        "latest": _V2,
        "canary": None,
        "last_green": _V1,
    }
    assert service.lineage("PII")[0] == {
        "relation": "supersedes",
        "from": _V1,
        "to": _V2,
        "reason": "promotion",
        "recorded_at": "2026-08-03T00:00:00+00:00",
        "gate_report_hash": "sha256:gate-v2",
    }

    service.rollback("PII", gate_report=_gate(_V1))
    persisted = json.loads(state.read_text(encoding="utf-8"))
    restored = RegistryService(manifest_path=manifest, state_path=state)

    assert restored.state == persisted
    assert restored.pointers("pii")["latest"] == _V1
    assert restored.pointers("PII")["canary"] is None
    assert restored.lineage("PII")[-1]["relation"] == "rolled-back-from"
    assert restored.lineage("PII")[-1]["from"] == _V2
    assert restored.lineage("PII")[-1]["to"] == _V1


@pytest.mark.parametrize("decision", [None, "QUARANTINED"])
def test_pointer_flip_requires_releasable_gate_without_mutating_state(
    tmp_path: Path,
    decision: str | None,
) -> None:
    manifest = _write_manifest(tmp_path)
    state = _write_state(tmp_path)
    before = state.read_bytes()
    service = RegistryService(manifest_path=manifest, state_path=state)
    report = None if decision is None else _gate(_V2, decision=decision)

    with pytest.raises(RegistryGateError, match="RELEASABLE"):
        service.flip_pointer("PII", "canary", _V2, gate_report=report)

    assert state.read_bytes() == before
    assert service.pointers("PII")["canary"] is None


def test_registry_coherence_reports_dangling_pointer_and_lineage() -> None:
    rows = [_row(_V1)]
    state = _state_payload()
    state["families"]["PII"]["pointers"]["canary"] = _V2
    state["families"]["PII"]["lineage"] = [
        {
            "relation": "supersedes",
            "from": _V1,
            "to": _V2,
            "reason": "synthetic-test",
            "recorded_at": "2026-08-03T00:00:00+00:00",
            "gate_report_hash": "sha256:synthetic",
        }
    ]

    errors = registry_state_errors(rows, state)

    assert any(
        "pointers.canary references missing manifest row" in error for error in errors
    )
    assert any(
        "lineage[0].to references missing manifest row" in error for error in errors
    )


_V1 = "OpenMed/synthetic-clinical-small-v1"
_V2 = "OpenMed/synthetic-clinical-small-v2"


def _row(repo_id: str) -> dict[str, object]:
    return {
        "repo_id": repo_id,
        "family": "PII",
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Small",
        "param_count": 44_000_000,
        "formats": ["pytorch"],
    }


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "models.jsonl"
    path.write_text(
        "".join(json.dumps(_row(repo_id)) + "\n" for repo_id in (_V1, _V2)),
        encoding="utf-8",
    )
    return path


def _state_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "families": {
            "PII": {
                "versions": {_V1: "1.0.0"},
                "pointers": {
                    "latest": _V1,
                    "canary": None,
                    "last_green": _V1,
                },
                "lineage": [],
            }
        },
    }


def _write_state(tmp_path: Path) -> Path:
    path = tmp_path / "registry_state.json"
    path.write_text(json.dumps(_state_payload()), encoding="utf-8")
    return path


def _gate(repo_id: str, *, decision: str = "RELEASABLE") -> dict[str, object]:
    version = semantic_version(repo_id).split(".", 1)[0]
    return {
        "decision": decision,
        "repo_id": repo_id,
        "family": "PII",
        "tier": "Small",
        "format": "pytorch",
        "repro_hash": f"sha256:gate-v{version}",
    }
