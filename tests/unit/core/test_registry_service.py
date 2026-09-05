"""Focused tests for the offline slot-keyed model-registry state service.

Fixtures mirror the committed manifest's real shapes: the PII pointer target is
the actual committed checkpoint (one ``-v1`` name, two formats), and the NER
rows are real untiered single-format entries — no invented two-version stems,
which occur zero times in ``models.jsonl``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openmed.core.baseline import write_baseline_store
from openmed.core.registry_service import (
    RegistryGateError,
    RegistryMigrationError,
    RegistryService,
    RegistryStateError,
    load_registry_state,
    migrate_registry_state,
    migrate_registry_state_file,
    registry_state_errors,
    semantic_version,
)

_PII_MLX = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx"
_PII_LARGE = "OpenMed/OpenMed-PII-SuperClinical-Large-434M"
_NER_A = "OpenMed/OpenMed-NER-AnatomyDetect-BigMed-278M"
_NER_B = "OpenMed/OpenMed-NER-AnatomyDetect-BigMed-560M"
_PII_SLOT = "pii::small::mlx-fp"
_NER_SLOT = "ner::none::pytorch"


def test_semantic_version_is_display_metadata_only() -> None:
    assert semantic_version("OpenMed/checkpoint-v3") == "3.0.0"
    assert semantic_version("OpenMed/checkpoint-v2.4-mlx") == "2.4.0"
    assert semantic_version(_NER_A) == "0.0.0"


def test_load_rejects_retired_v1_schema(tmp_path: Path) -> None:
    state = tmp_path / "registry_state.json"
    state.write_text(json.dumps(_v1_committed_payload()), encoding="utf-8")

    with pytest.raises(RegistryStateError, match="migrate"):
        load_registry_state(state)


def test_promote_creates_sparse_untiered_slot(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    state = tmp_path / "registry_state.json"
    service = RegistryService(manifest_path=manifest, state_path=state)

    service.promote(_NER_A, gate_report=_ner_gate(_NER_A))

    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert sorted(persisted["slots"]) == [_NER_SLOT]
    entry = persisted["slots"][_NER_SLOT]
    # Sparse channel: only the promoted checkpoint exists, no pre-population
    # of the other untiered NER manifest rows sharing the coarse slot.
    assert entry["checkpoints"] == {_NER_A: "1.0.0"}
    assert entry["pointers"] == {
        "latest": _NER_A,
        "canary": None,
        "last_green": _NER_A,
    }


def test_promotion_assigns_and_preserves_semver(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    state = tmp_path / "registry_state.json"
    service = RegistryService(manifest_path=manifest, state_path=state)

    service.promote(_NER_A, gate_report=_ner_gate(_NER_A))
    service.promote(_NER_B, gate_report=_ner_gate(_NER_B))
    service.promote(_NER_A, gate_report=_ner_gate(_NER_A))

    checkpoints = service.state["slots"][_NER_SLOT]["checkpoints"]
    assert checkpoints == {_NER_A: "1.0.0", _NER_B: "1.1.0"}

    restored = RegistryService(manifest_path=manifest, state_path=state)
    assert restored.state["slots"][_NER_SLOT]["checkpoints"] == checkpoints
    assert restored.pointers(_NER_SLOT)["latest"] == _NER_A


def test_promote_and_rollback_are_reproducible_from_local_state(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    state = tmp_path / "registry_state.json"
    service = RegistryService(
        manifest_path=manifest,
        state_path=state,
        clock=lambda: datetime(2026, 8, 10, tzinfo=timezone.utc),
    )

    service.promote(_NER_A, gate_report=_ner_gate(_NER_A))
    service.promote(_NER_B, gate_report=_ner_gate(_NER_B))
    assert service.pointers(_NER_SLOT) == {
        "latest": _NER_B,
        "canary": None,
        "last_green": _NER_A,
    }
    assert service.lineage(_NER_SLOT)[0] == {
        "relation": "supersedes",
        "from": _NER_A,
        "to": _NER_B,
        "reason": "promotion",
        "recorded_at": "2026-08-10T00:00:00+00:00",
        "gate_report_hash": f"sha256:gate-{_NER_B.rsplit('-', 1)[-1]}",
    }

    service.rollback(_NER_SLOT, gate_report=_ner_gate(_NER_A))
    persisted = json.loads(state.read_text(encoding="utf-8"))
    restored = RegistryService(manifest_path=manifest, state_path=state)

    assert restored.state == persisted
    assert restored.pointers("NER::none::PYTORCH")["latest"] == _NER_A
    assert restored.pointers(_NER_SLOT)["canary"] is None
    assert restored.lineage(_NER_SLOT)[-1]["relation"] == "rolled-back-from"
    assert restored.lineage(_NER_SLOT)[-1]["from"] == _NER_B
    assert restored.lineage(_NER_SLOT)[-1]["to"] == _NER_A


@pytest.mark.parametrize("decision", [None, "QUARANTINED"])
def test_pointer_flip_requires_releasable_gate_without_mutating_state(
    tmp_path: Path,
    decision: str | None,
) -> None:
    manifest = _write_manifest(tmp_path)
    state = _write_v2_state(tmp_path)
    before = state.read_bytes()
    service = RegistryService(manifest_path=manifest, state_path=state)
    report = None if decision is None else _ner_gate(_NER_B, decision=decision)

    with pytest.raises(RegistryGateError, match="RELEASABLE"):
        service.flip_pointer(_NER_SLOT, "canary", _NER_B, gate_report=report)

    assert state.read_bytes() == before
    assert service.pointers(_NER_SLOT)["canary"] is None


def test_gate_coordinates_must_match_manifest_and_slot(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    state = _write_v2_state(tmp_path)
    before = state.read_bytes()
    service = RegistryService(manifest_path=manifest, state_path=state)

    format_mismatch = _gate(_NER_A, family="NER", tier=None, format_name="mlx-fp")
    with pytest.raises(RegistryGateError, match="do not match the manifest"):
        service.promote(_NER_A, gate_report=format_mismatch)

    tier_mismatch = _gate(_NER_A, family="NER", tier="Small", format_name="pytorch")
    with pytest.raises(RegistryGateError, match="do not match the manifest"):
        service.promote(_NER_A, gate_report=tier_mismatch)

    # The PII checkpoint carries both formats; a pytorch-coordinate report is
    # manifest-valid but resolves to a different slot than the one being moved.
    other_slot = _gate(_PII_MLX, family="PII", tier="Small", format_name="pytorch")
    with pytest.raises(RegistryGateError, match="not .*mlx-fp"):
        service.flip_pointer(_PII_SLOT, "canary", _PII_MLX, gate_report=other_slot)

    assert state.read_bytes() == before


def test_flip_pointer_never_creates_a_slot(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    state = tmp_path / "registry_state.json"
    service = RegistryService(manifest_path=manifest, state_path=state)

    with pytest.raises(RegistryStateError, match="unknown registry slot"):
        service.flip_pointer(_NER_SLOT, "canary", _NER_A, gate_report=_ner_gate(_NER_A))


def test_stored_semver_is_validated_not_recomputed(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]

    # The committed checkpoint's name says "-v1"; an assigned 3.2.0 is still
    # coherent because registry versions are promotion state, not name parses.
    state = _v2_state_payload()
    state["slots"][_PII_SLOT]["checkpoints"][_PII_MLX] = "3.2.0"
    assert registry_state_errors(rows, state) == []

    state["slots"][_PII_SLOT]["checkpoints"][_PII_MLX] = "not-semver"
    errors = registry_state_errors(rows, state)
    assert any("MAJOR.MINOR.PATCH" in error for error in errors)


def test_registry_coherence_reports_dangling_and_cross_slot_targets(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]

    state = _v2_state_payload()
    entry = state["slots"][_PII_SLOT]
    entry["pointers"]["canary"] = "OpenMed/absent-from-manifest"
    errors = registry_state_errors(rows, state)
    assert any("missing manifest row" in error for error in errors)
    assert any("lacks a slot checkpoint entry" in error for error in errors)

    state = _v2_state_payload()
    entry = state["slots"][_PII_SLOT]
    entry["checkpoints"][_NER_A] = "1.1.0"
    errors = registry_state_errors(rows, state)
    assert any(
        "coordinates do not include slot" in error and _NER_A in error
        for error in errors
    )


def test_migration_maps_committed_v1_state_uniquely(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]

    migrated = migrate_registry_state(
        _v1_committed_payload(), rows, _baseline_entries()
    )

    assert migrated == {
        "schema_version": 2,
        "slots": {_PII_SLOT: _v2_state_payload()["slots"][_PII_SLOT]},
    }
    # The committed row declares ["mlx-fp", "pytorch"]; only the slot with
    # committed baseline evidence exists after migration.
    assert "pii::small::pytorch" not in migrated["slots"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        ("drop-baseline", "no committed baseline entry"),
        ("ambiguous-baseline", "multiple baseline slots"),
        ("different-slots", "different slots"),
        ("stray-version", "without pointer-target coordinate evidence"),
    ],
)
def test_migration_failures_leave_the_file_unchanged(
    tmp_path: Path,
    mutate: str,
    match: str,
) -> None:
    manifest = _write_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    v1 = _v1_committed_payload()
    baseline = _baseline_entries()

    if mutate == "drop-baseline":
        baseline.pop(_PII_SLOT)
    elif mutate == "ambiguous-baseline":
        baseline["pii::small::pytorch"] = {"repo_id": _PII_MLX}
    elif mutate == "different-slots":
        baseline["pii::large::pytorch"] = {"repo_id": _PII_LARGE}
        v1["families"]["PII"]["pointers"]["last_green"] = _PII_LARGE
        v1["families"]["PII"]["versions"][_PII_LARGE] = "1.0.0"
    elif mutate == "stray-version":
        v1["families"]["PII"]["versions"][_PII_LARGE] = "1.0.0"

    state = tmp_path / "registry_state.json"
    state.write_text(json.dumps(v1), encoding="utf-8")
    before = state.read_bytes()

    with pytest.raises(RegistryMigrationError, match=match):
        migrate_registry_state(v1, rows, baseline)

    assert state.read_bytes() == before


def test_migrate_file_writes_v2_and_the_service_loads_it(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    state = tmp_path / "registry_state.json"
    state.write_text(json.dumps(_v1_committed_payload()), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    write_baseline_store(_baseline_store(), baseline)

    migrated = migrate_registry_state_file(
        state_path=state,
        manifest_path=manifest,
        baseline_path=baseline,
    )

    assert migrated == json.loads(state.read_text(encoding="utf-8"))
    service = RegistryService(manifest_path=manifest, state_path=state)
    assert service.pointers(_PII_SLOT)["latest"] == _PII_MLX

    with pytest.raises(RegistryMigrationError, match="nothing to migrate"):
        migrate_registry_state_file(
            state_path=state,
            manifest_path=manifest,
            baseline_path=baseline,
        )


def test_post_migration_promotion_still_requires_matched_gate(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    state = tmp_path / "registry_state.json"
    state.write_text(json.dumps(_v1_committed_payload()), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    write_baseline_store(_baseline_store(), baseline)
    migrate_registry_state_file(
        state_path=state, manifest_path=manifest, baseline_path=baseline
    )
    service = RegistryService(manifest_path=manifest, state_path=state)

    with pytest.raises(RegistryGateError, match="RELEASABLE"):
        service.flip_pointer(_PII_SLOT, "canary", _PII_MLX, gate_report=None)

    mismatched = _gate(_PII_MLX, family="PII", tier="Large", format_name="mlx-fp")
    with pytest.raises(RegistryGateError, match="do not match the manifest"):
        service.rollback(_PII_SLOT, gate_report=mismatched)

    # The migrated checkpoint keeps its committed 1.0.0, so the next distinct
    # promotion into the slot assigns 1.1.0.
    second = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v2-mlx"
    _append_manifest_row(manifest, _pii_row(second))
    service = RegistryService(manifest_path=manifest, state_path=state)
    service.promote(
        second,
        gate_report=_gate(second, family="PII", tier="Small", format_name="mlx-fp"),
    )
    checkpoints = service.state["slots"][_PII_SLOT]["checkpoints"]
    assert checkpoints == {_PII_MLX: "1.0.0", second: "1.1.0"}


def _pii_row(repo_id: str = _PII_MLX) -> dict[str, object]:
    return {
        "repo_id": repo_id,
        "family": "PII",
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Small",
        "param_count": 44_000_000,
        "formats": ["mlx-fp", "pytorch"],
    }


def _pii_large_row() -> dict[str, object]:
    return {
        "repo_id": _PII_LARGE,
        "family": "PII",
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Large",
        "param_count": 434_000_000,
        "formats": ["pytorch"],
    }


def _ner_row(repo_id: str) -> dict[str, object]:
    return {
        "repo_id": repo_id,
        "family": "NER",
        "task": "token-classification",
        "languages": ["en"],
        "tier": None,
        "param_count": 278_000_000,
        "formats": ["pytorch"],
    }


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "models.jsonl"
    rows = [_pii_row(), _pii_large_row(), _ner_row(_NER_A), _ner_row(_NER_B)]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _append_manifest_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _v1_committed_payload() -> dict[str, object]:
    """Mirror the exact registry_state.json committed by schema v1."""

    return {
        "schema_version": 1,
        "families": {
            "PII": {
                "lineage": [],
                "pointers": {
                    "canary": None,
                    "last_green": _PII_MLX,
                    "latest": _PII_MLX,
                },
                "versions": {_PII_MLX: "1.0.0"},
            }
        },
    }


def _v2_state_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "slots": {
            _PII_SLOT: {
                "checkpoints": {_PII_MLX: "1.0.0"},
                "pointers": {
                    "latest": _PII_MLX,
                    "canary": None,
                    "last_green": _PII_MLX,
                },
                "lineage": [],
            },
            _NER_SLOT: {
                "checkpoints": {_NER_A: "1.0.0"},
                "pointers": {
                    "latest": _NER_A,
                    "canary": None,
                    "last_green": _NER_A,
                },
                "lineage": [],
            },
        },
    }


def _write_v2_state(tmp_path: Path) -> Path:
    path = tmp_path / "registry_state.json"
    path.write_text(json.dumps(_v2_state_payload()), encoding="utf-8")
    return path


def _baseline_entries() -> dict[str, dict[str, object]]:
    """Migration coordinate evidence shaped like gates/baseline.json entries."""

    return {
        _PII_SLOT: {"repo_id": _PII_MLX},
        # Gate-only benchmark baseline: no repo_id, never migration evidence.
        "i18n-throughput::hi::pattern-only": {"family": "i18n-throughput"},
    }


def _baseline_store() -> dict[str, object]:
    """A structurally valid baseline store for the file-level migration."""

    return {
        "schema_version": 1,
        "entries": {
            _PII_SLOT: {
                "key": _PII_SLOT,
                "family": "PII",
                "tier": "Small",
                "format": "mlx-fp",
                "metrics": {"micro_f1": None, "recall": None, "dataset": None},
                "reproducibility_hash": "sha256:" + "ab" * 32,
                "repo_id": _PII_MLX,
            }
        },
    }


def _gate(
    repo_id: str,
    *,
    family: str,
    tier: str | None,
    format_name: str,
    decision: str = "RELEASABLE",
) -> dict[str, object]:
    return {
        "decision": decision,
        "repo_id": repo_id,
        "family": family,
        "tier": tier,
        "format": format_name,
        "repro_hash": f"sha256:gate-{repo_id.rsplit('-', 1)[-1]}",
    }


def _ner_gate(repo_id: str, *, decision: str = "RELEASABLE") -> dict[str, object]:
    return _gate(
        repo_id,
        family="NER",
        tier=None,
        format_name="pytorch",
        decision=decision,
    )
