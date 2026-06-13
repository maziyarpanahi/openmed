"""Coherence checks for manifest-derived catalog surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.manifest import regenerate_surfaces


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "coherence.yml"


def test_regenerated_surfaces_are_current():
    assert regenerate_surfaces.regenerate_surfaces(check=True) == []


def test_canonical_labels_must_exist_in_core_taxonomy():
    rows = regenerate_surfaces.load_manifest()
    invalid_row = dict(rows[0])
    invalid_row["canonical_labels"] = ["NOT_A_CANONICAL_LABEL"]

    with pytest.raises(ValueError, match="outside CANONICAL_LABELS"):
        regenerate_surfaces.validate_canonical_labels([invalid_row])


def test_workflow_uses_committed_manifest_only():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "scripts/manifest/regenerate_surfaces.py" in text
    assert "git diff --exit-code" in text
    assert "scripts/manifest/generate_manifest.py" not in text
    assert "pip install" not in text
    assert "schedule:" not in text
    assert "cron:" not in text


def test_manifest_labels_are_within_core_taxonomy():
    rows = regenerate_surfaces.load_manifest()
    regenerate_surfaces.validate_canonical_labels(rows)


def test_registry_derivation_covers_manifest_rows():
    rows = regenerate_surfaces.load_manifest()
    regenerate_surfaces.validate_registry_derivation(rows)


def test_pii_defaults_are_derived_from_manifest_rows():
    rows = regenerate_surfaces.load_manifest()
    manifest_ids = {row["repo_id"] for row in rows}
    defaults = regenerate_surfaces.default_pii_models(rows)

    assert set(defaults) == set(regenerate_surfaces.supported_pii_languages(rows))
    assert set(defaults.values()) <= manifest_ids


def test_manifest_loader_rejects_invalid_json(tmp_path):
    bad_manifest = tmp_path / "models.jsonl"
    bad_manifest.write_text(json.dumps({"repo_id": "OpenMed/example"}) + "\n{", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        regenerate_surfaces.load_manifest(bad_manifest)
