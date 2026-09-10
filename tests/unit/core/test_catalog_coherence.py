"""Tests for the manifest <-> label-taxonomy coherence check (OM-007)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from openmed.core.catalog_coherence import manifest_label_errors
from openmed.core.labels import CANONICAL_LABELS, OTHER, is_recognized_label
from openmed.core.model_registry import load_manifest_rows
from scripts.manifest import regenerate_surfaces

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "coherence.yml"


def _write(path: Path, *rows: dict) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_committed_manifest_has_no_label_drift() -> None:
    # AC1: the check is green on the committed manifest at HEAD.
    assert manifest_label_errors() == []


def test_alias_forms_resolve() -> None:
    # Option A: CHEM / SIMPLE_CHEMICAL are aliases of CHEMICAL, not drift.
    assert is_recognized_label("CHEM")
    assert is_recognized_label("SIMPLE_CHEMICAL")


def test_literal_other_is_accepted() -> None:
    assert OTHER in CANONICAL_LABELS
    assert is_recognized_label("OTHER")


def test_unknown_label_is_flagged_despite_other_fallback(tmp_path: Path) -> None:
    # AC4 + the trap: normalize_label maps this to OTHER (a canonical member),
    # so a bare membership test would pass it. The check must still flag it.
    manifest = _write(
        tmp_path / "models.jsonl",
        {"repo_id": "acme/bogus", "canonical_labels": ["NOT_A_REAL_LABEL"]},
    )
    errors = manifest_label_errors(manifest_path=manifest)
    assert errors and "NOT_A_REAL_LABEL" in errors[0]


def test_clean_manifest_passes(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "models.jsonl",
        {"repo_id": "acme/ok", "canonical_labels": ["DISEASE", "CHEM", "OTHER"]},
    )
    assert manifest_label_errors(manifest_path=manifest) == []


def test_non_list_canonical_labels_is_flagged(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "models.jsonl",
        {"repo_id": "acme/bad", "canonical_labels": "DISEASE"},
    )
    assert manifest_label_errors(manifest_path=manifest)


def test_committed_catalog_surfaces_are_current() -> None:
    assert regenerate_surfaces.surface_errors() == []


def test_runtime_registry_and_pii_defaults_derive_from_manifest() -> None:
    rows = load_manifest_rows()

    regenerate_surfaces.validate_registry_derivation(
        rows, ROOT / "gates" / "registry_state.json"
    )
    regenerate_surfaces.validate_pii_derivation(rows)


@pytest.mark.parametrize(
    ("relative_path", "old", "new"),
    (
        (Path("README.md"), "2,266 manifest entries", "9,999 manifest entries"),
        (
            Path("docs/model-registry.md"),
            "The committed manifest contains 2,266 entries",
            "The committed manifest contains 9,999 entries",
        ),
        (
            Path("docs/model-cards/registry/pii-small-mlx-fp-latest.md"),
            "<!-- Registry pointer:",
            "<!-- Stale registry pointer:",
        ),
    ),
)
def test_manual_surface_mutation_is_detected(
    tmp_path: Path, relative_path: Path, old: str, new: str
) -> None:
    _copy_catalog_inputs(tmp_path)
    target = tmp_path / relative_path
    current = target.read_text(encoding="utf-8")
    assert old in current
    target.write_text(current.replace(old, new, 1), encoding="utf-8")

    assert regenerate_surfaces.surface_errors(tmp_path)
    regenerate_surfaces.regenerate_surfaces(tmp_path)
    assert regenerate_surfaces.surface_errors(tmp_path) == []


def test_coherence_workflow_is_offline_and_diff_guarded() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    regenerator = (ROOT / "scripts/manifest/regenerate_surfaces.py").read_text(
        encoding="utf-8"
    )

    assert "scripts/manifest/regenerate_surfaces.py" in workflow
    assert "run: git diff --exit-code" in workflow
    assert "docs/i18n/readme_section_hashes.json" in workflow
    assert "generate_manifest.py" not in workflow
    assert "huggingface" not in workflow.casefold()
    assert "schedule:" not in workflow
    assert "cron:" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "--refresh-github-stars" not in regenerator
    assert "check_readme_drift.py" in regenerator


def _copy_catalog_inputs(destination: Path) -> None:
    (destination / "gates").mkdir()
    (destination / "docs" / "model-cards").mkdir(parents=True)
    shutil.copy2(ROOT / "models.jsonl", destination / "models.jsonl")
    shutil.copy2(
        ROOT / "gates" / "registry_state.json",
        destination / "gates" / "registry_state.json",
    )
    shutil.copy2(ROOT / "README.md", destination / "README.md")
    shutil.copy2(
        ROOT / "docs" / "model-registry.md",
        destination / "docs" / "model-registry.md",
    )
    shutil.copytree(
        ROOT / "docs" / "model-cards" / "registry",
        destination / "docs" / "model-cards" / "registry",
    )
