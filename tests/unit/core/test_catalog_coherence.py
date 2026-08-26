"""Tests for the manifest <-> label-taxonomy coherence check (OM-007)."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.core.catalog_coherence import manifest_label_errors
from openmed.core.labels import CANONICAL_LABELS, OTHER, is_recognized_label


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
