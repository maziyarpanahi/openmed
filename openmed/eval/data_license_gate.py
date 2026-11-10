"""Fail-closed gate for declared training-data licenses in the model manifest.

Reads ``training_data_licenses`` entries from the committed ``models.jsonl``
manifest only. This gate is read-only: it never re-fetches data or calls the
Hugging Face API, and it never mutates the manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from openmed.core.model_registry import MANIFEST_PATH, load_manifest_rows
from openmed.eval.release_gates import GateCheck

DATA_LICENSE_GATE = "data_license"


def evaluate_data_license_gate(
    manifest_path: str | Path = MANIFEST_PATH,
) -> GateCheck:
    """Fail closed when a published checkpoint trains on non-redistributable data.

    A manifest row counts as published once it carries a non-empty ``released``
    date. For each published row, a ``training_data_licenses`` entry with
    ``redistributable: false`` (a DUA/UMLS/SNOMED/CPT-style source, for
    example) must declare ``role: eval``; the same source declared with
    ``role: train`` fails the gate.
    """
    path = Path(manifest_path)
    rows = load_manifest_rows(path)

    violations: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("released"):
            continue
        repo_id = str(row.get("repo_id") or "<unknown>")
        violations.extend(_row_violations(row, repo_id))

    if violations:
        return GateCheck(
            DATA_LICENSE_GATE,
            False,
            reason=(
                "published checkpoint declares a non-redistributable training "
                "source with role=train"
            ),
            details={"violations": violations},
        )
    return GateCheck(
        DATA_LICENSE_GATE,
        True,
        details={"manifest_path": str(path), "rows_checked": len(rows)},
    )


def data_license_gate_errors(manifest_path: str | Path = MANIFEST_PATH) -> list[str]:
    """Return catalog-check-style error strings for the registry coherence CLI."""
    check = evaluate_data_license_gate(manifest_path)
    if check.passed:
        return []
    return [
        f"{item['repo_id']}: non-redistributable training source "
        f"{item['name']!r} ({item['license']}) declares role=train"
        for item in check.details["violations"]
    ]


def _row_violations(row: Mapping[str, Any], repo_id: str) -> list[dict[str, str]]:
    entries = row.get("training_data_licenses")
    if not isinstance(entries, list):
        return []

    found: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("role") == "train" and entry.get("redistributable") is False:
            found.append(
                {
                    "repo_id": repo_id,
                    "name": str(entry.get("name") or "<unnamed>"),
                    "license": str(entry.get("license") or "<unknown>"),
                }
            )
    return found


__all__ = [
    "DATA_LICENSE_GATE",
    "data_license_gate_errors",
    "evaluate_data_license_gate",
]
