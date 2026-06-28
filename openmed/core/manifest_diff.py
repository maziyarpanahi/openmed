"""Diff helpers for canonical OpenMed model manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .model_registry import load_manifest_rows

DIFF_FIELDS: tuple[str, ...] = (
    "tier",
    "param_count",
    "formats",
    "license",
    "benchmark",
)


@dataclass(frozen=True)
class ManifestFieldChange:
    """Before/after values for one changed manifest field."""

    before: Any
    after: Any

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the change."""
        return {"before": self.before, "after": self.after}


@dataclass(frozen=True)
class ManifestRepoChange:
    """Per-field changes for one repo present in both manifests."""

    repo_id: str
    changes: Mapping[str, ManifestFieldChange]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the repo change."""
        return {
            "repo_id": self.repo_id,
            "changes": {
                field: change.to_dict() for field, change in self.changes.items()
            },
        }


@dataclass(frozen=True)
class ManifestDiff:
    """Structured diff between two canonical model manifests."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[ManifestRepoChange, ...]

    @property
    def has_removed(self) -> bool:
        """Return whether any repo disappeared from the new manifest."""
        return bool(self.removed)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest diff."""
        return {
            "added": list(self.added),
            "removed": list(self.removed),
            "changed": [change.to_dict() for change in self.changed],
        }


def diff_manifests(old_path: str | Path, new_path: str | Path) -> ManifestDiff:
    """Return a structured diff between two local manifest JSONL files.

    Rows are keyed by ``repo_id``. The diff tracks the release-review fields
    ``tier``, ``param_count``, ``formats``, ``license``, and ``benchmark``.
    ``formats`` and benchmark structures are compared order-insensitively so
    equivalent reordering does not produce a changed repo.
    """

    old_manifest = Path(old_path)
    new_manifest = Path(new_path)
    for manifest in (old_manifest, new_manifest):
        if not manifest.is_file():
            raise FileNotFoundError(manifest)

    old_rows = _rows_by_repo(load_manifest_rows(old_manifest), old_manifest)
    new_rows = _rows_by_repo(load_manifest_rows(new_manifest), new_manifest)

    old_repo_ids = set(old_rows)
    new_repo_ids = set(new_rows)
    added = tuple(sorted(new_repo_ids - old_repo_ids))
    removed = tuple(sorted(old_repo_ids - new_repo_ids))

    changed: list[ManifestRepoChange] = []
    for repo_id in sorted(old_repo_ids & new_repo_ids):
        field_changes: dict[str, ManifestFieldChange] = {}
        old_row = old_rows[repo_id]
        new_row = new_rows[repo_id]
        for field in DIFF_FIELDS:
            old_value = old_row.get(field)
            new_value = new_row.get(field)
            if _normalized_field(field, old_value) == _normalized_field(
                field, new_value
            ):
                continue
            field_changes[field] = ManifestFieldChange(
                before=_display_field(field, old_value),
                after=_display_field(field, new_value),
            )

        if field_changes:
            changed.append(ManifestRepoChange(repo_id=repo_id, changes=field_changes))

    return ManifestDiff(added=added, removed=removed, changed=tuple(changed))


def _rows_by_repo(
    rows: list[dict[str, Any]], manifest_path: Path
) -> dict[str, dict[str, Any]]:
    by_repo: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        repo_id = row.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id:
            raise ValueError(
                f"Manifest row in {manifest_path} line {line_number} has no repo_id"
            )
        if repo_id in by_repo:
            raise ValueError(f"Duplicate repo_id in {manifest_path}: {repo_id}")
        by_repo[repo_id] = row
    return by_repo


def _normalized_field(field: str, value: Any) -> Any:
    if field == "formats":
        return _normalized_formats(value)
    if field == "benchmark":
        return _normalized_structured(value)
    return value


def _display_field(field: str, value: Any) -> Any:
    if field == "formats":
        return list(_normalized_formats(value))
    return value


def _normalized_formats(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        return (str(value),)
    return tuple(sorted({str(item) for item in value}))


def _normalized_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalized_structured(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        encoded_items = {
            json.dumps(
                _normalized_structured(item),
                sort_keys=True,
                separators=(",", ":"),
            )
            for item in value
        }
        return [json.loads(item) for item in sorted(encoded_items)]
    return value
