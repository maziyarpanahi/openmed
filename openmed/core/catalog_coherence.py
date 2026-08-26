"""Coherence check binding the committed manifest to the label taxonomy.

Complements the manifest-derived surfaces already gated elsewhere —
``manifest_diff``'s README counts and registry model cards, the ``pii_i18n``
import-time language guards, and ``stage_pages`` leaderboard parity — by
validating the ``canonical_labels`` column of ``models.jsonl`` against
:data:`openmed.core.labels.CANONICAL_LABELS`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openmed.core.labels import is_recognized_label

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "models.jsonl"


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_number}: {exc}") from exc
    return rows


def manifest_label_errors(*, manifest_path: str | Path = MANIFEST_PATH) -> list[str]:
    """Return drift errors for ``canonical_labels`` values outside the taxonomy.

    A value is accepted when it resolves to a real taxonomy member — already
    canonical, or a known alias such as ``CHEM``/``SIMPLE_CHEMICAL`` ->
    ``CHEMICAL`` — and rejected when it would only survive via
    :func:`~openmed.core.labels.normalize_label`'s ``OTHER`` fallthrough. Mirrors
    the ``list[str]`` contract of ``manifest_diff.registry_surface_errors`` so a
    single gate can consume both. An empty list means the column is coherent.
    """

    try:
        rows = _load_manifest_rows(Path(manifest_path))
    except (OSError, ValueError) as exc:
        return [str(exc)]

    offenders: dict[str, set[str]] = {}
    for line_number, row in enumerate(rows, start=1):
        repo_id = str(row.get("repo_id") or f"<row {line_number}>")
        values = row.get("canonical_labels")
        if values is None:
            continue
        if not isinstance(values, list):
            offenders.setdefault("<canonical_labels not a list>", set()).add(repo_id)
            continue
        for value in values:
            if not isinstance(value, str) or not is_recognized_label(value):
                offenders.setdefault(repr(value), set()).add(repo_id)

    errors: list[str] = []
    for label, repos in sorted(offenders.items()):
        sample = ", ".join(sorted(repos)[:3])
        errors.append(
            f"canonical_labels value {label} is not in CANONICAL_LABELS "
            f"({len(repos)} row(s), e.g. {sample})"
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Command-line gate: exit non-zero when any ``canonical_labels`` value drifts."""

    parser = argparse.ArgumentParser(
        description="Check manifest canonical_labels against the label taxonomy.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to models.jsonl (defaults to the committed manifest).",
    )
    args = parser.parse_args(argv)

    errors = manifest_label_errors(manifest_path=args.manifest)
    if errors:
        print("Catalog coherence check failed:")
        for line in errors:
            print(f"  - {line}")
        return 1
    print("Catalog canonical_labels are coherent with CANONICAL_LABELS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
