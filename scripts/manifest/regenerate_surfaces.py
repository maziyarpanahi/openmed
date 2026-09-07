#!/usr/bin/env python3
"""Regenerate repository surfaces from the committed model manifest.

This command is deliberately offline. Refreshing ``models.jsonl`` from a
remote registry belongs to ``generate_manifest.py`` and is never performed
here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from openmed.core.catalog_coherence import manifest_label_errors
from openmed.core.language_pack_catalog import DEFAULT_MODEL_PLACEHOLDER_LANGUAGES
from openmed.core.manifest_diff import (
    regenerate_registry_surfaces,
    registry_surface_errors,
)
from openmed.core.model_registry import build_registry, load_manifest_rows
from openmed.core.pii_i18n import DEFAULT_PII_MODELS, SUPPORTED_LANGUAGES
from openmed.core.registry_service import load_registry_state

ROOT = Path(__file__).resolve().parents[2]
_BRAND_CLAIMS_SCRIPT = ROOT / "scripts" / "brand" / "update_claims.py"
_BRAND_README_SCRIPT = ROOT / "scripts" / "brand" / "update_readme_brand.py"
_README_HASH_SCRIPT = ROOT / "scripts" / "i18n" / "check_readme_drift.py"


def _paths(root: Path) -> dict[str, Path]:
    return {
        "manifest": root / "models.jsonl",
        "state": root / "gates" / "registry_state.json",
        "readme": root / "README.md",
        "cards": root / "docs" / "model-cards" / "registry",
        "catalog_doc": root / "docs" / "model-registry.md",
    }


def _manifest_pii_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        repo_id = str(row.get("repo_id") or "").casefold()
        family = str(row.get("family") or "").casefold()
        if family == "pii" or "pii" in repo_id or "privacy" in repo_id:
            result.append(row)
    return result


def validate_canonical_labels(manifest_path: Path) -> None:
    """Raise when a manifest label is outside the core label taxonomy."""

    errors = manifest_label_errors(manifest_path=manifest_path)
    if errors:
        raise ValueError("; ".join(errors))


def validate_registry_derivation(
    rows: Sequence[dict[str, Any]], state_path: Path
) -> None:
    """Require every manifest row to appear in the generated runtime registry."""

    registry = build_registry(rows, load_registry_state(state_path))
    manifest_ids = {
        str(row["repo_id"])
        for row in rows
        if isinstance(row.get("repo_id"), str) and row["repo_id"]
    }
    registry_ids = {model.model_id for model in registry.values()}
    missing = sorted(manifest_ids - registry_ids)
    if missing:
        raise ValueError(
            "manifest rows missing from OPENMED_MODELS derivation: "
            + ", ".join(missing[:20])
        )


def validate_pii_derivation(rows: Sequence[dict[str, Any]]) -> None:
    """Require public PII languages and defaults to agree with manifest rows."""

    pii_rows = _manifest_pii_rows(rows)
    manifest_languages = {
        str(language)
        for row in pii_rows
        for language in row.get("languages") or []
        if str(language)
    }
    expected_supported = manifest_languages | set(DEFAULT_MODEL_PLACEHOLDER_LANGUAGES)
    if set(SUPPORTED_LANGUAGES) != expected_supported:
        raise ValueError(
            "pii_i18n.SUPPORTED_LANGUAGES differs from manifest-backed routes"
        )

    rows_by_id = {
        str(row["repo_id"]): row
        for row in pii_rows
        if isinstance(row.get("repo_id"), str) and row["repo_id"]
    }
    errors: list[str] = []
    for language in sorted(manifest_languages):
        model_id = DEFAULT_PII_MODELS.get(language)
        row = rows_by_id.get(str(model_id))
        if row is None:
            errors.append(f"{language}: default model is absent from models.jsonl")
            continue
        if language not in (row.get("languages") or []):
            errors.append(f"{language}: default model does not claim the language")
    if errors:
        raise ValueError("invalid pii_i18n.DEFAULT_PII_MODELS: " + "; ".join(errors))


def validate_inputs(root: Path = ROOT) -> list[dict[str, Any]]:
    """Load and validate all local inputs used by surface regeneration."""

    paths = _paths(root)
    rows = load_manifest_rows(paths["manifest"])
    validate_canonical_labels(paths["manifest"])
    validate_registry_derivation(rows, paths["state"])
    validate_pii_derivation(rows)
    return rows


def surface_errors(root: Path = ROOT) -> list[str]:
    """Return coherence errors without modifying any repository file."""

    paths = _paths(root)
    try:
        validate_inputs(root)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    errors = registry_surface_errors(
        manifest_path=paths["manifest"],
        state_path=paths["state"],
        readme_path=paths["readme"],
        card_dir=paths["cards"],
        catalog_doc_path=paths["catalog_doc"],
    )
    if root.resolve() == ROOT:
        errors.extend(_governed_surface_errors())
    return errors


def regenerate_surfaces(root: Path = ROOT) -> dict[str, int]:
    """Regenerate all committed catalog surfaces from local inputs."""

    paths = _paths(root)
    rows = validate_inputs(root)
    if root.resolve() == ROOT:
        _run_governed_tool(_BRAND_CLAIMS_SCRIPT, "--write")
    snapshot = regenerate_registry_surfaces(
        manifest_path=paths["manifest"],
        state_path=paths["state"],
        readme_path=paths["readme"],
        card_dir=paths["cards"],
        catalog_doc_path=paths["catalog_doc"],
    )
    if root.resolve() == ROOT:
        _run_governed_tool(_BRAND_README_SCRIPT, "--write")
        _run_governed_tool(_README_HASH_SCRIPT, "--update")
    return {
        "manifest_entries": len(rows),
        "registry_keys": len(snapshot.registry_keys),
        "supported_languages": len(snapshot.supported_languages),
        "registry_cards": len(snapshot.cards),
    }


def _governed_surface_errors() -> list[str]:
    errors: list[str] = []
    checks = (
        (_BRAND_CLAIMS_SCRIPT, ("--check",)),
        (_BRAND_README_SCRIPT, ("--check",)),
        (_README_HASH_SCRIPT, ()),
    )
    for script, args in checks:
        completed = _run_governed_tool(script, *args, check=False)
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            errors.append(detail or f"{script.name} reported generated-surface drift")
    return errors


def _run_governed_tool(
    script: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(detail or f"{script.name} failed")
    return completed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root containing models.jsonl.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check for drift without writing generated surfaces.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline regeneration or read-only drift check."""

    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    if args.check:
        errors = surface_errors(root)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print("Catalog surfaces are coherent with committed models.jsonl.")
        return 0

    try:
        summary = regenerate_surfaces(root)
    except (OSError, ValueError) as exc:
        print(f"Catalog regeneration failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
