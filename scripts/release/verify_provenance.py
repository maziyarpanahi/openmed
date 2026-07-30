#!/usr/bin/env python3
"""Verify checkpoint training provenance against pinned inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from openmed.core.repro_hash import (
    TRAINING_PROVENANCE_FILENAME,
    ReproducibilityVerificationError,
    load_training_provenance,
    verify_reproducibility,
)

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """Run the training provenance verifier."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provenance",
        type=Path,
        help="Path to training_provenance.json.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(f"Checkpoint directory containing {TRAINING_PROVENANCE_FILENAME}."),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional models.jsonl path for manifest hash cross-checking.",
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        help="Optional rendered README.md model card for hash cross-checking.",
    )
    parser.add_argument(
        "--repo-id",
        help="Repo id to select from --manifest; defaults to provenance repo_id.",
    )
    args = parser.parse_args(argv)

    try:
        provenance_path = _resolve_provenance_path(args.provenance, args.checkpoint_dir)
        provenance = load_training_provenance(provenance_path)
        repo_id = args.repo_id or provenance.get("repo_id")
        manifest_row = (
            _load_manifest_row(args.manifest, repo_id) if args.manifest else None
        )
        model_card_text = (
            args.model_card.read_text(encoding="utf-8") if args.model_card else None
        )
        verified_hash = verify_reproducibility(
            provenance,
            manifest_row=manifest_row,
            model_card_text=model_card_text,
        )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        ReproducibilityVerificationError,
        ValueError,
    ) as exc:
        print(f"Training provenance quarantined: {exc}", file=sys.stderr)
        return 1

    print("Training provenance verified")
    print(f"- provenance: {provenance_path}")
    print(f"- reproducibility_hash: {verified_hash}")
    return 0


def _resolve_provenance_path(
    provenance: Path | None,
    checkpoint_dir: Path | None,
) -> Path:
    if provenance is not None and checkpoint_dir is not None:
        raise ValueError("use either --provenance or --checkpoint-dir, not both")
    if provenance is not None:
        return provenance
    if checkpoint_dir is None:
        raise ValueError("one of --provenance or --checkpoint-dir is required")
    path = checkpoint_dir / TRAINING_PROVENANCE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{TRAINING_PROVENANCE_FILENAME} is missing from {checkpoint_dir}"
        )
    return path


def _load_manifest_row(path: Path, repo_id: Any) -> dict[str, Any]:
    selected_repo_id = "" if repo_id is None else str(repo_id).strip()
    if not selected_repo_id:
        raise ValueError("--manifest requires --repo-id or provenance repo_id")

    manifest_path = path if path.is_absolute() else ROOT / path
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("repo_id") == selected_repo_id:
                return row

    raise ValueError(f"repo_id {selected_repo_id!r} was not found in {manifest_path}")


if __name__ == "__main__":
    raise SystemExit(main())
