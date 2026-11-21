#!/usr/bin/env python3
"""Apply reviewed family/task rules to the local model manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from openmed.core.intended_use import load_intended_use_registry
from openmed.core.manifest import load_manifest_rows, write_manifest_rows

RULE_VERSION = 1
# Regulatory tiers are read from the corresponding reviewed declarations.
CAPABILITIES = {
    ("PII", "token-classification"): "extract_pii",
    ("NER", "token-classification"): "analyze_text",
    ("ZeroShot", "token-classification"): "analyze_text",
    ("General", "text-generation"): "generate_text",
    ("Vision", "visual-question-answering"): "visual_qa",
}


def backfill(path: Path, *, dry_run: bool) -> dict:
    """Fill missing metadata and list unclassified or conflicting model IDs."""
    registry = load_intended_use_registry()
    rows = load_manifest_rows(path)
    counts = Counter()
    review = []
    changed = 0
    for row in rows:
        pair = (row.get("family"), row.get("task"))
        code = CAPABILITIES.get(pair)
        if code is None:
            review.append({"repo_id": row["repo_id"], "reason": "unknown_family_task"})
            continue
        tier = registry.resolve(code).regulatory_tier.value
        fields = {"capability": code, "regulatory_tier": tier}
        if any(key in row and row[key] != value for key, value in fields.items()):
            review.append(
                {"repo_id": row["repo_id"], "reason": "existing_declaration_differs"}
            )
            continue
        counts[tier] += 1
        if not fields.keys() <= row.keys():
            row.update(fields)
            changed += 1
    if changed and not dry_run:
        write_manifest_rows(rows, path)
    return {
        "rule_version": RULE_VERSION,
        "changed": changed,
        "counts": dict(sorted(counts.items())),
        "review": sorted(review, key=lambda item: item["repo_id"]),
    }


def main(argv=None) -> int:
    """Print a reproducible summary and optionally apply the reviewed rules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("models.jsonl"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report proposed changes without writing.",
    )
    args = parser.parse_args(argv)
    print(json.dumps(backfill(args.manifest, dry_run=args.dry_run), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
