#!/usr/bin/env python3
"""Build the Android on-device model catalog from the canonical manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

DEFAULT_MANIFEST = Path("models.jsonl")
DEFAULT_OUTPUT = Path("android/openmedkit/src/main/assets/openmed_model_catalog.jsonl")

ANDROID_CATALOG_FILENAME = "openmed_model_catalog.jsonl"
ANDROID_CATALOG_FIELDS = (
    "repo_id",
    "formats",
    "tier",
    "param_count",
    "languages",
    "license",
)

PERMISSIVE_LICENSES = {
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-3.0",
    "cc-by-4.0",
    "cc0-1.0",
    "isc",
    "mit",
    "unlicense",
}


def normalize_token(value: object) -> str:
    """Return a lowercase, hyphen-normalized manifest token."""
    return str(value).strip().lower().replace("_", "-")


def is_permissive_license(value: object) -> bool:
    """Return whether *value* is in the Android catalog license allowlist."""
    if value is None:
        return False
    return normalize_token(value) in PERMISSIVE_LICENSES


def is_android_runnable_format(value: object) -> bool:
    """Return whether *value* describes an Android-runnable model artifact."""
    normalized = normalize_token(value)
    return normalized.startswith("onnx") or normalized.startswith("tflite")


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    """Load JSONL manifest rows from *path*."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                message = f"{path}:{line_number}: invalid JSON: {exc.msg}"
                raise ValueError(message) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def build_catalog_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return Android catalog rows derived from canonical manifest *rows*."""
    catalog_rows: list[dict[str, Any]] = []
    for row in rows:
        license_name = row.get("license")
        if not is_permissive_license(license_name):
            continue

        android_formats = [
            str(format_name)
            for format_name in row.get("formats") or []
            if is_android_runnable_format(format_name)
        ]
        if not android_formats:
            continue

        repo_id = row.get("repo_id")
        if not repo_id:
            continue

        catalog_rows.append(
            {
                "repo_id": str(repo_id),
                "formats": android_formats,
                "tier": row.get("tier"),
                "param_count": row.get("param_count"),
                "languages": [str(language) for language in row.get("languages") or []],
                "license": str(license_name),
            }
        )

    return sorted(catalog_rows, key=lambda item: item["repo_id"].lower())


def write_catalog(rows: Iterable[Mapping[str, Any]], output: Path) -> None:
    """Write *rows* as deterministic JSONL to *output*."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            ordered = {field: row.get(field) for field in ANDROID_CATALOG_FIELDS}
            handle.write(json.dumps(ordered, separators=(",", ":")))
            handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Derive the Android OpenMedKit model catalog from the canonical "
            "models.jsonl manifest."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the canonical model manifest JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the derived Android catalog JSONL asset.",
    )
    return parser


def run(
    manifest: Path,
    output: Path,
    *,
    stdout: TextIO | None = None,
) -> int:
    """Build the Android catalog and return a process exit code."""
    stdout = stdout or None
    rows = load_manifest_rows(manifest)
    catalog_rows = build_catalog_rows(rows)
    write_catalog(catalog_rows, output)
    if stdout is not None:
        stdout.write(f"Wrote {len(catalog_rows)} catalog rows to {output}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Android catalog builder."""
    args = build_parser().parse_args(argv)
    return run(args.manifest, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
