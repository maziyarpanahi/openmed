#!/usr/bin/env python3
"""Build the typed ONNX model catalog shipped by the ``openmed`` web package.

The catalog is derived from the canonical ``models.jsonl`` manifest with the
same selection rules and the same row fields as the Android catalog
(``scripts/android/build_android_catalog.py``), restricted to ONNX formats
because ONNX Runtime Web and Transformers.js only load ONNX artifacts:

* the row's license is in the shared permissive allowlist;
* at least one manifest format starts with ``onnx``; only those formats are
  kept;
* the row has a ``repo_id``.

Rows are sorted by case-folded ``repo_id`` and written to
``js/openmedkit-web/src/catalog.generated.ts`` as one JSON object per line
inside a frozen TypeScript array, so a manifest refresh shows up as a readable
per-model diff. The file also carries ``MODEL_CATALOG_DIGEST``: the SHA-256 of
the same rows serialized as compact JSONL in Android catalog field order,
which lets any platform confirm it holds the same rows.

The tool is dependency-free and performs no network operation. ``--check``
fails instead of writing when the committed file is stale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.android.build_android_catalog import (  # noqa: E402
    ANDROID_CATALOG_FIELDS,
    is_permissive_license,
    load_manifest_rows,
    normalize_token,
)

DEFAULT_MANIFEST = ROOT / "models.jsonl"
DEFAULT_OUTPUT = ROOT / "js" / "openmedkit-web" / "src" / "catalog.generated.ts"

#: Version tag of the generated catalog file; bump it when the row shape changes.
WEB_CATALOG_FORMAT = "openmed.web-model-catalog.v1"
#: Row fields, in order: the leading seven Android catalog fields. The family,
#: task and canonical-label fields Android added in 4.1 are not in the web
#: catalog yet (#5833), so the bundled catalog does not change shape.
WEB_CATALOG_FIELDS = (
    "repo_id",
    "formats",
    "tier",
    "param_count",
    "languages",
    "license",
    "reproducibility_hash",
)
#: Largest integer a JavaScript number holds exactly (2**53 - 1).
MAX_SAFE_INTEGER = 2**53 - 1
REGENERATE_COMMAND = "python scripts/web/build_web_catalog.py"

HEADER = f"""\
// GENERATED FILE. Do not edit by hand.
//
// Source: models.jsonl, through scripts/web/build_web_catalog.py.
// Regenerate: {REGENERATE_COMMAND}
// Check:      {REGENERATE_COMMAND} --check
//
// Each row is one permissively licensed manifest model with at least one ONNX
// format, using the Android catalog fields. Import the typed API from
// "openmed/catalog" instead of this file.

import type {{ ModelCatalogEntry }} from "./catalog";
"""


class WebCatalogError(ValueError):
    """Raised when the manifest cannot produce a valid web catalog."""


def is_web_runnable_format(value: object) -> bool:
    """Return whether *value* names an ONNX artifact the web runtimes can load.

    Args:
        value: A manifest ``formats`` entry.

    Returns:
        ``True`` when the normalized format token starts with ``onnx``.
    """
    return normalize_token(value).startswith("onnx")


def _optional_string(value: object, field: str, repo_id: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise WebCatalogError(f"{repo_id}: {field} must be a string or null")
    return value


def _param_count(value: object, repo_id: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise WebCatalogError(f"{repo_id}: param_count must be an integer or null")
    if value < 0 or value > MAX_SAFE_INTEGER:
        raise WebCatalogError(
            f"{repo_id}: param_count must be a non-negative safe integer"
        )
    return value


def build_web_catalog_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return web catalog rows derived from canonical manifest *rows*.

    Args:
        rows: Parsed ``models.jsonl`` rows.

    Returns:
        Rows holding exactly :data:`WEB_CATALOG_FIELDS`, sorted by case-folded
        ``repo_id``.

    Raises:
        WebCatalogError: If a selected row has a field of the wrong type, a
            ``param_count`` outside the JavaScript safe-integer range, or a
            repo id that appears twice.
    """
    catalog_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        license_name = row.get("license")
        if not is_permissive_license(license_name):
            continue
        formats = [
            str(format_name)
            for format_name in row.get("formats") or []
            if is_web_runnable_format(format_name)
        ]
        if not formats:
            continue
        repo_id = row.get("repo_id")
        if not repo_id:
            continue
        repo_id = str(repo_id)
        if repo_id in seen:
            raise WebCatalogError(f"{repo_id}: duplicate repo_id")
        seen.add(repo_id)
        reproducibility_hash = row.get("reproducibility_hash")
        if not isinstance(reproducibility_hash, str) or not reproducibility_hash:
            raise WebCatalogError(f"{repo_id}: reproducibility_hash is required")
        catalog_rows.append(
            {
                "repo_id": repo_id,
                "formats": formats,
                "tier": _optional_string(row.get("tier"), "tier", repo_id),
                "param_count": _param_count(row.get("param_count"), repo_id),
                "languages": [str(language) for language in row.get("languages") or []],
                "license": str(license_name),
                "reproducibility_hash": reproducibility_hash,
            }
        )
    return sorted(catalog_rows, key=lambda item: item["repo_id"].lower())


def _row_json(row: Mapping[str, Any]) -> str:
    ordered = {field: row.get(field) for field in WEB_CATALOG_FIELDS}
    return json.dumps(ordered, separators=(",", ":"))


def catalog_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return the ``sha256:`` digest of *rows* as compact JSONL.

    The bytes hashed are exactly what the Android catalog writer emits for the
    same rows: one compact JSON object per line in field order, each line
    terminated by ``\\n``.

    Args:
        rows: Web catalog rows.

    Returns:
        ``"sha256:"`` followed by the lowercase hex digest.
    """
    payload = "".join(f"{_row_json(row)}\n" for row in rows)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_catalog_module(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the generated TypeScript module for *rows*.

    Args:
        rows: Web catalog rows from :func:`build_web_catalog_rows`.

    Returns:
        The full text of ``catalog.generated.ts``.
    """
    lines = [
        HEADER,
        "/** Version tag of this generated catalog file. */",
        f'export const MODEL_CATALOG_FORMAT = "{WEB_CATALOG_FORMAT}";',
        "",
        "/**",
        " * SHA-256 of the rows below as compact JSONL in field order (the bytes",
        " * the Android catalog writer emits for the same rows).",
        " */",
        f'export const MODEL_CATALOG_DIGEST = "{catalog_digest(rows)}";',
        "",
        "/** Bundled catalog rows, sorted by case-folded repo id. */",
        "export const MODEL_CATALOG_ROWS: readonly ModelCatalogEntry[] = [",
    ]
    lines.extend(f"  {_row_json(row)}," for row in rows)
    lines.append("];")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Derive the openmed web package ONNX model catalog from the "
            "canonical models.jsonl manifest."
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
        help="Path of the generated TypeScript catalog module.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when the generated module is stale.",
    )
    return parser


def run(
    manifest: Path,
    output: Path,
    *,
    check: bool = False,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Build or check the web catalog module.

    Args:
        manifest: Canonical manifest path.
        output: Generated TypeScript module path.
        check: Compare instead of writing.
        stdout: Stream for the success summary.
        stderr: Stream for failure messages.

    Returns:
        ``0`` when the module is current or was written, ``1`` when ``check``
        found it stale and ``2`` when the manifest is unusable.
    """
    try:
        rows = build_web_catalog_rows(load_manifest_rows(manifest))
    except (OSError, ValueError) as error:
        if stderr is not None:
            # Name the manifest by file name only; never echo a local path.
            reason = str(error).replace(str(manifest), manifest.name)
            if isinstance(error, OSError):
                reason = f"{manifest.name}: {type(error).__name__}"
            stderr.write(f"web model catalog not built: {reason}\n")
        return 2
    if not rows:
        if stderr is not None:
            stderr.write(
                "Refusing to write an empty web model catalog: the manifest has "
                "no permissively licensed ONNX rows.\n"
            )
        return 2
    rendered = render_catalog_module(rows)
    current = output.read_text(encoding="utf-8") if output.is_file() else None
    if check:
        if current != rendered:
            if stderr is not None:
                stderr.write(
                    f"web model catalog is stale ({output.name}); run "
                    f"{REGENERATE_COMMAND}\n"
                )
            return 1
        if stdout is not None:
            stdout.write(f"web model catalog current: {len(rows)} rows\n")
        return 0
    if current != rendered:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    if stdout is not None:
        stdout.write(f"web model catalog written: {len(rows)} rows\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the web catalog builder."""
    args = build_parser().parse_args(argv)
    return run(
        args.manifest,
        args.output,
        check=args.check,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
