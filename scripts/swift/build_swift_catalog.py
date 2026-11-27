#!/usr/bin/env python3
"""Build the OpenMedKit Swift model catalog from the canonical manifest.

The Swift resource uses the catalog format of the Android asset built by
``scripts/android/build_android_catalog.py``:

* one compact JSON object per line, fields in ``ANDROID_CATALOG_FIELDS`` order
  (``repo_id``, ``formats``, ``tier``, ``param_count``, ``languages``,
  ``license``, ``reproducibility_hash``);
* permissively licensed rows only (the same license allowlist);
* rows sorted case-insensitively by ``repo_id``;
* ``formats`` narrowed to what the platform can run.

Only the format filter differs: the Swift catalog keeps the MLX and Core ML
formats Apple platforms run. Rows without a ``reproducibility_hash`` are
refused, because apps verify downloads against it.

Unlike the Android asset, which Gradle derives at build time, SwiftPM cannot
run the generator during a consumer build, so the resource is committed.
``--check`` rebuilds it in memory and fails when the committed file differs.

Usage::

    python3 scripts/swift/build_swift_catalog.py
    python3 scripts/swift/build_swift_catalog.py --check

Exit codes: 0 on success, 1 when the catalog would be empty, a row lacks its
reproducibility hash, or ``--check`` finds a stale resource.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.android.build_android_catalog import (  # noqa: E402
    ANDROID_CATALOG_FIELDS,
    build_catalog_rows,
    load_manifest_rows,
    normalize_token,
    render_catalog,
)

SWIFT_CATALOG_FILENAME = "openmed_model_catalog.jsonl"
SWIFT_CATALOG_FIELDS = ANDROID_CATALOG_FIELDS
DEFAULT_MANIFEST = ROOT / "models.jsonl"
DEFAULT_OUTPUT = (
    ROOT
    / "swift"
    / "OpenMedKit"
    / "Sources"
    / "OpenMedKit"
    / "Resources"
    / SWIFT_CATALOG_FILENAME
)
APPLE_FORMAT_PREFIXES = ("mlx", "coreml")


def is_apple_runnable_format(value: object) -> bool:
    """Return whether *value* names an artifact OpenMedKit can run on Apple.

    Args:
        value: A manifest ``formats`` entry such as ``mlx-fp`` or ``onnx``.

    Returns:
        ``True`` for MLX (``mlx``, ``mlx-fp``, ``mlx-8bit``, ...) and Core ML
        (``coreml``, ``coreml-int8``, ...) formats.
    """
    return normalize_token(value).startswith(APPLE_FORMAT_PREFIXES)


def build_swift_catalog_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return Swift catalog rows derived from canonical manifest *rows*.

    Args:
        rows: Canonical ``models.jsonl`` rows.

    Returns:
        Permissively licensed rows with at least one Apple-runnable format,
        sorted case-insensitively by ``repo_id``.

    Raises:
        ValueError: If a kept row has no ``reproducibility_hash``. The message
            names the row index in the derived catalog, not the repository.
    """
    catalog_rows = build_catalog_rows(rows, is_runnable_format=is_apple_runnable_format)
    for index, row in enumerate(catalog_rows):
        if not row.get("reproducibility_hash"):
            raise ValueError(
                f"catalog row {index} has no reproducibility_hash; "
                "regenerate models.jsonl before building the Swift catalog"
            )
    return catalog_rows


def render_swift_catalog(manifest: Path) -> str:
    """Return the Swift catalog text derived from *manifest*.

    Args:
        manifest: Path to a canonical ``models.jsonl`` file.

    Returns:
        Deterministic JSONL text, empty when no row qualifies.
    """
    return render_catalog(build_swift_catalog_rows(load_manifest_rows(manifest)))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Derive the OpenMedKit Swift model catalog resource from the "
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
        help="Path of the Swift catalog JSONL resource.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when the resource is missing or stale.",
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
    """Build or check the Swift catalog and return a process exit code.

    Args:
        manifest: Canonical ``models.jsonl`` path.
        output: Swift catalog resource path.
        check: Compare instead of writing.
        stdout: Stream for the success summary, or ``None`` for silence.
        stderr: Stream for failures, or ``None`` for silence.

    Returns:
        ``0`` on success, ``1`` on an empty catalog, a missing hash, or a
        stale resource in check mode.
    """

    def fail(message: str) -> int:
        if stderr is not None:
            stderr.write(message + "\n")
        return 1

    try:
        text = render_swift_catalog(manifest)
    except ValueError as exc:
        return fail(f"Refusing to build the Swift model catalog: {exc}.")
    if not text:
        return fail(
            "Refusing to write an empty Swift model catalog: the manifest "
            "contains no permissively licensed MLX/Core ML rows."
        )
    count = text.count("\n")
    if check:
        current = output.read_bytes() if output.is_file() else None
        if current != text.encode("utf-8"):
            return fail(
                "Swift model catalog is stale. Regenerate it with "
                "`python3 scripts/swift/build_swift_catalog.py` and commit "
                f"{output.name}."
            )
        if stdout is not None:
            stdout.write(f"Swift model catalog is current ({count} rows).\n")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(text.encode("utf-8"))
    if stdout is not None:
        stdout.write(f"Wrote {count} Swift catalog rows to {output.name}.\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Swift catalog builder."""
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
