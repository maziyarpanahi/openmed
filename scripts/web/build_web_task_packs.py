#!/usr/bin/env python3
"""Build the web runtime view of the portable task packs for the web package.

The ``openmed`` web package resolves task packs such as ``pii`` from the same
manifest the Python resolver reads, ``openmed/core/data/task_packs.json``.
Browsers cannot read that file, so this tool validates it with
:func:`openmed.core.task_packs.validate_task_packs`, keeps each pack's ``web``
runtime entry with the fields :func:`openmed.core.task_packs.resolve`
returns, and joins every candidate with its row in the web ONNX model catalog
(the rows ``scripts/web/build_web_catalog.py`` ships). A candidate that is
missing from the catalog, or whose license or parameter count differs, fails
the build with ``catalog_mismatch``. The result is written to
``js/openmedkit-web/src/task-packs.generated.ts`` with the notice texts the
packs reference and the SHA-256 of the manifest bytes.

Joining at build time keeps the catalog rows out of the web package's main
entry. The tool is offline and byte-stable; ``--check`` writes nothing and
exits non-zero when the committed module is stale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openmed.core.task_packs import (  # noqa: E402
    TASK_PACKS_DATA_FILE,
    TaskPackError,
    validate_task_packs,
)
from scripts.android.build_android_catalog import load_manifest_rows  # noqa: E402
from scripts.web.build_web_catalog import build_web_catalog_rows  # noqa: E402

DEFAULT_SOURCE = TASK_PACKS_DATA_FILE
DEFAULT_MODELS = ROOT / "models.jsonl"
DEFAULT_OUTPUT = ROOT / "js" / "openmedkit-web" / "src" / "task-packs.generated.ts"
WEB_RUNTIME = "web"
REGENERATE_COMMAND = "python scripts/web/build_web_task_packs.py"

HEADER = f"""\
// GENERATED FILE. Do not edit by hand.
//
// Source: openmed/core/data/task_packs.json and the web ONNX model catalog
// (models.jsonl), through scripts/web/build_web_task_packs.py.
// Regenerate: {REGENERATE_COMMAND}
// Check:      {REGENERATE_COMMAND} --check
//
// The web runtime entry of every portable task pack, with the fields the
// Python resolver (openmed.core.task_packs.resolve) returns and each
// candidate's catalog reproducibility hash. Use the typed API in "./packs"
// instead of this file.

import type {{ WebTaskPackData }} from "./packs";
"""


def web_task_packs(
    document: Mapping[str, Any],
    catalog_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the web runtime view of every pack, sorted by pack id.

    Args:
        document: A manifest that :func:`validate_task_packs` accepts.
        catalog_rows: Web catalog rows from
            :func:`scripts.web.build_web_catalog.build_web_catalog_rows`.

    Returns:
        One record per pack with the fields of ``resolve(pack_id, "web")``;
        each candidate also carries its catalog ``reproducibility_hash``.

    Raises:
        TaskPackError: ``catalog_mismatch`` when a candidate is not a catalog
            row with the same license and parameter count.
    """

    catalog = {row["repo_id"]: row for row in catalog_rows}
    packs = []
    for pack in sorted(document["packs"], key=lambda item: item["pack_id"]):
        entry = pack["runtimes"][WEB_RUNTIME]
        candidates = []
        for item in entry["candidates"]:
            row = catalog.get(item["repo_id"])
            if (
                row is None
                or str(row["license"]).lower() != str(item["license"]).lower()
                or row["param_count"] != item["param_count"]
            ):
                raise TaskPackError("catalog_mismatch")
            candidates.append(
                {
                    "rank": item["rank"],
                    "repo_id": item["repo_id"],
                    "param_count": item["param_count"],
                    "license": item["license"],
                    "reproducibility_hash": row["reproducibility_hash"],
                }
            )
        packs.append(
            {
                "pack_id": pack["pack_id"],
                "pack_version": pack["version"],
                "task": pack["task"],
                "runtime": WEB_RUNTIME,
                "format": entry["format"],
                "status": entry["status"],
                "candidates": candidates,
                "canonical_labels": list(pack["canonical_labels"]),
                "notice_key": pack["notice_key"],
            }
        )
    return packs


def render_module(source: bytes, catalog_rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the generated TypeScript module.

    Args:
        source: The ``task_packs.json`` bytes.
        catalog_rows: Web catalog rows the candidates are joined with.

    Returns:
        The module text, ending with a newline.

    Raises:
        TaskPackError: ``manifest_unreadable`` when the bytes are not JSON,
            ``catalog_mismatch`` from :func:`web_task_packs`, or a validation
            reason from :func:`validate_task_packs`.
    """

    try:
        document = json.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise TaskPackError("manifest_unreadable") from None
    validate_task_packs(document)
    packs = web_task_packs(document, catalog_rows)
    used = sorted({pack["notice_key"] for pack in packs})
    notices = {key: document["notices"][key] for key in used}

    def literal(value: Any) -> str:
        return json.dumps(value, indent=2, ensure_ascii=True)

    digest = hashlib.sha256(source).hexdigest()
    return (
        f"{HEADER}\n"
        f'export const TASK_PACKS_SCHEMA = "{document["schema"]}";\n'
        f'export const TASK_PACKS_SCHEMA_VERSION = "{document["schema_version"]}";\n'
        "\n"
        "/** SHA-256 of the task_packs.json bytes this module was built from. */\n"
        f'export const TASK_PACKS_SOURCE_DIGEST = "sha256:{digest}";\n'
        "\n"
        "/** Default English notice text by key. Kits localize by key. */\n"
        "export const TASK_PACK_NOTICES: Readonly<Record<string, string>> = "
        f"{literal(notices)};\n"
        "\n"
        "/** The web runtime entry of each pack, sorted by pack id. */\n"
        "export const WEB_TASK_PACKS: readonly WebTaskPackData[] = "
        f"{literal(packs)};\n"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the committed module is stale; write nothing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build or check the generated module and return an exit code."""

    args = build_parser().parse_args(argv)
    try:
        source = args.source.read_bytes()
        catalog_rows = build_web_catalog_rows(load_manifest_rows(args.models))
        text = render_module(source, catalog_rows)
    except TaskPackError as exc:
        print(f"web task pack build failed: {exc.reason}", file=sys.stderr)
        return 1
    except (OSError, ValueError):
        # Name no path or row: the inputs are the committed manifests.
        print("web task pack build failed: source_unreadable", file=sys.stderr)
        return 1
    if args.check:
        try:
            committed = args.output.read_text(encoding="utf-8")
        except OSError:
            committed = None
        if committed != text:
            print(
                f"{args.output.name} is stale; run {REGENERATE_COMMAND}",
                file=sys.stderr,
            )
            return 1
        print(f"{args.output.name} matches task_packs.json and the catalog.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
