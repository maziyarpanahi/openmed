"""Command-line orchestration for the PII tokenizer script audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from openmed.core.manifest_schema import validate_manifest_row
from openmed.core.model_registry import load_manifest_rows
from openmed.eval.coverage import (
    TokenizerCoverageReport,
    audit_pii_tokenizers,
    load_transformers_tokenizer,
    update_manifest_script_coverage,
)

DEFAULT_MANIFEST = Path("models.jsonl")
DEFAULT_JSON_REPORT = Path("docs/model-tokenizer-script-coverage.json")
DEFAULT_MARKDOWN_REPORT = Path("docs/model-tokenizer-script-coverage.md")


def build_parser() -> argparse.ArgumentParser:
    """Build the tokenizer audit argument parser."""
    parser = argparse.ArgumentParser(
        description=("Audit all PII model tokenizers over Han and nine Indic scripts.")
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_REPORT,
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Populate script_coverage on every PII manifest entry.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume complete model results from the JSON output file.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write a resumable JSON checkpoint after this many models.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Bounded number of tokenizers to audit concurrently.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the complete tokenizer audit and write committed artifacts."""
    args = build_parser().parse_args(argv)
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be at least 1")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    rows = load_manifest_rows(args.manifest)
    pii_count = sum(row.get("family") == "PII" for row in rows)
    existing_models = _load_existing_models(args.json_output) if args.resume else {}
    completed = dict(existing_models)
    audited_since_checkpoint = 0

    def checkpoint(model_id: str, result: Mapping[str, object]) -> None:
        nonlocal audited_since_checkpoint
        completed[model_id] = result
        audited_since_checkpoint += 1
        print(
            f"Audited {len(completed)}/{pii_count}: {model_id}",
            file=sys.stderr,
            flush=True,
        )
        if audited_since_checkpoint >= args.checkpoint_every:
            _write_json(
                args.json_output,
                TokenizerCoverageReport(
                    models=completed,
                    model_count=len(completed),
                ).to_dict(),
            )
            audited_since_checkpoint = 0

    report = audit_pii_tokenizers(
        rows,
        tokenizer_loader=load_transformers_tokenizer,
        existing_models=existing_models,
        on_model=checkpoint,
        max_workers=args.workers,
    )
    _write_json(args.json_output, report.to_dict())
    _write_text(args.markdown_output, report.to_markdown())

    if args.update_manifest:
        updated_rows = update_manifest_script_coverage(rows, report)
        violations = [
            str(violation)
            for line_number, row in enumerate(updated_rows, start=1)
            for violation in validate_manifest_row(row, line_number)
        ]
        if violations:
            raise RuntimeError(
                "refusing to write an invalid manifest:\n" + "\n".join(violations)
            )
        _write_manifest(args.manifest, updated_rows)

    print(
        f"Audited {report.model_count} PII models across "
        f"{report.script_count} scripts.",
        file=sys.stdout,
    )
    return 0


def _load_existing_models(path: Path) -> dict[str, Mapping[str, object]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models") if isinstance(payload, Mapping) else None
    if not isinstance(models, Mapping):
        raise ValueError(f"resume report has no models object: {path}")
    return {
        str(model_id): result
        for model_id, result in models.items()
        if isinstance(result, Mapping)
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )


def _write_manifest(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    _write_text(path, text)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
