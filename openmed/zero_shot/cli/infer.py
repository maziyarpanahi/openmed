"""CLI entry point for zero-shot NER inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from openmed.ner import MissingDependencyError, NerRequest, infer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m openmed.zero_shot.cli.infer",
        description="Run zero-shot NER inference using the zero-shot toolkit.",
    )
    parser.add_argument("--model-id", required=True, help="Model identifier from the index.")
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Inline text to analyse.")
    text_group.add_argument(
        "--input-file",
        type=Path,
        help="Path to a UTF-8 encoded file containing text to analyse.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score to keep an entity (default 0.5).",
    )
    parser.add_argument(
        "--labels",
        help="Comma separated list of labels to use (overrides domain defaults).",
    )
    parser.add_argument(
        "--domain",
        help="Domain to use when labels are not provided (falls back to model/domain defaults).",
    )
    parser.add_argument(
        "--index",
        type=Path,
        help="Path to models/index.json (defaults to packaged location).",
    )
    parser.add_argument(
        "--output",
        choices=["json", "pretty"],
        default="pretty",
        help="Output format (default pretty).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.input_file:
        try:
            text = args.input_file.read_text(encoding="utf-8")
        except OSError as exc:
            sys.stderr.write(f"Failed to read {args.input_file}: {exc}\n")
            return 1
    else:
        text = args.text

    labels = _parse_labels(args.labels)

    request = NerRequest(
        model_id=args.model_id,
        text=text,
        threshold=args.threshold,
        labels=labels,
        domain=args.domain,
    )

    try:
        response = infer(request, index_path=args.index)
    except FileNotFoundError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1
    except MissingDependencyError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1
    except Exception as exc:  # pragma: no cover - defensive path
        sys.stderr.write(f"Inference failed: {exc}\n")
        return 1

    if args.output == "json":
        sys.stdout.write(json.dumps(response.to_dict(), indent=2, sort_keys=True) + "\n")
    else:
        _print_pretty(response, stream=sys.stdout)
    return 0


def _parse_labels(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    labels = [label.strip() for label in raw.split(",")]
    cleaned = [label for label in labels if label]
    return cleaned or None


def _print_pretty(response, stream) -> None:
    stream.write(
        f"Model: {response.meta.get('model_id')} ({response.meta.get('family')})\n"
    )
    labels = response.meta.get("labels_used") or []
    if labels:
        stream.write("Labels: " + ", ".join(labels) + "\n")
    stream.write(f"Threshold: {response.meta.get('threshold')}\n")
    if not response.entities:
        stream.write("No entities found.\n")
        return
    for entity in response.entities:
        group_suffix = f" [group={entity.group}]" if entity.group else ""
        stream.write(
            f"- {entity.label}: '{entity.text}' ({entity.start}-{entity.end}) score={entity.score:.4f}{group_suffix}\n"
        )


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
