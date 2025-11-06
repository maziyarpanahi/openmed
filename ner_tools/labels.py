"""Command-line helpers for managing zero-shot label defaults."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from openmed.ner.labels import (
    available_domains,
    get_default_labels,
    load_default_label_map,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ner_tools.labels",
        description="Inspect default zero-shot NER label maps.",
    )
    subparsers = parser.add_subparsers(dest="command")

    dump_parser = subparsers.add_parser(
        "dump-defaults",
        help="List all domains and their default label sets.",
    )
    dump_parser.add_argument(
        "--domain",
        help="Only show defaults for the specified domain.",
    )
    dump_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable table.",
    )
    dump_parser.set_defaults(handler=_handle_dump_defaults)

    return parser


def _handle_dump_defaults(args: argparse.Namespace) -> int:
    label_map = load_default_label_map()

    if args.domain:
        domain = args.domain.lower()
        labels = get_default_labels(domain, label_map=label_map)
        if not labels:
            sys.stderr.write(f"No defaults found for domain '{domain}'.\n")
            return 1
        if args.json:
            sys.stdout.write(
                json.dumps({domain: labels}, indent=2, sort_keys=True) + "\n"
            )
        else:
            joined = ", ".join(labels)
            sys.stdout.write(f"{domain}: {joined}\n")
        return 0

    domains = available_domains(label_map)
    payload = {domain: get_default_labels(domain, label_map=label_map) for domain in domains}

    if args.json:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0

    for domain in domains:
        joined = ", ".join(payload[domain]) or "(none)"
        sys.stdout.write(f"{domain}: {joined}\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0

    return handler(args)


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    raise SystemExit(main())
