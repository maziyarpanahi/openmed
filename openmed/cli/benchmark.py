"""Benchmark command wiring for the OpenMed CLI."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.model_registry import MANIFEST_PATH, load_manifest_rows
from openmed.eval.golden import load_benchmark_fixtures
from openmed.eval.harness import run_benchmark
from openmed.eval.report import BenchmarkReport


PII_SUITES = ("golden", "i2b2", "n2c2", "shield")
PII_DEVICES = ("cpu", "mlx", "coreml")
CLINICAL_SUITES = ("drugprot", "medmentions", "context")
CLINICAL_TASKS = ("ner", "linking", "assertion", "relation")
MOBILE_DEVICES = ("mlx", "coreml")
MOBILE_TIERS = ("phone", "laptop", "workstation", "server")
DEFAULT_OUTPUT_DIR = Path("benchmark-reports")

_PATH_TOKEN_RE = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass(frozen=True)
class BenchmarkOutput:
    """Paths produced for one model benchmark run."""

    model_name: str
    report: BenchmarkReport
    json_path: Path
    markdown_path: Path


def add_benchmark_command(subparsers: argparse._SubParsersAction) -> None:
    """Add benchmark subcommands to the argparse CLI."""
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run OpenMed benchmark suites.",
    )
    benchmark_parser.set_defaults(handler=_handle_missing_benchmark_command)
    benchmark_sub = benchmark_parser.add_subparsers(dest="benchmark_command")

    pii_parser = benchmark_sub.add_parser(
        "pii",
        help="Run PII benchmark suites.",
    )
    pii_parser.add_argument(
        "--suite",
        choices=PII_SUITES,
        required=True,
        help="PII suite to run.",
    )
    pii_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="MODEL",
        help="Model id(s), comma-separated ids, or @manifest.",
    )
    pii_parser.add_argument(
        "--device",
        choices=PII_DEVICES,
        default="cpu",
        help="Runtime device label.",
    )
    pii_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSON and Markdown reports are written.",
    )
    pii_parser.add_argument(
        "--fixture-path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    pii_parser.set_defaults(handler=handle_pii_benchmark)

    clinical_parser = benchmark_sub.add_parser(
        "clinical",
        help="Parse clinical benchmark options.",
    )
    clinical_parser.add_argument(
        "--suite",
        choices=CLINICAL_SUITES,
        required=True,
        help="Clinical suite to run.",
    )
    clinical_parser.add_argument(
        "--task",
        choices=CLINICAL_TASKS,
        required=True,
        help="Clinical scoring task.",
    )
    clinical_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where benchmark reports will be written.",
    )
    clinical_parser.set_defaults(handler=handle_clinical_benchmark)

    mobile_parser = benchmark_sub.add_parser(
        "mobile",
        help="Parse mobile benchmark options.",
    )
    mobile_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="MODEL",
        help="Model id(s) or comma-separated ids.",
    )
    mobile_parser.add_argument(
        "--device",
        choices=MOBILE_DEVICES,
        required=True,
        help="Mobile runtime device.",
    )
    mobile_parser.add_argument(
        "--tier",
        choices=MOBILE_TIERS,
        required=True,
        help="Device tier to benchmark.",
    )
    mobile_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where benchmark reports will be written.",
    )
    mobile_parser.set_defaults(handler=handle_mobile_benchmark)


def resolve_model_ids(
    values: Sequence[str],
    *,
    manifest_path: Path = MANIFEST_PATH,
) -> list[str]:
    """Resolve explicit model ids or the canonical @manifest shortcut."""
    tokens = _split_model_tokens(values)
    if not tokens:
        raise ValueError("--models requires at least one model id")

    if tokens == ["@manifest"]:
        rows = load_manifest_rows(manifest_path)
        model_ids = [
            str(row["repo_id"])
            for row in rows
            if isinstance(row.get("repo_id"), str) and row["repo_id"]
        ]
        if not model_ids:
            raise ValueError(f"model manifest is empty: {manifest_path}")
        return model_ids

    if "@manifest" in tokens:
        raise ValueError("--models @manifest cannot be combined with explicit ids")

    return tokens


def run_pii_benchmark(
    *,
    suite: str,
    models: Sequence[str],
    device: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    fixture_path: Path | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[BenchmarkOutput]:
    """Run the PII benchmark suite for the requested models."""
    model_ids = resolve_model_ids(models)
    if suite != "golden":
        raise NotImplementedError(
            f"PII benchmark suite '{suite}' is not implemented yet; "
            "only 'golden' is currently runnable."
        )

    fixtures = load_benchmark_fixtures(fixture_path)
    timestamp = generated_at or _utc_timestamp()
    outputs: list[BenchmarkOutput] = []
    for model_name in model_ids:
        json_path, markdown_path = report_paths(
            output_dir=output_dir,
            domain="pii",
            suite=suite,
            model_name=model_name,
            device=device,
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        report_metadata = {
            "benchmark_domain": "pii",
            "source_suite": suite,
            "output_json": str(json_path),
            "output_markdown": str(markdown_path),
        }
        report_metadata.update(dict(metadata or {}))
        report = run_benchmark(
            fixtures,
            suite=suite,
            model_name=model_name,
            device=device,
            generated_at=timestamp,
            metadata=report_metadata,
        )
        report.write_json(json_path)
        report.write_markdown(markdown_path)
        outputs.append(
            BenchmarkOutput(
                model_name=model_name,
                report=report,
                json_path=json_path,
                markdown_path=markdown_path,
            )
        )
    return outputs


def report_paths(
    *,
    output_dir: Path,
    domain: str,
    suite: str,
    model_name: str,
    device: str,
) -> tuple[Path, Path]:
    """Return stable JSON and Markdown paths for a benchmark report."""
    stem = f"{_path_token(model_name)}-{_path_token(device)}"
    directory = output_dir / domain / suite
    return directory / f"{stem}.json", directory / f"{stem}.md"


def handle_pii_benchmark(args: argparse.Namespace) -> int:
    """Run a PII benchmark from parsed CLI args."""
    try:
        outputs = run_pii_benchmark(
            suite=args.suite,
            models=args.models,
            device=args.device,
            output_dir=args.output_dir,
            fixture_path=args.fixture_path,
        )
    except (NotImplementedError, ValueError, OSError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    sys.stdout.write("Benchmark reports written:\n")
    for output in outputs:
        sys.stdout.write(f"  - {output.model_name}\n")
        sys.stdout.write(f"    JSON: {output.json_path}\n")
        sys.stdout.write(f"    Markdown: {output.markdown_path}\n")
    return 0


def handle_clinical_benchmark(args: argparse.Namespace) -> int:
    """Parse clinical flags and report the current implementation boundary."""
    sys.stderr.write(
        "Clinical benchmark execution is not implemented yet "
        f"for suite '{args.suite}' task '{args.task}'.\n"
    )
    return 1


def handle_mobile_benchmark(args: argparse.Namespace) -> int:
    """Parse mobile flags and report the current implementation boundary."""
    try:
        models = ", ".join(resolve_model_ids(args.models))
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stderr.write(
        "Mobile benchmark execution is not implemented yet "
        f"for tier '{args.tier}' on device '{args.device}' "
        f"with model(s): {models}.\n"
    )
    return 1


def _handle_missing_benchmark_command(args: argparse.Namespace) -> int:
    sys.stderr.write("Choose a benchmark command: pii, clinical, or mobile.\n")
    return 1


def _split_model_tokens(values: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(token.strip() for token in str(value).split(","))
    return [token for token in tokens if token]


def _path_token(value: str) -> str:
    token = _PATH_TOKEN_RE.sub("-", value).strip("-._")
    return token or "model"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


__all__ = [
    "CLINICAL_SUITES",
    "CLINICAL_TASKS",
    "DEFAULT_OUTPUT_DIR",
    "MOBILE_DEVICES",
    "MOBILE_TIERS",
    "PII_DEVICES",
    "PII_SUITES",
    "BenchmarkOutput",
    "add_benchmark_command",
    "handle_clinical_benchmark",
    "handle_mobile_benchmark",
    "handle_pii_benchmark",
    "report_paths",
    "resolve_model_ids",
    "run_pii_benchmark",
]
