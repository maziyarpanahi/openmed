#!/usr/bin/env python3
"""Merge benchmark and device measurements into model manifest rows."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("models.jsonl")

REQUIRED_FIELDS = frozenset(
    {
        "repo_id",
        "family",
        "task",
        "languages",
        "tier",
        "param_count",
        "architecture",
        "base_model",
        "formats",
        "canonical_labels",
        "benchmark",
        "arxiv",
        "license",
        "reproducibility_hash",
        "released",
    }
)
OPTIONAL_ENRICHMENT_FIELDS = frozenset(
    {"latency_ms", "peak_ram_mb", "recommended_tier"}
)
MANIFEST_FIELDS = REQUIRED_FIELDS | OPTIONAL_ENRICHMENT_FIELDS
RECOMMENDED_TIERS = frozenset({"phone", "laptop", "workstation", "server"})

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_REPRO_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")


def validate_manifest_row(row: Mapping[str, Any]) -> None:
    """Validate one manifest row, including optional enrichment fields."""

    missing = REQUIRED_FIELDS - set(row)
    if missing:
        raise ValueError(f"manifest row is missing required fields: {sorted(missing)}")

    unknown = set(row) - MANIFEST_FIELDS
    if unknown:
        raise ValueError(f"manifest row has unknown fields: {sorted(unknown)}")

    _validate_string(row, "repo_id", prefix="OpenMed/")
    _validate_string(row, "family")
    _validate_string(row, "task")
    _validate_string_list(row, "languages", allow_empty=True)
    _validate_nullable_string(row, "tier")
    _validate_nullable_positive_int(row, "param_count")
    _validate_nullable_string(row, "architecture")
    _validate_nullable_string(row, "base_model")
    _validate_string_list(row, "formats", allow_empty=False)
    _validate_string_list(row, "canonical_labels", allow_empty=True)
    _validate_benchmark(row["benchmark"])
    _validate_nullable_string(row, "arxiv")
    _validate_nullable_string(row, "license")
    _validate_reproducibility_hash(row["reproducibility_hash"])
    _validate_released(row["released"])

    if "latency_ms" in row:
        _validate_number_map(row["latency_ms"], "latency_ms")
    if "peak_ram_mb" in row:
        _validate_number_map(row["peak_ram_mb"], "peak_ram_mb")
    if "recommended_tier" in row:
        recommended_tier = row["recommended_tier"]
        if recommended_tier not in RECOMMENDED_TIERS:
            allowed = ", ".join(sorted(RECOMMENDED_TIERS))
            raise ValueError(
                f"recommended_tier must be one of {allowed}: {recommended_tier!r}"
            )


def load_measurements(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load enrichment measurements keyed by ``repo_id``."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    measurements: dict[str, dict[str, Any]] = {}

    for record in _measurement_records(payload):
        repo_id, measurement = _normalize_measurement(record)
        if repo_id in measurements:
            raise ValueError(f"duplicate measurement for repo_id: {repo_id}")
        measurements[repo_id] = measurement

    return measurements


def enrich_manifest_file(
    manifest_path: str | Path,
    results_path: str | Path,
    output_path: str | Path,
) -> int:
    """Write an enriched manifest and return the number of updated rows."""

    measurements = load_measurements(results_path)
    output_lines, updated = enrich_manifest_lines(
        Path(manifest_path).read_text(encoding="utf-8").splitlines(keepends=True),
        measurements,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(output_lines), encoding="utf-8")
    return updated


def enrich_manifest_lines(
    lines: Sequence[str], measurements: Mapping[str, Mapping[str, Any]]
) -> tuple[list[str], int]:
    """Return manifest JSONL lines with measurements merged by ``repo_id``."""

    output_lines: list[str] = []
    updated = 0

    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            output_lines.append(raw_line)
            continue

        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid manifest JSON at line {line_number}: {exc}"
            ) from exc

        if not isinstance(row, dict):
            raise ValueError(f"manifest line {line_number} must be a JSON object")
        validate_manifest_row(row)

        repo_id = row["repo_id"]
        measurement = measurements.get(repo_id)
        if measurement is None:
            output_lines.append(raw_line)
            continue

        enriched = dict(row)
        for field in ("benchmark", "latency_ms", "peak_ram_mb", "recommended_tier"):
            if field in measurement:
                enriched[field] = measurement[field]

        validate_manifest_row(enriched)
        output_lines.append(json.dumps(enriched, separators=(",", ":")) + "\n")
        updated += 1

    return output_lines, updated


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for manifest enrichment."""

    parser = argparse.ArgumentParser(
        description="Merge benchmark and latency results into models.jsonl."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="JSON measurements keyed by repo_id or a list under results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the enriched JSONL manifest.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the manifest enrichment command."""

    args = parse_args(argv)
    updated = enrich_manifest_file(args.manifest, args.results, args.output)
    print(f"Wrote enriched manifest to {args.output} ({updated} rows updated)")


def _measurement_records(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, list):
        return _ensure_mapping_records(payload)

    if isinstance(payload, dict):
        for key in ("results", "measurements", "models"):
            records = payload.get(key)
            if isinstance(records, list):
                return _ensure_mapping_records(records)
            if isinstance(records, dict):
                return _records_from_repo_mapping(records)
        return _records_from_repo_mapping(payload)

    raise ValueError("measurement results must be a JSON object or list")


def _ensure_mapping_records(records: Iterable[Any]) -> list[Mapping[str, Any]]:
    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"measurement record {index} must be a JSON object")
        normalized.append(record)
    return normalized


def _records_from_repo_mapping(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for repo_id, record in payload.items():
        if not isinstance(record, Mapping):
            continue
        if "repo_id" in record:
            records.append(record)
        else:
            records.append({"repo_id": repo_id, **record})
    return records


def _normalize_measurement(record: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    repo_id = record.get("repo_id") or record.get("model_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("measurement record must include repo_id")

    measurement: dict[str, Any] = {}
    for field in ("latency_ms", "peak_ram_mb", "recommended_tier"):
        if field in record:
            measurement[field] = record[field]

    if "benchmark" in record:
        measurement["benchmark"] = record["benchmark"]
    elif "benchmarks" in record:
        measurement["benchmark"] = record["benchmarks"]
    elif "benchmark_suites" in record:
        measurement["benchmark"] = record["benchmark_suites"]

    return repo_id, measurement


def _validate_string(
    row: Mapping[str, Any], field: str, *, prefix: str | None = None
) -> None:
    value = row[field]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if prefix and not value.startswith(prefix):
        raise ValueError(f"{field} must start with {prefix!r}")


def _validate_nullable_string(row: Mapping[str, Any], field: str) -> None:
    value = row[field]
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")


def _validate_string_list(
    row: Mapping[str, Any], field: str, *, allow_empty: bool
) -> None:
    value = row[field]
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(f"{field} must be a list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must contain only strings")


def _validate_nullable_positive_int(row: Mapping[str, Any], field: str) -> None:
    value = row[field]
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer or null")


def _validate_benchmark(value: Any) -> None:
    if isinstance(value, Mapping):
        missing = {"dataset", "micro_f1", "recall"} - set(value)
        if missing:
            raise ValueError(f"benchmark is missing fields: {sorted(missing)}")
        _validate_nullable_text(value["dataset"], "benchmark.dataset")
        _validate_metric(value["micro_f1"], "benchmark.micro_f1")
        _validate_metric(value["recall"], "benchmark.recall")
        if "leakage" in value:
            _validate_metric(value["leakage"], "benchmark.leakage")
        return

    if isinstance(value, list):
        for index, suite in enumerate(value):
            _validate_benchmark_suite(suite, index)
        return

    raise ValueError("benchmark must be a legacy object or suite list")


def _validate_benchmark_suite(value: Any, index: int) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"benchmark suite {index} must be a JSON object")

    required = {"suite", "dataset", "micro_f1", "recall", "leakage"}
    missing = required - set(value)
    if missing:
        raise ValueError(
            f"benchmark suite {index} is missing fields: {sorted(missing)}"
        )

    suite = value["suite"]
    if not isinstance(suite, str) or not suite:
        raise ValueError(f"benchmark suite {index}.suite must be a non-empty string")
    _validate_nullable_text(value["dataset"], f"benchmark suite {index}.dataset")
    _validate_metric(value["micro_f1"], f"benchmark suite {index}.micro_f1")
    _validate_metric(value["recall"], f"benchmark suite {index}.recall")
    _validate_metric(value["leakage"], f"benchmark suite {index}.leakage")


def _validate_number_map(value: Any, field: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a per-device object")
    for device, measurement in value.items():
        if not isinstance(device, str) or not device:
            raise ValueError(f"{field} device names must be non-empty strings")
        if not _is_number(measurement) or float(measurement) < 0:
            raise ValueError(f"{field}.{device} must be a non-negative number")


def _validate_nullable_text(value: Any, field: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")


def _validate_metric(value: Any, field: str) -> None:
    if value is not None and not _is_number(value):
        raise ValueError(f"{field} must be a number or null")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_reproducibility_hash(value: Any) -> None:
    if not isinstance(value, str) or _REPRO_HASH_RE.fullmatch(value) is None:
        raise ValueError("reproducibility_hash must be sha256:<64 lower hex chars>")


def _validate_released(value: Any) -> None:
    if value is not None and (
        not isinstance(value, str) or not _DATE_RE.fullmatch(value)
    ):
        raise ValueError("released must be YYYY-MM-DD or null")


if __name__ == "__main__":
    main()
