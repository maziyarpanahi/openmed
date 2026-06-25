"""Schema validation for the canonical OpenMed model manifest."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "models.jsonl"

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
BENCHMARK_FIELDS = frozenset({"dataset", "micro_f1", "recall"})

ALLOWED_TIERS = ("Tiny", "Small", "Base", "Medium", "Large", "XLarge")
ALLOWED_FORMATS = ("pytorch", "mlx-fp", "mlx-8bit", "onnx", "gguf", "unknown")
ALLOWED_LICENSES = ("apache-2.0", "other")

REPRODUCIBILITY_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}$")
RELEASED_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class ManifestViolation:
    """A schema violation found on a manifest line."""

    line_number: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line_number}: {self.message}"


@dataclass(frozen=True)
class ManifestValidationResult:
    """Validation result for a model manifest."""

    path: Path
    row_count: int
    violations: tuple[ManifestViolation, ...]

    @property
    def ok(self) -> bool:
        """Return whether the manifest passed schema validation."""
        return not self.violations


def validate_manifest_file(
    path: str | Path = MANIFEST_PATH,
) -> ManifestValidationResult:
    """Validate every JSONL row in *path* against the manifest schema."""
    manifest_path = Path(path)
    row_count = 0
    violations: list[ManifestViolation] = []

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            row_count += 1
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                violations.append(
                    ManifestViolation(
                        line_number,
                        f"invalid JSON: {exc.msg}",
                    )
                )
                continue

            violations.extend(validate_manifest_row(row, line_number))

    return ManifestValidationResult(
        path=manifest_path,
        row_count=row_count,
        violations=tuple(violations),
    )


def validate_manifest_row(row: Any, line_number: int) -> list[ManifestViolation]:
    """Return schema violations for one decoded manifest row."""
    violations: list[ManifestViolation] = []
    if not isinstance(row, dict):
        return [
            ManifestViolation(
                line_number,
                "row must be a JSON object",
            )
        ]

    fields = set(row)
    for field in sorted(REQUIRED_FIELDS - fields):
        violations.append(
            ManifestViolation(line_number, f"missing required key: {field}")
        )
    for field in sorted(fields - REQUIRED_FIELDS):
        violations.append(ManifestViolation(line_number, f"unexpected key: {field}"))

    _validate_non_empty_string(violations, line_number, row, "family")
    _validate_non_empty_string(violations, line_number, row, "task")
    _validate_optional_string(violations, line_number, row, "architecture")
    _validate_optional_string(violations, line_number, row, "base_model")
    _validate_optional_string(violations, line_number, row, "arxiv")

    if "repo_id" in row:
        value = row["repo_id"]
        if not isinstance(value, str) or not value.startswith("OpenMed/"):
            violations.append(
                ManifestViolation(
                    line_number,
                    "repo_id must be a string starting with 'OpenMed/'",
                )
            )

    if "languages" in row:
        _validate_string_list(violations, line_number, row["languages"], "languages")

    if "tier" in row:
        value = row["tier"]
        if value is not None and not isinstance(value, str):
            violations.append(
                ManifestViolation(line_number, "tier must be a string or null")
            )
        elif value is not None and value not in ALLOWED_TIERS:
            violations.append(
                ManifestViolation(
                    line_number,
                    f"tier must be one of: {_allowed(ALLOWED_TIERS, allow_null=True)}",
                )
            )

    if "param_count" in row:
        value = row["param_count"]
        if value is not None and (type(value) is not int or value <= 0):
            violations.append(
                ManifestViolation(
                    line_number,
                    "param_count must be a positive integer or null",
                )
            )

    if "formats" in row:
        value = row["formats"]
        if not isinstance(value, list):
            violations.append(ManifestViolation(line_number, "formats must be a list"))
        elif not value:
            violations.append(
                ManifestViolation(line_number, "formats must not be empty")
            )
        else:
            for index, format_name in enumerate(value):
                if not isinstance(format_name, str):
                    violations.append(
                        ManifestViolation(
                            line_number,
                            f"formats[{index}] must be a string",
                        )
                    )
                elif format_name not in ALLOWED_FORMATS:
                    violations.append(
                        ManifestViolation(
                            line_number,
                            f"formats must contain only: {_allowed(ALLOWED_FORMATS)}",
                        )
                    )

    if "canonical_labels" in row:
        _validate_string_list(
            violations,
            line_number,
            row["canonical_labels"],
            "canonical_labels",
        )

    if "benchmark" in row:
        _validate_benchmark(violations, line_number, row["benchmark"])

    if "license" in row:
        value = row["license"]
        if value is not None and not isinstance(value, str):
            violations.append(
                ManifestViolation(line_number, "license must be a string or null")
            )
        elif value is not None and value not in ALLOWED_LICENSES:
            violations.append(
                ManifestViolation(
                    line_number,
                    "license must be one of: "
                    f"{_allowed(ALLOWED_LICENSES, allow_null=True)}",
                )
            )

    if "reproducibility_hash" in row:
        value = row["reproducibility_hash"]
        if not isinstance(value, str) or not REPRODUCIBILITY_HASH_RE.fullmatch(value):
            violations.append(
                ManifestViolation(
                    line_number,
                    "reproducibility_hash must match "
                    "sha256:<64 lowercase hex characters>",
                )
            )

    if "released" in row:
        value = row["released"]
        if value is not None and not isinstance(value, str):
            violations.append(
                ManifestViolation(line_number, "released must be a string or null")
            )
        elif value is not None and not RELEASED_DATE_RE.fullmatch(value):
            violations.append(
                ManifestViolation(line_number, "released must match YYYY-MM-DD")
            )

    return violations


def format_manifest_validation(result: ManifestValidationResult) -> list[str]:
    """Format a validation result for CLI output."""
    if result.ok:
        return [f"{result.path}: OK ({result.row_count} rows checked)"]
    return [str(violation) for violation in result.violations]


def _validate_non_empty_string(
    violations: list[ManifestViolation],
    line_number: int,
    row: Mapping[str, Any],
    field: str,
) -> None:
    if field not in row:
        return
    value = row[field]
    if not isinstance(value, str) or not value:
        violations.append(
            ManifestViolation(line_number, f"{field} must be a non-empty string")
        )


def _validate_optional_string(
    violations: list[ManifestViolation],
    line_number: int,
    row: Mapping[str, Any],
    field: str,
) -> None:
    if field not in row:
        return
    value = row[field]
    if value is not None and not isinstance(value, str):
        violations.append(
            ManifestViolation(line_number, f"{field} must be a string or null")
        )


def _validate_string_list(
    violations: list[ManifestViolation],
    line_number: int,
    value: Any,
    field: str,
) -> None:
    if not isinstance(value, list):
        violations.append(ManifestViolation(line_number, f"{field} must be a list"))
        return

    for index, item in enumerate(value):
        if not isinstance(item, str):
            violations.append(
                ManifestViolation(line_number, f"{field}[{index}] must be a string")
            )


def _validate_benchmark(
    violations: list[ManifestViolation],
    line_number: int,
    value: Any,
) -> None:
    if not isinstance(value, dict):
        violations.append(ManifestViolation(line_number, "benchmark must be an object"))
        return

    fields = set(value)
    for field in sorted(BENCHMARK_FIELDS - fields):
        violations.append(
            ManifestViolation(line_number, f"benchmark missing required key: {field}")
        )
    if "dataset" in value and value["dataset"] is not None:
        if not isinstance(value["dataset"], str):
            violations.append(
                ManifestViolation(
                    line_number,
                    "benchmark.dataset must be a string or null",
                )
            )
    for metric in ("micro_f1", "recall"):
        if metric not in value or value[metric] is None:
            continue
        if not isinstance(value[metric], (int, float)) or isinstance(
            value[metric], bool
        ):
            violations.append(
                ManifestViolation(
                    line_number,
                    f"benchmark.{metric} must be a number or null",
                )
            )


def _allowed(values: tuple[str, ...], *, allow_null: bool = False) -> str:
    allowed = list(values)
    if allow_null:
        allowed.append("null")
    if len(allowed) == 1:
        return allowed[0]
    return f"{', '.join(allowed[:-1])}, or {allowed[-1]}"
