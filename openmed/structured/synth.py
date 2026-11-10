"""Differentially private synthetic table generation and release gates.

Source rows remain inside this process. The optional synthesizer subprocess
receives only column schema and aggregate one/two-way statistics through the
strict bridge in :mod:`openmed.interop.bridges.dp_synth`. A release is written
only after the active epsilon policy, marginal/correlation utility caps, and a
nearest-row membership-inference advantage ceiling all pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from openmed.interop.bridges.dp_synth import (
    DEFAULT_ENGINE_TIMEOUT_SECONDS,
    DPSynthBridge,
)
from openmed.risk.budget import DPGenerationBudgetAccountant
from openmed.structured.table_io import read_table, write_table

SYNTHETIC_PRIVACY_REPORT_SCHEMA_VERSION = 1
DEFAULT_DP_SYNTH_SCOPE = "clinical_release_default"
DEFAULT_EVALUATION_HOLDOUT_FRACTION = 0.20
DEFAULT_SYNTHETIC_MARGINAL_MAE_CAP = 0.15
DEFAULT_SYNTHETIC_CORRELATION_MAE_CAP = 0.15
DEFAULT_MEMBERSHIP_ADVANTAGE_CEILING = 0.10
SUPPORTED_SYNTHETIC_OUTPUT_SUFFIXES = frozenset({".csv", ".parquet"})


class SyntheticDataGateError(ValueError):
    """Raised when a generated table fails utility or membership-risk gates."""

    def __init__(self, privacy_report: Mapping[str, Any]) -> None:
        self.privacy_report = dict(privacy_report)
        utility = privacy_report["utility"]
        membership = privacy_report["membership_inference"]
        super().__init__(
            "Synthetic table failed its release gate: "
            f"one-way MAE={utility['one_way_marginal_mae']:.6g}, "
            f"two-way MAE={utility['two_way_marginal_mae']:.6g}, "
            f"membership advantage={membership['advantage']:.6g}"
        )


@dataclass(frozen=True)
class SyntheticGenerationResult:
    """Paths and aggregate privacy evidence for a synthetic table release."""

    output_path: Path
    report_path: Path
    row_count: int
    privacy_report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible release summary."""

        return {
            "output_path": str(self.output_path),
            "report_path": str(self.report_path),
            "row_count": self.row_count,
            "privacy_report": dict(self.privacy_report),
        }


def generate_synthetic(
    path: str | Path,
    *,
    epsilon: float,
    delta: float,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    scope: str = DEFAULT_DP_SYNTH_SCOPE,
    accountant: DPGenerationBudgetAccountant | None = None,
    engine_command: str | Sequence[str] | None = None,
    heldout_path: str | Path | None = None,
    row_count: int | None = None,
    seed: int = 0,
    marginal_mae_cap: float = DEFAULT_SYNTHETIC_MARGINAL_MAE_CAP,
    correlation_mae_cap: float = DEFAULT_SYNTHETIC_CORRELATION_MAE_CAP,
    membership_advantage_ceiling: float = DEFAULT_MEMBERSHIP_ADVANTAGE_CEILING,
    enforce_gates: bool = True,
    overwrite: bool = False,
    timeout_seconds: float = DEFAULT_ENGINE_TIMEOUT_SECONDS,
) -> SyntheticGenerationResult:
    """Generate and gate a synthetic CSV or Parquet table.

    Args:
        path: Local CSV, TSV, JSONL, NDJSON, or Parquet source table.
        epsilon: Requested privacy epsilon charged before aggregate fitting.
        delta: Requested privacy delta charged before aggregate fitting.
        output_path: CSV or Parquet release path. Defaults beside ``path``.
        report_path: JSON privacy-report path. Defaults beside the output.
        scope: Active committed epsilon-policy scope.
        accountant: Reusable budget accountant. Passing one persists cumulative
            spend across calls; otherwise a fresh configured accountant is used.
        engine_command: User-supplied permissive engine executable and arguments.
        heldout_path: Optional separate real evaluation table. When omitted, a
            deterministic 20% source split is withheld from aggregate fitting.
        row_count: Number of synthetic rows. Defaults to the source row count.
        seed: Non-negative deterministic engine seed.
        marginal_mae_cap: Maximum one/two-way marginal MAE.
        correlation_mae_cap: Maximum numeric-correlation MAE.
        membership_advantage_ceiling: Maximum nearest-row attack advantage.
        enforce_gates: Refuse to write artifacts when a regression gate fails.
        overwrite: Permit replacement of existing output and report artifacts.
        timeout_seconds: Per-operation engine timeout.

    Returns:
        Release paths and a raw-value-free privacy report.

    Raises:
        BudgetExceeded: If the requested spend exceeds the active policy.
        DPSynthEngineUnavailable: If the optional engine is not installed.
        SyntheticDataGateError: If utility or membership risk exceeds its cap.
    """

    from openmed.eval.utility import (
        membership_inference_risk_report,
        synthetic_tabular_utility_report,
    )

    source_path = Path(path)
    selected_output = _output_path(source_path, output_path)
    selected_report = _report_path(selected_output, report_path)
    _validate_artifact_paths(
        source_path,
        heldout_path=Path(heldout_path) if heldout_path is not None else None,
        output_path=selected_output,
        report_path=selected_report,
        overwrite=overwrite,
    )
    if not isinstance(enforce_gates, bool):
        raise TypeError("enforce_gates must be boolean")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be boolean")

    bridge = DPSynthBridge(
        engine_command,
        timeout_seconds=timeout_seconds,
    )
    engine = bridge.capabilities()
    source_rows = read_table(source_path)
    if len(source_rows) < 2:
        raise ValueError(
            "DP synthesis requires at least two source rows for a held-out risk gate"
        )
    if heldout_path is None:
        training_rows, heldout_rows = _deterministic_holdout(source_rows)
        evaluation_source = "deterministic-source-holdout"
    else:
        training_rows = source_rows
        heldout_rows = read_table(heldout_path)
        evaluation_source = "separate-heldout-table"
        _require_same_schema(training_rows, heldout_rows, label="heldout")

    schema = _table_schema(training_rows)
    statistics = _aggregate_statistics(training_rows, schema)
    requested_rows = (
        len(source_rows)
        if row_count is None
        else _positive_integer(row_count, field_name="row_count")
    )
    normalized_seed = _non_negative_integer(seed, field_name="seed")
    selected_accountant = accountant or DPGenerationBudgetAccountant.from_config()
    if not isinstance(selected_accountant, DPGenerationBudgetAccountant):
        raise TypeError("accountant must be a DPGenerationBudgetAccountant")
    decision = selected_accountant.guard_generation(
        epsilon,
        delta,
        scope,
        family=engine.family,
        mechanism=f"{engine.family}-aggregate",
    )
    response = bridge.fit_synthesize(
        schema,
        statistics,
        epsilon=epsilon,
        delta=delta,
        row_count=requested_rows,
        seed=normalized_seed,
    )
    synthetic_rows = list(response.rows)
    utility = synthetic_tabular_utility_report(
        heldout_rows,
        synthetic_rows,
        marginal_mae_cap=marginal_mae_cap,
        correlation_mae_cap=correlation_mae_cap,
    )
    membership = membership_inference_risk_report(
        training_rows,
        heldout_rows,
        synthetic_rows,
        advantage_ceiling=membership_advantage_ceiling,
    )
    gates_passed = utility.passed and membership.passed
    privacy_report = _privacy_report(
        source_row_count=len(source_rows),
        training_row_count=len(training_rows),
        evaluation_source=evaluation_source,
        engine=response.engine.to_dict(),
        epsilon_spent=response.epsilon_spent,
        delta_spent=response.delta_spent,
        scope=scope,
        budget_decision=decision.to_dict(),
        budget_composition=selected_accountant.compose(scope).to_dict(),
        utility=utility.to_dict(),
        membership=membership.to_dict(),
        output_path=selected_output,
        output_row_count=len(synthetic_rows),
        gates_passed=gates_passed,
    )
    if enforce_gates and not gates_passed:
        raise SyntheticDataGateError(privacy_report)

    write_table(selected_output, synthetic_rows, overwrite=overwrite)
    privacy_report["output"]["sha256"] = _file_sha256(selected_output)
    _write_json_atomic(selected_report, privacy_report, overwrite=overwrite)
    return SyntheticGenerationResult(
        output_path=selected_output,
        report_path=selected_report,
        row_count=len(synthetic_rows),
        privacy_report=privacy_report,
    )


def _output_path(source: Path, output: str | Path | None) -> Path:
    if output is not None:
        selected = Path(output)
    else:
        suffix = ".parquet" if source.suffix.casefold() == ".parquet" else ".csv"
        selected = source.with_name(f"{source.stem}.synthetic{suffix}")
    if selected.suffix.casefold() not in SUPPORTED_SYNTHETIC_OUTPUT_SUFFIXES:
        raise ValueError("synthetic output_path must end in .csv or .parquet")
    return selected


def _report_path(output: Path, report: str | Path | None) -> Path:
    selected = (
        Path(report)
        if report is not None
        else output.with_name(f"{output.stem}.privacy.json")
    )
    if selected.suffix.casefold() != ".json":
        raise ValueError("privacy report_path must end in .json")
    return selected


def _validate_artifact_paths(
    source_path: Path,
    *,
    heldout_path: Path | None,
    output_path: Path,
    report_path: Path,
    overwrite: bool,
) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(f"Source table does not exist: {source_path}")
    if heldout_path is not None and not heldout_path.is_file():
        raise FileNotFoundError(f"Held-out table does not exist: {heldout_path}")
    resolved_source = source_path.resolve()
    resolved_heldout = heldout_path.resolve() if heldout_path is not None else None
    resolved_output = output_path.resolve()
    resolved_report = report_path.resolve()
    protected = {resolved_source, resolved_heldout}
    if resolved_output in protected or resolved_report in protected:
        raise ValueError("synthetic artifacts must not replace source or held-out data")
    if resolved_output == resolved_report:
        raise ValueError("output_path and report_path must be different")
    for artifact in (output_path, report_path):
        if not artifact.parent.is_dir():
            raise FileNotFoundError(
                f"Artifact directory does not exist: {artifact.parent}"
            )
        if (artifact.exists() or artifact.is_symlink()) and not overwrite:
            raise FileExistsError(f"Artifact already exists: {artifact}")


def _deterministic_holdout(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    holdout_count = max(1, math.ceil(len(rows) * DEFAULT_EVALUATION_HOLDOUT_FRACTION))
    heldout_indexes = set(range(len(rows) - holdout_count, len(rows)))
    training = [
        dict(row) for index, row in enumerate(rows) if index not in heldout_indexes
    ]
    heldout = [dict(row) for index, row in enumerate(rows) if index in heldout_indexes]
    return training, heldout


def _require_same_schema(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    if not candidate:
        raise ValueError(f"{label} table must contain at least one row")
    expected = set(reference[0])
    if any(set(row) != expected for row in candidate):
        raise ValueError(f"every {label} row must match the source schema")


def _table_schema(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("training table must contain at least one row")
    names = tuple(rows[0])
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("source table requires non-empty string column names")
    expected = set(names)
    if any(set(row) != expected for row in rows):
        raise ValueError("every source row must contain the same columns")
    schema: list[dict[str, Any]] = []
    for name in names:
        values = [row[name] for row in rows]
        kind = _column_kind(values)
        for value in values:
            _json_scalar(value, kind=kind)
        schema.append(
            {
                "name": name,
                "kind": kind,
                "nullable": any(value is None for value in values),
            }
        )
    return schema


def _column_kind(values: Sequence[Any]) -> str:
    present = [value for value in values if value is not None]
    if present and all(isinstance(value, bool) for value in present):
        return "boolean"
    if present and all(
        isinstance(value, Integral) and not isinstance(value, bool) for value in present
    ):
        return "integer"
    if present and all(
        isinstance(value, Real) and not isinstance(value, bool) for value in present
    ):
        if any(not math.isfinite(float(value)) for value in present):
            raise ValueError("source numeric columns must contain finite values")
        return "number"
    if all(isinstance(value, str) for value in present):
        return "string"
    raise TypeError(
        "DP synthesis supports only null, boolean, finite numeric, and string cells"
    )


def _json_scalar(value: Any, *, kind: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"source column of kind {kind!r} contains a non-JSON scalar")


def _aggregate_statistics(
    rows: Sequence[Mapping[str, Any]],
    schema: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    names = tuple(str(column["name"]) for column in schema)
    columns: list[dict[str, Any]] = []
    for column in schema:
        name = str(column["name"])
        values = [_json_scalar(row[name], kind=str(column["kind"])) for row in rows]
        counts = Counter(_typed_value(value) for value in values)
        distribution = [
            {
                "value": _untyped_value(key),
                "count": count,
            }
            for key, count in sorted(counts.items(), key=lambda item: item[0])
        ]
        entry: dict[str, Any] = {
            "name": name,
            "kind": column["kind"],
            "missing_count": sum(value is None for value in values),
            "non_null_count": sum(value is not None for value in values),
            "distinct_count": len(counts),
            "distribution": distribution,
        }
        numeric = [float(value) for value in values if _is_numeric(value)]
        if numeric:
            entry["numeric_summary"] = {
                "minimum": min(numeric),
                "maximum": max(numeric),
                "mean": sum(numeric) / len(numeric),
            }
        columns.append(entry)

    pairs: list[dict[str, Any]] = []
    correlations: list[dict[str, Any]] = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            counts = Counter(
                (_typed_value(row[left]), _typed_value(row[right])) for row in rows
            )
            pairs.append(
                {
                    "columns": [left, right],
                    "cell_counts": [
                        {
                            "values": [
                                _untyped_value(values[0]),
                                _untyped_value(values[1]),
                            ],
                            "count": count,
                        }
                        for values, count in sorted(
                            counts.items(), key=lambda item: item[0]
                        )
                    ],
                }
            )
            correlation = _numeric_correlation(rows, left, right)
            if correlation is not None:
                correlations.append({"columns": [left, right], "pearson": correlation})
    return {
        "schema_version": 1,
        "source_row_count": len(rows),
        "columns": columns,
        "pairs": pairs,
        "correlations": correlations,
    }


def _typed_value(value: Any) -> tuple[str, str]:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (type(value).__name__, encoded)


def _untyped_value(value: tuple[str, str]) -> Any:
    return json.loads(value[1])


def _is_numeric(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _numeric_correlation(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
) -> float | None:
    pairs = [
        (float(row[left]), float(row[right]))
        for row in rows
        if _is_numeric(row[left]) and _is_numeric(row[right])
    ]
    if len(pairs) < 2:
        return None
    left_mean = sum(pair[0] for pair in pairs) / len(pairs)
    right_mean = sum(pair[1] for pair in pairs) / len(pairs)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in pairs
    )
    left_variance = sum((pair[0] - left_mean) ** 2 for pair in pairs)
    right_variance = sum((pair[1] - right_mean) ** 2 for pair in pairs)
    denominator = math.sqrt(left_variance * right_variance)
    return 0.0 if denominator == 0.0 else float(numerator / denominator)


def _privacy_report(
    *,
    source_row_count: int,
    training_row_count: int,
    evaluation_source: str,
    engine: Mapping[str, Any],
    epsilon_spent: float,
    delta_spent: float,
    scope: str,
    budget_decision: Mapping[str, Any],
    budget_composition: Mapping[str, Any],
    utility: Mapping[str, Any],
    membership: Mapping[str, Any],
    output_path: Path,
    output_row_count: int,
    gates_passed: bool,
) -> dict[str, Any]:
    per_column = {
        name: {
            "marginal_mae": error,
            "fidelity": max(0.0, 1.0 - error),
        }
        for name, error in utility["per_column_mae"].items()
    }
    return {
        "schema_version": SYNTHETIC_PRIVACY_REPORT_SCHEMA_VERSION,
        "provenance": "synthetic-offline",
        "engine": dict(engine),
        "privacy": {
            "epsilon_spent": epsilon_spent,
            "delta_spent": delta_spent,
            "scope": scope,
            "budget_decision": dict(budget_decision),
            "cumulative_composition": dict(budget_composition),
        },
        "evaluation": {
            "reference": evaluation_source,
            "source_row_count": source_row_count,
            "training_row_count": training_row_count,
            "heldout_row_count": membership["nonmember_count"],
        },
        "per_column_fidelity": per_column,
        "utility": dict(utility),
        "membership_inference": dict(membership),
        "gates": {
            "passed": gates_passed,
            "utility_passed": utility["passed"],
            "membership_inference_passed": membership["passed"],
        },
        "output": {
            "format": output_path.suffix.casefold().lstrip("."),
            "row_count": output_row_count,
            "sha256": None,
        },
    }


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    if (path.exists() or path.is_symlink()) and not overwrite:
        raise FileExistsError(f"Artifact already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value: Any, *, field_name: str) -> int:
    integer = _non_negative_integer(value, field_name=field_name)
    if integer <= 0:
        raise ValueError(f"{field_name} must be positive")
    return integer


def _non_negative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return integer


__all__ = [
    "DEFAULT_DP_SYNTH_SCOPE",
    "DEFAULT_EVALUATION_HOLDOUT_FRACTION",
    "SUPPORTED_SYNTHETIC_OUTPUT_SUFFIXES",
    "SYNTHETIC_PRIVACY_REPORT_SCHEMA_VERSION",
    "SyntheticDataGateError",
    "SyntheticGenerationResult",
    "generate_synthetic",
]
