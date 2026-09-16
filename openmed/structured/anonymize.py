"""Role-directed tabular anonymization orchestration.

This module composes the independently shipped structured privacy primitives:
column-role scanning, free-text de-identification, direct-identifier removal,
pure-Python k/l/t enforcement, optional ARX execution, local differential
privacy for explicitly bounded numeric attributes, release writing, and the
structured re-identification report.
"""

from __future__ import annotations

import math
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from openmed.core.date_shift import DEFAULT_DATE_SHIFT_MAX_DAYS
from openmed.core.labels import AGE, DATE, DATE_OF_BIRTH, ZIPCODE
from openmed.interop.bridges.arx import ArxBridge, ArxNotAvailableError
from openmed.risk.kanon import kanon_report

from .generalize import (
    DEFAULT_TARGET_K,
    DEFAULT_TARGET_L,
    DEFAULT_TARGET_T,
    AnonymizationError,
)
from .generalize import (
    AnonymizationResult as GeneralizationResult,
)
from .generalize import _output_hash as _table_output_hash
from .generalize import anonymize_table as _generalize_table
from .hierarchies import COLUMN_TYPE_AGE, COLUMN_TYPE_DATE, COLUMN_TYPE_ZIP
from .reid_report import reid_report as _reid_report
from .scan import ColumnRole, TableRoleScan, scan_table
from .table_io import read_table, write_table

MODEL_K_ANON: Final = "k-anon"
MODEL_L_DIVERSITY: Final = "l-diversity"
MODEL_T_CLOSENESS: Final = "t-closeness"
MODEL_DP: Final = "dp"
SUPPORTED_TABLE_MODELS: Final = frozenset(
    {MODEL_DP, MODEL_K_ANON, MODEL_L_DIVERSITY, MODEL_T_CLOSENESS}
)

ENGINE_AUTO: Final = "auto"
ENGINE_PYTHON: Final = "python"
ENGINE_ARX: Final = "arx"
SUPPORTED_ENGINES: Final = frozenset({ENGINE_ARX, ENGINE_AUTO, ENGINE_PYTHON})


@dataclass(frozen=True)
class TableAnonymizationResult(GeneralizationResult):
    """Transformed records plus scan, manifest, and residual-risk evidence.

    Attributes:
        records: Released row mappings.
        manifest: Raw-value-free transformation and routing evidence.
        risk_report: Structured prosecutor/journalist/marketer risk evidence.
        role_scan: Reviewed column-role classification used for routing.
        output_path: Written release path, or ``None`` for an in-memory result.
    """

    risk_report: dict[str, Any] = field(default_factory=dict)
    role_scan: TableRoleScan | None = None
    output_path: Path | None = None


def anonymize_table(
    table: Any,
    quasi_identifiers: Mapping[str, str] | None = None,
    *,
    model: str = MODEL_K_ANON,
    k: int | None = None,
    l: int | None = None,
    t: float | None = None,
    target_k: int = DEFAULT_TARGET_K,
    target_l: int = DEFAULT_TARGET_L,
    target_t: float = DEFAULT_TARGET_T,
    sensitive_attributes: Sequence[str] | None = None,
    text_columns: Sequence[str] | None = None,
    role_overrides: Mapping[str, str | ColumnRole] | None = None,
    profile_backend: str = "auto",
    text_deidentifier: Callable[[str], Any] | None = None,
    engine: str = ENGINE_AUTO,
    arx_bridge: ArxBridge | None = None,
    suppression_limit: int | None = None,
    suppression_rate: float = 0.0,
    clinical_code_hierarchies: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    subject_id_column: str | None = None,
    date_shift_secret: str | bytes | None = None,
    date_shift_max_days: int = DEFAULT_DATE_SHIFT_MAX_DAYS,
    seed: int | str | bytes | None = None,
    l_metric: str = "distinct",
    dp_bounds: Mapping[str, tuple[float, float]] | None = None,
    epsilon: float = 1.0,
    delta: float = 0.0,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> TableAnonymizationResult:
    """Anonymize a local table and return residual re-identification risk.

    ``quasi_identifiers`` remains compatible with the lower-level generalizer's
    explicit ``{column: hierarchy_type}`` contract. When omitted, age, ZIP, and
    date QIs are inferred from the reviewed role scan; an unsupported QI fails
    closed and asks the caller for an explicit mapping.

    Direct-identifier columns are removed. Explicit ``text_columns`` and
    high-confidence native free-text columns are routed cell-by-cell through
    :func:`openmed.deidentify`. The pure-Python engine is the default and the
    fallback whenever ``engine="auto"`` has no available ARX bridge.

    ``model="dp"`` first performs the requested k/l/t transform, then applies
    epsilon-local differential privacy to numeric attributes with public bounds
    supplied through ``dp_bounds``. Laplace noise provides an ``(epsilon, 0)``
    guarantee, which is at least as strong as the caller's reported nonnegative
    ``delta``. Bounds are mandatory because deriving them from private data
    would invalidate the sensitivity claim.

    Args:
        table: CSV/TSV, JSONL/NDJSON, or Parquet path; row mappings; or a
            DataFrame-like object.
        quasi_identifiers: Optional reviewed column-to-hierarchy mapping.
        model: ``k-anon``, ``l-diversity``, ``t-closeness``, or ``dp``.
        k: Compact alias for ``target_k``.
        l: Compact alias for ``target_l``.
        t: Compact alias for ``target_t``.
        target_k: Minimum equivalence-class size.
        target_l: Minimum sensitive-value diversity.
        target_t: Maximum variational t-closeness distance.
        sensitive_attributes: Reviewed sensitive columns for l/t enforcement.
        text_columns: Reviewed free-text columns to de-identify.
        role_overrides: Reviewed role pins passed to :func:`scan_table`.
        profile_backend: Native or optional DataProfiler scan backend.
        text_deidentifier: Optional compatible callable, primarily for custom
            local model routing and deterministic synthetic tests.
        engine: ``auto``, ``python``, or ``arx``.
        arx_bridge: Configured user-supplied ARX adapter.
        suppression_limit: Absolute maximum number of rows to suppress.
        suppression_rate: Fractional suppression cap in ``[0, 1]``.
        clinical_code_hierarchies: Caller-supplied clinical-code parent chains.
        subject_id_column: Transient subject key for consistent date shifting.
        date_shift_secret: Secret key material for subject-level date shifts.
        date_shift_max_days: Maximum absolute date-shift distance.
        seed: Reproducibility key accepted by the date generalizer.
        l_metric: ``distinct`` or ``entropy`` l-diversity measurement.
        dp_bounds: Public numeric ``{column: (lower, upper)}`` bounds.
        epsilon: Total per-record local-DP budget, split across DP columns.
        delta: Reported approximate-DP delta; the implemented Laplace path has
            the stronger delta-zero guarantee.
        output_path: Optional local destination for the transformed table.
        overwrite: Whether an existing output may be replaced atomically.

    Returns:
        A :class:`TableAnonymizationResult` containing transformed records, a
        raw-value-free manifest, a section-5.10 risk report, the role scan, and
        the optional output path.

    Raises:
        AnonymizationError: If configuration is invalid or a requested privacy
            bound cannot be verified on the transformed release.
        ArxNotAvailableError: If ``engine="arx"`` has no usable adapter.
    """

    requested_model = _validated_model(model)
    selected_engine = _validated_engine(engine)
    resolved_k = _resolve_alias("k", target_k, k, DEFAULT_TARGET_K)
    resolved_l = _resolve_alias("l", target_l, l, DEFAULT_TARGET_L)
    resolved_t = _resolve_alias("t", target_t, t, DEFAULT_TARGET_T)
    _validate_privacy_targets(resolved_k, resolved_l, resolved_t)
    _validate_model_targets(requested_model, resolved_l, resolved_t)
    if selected_engine == ENGINE_ARX and requested_model == MODEL_DP:
        raise AnonymizationError("the ARX bridge does not implement the dp model")
    use_arx = selected_engine == ENGINE_ARX or (
        selected_engine == ENGINE_AUTO
        and arx_bridge is not None
        and arx_bridge.available
        and requested_model != MODEL_DP
    )

    source_records = _load_records(table)
    role_scan = scan_table(
        source_records,
        overrides=role_overrides,
        profile_backend=profile_backend,
    )
    column_types = (
        dict(quasi_identifiers)
        if quasi_identifiers is not None
        else _infer_quasi_identifier_types(role_scan)
    )
    if not column_types:
        raise AnonymizationError(
            "no supported quasi-identifiers were found; provide an explicit "
            "quasi_identifiers mapping"
        )
    _validate_known_columns(source_records, column_types, label="quasi-identifier")

    routed_text_columns = _resolve_text_columns(
        role_scan,
        text_columns,
        source_records,
    )
    routed_records = _deidentify_text_columns(
        source_records,
        routed_text_columns,
        text_deidentifier=text_deidentifier,
    )
    removable_direct_ids = tuple(
        column
        for column in role_scan.direct_identifiers
        if column not in column_types
        and not (
            column == subject_id_column
            and not use_arx
            and COLUMN_TYPE_DATE in column_types.values()
        )
    )
    routed_records = _remove_columns(routed_records, removable_direct_ids)

    sensitive = tuple(dict.fromkeys(sensitive_attributes or ()))
    _validate_known_columns(routed_records, sensitive, label="sensitive-attribute")
    if (resolved_l > 1 or resolved_t < 1.0) and not sensitive:
        raise AnonymizationError(
            "l-diversity or t-closeness requires sensitive_attributes"
        )

    if use_arx:
        if arx_bridge is None:
            raise ArxNotAvailableError(
                "ARX is not available; configure an adapter or use engine='python'"
            )
        transformed = arx_bridge.anonymize(
            routed_records,
            quasi_identifiers=column_types,
            sensitive_attributes=sensitive,
            k=resolved_k,
            l=resolved_l,
            t=resolved_t,
        )
        released = transformed.records
        manifest = dict(transformed.manifest)
        effective_engine = ENGINE_ARX
    else:
        if selected_engine == ENGINE_ARX:
            raise ArxNotAvailableError(
                "ARX is not available; configure an adapter or use engine='python'"
            )
        generalized = _generalize_table(
            routed_records,
            column_types,
            target_k=resolved_k,
            target_l=resolved_l,
            target_t=resolved_t,
            suppression_limit=suppression_limit,
            suppression_rate=suppression_rate,
            sensitive_attributes=sensitive,
            model=MODEL_K_ANON,
            clinical_code_hierarchies=clinical_code_hierarchies,
            subject_id_column=subject_id_column,
            date_shift_secret=date_shift_secret,
            date_shift_max_days=date_shift_max_days,
            seed=seed,
            l_metric=l_metric,
        )
        released = generalized.records
        manifest = dict(generalized.manifest)
        effective_engine = ENGINE_PYTHON

    dp_manifest: dict[str, Any] | None = None
    if requested_model == MODEL_DP:
        released, dp_manifest = _apply_local_dp(
            released,
            bounds=dp_bounds,
            quasi_identifiers=column_types,
            epsilon=epsilon,
            delta=delta,
        )

    _validate_privacy_release(
        released,
        quasi_identifiers=column_types,
        sensitive_attributes=sensitive,
        k=resolved_k,
        l=resolved_l,
        t=resolved_t,
        l_metric=l_metric,
    )
    risk = _reid_report(released, quasi_identifiers=list(column_types))
    manifest.update(
        {
            "model": requested_model,
            "engine": effective_engine,
            "output_hash": _table_output_hash(released),
            "column_routing": {
                "direct_identifiers_removed": list(removable_direct_ids),
                "free_text_deidentified": list(routed_text_columns),
                "structured_quasi_identifiers": dict(column_types),
            },
            "risk_report_schema_version": risk["schema_version"],
        }
    )
    if dp_manifest is not None:
        manifest["differential_privacy"] = dp_manifest

    resolved_output: Path | None = None
    if output_path is not None:
        resolved_output = write_table(output_path, released, overwrite=overwrite)
    return TableAnonymizationResult(
        records=tuple(dict(row) for row in released),
        manifest=manifest,
        risk_report=risk,
        role_scan=role_scan,
        output_path=resolved_output,
    )


def _validated_model(model: str) -> str:
    if model not in SUPPORTED_TABLE_MODELS:
        supported = ", ".join(sorted(SUPPORTED_TABLE_MODELS))
        raise AnonymizationError(f"unknown model {model!r}; supported: {supported}")
    return model


def _validated_engine(engine: str) -> str:
    if engine not in SUPPORTED_ENGINES:
        supported = ", ".join(sorted(SUPPORTED_ENGINES))
        raise AnonymizationError(f"unknown engine {engine!r}; supported: {supported}")
    return engine


def _resolve_alias(
    name: str,
    target: int | float,
    alias: int | float | None,
    default: int | float,
) -> Any:
    if alias is None:
        return target
    if target != default and target != alias:
        raise AnonymizationError(
            f"conflicting {name} targets were supplied via target_{name} and {name}"
        )
    return alias


def _validate_model_targets(model: str, l: int, t: float) -> None:
    if model == MODEL_L_DIVERSITY and l < 2:
        raise AnonymizationError("l-diversity requires l >= 2")
    if model == MODEL_T_CLOSENESS and t >= 1.0:
        raise AnonymizationError("t-closeness requires t < 1")


def _validate_privacy_targets(k: Any, l: Any, t: Any) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise AnonymizationError("k must be an integer >= 1")
    if isinstance(l, bool) or not isinstance(l, int) or l < 1:
        raise AnonymizationError("l must be an integer >= 1")
    if (
        isinstance(t, bool)
        or not isinstance(t, (int, float))
        or not math.isfinite(float(t))
        or not 0.0 <= float(t) <= 1.0
    ):
        raise AnonymizationError("t must be between 0 and 1")


def _validate_privacy_release(
    records: Sequence[Mapping[str, Any]],
    *,
    quasi_identifiers: Mapping[str, str],
    sensitive_attributes: Sequence[str],
    k: int,
    l: int,
    t: float,
    l_metric: str,
) -> None:
    """Independently fail closed if an engine misses a requested bound."""

    try:
        report = kanon_report(
            records,
            quasi_identifiers=list(quasi_identifiers),
            sensitive_attributes=list(sensitive_attributes),
            l_metric=l_metric,
        )
    except (TypeError, ValueError) as exc:
        raise AnonymizationError(
            "the transformed release could not be validated"
        ) from exc
    if int(report["k"]) < k:
        raise AnonymizationError("the transformed release did not reach target k")
    required_l = math.log2(l) if l_metric == "entropy" else float(l)
    l_diversity = report.get("l", {})
    if sensitive_attributes and any(
        float(l_diversity.get(column, -math.inf)) + 1e-12 < required_l
        for column in sensitive_attributes
    ):
        raise AnonymizationError("the transformed release did not reach target l")
    t_closeness = report.get("t_closeness", {})
    if sensitive_attributes and any(
        float(t_closeness.get(column, math.inf)) > float(t) + 1e-12
        for column in sensitive_attributes
    ):
        raise AnonymizationError("the transformed release did not reach target t")


def _load_records(table: Any) -> list[dict[str, Any]]:
    if isinstance(table, (str, Path)):
        records = read_table(table)
    else:
        to_dicts = getattr(table, "to_dicts", None)
        if callable(to_dicts):
            records = to_dicts()
        else:
            to_dict = getattr(table, "to_dict", None)
            if callable(to_dict) and not isinstance(table, Mapping):
                records = to_dict("records")
            elif isinstance(table, Sequence) and not isinstance(
                table, (str, bytes, bytearray)
            ):
                records = table
            else:
                raise AnonymizationError(
                    "table must be a local path, row mappings, or a DataFrame-like object"
                )
    if not records or not all(isinstance(row, Mapping) for row in records):
        raise AnonymizationError("the input table must contain row mappings")
    return [dict(row) for row in records]


def _infer_quasi_identifier_types(scan: TableRoleScan) -> dict[str, str]:
    inferred: dict[str, str] = {}
    unsupported: list[str] = []
    for classification in scan.classifications:
        if classification.role is not ColumnRole.QUASI_ID:
            continue
        if classification.canonical_label == AGE:
            inferred[classification.column] = COLUMN_TYPE_AGE
        elif classification.canonical_label == ZIPCODE:
            inferred[classification.column] = COLUMN_TYPE_ZIP
        elif classification.canonical_label in {DATE, DATE_OF_BIRTH} or any(
            signal.startswith("date_like_ratio=") for signal in classification.signals
        ):
            inferred[classification.column] = COLUMN_TYPE_DATE
        else:
            unsupported.append(classification.column)
    if unsupported:
        raise AnonymizationError(
            "reviewed hierarchy types are required for quasi-identifier columns: "
            + ", ".join(unsupported)
        )
    return inferred


def _resolve_text_columns(
    scan: TableRoleScan,
    requested: Sequence[str] | None,
    records: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    if requested is not None:
        explicit = tuple(dict.fromkeys(requested))
        _validate_known_columns(records, explicit, label="text")
        return explicit
    detected = tuple(
        item.column
        for item in scan.classifications
        if item.role is ColumnRole.SENSITIVE and "free_text_shape=true" in item.signals
    )
    return detected


def _deidentify_text_columns(
    records: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    text_deidentifier: Callable[[str], Any] | None,
) -> list[dict[str, Any]]:
    if not columns:
        return [dict(row) for row in records]
    deidentifier = text_deidentifier or _default_text_deidentifier
    transformed: list[dict[str, Any]] = []
    for row_index, row in enumerate(records):
        rewritten = dict(row)
        for column in columns:
            value = row[column]
            if value is None:
                continue
            if not isinstance(value, str):
                raise AnonymizationError(
                    f"text column {column!r} has a non-string value at row {row_index}"
                )
            try:
                result = deidentifier(value)
            except Exception:
                raise AnonymizationError(
                    f"free-text de-identification failed for column {column!r} "
                    f"at row {row_index}"
                ) from None
            if isinstance(result, str):
                rewritten[column] = result
            else:
                deidentified_text = getattr(result, "deidentified_text", None)
                if not isinstance(deidentified_text, str):
                    raise AnonymizationError(
                        "text_deidentifier must return a string or a "
                        "DeidentificationResult-compatible object"
                    )
                rewritten[column] = deidentified_text
        transformed.append(rewritten)
    return transformed


def _default_text_deidentifier(text: str) -> Any:
    from openmed.core.pii import deidentify

    return deidentify(text)


def _remove_columns(
    records: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    blocked = set(columns)
    return [
        {column: value for column, value in row.items() if column not in blocked}
        for row in records
    ]


def _validate_known_columns(
    records: Sequence[Mapping[str, Any]],
    columns: Mapping[str, Any] | Sequence[str],
    *,
    label: str,
) -> None:
    requested = tuple(columns)
    for column in requested:
        if not isinstance(column, str) or not column:
            raise AnonymizationError(f"{label} column names must be non-empty strings")
        if any(column not in row for row in records):
            raise AnonymizationError(
                f"{label} column {column!r} is absent from the table"
            )


def _apply_local_dp(
    records: Sequence[Mapping[str, Any]],
    *,
    bounds: Mapping[str, tuple[float, float]] | None,
    quasi_identifiers: Mapping[str, str],
    epsilon: float,
    delta: float,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    if not isinstance(bounds, Mapping) or not bounds:
        raise AnonymizationError("model='dp' requires public dp_bounds")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise AnonymizationError("epsilon must be a positive finite number")
    if not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise AnonymizationError("epsilon must be a positive finite number")
    if isinstance(delta, bool) or not isinstance(delta, (int, float)):
        raise AnonymizationError("delta must be a finite number in [0, 1)")
    if not math.isfinite(float(delta)) or not 0 <= delta < 1:
        raise AnonymizationError("delta must be a finite number in [0, 1)")
    overlap = sorted(set(bounds) & set(quasi_identifiers))
    if overlap:
        raise AnonymizationError(
            "dp_bounds cannot include quasi-identifiers because noise would "
            "invalidate equivalence classes: " + ", ".join(overlap)
        )
    _validate_known_columns(records, bounds, label="DP")
    per_column_epsilon = float(epsilon) / len(bounds)
    validated_bounds: dict[str, tuple[float, float]] = {}
    for column, interval in bounds.items():
        if (
            not isinstance(interval, Sequence)
            or len(interval) != 2
            or any(isinstance(value, bool) for value in interval)
        ):
            raise AnonymizationError("each DP bound must be a (lower, upper) pair")
        try:
            lower, upper = float(interval[0]), float(interval[1])
        except (TypeError, ValueError):
            raise AnonymizationError(
                "each DP bound must contain numeric values"
            ) from None
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise AnonymizationError("each DP bound must satisfy finite lower < upper")
        validated_bounds[column] = (lower, upper)

    transformed: list[dict[str, Any]] = []
    for row_index, row in enumerate(records):
        rewritten = dict(row)
        for column, (lower, upper) in validated_bounds.items():
            value = row[column]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AnonymizationError(
                    f"DP column {column!r} has a non-numeric value at row {row_index}"
                )
            number = float(value)
            if not math.isfinite(number):
                raise AnonymizationError(
                    f"DP column {column!r} has a non-finite value at row {row_index}"
                )
            clipped = min(upper, max(lower, number))
            scale = (upper - lower) / per_column_epsilon
            rewritten[column] = clipped + _laplace_noise(scale)
        transformed.append(rewritten)

    manifest = {
        "mechanism": "local_laplace",
        "epsilon": float(epsilon),
        "delta": float(delta),
        "guaranteed_delta": 0.0,
        "composition": "sequential_per_record",
        "per_column_epsilon": per_column_epsilon,
        "columns": [
            {"column": column, "lower": lower, "upper": upper}
            for column, (lower, upper) in validated_bounds.items()
        ],
    }
    return tuple(transformed), manifest


def _laplace_noise(scale: float) -> float:
    uniform = secrets.SystemRandom().random() - 0.5
    # SystemRandom.random() is in [0, 1); avoid log(0) for the exact lower edge.
    uniform = max(-0.5 + 2**-53, uniform)
    return -scale * math.copysign(math.log1p(-2.0 * abs(uniform)), uniform)


__all__ = [
    "ENGINE_ARX",
    "ENGINE_AUTO",
    "ENGINE_PYTHON",
    "MODEL_DP",
    "MODEL_K_ANON",
    "MODEL_L_DIVERSITY",
    "MODEL_T_CLOSENESS",
    "SUPPORTED_ENGINES",
    "SUPPORTED_TABLE_MODELS",
    "TableAnonymizationResult",
    "anonymize_table",
]
