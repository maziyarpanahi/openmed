"""Reach target k/l/t privacy bounds through generalization and suppression.

The monotone generalization-lattice search with bounded suppression already
lives in :func:`openmed.risk.kanon.enforce_kanon`. This module is the thin glue
that lets the declarative per-column generalization family in
:mod:`openmed.structured.hierarchies` drive that engine: given a table and a
quasi-identifier ``column -> column type`` mapping, it materializes the family
into ``enforce_kanon``-compatible generalization hierarchies via
:func:`openmed.structured.hierarchies.build_enforcement_hierarchies`, runs the
enforcement search, and shapes the engine's report into the
:class:`AnonymizationResult` the ``anonymize_table`` entrypoint promises.

No search is reimplemented here. ``enforce_kanon`` owns the lattice search, the
suppression selection, and the utility scoring; this module contributes the
column-type-to-hierarchy binding, table coercion, patient-consistent date shift,
and a raw-value-free manifest. Quasi-identifiers are supplied explicitly as a
``column -> type`` mapping; detection is a separate concern and is never inferred
here. The path is pure Python: no JVM, no bundled terminology, and no network
access.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from openmed.core.audit import stable_hash
from openmed.core.date_shift import DEFAULT_DATE_SHIFT_MAX_DAYS
from openmed.risk.kanon import enforce_kanon
from openmed.structured.hierarchies import (
    COLUMN_TYPE_DATE,
    HIERARCHY_SCHEMA_VERSION,
    SUPPORTED_COLUMN_TYPES,
    HierarchyError,
    build_enforcement_hierarchies,
    generalize_value,
)
from openmed.structured.table_io import read_table

MANIFEST_SCHEMA_VERSION: Final = "1.1.0"

#: The anonymization models this entrypoint understands.
MODEL_K_ANON: Final = "k-anon"
SUPPORTED_MODELS: Final = frozenset({MODEL_K_ANON})

#: Default target group size when a caller does not specify ``target_k``.
DEFAULT_TARGET_K: Final = 2
DEFAULT_TARGET_L: Final = 1
DEFAULT_TARGET_T: Final = 1.0

#: Regression cap for the synthetic reference fixture in this module's tests.
REFERENCE_AVERAGE_GENERALIZATION_HEIGHT_CAP: Final = 0.5
#: Regression cap for row suppression on the synthetic reference fixture.
REFERENCE_SUPPRESSION_RATE_CAP: Final = 0.1


class AnonymizationError(ValueError):
    """Raised for an invalid anonymization request (unknown model or column
    type, missing quasi-identifier column, empty inputs) or when the underlying
    enforcement engine cannot reach the target within the suppression bound."""


@dataclass(frozen=True)
class AnonymizationResult:
    """The released table plus a raw-value-free transformation manifest.

    ``records`` is the generalized, suppression-filtered table returned by the
    enforcement engine (quasi-identifier columns replaced by their generalized
    labels, other columns preserved, rows in classes smaller than the target k
    removed). ``manifest`` records per-column levels and suppression counts,
    aggregate utility metrics, date-shift provenance, and an output hash -- never
    a raw source value -- so it is safe to persist alongside a release.
    """

    records: tuple[dict[str, Any], ...]
    manifest: dict[str, Any] = field(default_factory=dict)


def anonymize_table(
    table: Any,
    quasi_identifiers: Mapping[str, str],
    *,
    target_k: int = DEFAULT_TARGET_K,
    target_l: int = DEFAULT_TARGET_L,
    target_t: float = DEFAULT_TARGET_T,
    suppression_limit: int | None = None,
    suppression_rate: float = 0.0,
    sensitive_attributes: Sequence[str] | None = None,
    model: str = MODEL_K_ANON,
    clinical_code_hierarchies: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    subject_id_column: str | None = None,
    date_shift_secret: str | bytes | None = None,
    date_shift_max_days: int = DEFAULT_DATE_SHIFT_MAX_DAYS,
    seed: int | str | bytes | None = None,
    l_metric: str = "distinct",
    k: int | None = None,
    l: int | None = None,
    t: float | None = None,
) -> AnonymizationResult:
    """Anonymize a table to k-anonymity via the ``enforce_kanon`` search.

    The declarative generalization family is materialized into
    ``enforce_kanon``-compatible hierarchies and the enforcement engine performs
    the monotone lattice search and bounded suppression; this function only
    binds the inputs and reshapes the result.

    Args:
        table: A local table path (any suffix accepted by
            :func:`openmed.structured.read_table`), a sequence of row mappings,
            or a DataFrame-like object exposing ``to_dicts``/``to_dict``.
        quasi_identifiers: Mapping of quasi-identifier column name to its
            generalization column type (one of
            :data:`openmed.structured.SUPPORTED_COLUMN_TYPES`). Column detection
            is out of scope; the mapping is taken as given.
        target_k: Target minimum equivalence-class size (an integer ``>= 1``).
        target_l: Target l-diversity for every declared sensitive attribute.
        target_t: Maximum variational t-closeness distance for every declared
            sensitive attribute.
        suppression_limit: Absolute cap on suppressed rows, or ``None``.
        suppression_rate: Fractional cap on suppressed rows in ``[0, 1]``. When
            both bounds are given the tighter one applies.
        sensitive_attributes: Optional sensitive-attribute columns; forwarded to
            the engine so the report carries their disclosure bounds.
        model: Anonymization model; only ``"k-anon"`` is supported.
        clinical_code_hierarchies: Caller-supplied clinical terminology data,
            mapping each clinical-code QI column to ``leaf -> parent chain``.
            Parent chains are ordered from immediate parent to root. OpenMed
            never bundles ICD, SNOMED CT, or other terminology content.
        subject_id_column: Transient subject-key column used to assign one
            stable date-shift offset to every date QI for the same subject. It
            is removed before enforcement and never released.
        date_shift_secret: HMAC key material for patient-consistent date shifts.
            Required for date QIs unless ``seed`` is supplied.
        date_shift_max_days: Maximum absolute date shift in days.
        seed: Reproducibility key accepted as an alternative to
            ``date_shift_secret``. For production date shifting, use secret
            high-entropy key material rather than a guessable integer.
        l_metric: l-diversity variant forwarded to the enforcement engine.
        k: Alias for ``target_k`` matching the compact public entrypoint.
        l: Alias for ``target_l`` matching the compact public entrypoint.
        t: Alias for ``target_t`` matching the compact public entrypoint.

    Returns:
        An :class:`AnonymizationResult` whose ``records`` reach ``target_k`` with
        zero equivalence classes below it and whose ``manifest`` records the
        chosen per-column generalization level, the achieved k, and the
        suppression count.

    Raises:
        AnonymizationError: For an unknown model, a quasi-identifier with an
            unknown column type or absent from the table, empty inputs, or when
            no assignment reaches ``target_k`` within the suppression bound.
    """
    if model not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise AnonymizationError(f"unknown model {model!r}; supported: {supported}")

    target_k = _resolve_policy_alias("k", target_k, k, DEFAULT_TARGET_K)
    target_l = _resolve_policy_alias("l", target_l, l, DEFAULT_TARGET_L)
    target_t = _resolve_policy_alias("t", target_t, t, DEFAULT_TARGET_T)

    column_types = _validated_quasi_identifiers(quasi_identifiers)
    records = _load_records(table)
    _validate_columns_present(records, column_types)
    records, shifted_date_columns = _prepare_records_for_enforcement(
        records,
        column_types,
        sensitive_attributes=sensitive_attributes,
        subject_id_column=subject_id_column,
        date_shift_secret=date_shift_secret,
        date_shift_max_days=date_shift_max_days,
        seed=seed,
    )

    try:
        hierarchies = build_enforcement_hierarchies(
            column_types,
            records,
            clinical_code_hierarchies=clinical_code_hierarchies,
        )
    except HierarchyError as exc:
        raise AnonymizationError(_safe_hierarchy_error(exc)) from exc

    try:
        report = enforce_kanon(
            records,
            quasi_identifiers=list(column_types),
            sensitive_attributes=(
                list(sensitive_attributes) if sensitive_attributes else None
            ),
            hierarchies=hierarchies,
            target_k=target_k,
            target_l=target_l,
            target_t=target_t,
            suppression_limit=suppression_limit,
            suppression_rate=suppression_rate,
            l_metric=l_metric,
        )
    except (TypeError, ValueError) as exc:
        raise AnonymizationError(str(exc)) from exc

    released = tuple(dict(record) for record in report["records"])
    manifest = _build_manifest(
        column_types,
        report,
        hierarchies=hierarchies,
        records=released,
        shifted_date_columns=shifted_date_columns,
        date_shift_max_days=date_shift_max_days,
        model=model,
    )
    return AnonymizationResult(records=released, manifest=manifest)


# --------------------------------------------------------------------------- #
# Input handling                                                              #
# --------------------------------------------------------------------------- #
def _validated_quasi_identifiers(
    quasi_identifiers: Mapping[str, str],
) -> dict[str, str]:
    """Return a validated ``column -> column type`` mapping (insertion order)."""
    if not isinstance(quasi_identifiers, Mapping):
        raise AnonymizationError(
            "quasi_identifiers must be a mapping of column name to column type"
        )
    if not quasi_identifiers:
        raise AnonymizationError("quasi_identifiers must not be empty")
    column_types: dict[str, str] = {}
    for column, column_type in quasi_identifiers.items():
        if not isinstance(column, str) or not column:
            raise AnonymizationError(
                "quasi_identifiers keys must be non-empty column names"
            )
        if (
            not isinstance(column_type, str)
            or column_type not in SUPPORTED_COLUMN_TYPES
        ):
            supported = ", ".join(sorted(SUPPORTED_COLUMN_TYPES))
            raise AnonymizationError(
                f"unknown column type {column_type!r} for column {column!r}; "
                f"supported: {supported}"
            )
        column_types[column] = column_type
    return column_types


def _load_records(table: Any) -> list[dict[str, Any]]:
    """Coerce a path, row sequence, or DataFrame-like into a list of row dicts."""
    if isinstance(table, (str, Path)):
        return list(read_table(table))

    to_dicts = getattr(table, "to_dicts", None)
    if callable(to_dicts):
        rows = to_dicts()
    else:
        to_dict = getattr(table, "to_dict", None)
        if callable(to_dict) and not isinstance(table, Mapping):
            rows = to_dict("records")
        elif isinstance(table, Sequence) and not isinstance(
            table, (str, bytes, bytearray)
        ):
            rows = table
        else:
            raise AnonymizationError(
                "table must be a table path, a sequence of row mappings, or a "
                "DataFrame-like object"
            )
    if not all(isinstance(row, Mapping) for row in rows):
        raise AnonymizationError("every table row must be a mapping of column to value")
    return [dict(row) for row in rows]


def _validate_columns_present(
    records: Sequence[Mapping[str, Any]],
    column_types: Mapping[str, str],
) -> None:
    if not records:
        raise AnonymizationError("the input table must contain at least one row")
    for column in column_types:
        for index, row in enumerate(records):
            if column not in row:
                raise AnonymizationError(
                    f"quasi-identifier column {column!r} is missing at row {index}"
                )


def _resolve_policy_alias(
    name: str,
    target: int | float,
    alias: int | float | None,
    default: int | float,
) -> Any:
    """Resolve a compact k/l/t alias without accepting conflicting targets."""
    if alias is None:
        return target
    if target != default and target != alias:
        raise AnonymizationError(
            f"conflicting {name} targets were supplied via target_{name} and {name}"
        )
    return alias


def _prepare_records_for_enforcement(
    records: Sequence[Mapping[str, Any]],
    column_types: Mapping[str, str],
    *,
    sensitive_attributes: Sequence[str] | None,
    subject_id_column: str | None,
    date_shift_secret: str | bytes | None,
    date_shift_max_days: int,
    seed: int | str | bytes | None,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Shift date QIs consistently and discard the transient subject key."""
    date_columns = tuple(
        column
        for column, column_type in column_types.items()
        if column_type == COLUMN_TYPE_DATE
    )
    if not date_columns:
        return [dict(record) for record in records], ()

    if not isinstance(subject_id_column, str) or not subject_id_column:
        raise AnonymizationError("date quasi-identifiers require subject_id_column")
    if subject_id_column in column_types:
        raise AnonymizationError(
            "subject_id_column is transient and cannot also be a quasi-identifier"
        )
    if sensitive_attributes and subject_id_column in sensitive_attributes:
        raise AnonymizationError(
            "subject_id_column is transient and cannot be a sensitive attribute"
        )
    if date_shift_secret is not None and seed is not None:
        raise AnonymizationError("supply date_shift_secret or seed, not both")
    secret = date_shift_secret if date_shift_secret is not None else _seed_bytes(seed)
    if secret is None:
        raise AnonymizationError(
            "date quasi-identifiers require date_shift_secret or seed"
        )

    shifted: list[dict[str, Any]] = []
    for row_index, record in enumerate(records):
        if subject_id_column not in record:
            raise AnonymizationError(f"subject_id_column is missing at row {row_index}")
        subject_key = _subject_key(record[subject_id_column], row_index=row_index)
        rewritten = dict(record)
        for column in date_columns:
            try:
                rewritten[column] = generalize_value(
                    COLUMN_TYPE_DATE,
                    record[column],
                    0,
                    patient_key=subject_key,
                    secret=secret,
                    max_days=date_shift_max_days,
                )
            except (HierarchyError, TypeError, ValueError):
                raise AnonymizationError(
                    f"date shift failed for column {column!r} at row {row_index}"
                ) from None
        rewritten.pop(subject_id_column, None)
        shifted.append(rewritten)
    return shifted, date_columns


def _seed_bytes(seed: int | str | bytes | None) -> bytes | None:
    """Return domain-separated bytes for a reproducibility seed."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, (int, str, bytes)):
        raise AnonymizationError("seed must be an integer, string, or bytes")
    if isinstance(seed, int):
        payload = str(seed).encode("ascii")
    elif isinstance(seed, str):
        payload = seed.encode("utf-8")
    else:
        payload = seed
    if not payload:
        raise AnonymizationError("seed must be non-empty")
    return b"openmed.structured.generalize.seed.v1\x00" + payload


def _subject_key(value: Any, *, row_index: int) -> str | bytes:
    """Coerce a scalar subject key without including it in an error message."""
    if isinstance(value, bool):
        raise AnonymizationError(
            f"subject_id_column has an invalid value at row {row_index}"
        )
    if isinstance(value, (str, bytes)):
        result: str | bytes = value
    elif isinstance(value, int):
        result = str(value)
    else:
        raise AnonymizationError(
            f"subject_id_column has an invalid value at row {row_index}"
        )
    if not result:
        raise AnonymizationError(
            f"subject_id_column has an invalid value at row {row_index}"
        )
    return result


def _safe_hierarchy_error(error: HierarchyError) -> str:
    """Return a useful hierarchy error without echoing an input value.

    Leaf generalizers include offending values in some validation messages so
    that their standalone API is diagnosable. This public table boundary may
    handle PHI, so it only preserves fixed, value-independent clinical-policy
    diagnostics and replaces all other hierarchy text with a safe summary.
    """
    message = str(error)
    if "requires clinical code parent-chain data" in message:
        return "clinical code quasi-identifiers require parent-chain data"
    if "clinical code hierarchy is missing" in message:
        return "clinical code hierarchy is missing observed value mappings"
    if "clinical code parent chains" in message:
        return "clinical code parent-chain data is not monotone"
    if "clinical code hierarchy was supplied for non-code column" in message:
        return "clinical code hierarchy was supplied for a non-code column"
    if "undeclared columns" in message:
        return "clinical code hierarchy targets an undeclared column"
    return "one or more quasi-identifier values are invalid for the declared type"


# --------------------------------------------------------------------------- #
# Manifest                                                                    #
# --------------------------------------------------------------------------- #
def _build_manifest(
    column_types: Mapping[str, str],
    report: Mapping[str, Any],
    *,
    hierarchies: Mapping[str, Sequence[Mapping[str, Any]]],
    records: Sequence[Mapping[str, Any]],
    shifted_date_columns: Sequence[str],
    date_shift_max_days: int,
    model: str,
) -> dict[str, Any]:
    """Shape ``enforce_kanon``'s report into a raw-value-free manifest.

    Only level metadata (index, declarative family key, loss) and counts are
    copied out. No raw source value or generalized cell value is recorded.
    """
    generalization = report.get("generalization", {})
    engine_levels = generalization.get("levels", {})

    columns: list[dict[str, Any]] = []
    generalization_levels: dict[str, int] = {}
    normalized_heights: list[float] = []
    suppressed_count = int(report.get("suppressed_count", 0))
    for column, column_type in column_types.items():
        level_info = engine_levels.get(column, {})
        level_index = int(level_info.get("level", 0))
        maximum_level = max(0, len(hierarchies.get(column, ())) - 1)
        normalized_height = level_index / maximum_level if maximum_level else 0.0
        columns.append(
            {
                "column": column,
                "column_type": column_type,
                "level": level_index,
                "level_name": level_info.get("name"),
                "loss": level_info.get("loss"),
                "suppression_count": suppressed_count,
            }
        )
        generalization_levels[column] = level_index
        normalized_heights.append(normalized_height)

    record_count = int(report.get("record_count", 0))
    suppression_rate = suppressed_count / record_count if record_count else 0.0
    average_height = (
        sum(normalized_heights) / len(normalized_heights) if normalized_heights else 0.0
    )

    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "hierarchy_schema_version": HIERARCHY_SCHEMA_VERSION,
        "model": model,
        "target_k": int(report.get("target_k", 0)),
        "target_l": int(report.get("target_l", 0)),
        "target_t": float(report.get("target_t", 0.0)),
        "l_metric": report.get("l_metric"),
        "achieved_k": int(report.get("kanon", {}).get("k", 0)),
        "quasi_identifiers": dict(column_types),
        "columns": columns,
        "generalization_levels": generalization_levels,
        "record_count": record_count,
        "released_count": int(report.get("released_count", 0)),
        "suppressed_count": suppressed_count,
        "suppression_limit": (
            None
            if report.get("suppression_limit") is None
            else int(report["suppression_limit"])
        ),
        "utility": {
            "information_loss": generalization.get("information_loss"),
            "generalization_loss": generalization.get("generalization_loss"),
            "suppression_loss": generalization.get("suppression_loss"),
            "average_generalization_height": average_height,
            "suppression_rate": suppression_rate,
        },
        "date_shift": {
            "applied": bool(shifted_date_columns),
            "columns": list(shifted_date_columns),
            "max_days": date_shift_max_days if shifted_date_columns else None,
            "subject_identifier_removed": bool(shifted_date_columns),
        },
        "output_hash": _output_hash(records),
        "search": {
            "engine": "openmed.risk.kanon.enforce_kanon",
            "strategy": generalization.get("search"),
            "nodes_evaluated": generalization.get("nodes_evaluated"),
            "search_space_size": generalization.get("search_space_size"),
        },
    }


def _output_hash(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the transformed table through a deterministic typed representation."""
    return stable_hash(_canonical_output_value(records))


def _canonical_output_value(value: Any) -> Any:
    """Return JSON-safe typed data without collapsing temporal or byte values."""
    if isinstance(value, Mapping):
        return {str(key): _canonical_output_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_output_value(item) for item in value]
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise AnonymizationError("output contains a non-finite float")
        return {"type": "float", "value": repr(value)}
    if type(value) is Decimal:
        if not value.is_finite():
            raise AnonymizationError("output contains a non-finite decimal")
        return {"type": "decimal", "value": str(value)}
    if type(value) is datetime:
        return {"type": "datetime", "value": value.isoformat()}
    if type(value) is date:
        return {"type": "date", "value": value.isoformat()}
    if type(value) is time:
        return {"type": "time", "value": value.isoformat()}
    if type(value) is bytes:
        return {"type": "bytes", "value": value.hex()}
    raise AnonymizationError(
        f"output contains unsupported scalar type {type(value).__name__}"
    )


__all__ = [
    "DEFAULT_TARGET_K",
    "DEFAULT_TARGET_L",
    "DEFAULT_TARGET_T",
    "MANIFEST_SCHEMA_VERSION",
    "MODEL_K_ANON",
    "REFERENCE_AVERAGE_GENERALIZATION_HEIGHT_CAP",
    "REFERENCE_SUPPRESSION_RATE_CAP",
    "SUPPORTED_MODELS",
    "AnonymizationError",
    "AnonymizationResult",
    "anonymize_table",
]
