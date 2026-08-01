"""Reach a target k-anonymity by delegating to the enforce_kanon engine.

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
column-type-to-hierarchy binding, table coercion, and a raw-value-free manifest.
Quasi-identifiers are supplied explicitly as a ``column -> type`` mapping;
detection is a separate concern and is never inferred here. The path is pure
Python: no JVM, no bundled terminology, and no network access.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from openmed.risk.kanon import enforce_kanon
from openmed.structured.hierarchies import (
    HIERARCHY_SCHEMA_VERSION,
    SUPPORTED_COLUMN_TYPES,
    HierarchyError,
    build_enforcement_hierarchies,
)
from openmed.structured.table_io import read_table

MANIFEST_SCHEMA_VERSION: Final = "1.0.0"

#: The anonymization models this entrypoint understands.
MODEL_K_ANON: Final = "k-anon"
SUPPORTED_MODELS: Final = frozenset({MODEL_K_ANON})

#: Default target group size when a caller does not specify ``target_k``.
DEFAULT_TARGET_K: Final = 2


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
    removed). ``manifest`` records the per-column generalization level and the
    suppression count only -- never any raw source value -- so it is safe to
    persist alongside a release.
    """

    records: tuple[dict[str, Any], ...]
    manifest: dict[str, Any] = field(default_factory=dict)


def anonymize_table(
    table: Any,
    quasi_identifiers: Mapping[str, str],
    *,
    target_k: int = DEFAULT_TARGET_K,
    suppression_limit: int | None = None,
    suppression_rate: float = 0.0,
    sensitive_attributes: Sequence[str] | None = None,
    model: str = MODEL_K_ANON,
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
        suppression_limit: Absolute cap on suppressed rows, or ``None``.
        suppression_rate: Fractional cap on suppressed rows in ``[0, 1]``. When
            both bounds are given the tighter one applies.
        sensitive_attributes: Optional sensitive-attribute columns; forwarded to
            the engine so the report carries their disclosure bounds.
        model: Anonymization model; only ``"k-anon"`` is supported.

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

    column_types = _validated_quasi_identifiers(quasi_identifiers)
    records = _load_records(table)
    _validate_columns_present(records, column_types)

    try:
        hierarchies = build_enforcement_hierarchies(column_types, records)
    except HierarchyError as exc:
        raise AnonymizationError(str(exc)) from exc

    try:
        report = enforce_kanon(
            records,
            quasi_identifiers=list(column_types),
            sensitive_attributes=(
                list(sensitive_attributes) if sensitive_attributes else None
            ),
            hierarchies=hierarchies,
            target_k=target_k,
            suppression_limit=suppression_limit,
            suppression_rate=suppression_rate,
        )
    except ValueError as exc:
        raise AnonymizationError(str(exc)) from exc

    released = tuple(dict(record) for record in report["records"])
    manifest = _build_manifest(
        column_types,
        report,
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
        if column_type not in SUPPORTED_COLUMN_TYPES:
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


# --------------------------------------------------------------------------- #
# Manifest                                                                    #
# --------------------------------------------------------------------------- #
def _build_manifest(
    column_types: Mapping[str, str],
    report: Mapping[str, Any],
    *,
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
    for column, column_type in column_types.items():
        level_info = engine_levels.get(column, {})
        level_index = int(level_info.get("level", 0))
        columns.append(
            {
                "column": column,
                "column_type": column_type,
                "level": level_index,
                "level_name": level_info.get("name"),
                "loss": level_info.get("loss"),
            }
        )
        generalization_levels[column] = level_index

    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "hierarchy_schema_version": HIERARCHY_SCHEMA_VERSION,
        "model": model,
        "target_k": int(report.get("target_k", 0)),
        "achieved_k": int(report.get("kanon", {}).get("k", 0)),
        "quasi_identifiers": dict(column_types),
        "columns": columns,
        "generalization_levels": generalization_levels,
        "record_count": int(report.get("record_count", 0)),
        "released_count": int(report.get("released_count", 0)),
        "suppressed_count": int(report.get("suppressed_count", 0)),
        "suppression_limit": (
            None
            if report.get("suppression_limit") is None
            else int(report["suppression_limit"])
        ),
        "utility": {
            "information_loss": generalization.get("information_loss"),
            "generalization_loss": generalization.get("generalization_loss"),
            "suppression_loss": generalization.get("suppression_loss"),
        },
        "search": {
            "engine": "openmed.risk.kanon.enforce_kanon",
            "strategy": generalization.get("search"),
            "nodes_evaluated": generalization.get("nodes_evaluated"),
            "search_space_size": generalization.get("search_space_size"),
        },
    }


__all__ = [
    "DEFAULT_TARGET_K",
    "MANIFEST_SCHEMA_VERSION",
    "MODEL_K_ANON",
    "SUPPORTED_MODELS",
    "AnonymizationError",
    "AnonymizationResult",
    "anonymize_table",
]
