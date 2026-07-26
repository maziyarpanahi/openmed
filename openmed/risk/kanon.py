"""k-anonymity, l-diversity and t-closeness for tabular records.

The module exposes both measurement and a small full-domain enforcement
engine for structured quasi-identifiers. Enforcement searches the
generalization lattice over age, geography, date and user-supplied hierarchies,
then suppresses only the equivalence classes that still violate the policy and
fit within the declared suppression cap.

Quasi-identifier handling reuses :mod:`openmed.risk.reid` so equivalence-class
keys match :func:`openmed.risk.risk_report`: auto-detection uses the same
``_profile_record`` key, and an explicit ``quasi_identifiers`` list builds the
class key from those fields with the same value normalization.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
from itertools import product
from typing import Any, Mapping, Sequence

from openmed.core.audit import stable_hash

from .reid import (
    _coerce_records as _legacy_coerce_records,
)
from .reid import (
    _field_category,
    _field_is_direct_identifier,
    _normalize_qi_value,
    _profile_record,
    _Record,
)

__all__ = ["build_generalization_hierarchies", "enforce_kanon", "kanon_report"]

_SUPPORTED_L_METRICS = ("distinct", "entropy")
_SUPPORTED_T_DISTANCES = ("variational",)
_SUPPRESSED_VALUE = "*"
_INTERNAL_QI_TOKEN_PREFIX = "__OPENMED_INTERNAL_QI__:"
_SUPPORTED_USER_LEVEL_KEYS = frozenset({"name", "values", "default", "loss"})
_DEFAULT_MAX_LATTICE_NODES = 100_000
_DEFAULT_MAX_SUPPRESSION_SUBSETS = 100_000
_TEXT_FIELDS = (
    "text",
    "note",
    "content",
    "document",
    "deidentified_text",
    "original_text",
)


@dataclass(frozen=True)
class _GeneralizationLevel:
    name: str
    loss: float
    transform: Callable[[Any], Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "loss": float(self.loss)}


@dataclass(frozen=True)
class _Candidate:
    node: tuple[int, ...]
    records: tuple[dict[str, Any], ...]
    report: Mapping[str, Any]
    suppressed_positions: tuple[int, ...]
    information_loss: float
    generalization_loss: float
    suppression_loss: float


@dataclass(frozen=True)
class _InternalQIState:
    kind: str
    values: tuple[str, ...] = ()


@dataclass
class _SuppressionSearchState:
    evaluated: int
    maximum: int

    def consume(self) -> None:
        self.evaluated += 1
        if self.evaluated > self.maximum:
            raise ValueError(
                "Suppression subset search exceeds the configured search budget: "
                f"more than {self.maximum} eligible class subsets. Reduce the "
                "suppression cap, provide coarser hierarchies, or raise "
                "max_suppression_subsets deliberately."
            )


_MISSING_QI = _InternalQIState("missing")
_NULL_QI = _InternalQIState("null")
_EMPTY_QI = _InternalQIState("empty")


def _validated_columns(
    value: Sequence[str] | None,
    *,
    name: str,
    allow_none: bool,
) -> tuple[str, ...] | None:
    if value is None:
        if allow_none:
            return None
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of column names, not a string")
    if not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of column names")
    columns: list[str] = []
    for column in value:
        if not isinstance(column, str) or not column:
            raise ValueError(f"{name} must contain non-empty string column names")
        columns.append(column)
    return tuple(sorted(dict.fromkeys(columns)))


def _coerce_records(data: Any, *, source: str) -> list[_Record]:
    """Coerce structured rows without applying document-container semantics."""

    _validate_dataframe_temporal_precision(data)
    dataframe_columns = getattr(data, "columns", None)
    is_dataframe_like = dataframe_columns is not None
    if dataframe_columns is not None:
        try:
            columns = list(dataframe_columns)
        except TypeError:
            raise TypeError("DataFrame columns must be an iterable schema") from None
        if any(type(column) is not str for column in columns):
            raise TypeError("DataFrame column names must be strings")
        if len(columns) != len(set(columns)):
            raise ValueError("DataFrame column names must be unique")

    to_dicts = getattr(data, "to_dicts", None)
    if callable(to_dicts):
        data = to_dicts()
    else:
        to_dict = getattr(data, "to_dict", None)
        if callable(to_dict) and not isinstance(data, Mapping):
            data = to_dict("records")

    if isinstance(data, Mapping):
        rows: Sequence[Mapping[Any, Any]] = [data]
    elif isinstance(data, Sequence) and not isinstance(
        data,
        (str, bytes, bytearray),
    ):
        if not all(isinstance(row, Mapping) for row in data):
            return _legacy_coerce_records(data, source=source)
        rows = data
    else:
        return _legacy_coerce_records(data, source=source)

    records: list[_Record] = []
    for index, row in enumerate(rows):
        fields: dict[str, Any] = {}
        for column_index, (field, value) in enumerate(row.items()):
            if type(field) is not str:
                raise TypeError(
                    "Structured column names must be strings; unsupported name "
                    f"at row offset {index}, column offset {column_index}"
                )
            fields[field] = (
                _normalized_dataframe_scalar(value) if is_dataframe_like else value
            )
        text = next(
            (
                value
                for field in _TEXT_FIELDS
                if isinstance((value := fields.get(field)), str)
            ),
            "",
        )
        records.append(
            _Record(
                index=index,
                record_id=None,
                text=text,
                fields=fields,
                spans=(),
                source=source,
            )
        )
    return records


def _validate_dataframe_temporal_precision(data: Any) -> None:
    """Reject DataFrame temporal dtypes that lose precision on conversion."""

    if not type(data).__module__.startswith("polars"):
        return
    schema = getattr(data, "schema", None)
    if callable(schema):
        schema = schema()
    if not isinstance(schema, Mapping):
        return
    for dtype in schema.values():
        rendered = str(dtype)
        if rendered == "Time" or (
            rendered.startswith("Datetime(")
            and getattr(dtype, "time_unit", None) == "ns"
        ):
            raise ValueError(
                "Polars temporal columns with sub-microsecond precision are unsupported"
            )


def _normalized_dataframe_scalar(value: Any) -> Any:
    """Convert common DataFrame scalar wrappers to supported Python scalars."""

    module = type(value).__module__
    type_name = type(value).__name__
    if module.startswith("pandas") and type_name in {"NAType", "NaTType"}:
        return None
    if module.startswith("pandas"):
        to_pydatetime = getattr(value, "to_pydatetime", None)
        if callable(to_pydatetime):
            if getattr(value, "nanosecond", 0):
                raise ValueError(
                    "DataFrame timestamps with sub-microsecond precision are "
                    "unsupported"
                )
            converted = to_pydatetime()
            if type(converted) is datetime:
                return converted
    if module.startswith("numpy"):
        if type_name == "datetime64":
            microseconds = value.astype("datetime64[us]")
            converted = microseconds.item()
            if converted is not None and bool(
                microseconds.astype(value.dtype) != value
            ):
                raise ValueError(
                    "DataFrame timestamps with sub-microsecond precision are "
                    "unsupported"
                )
        elif type_name == "timedelta64":
            raise TypeError("DataFrame time durations are unsupported")
        else:
            item = getattr(value, "item", None)
            converted = item() if callable(item) else value
        if converted is not value:
            return converted
    return value


def _typed_sensitive_value(value: Any) -> str:
    return _typed_scalar_token(value)


def _typed_sensitive_distribution_value(value: Any) -> str:
    payload = _exact_qi_scalar_payload(value)
    return (
        _INTERNAL_QI_TOKEN_PREFIX
        + "sensitive-distribution:"
        + json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _typed_qi_value(field: str, value: Any) -> str:
    del field
    payload = _exact_qi_scalar_payload(value)
    return (
        _INTERNAL_QI_TOKEN_PREFIX
        + "typed:"
        + json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _typed_scalar_token(value: Any) -> str:
    payload = _canonical_scalar_payload(value)
    return (
        _INTERNAL_QI_TOKEN_PREFIX
        + "typed:"
        + json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _exact_qi_scalar_payload(value: Any) -> dict[str, Any]:
    """Return the typed representation materialized for a released QI."""

    payload = _canonical_scalar_payload(value)
    if type(value) is float:
        return {"type": "float", "value": repr(value)}
    if type(value) is str:
        return {"type": "str", "value": value}
    if type(value) is Decimal:
        return {"type": "decimal", "value": str(value)}
    if type(value) is datetime:
        if value.tzinfo is not None and value.utcoffset() is None:
            raise ValueError("datetime timezone offsets must be determinate")
        return {"type": "datetime", "value": value.isoformat()}
    return payload


def _canonical_scalar_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, _InternalQIState):
        return {
            "type": value.kind,
            "value": list(value.values) if value.values else None,
        }
    if value is None:
        return {"type": "null", "value": None}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("floating-point values must be finite")
        return {
            "type": "float",
            "value": "0" if value == 0.0 else format(value, ".17g"),
        }
    if type(value) is str:
        return {"type": "str", "value": unicodedata.normalize("NFC", value)}
    if type(value) is Decimal:
        if not value.is_finite():
            raise ValueError("decimal values must be finite")
        return {"type": "decimal", "value": _canonical_decimal_text(value)}
    if type(value) is datetime:
        if value.tzinfo is not None and value.utcoffset() is None:
            raise ValueError("datetime timezone offsets must be determinate")
        canonical = (
            value.astimezone(timezone.utc) if value.tzinfo is not None else value
        )
        return {"type": "datetime", "value": canonical.isoformat()}
    if type(value) is date:
        return {"type": "date", "value": value.isoformat()}
    if type(value) is time:
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ValueError("timezone-aware time values are unsupported")
        return {"type": "time", "value": value.isoformat()}
    if type(value) is bytes:
        return {"type": "bytes", "value": value.hex()}
    raise TypeError("structured values must be supported tabular scalars")


def _canonical_decimal_text(value: Decimal) -> str:
    """Canonicalize a finite Decimal without applying context precision."""

    if value.is_zero():
        return "0"
    components = value.as_tuple()
    if not isinstance(components.exponent, int):
        raise ValueError("decimal values must be finite")
    sign, digits, exponent = (
        components.sign,
        components.digits,
        components.exponent,
    )
    trimmed = list(digits)
    while trimmed[-1] == 0:
        trimmed.pop()
        exponent += 1
    coefficient = "".join(str(digit) for digit in trimmed)
    prefix = "-" if sign else ""
    return f"{prefix}{coefficient}e{exponent}"


def _escaped_literal_string(value: str) -> str:
    if not value.startswith(_INTERNAL_QI_TOKEN_PREFIX):
        return value
    return (
        _INTERNAL_QI_TOKEN_PREFIX
        + "literal:"
        + json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def _canonical_hash_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(field): _canonical_hash_value(item) for field, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_canonical_hash_value(item) for item in value]
    return _canonical_scalar_payload(value)


def _record_hash(record: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "kind": "openmed-kanon-record",
            "fields": _canonical_hash_value(record),
        }
    )


def kanon_report(
    records: Any,
    quasi_identifiers: Sequence[str] | None = None,
    sensitive_attributes: Sequence[str] | None = None,
    *,
    l_metric: str = "distinct",
    t_distance: str = "variational",
) -> dict[str, Any]:
    """Measure k-anonymity, l-diversity and t-closeness for ``records``.

    Args:
        records: Tabular records in any shape accepted by ``risk_report``
            (mapping, sequence of mappings, or a DataFrame-like object).
        quasi_identifiers: Explicit quasi-identifier field names. When omitted,
            quasi-identifiers are auto-detected consistently with
            ``risk_report``'s profiling.
        sensitive_attributes: Field names whose distribution drives l-diversity
            and t-closeness. When omitted, only k-anonymity is reported.
        l_metric: Reserved selector for the headline l-diversity metric; both
            distinct count and Shannon entropy are always reported per class.
        t_distance: Distance used for t-closeness. Only ``"variational"``
            (total-variation distance) is currently supported.

    Returns:
        A deterministic, JSON-serializable mapping with equivalence-class sizes,
        k (min class size), per-class l-diversity and t-closeness, and the
        worst-case (overall) l-diversity and t-closeness.
    """
    if l_metric not in _SUPPORTED_L_METRICS:
        raise ValueError(
            f"Unsupported l_metric {l_metric!r}; "
            f"supported: {', '.join(_SUPPORTED_L_METRICS)}."
        )
    if t_distance not in _SUPPORTED_T_DISTANCES:
        raise ValueError(
            f"Unsupported t_distance {t_distance!r}; "
            f"supported: {', '.join(_SUPPORTED_T_DISTANCES)}."
        )

    qis = _validated_columns(
        quasi_identifiers,
        name="quasi_identifiers",
        allow_none=True,
    )
    sensitive = (
        _validated_columns(
            sensitive_attributes,
            name="sensitive_attributes",
            allow_none=True,
        )
        or ()
    )
    coerced = _coerce_records(records, source="deidentified")

    members: defaultdict[Any, list[int]] = defaultdict(list)
    json_keys: dict[Any, list[Any]] = {}
    sensitive_values: dict[int, dict[str, str]] = {}
    sensitive_distribution_values: dict[int, dict[str, str]] = {}

    for record in coerced:
        hash_key, json_key = _equivalence_key(record, qis)
        members[hash_key].append(record.index)
        json_keys.setdefault(hash_key, json_key)
        sensitive_values[record.index] = {}
        sensitive_distribution_values[record.index] = {}
        for attr in sensitive:
            if attr not in record.fields:
                raise ValueError(
                    f"Sensitive attribute {attr!r} is missing at record offset "
                    f"{record.index}; missing values cannot count toward "
                    "l-diversity"
                )
            value = record.fields[attr]
            if (
                value is None
                or (
                    isinstance(value, str)
                    and (not value.strip() or value != value.strip())
                )
                or (type(value) is bytes and not value)
            ):
                raise ValueError(
                    f"Sensitive attribute {attr!r} is empty or has surrounding "
                    f"whitespace at record offset {record.index}; ambiguous "
                    "values cannot count toward l-diversity"
                )
            try:
                sensitive_values[record.index][attr] = _typed_sensitive_value(value)
                sensitive_distribution_values[record.index][attr] = (
                    _typed_sensitive_distribution_value(value)
                )
            except (TypeError, ValueError) as exc:
                raise type(exc)(
                    f"Sensitive attribute {attr!r} is unsupported or non-finite "
                    f"at record offset {record.index}"
                ) from None

    global_dist = {
        attr: _distribution(
            sensitive_distribution_values[idx][attr]
            for idx in sensitive_distribution_values
            if attr in sensitive_distribution_values[idx]
        )
        for attr in sensitive
    }

    classes: list[dict[str, Any]] = []
    for hash_key in members:
        indices = sorted(members[hash_key])
        per_class_l: dict[str, Any] = {}
        per_class_t: dict[str, float] = {}
        for attr in sensitive:
            l_values = [sensitive_values[idx][attr] for idx in indices]
            distribution_values = [
                sensitive_distribution_values[idx][attr] for idx in indices
            ]
            counts = Counter(l_values)
            per_class_l[attr] = {
                "distinct": len(counts),
                "entropy": _entropy(counts),
            }
            per_class_t[attr] = _variational_distance(
                _distribution(distribution_values), global_dist[attr]
            )
        classes.append(
            {
                "key": json_keys[hash_key],
                "size": len(indices),
                "members": indices,
                "l_diversity": per_class_l,
                "t_closeness": per_class_t,
            }
        )

    classes.sort(key=lambda cls: json.dumps(cls["key"], sort_keys=True))

    sizes: list[int] = [int(cls["size"]) for cls in classes]
    overall_l = {
        attr: {
            "min_distinct": min(
                (cls["l_diversity"][attr]["distinct"] for cls in classes),
                default=0,
            ),
            "min_entropy": min(
                (cls["l_diversity"][attr]["entropy"] for cls in classes),
                default=0.0,
            ),
        }
        for attr in sensitive
    }
    overall_t = {
        attr: max(
            (cls["t_closeness"][attr] for cls in classes),
            default=0.0,
        )
        for attr in sensitive
    }

    return {
        "record_count": len(coerced),
        "quasi_identifiers": _reported_quasi_identifiers(qis, classes),
        "sensitive_attributes": sorted(sensitive),
        "k": min(sizes) if sizes else 0,
        "class_count": len(classes),
        "class_size_distribution": _size_distribution(sizes),
        "equivalence_classes": classes,
        "l": _headline_l_diversity(overall_l, l_metric),
        "l_diversity": overall_l,
        "t_closeness": overall_t,
        "l_metric": l_metric,
        "t_distance": t_distance,
    }


def build_generalization_hierarchies(
    records: Any,
    quasi_identifiers: Sequence[str] | None = None,
    *,
    hierarchies: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return the field-level generalization hierarchies used by enforcement.

    Default hierarchies cover common structured quasi-identifiers: ages roll up
    from exact ages to five-year, ten-year, twenty-year and suppressed bands;
    dates roll up to month, year, decade and suppression; geography rolls up
    from exact values to postal/state-style regions and suppression. A
    user-supplied hierarchy may provide levels with ``name``, ``values``,
    optional ``default`` and optional ``loss`` keys.
    """

    explicit_qis = _validated_columns(
        quasi_identifiers,
        name="quasi_identifiers",
        allow_none=True,
    )
    coerced = _coerce_records(records, source="deidentified")
    qis = _resolve_quasi_identifier_fields(coerced, explicit_qis)
    levels = _build_hierarchy_levels(coerced, qis, hierarchies)
    return {
        field: [level.to_dict() for level in field_levels]
        for field, field_levels in levels.items()
    }


def enforce_kanon(
    records: Any,
    quasi_identifiers: Sequence[str] | None = None,
    sensitive_attributes: Sequence[str] | None = None,
    *,
    target_k: int = 2,
    target_l: int = 1,
    target_t: float = 1.0,
    suppression_limit: int | None = None,
    suppression_rate: float = 0.0,
    hierarchies: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    remove_direct_identifiers: bool = True,
    l_metric: str = "distinct",
    max_lattice_nodes: int = _DEFAULT_MAX_LATTICE_NODES,
    max_suppression_subsets: int = _DEFAULT_MAX_SUPPRESSION_SUBSETS,
) -> dict[str, Any]:
    """Generalize and suppress records until the declared k/l/t policy holds.

    The search is full-domain: one level per quasi-identifier is applied to the
    whole corpus, then violating equivalence classes are suppressed if the
    suppression cap allows it. The selected node minimizes a documented
    information-loss metric over the exhaustive lattice, so small-fixture
    optimum checks can use zero tolerance.

    Proof sketch for the identity bound: after enforcement, each released
    equivalence class contains at least ``target_k`` records. Any attacker that
    matches only on released quasi-identifiers cannot distinguish records inside
    a class, so each released record has re-identification probability at most
    ``1 / class_size <= 1 / target_k``. l-diversity and t-closeness are emitted
    as separate sensitive-attribute disclosure bounds; they do not reduce the
    identity bound itself, but they do tighten the reported upper bound for
    sensitive value confidence and the joint identity-plus-sensitive event.
    """

    _validate_policy(
        target_k,
        target_l,
        target_t,
        suppression_rate,
        l_metric=l_metric,
        max_lattice_nodes=max_lattice_nodes,
        max_suppression_subsets=max_suppression_subsets,
    )
    explicit_qis = _validated_columns(
        quasi_identifiers,
        name="quasi_identifiers",
        allow_none=True,
    )
    sensitive = (
        _validated_columns(
            sensitive_attributes,
            name="sensitive_attributes",
            allow_none=True,
        )
        or ()
    )
    coerced = _coerce_records(records, source="deidentified")
    qis = _resolve_quasi_identifier_fields(coerced, explicit_qis)
    if target_l > 1 and not sensitive:
        raise ValueError("target_l > 1 requires at least one sensitive attribute")
    if target_t < 1.0 and not sensitive:
        raise ValueError("target_t < 1.0 requires at least one sensitive attribute")

    if not coerced:
        empty_report = kanon_report(
            [],
            quasi_identifiers=qis,
            sensitive_attributes=sensitive,
            l_metric=l_metric,
        )
        return {
            "schema_version": 1,
            "record_count": 0,
            "released_count": 0,
            "suppressed_count": 0,
            "target_k": int(target_k),
            "target_l": int(target_l),
            "l_metric": l_metric,
            "target_t": float(target_t),
            "quasi_identifiers": qis,
            "sensitive_attributes": sensitive,
            "records": [],
            "kanon": empty_report,
            "suppressed_records": [],
            "generalization": {
                "node": {},
                "levels": {},
                "information_loss": 0.0,
                "generalization_loss": 0.0,
                "suppression_loss": 0.0,
                "optimality_tolerance": 0.0,
                "search": "full-domain exhaustive lattice",
                "search_space_size": 0,
                "nodes_evaluated": 0,
                "max_lattice_nodes": int(max_lattice_nodes),
                "suppression_search": "exhaustive equivalence-class subsets",
                "suppression_subsets_evaluated": 0,
                "suppression_subsets_possible": 0,
                "max_suppression_subsets": int(max_suppression_subsets),
                "search_complete": True,
            },
            "bounds": _bound_report(
                [],
                empty_report,
                (),
                target_k=target_k,
                target_l=target_l,
                target_t=target_t,
                sensitive_attributes=sensitive,
                l_metric=l_metric,
            ),
        }

    levels = _build_hierarchy_levels(coerced, qis, hierarchies)
    search_space_size = math.prod(len(levels[field]) for field in qis)
    if search_space_size > max_lattice_nodes:
        raise ValueError(
            "Generalization lattice exceeds the configured search budget: "
            f"{search_space_size} nodes for {len(qis)} quasi-identifiers, "
            f"max_lattice_nodes={max_lattice_nodes}. Reduce the quasi-identifier "
            "set, provide coarser explicit hierarchies, or raise the budget "
            "deliberately."
        )
    budget = _suppression_budget(
        len(coerced),
        suppression_limit=suppression_limit,
        suppression_rate=suppression_rate,
    )
    suppression_search = _SuppressionSearchState(
        evaluated=0,
        maximum=max_suppression_subsets,
    )
    candidate = _search_lattice(
        coerced,
        qis,
        sensitive,
        levels,
        target_k=target_k,
        target_l=target_l,
        target_t=target_t,
        suppression_budget=budget,
        remove_direct_identifiers=remove_direct_identifiers,
        l_metric=l_metric,
        suppression_search=suppression_search,
    )
    if candidate is None:
        raise ValueError(
            "No generalization satisfies the requested k/l/t targets within "
            f"the suppression cap ({budget} of {len(coerced)} records)."
        )

    field_order = tuple(qis)
    node_by_field = {
        field: {
            "level": candidate.node[index],
            **levels[field][candidate.node[index]].to_dict(),
        }
        for index, field in enumerate(field_order)
    }
    suppressed = _suppressed_records(
        coerced,
        candidate.suppressed_positions,
        reason="privacy_class_violation",
    )
    bounds = _bound_report(
        candidate.records,
        candidate.report,
        candidate.suppressed_positions,
        target_k=target_k,
        target_l=target_l,
        target_t=target_t,
        sensitive_attributes=sensitive,
        l_metric=l_metric,
    )
    return {
        "schema_version": 1,
        "record_count": len(coerced),
        "released_count": len(candidate.records),
        "suppressed_count": len(candidate.suppressed_positions),
        "suppression_limit": budget,
        "target_k": int(target_k),
        "target_l": int(target_l),
        "l_metric": l_metric,
        "target_t": float(target_t),
        "quasi_identifiers": field_order,
        "sensitive_attributes": sensitive,
        "records": [dict(record) for record in candidate.records],
        "kanon": candidate.report,
        "suppressed_records": suppressed,
        "generalization": {
            "node": {
                field: candidate.node[index] for index, field in enumerate(field_order)
            },
            "levels": node_by_field,
            "information_loss": candidate.information_loss,
            "generalization_loss": candidate.generalization_loss,
            "suppression_loss": candidate.suppression_loss,
            "optimality_tolerance": 0.0,
            "search": "full-domain exhaustive lattice",
            "search_space_size": int(search_space_size),
            "nodes_evaluated": int(search_space_size),
            "max_lattice_nodes": int(max_lattice_nodes),
            "suppression_search": "exhaustive equivalence-class subsets",
            "suppression_subsets_evaluated": suppression_search.evaluated,
            "suppression_subsets_possible": suppression_search.evaluated,
            "max_suppression_subsets": int(max_suppression_subsets),
            "search_complete": True,
        },
        "bounds": bounds,
    }


def _equivalence_key(
    record: Any,
    quasi_identifiers: Sequence[str] | None,
) -> tuple[Any, list[Any]]:
    """Return (hashable grouping key, JSON-serializable key) for a record."""
    if quasi_identifiers:
        pairs = tuple(
            (field, _explicit_qi_value(record.fields, field))
            for field in sorted(quasi_identifiers)
        )
        return pairs, [[field, value] for field, value in pairs]

    profile_key = _profile_record(record).key
    json_key = [[category, list(values)] for category, values in profile_key]
    return profile_key, json_key


def _explicit_qi_value(fields: Mapping[str, Any], field: str) -> str:
    """Return a collision-resistant value for one explicitly declared QI."""

    if field not in fields:
        return _typed_qi_value(field, _MISSING_QI)
    value = fields[field]
    if value is None:
        return _typed_qi_value(field, _NULL_QI)
    if isinstance(value, str) and not value:
        return _typed_qi_value(field, _EMPTY_QI)
    return _typed_qi_value(field, value)


def _exact_qi_value(field: str, value: Any) -> Any:
    if _is_internal_qi_token(value):
        suffix = (
            ""
            if not value.values
            else ":"
            + json.dumps(
                value.values,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return f"{_INTERNAL_QI_TOKEN_PREFIX}state:{value.kind}{suffix}"
    _typed_qi_value(field, value)
    if isinstance(value, str):
        return _escaped_literal_string(value)
    return value


def _is_internal_qi_token(value: Any) -> bool:
    return isinstance(value, _InternalQIState)


def _distribution(values: Any) -> dict[str, float]:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {value: count / total for value, count in counts.items()}


def _entropy(counts: Mapping[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count == 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    # Normalize -0.0 to 0.0 for clean, JSON-stable output.
    return entropy + 0.0


def _variational_distance(
    class_dist: Mapping[str, float],
    global_dist: Mapping[str, float],
) -> float:
    values = set(class_dist) | set(global_dist)
    total = sum(
        abs(class_dist.get(value, 0.0) - global_dist.get(value, 0.0))
        for value in values
    )
    return 0.5 * total


def _size_distribution(sizes: Sequence[int]) -> list[list[int]]:
    counts = Counter(sizes)
    return [[size, counts[size]] for size in sorted(counts)]


def _headline_l_diversity(
    overall_l: Mapping[str, Mapping[str, int | float]],
    l_metric: str,
) -> dict[str, int | float]:
    field = "min_distinct" if l_metric == "distinct" else "min_entropy"
    return {attr: metrics[field] for attr, metrics in overall_l.items()}


def _reported_quasi_identifiers(
    quasi_identifiers: Sequence[str] | None,
    classes: Sequence[Mapping[str, Any]],
) -> list[str]:
    if quasi_identifiers:
        return sorted(quasi_identifiers)
    categories: set[str] = set()
    for cls in classes:
        for entry in cls["key"]:
            categories.add(str(entry[0]))
    return sorted(categories)


def _validate_policy(
    target_k: int,
    target_l: int,
    target_t: float,
    suppression_rate: float,
    *,
    l_metric: str = "distinct",
    max_lattice_nodes: int = _DEFAULT_MAX_LATTICE_NODES,
    max_suppression_subsets: int = _DEFAULT_MAX_SUPPRESSION_SUBSETS,
) -> None:
    if type(target_k) is not int or target_k < 1:
        raise ValueError("target_k must be an integer >= 1")
    if type(target_l) is not int or target_l < 1:
        raise ValueError("target_l must be an integer >= 1")
    if (
        not isinstance(target_t, (int, float))
        or isinstance(target_t, bool)
        or not math.isfinite(float(target_t))
        or not 0.0 <= float(target_t) <= 1.0
    ):
        raise ValueError("target_t must be between 0.0 and 1.0")
    if (
        not isinstance(suppression_rate, (int, float))
        or isinstance(suppression_rate, bool)
        or not math.isfinite(float(suppression_rate))
        or not 0.0 <= float(suppression_rate) <= 1.0
    ):
        raise ValueError("suppression_rate must be between 0.0 and 1.0")
    if l_metric not in _SUPPORTED_L_METRICS:
        raise ValueError(
            f"Unsupported l_metric {l_metric!r}; "
            f"supported: {', '.join(_SUPPORTED_L_METRICS)}."
        )
    if type(max_lattice_nodes) is not int or max_lattice_nodes < 1:
        raise ValueError("max_lattice_nodes must be an integer >= 1")
    if type(max_suppression_subsets) is not int or max_suppression_subsets < 1:
        raise ValueError("max_suppression_subsets must be an integer >= 1")


def _resolve_quasi_identifier_fields(
    records: Sequence[Any],
    quasi_identifiers: Sequence[str] | None,
) -> list[str]:
    if quasi_identifiers:
        return sorted({str(field) for field in quasi_identifiers})

    fields: set[str] = set()
    for record in records:
        for field in record.fields:
            if _field_category(field) is not None:
                fields.add(str(field))
    if fields:
        return sorted(fields)

    categories: set[str] = set()
    for record in records:
        profile = _profile_record(record)
        for category, values in profile.key:
            if values:
                categories.add(category)
    return sorted(categories)


def _build_hierarchy_levels(
    records: Sequence[Any],
    quasi_identifiers: Sequence[str],
    supplied: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> dict[str, tuple[_GeneralizationLevel, ...]]:
    supplied = supplied or {}
    unknown = sorted(set(supplied) - set(quasi_identifiers))
    if unknown:
        raise ValueError(
            f"Hierarchies were supplied for undeclared quasi-identifiers: {unknown!r}"
        )
    result: dict[str, tuple[_GeneralizationLevel, ...]] = {}
    for field in quasi_identifiers:
        if field in supplied:
            result[field] = _user_hierarchy(field, supplied[field])
        else:
            result[field] = _default_hierarchy(field, records)
    return result


def _user_hierarchy(
    field: str,
    levels: Sequence[Mapping[str, Any]],
) -> tuple[_GeneralizationLevel, ...]:
    if not levels:
        raise ValueError(f"Hierarchy for {field!r} must contain at least one level")

    built: list[_GeneralizationLevel] = []
    max_index = max(1, len(levels) - 1)
    for index, level in enumerate(levels):
        unknown = set(level) - _SUPPORTED_USER_LEVEL_KEYS
        if unknown:
            raise ValueError(
                f"Unsupported hierarchy keys for {field!r}: {sorted(unknown)}"
            )
        values = level.get("values")
        if values is not None and not isinstance(values, Mapping):
            raise ValueError(f"Hierarchy level {field!r}[{index}] values must be a map")
        if index == 0 and (values is not None or "default" in level):
            raise ValueError(
                f"Hierarchy level {field!r}[0] must be a canonical identity "
                "level without values or a default"
            )
        value_map = {str(key): str(value) for key, value in dict(values or {}).items()}
        default = level.get("default")
        if any(
            value.startswith(_INTERNAL_QI_TOKEN_PREFIX) for value in value_map.values()
        ) or (
            default is not None and str(default).startswith(_INTERNAL_QI_TOKEN_PREFIX)
        ):
            raise ValueError(
                f"Hierarchy level {field!r}[{index}] outputs cannot use the "
                "reserved internal namespace"
            )
        loss = _optional_float(level.get("loss"))
        if "loss" in level and loss is None:
            raise ValueError(f"Hierarchy level {field!r}[{index}] loss must be finite")
        if loss is None:
            loss = index / max_index
        if index == 0 and loss != 0.0:
            raise ValueError(f"Hierarchy level {field!r}[0] identity loss must be 0")
        if index > 0 and loss <= 0.0:
            raise ValueError(
                f"Hierarchy coarsening level {field!r}[{index}] loss must be "
                "greater than 0"
            )
        if not 0.0 <= loss <= 1.0:
            raise ValueError(
                f"Hierarchy level {field!r}[{index}] loss must be between 0 and 1"
            )
        if built and loss < built[-1].loss:
            raise ValueError(f"Hierarchy losses for {field!r} must be non-decreasing")

        def transform(
            value: Any,
            *,
            mapping: Mapping[str, str] = value_map,
            default_value: Any = default,
            level_index: int = index,
        ) -> Any:
            if level_index == 0:
                return _exact_qi_value(field, value)
            if _is_internal_qi_token(value):
                return (
                    str(default_value)
                    if default_value is not None
                    else _SUPPRESSED_VALUE
                )
            exact = str(value)
            normalized = _normalize_qi_value(field, value)
            if exact in mapping:
                return mapping[exact]
            if normalized in mapping:
                return mapping[normalized]
            if default_value is not None:
                return str(default_value)
            return _escaped_literal_string(normalized)

        built.append(
            _GeneralizationLevel(
                name=str(level.get("name") or f"level_{index}"),
                loss=float(loss),
                transform=transform,
            )
        )
    return tuple(built)


def _default_hierarchy(
    field: str,
    records: Sequence[Any],
) -> tuple[_GeneralizationLevel, ...]:
    category = _field_category(field) or field
    if category == "age":
        return (
            _level("exact", 0.0, lambda value: _exact_qi_value("age", value)),
            _level("age_5_year_band", 0.25, lambda value: _age_band(value, 5)),
            _level("age_10_year_band", 0.5, lambda value: _age_band(value, 10)),
            _level("age_20_year_band", 0.75, lambda value: _age_band(value, 20)),
            _level("suppressed", 1.0, lambda value: _SUPPRESSED_VALUE),
        )
    if category == "date":
        return (
            _level("exact", 0.0, lambda value: _exact_qi_value("date", value)),
            _level("month", 0.25, _date_month),
            _level("year", 0.5, _date_year),
            _level("decade", 0.75, _date_decade),
            _level("suppressed", 1.0, lambda value: _SUPPRESSED_VALUE),
        )
    if category == "geography":
        return (
            _level(
                "exact",
                0.0,
                lambda value: _exact_qi_value("geography", value),
            ),
            _level("regional", 0.4, _geography_region),
            _level("broad_region", 0.7, _geography_broad_region),
            _level("suppressed", 1.0, lambda value: _SUPPRESSED_VALUE),
        )

    # Arbitrary categories, facilities, and clinical codes do not have a
    # defensible hierarchy that can be inferred from their spelling. In
    # particular, a first-character prefix is not a semantic parent. Callers
    # that need useful intermediate levels must provide an explicit, reviewed
    # hierarchy; the safe fallback is exact-or-suppressed.
    return (
        _level("exact", 0.0, lambda value: _exact_qi_value(field, value)),
        _level("suppressed", 1.0, lambda value: _SUPPRESSED_VALUE),
    )


def _level(
    name: str,
    loss: float,
    transform: Callable[[Any], Any],
) -> _GeneralizationLevel:
    return _GeneralizationLevel(name=name, loss=loss, transform=transform)


def _age_band(value: Any, width: int) -> str:
    if _is_internal_qi_token(value):
        return _SUPPRESSED_VALUE
    normalized = _normalize_qi_value("age", value)
    parsed = _optional_int(normalized)
    if parsed is None:
        return _SUPPRESSED_VALUE
    lower = (parsed // width) * width
    upper = lower + width - 1
    return f"{lower}-{upper}"


def _date_parts(value: Any) -> tuple[int, int | None] | None:
    if _is_internal_qi_token(value):
        return None
    text = _normalize_qi_value("date", value)
    match = re.match(r"^(\d{4})(?:[-/](\d{1,2}))?", text)
    if not match:
        return None
    year = _optional_int(match.group(1))
    month = _optional_int(match.group(2))
    if year is None:
        return None
    if month is not None and not 1 <= month <= 12:
        month = None
    return year, month


def _date_month(value: Any) -> str:
    parts = _date_parts(value)
    if parts is None:
        return _SUPPRESSED_VALUE
    year, month = parts
    if month is None:
        return str(year)
    return f"{year:04d}-{month:02d}"


def _date_year(value: Any) -> str:
    parts = _date_parts(value)
    if parts is None:
        return _SUPPRESSED_VALUE
    return f"{parts[0]:04d}"


def _date_decade(value: Any) -> str:
    parts = _date_parts(value)
    if parts is None:
        return _SUPPRESSED_VALUE
    decade = (parts[0] // 10) * 10
    return f"{decade:04d}s"


def _geography_region(value: Any) -> str:
    if _is_internal_qi_token(value):
        return _SUPPRESSED_VALUE
    text = _normalize_qi_value("geography", value)
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 5:
        return f"{digits[:3]}**"
    if "," in text:
        return text.rsplit(",", 1)[-1].strip() or _SUPPRESSED_VALUE
    pieces = text.split()
    if len(pieces) > 1:
        return pieces[-1]
    return text[:3] + "*" if len(text) > 3 else text or _SUPPRESSED_VALUE


def _geography_broad_region(value: Any) -> str:
    if _is_internal_qi_token(value):
        return _SUPPRESSED_VALUE
    text = _normalize_qi_value("geography", value)
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 5:
        return f"{digits[:2]}***"
    region = _geography_region(text)
    return region[:1] + "*" if len(region) > 1 else region


def _suppression_budget(
    record_count: int,
    *,
    suppression_limit: int | None,
    suppression_rate: float,
) -> int:
    if suppression_limit is not None and (
        type(suppression_limit) is not int or suppression_limit < 0
    ):
        raise ValueError("suppression_limit must be an integer >= 0")
    rate_budget = math.floor(record_count * suppression_rate)
    if suppression_limit is None:
        return rate_budget
    if suppression_rate > 0.0:
        return min(int(suppression_limit), rate_budget)
    return int(suppression_limit)


def _search_lattice(
    records: Sequence[Any],
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str],
    levels: Mapping[str, Sequence[_GeneralizationLevel]],
    *,
    target_k: int,
    target_l: int,
    target_t: float,
    suppression_budget: int,
    remove_direct_identifiers: bool,
    l_metric: str = "distinct",
    suppression_search: _SuppressionSearchState,
) -> _Candidate | None:
    field_order = tuple(quasi_identifiers)
    best: _Candidate | None = None
    ranges = [range(len(levels[field])) for field in field_order]
    for node in product(*ranges):
        candidate = _evaluate_lattice_node(
            records,
            field_order,
            sensitive_attributes,
            levels,
            node,
            target_k=target_k,
            target_l=target_l,
            target_t=target_t,
            suppression_budget=suppression_budget,
            remove_direct_identifiers=remove_direct_identifiers,
            l_metric=l_metric,
            suppression_search=suppression_search,
        )
        if candidate is None:
            continue
        if best is None or _candidate_sort_key(candidate) < _candidate_sort_key(best):
            best = candidate
    return best


def _candidate_sort_key(candidate: _Candidate) -> tuple[Any, ...]:
    return (
        candidate.information_loss,
        candidate.suppression_loss,
        sum(candidate.node),
        candidate.node,
        candidate.suppressed_positions,
    )


def _evaluate_lattice_node(
    records: Sequence[Any],
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str],
    levels: Mapping[str, Sequence[_GeneralizationLevel]],
    node: tuple[int, ...],
    *,
    target_k: int,
    target_l: int,
    target_t: float,
    suppression_budget: int,
    remove_direct_identifiers: bool,
    l_metric: str = "distinct",
    suppression_search: _SuppressionSearchState | None = None,
) -> _Candidate | None:
    if suppression_search is None:
        suppression_search = _SuppressionSearchState(
            evaluated=0,
            maximum=_DEFAULT_MAX_SUPPRESSION_SUBSETS,
        )
    transformed = tuple(
        _transform_record(
            record,
            quasi_identifiers,
            levels,
            node,
            remove_direct_identifiers=remove_direct_identifiers,
        )
        for record in records
    )
    initial_report = kanon_report(
        transformed,
        quasi_identifiers=quasi_identifiers,
        sensitive_attributes=sensitive_attributes,
        l_metric=l_metric,
    )
    classes = [
        item
        for item in initial_report.get("equivalence_classes", [])
        if isinstance(item, Mapping)
    ]
    class_positions = tuple(
        tuple(
            member
            for member in (
                _optional_int(value) for value in equivalence_class.get("members", [])
            )
            if member is not None and 0 <= member < len(transformed)
        )
        for equivalence_class in classes
    )
    mandatory_indices = {
        index
        for index, equivalence_class in enumerate(classes)
        if not _class_satisfies_k_l(
            equivalence_class,
            target_k=target_k,
            target_l=target_l,
            sensitive_attributes=sensitive_attributes,
            l_metric=l_metric,
        )
    }
    mandatory_positions = {
        position for index in mandatory_indices for position in class_positions[index]
    }
    if len(mandatory_positions) > suppression_budget:
        return None

    generalization_loss = _generalization_loss(quasi_identifiers, levels, node)
    optional_classes = tuple(
        positions
        for index, positions in enumerate(class_positions)
        if index not in mandatory_indices
    )
    remaining_budget = suppression_budget - len(mandatory_positions)
    best: _Candidate | None = None
    for optional_positions in _class_subset_positions(
        optional_classes,
        remaining_budget,
    ):
        suppression_search.consume()
        suppressed = mandatory_positions | set(optional_positions)
        final_positions = [
            position
            for position in range(len(transformed))
            if position not in suppressed
        ]
        if not final_positions:
            continue
        final_records = tuple(transformed[position] for position in final_positions)
        final_report = kanon_report(
            final_records,
            quasi_identifiers=quasi_identifiers,
            sensitive_attributes=sensitive_attributes,
            l_metric=l_metric,
        )
        if not _report_satisfies(
            final_report,
            target_k=target_k,
            target_l=target_l,
            target_t=target_t,
            sensitive_attributes=sensitive_attributes,
            l_metric=l_metric,
        ):
            continue
        suppression_loss = len(suppressed) / len(records) if records else 0.0
        candidate = _Candidate(
            node=node,
            records=final_records,
            report=final_report,
            suppressed_positions=tuple(sorted(suppressed)),
            information_loss=generalization_loss + suppression_loss,
            generalization_loss=generalization_loss,
            suppression_loss=suppression_loss,
        )
        if best is None or _candidate_sort_key(candidate) < _candidate_sort_key(best):
            best = candidate
    return best


def _class_subset_positions(
    classes: Sequence[Sequence[int]],
    budget: int,
) -> Any:
    """Yield every class subset whose total row count fits ``budget``."""

    stack: list[tuple[int, int, tuple[int, ...]]] = [(0, 0, ())]
    while stack:
        index, used, selected = stack.pop()
        if index == len(classes):
            yield selected
            continue
        positions = tuple(classes[index])
        next_used = used + len(positions)
        if next_used <= budget:
            stack.append((index + 1, next_used, (*selected, *positions)))
        stack.append((index + 1, used, selected))


def _class_satisfies_k_l(
    equivalence_class: Mapping[str, Any],
    *,
    target_k: int,
    target_l: int,
    sensitive_attributes: Sequence[str],
    l_metric: str,
) -> bool:
    if int(equivalence_class.get("size", 0)) < target_k:
        return False
    l_diversity = _mapping(equivalence_class.get("l_diversity"))
    for attribute in sensitive_attributes:
        attribute_l = _mapping(l_diversity.get(attribute))
        if l_metric == "entropy":
            achieved_l = float(attribute_l.get("entropy", 0.0))
            required_l = math.log2(target_l)
        else:
            achieved_l = float(attribute_l.get("distinct", 0))
            required_l = float(target_l)
        if achieved_l + 1e-12 < required_l:
            return False
    return True


def _transform_record(
    record: Any,
    quasi_identifiers: Sequence[str],
    levels: Mapping[str, Sequence[_GeneralizationLevel]],
    node: Sequence[int],
    *,
    remove_direct_identifiers: bool,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name, value in record.fields.items():
        if remove_direct_identifiers and _field_is_direct_identifier(name):
            continue
        fields[name] = value

    for index, field in enumerate(quasi_identifiers):
        if remove_direct_identifiers and _field_is_direct_identifier(field):
            fields.pop(field, None)
            continue
        level = levels[field][node[index]]
        fields[field] = level.transform(_hierarchy_input(record.fields, field))
    return fields


def _hierarchy_input(fields: Mapping[str, Any], field: str) -> Any:
    if field not in fields:
        return _MISSING_QI
    value = fields[field]
    if value is None:
        return _NULL_QI
    if isinstance(value, str) and not value:
        return _EMPTY_QI
    return value


def _failing_positions(
    report: Mapping[str, Any],
    position_map: Sequence[int],
    *,
    target_k: int,
    target_l: int,
    target_t: float,
    sensitive_attributes: Sequence[str],
    l_metric: str = "distinct",
) -> set[int]:
    failing: set[int] = set()
    for cls in report.get("equivalence_classes", []):
        if not isinstance(cls, Mapping):
            continue
        if _class_satisfies(
            cls,
            target_k=target_k,
            target_l=target_l,
            target_t=target_t,
            sensitive_attributes=sensitive_attributes,
            l_metric=l_metric,
        ):
            continue
        for member in cls.get("members", []):
            parsed = _optional_int(member)
            if parsed is not None and 0 <= parsed < len(position_map):
                failing.add(position_map[parsed])
    return failing


def _report_satisfies(
    report: Mapping[str, Any],
    *,
    target_k: int,
    target_l: int,
    target_t: float,
    sensitive_attributes: Sequence[str],
    l_metric: str = "distinct",
) -> bool:
    if int(report.get("record_count", 0)) <= 0:
        return False
    if int(report.get("k", 0)) < target_k:
        return False
    for cls in report.get("equivalence_classes", []):
        if not isinstance(cls, Mapping):
            return False
        if not _class_satisfies(
            cls,
            target_k=target_k,
            target_l=target_l,
            target_t=target_t,
            sensitive_attributes=sensitive_attributes,
            l_metric=l_metric,
        ):
            return False
    return True


def _class_satisfies(
    cls: Mapping[str, Any],
    *,
    target_k: int,
    target_l: int,
    target_t: float,
    sensitive_attributes: Sequence[str],
    l_metric: str = "distinct",
) -> bool:
    if int(cls.get("size", 0)) < target_k:
        return False
    l_diversity = _mapping(cls.get("l_diversity"))
    t_closeness = _mapping(cls.get("t_closeness"))
    for attr in sensitive_attributes:
        attr_l = _mapping(l_diversity.get(attr))
        if l_metric == "entropy":
            achieved_l = float(attr_l.get("entropy", 0.0))
            required_l = math.log2(target_l)
        else:
            achieved_l = float(attr_l.get("distinct", 0))
            required_l = float(target_l)
        if achieved_l + 1e-12 < required_l:
            return False
        parsed_t = _optional_float(t_closeness.get(attr))
        if parsed_t is None or parsed_t > target_t + 1e-12:
            return False
    return True


def _generalization_loss(
    quasi_identifiers: Sequence[str],
    levels: Mapping[str, Sequence[_GeneralizationLevel]],
    node: Sequence[int],
) -> float:
    if not quasi_identifiers:
        return 0.0
    return sum(
        float(levels[field][node[index]].loss)
        for index, field in enumerate(quasi_identifiers)
    ) / len(quasi_identifiers)


def _suppressed_records(
    records: Sequence[Any],
    positions: Sequence[int],
    *,
    reason: str,
) -> list[dict[str, Any]]:
    return [
        {
            "record_index": int(position),
            "offset": int(position),
            "record_hash": _record_hash(records[position].fields),
            "reason": reason,
        }
        for position in positions
    ]


def _bound_report(
    records: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    suppressed_positions: Sequence[int],
    *,
    target_k: int,
    target_l: int,
    target_t: float,
    sensitive_attributes: Sequence[str],
    l_metric: str = "distinct",
) -> dict[str, Any]:
    class_by_member: dict[int, Mapping[str, Any]] = {}
    for cls in report.get("equivalence_classes", []):
        if not isinstance(cls, Mapping):
            continue
        for member in cls.get("members", []):
            parsed = _optional_int(member)
            if parsed is not None:
                class_by_member[parsed] = cls

    global_dist = {
        attr: _distribution(
            _typed_sensitive_distribution_value(record[attr])
            for record in records
            if attr in record and record.get(attr) is not None
        )
        for attr in sensitive_attributes
    }
    per_record = []
    violations = []
    target_bound = 1.0 / target_k
    for index, record in enumerate(records):
        cls = class_by_member.get(index)
        class_size = int(cls.get("size", 0)) if cls is not None else 0
        identity_bound = 1.0 / class_size if class_size else 1.0
        sensitive_bounds = _sensitive_bounds(
            records,
            cls,
            record,
            global_dist,
            sensitive_attributes,
        )
        joint_bound = min(
            [
                identity_bound,
                *[
                    bound["value_confidence_upper_bound"]
                    for bound in sensitive_bounds.values()
                ],
            ]
        )
        if identity_bound > target_bound + 1e-12:
            violations.append(
                {
                    "record_index": index,
                    "bound": identity_bound,
                    "target_bound": target_bound,
                }
            )
        per_record.append(
            {
                "record_index": index,
                "record_hash": _record_hash(record),
                "equivalence_class_size": class_size,
                "reidentification_upper_bound": identity_bound,
                "sensitive_attribute_upper_bounds": sensitive_bounds,
                "joint_sensitive_reidentification_upper_bound": joint_bound,
            }
        )

    max_bound = max(
        (item["reidentification_upper_bound"] for item in per_record),
        default=0.0,
    )
    l_ok = _l_targets_satisfied(
        report,
        sensitive_attributes,
        target_l,
        l_metric=l_metric,
    )
    t_ok = _t_targets_satisfied(report, sensitive_attributes, target_t)
    numeric_self_check = {
        "passed": not violations and l_ok and t_ok,
        "identity_bound_violations": violations,
        "l_diversity_satisfied": l_ok,
        "t_closeness_satisfied": t_ok,
    }
    return {
        "proof_sketch": (
            "Released records are partitioned by the published "
            "quasi-identifier key. Each class has size at least target_k, so "
            "any quasi-identifier-only linkage attack has probability at most "
            "1/class_size for each member, which is <= 1/target_k. The selected "
            "l-diversity variant and variational t-closeness are checked per "
            "class and reported separately from the identity bound."
        ),
        "target_reidentification_upper_bound": target_bound,
        "max_reidentification_upper_bound": max_bound,
        "target_k": int(target_k),
        "target_l": int(target_l),
        "l_metric": l_metric,
        "target_t": float(target_t),
        "suppressed_count": len(suppressed_positions),
        "numeric_self_check": numeric_self_check,
        "per_record": per_record,
    }


def _sensitive_bounds(
    records: Sequence[Mapping[str, Any]],
    cls: Mapping[str, Any] | None,
    record: Mapping[str, Any],
    global_dist: Mapping[str, Mapping[str, float]],
    sensitive_attributes: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if cls is None:
        return {}
    members = [
        parsed
        for parsed in (_optional_int(member) for member in cls.get("members", []))
        if parsed is not None and 0 <= parsed < len(records)
    ]
    class_size = len(members)
    result: dict[str, dict[str, Any]] = {}
    for attr in sensitive_attributes:
        values = [
            _typed_sensitive_distribution_value(records[index][attr])
            for index in members
        ]
        counts = Counter(values)
        record_value = _typed_sensitive_distribution_value(record[attr])
        observed = counts.get(record_value, 0) / class_size if class_size else 1.0
        attr_t = _optional_float(_mapping(cls.get("t_closeness")).get(attr)) or 0.0
        t_cap = min(1.0, global_dist.get(attr, {}).get(record_value, 0.0) + attr_t)
        distinct = int(
            _mapping(_mapping(cls.get("l_diversity")).get(attr)).get("distinct", 0)
        )
        result[attr] = {
            "distinct_values": distinct,
            "class_value_frequency": counts.get(record_value, 0),
            "value_confidence_upper_bound": min(observed, t_cap),
            "t_closeness_cap": t_cap,
        }
    return result


def _l_targets_satisfied(
    report: Mapping[str, Any],
    sensitive_attributes: Sequence[str],
    target_l: int,
    *,
    l_metric: str = "distinct",
) -> bool:
    for cls in report.get("equivalence_classes", []):
        if not isinstance(cls, Mapping):
            return False
        l_diversity = _mapping(cls.get("l_diversity"))
        for attr in sensitive_attributes:
            attr_l = _mapping(l_diversity.get(attr))
            if l_metric == "entropy":
                achieved_l = float(attr_l.get("entropy", 0.0))
                required_l = math.log2(target_l)
            else:
                achieved_l = float(attr_l.get("distinct", 0))
                required_l = float(target_l)
            if achieved_l + 1e-12 < required_l:
                return False
    return True


def _t_targets_satisfied(
    report: Mapping[str, Any],
    sensitive_attributes: Sequence[str],
    target_t: float,
) -> bool:
    for cls in report.get("equivalence_classes", []):
        if not isinstance(cls, Mapping):
            return False
        t_closeness = _mapping(cls.get("t_closeness"))
        for attr in sensitive_attributes:
            parsed = _optional_float(t_closeness.get(attr))
            if parsed is None or parsed > target_t + 1e-12:
                return False
    return True


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
