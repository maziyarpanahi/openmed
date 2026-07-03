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
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence

from openmed.core.audit import stable_hash

from .reid import (
    _coerce_records,
    _field_category,
    _field_is_direct_identifier,
    _normalize_qi_value,
    _profile_record,
)

__all__ = ["build_generalization_hierarchies", "enforce_kanon", "kanon_report"]

_SUPPORTED_L_METRICS = ("distinct", "entropy")
_SUPPORTED_T_DISTANCES = ("variational",)
_SUPPRESSED_VALUE = "*"
_SUPPORTED_USER_LEVEL_KEYS = frozenset({"name", "values", "default", "loss"})


@dataclass(frozen=True)
class _GeneralizationLevel:
    name: str
    loss: float
    transform: Callable[[Any], str]

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

    coerced = _coerce_records(records, source="deidentified")
    sensitive = sorted(dict.fromkeys(sensitive_attributes or []))

    members: defaultdict[Any, list[int]] = defaultdict(list)
    json_keys: dict[Any, list[Any]] = {}
    sensitive_values: dict[int, dict[str, str]] = {}

    for record in coerced:
        hash_key, json_key = _equivalence_key(record, quasi_identifiers)
        members[hash_key].append(record.index)
        json_keys.setdefault(hash_key, json_key)
        sensitive_values[record.index] = {
            attr: str(record.fields.get(attr)) for attr in sensitive
        }

    global_dist = {
        attr: _distribution(
            sensitive_values[idx][attr]
            for idx in sensitive_values
            if attr in sensitive_values[idx]
        )
        for attr in sensitive
    }

    classes = []
    for hash_key in members:
        indices = sorted(members[hash_key])
        per_class_l: dict[str, Any] = {}
        per_class_t: dict[str, float] = {}
        for attr in sensitive:
            values = [sensitive_values[idx][attr] for idx in indices]
            counts = Counter(values)
            per_class_l[attr] = {
                "distinct": len(counts),
                "entropy": _entropy(counts),
            }
            per_class_t[attr] = _variational_distance(
                _distribution(values), global_dist[attr]
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

    sizes = [cls["size"] for cls in classes]
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
        "quasi_identifiers": _reported_quasi_identifiers(quasi_identifiers, classes),
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

    coerced = _coerce_records(records, source="deidentified")
    qis = _resolve_quasi_identifier_fields(coerced, quasi_identifiers)
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

    _validate_policy(target_k, target_l, target_t, suppression_rate)
    coerced = _coerce_records(records, source="deidentified")
    qis = _resolve_quasi_identifier_fields(coerced, quasi_identifiers)
    sensitive = sorted(str(attr) for attr in sensitive_attributes or [])
    if target_l > 1 and not sensitive:
        raise ValueError("target_l > 1 requires at least one sensitive attribute")
    if target_t < 1.0 and not sensitive:
        raise ValueError("target_t < 1.0 requires at least one sensitive attribute")

    if not coerced:
        empty_report = kanon_report(
            [], quasi_identifiers=qis, sensitive_attributes=sensitive
        )
        return {
            "schema_version": 1,
            "record_count": 0,
            "released_count": 0,
            "suppressed_count": 0,
            "target_k": int(target_k),
            "target_l": int(target_l),
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
            },
            "bounds": _bound_report(
                [],
                empty_report,
                (),
                target_k=target_k,
                target_l=target_l,
                target_t=target_t,
                sensitive_attributes=sensitive,
            ),
        }

    levels = _build_hierarchy_levels(coerced, qis, hierarchies)
    budget = _suppression_budget(
        len(coerced),
        suppression_limit=suppression_limit,
        suppression_rate=suppression_rate,
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
    )
    return {
        "schema_version": 1,
        "record_count": len(coerced),
        "released_count": len(candidate.records),
        "suppressed_count": len(candidate.suppressed_positions),
        "suppression_limit": budget,
        "target_k": int(target_k),
        "target_l": int(target_l),
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
            (field, _normalize_qi_value(field, record.fields.get(field, "")))
            for field in sorted(quasi_identifiers)
        )
        return pairs, [[field, value] for field, value in pairs]

    profile_key = _profile_record(record).key
    json_key = [[category, list(values)] for category, values in profile_key]
    return profile_key, json_key


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
) -> None:
    if target_k < 1:
        raise ValueError("target_k must be >= 1")
    if target_l < 1:
        raise ValueError("target_l must be >= 1")
    if not 0.0 <= float(target_t) <= 1.0:
        raise ValueError("target_t must be between 0.0 and 1.0")
    if not 0.0 <= float(suppression_rate) <= 1.0:
        raise ValueError("suppression_rate must be between 0.0 and 1.0")


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
        value_map = {str(key): str(value) for key, value in dict(values or {}).items()}
        default = level.get("default")
        loss = _optional_float(level.get("loss"))
        if loss is None:
            loss = index / max_index

        def transform(
            value: Any,
            *,
            mapping: Mapping[str, str] = value_map,
            default_value: Any = default,
        ) -> str:
            exact = str(value)
            normalized = _normalize_qi_value(field, value)
            if exact in mapping:
                return mapping[exact]
            if normalized in mapping:
                return mapping[normalized]
            if default_value is not None:
                return str(default_value)
            return normalized

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
            _level("exact", 0.0, lambda value: _normalize_qi_value("age", value)),
            _level("age_5_year_band", 0.25, lambda value: _age_band(value, 5)),
            _level("age_10_year_band", 0.5, lambda value: _age_band(value, 10)),
            _level("age_20_year_band", 0.75, lambda value: _age_band(value, 20)),
            _level("suppressed", 1.0, lambda value: _SUPPRESSED_VALUE),
        )
    if category == "date":
        return (
            _level("exact", 0.0, lambda value: _normalize_qi_value("date", value)),
            _level("month", 0.25, _date_month),
            _level("year", 0.5, _date_year),
            _level("decade", 0.75, _date_decade),
            _level("suppressed", 1.0, lambda value: _SUPPRESSED_VALUE),
        )
    if category == "geography":
        return (
            _level("exact", 0.0, lambda value: _normalize_qi_value("geography", value)),
            _level("regional", 0.4, _geography_region),
            _level("broad_region", 0.7, _geography_broad_region),
            _level("suppressed", 1.0, lambda value: _SUPPRESSED_VALUE),
        )

    values = {
        _normalize_qi_value(field, record.fields.get(field, ""))
        for record in records
        if record.fields.get(field) is not None
    }
    has_prefix_signal = any(len(value) > 1 for value in values)
    if has_prefix_signal:
        return (
            _level("exact", 0.0, lambda value: _normalize_qi_value(field, value)),
            _level("prefix", 0.5, lambda value: _generic_prefix(field, value)),
            _level("suppressed", 1.0, lambda value: _SUPPRESSED_VALUE),
        )
    return (
        _level("exact", 0.0, lambda value: _normalize_qi_value(field, value)),
        _level("suppressed", 1.0, lambda value: _SUPPRESSED_VALUE),
    )


def _level(
    name: str,
    loss: float,
    transform: Callable[[Any], str],
) -> _GeneralizationLevel:
    return _GeneralizationLevel(name=name, loss=loss, transform=transform)


def _age_band(value: Any, width: int) -> str:
    normalized = _normalize_qi_value("age", value)
    parsed = _optional_int(normalized)
    if parsed is None:
        return _SUPPRESSED_VALUE
    lower = (parsed // width) * width
    upper = lower + width - 1
    return f"{lower}-{upper}"


def _date_parts(value: Any) -> tuple[int, int | None] | None:
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
    text = _normalize_qi_value("geography", value)
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 5:
        return f"{digits[:2]}***"
    region = _geography_region(text)
    return region[:1] + "*" if len(region) > 1 else region


def _generic_prefix(field: str, value: Any) -> str:
    normalized = _normalize_qi_value(field, value)
    if not normalized:
        return _SUPPRESSED_VALUE
    return normalized[:1] + "*"


def _suppression_budget(
    record_count: int,
    *,
    suppression_limit: int | None,
    suppression_rate: float,
) -> int:
    if suppression_limit is not None and suppression_limit < 0:
        raise ValueError("suppression_limit must be >= 0")
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
) -> _Candidate | None:
    field_order = tuple(quasi_identifiers)
    candidates: list[_Candidate] = []
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
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(
        key=lambda candidate: (
            candidate.information_loss,
            candidate.suppression_loss,
            sum(candidate.node),
            candidate.node,
        )
    )
    return candidates[0]


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
) -> _Candidate | None:
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
    suppressed: set[int] = set()

    while True:
        remaining_positions = [
            position
            for position in range(len(transformed))
            if position not in suppressed
        ]
        if not remaining_positions:
            return None
        remaining_records = [transformed[position] for position in remaining_positions]
        report = kanon_report(
            remaining_records,
            quasi_identifiers=quasi_identifiers,
            sensitive_attributes=sensitive_attributes,
        )
        failing_positions = _failing_positions(
            report,
            remaining_positions,
            target_k=target_k,
            target_l=target_l,
            target_t=target_t,
            sensitive_attributes=sensitive_attributes,
        )
        if not failing_positions:
            break
        before = len(suppressed)
        suppressed.update(failing_positions)
        if len(suppressed) > suppression_budget or len(suppressed) == before:
            return None

    final_positions = [
        position for position in range(len(transformed)) if position not in suppressed
    ]
    if not final_positions:
        return None
    final_records = tuple(transformed[position] for position in final_positions)
    final_report = kanon_report(
        final_records,
        quasi_identifiers=quasi_identifiers,
        sensitive_attributes=sensitive_attributes,
    )
    if not _report_satisfies(
        final_report,
        target_k=target_k,
        target_l=target_l,
        target_t=target_t,
        sensitive_attributes=sensitive_attributes,
    ):
        return None

    generalization_loss = _generalization_loss(quasi_identifiers, levels, node)
    suppression_loss = len(suppressed) / len(records) if records else 0.0
    information_loss = generalization_loss + suppression_loss
    return _Candidate(
        node=node,
        records=final_records,
        report=final_report,
        suppressed_positions=tuple(sorted(suppressed)),
        information_loss=information_loss,
        generalization_loss=generalization_loss,
        suppression_loss=suppression_loss,
    )


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
        fields[field] = level.transform(record.fields.get(field, ""))
    return fields


def _failing_positions(
    report: Mapping[str, Any],
    position_map: Sequence[int],
    *,
    target_k: int,
    target_l: int,
    target_t: float,
    sensitive_attributes: Sequence[str],
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
) -> bool:
    if int(cls.get("size", 0)) < target_k:
        return False
    l_diversity = _mapping(cls.get("l_diversity"))
    t_closeness = _mapping(cls.get("t_closeness"))
    for attr in sensitive_attributes:
        attr_l = _mapping(l_diversity.get(attr))
        if int(attr_l.get("distinct", 0)) < target_l:
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
            "record_hash": stable_hash(records[position].fields),
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
            str(record.get(attr))
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
                "record_hash": stable_hash(record),
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
    l_ok = _l_targets_satisfied(report, sensitive_attributes, target_l)
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
            "1/class_size for each member, which is <= 1/target_k. Distinct "
            "l-diversity and variational t-closeness are checked per class and "
            "reported as sensitive-attribute confidence caps."
        ),
        "target_reidentification_upper_bound": target_bound,
        "max_reidentification_upper_bound": max_bound,
        "target_k": int(target_k),
        "target_l": int(target_l),
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
        values = [str(records[index].get(attr)) for index in members]
        counts = Counter(values)
        record_value = str(record.get(attr))
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
) -> bool:
    for cls in report.get("equivalence_classes", []):
        if not isinstance(cls, Mapping):
            return False
        l_diversity = _mapping(cls.get("l_diversity"))
        for attr in sensitive_attributes:
            attr_l = _mapping(l_diversity.get(attr))
            if int(attr_l.get("distinct", 0)) < target_l:
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
