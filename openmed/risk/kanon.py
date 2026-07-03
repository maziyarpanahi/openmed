"""k-anonymity, l-diversity and t-closeness measurement for tabular records.

These are *measurement-only* disclosure-control metrics over already
de-identified records. They report on the realized privacy of an output; they
do NOT generalize, suppress or microaggregate to achieve a target k (that is
the ARX-style engine, a separate epic).

Quasi-identifier handling reuses :mod:`openmed.risk.reid` so equivalence-class
keys match :func:`openmed.risk.risk_report`: auto-detection uses the same
``_profile_record`` key, and an explicit ``quasi_identifiers`` list builds the
class key from those fields with the same value normalization.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .reid import _coerce_records, _normalize_qi_value, _profile_record

__all__ = ["kanon_report"]

_SUPPORTED_L_METRICS = ("distinct", "entropy")
_SUPPORTED_T_DISTANCES = ("variational",)


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
