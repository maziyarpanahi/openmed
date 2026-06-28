"""External quasi-identifier linkage attack for de-identified records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openmed.risk.reid import _coerce_records, _normalize_qi_value, _profile_record


@dataclass(frozen=True)
class LinkageAttackResult:
    """Result of joining de-identified records to an external QI table."""

    unique_match_rate: float
    ambiguous_match_rate: float
    no_match_rate: float
    unique_matches: int
    ambiguous_matches: int
    no_matches: int
    record_count: int
    details: tuple[dict[str, Any], ...]

    def to_metric(self) -> dict[str, Any]:
        """Return a JSON-serializable BenchmarkReport metric payload."""

        details = [dict(detail) for detail in self.details]
        return {
            "linkage_unique_match_rate": float(self.unique_match_rate),
            "unique_match_rate": float(self.unique_match_rate),
            "ambiguous_match_rate": float(self.ambiguous_match_rate),
            "no_match_rate": float(self.no_match_rate),
            "unique_matches": int(self.unique_matches),
            "ambiguous_matches": int(self.ambiguous_matches),
            "no_matches": int(self.no_matches),
            "denominator": int(self.record_count),
            "details": details,
        }


def linkage_attack(
    deidentified_records: Any,
    quasi_id_table: Any,
    quasi_identifiers: Sequence[str] | None = None,
) -> LinkageAttackResult:
    """Run a Sweeney-style external quasi-identifier linkage attack.

    Args:
        deidentified_records: De-identified records in any shape accepted by
            ``risk_report``.
        quasi_id_table: External auxiliary table containing the same
            quasi-identifier fields or detectable quasi-identifier values.
        quasi_identifiers: Explicit quasi-identifier field names. When omitted,
            records are profiled with the same auto-detection path used by
            ``risk_report``.

    Returns:
        A result with unique, ambiguous, and no-match rates plus per-record
        match details. A record is counted as re-identified only when its QI
        key maps to exactly one row in the external table.
    """

    deidentified = _coerce_records(deidentified_records, source="deidentified")
    external_rows = _coerce_records(quasi_id_table, source="aux")

    external_index: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in external_rows:
        key, _ = _linkage_key(row, quasi_identifiers)
        if not key:
            continue
        external_index[key].append(
            {
                "external_record_index": row.index,
                "external_record_id": row.record_id,
            }
        )

    unique_matches = 0
    ambiguous_matches = 0
    no_matches = 0
    details: list[dict[str, Any]] = []
    for record in deidentified:
        key, json_key = _linkage_key(record, quasi_identifiers)
        matches = external_index.get(key, []) if key else []
        match_count = len(matches)

        detail: dict[str, Any] = {
            "record_index": record.index,
            "record_id": record.record_id,
            "key": json_key,
            "match_count": match_count,
        }
        if match_count == 1:
            unique_matches += 1
            detail["outcome"] = "unique"
            detail.update(matches[0])
        elif match_count > 1:
            ambiguous_matches += 1
            detail["outcome"] = "ambiguous"
        else:
            no_matches += 1
            detail["outcome"] = "no_match"
        details.append(detail)

    denominator = len(deidentified)
    return LinkageAttackResult(
        unique_match_rate=_rate(unique_matches, denominator),
        ambiguous_match_rate=_rate(ambiguous_matches, denominator),
        no_match_rate=_rate(no_matches, denominator),
        unique_matches=unique_matches,
        ambiguous_matches=ambiguous_matches,
        no_matches=no_matches,
        record_count=denominator,
        details=tuple(details),
    )


def _linkage_key(
    record: Any,
    quasi_identifiers: Sequence[str] | None,
) -> tuple[Any, list[Any]]:
    if quasi_identifiers:
        pairs = []
        for field in sorted(quasi_identifiers):
            normalized = _normalize_qi_value(field, record.fields.get(field, ""))
            if not normalized:
                return (), []
            pairs.append((field, normalized))
        return tuple(pairs), [[field, value] for field, value in pairs]

    profile_key = _profile_record(record).key
    json_key = [[category, list(values)] for category, values in profile_key]
    return profile_key, json_key


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


__all__ = ["LinkageAttackResult", "linkage_attack"]
