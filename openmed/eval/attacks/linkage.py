"""External quasi-identifier linkage attack for de-identified records."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openmed.risk.reid import (
    _coerce_records,
    _normalize_qi_value,
    _profile_record,
    build_longitudinal_corpus,
    longitudinal_attack_fingerprint,
    longitudinal_risk_report,
)


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


@dataclass(frozen=True)
class LongitudinalLinkageAttackResult:
    """Result of a same-patient cross-document linkage attack."""

    realized_success_rate: float
    reported_upper_bound: float
    bound_violated: bool
    successful_links: int
    patient_count: int
    details: tuple[dict[str, Any], ...]
    risk: dict[str, Any]

    def to_metric(self) -> dict[str, Any]:
        """Return a JSON-serializable longitudinal linkage metric payload."""

        return {
            "longitudinal_linkage_success_rate": float(self.realized_success_rate),
            "reported_upper_bound": float(self.reported_upper_bound),
            "bound_violated": bool(self.bound_violated),
            "successful_links": int(self.successful_links),
            "denominator": int(self.patient_count),
            "details": [dict(detail) for detail in self.details],
            "risk": dict(self.risk),
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


def longitudinal_linkage_attack(
    records: Any,
    *,
    hmac_key: bytes | str = "openmed-longitudinal-linkage-local-key",
) -> LongitudinalLinkageAttackResult:
    """Run a synthetic cross-document same-patient linkage attack.

    The attack groups each patient by the hashed longitudinal fingerprint
    reported by :func:`openmed.risk.longitudinal_risk_report`. A patient is
    counted as linked only when its non-empty fingerprint is unique across the
    corpus. The result is therefore an empirical success rate that should be at
    or below the report's conservative upper bound.
    """

    corpus = build_longitudinal_corpus(records, hmac_key=hmac_key)
    risk = longitudinal_risk_report(records, hmac_key=hmac_key)
    fingerprints = {
        patient.patient_pseudonym: longitudinal_attack_fingerprint(patient)
        for patient in corpus.patients
    }
    fingerprint_counts = Counter(
        fingerprint for fingerprint in fingerprints.values() if fingerprint
    )

    successful_links = 0
    details: list[dict[str, Any]] = []
    for patient in corpus.patients:
        fingerprint = fingerprints[patient.patient_pseudonym]
        candidate_count = fingerprint_counts[fingerprint] if fingerprint else 0
        linked = bool(fingerprint and candidate_count == 1)
        if linked:
            successful_links += 1
        details.append(
            {
                "patient_pseudonym": patient.patient_pseudonym,
                "document_count": patient.document_count,
                "fingerprint_size": len(fingerprint),
                "candidate_count": candidate_count,
                "outcome": "unique_link" if linked else "not_unique",
                "attack_fingerprint": [
                    {"category": category, "value_hash": value_hash}
                    for category, value_hash in fingerprint
                ],
            }
        )

    patient_count = corpus.patient_count
    success_rate = _rate(successful_links, patient_count)
    reported_upper_bound = float(risk.get("linkage_success_upper_bound", 0.0))
    return LongitudinalLinkageAttackResult(
        realized_success_rate=success_rate,
        reported_upper_bound=reported_upper_bound,
        bound_violated=success_rate > reported_upper_bound + 1e-12,
        successful_links=successful_links,
        patient_count=patient_count,
        details=tuple(details),
        risk=risk,
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


__all__ = [
    "LinkageAttackResult",
    "LongitudinalLinkageAttackResult",
    "linkage_attack",
    "longitudinal_linkage_attack",
]
