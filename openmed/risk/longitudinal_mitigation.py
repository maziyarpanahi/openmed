"""Deterministic mitigations for longitudinal cross-document linkage risk."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, cast

from openmed.core.audit import stable_hash

from .reid import (
    _AGE_PATTERN,
    _DATE_PATTERN,
    _ID_KEYS,
    _PATIENT_KEY_FIELDS,
    _RARE_CONDITION_PATTERN,
    _SPAN_KEYS,
    _TEXT_KEYS,
    _field_category,
    _flatten_longitudinal_items,
    _hmac_digest,
    _record_id,
    _source_patient_key,
    _span_category,
    _span_label,
    longitudinal_risk_report,
)

LONGITUDINAL_MITIGATION_SCHEMA_VERSION = "openmed.longitudinal_mitigation.v1"
_MITIGATION_SPAN_KEYS = (*_SPAN_KEYS, "audit_spans")


@dataclass(frozen=True)
class LongitudinalMitigationPolicy:
    """Controls for balancing longitudinal consistency and uniqueness.

    A cohort size of one applies a distinct deterministic surrogate or
    trajectory offset to every note. Larger sizes preserve consistency within
    consecutive note cohorts while reducing consistency across the release.
    """

    linkage_ceiling: float = 0.0
    surrogate_cohort_size: int = 1
    age_cohort_size: int = 1
    age_perturbation_years: int = 20
    date_cohort_size: int = 1
    date_perturbation_days: int = 180
    suppress_rare_attributes: bool = True

    def __post_init__(self) -> None:
        _validate_rate(self.linkage_ceiling, "linkage_ceiling")
        for name in ("surrogate_cohort_size", "age_cohort_size", "date_cohort_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        for name in ("age_perturbation_years", "date_perturbation_days"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")

    def to_dict(self) -> dict[str, Any]:
        """Return the public, secret-free policy representation."""

        return {
            "age_cohort_size": self.age_cohort_size,
            "age_perturbation_years": self.age_perturbation_years,
            "date_cohort_size": self.date_cohort_size,
            "date_perturbation_days": self.date_perturbation_days,
            "linkage_ceiling": float(self.linkage_ceiling),
            "suppress_rare_attributes": self.suppress_rare_attributes,
            "surrogate_cohort_size": self.surrogate_cohort_size,
        }


@dataclass(frozen=True)
class LongitudinalMitigationAction:
    """One hashes-and-offsets-only mitigation audit entry."""

    patient_pseudonym: str
    note_hash: str
    category: str
    operation: str
    source: str
    before_hash: str
    after_hash: str | None = None
    start: int | None = None
    end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready action without raw source or replacement values."""

        payload: dict[str, Any] = {
            "before_hash": self.before_hash,
            "category": self.category,
            "note_hash": self.note_hash,
            "operation": self.operation,
            "patient_pseudonym": self.patient_pseudonym,
            "source": self.source,
        }
        if self.after_hash is not None:
            payload["after_hash"] = self.after_hash
        if self.start is not None:
            payload["start"] = self.start
        if self.end is not None:
            payload["end"] = self.end
        return payload


@dataclass(frozen=True)
class LongitudinalMitigationResult:
    """Mitigated records plus a separately serializable privacy-safe report."""

    records: tuple[dict[str, Any], ...]
    before_report: Mapping[str, Any]
    after_report: Mapping[str, Any]
    actions: tuple[LongitudinalMitigationAction, ...]
    policy: LongitudinalMitigationPolicy

    @property
    def meets_ceiling(self) -> bool:
        """Return whether the mitigated release is within the configured bound."""

        return (
            float(self.after_report["linkage_success_upper_bound"])
            <= self.policy.linkage_ceiling
        )

    def to_report_dict(self) -> dict[str, Any]:
        """Return hashes, offsets, counts, and scores only.

        The de-identified release records are intentionally excluded. Callers
        should serialize this report, not the containing result object.
        """

        before_bound = float(self.before_report["linkage_success_upper_bound"])
        after_bound = float(self.after_report["linkage_success_upper_bound"])
        payload: dict[str, Any] = {
            "action_count": len(self.actions),
            "actions": [action.to_dict() for action in self.actions],
            "after_linkage_success_upper_bound": after_bound,
            "before_linkage_success_upper_bound": before_bound,
            "document_count": int(self.after_report["document_count"]),
            "linkage_ceiling": float(self.policy.linkage_ceiling),
            "meets_ceiling": self.meets_ceiling,
            "mitigated_patient_count": len(
                {action.patient_pseudonym for action in self.actions}
            ),
            "patient_count": int(self.after_report["patient_count"]),
            "policy": self.policy.to_dict(),
            "schema_version": LONGITUDINAL_MITIGATION_SCHEMA_VERSION,
        }
        payload["report_hash"] = stable_hash(payload)
        return payload


def mitigate_longitudinal_linkage(
    records: Any,
    *,
    hmac_key: bytes | str,
    policy: LongitudinalMitigationPolicy | None = None,
    patient_key_fields: Sequence[str] = _PATIENT_KEY_FIELDS,
) -> LongitudinalMitigationResult:
    """Mitigate only patient cohorts whose linkage bound exceeds the ceiling.

    The returned records are detached copies. The audit report deliberately
    excludes record content and contains HMAC digests for every changed value.
    """

    resolved_policy = policy or LongitudinalMitigationPolicy()
    flattened = _copy_flat_records(records, patient_key_fields)
    before = longitudinal_risk_report(
        flattened,
        hmac_key=hmac_key,
        patient_key_fields=patient_key_fields,
    )
    targeted = {
        str(patient["patient_pseudonym"])
        for patient in before["patient_risks"]
        if float(patient["linkage_upper_bound"]) > resolved_policy.linkage_ceiling
    }

    actions: list[LongitudinalMitigationAction] = []
    note_ordinals: dict[str, int] = {}
    for note_index, record in enumerate(flattened):
        patient_pseudonym = _patient_pseudonym(
            record,
            note_index,
            hmac_key=hmac_key,
            patient_key_fields=patient_key_fields,
        )
        note_ordinal = note_ordinals.get(patient_pseudonym, 0)
        note_ordinals[patient_pseudonym] = note_ordinal + 1
        if patient_pseudonym not in targeted:
            continue
        note_hash = _note_hash(record, note_index, hmac_key)
        _mitigate_record(
            record,
            patient_pseudonym=patient_pseudonym,
            note_hash=note_hash,
            note_ordinal=note_ordinal,
            hmac_key=hmac_key,
            patient_key_fields=patient_key_fields,
            policy=resolved_policy,
            actions=actions,
        )

    after = longitudinal_risk_report(
        flattened,
        hmac_key=hmac_key,
        patient_key_fields=patient_key_fields,
    )
    return LongitudinalMitigationResult(
        records=tuple(flattened),
        before_report=before,
        after_report=after,
        actions=tuple(actions),
        policy=resolved_policy,
    )


def _copy_flat_records(
    records: Any,
    patient_key_fields: Sequence[str],
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for item in _flatten_longitudinal_items(records, patient_key_fields):
        if not isinstance(item, Mapping):
            raise TypeError("longitudinal mitigation records must be mappings")
        copied.append(copy.deepcopy(dict(item)))
    return copied


def _patient_pseudonym(
    record: Mapping[str, Any],
    note_index: int,
    *,
    hmac_key: bytes | str,
    patient_key_fields: Sequence[str],
) -> str:
    source_key = (
        _source_patient_key(record, patient_key_fields)
        or _record_id(record)
        or f"note:{note_index}"
    )
    return _hmac_digest(hmac_key, f"patient:{source_key}")


def _note_hash(
    record: Mapping[str, Any],
    note_index: int,
    hmac_key: bytes | str,
) -> str:
    return _hmac_digest(hmac_key, f"note:{_record_id(record) or note_index}")


def _mitigate_record(
    record: dict[str, Any],
    *,
    patient_pseudonym: str,
    note_hash: str,
    note_ordinal: int,
    hmac_key: bytes | str,
    patient_key_fields: Sequence[str],
    policy: LongitudinalMitigationPolicy,
    actions: list[LongitudinalMitigationAction],
) -> None:
    context = _ActionContext(patient_pseudonym, note_hash, hmac_key, actions)
    age_offset = _cohort_offset(
        note_ordinal,
        policy.age_cohort_size,
        policy.age_perturbation_years,
    )
    date_offset = _cohort_offset(
        note_ordinal,
        policy.date_cohort_size,
        policy.date_perturbation_days,
    )

    for key in _TEXT_KEYS:
        value = record.get(key)
        if not isinstance(value, str):
            continue
        record[key] = _mitigate_text(
            value,
            age_offset=age_offset,
            date_offset=date_offset,
            suppress_rare_attributes=policy.suppress_rare_attributes,
            source="text",
            context=context,
        )

    for key in tuple(record):
        if key in _TEXT_KEYS or key in _MITIGATION_SPAN_KEYS or key in _ID_KEYS:
            continue
        if key in patient_key_fields:
            continue
        value = record[key]
        normalized_name = _name_key(key)
        if "surrogate" in normalized_name and value is not None:
            record[key] = _diversified_surrogate(
                value,
                patient_pseudonym=patient_pseudonym,
                note_ordinal=note_ordinal,
                cohort_size=policy.surrogate_cohort_size,
                hmac_key=hmac_key,
            )
            context.add("stable_surrogate", "diversify", "field", value, record[key])
            continue

        category = _field_category(key)
        if category == "rare_condition" and policy.suppress_rare_attributes:
            context.add(category, "suppress", "field", value)
            del record[key]
        elif category == "age" and age_offset and value is not None:
            shifted = _shift_age_value(value, age_offset)
            if shifted != value:
                record[key] = shifted
                context.add(category, "perturb", "field", value, shifted)
        elif category == "date" and date_offset and value is not None:
            shifted = _shift_date_value(value, date_offset)
            if shifted != value:
                record[key] = shifted
                context.add(category, "perturb", "field", value, shifted)

    for key in _MITIGATION_SPAN_KEYS:
        spans = record.get(key)
        if isinstance(spans, Mapping):
            mitigated = _mitigate_spans(
                [spans],
                patient_pseudonym=patient_pseudonym,
                note_ordinal=note_ordinal,
                age_offset=age_offset,
                date_offset=date_offset,
                policy=policy,
                context=context,
            )
            record[key] = mitigated[0] if mitigated else []
        elif _is_sequence(spans):
            record[key] = _mitigate_spans(
                cast(Sequence[Any], spans),
                patient_pseudonym=patient_pseudonym,
                note_ordinal=note_ordinal,
                age_offset=age_offset,
                date_offset=date_offset,
                policy=policy,
                context=context,
            )


@dataclass(frozen=True)
class _ActionContext:
    patient_pseudonym: str
    note_hash: str
    hmac_key: bytes | str
    actions: list[LongitudinalMitigationAction]

    def add(
        self,
        category: str,
        operation: str,
        source: str,
        before: Any,
        after: Any | None = None,
        *,
        start: int | None = None,
        end: int | None = None,
    ) -> None:
        self.actions.append(
            LongitudinalMitigationAction(
                patient_pseudonym=self.patient_pseudonym,
                note_hash=self.note_hash,
                category=category,
                operation=operation,
                source=source,
                before_hash=_hmac_digest(self.hmac_key, f"before:{before}"),
                after_hash=(
                    _hmac_digest(self.hmac_key, f"after:{after}")
                    if after is not None
                    else None
                ),
                start=start,
                end=end,
            )
        )


def _mitigate_text(
    text: str,
    *,
    age_offset: int,
    date_offset: int,
    suppress_rare_attributes: bool,
    source: str,
    context: _ActionContext,
) -> str:
    replacements: list[tuple[int, int, str, str, str]] = []
    if age_offset:
        for match in _AGE_PATTERN.finditer(text):
            start, end = match.span(1)
            shifted = str(_bounded_age(int(match.group(1)), age_offset))
            replacements.append((start, end, shifted, "age", "perturb"))
    if date_offset:
        for match in _DATE_PATTERN.finditer(text):
            shifted_date = _shift_date_text(match.group(0), date_offset)
            if shifted_date is not None:
                replacements.append(
                    (match.start(), match.end(), shifted_date, "date", "perturb")
                )
    if suppress_rare_attributes:
        for match in _RARE_CONDITION_PATTERN.finditer(text):
            replacements.append(
                (
                    match.start(),
                    match.end(),
                    "*" * len(match.group(0)),
                    "rare_condition",
                    "suppress",
                )
            )

    rendered = text
    for start, end, replacement, category, operation in sorted(
        replacements,
        reverse=True,
    ):
        before = rendered[start:end]
        replacement = _fit_width(replacement, end - start)
        rendered = rendered[:start] + replacement + rendered[end:]
        context.add(
            category,
            operation,
            source,
            before,
            replacement,
            start=start,
            end=end,
        )
    return rendered


def _mitigate_spans(
    spans: Sequence[Any],
    *,
    patient_pseudonym: str,
    note_ordinal: int,
    age_offset: int,
    date_offset: int,
    policy: LongitudinalMitigationPolicy,
    context: _ActionContext,
) -> list[Any]:
    mitigated: list[Any] = []
    for item in spans:
        if not isinstance(item, Mapping):
            mitigated.append(copy.deepcopy(item))
            continue
        span = copy.deepcopy(dict(item))
        category = _span_category(span)
        source = "span"
        if category == "rare_condition" and policy.suppress_rare_attributes:
            context.add(category, "suppress", source, _span_audit_value(span))
            continue
        surrogate = span.get("surrogate")
        if surrogate is not None:
            diversified = _diversified_surrogate(
                surrogate,
                patient_pseudonym=patient_pseudonym,
                note_ordinal=note_ordinal,
                cohort_size=policy.surrogate_cohort_size,
                hmac_key=context.hmac_key,
                label=_span_label(span),
            )
            span["surrogate"] = diversified
            context.add("stable_surrogate", "diversify", source, surrogate, diversified)
        if category == "age" and age_offset:
            _shift_span_values(span, category, age_offset, source, context)
        elif category == "date" and date_offset:
            _shift_span_values(span, category, date_offset, source, context)
        mitigated.append(span)
    return mitigated


def _shift_span_values(
    span: dict[str, Any],
    category: str,
    offset: int,
    source: str,
    context: _ActionContext,
) -> None:
    for key in ("text", "word", "value", "surface", "replacement"):
        value = span.get(key)
        if value is None:
            continue
        shifted = (
            _shift_age_value(value, offset)
            if category == "age"
            else _shift_date_value(value, offset)
        )
        if shifted == value:
            continue
        span[key] = shifted
        context.add(category, "perturb", source, value, shifted)


def _diversified_surrogate(
    value: Any,
    *,
    patient_pseudonym: str,
    note_ordinal: int,
    cohort_size: int,
    hmac_key: bytes | str,
    label: str = "field",
) -> str:
    cohort = note_ordinal // cohort_size
    digest = _hmac_digest(
        hmac_key,
        f"diversify:{patient_pseudonym}:{cohort}:{label}:{value}",
    )
    return f"surrogate-{digest.removeprefix('hmac-sha256:')[:20]}"


def _cohort_offset(note_ordinal: int, cohort_size: int, magnitude: int) -> int:
    if magnitude == 0:
        return 0
    cohort = note_ordinal // cohort_size
    return magnitude if cohort % 2 == 0 else -magnitude


def _shift_age_value(value: Any, offset: int) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return _bounded_age(value, offset)
    match = re.search(r"\b(?:1[01]\d|[1-9]?\d)\b", str(value))
    if match is None:
        return value
    shifted = str(_bounded_age(int(match.group(0)), offset))
    return str(value)[: match.start()] + shifted + str(value)[match.end() :]


def _bounded_age(value: int, offset: int) -> int:
    upper_bound = 99 if value < 100 else 119
    return max(0, min(upper_bound, value + offset))


def _shift_date_value(value: Any, offset_days: int) -> Any:
    if isinstance(value, datetime):
        return value + timedelta(days=offset_days)
    if isinstance(value, date):
        return value + timedelta(days=offset_days)
    shifted = _shift_date_text(str(value), offset_days)
    return shifted if shifted is not None else value


def _shift_date_text(value: str, offset_days: int) -> str | None:
    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%m/%d/%y",
        "%m-%d-%y",
        "%b %d, %Y",
        "%b %d %Y",
        "%B %d, %Y",
        "%B %d %Y",
    )
    for date_format in formats:
        try:
            shifted = datetime.strptime(value, date_format) + timedelta(
                days=offset_days
            )
        except ValueError:
            continue
        return shifted.strftime(date_format)
    return None


def _fit_width(value: str, width: int) -> str:
    if len(value) == width:
        return value
    if len(value) < width:
        return value.rjust(width, "0")
    return value[-width:]


def _span_audit_value(span: Mapping[str, Any]) -> str:
    for key in ("surrogate", "replacement", "text", "word", "value", "surface"):
        value = span.get(key)
        if value is not None:
            return str(value)
    return _span_label(span)


def _name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _validate_rate(value: float, name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1]")


__all__ = [
    "LONGITUDINAL_MITIGATION_SCHEMA_VERSION",
    "LongitudinalMitigationAction",
    "LongitudinalMitigationPolicy",
    "LongitudinalMitigationResult",
    "mitigate_longitudinal_linkage",
]
