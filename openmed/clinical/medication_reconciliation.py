"""Deterministic, privacy-conscious medication reconciliation helpers.

The functions in this module compare already normalized medication candidates.
They do not ground medication names, call a remote vocabulary, infer a dose,
or make a clinical decision.  A pair is merged only when its name and
available regimen evidence meet the configured threshold.  Known dose, route,
identity, and overlapping temporal-status conflicts always abstain.

The in-memory candidate keeps the normalized values needed by a caller to
render a review UI.  ``to_dict`` methods are audit-safe: candidate identities,
names, dose values, and source identifiers are represented by stable hashes,
not copied into reports or logs.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import Literal

from .medication_sig import normalize_dose

MEDICATION_RECONCILIATION_SCHEMA_VERSION = 1
MEDICATION_RECONCILIATION_ADVISORY = (
    "Medication reconciliation is deterministic support tooling for clinician "
    "review; it does not establish medication identity, treatment, or safety."
)

DecisionStatus = Literal["matched", "abstained"]
FeatureStatus = Literal["match", "mismatch", "partial", "unknown"]
GroupStatus = Literal["merged", "singleton"]

_FEATURES = ("name", "dose", "route", "temporal")
_DEFAULT_WEIGHTS = {
    "name": 0.45,
    "dose": 0.25,
    "route": 0.15,
    "temporal": 0.15,
}
_ROUTE_ALIASES = {
    "by mouth": "oral",
    "enteral": "enteral",
    "gastrostomy": "enteral",
    "im": "intramuscular",
    "inhalation": "inhaled",
    "inhaled": "inhaled",
    "intramuscular": "intramuscular",
    "intranasal": "intranasal",
    "intravenous": "intravenous",
    "iv": "intravenous",
    "nasal": "intranasal",
    "oral": "oral",
    "per os": "oral",
    "po": "oral",
    "sc": "subcutaneous",
    "sq": "subcutaneous",
    "sub q": "subcutaneous",
    "sub-q": "subcutaneous",
    "subcutaneous": "subcutaneous",
    "sublingual": "sublingual",
    "topical": "topical",
    "transdermal": "transdermal",
    "buccal": "buccal",
    "ophthalmic": "ophthalmic",
    "otic": "otic",
    "rectal": "rectal",
    "vaginal": "vaginal",
}
_CURRENT_TEMPORAL_LABELS = frozenset({"active", "current", "ongoing", "recent"})
_STOPPED_TEMPORAL_LABELS = frozenset(
    {"discontinued", "ended", "inactive", "stopped", "past"}
)
_WHITESPACE_RE = re.compile(r"\s+")
_NON_NAME_RE = re.compile(r"[^\w]+", re.UNICODE)


def _stable_hash(value: object) -> str:
    """Return a stable identifier suitable for an audit record."""

    return "sha256:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _normalize_name(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("medication name must be a string")
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = _NON_NAME_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def _normalize_optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string when provided")
    normalized = _WHITESPACE_RE.sub(" ", value.strip())
    return normalized or None


def _normalize_code(value: object) -> str | None:
    normalized = _normalize_optional_text(value, "medication code")
    return normalized.casefold() if normalized is not None else None


def _normalize_route(value: object) -> str | None:
    normalized = _normalize_optional_text(value, "route")
    if normalized is None:
        return None
    key = unicodedata.normalize("NFKC", normalized).casefold()
    key = _WHITESPACE_RE.sub(" ", key)
    return _ROUTE_ALIASES.get(key, key)


def _normalize_temporal_label(value: object) -> str | None:
    normalized = _normalize_optional_text(value, "temporal label")
    if normalized is None:
        return None
    return unicodedata.normalize("NFKC", normalized).casefold()


def _coerce_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        if "T" in text or " " in text:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        return date.fromisoformat(text)
    except ValueError:
        return None


def _field(source: object, *names: str, default: object = None) -> object:
    if isinstance(source, Mapping):
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
        return default
    for name in names:
        value = getattr(source, name, None)
        if value is not None:
            return value
    return default


@dataclass(frozen=True, repr=False)
class MedicationReconciliationPolicy:
    """Weights and conservative gates used by medication comparison.

    The default score is the weighted sum of four independent features.  An
    unknown feature contributes zero, so a name-only match cannot reach the
    default threshold.  Weights are normalized to sum to one at comparison
    time, which keeps custom policies in the same ``0.0``-to-``1.0`` range.
    """

    name_weight: float = _DEFAULT_WEIGHTS["name"]
    dose_weight: float = _DEFAULT_WEIGHTS["dose"]
    route_weight: float = _DEFAULT_WEIGHTS["route"]
    temporal_weight: float = _DEFAULT_WEIGHTS["temporal"]
    match_threshold: float = 0.80
    temporal_gap_days: float = 365.0
    dose_relative_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        weights = (
            self.name_weight,
            self.dose_weight,
            self.route_weight,
            self.temporal_weight,
        )
        if any(
            isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
            for weight in weights
        ):
            raise ValueError(
                "medication feature weights must be finite and non-negative"
            )
        if sum(float(weight) for weight in weights) <= 0.0:
            raise ValueError("at least one medication feature weight is required")
        if (
            isinstance(self.match_threshold, bool)
            or not math.isfinite(float(self.match_threshold))
            or not 0.0 <= float(self.match_threshold) <= 1.0
        ):
            raise ValueError("match_threshold must be between 0 and 1")
        if (
            isinstance(self.temporal_gap_days, bool)
            or not math.isfinite(float(self.temporal_gap_days))
            or float(self.temporal_gap_days) <= 0.0
        ):
            raise ValueError("temporal_gap_days must be greater than zero")
        if (
            isinstance(self.dose_relative_tolerance, bool)
            or not math.isfinite(float(self.dose_relative_tolerance))
            or float(self.dose_relative_tolerance) < 0.0
        ):
            raise ValueError("dose_relative_tolerance must be non-negative")

    @property
    def weights(self) -> Mapping[str, float]:
        """Return normalized feature weights in a stable key order."""

        raw = {
            "name": float(self.name_weight),
            "dose": float(self.dose_weight),
            "route": float(self.route_weight),
            "temporal": float(self.temporal_weight),
        }
        total = sum(raw.values())
        return MappingProxyType({key: raw[key] / total for key in _FEATURES})


@dataclass(frozen=True, repr=False)
class MedicationReconciliationCandidate:
    """A normalized medication mention supplied by an upstream extractor.

    ``name`` should be a normalized medication surface or canonical display.
    ``dose`` accepts the same local formats as :func:`normalize_dose`, such as
    ``"500 mg"`` or ``{"value": 500, "unit": "mg"}``.  Dates must be ISO
    strings, ``date`` objects, or ``datetime`` objects.  No field is fetched
    from a remote vocabulary.
    """

    candidate_id: str
    name: str | None = field(default=None, repr=False)
    dose: object | None = field(default=None, repr=False)
    route: str | None = None
    temporal_start: object | None = field(default=None, repr=False)
    temporal_end: object | None = field(default=None, repr=False)
    temporal_label: str | None = None
    medication_code: str | None = field(default=None, repr=False)
    code_system: str | None = field(default=None, repr=False)
    source_id: str | None = field(default=None, repr=False)
    _normalized_name: str = field(init=False, repr=False, compare=False)
    _normalized_code: str | None = field(init=False, repr=False, compare=False)
    _normalized_system: str | None = field(init=False, repr=False, compare=False)
    _normalized_route: str | None = field(init=False, repr=False, compare=False)
    _dose: Mapping[str, object] = field(init=False, repr=False, compare=False)
    _temporal_start: date | None = field(init=False, repr=False, compare=False)
    _temporal_end: date | None = field(init=False, repr=False, compare=False)
    _temporal_invalid: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        object.__setattr__(self, "candidate_id", self.candidate_id.strip())
        normalized_name = "" if self.name is None else _normalize_name(self.name)
        normalized_code = _normalize_code(self.medication_code)
        if not normalized_name and not normalized_code:
            raise ValueError("candidate must include a medication name or code")
        object.__setattr__(self, "_normalized_name", normalized_name)
        object.__setattr__(self, "_normalized_code", normalized_code)
        object.__setattr__(
            self,
            "_normalized_system",
            _normalize_code(self.code_system),
        )
        object.__setattr__(self, "_normalized_route", _normalize_route(self.route))
        object.__setattr__(self, "_dose", _normalize_dose(self.dose))
        object.__setattr__(
            self,
            "temporal_label",
            _normalize_temporal_label(self.temporal_label),
        )

        start = _coerce_date(self.temporal_start)
        end = _coerce_date(self.temporal_end)
        invalid_temporal = (self.temporal_start is not None and start is None) or (
            self.temporal_end is not None and end is None
        )
        if start is not None and end is not None and end < start:
            invalid_temporal = True
            start = None
            end = None
        object.__setattr__(self, "_temporal_start", start)
        object.__setattr__(self, "_temporal_end", end)
        object.__setattr__(self, "_temporal_invalid", invalid_temporal)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object] | object,
        *,
        default_id: str = "candidate-1",
    ) -> "MedicationReconciliationCandidate":
        """Build a candidate from common extractor mapping/object fields."""

        if isinstance(value, cls):
            return value
        candidate_id = _field(
            value,
            "candidate_id",
            "mention_id",
            "id",
            default=default_id,
        )
        name = _field(
            value,
            "normalized_name",
            "canonical_name",
            "medication_name",
            "name",
            "drug",
            "text",
        )
        medication_code = _field(
            value,
            "medication_code",
            "concept_code",
            "rxnorm_code",
            "rx_cui",
            "code",
        )
        code_system = _field(
            value,
            "code_system",
            "concept_system",
            "medication_system",
            "rxnorm_system",
            "system",
        )
        dose = _field(value, "dose", "normalized_dose")
        dose_unit = _field(value, "dose_unit", "dose_units", "unit", "units")
        if dose is not None and dose_unit is not None and not isinstance(dose, Mapping):
            dose = {"value": dose, "unit": dose_unit}
        if dose is None:
            dose_value = _field(value, "dose_value", "dose_amount", "amount")
            if dose_value is not None:
                dose = {"value": dose_value, "unit": dose_unit}
        route = _field(value, "route", "administration_route")

        temporal = _field(value, "temporal", "temporal_evidence", "temporal_window")
        temporal_start = _field(
            value,
            "temporal_start",
            "effective_start",
            "start_date",
            "effective_date",
            "event_date",
            "date",
            "reference_date",
            "as_of",
            "document_date",
        )
        temporal_end = _field(value, "temporal_end", "effective_end", "end_date")
        temporal_label = _field(
            value,
            "temporal_label",
            "temporality",
            "temporal_status",
            "medication_status",
        )
        if isinstance(temporal, Mapping):
            temporal_start = _field(
                temporal,
                "start",
                "from",
                "effective_start",
                "date",
                default=temporal_start,
            )
            temporal_end = _field(
                temporal,
                "end",
                "to",
                "effective_end",
                default=temporal_end,
            )
            temporal_label = _field(
                temporal,
                "label",
                "temporality",
                "status",
                default=temporal_label,
            )
        elif temporal is not None and temporal_label is None:
            temporal_label = temporal

        return cls(
            candidate_id=candidate_id,
            name=name,
            dose=dose,
            route=route,
            temporal_start=temporal_start,
            temporal_end=temporal_end,
            temporal_label=temporal_label,
            medication_code=medication_code,
            code_system=code_system,
            source_id=_field(
                value,
                "source_id",
                "source_document_id",
                "document_id",
            ),
        )

    @property
    def normalized_name(self) -> str:
        """Return the normalized name kept in memory for comparison."""

        return self._normalized_name

    @property
    def normalized_code(self) -> str | None:
        """Return the normalized coded identity, when supplied."""

        return self._normalized_code

    @property
    def normalized_system(self) -> str | None:
        """Return the normalized coding system, when supplied."""

        return self._normalized_system

    @property
    def normalized_route(self) -> str | None:
        """Return the controlled route value, when supplied."""

        return self._normalized_route

    @property
    def dose_recognized(self) -> bool:
        """Return whether the supplied dose was locally normalized."""

        return bool(self._dose["recognized"])

    @property
    def temporal_start_date(self) -> date | None:
        """Return the normalized temporal start date."""

        return self._temporal_start

    @property
    def temporal_end_date(self) -> date | None:
        """Return the normalized temporal end date."""

        return self._temporal_end

    @property
    def temporal_invalid(self) -> bool:
        """Return whether a supplied temporal value could not be parsed."""

        return self._temporal_invalid

    @property
    def name_hash(self) -> str:
        """Return a stable hash for safe provenance records."""

        return _stable_hash(self._normalized_name or self._normalized_code)

    def to_dict(self) -> dict[str, object]:
        """Return a PHI-safe candidate record without raw medication values."""

        dose_hash = None
        if self.dose is not None:
            dose_hash = _stable_hash(
                (
                    self._dose.get("canonical_value"),
                    self._dose.get("canonical_unit"),
                    tuple(sorted(self._dose.get("dimension", {}).items())),
                )
            )
        return {
            "candidate_id_hash": _stable_hash(self.candidate_id),
            "name_hash": self.name_hash,
            "code_hash": (
                _stable_hash((self._normalized_system, self._normalized_code))
                if self._normalized_code is not None
                else None
            ),
            "route": self._normalized_route,
            "dose_hash": dose_hash,
            "dose_recognized": self.dose_recognized,
            "temporal_start": (
                self._temporal_start.isoformat() if self._temporal_start else None
            ),
            "temporal_end": (
                self._temporal_end.isoformat() if self._temporal_end else None
            ),
            "temporal_label": _normalize_temporal_label(self.temporal_label),
            "temporal_invalid": self._temporal_invalid,
            "source_id_hash": (
                _stable_hash(self.source_id) if self.source_id is not None else None
            ),
        }

    def __repr__(self) -> str:
        return (
            "MedicationReconciliationCandidate("
            f"candidate_id_hash={_stable_hash(self.candidate_id)!r}, "
            f"name_hash={self.name_hash!r})"
        )


# A short alias makes the module convenient without colliding with the legacy
# ``openmed.clinical.MedicationCandidate`` exported by medication_sig.py.
NormalizedMedicationCandidate = MedicationReconciliationCandidate


def _normalize_dose(value: object | None) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType(
            {"recognized": False, "canonical_value": None, "canonical_unit": None}
        )
    dose_value = value
    dose_unit: object | None = None
    if isinstance(value, Mapping):
        if "canonical_value" in value and "canonical_unit" in value:
            dose_value = value.get("canonical_value")
            dose_unit = value.get("canonical_unit")
        elif "value" in value or "amount" in value or "magnitude" in value:
            dose_value = value.get(
                "value",
                value.get("amount", value.get("magnitude")),
            )
            dose_unit = value.get("unit", value.get("units"))
    try:
        result = normalize_dose(dose_value, dose_unit)
    except (AttributeError, TypeError, ValueError):
        return MappingProxyType(
            {"recognized": False, "canonical_value": None, "canonical_unit": None}
        )
    recognized = result.get("recognized") is True
    canonical_value = result.get("canonical_value")
    if recognized and (
        not isinstance(canonical_value, (int, float))
        or isinstance(canonical_value, bool)
        or not math.isfinite(float(canonical_value))
    ):
        recognized = False
    dimension = result.get("dimension", {})
    if not isinstance(dimension, Mapping):
        dimension = {}
    normalized: dict[str, object] = {
        "recognized": recognized,
        "canonical_value": float(canonical_value) if recognized else None,
        "canonical_unit": result.get("canonical_unit") if recognized else None,
        "dimension": dict(dimension),
    }
    return MappingProxyType(normalized)


def coerce_medication_candidate(
    value: MedicationReconciliationCandidate | Mapping[str, object] | object,
    *,
    default_id: str = "candidate-1",
) -> MedicationReconciliationCandidate:
    """Coerce one extractor result into a validated local candidate."""

    return MedicationReconciliationCandidate.from_mapping(value, default_id=default_id)


@dataclass(frozen=True, repr=False)
class MedicationMatchDecision:
    """Explainable match or abstention decision for one candidate pair."""

    left_candidate_id: str
    right_candidate_id: str
    confidence: float
    status: DecisionStatus
    evidence: Mapping[str, FeatureStatus]
    feature_scores: Mapping[str, float]
    abstention_reasons: tuple[str, ...] = ()
    threshold: float = 0.80

    def __post_init__(self) -> None:
        if self.left_candidate_id == self.right_candidate_id:
            raise ValueError("candidate pair must contain two distinct candidates")
        if self.status not in {"matched", "abstained"}:
            raise ValueError("decision status must be matched or abstained")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("decision confidence must be between 0 and 1")
        normalized_evidence = {
            key: self.evidence[key] for key in _FEATURES if key in self.evidence
        }
        normalized_scores = {
            key: round(float(self.feature_scores.get(key, 0.0)), 6) for key in _FEATURES
        }
        object.__setattr__(self, "evidence", MappingProxyType(normalized_evidence))
        object.__setattr__(self, "feature_scores", MappingProxyType(normalized_scores))
        object.__setattr__(
            self,
            "abstention_reasons",
            tuple(dict.fromkeys(self.abstention_reasons)),
        )

    @property
    def matched(self) -> bool:
        """Return whether this pair may be merged."""

        return self.status == "matched"

    @property
    def abstained(self) -> bool:
        """Return whether this pair was withheld from merging."""

        return not self.matched

    @property
    def reason(self) -> str | None:
        """Return the first stable abstention reason, if any."""

        return self.abstention_reasons[0] if self.abstention_reasons else None

    @property
    def abstention_reason(self) -> str | None:
        """Alias for :attr:`reason` used by review clients."""

        return self.reason

    def to_dict(self) -> dict[str, object]:
        """Return a PHI-safe decision record."""

        return {
            "schema_version": MEDICATION_RECONCILIATION_SCHEMA_VERSION,
            "left_candidate_id_hash": _stable_hash(self.left_candidate_id),
            "right_candidate_id_hash": _stable_hash(self.right_candidate_id),
            "confidence": self.confidence,
            "status": self.status,
            "matched": self.matched,
            "evidence": dict(self.evidence),
            "feature_scores": dict(self.feature_scores),
            "abstention_reasons": list(self.abstention_reasons),
            "threshold": self.threshold,
            "advisory": MEDICATION_RECONCILIATION_ADVISORY,
        }

    def __repr__(self) -> str:
        return (
            "MedicationMatchDecision("
            f"left_candidate_id_hash={_stable_hash(self.left_candidate_id)!r}, "
            f"right_candidate_id_hash={_stable_hash(self.right_candidate_id)!r}, "
            f"confidence={self.confidence!r}, status={self.status!r})"
        )


def _compare_name(
    left: MedicationReconciliationCandidate,
    right: MedicationReconciliationCandidate,
) -> tuple[FeatureStatus, float, tuple[str, ...]]:
    same_code = (
        left.normalized_code is not None
        and right.normalized_code is not None
        and left.normalized_code == right.normalized_code
        and left.normalized_system == right.normalized_system
    )
    code_conflict = (
        left.normalized_code is not None
        and right.normalized_code is not None
        and not same_code
    )
    same_name = bool(
        left.normalized_name
        and right.normalized_name
        and left.normalized_name == right.normalized_name
    )
    if same_code or same_name:
        return "match", 1.0, ("coded_identity_conflict",) if code_conflict else ()
    reasons = ["name_mismatch"]
    if code_conflict:
        reasons.append("coded_identity_conflict")
    return "mismatch", 0.0, tuple(reasons)


def _compare_dose(
    left: MedicationReconciliationCandidate,
    right: MedicationReconciliationCandidate,
    policy: MedicationReconciliationPolicy,
) -> tuple[FeatureStatus, float, tuple[str, ...]]:
    left_present = left.dose is not None
    right_present = right.dose is not None
    if not left_present or not right_present:
        return "unknown", 0.0, ()
    if not left.dose_recognized or not right.dose_recognized:
        return "unknown", 0.0, ("unparseable_dose",)
    left_dimension = tuple(sorted(left._dose.get("dimension", {}).items()))
    right_dimension = tuple(sorted(right._dose.get("dimension", {}).items()))
    if left_dimension != right_dimension:
        return "mismatch", 0.0, ("dose_conflict",)
    left_value = float(left._dose["canonical_value"])
    right_value = float(right._dose["canonical_value"])
    if math.isclose(
        left_value,
        right_value,
        rel_tol=policy.dose_relative_tolerance,
        abs_tol=policy.dose_relative_tolerance,
    ):
        return "match", 1.0, ()
    return "mismatch", 0.0, ("dose_conflict",)


def _compare_route(
    left: MedicationReconciliationCandidate,
    right: MedicationReconciliationCandidate,
) -> tuple[FeatureStatus, float, tuple[str, ...]]:
    if left.normalized_route is None or right.normalized_route is None:
        return "unknown", 0.0, ()
    if left.normalized_route == right.normalized_route:
        return "match", 1.0, ()
    return "mismatch", 0.0, ("route_conflict",)


def _temporal_labels_conflict(left: str | None, right: str | None) -> bool:
    if left is None or right is None or left == right:
        return False
    return bool(
        (left in _CURRENT_TEMPORAL_LABELS and right in _STOPPED_TEMPORAL_LABELS)
        or (right in _CURRENT_TEMPORAL_LABELS and left in _STOPPED_TEMPORAL_LABELS)
    )


def _compare_temporal(
    left: MedicationReconciliationCandidate,
    right: MedicationReconciliationCandidate,
    policy: MedicationReconciliationPolicy,
) -> tuple[FeatureStatus, float, tuple[str, ...]]:
    left_label = _normalize_temporal_label(left.temporal_label)
    right_label = _normalize_temporal_label(right.temporal_label)
    left_has_date = (
        left.temporal_start_date is not None or left.temporal_end_date is not None
    )
    right_has_date = (
        right.temporal_start_date is not None or right.temporal_end_date is not None
    )
    if left.temporal_invalid or right.temporal_invalid:
        return "unknown", 0.0, ("unparseable_temporal",)
    if (
        not left_has_date
        and not right_has_date
        and left_label is None
        and right_label is None
    ):
        return "unknown", 0.0, ()
    if _temporal_labels_conflict(left_label, right_label):
        return "mismatch", 0.0, ("temporal_conflict",)
    if left_label is not None and right_label is not None and left_label == right_label:
        label_score = 1.0
    elif left_label is not None and right_label is not None:
        label_score = 0.25
    else:
        label_score = 0.0

    if not left_has_date or not right_has_date:
        if label_score == 1.0:
            return "match", 1.0, ()
        return "partial", label_score, ("temporal_mismatch",) if label_score else ()

    left_start = left.temporal_start_date or left.temporal_end_date
    left_end = left.temporal_end_date or left.temporal_start_date
    right_start = right.temporal_start_date or right.temporal_end_date
    right_end = right.temporal_end_date or right.temporal_start_date
    assert left_start is not None
    assert left_end is not None
    assert right_start is not None
    assert right_end is not None
    if left_start <= right_end and right_start <= left_end:
        date_score = 1.0
        reason = ()
    else:
        if left_end < right_start:
            gap = (right_start - left_end).days
        else:
            gap = (left_start - right_end).days
        date_score = max(0.0, 1.0 - gap / policy.temporal_gap_days)
        reason = ("temporal_gap",)
    score = min(date_score, label_score) if left_label and right_label else date_score
    if score >= 1.0:
        return "match", 1.0, reason
    return (
        "partial",
        score,
        reason + (("temporal_mismatch",) if label_score == 0.25 else ()),
    )


def score_medication_match(
    left: MedicationReconciliationCandidate | Mapping[str, object] | object,
    right: MedicationReconciliationCandidate | Mapping[str, object] | object,
    *,
    policy: MedicationReconciliationPolicy | None = None,
) -> MedicationMatchDecision:
    """Score one normalized medication pair and explain abstention.

    A decision is ``matched`` only when the weighted score reaches the policy
    threshold and no known identity, dose, route, or temporal-status conflict
    exists.  Missing fields reduce confidence; they never become implicit
    agreement.
    """

    selected_policy = policy or MedicationReconciliationPolicy()
    left_candidate = coerce_medication_candidate(left)
    right_candidate = coerce_medication_candidate(right)
    if left_candidate.candidate_id == right_candidate.candidate_id:
        raise ValueError("candidate pair must contain two distinct candidates")

    comparisons = (
        _compare_name(left_candidate, right_candidate),
        _compare_dose(left_candidate, right_candidate, selected_policy),
        _compare_route(left_candidate, right_candidate),
        _compare_temporal(left_candidate, right_candidate, selected_policy),
    )
    evidence = {
        feature: comparison[0] for feature, comparison in zip(_FEATURES, comparisons)
    }
    feature_scores = {
        feature: comparison[1] * selected_policy.weights[feature]
        for feature, comparison in zip(_FEATURES, comparisons)
    }
    confidence = round(max(0.0, min(1.0, sum(feature_scores.values()))), 6)
    reasons: list[str] = []
    for comparison in comparisons:
        for reason in comparison[2]:
            if reason not in reasons:
                reasons.append(reason)
    if confidence < selected_policy.match_threshold and not reasons:
        reasons.append(
            "insufficient_evidence" if confidence < 0.5 else "low_confidence"
        )
    matched = confidence >= selected_policy.match_threshold and not reasons
    return MedicationMatchDecision(
        left_candidate_id=left_candidate.candidate_id,
        right_candidate_id=right_candidate.candidate_id,
        confidence=confidence,
        status="matched" if matched else "abstained",
        evidence=evidence,
        feature_scores=feature_scores,
        abstention_reasons=() if matched else tuple(reasons),
        threshold=selected_policy.match_threshold,
    )


def score_medication_candidates(
    left: MedicationReconciliationCandidate | Mapping[str, object] | object,
    right: MedicationReconciliationCandidate | Mapping[str, object] | object,
    *,
    policy: MedicationReconciliationPolicy | None = None,
) -> MedicationMatchDecision:
    """Plural-named alias for :func:`score_medication_match`."""

    return score_medication_match(left, right, policy=policy)


@dataclass(frozen=True, repr=False)
class ReconciledMedicationGroup:
    """One safe group or explicit singleton produced by reconciliation."""

    group_id: str
    candidate_ids: tuple[str, ...]
    confidence: float
    status: GroupStatus
    abstention_reasons: tuple[str, ...] = ()

    @property
    def merged(self) -> bool:
        """Return whether more than one candidate was safely grouped."""

        return self.status == "merged"

    def to_dict(self) -> dict[str, object]:
        """Return a PHI-safe group record."""

        return {
            "group_id": self.group_id,
            "candidate_id_hashes": [
                _stable_hash(value) for value in self.candidate_ids
            ],
            "candidate_count": len(self.candidate_ids),
            "confidence": self.confidence,
            "status": self.status,
            "merged": self.merged,
            "abstention_reasons": list(self.abstention_reasons),
        }

    def __repr__(self) -> str:
        return (
            "ReconciledMedicationGroup("
            f"group_id={self.group_id!r}, candidate_count={len(self.candidate_ids)}, "
            f"confidence={self.confidence!r}, status={self.status!r})"
        )


@dataclass(frozen=True, repr=False)
class MedicationReconciliationResult:
    """All pair decisions and safe groups for one reconciliation run."""

    candidates: tuple[MedicationReconciliationCandidate, ...]
    decisions: tuple[MedicationMatchDecision, ...]
    groups: tuple[ReconciledMedicationGroup, ...]
    advisory: str = MEDICATION_RECONCILIATION_ADVISORY

    @property
    def abstentions(self) -> tuple[MedicationMatchDecision, ...]:
        """Return every pair withheld from merging."""

        return tuple(decision for decision in self.decisions if decision.abstained)

    @property
    def matched_pairs(self) -> tuple[MedicationMatchDecision, ...]:
        """Return every pair that passed the conservative match gate."""

        return tuple(decision for decision in self.decisions if decision.matched)

    @property
    def merged_groups(self) -> tuple[ReconciledMedicationGroup, ...]:
        """Return only groups containing at least two candidates."""

        return tuple(group for group in self.groups if group.merged)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic PHI-safe reconciliation report."""

        return {
            "schema_version": MEDICATION_RECONCILIATION_SCHEMA_VERSION,
            "candidate_count": len(self.candidates),
            "groups": [group.to_dict() for group in self.groups],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "abstention_count": len(self.abstentions),
            "advisory": self.advisory,
        }

    def __repr__(self) -> str:
        return (
            "MedicationReconciliationResult("
            f"candidate_count={len(self.candidates)}, "
            f"decision_count={len(self.decisions)}, group_count={len(self.groups)})"
        )


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _group_id(candidate_ids: Iterable[str]) -> str:
    return _stable_hash(("medication-group", tuple(sorted(candidate_ids))))


def reconcile_medications(
    candidates: Iterable[
        MedicationReconciliationCandidate | Mapping[str, object] | object
    ]
    | Mapping[str, object],
    *,
    policy: MedicationReconciliationPolicy | None = None,
) -> MedicationReconciliationResult:
    """Score all pairs and conservatively group compatible candidates.

    The function first computes every pair decision, then merges only groups
    whose complete cross-product is matched.  That all-pairs gate prevents a
    transitive chain (A matches B, B matches C) from silently joining A and C
    when their regimens conflict.  Every rejected pair remains in
    :attr:`MedicationReconciliationResult.abstentions` with stable reason
    codes.
    """

    selected_policy = policy or MedicationReconciliationPolicy()
    if isinstance(candidates, Mapping):
        values = [candidates]
    elif isinstance(candidates, (str, bytes)) or candidates is None:
        raise TypeError("candidates must be an iterable of medication candidates")
    else:
        try:
            values = list(candidates)
        except TypeError as exc:
            raise TypeError(
                "candidates must be an iterable of medication candidates"
            ) from exc
    normalized = tuple(
        coerce_medication_candidate(value, default_id=f"candidate-{index + 1}")
        for index, value in enumerate(values)
    )
    ids = tuple(candidate.candidate_id for candidate in normalized)
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_id values must be unique within a reconciliation")
    ordered = tuple(sorted(normalized, key=lambda candidate: candidate.candidate_id))
    decisions: list[MedicationMatchDecision] = []
    decision_by_pair: dict[tuple[str, str], MedicationMatchDecision] = {}
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            decision = score_medication_match(left, right, policy=selected_policy)
            decisions.append(decision)
            decision_by_pair[_pair_key(left.candidate_id, right.candidate_id)] = (
                decision
            )

    groups: list[set[str]] = [{candidate.candidate_id} for candidate in ordered]
    group_reasons: dict[str, set[str]] = {
        candidate.candidate_id: set() for candidate in ordered
    }

    def group_for(candidate_id: str) -> set[str]:
        for group in groups:
            if candidate_id in group:
                return group
        raise RuntimeError("internal medication group state is inconsistent")

    def can_join(left_group: set[str], right_group: set[str]) -> bool:
        return all(
            decision_by_pair[_pair_key(left_id, right_id)].matched
            for left_id in left_group
            for right_id in right_group
        )

    for decision in sorted(
        decisions,
        key=lambda item: (
            -item.confidence,
            item.left_candidate_id,
            item.right_candidate_id,
        ),
    ):
        left_group = group_for(decision.left_candidate_id)
        right_group = group_for(decision.right_candidate_id)
        if left_group is right_group:
            continue
        if not decision.matched:
            for reason in decision.abstention_reasons or ("no_safe_merge",):
                group_reasons[decision.left_candidate_id].add(reason)
                group_reasons[decision.right_candidate_id].add(reason)
            continue
        if can_join(left_group, right_group):
            left_group.update(right_group)
            groups.remove(right_group)
            continue
        for member in left_group | right_group:
            group_reasons[member].add("transitive_conflict")

    output_groups: list[ReconciledMedicationGroup] = []
    for group in sorted(groups, key=lambda item: min(item)):
        member_ids = tuple(sorted(group))
        internal_decisions = [
            decision_by_pair[_pair_key(left_id, right_id)]
            for index, left_id in enumerate(member_ids)
            for right_id in member_ids[index + 1 :]
        ]
        confidence = (
            round(min(decision.confidence for decision in internal_decisions), 6)
            if internal_decisions
            else 0.0
        )
        reasons = tuple(
            sorted(
                {reason for member in member_ids for reason in group_reasons[member]}
            )
        )
        output_groups.append(
            ReconciledMedicationGroup(
                group_id=_group_id(member_ids),
                candidate_ids=member_ids,
                confidence=confidence,
                status="merged" if len(member_ids) > 1 else "singleton",
                abstention_reasons=reasons,
            )
        )
    decisions.sort(key=lambda item: (item.left_candidate_id, item.right_candidate_id))
    return MedicationReconciliationResult(
        candidates=ordered,
        decisions=tuple(decisions),
        groups=tuple(output_groups),
    )


def reconcile_medication_candidates(
    candidates: Iterable[
        MedicationReconciliationCandidate | Mapping[str, object] | object
    ]
    | Mapping[str, object],
    *,
    policy: MedicationReconciliationPolicy | None = None,
) -> MedicationReconciliationResult:
    """Plural-named alias for :func:`reconcile_medications`."""

    return reconcile_medications(candidates, policy=policy)


__all__ = [
    "DecisionStatus",
    "FeatureStatus",
    "GroupStatus",
    "MEDICATION_RECONCILIATION_ADVISORY",
    "MEDICATION_RECONCILIATION_SCHEMA_VERSION",
    "MedicationMatchDecision",
    "MedicationReconciliationCandidate",
    "MedicationReconciliationPolicy",
    "MedicationReconciliationResult",
    "NormalizedMedicationCandidate",
    "ReconciledMedicationGroup",
    "coerce_medication_candidate",
    "reconcile_medication_candidates",
    "reconcile_medications",
    "score_medication_candidates",
    "score_medication_match",
]
