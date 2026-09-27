"""Typed, composable uncertainty sources for guarded clinical outputs.

Clinical uncertainty is not a single probability.  A result can be uncertain
because its evidence is incomplete, a model is ambiguous, a policy is
insufficient, a temporal relation is unresolved, or sources conflict.  This
module keeps those causes as separate, deterministic records so a guarded
output can disclose every active source without retaining source text.

Only controlled source types and reason codes are emitted.  References are
normalized to opaque SHA-256 identifiers before they are stored, serialized,
or represented.  The module is local-only and uses no filesystem, clock, or
network access.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

UNCERTAINTY_SOURCE_SCHEMA_VERSION: Final[int] = 1
UNCERTAINTY_SOURCES_SCHEMA_VERSION: Final[int] = UNCERTAINTY_SOURCE_SCHEMA_VERSION
UNCERTAINTY_SOURCE_DISCLAIMER: Final[str] = (
    "Typed uncertainty sources are assistive review metadata, not a clinical "
    "decision, diagnosis, treatment recommendation, or compliance certification."
)
UNCERTAINTY_SOURCES_DISCLAIMER: Final[str] = UNCERTAINTY_SOURCE_DISCLAIMER


class UncertaintySourceError(ValueError):
    """Raised when a typed uncertainty source cannot be constructed safely."""


class UncertaintySourceType(str, Enum):
    """The independent causes of uncertainty retained by a guarded output."""

    EVIDENCE = "evidence"
    MODEL = "model"
    POLICY = "policy"
    TEMPORAL = "temporal"
    CONFLICT = "conflict"


UNCERTAINTY_SOURCE_TYPES: Final[tuple[str, ...]] = tuple(
    source_type.value for source_type in UncertaintySourceType
)

# Reason codes are intentionally a closed vocabulary.  They make reports
# useful without allowing arbitrary caller text to become a log or audit field.
_REASON_CODES: dict[str, tuple[str, ...]] = {
    UncertaintySourceType.EVIDENCE.value: (
        "conflicting",
        "indirect",
        "insufficient",
        "insufficient_evidence",
        "low_quality",
        "missing",
        "stale",
        "unsupported",
        "unverified",
        "unspecified",
        "weak",
    ),
    UncertaintySourceType.MODEL.value: (
        "abstained",
        "ambiguous",
        "ambiguous_output",
        "calibration_gap",
        "low_confidence",
        "out_of_distribution",
        "unsupported",
        "uncertain",
        "unspecified",
    ),
    UncertaintySourceType.POLICY.value: (
        "incomplete",
        "insufficient",
        "insufficient_policy",
        "missing",
        "missing_policy",
        "policy_gap",
        "restricted",
        "review_required",
        "unsupported",
        "uncertain_policy",
        "unspecified",
    ),
    UncertaintySourceType.TEMPORAL.value: (
        "future",
        "future_context",
        "inconsistent",
        "missing",
        "stale",
        "unresolved",
        "unresolved_temporality",
        "unspecified",
    ),
    UncertaintySourceType.CONFLICT.value: (
        "conflict",
        "contradictory",
        "contradiction",
        "duplicate",
        "inconsistent",
        "priority_tie",
        "source_conflict",
        "unresolved",
        "unresolved_conflict",
        "unspecified",
    ),
}
UNCERTAINTY_SOURCE_REASON_CODES = MappingProxyType(
    {source_type: tuple(codes) for source_type, codes in _REASON_CODES.items()}
)
UNCERTAINTY_REASON_CODES = UNCERTAINTY_SOURCE_REASON_CODES

_OPAQUE_REFERENCE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SOURCE_TYPE_ORDER = {
    source_type.value: index for index, source_type in enumerate(UncertaintySourceType)
}
_MISSING = object()


def _source_type(value: Any) -> UncertaintySourceType:
    if isinstance(value, UncertaintySourceType):
        return value
    if not isinstance(value, str):
        raise UncertaintySourceError("uncertainty source type is unsupported")
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    try:
        return UncertaintySourceType(normalized)
    except ValueError:
        raise UncertaintySourceError("uncertainty source type is unsupported") from None


def _reason_code(value: Any, source_type: UncertaintySourceType) -> str:
    if value is None:
        return "unspecified"
    if not isinstance(value, str):
        raise UncertaintySourceError("uncertainty source reason code is unsupported")
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized not in _REASON_CODES[source_type.value]:
        raise UncertaintySourceError("uncertainty source reason code is unsupported")
    return normalized


def _opaque_reference(value: Any) -> str:
    if not isinstance(value, str):
        raise UncertaintySourceError("uncertainty source references must be strings")
    reference = value.strip()
    if not reference:
        raise UncertaintySourceError("uncertainty source references must not be empty")
    if _OPAQUE_REFERENCE_RE.fullmatch(reference):
        return reference
    digest = hashlib.sha256(
        ("openmed:uncertainty-reference:" + reference).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _references(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (bytes, bytearray, Mapping)):
        raise UncertaintySourceError("uncertainty source references have invalid shape")
    else:
        try:
            values = tuple(value)
        except TypeError:
            raise UncertaintySourceError(
                "uncertainty source references have invalid shape"
            ) from None
    return tuple(sorted({_opaque_reference(item) for item in values}))


def _alias_value(
    primary: Any,
    aliases: tuple[Any, ...],
    *,
    error: str,
) -> Any:
    present = [value for value in (primary, *aliases) if value is not _MISSING]
    if len(present) > 1:
        raise UncertaintySourceError(error)
    return present[0] if present else _MISSING


@dataclass(frozen=True, slots=True, init=False)
class UncertaintySource:
    """One typed, value-free cause of clinical uncertainty.

    Args:
        source_type: One of ``evidence``, ``model``, ``policy``, ``temporal``,
            or ``conflict``.
        code: Controlled reason code for the source.  ``reason_code`` and
            ``reason`` are accepted as keyword aliases.
        active: Whether the source currently contributes to uncertainty.
        references: Optional source/provenance identifiers.  Non-opaque input
            is immediately replaced by a SHA-256 identifier and is never
            retained.

    The class deliberately has no aggregate score.  Multiple sources remain
    independently inspectable and are ordered deterministically in a
    :class:`UncertaintySources` collection.
    """

    source_type: UncertaintySourceType
    code: str
    active: bool
    references: tuple[str, ...]

    def __init__(
        self,
        source_type: UncertaintySourceType | str | None = None,
        code: str | None = None,
        active: bool = True,
        references: Iterable[str] | str | None = None,
        *,
        kind: UncertaintySourceType | str | None = None,
        source: UncertaintySourceType | str | None = None,
        source_kind: UncertaintySourceType | str | None = None,
        reason_code: str | None = None,
        reason: str | None = None,
        evidence_refs: Iterable[str] | str | None | object = _MISSING,
    ) -> None:
        selected_type = _alias_value(
            source_type if source_type is not None else _MISSING,
            tuple(
                value if value is not None else _MISSING
                for value in (kind, source, source_kind)
            ),
            error="uncertainty source type was supplied more than once",
        )
        if selected_type is _MISSING:
            raise UncertaintySourceError("uncertainty source type is required")
        selected_code = _alias_value(
            code if code is not None else _MISSING,
            tuple(
                value if value is not None else _MISSING
                for value in (reason_code, reason)
            ),
            error="uncertainty source reason was supplied more than once",
        )
        if selected_code is _MISSING:
            selected_code = "unspecified"

        selected_references = _alias_value(
            references if references is not None else _MISSING,
            (evidence_refs,),
            error="uncertainty source references were supplied more than once",
        )
        if selected_references is _MISSING:
            selected_references = ()

        normalized_type = _source_type(selected_type)
        if type(active) is not bool:
            raise UncertaintySourceError("uncertainty source active flag is invalid")
        object.__setattr__(self, "source_type", normalized_type)
        object.__setattr__(self, "code", _reason_code(selected_code, normalized_type))
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "references", _references(selected_references))

    @property
    def kind(self) -> UncertaintySourceType:
        """Return :attr:`source_type` as an ergonomic alias."""

        return self.source_type

    @property
    def type(self) -> UncertaintySourceType:
        """Return :attr:`source_type` for type-oriented callers."""

        return self.source_type

    @property
    def reason_code(self) -> str:
        """Return the controlled reason code."""

        return self.code

    @property
    def reason(self) -> str:
        """Return :attr:`code` as a readable alias."""

        return self.code

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        """Return the opaque references attached to this source."""

        return self.references

    def identity_key(self) -> tuple[str, str, tuple[str, ...]]:
        """Return the stable key used to deduplicate composed sources."""

        return self.source_type.value, self.code, self.references

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible source record without raw input values."""

        payload: dict[str, Any] = {
            "source_type": self.source_type.value,
            "reason_code": self.code,
            "active": self.active,
        }
        if self.references:
            payload["references"] = list(self.references)
        return payload

    @classmethod
    def from_obj(cls, value: Any) -> "UncertaintySource":
        """Build a source from a source object or a safe mapping shape."""

        if isinstance(value, UncertaintySource):
            return value
        if not isinstance(value, Mapping):
            raise UncertaintySourceError("uncertainty source has invalid shape")
        source_type = _first_present(
            value,
            ("source_type", "type", "kind", "source", "source_kind"),
        )
        if source_type is _MISSING:
            raise UncertaintySourceError("uncertainty source type is required")
        code = _first_present(value, ("reason_code", "code", "reason"))
        active = value.get("active", True)
        references = _first_present(
            value,
            ("references", "evidence_refs", "provenance_refs"),
        )
        return cls(
            source_type,
            None if code is _MISSING else code,
            active,
            () if references is _MISSING else references,
        )


class EvidenceUncertaintySource(UncertaintySource):
    """Typed convenience constructor for evidence uncertainty."""

    def __init__(
        self,
        code: str | None = "insufficient",
        active: bool = True,
        references: Iterable[str] | str | None = None,
        *,
        reason_code: str | None = None,
        reason: str | None = None,
        evidence_refs: Iterable[str] | str | None | object = _MISSING,
    ) -> None:
        super().__init__(
            UncertaintySourceType.EVIDENCE,
            code,
            active,
            references,
            reason_code=reason_code,
            reason=reason,
            evidence_refs=evidence_refs,
        )


class ModelUncertaintySource(UncertaintySource):
    """Typed convenience constructor for model uncertainty."""

    def __init__(
        self,
        code: str | None = "ambiguous",
        active: bool = True,
        references: Iterable[str] | str | None = None,
        *,
        reason_code: str | None = None,
        reason: str | None = None,
        evidence_refs: Iterable[str] | str | None | object = _MISSING,
    ) -> None:
        super().__init__(
            UncertaintySourceType.MODEL,
            code,
            active,
            references,
            reason_code=reason_code,
            reason=reason,
            evidence_refs=evidence_refs,
        )


class PolicyUncertaintySource(UncertaintySource):
    """Typed convenience constructor for policy uncertainty."""

    def __init__(
        self,
        code: str | None = "insufficient",
        active: bool = True,
        references: Iterable[str] | str | None = None,
        *,
        reason_code: str | None = None,
        reason: str | None = None,
        evidence_refs: Iterable[str] | str | None | object = _MISSING,
    ) -> None:
        super().__init__(
            UncertaintySourceType.POLICY,
            code,
            active,
            references,
            reason_code=reason_code,
            reason=reason,
            evidence_refs=evidence_refs,
        )


class TemporalUncertaintySource(UncertaintySource):
    """Typed convenience constructor for temporal uncertainty."""

    def __init__(
        self,
        code: str | None = "unresolved",
        active: bool = True,
        references: Iterable[str] | str | None = None,
        *,
        reason_code: str | None = None,
        reason: str | None = None,
        evidence_refs: Iterable[str] | str | None | object = _MISSING,
    ) -> None:
        super().__init__(
            UncertaintySourceType.TEMPORAL,
            code,
            active,
            references,
            reason_code=reason_code,
            reason=reason,
            evidence_refs=evidence_refs,
        )


class ConflictUncertaintySource(UncertaintySource):
    """Typed convenience constructor for unresolved source conflicts."""

    def __init__(
        self,
        code: str | None = "unresolved",
        active: bool = True,
        references: Iterable[str] | str | None = None,
        *,
        reason_code: str | None = None,
        reason: str | None = None,
        evidence_refs: Iterable[str] | str | None | object = _MISSING,
    ) -> None:
        super().__init__(
            UncertaintySourceType.CONFLICT,
            code,
            active,
            references,
            reason_code=reason_code,
            reason=reason,
            evidence_refs=evidence_refs,
        )


# Short aliases make the five typed constructors discoverable without making
# callers depend on a particular noun ordering.
EvidenceUncertainty = EvidenceUncertaintySource
ModelUncertainty = ModelUncertaintySource
PolicyUncertainty = PolicyUncertaintySource
TemporalUncertainty = TemporalUncertaintySource
ConflictUncertainty = ConflictUncertaintySource


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return _MISSING


def _source_sort_key(
    source: UncertaintySource,
) -> tuple[int, str, tuple[str, ...], int]:
    return (
        _SOURCE_TYPE_ORDER[source.source_type.value],
        source.code,
        source.references,
        0 if source.active else 1,
    )


def _source_values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (UncertaintySource, Mapping)):
        if isinstance(value, Mapping) and not any(
            key in value
            for key in ("source_type", "type", "kind", "source", "source_kind")
        ):
            nested = _first_present(value, ("active_sources", "sources"))
            if nested is not _MISSING:
                return _source_values(nested)
        return (value,)
    if isinstance(value, (str, bytes, bytearray)):
        raise UncertaintySourceError("uncertainty source collection has invalid shape")
    try:
        return tuple(value)
    except TypeError:
        raise UncertaintySourceError(
            "uncertainty source collection has invalid shape"
        ) from None


@dataclass(frozen=True, slots=True)
class UncertaintySources:
    """A deterministic composition of independently disclosed sources."""

    sources: tuple[UncertaintySource, ...] = ()
    schema_version: int = UNCERTAINTY_SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != UNCERTAINTY_SOURCE_SCHEMA_VERSION
        ):
            raise UncertaintySourceError("unsupported uncertainty source schema")

        by_identity: dict[tuple[str, str, tuple[str, ...]], UncertaintySource] = {}
        for value in _source_values(self.sources):
            source = UncertaintySource.from_obj(value)
            if type(source) is not UncertaintySource:
                source = UncertaintySource(
                    source.source_type,
                    source.code,
                    source.active,
                    source.references,
                )
            existing = by_identity.get(source.identity_key())
            if existing is None or source.active:
                by_identity[source.identity_key()] = source
        normalized = tuple(sorted(by_identity.values(), key=_source_sort_key))
        object.__setattr__(self, "sources", normalized)

    @classmethod
    def compose(cls, *source_groups: Any) -> "UncertaintySources":
        """Compose source objects, collections, or disclosure mappings."""

        values: list[Any] = []
        for group in source_groups:
            values.extend(_source_values(group))
        return cls(tuple(values))

    @classmethod
    def from_obj(cls, value: Any) -> "UncertaintySources":
        """Build a collection from a collection or serialized mapping."""

        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            source_values = _first_present(value, ("active_sources", "sources"))
            if source_values is _MISSING:
                source_values = value.get("uncertainty_sources", ())
            return cls(_source_values(source_values))
        return cls(_source_values(value))

    @property
    def active_sources(self) -> tuple[UncertaintySource, ...]:
        """Return all currently active sources in canonical order."""

        return tuple(source for source in self.sources if source.active)

    @property
    def active(self) -> tuple[UncertaintySource, ...]:
        """Return :attr:`active_sources` as a short alias."""

        return self.active_sources

    @property
    def active_source_types(self) -> tuple[UncertaintySourceType, ...]:
        """Return distinct active source types in canonical type order."""

        present = {source.source_type for source in self.active_sources}
        return tuple(
            source_type
            for source_type in UncertaintySourceType
            if source_type in present
        )

    @property
    def has_active_sources(self) -> bool:
        """Return whether at least one source must be disclosed."""

        return bool(self.active_sources)

    @property
    def is_empty(self) -> bool:
        """Return whether no active uncertainty source is present."""

        return not self.has_active_sources

    def __iter__(self) -> Iterator[UncertaintySource]:
        return iter(self.sources)

    def disclose(self) -> tuple[dict[str, Any], ...]:
        """Return every active source as safe, JSON-compatible mappings."""

        return tuple(source.to_dict() for source in self.active_sources)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic disclosure with no aggregate score."""

        active_sources = self.disclose()
        return {
            "schema_version": self.schema_version,
            "active_source_count": len(active_sources),
            "active_source_types": [
                source_type.value for source_type in self.active_source_types
            ],
            "active_sources": list(active_sources),
            "disclaimer": UNCERTAINTY_SOURCE_DISCLAIMER,
        }

    def to_json(self) -> str:
        """Return compact, byte-stable JSON for an audit or guarded output."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


UncertaintySourceSet = UncertaintySources
ClinicalUncertaintySources = UncertaintySources


def evidence_uncertainty(
    code: str | None = "insufficient",
    *,
    active: bool = True,
    references: Iterable[str] | str | None = None,
    reason_code: str | None = None,
    reason: str | None = None,
    evidence_refs: Iterable[str] | str | None | object = _MISSING,
) -> EvidenceUncertaintySource:
    """Create a typed evidence uncertainty source."""

    return EvidenceUncertaintySource(
        code,
        active,
        references,
        reason_code=reason_code,
        reason=reason,
        evidence_refs=evidence_refs,
    )


def model_uncertainty(
    code: str | None = "ambiguous",
    *,
    active: bool = True,
    references: Iterable[str] | str | None = None,
    reason_code: str | None = None,
    reason: str | None = None,
    evidence_refs: Iterable[str] | str | None | object = _MISSING,
) -> ModelUncertaintySource:
    """Create a typed model uncertainty source."""

    return ModelUncertaintySource(
        code,
        active,
        references,
        reason_code=reason_code,
        reason=reason,
        evidence_refs=evidence_refs,
    )


def policy_uncertainty(
    code: str | None = "insufficient",
    *,
    active: bool = True,
    references: Iterable[str] | str | None = None,
    reason_code: str | None = None,
    reason: str | None = None,
    evidence_refs: Iterable[str] | str | None | object = _MISSING,
) -> PolicyUncertaintySource:
    """Create a typed policy uncertainty source."""

    return PolicyUncertaintySource(
        code,
        active,
        references,
        reason_code=reason_code,
        reason=reason,
        evidence_refs=evidence_refs,
    )


def temporal_uncertainty(
    code: str | None = "unresolved",
    *,
    active: bool = True,
    references: Iterable[str] | str | None = None,
    reason_code: str | None = None,
    reason: str | None = None,
    evidence_refs: Iterable[str] | str | None | object = _MISSING,
) -> TemporalUncertaintySource:
    """Create a typed temporal uncertainty source."""

    return TemporalUncertaintySource(
        code,
        active,
        references,
        reason_code=reason_code,
        reason=reason,
        evidence_refs=evidence_refs,
    )


def conflict_uncertainty(
    code: str | None = "unresolved",
    *,
    active: bool = True,
    references: Iterable[str] | str | None = None,
    reason_code: str | None = None,
    reason: str | None = None,
    evidence_refs: Iterable[str] | str | None | object = _MISSING,
) -> ConflictUncertaintySource:
    """Create a typed source for unresolved or contradictory evidence."""

    return ConflictUncertaintySource(
        code,
        active,
        references,
        reason_code=reason_code,
        reason=reason,
        evidence_refs=evidence_refs,
    )


evidence_source = evidence_uncertainty
model_source = model_uncertainty
policy_source = policy_uncertainty
temporal_source = temporal_uncertainty
conflict_source = conflict_uncertainty


def compose_uncertainty_sources(*source_groups: Any) -> UncertaintySources:
    """Compose one or more source values into a deterministic disclosure."""

    return UncertaintySources.compose(*source_groups)


def build_uncertainty_sources(
    sources: Any = None,
    *,
    evidence: Any = None,
    model: Any = None,
    policy: Any = None,
    temporal: Any = None,
    conflict: Any = None,
) -> UncertaintySources:
    """Build sources from a collection plus optional typed groups."""

    groups = [sources, evidence, model, policy, temporal, conflict]
    return compose_uncertainty_sources(*groups)


def coerce_uncertainty_sources(value: Any) -> UncertaintySources:
    """Normalize a source collection or serialized disclosure mapping."""

    return UncertaintySources.from_obj(value)


def disclose_uncertainty_sources(value: Any) -> dict[str, Any]:
    """Return every active source in a safe guarded-output disclosure."""

    return coerce_uncertainty_sources(value).to_dict()


__all__ = [
    "UNCERTAINTY_REASON_CODES",
    "UNCERTAINTY_SOURCE_DISCLAIMER",
    "UNCERTAINTY_SOURCE_REASON_CODES",
    "UNCERTAINTY_SOURCE_SCHEMA_VERSION",
    "UNCERTAINTY_SOURCE_TYPES",
    "UNCERTAINTY_SOURCES_DISCLAIMER",
    "UNCERTAINTY_SOURCES_SCHEMA_VERSION",
    "ClinicalUncertaintySources",
    "ConflictUncertainty",
    "ConflictUncertaintySource",
    "EvidenceUncertainty",
    "EvidenceUncertaintySource",
    "ModelUncertainty",
    "ModelUncertaintySource",
    "PolicyUncertainty",
    "PolicyUncertaintySource",
    "TemporalUncertainty",
    "TemporalUncertaintySource",
    "UncertaintySource",
    "UncertaintySourceError",
    "UncertaintySourceSet",
    "UncertaintySourceType",
    "UncertaintySources",
    "build_uncertainty_sources",
    "coerce_uncertainty_sources",
    "compose_uncertainty_sources",
    "conflict_source",
    "conflict_uncertainty",
    "disclose_uncertainty_sources",
    "evidence_source",
    "evidence_uncertainty",
    "model_source",
    "model_uncertainty",
    "policy_source",
    "policy_uncertainty",
    "temporal_source",
    "temporal_uncertainty",
]
