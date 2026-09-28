"""Experiencer-aware clinical natural-language-inference pair inputs.

Clinical NLI must not turn a statement about a relative or caregiver into a
claim about the patient.  This module keeps the experiencer axis beside both
sides of a pair and applies a conservative gate before an NLI prediction can
be treated as entailment.

The implementation is intentionally dependency-free and local-first.  Pair
text is retained only for the explicit local model-input boundary.  Regular
representations, JSON, validation errors, and audit payloads contain only
experiencer classes, offsets, lengths, hashes, and controlled decisions; they
never contain source text.  The output is assistive metadata for qualified
human review, not a clinical decision.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Literal, TypeAlias, cast

from openmed.core.audit import hash_text, stable_hash

from .context import FAMILY_EXPERIENCER, PATIENT_EXPERIENCER

NLI_EXPERIENCER_PAIR_SCHEMA_VERSION: Final = 1
NLI_EXPERIENCER_PAIR_ADVISORY: Final = (
    "Experiencer-aware clinical NLI pairs are deterministic assistive inputs "
    "for qualified human review. Conflicting or unresolved experiencers are "
    "never treated as entailment or an autonomous clinical decision."
)
CLINICAL_NLI_EXPERIENCER_PAIR_ADVISORY: Final = NLI_EXPERIENCER_PAIR_ADVISORY

CAREGIVER_EXPERIENCER: Final = "caregiver"
UNKNOWN_EXPERIENCER: Final = "unknown"

NliExperiencerClass: TypeAlias = Literal[
    "patient",
    "family",
    "caregiver",
    "unknown",
]

# The NLI contract is deliberately distinct from the older context resolver's
# patient/family/other vocabulary.  ``other`` is not safe to reinterpret as a
# caregiver, so it is normalized to the unresolved ``unknown`` class below.
NLI_EXPERIENCER_VALUES: Final[tuple[NliExperiencerClass, ...]] = (
    PATIENT_EXPERIENCER,
    FAMILY_EXPERIENCER,
    CAREGIVER_EXPERIENCER,
    UNKNOWN_EXPERIENCER,
)
EXPERIENCER_NLI_VALUES = NLI_EXPERIENCER_VALUES
NLI_EXPERIENCER_CLASSES = NLI_EXPERIENCER_VALUES

EXPERIENCER_COMPATIBILITY_VALUES: Final = (
    "compatible",
    "incompatible",
    "unresolved",
)

SpanOffset: TypeAlias = tuple[int, int]
ExperiencerSource: TypeAlias = Literal[
    "provided",
    "context",
    "default",
    "unresolved",
    "cue",
    "section",
    "span",
    "explicit",
    "metadata",
]

_MISSING = object()
_TEXT_KEYS = (
    "text",
    "surface",
    "content",
    "claim",
    "evidence",
    "value",
    "premise",
    "hypothesis",
)
_OFFSET_KEYS = ("offset", "span", "source_offset", "source_span")
_START_KEYS = ("start", "source_start", "start_offset", "begin")
_END_KEYS = ("end", "source_end", "end_offset", "stop")
_EXPERIENCER_KEYS = (
    "experiencer",
    "subject",
    "experiencer_class",
    "subject_class",
    "experiencer_type",
    "class",
)
_EXPERIENCER_CONTAINER_KEYS = (
    "experiencer_metadata",
    "experiencer_context",
    "assignment",
    "clinical_assertion",
    "clinical_context",
    "assertion",
    "context",
    "metadata",
)
_RESOLVED_KEYS = ("resolved", "is_resolved", "experiencer_resolved")
_SOURCE_KEYS = ("source", "experiencer_source", "resolution_source")
_CUE_OFFSET_KEYS = ("cue_offset", "experiencer_cue_offset")
_SAFE_SOURCES = frozenset(
    {
        "provided",
        "context",
        "default",
        "unresolved",
        "cue",
        "section",
        "span",
        "explicit",
        "metadata",
    }
)
_NLI_LABEL_ALIASES = {
    "entail": "entailment",
    "entails": "entailment",
    "entailment": "entailment",
    "contradict": "contradiction",
    "contradicts": "contradiction",
    "contradiction": "contradiction",
    "neutral": "neutral",
    "abstain": "abstention",
    "abstention": "abstention",
    "review_required": "review_required",
}


class ExperiencerCompatibility(str, Enum):
    """Conservative compatibility state for two NLI pair subjects."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    CONFLICTING = "incompatible"
    CONFLICT = "incompatible"
    UNRESOLVED = "unresolved"


class NliExperiencerPairError(ValueError):
    """Base error for malformed or unsafe experiencer-aware pair inputs."""


class InconsistentExperiencerMetadataError(NliExperiencerPairError):
    """Raised when redundant experiencer fields disagree."""


class NliExperiencerPairValidationError(NliExperiencerPairError):
    """Raised when pair text, offsets, labels, or metadata are invalid."""


def _invalid(field_name: str) -> NliExperiencerPairValidationError:
    """Create a value-free validation error."""

    return NliExperiencerPairValidationError(f"{field_name} is invalid")


def _inconsistent(field_name: str) -> InconsistentExperiencerMetadataError:
    """Create a value-free inconsistency error."""

    return InconsistentExperiencerMetadataError(f"{field_name} is inconsistent")


def _mapping_from_object(value: object) -> Mapping[str, object] | None:
    """Read known fields from mappings or small record-like objects only."""

    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return None

    fields: dict[str, object] = {}
    for name in (
        *_TEXT_KEYS,
        *_OFFSET_KEYS,
        *_START_KEYS,
        *_END_KEYS,
        *_EXPERIENCER_KEYS,
        *_EXPERIENCER_CONTAINER_KEYS,
        *_RESOLVED_KEYS,
        *_SOURCE_KEYS,
        *_CUE_OFFSET_KEYS,
    ):
        try:
            candidate = getattr(value, name, _MISSING)
        except Exception:
            continue
        if candidate is not _MISSING:
            fields[name] = candidate
    return fields or None


def _first_value(source: Mapping[str, object], keys: Sequence[str]) -> object:
    """Return the first present value without converting caller data to text."""

    for key in keys:
        if key in source:
            return source[key]
    return _MISSING


def _normalize_class(value: object) -> NliExperiencerClass:
    """Normalize one class without echoing an untrusted value in errors."""

    if isinstance(value, Mapping):
        nested = _first_value(cast(Mapping[str, object], value), _EXPERIENCER_KEYS)
        if nested is _MISSING:
            raise _invalid("experiencer")
        value = nested
    if not isinstance(value, str):
        raise _invalid("experiencer")

    normalized = " ".join(value.strip().casefold().replace("_", " ").split())
    aliases: dict[str, NliExperiencerClass] = {
        "patient": PATIENT_EXPERIENCER,
        "self": PATIENT_EXPERIENCER,
        "index patient": PATIENT_EXPERIENCER,
        "family": FAMILY_EXPERIENCER,
        "family member": FAMILY_EXPERIENCER,
        "relative": FAMILY_EXPERIENCER,
        "family history": FAMILY_EXPERIENCER,
        "caregiver": CAREGIVER_EXPERIENCER,
        "care giver": CAREGIVER_EXPERIENCER,
        "carer": CAREGIVER_EXPERIENCER,
        "care provider": CAREGIVER_EXPERIENCER,
        "unknown": UNKNOWN_EXPERIENCER,
        "unresolved": UNKNOWN_EXPERIENCER,
        "indeterminate": UNKNOWN_EXPERIENCER,
        "unspecified": UNKNOWN_EXPERIENCER,
        # The existing cue resolver's non-relative class is intentionally not
        # guessed to be a caregiver (a donor or roommate is not one).
        "other": UNKNOWN_EXPERIENCER,
    }
    try:
        return aliases[normalized]
    except KeyError:
        raise _invalid("experiencer") from None


def _normalize_source(
    value: object, *, default: ExperiencerSource
) -> ExperiencerSource:
    """Keep provenance to a short controlled token."""

    if value is _MISSING or value is None:
        return default
    if not isinstance(value, str):
        raise _invalid("experiencer source")
    normalized = value.strip().casefold().replace(" ", "_")
    if normalized not in _SAFE_SOURCES:
        return "provided"
    return cast(ExperiencerSource, normalized)


def _coerce_bool(value: object, *, field_name: str) -> bool:
    """Validate a strict boolean flag."""

    if type(value) is not bool:
        raise _invalid(field_name)
    return value


def _coerce_offset(value: object, *, field_name: str) -> SpanOffset:
    """Validate a positive half-open offset pair."""

    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        start = _first_value(mapping, _START_KEYS)
        end = _first_value(mapping, _END_KEYS)
        if start is _MISSING or end is _MISSING:
            raise _invalid(field_name)
        value = (start, end)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
    ):
        raise _invalid(field_name)
    start, end = value
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        raise _invalid(field_name)
    return start, end


def _coerce_text(
    value: object,
    *,
    side_name: str,
) -> tuple[str, Mapping[str, object] | None]:
    """Extract one side's text while preserving it only in the pair object."""

    if type(value) is str:
        text = value
        mapping = None
    else:
        mapping = _mapping_from_object(value)
        if mapping is None:
            raise _invalid(f"{side_name} text")
        raw_text = _first_value(mapping, _TEXT_KEYS)
        if raw_text is _MISSING:
            raise NliExperiencerPairValidationError(f"{side_name} text is required")
        if type(raw_text) is not str:
            raise _invalid(f"{side_name} text")
        text = raw_text
    if not text.strip():
        raise NliExperiencerPairValidationError(f"{side_name} text is required")
    return text, mapping


def _offset_from_mapping(
    mapping: Mapping[str, object] | None,
    *,
    side_name: str,
) -> SpanOffset | None:
    """Read redundant offset forms and reject disagreements."""

    if mapping is None:
        return None
    candidates: list[SpanOffset] = []
    for key in _OFFSET_KEYS:
        value = mapping.get(key, _MISSING)
        if value is not _MISSING:
            candidates.append(_coerce_offset(value, field_name=f"{side_name} offset"))
    start = _first_value(mapping, _START_KEYS)
    end = _first_value(mapping, _END_KEYS)
    if start is not _MISSING or end is not _MISSING:
        if start is _MISSING or end is _MISSING:
            raise _invalid(f"{side_name} offset")
        candidates.append(
            _coerce_offset((start, end), field_name=f"{side_name} offset")
        )
    if not candidates:
        return None
    if len(set(candidates)) != 1:
        raise _inconsistent(f"{side_name} offset")
    return candidates[0]


def _nested_experiencer_sources(value: object) -> tuple[object, ...]:
    """Return known metadata containers without reading source text."""

    if isinstance(value, str):
        # A raw pair side is usually source text.  String metadata is handled
        # explicitly by ``ExperiencerMetadata.from_value`` instead of being
        # guessed from the side text.
        return ()
    mapping = _mapping_from_object(value)
    if mapping is None:
        return ()

    sources: list[object] = []
    for key in _EXPERIENCER_CONTAINER_KEYS:
        candidate = mapping.get(key, _MISSING)
        if candidate is _MISSING:
            continue
        sources.append(candidate)
    sources.append(mapping)
    return tuple(sources)


def _parse_metadata_source(value: object) -> "ExperiencerMetadata | None":
    """Parse one direct metadata source, ignoring unrelated span fields."""

    if isinstance(value, str):
        return ExperiencerMetadata(value)
    mapping = _mapping_from_object(value)
    if mapping is None:
        return None

    raw_class = _first_value(mapping, _EXPERIENCER_KEYS)
    resolved_value = _first_value(mapping, _RESOLVED_KEYS)
    raw_source = _first_value(mapping, _SOURCE_KEYS)
    raw_cue_offset = _first_value(mapping, _CUE_OFFSET_KEYS)
    if (
        raw_class is _MISSING
        and resolved_value is _MISSING
        and raw_cue_offset is _MISSING
    ):
        return None

    experiencer: NliExperiencerClass = UNKNOWN_EXPERIENCER
    if raw_class is not _MISSING and raw_class is not None:
        experiencer = _normalize_class(raw_class)
    classes = {
        UNKNOWN_EXPERIENCER if mapping[key] is None else _normalize_class(mapping[key])
        for key in _EXPERIENCER_KEYS
        if key in mapping
    }
    if len(classes) > 1:
        raise _inconsistent("experiencer")
    resolved_flags = {
        _coerce_bool(mapping[key], field_name="experiencer resolved")
        for key in _RESOLVED_KEYS
        if key in mapping and mapping[key] is not None
    }
    if len(resolved_flags) > 1:
        raise _inconsistent("experiencer resolved")
    if resolved_value is _MISSING or resolved_value is None:
        resolved: bool | None = None
    else:
        resolved = _coerce_bool(resolved_value, field_name="experiencer resolved")
    source_default: ExperiencerSource = (
        "unresolved" if experiencer == UNKNOWN_EXPERIENCER else "provided"
    )
    source = _normalize_source(raw_source, default=source_default)
    cue_offset = None
    if raw_cue_offset is not _MISSING and raw_cue_offset is not None:
        cue_offset = _coerce_offset(raw_cue_offset, field_name="experiencer cue offset")
    return ExperiencerMetadata(
        experiencer=experiencer,
        source=source,
        resolved=resolved,
        cue_offset=cue_offset,
    )


@dataclass(frozen=True, slots=True)
class ExperiencerMetadata:
    """Validated experiencer class and safe resolution provenance.

    ``experiencer`` is one of ``patient``, ``family``, ``caregiver``, or
    ``unknown``.  Unknown is the fail-closed representation for a missing or
    explicitly unresolved subject.  ``cue_offset`` can point to a local
    subject cue, but no cue surface text is retained.
    """

    experiencer: NliExperiencerClass = UNKNOWN_EXPERIENCER
    source: ExperiencerSource = "provided"
    resolved: bool | None = None
    cue_offset: SpanOffset | None = None

    def __post_init__(self) -> None:
        normalized = _normalize_class(self.experiencer)
        source = _normalize_source(self.source, default="provided")
        if self.resolved is not None and type(self.resolved) is not bool:
            raise _invalid("experiencer resolved")
        resolved = self.resolved
        if normalized == UNKNOWN_EXPERIENCER or resolved is False:
            normalized = UNKNOWN_EXPERIENCER
            resolved = False
            if source == "provided":
                source = "unresolved"
        else:
            resolved = True
        object.__setattr__(self, "experiencer", normalized)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "resolved", resolved)
        if self.cue_offset is not None:
            object.__setattr__(
                self,
                "cue_offset",
                _coerce_offset(self.cue_offset, field_name="experiencer cue offset"),
            )

    @classmethod
    def from_value(cls, value: object) -> "ExperiencerMetadata":
        """Coerce a string, assertion, or metadata mapping to one sidecar."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        parsed = [
            metadata
            for source in _nested_experiencer_sources(value)
            if (metadata := _parse_metadata_source(source)) is not None
        ]
        if not parsed:
            return cls(
                experiencer=UNKNOWN_EXPERIENCER,
                source="default",
                resolved=False,
            )
        return _merge_metadata(parsed, side_name="experiencer")

    @property
    def value(self) -> NliExperiencerClass:
        """Return the canonical class value."""

        return self.experiencer

    @property
    def experiencer_class(self) -> NliExperiencerClass:
        """Return the canonical class under an explicit field name."""

        return self.experiencer

    @property
    def is_resolved(self) -> bool:
        """Return whether this side has a known subject class."""

        return self.resolved is True and self.experiencer != UNKNOWN_EXPERIENCER

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata without subject surface text."""

        payload: dict[str, Any] = {
            "experiencer": self.experiencer,
            "resolved": self.is_resolved,
            "source": self.source,
        }
        if self.cue_offset is not None:
            payload["cue_offset"] = list(self.cue_offset)
        return payload

    def __eq__(self, other: object) -> bool:
        """Compare naturally with the canonical string used by callers."""

        if isinstance(other, str):
            try:
                return self.experiencer == _normalize_class(other)
            except NliExperiencerPairValidationError:
                return False
        if not isinstance(other, ExperiencerMetadata):
            return NotImplemented
        return (
            self.experiencer,
            self.source,
            self.resolved,
            self.cue_offset,
        ) == (
            other.experiencer,
            other.source,
            other.resolved,
            other.cue_offset,
        )


NliExperiencerMetadata = ExperiencerMetadata
ClinicalNliExperiencerMetadata = ExperiencerMetadata


def _merge_metadata(
    values: Iterable[ExperiencerMetadata],
    *,
    side_name: str,
) -> ExperiencerMetadata:
    """Merge redundant metadata while preserving the first safe provenance."""

    parsed = tuple(values)
    if not parsed:
        return ExperiencerMetadata(
            experiencer=UNKNOWN_EXPERIENCER,
            source="default",
            resolved=False,
        )
    classes = {item.experiencer for item in parsed}
    if len(classes) != 1:
        raise _inconsistent(f"{side_name} experiencer metadata")
    first = parsed[0]
    return first


def validate_experiencer_metadata(value: object) -> ExperiencerMetadata:
    """Validate and return one canonical experiencer sidecar."""

    return ExperiencerMetadata.from_value(value)


def _merge_side_experiencer(
    side_value: object,
    explicit_values: Sequence[object],
    *,
    side_name: str,
) -> ExperiencerMetadata:
    """Merge explicit keywords with side-attached metadata."""

    parsed: list[ExperiencerMetadata] = []
    for value in explicit_values:
        metadata = ExperiencerMetadata.from_value(value)
        parsed.append(metadata)
    for source in _nested_experiencer_sources(side_value):
        metadata = _parse_metadata_source(source)
        if metadata is not None:
            parsed.append(metadata)
    return _merge_metadata(parsed, side_name=side_name)


def _normalize_label(value: object) -> str | None:
    """Normalize an optional provider prediction without echoing it."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid("NLI label")
    normalized = value.strip().casefold().replace(" ", "_")
    try:
        return _NLI_LABEL_ALIASES[normalized]
    except KeyError:
        raise _invalid("NLI label") from None


def _normalize_score(value: object) -> float | None:
    """Validate an optional finite probability in the closed unit interval."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _invalid("NLI score")
    if not 0.0 <= value <= 1.0:
        raise _invalid("NLI score")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise _invalid("NLI score")
    return score


@dataclass(frozen=True, slots=True)
class ExperiencerComparison:
    """Safe comparison result for the two pair-side experiencers."""

    premise: ExperiencerMetadata
    hypothesis: ExperiencerMetadata
    status: ExperiencerCompatibility
    reason: str

    @property
    def compatible(self) -> bool:
        """Return whether entailment is permitted by the subject gate."""

        return self.status is ExperiencerCompatibility.COMPATIBLE

    @property
    def resolved(self) -> bool:
        """Return whether both sides have resolved subject classes."""

        return self.premise.is_resolved and self.hypothesis.is_resolved

    def to_dict(self) -> dict[str, str | bool]:
        """Return the value-free compatibility decision."""

        return {
            "status": self.status.value,
            "reason": self.reason,
            "compatible": self.compatible,
            "resolved": self.resolved,
        }


def compare_experiencers(
    premise: ExperiencerMetadata | object,
    hypothesis: ExperiencerMetadata | object,
) -> ExperiencerComparison:
    """Compare two subjects using the fail-closed entailment policy.

    Known identical classes are compatible.  Any different known classes are
    incompatible, and any unknown side is unresolved.  Unknown is never
    treated as a wildcard because doing so could convert a family or caregiver
    statement into a patient assertion.
    """

    premise_metadata = ExperiencerMetadata.from_value(premise)
    hypothesis_metadata = ExperiencerMetadata.from_value(hypothesis)
    if not premise_metadata.is_resolved or not hypothesis_metadata.is_resolved:
        return ExperiencerComparison(
            premise=premise_metadata,
            hypothesis=hypothesis_metadata,
            status=ExperiencerCompatibility.UNRESOLVED,
            reason="experiencer_unresolved",
        )
    if premise_metadata.experiencer != hypothesis_metadata.experiencer:
        return ExperiencerComparison(
            premise=premise_metadata,
            hypothesis=hypothesis_metadata,
            status=ExperiencerCompatibility.INCOMPATIBLE,
            reason="experiencer_conflict",
        )
    return ExperiencerComparison(
        premise=premise_metadata,
        hypothesis=hypothesis_metadata,
        status=ExperiencerCompatibility.COMPATIBLE,
        reason="experiencer_match",
    )


compare_experiencer_classes = compare_experiencers
classify_experiencer_compatibility = compare_experiencers


@dataclass(frozen=True, slots=True, repr=False)
class NliExperiencerPair:
    """One NLI pair with an explicit experiencer sidecar on both sides.

    The raw ``premise`` and ``hypothesis`` are available only through the
    explicit local model-input methods.  ``to_dict()``, ``to_json()``, and
    ``repr`` contain hashes, lengths, offsets, metadata, and safety decisions.
    """

    premise: str
    hypothesis: str
    premise_experiencer: ExperiencerMetadata | NliExperiencerClass | str = (
        UNKNOWN_EXPERIENCER
    )
    hypothesis_experiencer: ExperiencerMetadata | NliExperiencerClass | str = (
        UNKNOWN_EXPERIENCER
    )
    premise_offset: SpanOffset | None = None
    hypothesis_offset: SpanOffset | None = None
    predicted_label: str | None = None
    predicted_score: float | None = None
    schema_version: int = NLI_EXPERIENCER_PAIR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.premise) is not str or not self.premise.strip():
            raise NliExperiencerPairValidationError("premise text is required")
        if type(self.hypothesis) is not str or not self.hypothesis.strip():
            raise NliExperiencerPairValidationError("hypothesis text is required")
        object.__setattr__(
            self,
            "premise_experiencer",
            ExperiencerMetadata.from_value(self.premise_experiencer),
        )
        object.__setattr__(
            self,
            "hypothesis_experiencer",
            ExperiencerMetadata.from_value(self.hypothesis_experiencer),
        )
        if self.premise_offset is not None:
            object.__setattr__(
                self,
                "premise_offset",
                _coerce_offset(self.premise_offset, field_name="premise offset"),
            )
        if self.hypothesis_offset is not None:
            object.__setattr__(
                self,
                "hypothesis_offset",
                _coerce_offset(
                    self.hypothesis_offset,
                    field_name="hypothesis offset",
                ),
            )
        object.__setattr__(
            self, "predicted_label", _normalize_label(self.predicted_label)
        )
        object.__setattr__(
            self, "predicted_score", _normalize_score(self.predicted_score)
        )
        if type(self.schema_version) is not int or (
            self.schema_version != NLI_EXPERIENCER_PAIR_SCHEMA_VERSION
        ):
            raise _invalid("schema version")

    @property
    def premise_experiencer_class(self) -> NliExperiencerClass:
        """Return the premise's canonical class."""

        return cast(ExperiencerMetadata, self.premise_experiencer).experiencer

    @property
    def hypothesis_experiencer_class(self) -> NliExperiencerClass:
        """Return the hypothesis's canonical class."""

        return cast(ExperiencerMetadata, self.hypothesis_experiencer).experiencer

    @property
    def premise_experiencer_metadata(self) -> ExperiencerMetadata:
        """Return the premise sidecar."""

        return cast(ExperiencerMetadata, self.premise_experiencer)

    @property
    def hypothesis_experiencer_metadata(self) -> ExperiencerMetadata:
        """Return the hypothesis sidecar."""

        return cast(ExperiencerMetadata, self.hypothesis_experiencer)

    @property
    def experiencer_metadata(self) -> dict[str, dict[str, Any]]:
        """Return both sidecars without source text."""

        return {
            "premise": self.premise_experiencer_metadata.to_dict(),
            "hypothesis": self.hypothesis_experiencer_metadata.to_dict(),
        }

    @property
    def experiencer_comparison(self) -> ExperiencerComparison:
        """Return the conservative subject compatibility decision."""

        return compare_experiencers(
            self.premise_experiencer_metadata,
            self.hypothesis_experiencer_metadata,
        )

    @property
    def experiencer_compatibility(self) -> str:
        """Return ``compatible``, ``incompatible``, or ``unresolved``."""

        return self.experiencer_comparison.status.value

    @property
    def experiencer_status(self) -> str:
        """Return the compatibility state under a concise alias."""

        return self.experiencer_compatibility

    @property
    def experiencer_relation(self) -> str:
        """Return the compatibility state under a relation-oriented alias."""

        return self.experiencer_compatibility

    @property
    def experiencer_compatible(self) -> bool:
        """Return whether the pair may be treated as entailment."""

        return self.experiencer_comparison.compatible

    @property
    def is_experiencer_compatible(self) -> bool:
        """Return :attr:`experiencer_compatible` under an explicit alias."""

        return self.experiencer_compatible

    @property
    def entailment_allowed(self) -> bool:
        """Return whether experiencer evidence permits entailment."""

        return self.experiencer_compatible

    @property
    def can_entail(self) -> bool:
        """Return :attr:`entailment_allowed` under a concise alias."""

        return self.entailment_allowed

    @property
    def entailment_blocked(self) -> bool:
        """Return whether the pair must not be accepted as entailment."""

        return not self.entailment_allowed

    @property
    def block_reason(self) -> str | None:
        """Return the controlled reason for withholding entailment."""

        comparison = self.experiencer_comparison
        return None if comparison.compatible else comparison.reason

    @property
    def experiencer_reason(self) -> str | None:
        """Return :attr:`block_reason` under the experiencer vocabulary."""

        return self.block_reason

    @property
    def review_required(self) -> bool:
        """Return whether the subject gate requires additional review."""

        return self.entailment_blocked

    @property
    def requires_review(self) -> bool:
        """Return :attr:`review_required` under a concise alias."""

        return self.review_required

    @property
    def requires_clinician_review(self) -> Literal[True]:
        """Return the always-on clinical NLI review requirement."""

        return True

    @property
    def label(self) -> str | None:
        """Return the effective label after the experiencer safety gate."""

        if self.review_required:
            return "review_required"
        return self.predicted_label

    @property
    def effective_label(self) -> str | None:
        """Return :attr:`label` under an explicit classifier alias."""

        return self.label

    @property
    def pair_id(self) -> str:
        """Return a deterministic identifier derived from safe fingerprints."""

        return stable_hash(self._fingerprint_payload())

    def _fingerprint_payload(self) -> dict[str, Any]:
        """Return the value-free material used for the stable pair id."""

        return {
            "schema_version": self.schema_version,
            "premise_hash": hash_text(self.premise),
            "hypothesis_hash": hash_text(self.hypothesis),
            "premise_experiencer": self.premise_experiencer_metadata.to_dict(),
            "hypothesis_experiencer": self.hypothesis_experiencer_metadata.to_dict(),
            "premise_offset": (
                list(self.premise_offset) if self.premise_offset is not None else None
            ),
            "hypothesis_offset": (
                list(self.hypothesis_offset)
                if self.hypothesis_offset is not None
                else None
            ),
            "predicted_label": self.predicted_label,
            "predicted_score": self.predicted_score,
        }

    def to_model_input(self) -> dict[str, Any]:
        """Return raw text and sidecars for a caller-controlled local model."""

        payload: dict[str, Any] = {
            "premise": self.premise,
            "hypothesis": self.hypothesis,
            "premise_experiencer": self.premise_experiencer_metadata.to_dict(),
            "hypothesis_experiencer": self.hypothesis_experiencer_metadata.to_dict(),
        }
        if self.predicted_label is not None:
            payload["predicted_label"] = self.predicted_label
        if self.predicted_score is not None:
            payload["predicted_score"] = self.predicted_score
        return payload

    def to_text_pair(self) -> tuple[str, str]:
        """Return only the raw text tuple for a local backend adapter."""

        return self.premise, self.hypothesis

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic pair metadata without raw source values."""

        return {
            "schema_version": self.schema_version,
            "pair_id": self.pair_id,
            "premise": {
                "text_hash": hash_text(self.premise),
                "text_length": len(self.premise),
                "offset": list(self.premise_offset)
                if self.premise_offset is not None
                else None,
                "experiencer": self.premise_experiencer_metadata.to_dict(),
                "experiencer_class": self.premise_experiencer_class,
            },
            "hypothesis": {
                "text_hash": hash_text(self.hypothesis),
                "text_length": len(self.hypothesis),
                "offset": list(self.hypothesis_offset)
                if self.hypothesis_offset is not None
                else None,
                "experiencer": self.hypothesis_experiencer_metadata.to_dict(),
                "experiencer_class": self.hypothesis_experiencer_class,
            },
            "experiencer_metadata": self.experiencer_metadata,
            "experiencer_compatibility": self.experiencer_compatibility,
            "experiencer_reason": self.experiencer_reason,
            "entailment_allowed": self.entailment_allowed,
            "entailment_blocked": self.entailment_blocked,
            "review_required": self.review_required,
            "requires_clinician_review": self.requires_clinician_review,
            "label": self.label,
            "predicted_label": self.predicted_label,
            "predicted_score": self.predicted_score,
            "advisory": NLI_EXPERIENCER_PAIR_ADVISORY,
        }

    def to_audit_dict(self) -> dict[str, Any]:
        """Return the PHI-safe representation under an audit-specific name."""

        return self.to_dict()

    def to_json(self) -> str:
        """Serialize pair metadata with stable key ordering."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def __repr__(self) -> str:
        return (
            "NliExperiencerPair("
            f"pair_id={self.pair_id!r}, "
            f"premise_hash={hash_text(self.premise)!r}, "
            f"hypothesis_hash={hash_text(self.hypothesis)!r}, "
            f"experiencer_compatibility={self.experiencer_compatibility!r}, "
            f"label={self.label!r})"
        )


ExperiencerAwareNliPair = NliExperiencerPair
ExperiencerNliPair = NliExperiencerPair
ClinicalNliExperiencerPair = NliExperiencerPair
ClinicalExperiencerNliPair = NliExperiencerPair
ClinicalNliPair = NliExperiencerPair
NliPair = NliExperiencerPair


def _alias_value(values: Sequence[object], *, field_name: str) -> object:
    """Resolve equal aliases without exposing their values in errors."""

    present = [value for value in values if value is not _MISSING]
    if not present:
        return _MISSING
    first = present[0]
    if any(value != first for value in present[1:]):
        raise _inconsistent(field_name)
    return first


def build_experiencer_nli_pair(
    premise: object,
    hypothesis: object,
    *,
    premise_experiencer: object = _MISSING,
    hypothesis_experiencer: object = _MISSING,
    premise_experiencer_metadata: object = _MISSING,
    hypothesis_experiencer_metadata: object = _MISSING,
    premise_metadata: object = _MISSING,
    hypothesis_metadata: object = _MISSING,
    premise_offset: object = _MISSING,
    hypothesis_offset: object = _MISSING,
    predicted_label: object = _MISSING,
    nli_label: object = _MISSING,
    label: object = _MISSING,
    predicted_score: object = _MISSING,
    nli_score: object = _MISSING,
    score: object = _MISSING,
) -> NliExperiencerPair:
    """Construct one experiencer-aware clinical NLI pair.

    Each side may be a string plus explicit experiencer keyword arguments, or
    a span-like mapping carrying ``text`` and direct/nested experiencer
    metadata.  Missing or unresolved subjects become ``unknown`` and block
    entailment.  Different known classes are incompatible and also block
    entailment.  No model, filesystem, network, clock, or provider is used.
    """

    premise_text, premise_mapping = _coerce_text(premise, side_name="premise")
    hypothesis_text, hypothesis_mapping = _coerce_text(
        hypothesis,
        side_name="hypothesis",
    )
    premise_metadata_values = [
        value
        for value in (
            premise_experiencer,
            premise_experiencer_metadata,
            premise_metadata,
        )
        if value is not _MISSING
    ]
    hypothesis_metadata_values = [
        value
        for value in (
            hypothesis_experiencer,
            hypothesis_experiencer_metadata,
            hypothesis_metadata,
        )
        if value is not _MISSING
    ]
    resolved_premise = _merge_side_experiencer(
        premise,
        premise_metadata_values,
        side_name="premise",
    )
    resolved_hypothesis = _merge_side_experiencer(
        hypothesis,
        hypothesis_metadata_values,
        side_name="hypothesis",
    )

    resolved_premise_offset = (
        _coerce_offset(premise_offset, field_name="premise offset")
        if premise_offset is not _MISSING
        else _offset_from_mapping(premise_mapping, side_name="premise")
    )
    resolved_hypothesis_offset = (
        _coerce_offset(hypothesis_offset, field_name="hypothesis offset")
        if hypothesis_offset is not _MISSING
        else _offset_from_mapping(hypothesis_mapping, side_name="hypothesis")
    )
    label_values = tuple(
        _MISSING if value is _MISSING else _normalize_label(value)
        for value in (predicted_label, nli_label, label)
    )
    resolved_label = _alias_value(
        label_values,
        field_name="NLI label",
    )
    score_values = tuple(
        _MISSING if value is _MISSING else _normalize_score(value)
        for value in (predicted_score, nli_score, score)
    )
    resolved_score = _alias_value(
        score_values,
        field_name="NLI score",
    )
    return NliExperiencerPair(
        premise=premise_text,
        hypothesis=hypothesis_text,
        premise_experiencer=resolved_premise,
        hypothesis_experiencer=resolved_hypothesis,
        premise_offset=resolved_premise_offset,
        hypothesis_offset=resolved_hypothesis_offset,
        predicted_label=None
        if resolved_label is _MISSING
        else cast(str, resolved_label),
        predicted_score=None
        if resolved_score is _MISSING
        else cast(float, resolved_score),
    )


build_nli_experiencer_pair = build_experiencer_nli_pair
build_experiencer_aware_pair = build_experiencer_nli_pair
build_experiencer_aware_nli_pair = build_experiencer_nli_pair
construct_experiencer_nli_pair = build_experiencer_nli_pair
construct_experiencer_aware_pair = build_experiencer_nli_pair
build_nli_pair = build_experiencer_nli_pair


def build_experiencer_nli_pairs(
    pair_inputs: Iterable[object],
    **defaults: object,
) -> tuple[NliExperiencerPair, ...]:
    """Build pairs in input order without deduplication or external I/O.

    An item may already be a pair, a mapping with ``premise`` and
    ``hypothesis`` fields, or a two-item sequence.  Keyword defaults are
    applied to sequence items and overridden by explicit mapping fields.
    """

    pairs: list[NliExperiencerPair] = []
    parameter_names = (
        "premise_experiencer",
        "hypothesis_experiencer",
        "premise_experiencer_metadata",
        "hypothesis_experiencer_metadata",
        "premise_metadata",
        "hypothesis_metadata",
        "premise_offset",
        "hypothesis_offset",
        "predicted_label",
        "nli_label",
        "label",
        "predicted_score",
        "nli_score",
        "score",
    )
    for item in pair_inputs:
        if isinstance(item, NliExperiencerPair):
            pairs.append(item)
            continue
        if isinstance(item, Mapping):
            premise = item.get("premise", _MISSING)
            hypothesis = item.get("hypothesis", _MISSING)
            if premise is _MISSING or hypothesis is _MISSING:
                raise _invalid("pair premise and hypothesis")
            parameters = dict(defaults)
            for name in parameter_names:
                if name in item:
                    parameters[name] = item[name]
            pairs.append(build_experiencer_nli_pair(premise, hypothesis, **parameters))
            continue
        if (
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes, bytearray))
            and len(item) == 2
        ):
            pairs.append(build_experiencer_nli_pair(item[0], item[1], **defaults))
            continue
        raise _invalid("pair input")
    return tuple(pairs)


build_nli_experiencer_pairs = build_experiencer_nli_pairs
build_experiencer_aware_pairs = build_experiencer_nli_pairs
build_experiencer_aware_nli_pairs = build_experiencer_nli_pairs
construct_experiencer_nli_pairs = build_experiencer_nli_pairs
construct_experiencer_aware_pairs = build_experiencer_nli_pairs
build_nli_pairs = build_experiencer_nli_pairs


def validate_experiencer_nli_pair(value: object) -> NliExperiencerPair:
    """Revalidate a pair through the immutable, value-free boundary."""

    if not isinstance(value, NliExperiencerPair):
        raise _invalid("NLI experiencer pair")
    return NliExperiencerPair(
        premise=value.premise,
        hypothesis=value.hypothesis,
        premise_experiencer=ExperiencerMetadata.from_value(value.premise_experiencer),
        hypothesis_experiencer=ExperiencerMetadata.from_value(
            value.hypothesis_experiencer
        ),
        premise_offset=value.premise_offset,
        hypothesis_offset=value.hypothesis_offset,
        predicted_label=value.predicted_label,
        predicted_score=value.predicted_score,
        schema_version=value.schema_version,
    )


validate_nli_experiencer_pair = validate_experiencer_nli_pair
validate_experiencer_aware_pair = validate_experiencer_nli_pair
validate_nli_pair = validate_experiencer_nli_pair


__all__ = [
    "NLI_EXPERIENCER_PAIR_SCHEMA_VERSION",
    "NLI_EXPERIENCER_PAIR_ADVISORY",
    "CLINICAL_NLI_EXPERIENCER_PAIR_ADVISORY",
    "CAREGIVER_EXPERIENCER",
    "UNKNOWN_EXPERIENCER",
    "NliExperiencerClass",
    "NLI_EXPERIENCER_VALUES",
    "EXPERIENCER_NLI_VALUES",
    "NLI_EXPERIENCER_CLASSES",
    "EXPERIENCER_COMPATIBILITY_VALUES",
    "SpanOffset",
    "ExperiencerSource",
    "ExperiencerCompatibility",
    "NliExperiencerPairError",
    "InconsistentExperiencerMetadataError",
    "NliExperiencerPairValidationError",
    "ExperiencerMetadata",
    "NliExperiencerMetadata",
    "ClinicalNliExperiencerMetadata",
    "ExperiencerComparison",
    "compare_experiencers",
    "compare_experiencer_classes",
    "classify_experiencer_compatibility",
    "NliExperiencerPair",
    "ExperiencerAwareNliPair",
    "ExperiencerNliPair",
    "ClinicalNliExperiencerPair",
    "ClinicalExperiencerNliPair",
    "ClinicalNliPair",
    "NliPair",
    "validate_experiencer_metadata",
    "build_experiencer_nli_pair",
    "build_nli_experiencer_pair",
    "build_experiencer_aware_pair",
    "build_experiencer_aware_nli_pair",
    "construct_experiencer_nli_pair",
    "construct_experiencer_aware_pair",
    "build_nli_pair",
    "build_experiencer_nli_pairs",
    "build_nli_experiencer_pairs",
    "build_experiencer_aware_pairs",
    "build_experiencer_aware_nli_pairs",
    "construct_experiencer_nli_pairs",
    "construct_experiencer_aware_pairs",
    "build_nli_pairs",
    "validate_experiencer_nli_pair",
    "validate_nli_experiencer_pair",
    "validate_experiencer_aware_pair",
    "validate_nli_pair",
]
