"""Assertion-aware clinical NLI pair construction.

Clinical NLI must not silently treat a negated, uncertain, or conditional
source span as affirmed evidence.  This module provides a small, dependency-
free boundary for constructing premise/hypothesis pairs with explicit
assertion metadata on both sides.

The pair retains source text only for the transient model-input boundary.  Its
representations, JSON serialization, and validation errors are deliberately
PHI-safe: they contain assertion axes, offsets, lengths, and SHA-256 digests,
never raw premise or hypothesis text.  The pair is an assistive review input,
not a clinical decision or an NLI result.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias, cast

from openmed.core.audit import hash_text, stable_hash

from .context import (
    AFFIRMED,
    CERTAIN,
    CERTAINTY_VALUES,
    HYPOTHETICAL,
    NEGATED,
    NEGATION_VALUES,
    RECENT,
    TEMPORALITY_VALUES,
    UNCERTAIN,
    ClinicalAssertion,
)

NLI_ASSERTION_PAIR_SCHEMA_VERSION: Final = 1
NLI_ASSERTION_PAIR_ADVISORY: Final = (
    "Assertion-aware clinical NLI pairs are deterministic assistive inputs for "
    "qualified human review. They do not diagnose, recommend treatment, or "
    "make an autonomous clinical decision."
)
# A descriptive alias for callers that use the repository's disclaimer naming.
CLINICAL_NLI_ASSERTION_PAIR_ADVISORY: Final = NLI_ASSERTION_PAIR_ADVISORY

AssertionNegation: TypeAlias = Literal["affirmed", "negated"]
AssertionCertainty: TypeAlias = Literal["certain", "uncertain"]
AssertionTemporality: TypeAlias = Literal[
    "recent",
    "historical",
    "hypothetical",
]
SpanOffset: TypeAlias = tuple[int, int]

_MISSING = object()
_ASSERTION_CONTAINER_KEYS = (
    "assertion",
    "clinical_assertion",
    "assertion_metadata",
    "clinical_context",
)
_METADATA_CONTAINER_KEYS = ("metadata", "context")
_TEXT_KEYS = ("text", "surface", "content", "value")
_OFFSET_KEYS = ("offset", "span", "source_offset")
_START_KEYS = ("start", "source_start", "start_offset")
_END_KEYS = ("end", "source_end", "end_offset")
_EXPERIENCER_VALUES = frozenset({"patient", "family", "other"})


class NliAssertionPairError(ValueError):
    """Base error for malformed assertion-aware NLI pair inputs."""


class MissingAssertionMetadataError(NliAssertionPairError):
    """Raised when one side of a pair has no complete assertion metadata."""


class InconsistentAssertionMetadataError(NliAssertionPairError):
    """Raised when redundant assertion fields disagree."""


class NliAssertionPairValidationError(NliAssertionPairError):
    """Raised when pair text, offsets, or schema data is invalid."""


def _invalid(field_name: str) -> NliAssertionPairError:
    """Build a value-free validation error for ``field_name``."""

    return NliAssertionPairValidationError(f"{field_name} is invalid")


def _missing(field_name: str) -> MissingAssertionMetadataError:
    """Build a value-free missing-field error."""

    return MissingAssertionMetadataError(f"{field_name} is required")


def _inconsistent(field_name: str) -> InconsistentAssertionMetadataError:
    """Build a value-free inconsistency error."""

    return InconsistentAssertionMetadataError(f"{field_name} is inconsistent")


def _normalize_enum(
    value: object,
    *,
    field_name: str,
    allowed: Sequence[str],
) -> str:
    if type(value) is not str:
        raise _invalid(field_name)
    normalized = value.strip().casefold()
    if normalized not in allowed:
        raise _invalid(field_name)
    return normalized


def _normalize_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise _invalid(field_name)
    return value


def _first_value(source: Mapping[str, object], keys: Sequence[str]) -> object:
    for key in keys:
        if key in source:
            return source[key]
    return _MISSING


def _mapping_from_object(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    if isinstance(value, ClinicalAssertion):
        return {
            "negation": value.negation,
            "certainty": value.certainty,
            "temporality": value.temporality,
            "experiencer": value.experiencer,
        }

    fields: dict[str, object] = {}
    for name in (
        "negation",
        "negated",
        "is_negated",
        "certainty",
        "uncertainty",
        "uncertain",
        "is_uncertain",
        "certain",
        "temporality",
        "hypothetical",
        "is_hypothetical",
        "experiencer",
    ):
        try:
            candidate = getattr(value, name, _MISSING)
        except Exception:
            continue
        if candidate is not _MISSING:
            fields[name] = candidate
    return fields or None


def _nested_assertion_sources(value: object) -> tuple[object, ...]:
    """Return direct and nested assertion containers without reading text."""

    source = _mapping_from_object(value)
    if source is None:
        return ()

    nested: list[object] = []
    for key in _ASSERTION_CONTAINER_KEYS:
        candidate = source.get(key, _MISSING)
        if candidate is not _MISSING:
            nested.append(candidate)

    for key in _METADATA_CONTAINER_KEYS:
        container = source.get(key, _MISSING)
        if container is _MISSING:
            continue
        container_mapping = _mapping_from_object(container)
        if container_mapping is None:
            continue
        for assertion_key in _ASSERTION_CONTAINER_KEYS:
            candidate = container_mapping.get(assertion_key, _MISSING)
            if candidate is not _MISSING:
                nested.append(candidate)
        nested.append(container_mapping)

    nested.append(source)
    return tuple(nested)


def _coerce_negation_fields(
    source: Mapping[str, object],
) -> AssertionNegation | None:
    values: list[AssertionNegation] = []
    raw_negation = source.get("negation", _MISSING)
    if raw_negation is not _MISSING:
        if raw_negation is None:
            raise _missing("negation")
        values.append(
            cast(
                AssertionNegation,
                _normalize_enum(
                    raw_negation,
                    field_name="negation",
                    allowed=NEGATION_VALUES,
                ),
            )
        )

    for key in ("negated", "is_negated"):
        raw_value = source.get(key, _MISSING)
        if raw_value is not _MISSING:
            normalized = _normalize_bool(raw_value, field_name=key)
            values.append(NEGATED if normalized else AFFIRMED)

    if not values:
        return None
    if len(set(values)) != 1:
        raise _inconsistent("negation")
    return values[0]


def _certainty_from_raw(value: object, *, field_name: str) -> AssertionCertainty:
    if type(value) is bool:
        return UNCERTAIN if value else CERTAIN
    return cast(
        AssertionCertainty,
        _normalize_enum(
            value,
            field_name=field_name,
            allowed=CERTAINTY_VALUES,
        ),
    )


def _coerce_certainty_fields(
    source: Mapping[str, object],
) -> AssertionCertainty | None:
    values: list[AssertionCertainty] = []
    raw_certainty = source.get("certainty", _MISSING)
    if raw_certainty is not _MISSING:
        if raw_certainty is None:
            raise _missing("certainty")
        values.append(_certainty_from_raw(raw_certainty, field_name="certainty"))

    raw_uncertainty = source.get("uncertainty", _MISSING)
    if raw_uncertainty is not _MISSING:
        if raw_uncertainty is None:
            raise _missing("uncertainty")
        if type(raw_uncertainty) is bool:
            values.append(
                UNCERTAIN if raw_uncertainty else CERTAIN,
            )
        else:
            values.append(
                _certainty_from_raw(raw_uncertainty, field_name="uncertainty"),
            )

    for key in ("uncertain", "is_uncertain"):
        raw_value = source.get(key, _MISSING)
        if raw_value is not _MISSING:
            normalized = _normalize_bool(raw_value, field_name=key)
            values.append(UNCERTAIN if normalized else CERTAIN)

    raw_certain = source.get("certain", _MISSING)
    if raw_certain is not _MISSING:
        normalized = _normalize_bool(raw_certain, field_name="certain")
        values.append(CERTAIN if normalized else UNCERTAIN)

    if not values:
        return None
    if len(set(values)) != 1:
        raise _inconsistent("certainty/uncertainty")
    return values[0]


def _coerce_temporality_fields(
    source: Mapping[str, object],
) -> AssertionTemporality | None:
    values: list[AssertionTemporality] = []
    raw_temporality = source.get("temporality", _MISSING)
    if raw_temporality is not _MISSING:
        if raw_temporality is None:
            raise _missing("temporality")
        values.append(
            cast(
                AssertionTemporality,
                _normalize_enum(
                    raw_temporality,
                    field_name="temporality",
                    allowed=TEMPORALITY_VALUES,
                ),
            )
        )

    for key in ("hypothetical", "is_hypothetical"):
        raw_value = source.get(key, _MISSING)
        if raw_value is not _MISSING:
            normalized = _normalize_bool(raw_value, field_name=key)
            values.append(HYPOTHETICAL if normalized else RECENT)

    if not values:
        return None
    if len(set(values)) != 1:
        raise _inconsistent("temporality/hypothetical")
    return values[0]


def _coerce_experiencer(source: Mapping[str, object]) -> str | None:
    value = source.get("experiencer", _MISSING)
    if value is _MISSING or value is None:
        return None
    if type(value) is not str or not value.strip():
        raise _invalid("experiencer")
    normalized = value.strip().casefold()
    if normalized not in _EXPERIENCER_VALUES:
        raise _invalid("experiencer")
    return normalized


def _parse_assertion_source(
    source: object,
) -> dict[str, str | None] | None:
    mapping = _mapping_from_object(source)
    if mapping is None:
        raise NliAssertionPairValidationError("assertion metadata is invalid")
    negation = _coerce_negation_fields(mapping)
    certainty = _coerce_certainty_fields(mapping)
    temporality = _coerce_temporality_fields(mapping)
    experiencer = _coerce_experiencer(mapping)
    if negation is None and certainty is None and temporality is None:
        return None
    return {
        "negation": negation,
        "certainty": certainty,
        "temporality": temporality,
        "experiencer": experiencer,
    }


def _merge_assertion_fields(
    fields: Iterable[dict[str, str | None] | None],
    *,
    side_name: str,
) -> dict[str, str | None]:
    merged: dict[str, str | None] = {}
    found = False
    for parsed in fields:
        if parsed is None:
            continue
        found = True
        for key, value in parsed.items():
            if value is None:
                continue
            previous = merged.get(key)
            if previous is not None and previous != value:
                raise _inconsistent(f"{side_name} assertion metadata")
            merged[key] = value

    if not found:
        raise _missing(f"{side_name} assertion metadata")
    for key in ("negation", "certainty", "temporality"):
        if merged.get(key) is None:
            raise _missing(f"{side_name} assertion {key}")
    return merged


@dataclass(frozen=True, slots=True)
class AssertionMetadata:
    """Validated assertion axes attached to one NLI pair side.

    ``certainty`` is the canonical OpenMed axis for uncertainty.  The
    :attr:`uncertainty` property exposes the same value under the vocabulary
    used by NLI callers, while :attr:`hypothetical` makes the conditional
    temporality explicit without introducing a second mutable source of truth.
    """

    negation: AssertionNegation
    certainty: AssertionCertainty
    temporality: AssertionTemporality
    experiencer: str | None = None

    def __post_init__(self) -> None:
        normalized_negation = cast(
            AssertionNegation,
            _normalize_enum(
                self.negation,
                field_name="negation",
                allowed=NEGATION_VALUES,
            ),
        )
        normalized_certainty = cast(
            AssertionCertainty,
            _normalize_enum(
                self.certainty,
                field_name="certainty",
                allowed=CERTAINTY_VALUES,
            ),
        )
        normalized_temporality = cast(
            AssertionTemporality,
            _normalize_enum(
                self.temporality,
                field_name="temporality",
                allowed=TEMPORALITY_VALUES,
            ),
        )
        object.__setattr__(self, "negation", normalized_negation)
        object.__setattr__(self, "certainty", normalized_certainty)
        object.__setattr__(self, "temporality", normalized_temporality)
        if self.experiencer is not None:
            if type(self.experiencer) is not str or not self.experiencer.strip():
                raise _invalid("experiencer")
            normalized_experiencer = self.experiencer.strip().casefold()
            if normalized_experiencer not in _EXPERIENCER_VALUES:
                raise _invalid("experiencer")
            object.__setattr__(self, "experiencer", normalized_experiencer)

    @classmethod
    def from_value(cls, value: object, *, side_name: str = "assertion"):
        """Coerce a complete assertion from a record or context mapping.

        Accepted records may be :class:`ClinicalAssertion`, an
        :class:`AssertionMetadata`, a mapping with direct axes, or a span-like
        mapping containing ``assertion``/``clinical_context`` metadata.  When
        aliases such as ``uncertainty``/``certainty`` or
        ``hypothetical``/``temporality`` are both present, they must agree.
        """

        if isinstance(value, cls):
            return value

        sources = _nested_assertion_sources(value)
        parsed_sources = tuple(
            _parse_assertion_source(source)
            for source in sources
            if _mapping_from_object(source) is not None
        )
        fields = _merge_assertion_fields(parsed_sources, side_name=side_name)
        return cls(
            negation=cast(AssertionNegation, fields["negation"]),
            certainty=cast(AssertionCertainty, fields["certainty"]),
            temporality=cast(AssertionTemporality, fields["temporality"]),
            experiencer=fields.get("experiencer"),
        )

    @property
    def uncertainty(self) -> AssertionCertainty:
        """Return the certainty axis using NLI's uncertainty vocabulary."""

        return self.certainty

    @property
    def uncertain(self) -> bool:
        """Return whether the assertion is explicitly uncertain."""

        return self.certainty == UNCERTAIN

    @property
    def hypothetical(self) -> bool:
        """Return whether the assertion is conditional/non-present."""

        return self.temporality == HYPOTHETICAL

    @property
    def negated(self) -> bool:
        """Return whether the assertion is explicitly negated."""

        return self.negation == NEGATED

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, text-free assertion metadata."""

        payload: dict[str, Any] = {
            "negation": self.negation,
            "certainty": self.certainty,
            "temporality": self.temporality,
            "uncertainty": self.uncertainty,
            "hypothetical": self.hypothetical,
        }
        if self.experiencer is not None:
            payload["experiencer"] = self.experiencer
        return payload


NliAssertionMetadata = AssertionMetadata
ClinicalNliAssertionMetadata = AssertionMetadata


def validate_assertion_metadata(
    value: object,
    *,
    side_name: str = "assertion",
) -> AssertionMetadata:
    """Validate and return one complete assertion metadata record."""

    return AssertionMetadata.from_value(value, side_name=side_name)


def _coerce_text(
    value: object, *, side_name: str
) -> tuple[str, Mapping[str, object] | None]:
    if type(value) is str:
        text = value
        mapping = None
    elif isinstance(value, str):
        raise NliAssertionPairValidationError(f"{side_name} text is invalid")
    else:
        mapping = _mapping_from_object(value)
        if mapping is None:
            raise NliAssertionPairValidationError(f"{side_name} text is invalid")
        raw_text = _first_value(mapping, _TEXT_KEYS)
        if raw_text is _MISSING:
            raise _missing(f"{side_name} text")
        if type(raw_text) is not str:
            raise NliAssertionPairValidationError(f"{side_name} text is invalid")
        text = raw_text
    if not text.strip():
        raise _missing(f"{side_name} text")
    return text, mapping


def _coerce_offset(value: object, *, field_name: str) -> SpanOffset:
    if isinstance(value, Mapping):
        raw_start = _first_value(cast(Mapping[str, object], value), _START_KEYS)
        raw_end = _first_value(cast(Mapping[str, object], value), _END_KEYS)
        if raw_start is _MISSING or raw_end is _MISSING:
            raise _invalid(field_name)
        values: object = (raw_start, raw_end)
    else:
        values = value
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray))
        or len(values) != 2
    ):
        raise _invalid(field_name)
    start, end = values
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise _invalid(field_name)
    return start, end


def _offset_from_mapping(
    mapping: Mapping[str, object] | None,
    *,
    side_name: str,
) -> SpanOffset | None:
    if mapping is None:
        return None
    candidates: list[SpanOffset] = []
    for key in _OFFSET_KEYS:
        value = mapping.get(key, _MISSING)
        if value is not _MISSING:
            candidates.append(
                _coerce_offset(value, field_name=f"{side_name} offset"),
            )
    raw_start = _first_value(mapping, _START_KEYS)
    raw_end = _first_value(mapping, _END_KEYS)
    if raw_start is not _MISSING or raw_end is not _MISSING:
        if raw_start is _MISSING or raw_end is _MISSING:
            raise _invalid(f"{side_name} offset")
        candidates.append(
            _coerce_offset((raw_start, raw_end), field_name=f"{side_name} offset"),
        )
    if not candidates:
        return None
    if len(set(candidates)) != 1:
        raise _inconsistent(f"{side_name} offset")
    return candidates[0]


def _merge_side_assertion(
    side_value: object,
    explicit_assertion: object,
    *,
    side_name: str,
) -> AssertionMetadata:
    sources: list[object] = []
    if explicit_assertion is not _MISSING:
        sources.append(explicit_assertion)
    for source in _nested_assertion_sources(side_value):
        sources.append(source)
    if not sources:
        raise _missing(f"{side_name} assertion metadata")

    parsed = tuple(
        _parse_assertion_source(source)
        for source in sources
        if _mapping_from_object(source) is not None
    )
    fields = _merge_assertion_fields(parsed, side_name=side_name)
    return AssertionMetadata(
        negation=cast(AssertionNegation, fields["negation"]),
        certainty=cast(AssertionCertainty, fields["certainty"]),
        temporality=cast(AssertionTemporality, fields["temporality"]),
        experiencer=fields.get("experiencer"),
    )


@dataclass(frozen=True, slots=True, repr=False)
class NliAssertionPair:
    """One clinical NLI pair with explicit assertion state on both sides.

    ``premise`` and ``hypothesis`` are retained for a caller-controlled local
    model-input boundary.  Use :meth:`to_model_input` only when sending the
    pair to that local boundary.  ``to_dict()``, ``to_json()``, and ``repr``
    expose only hashes, lengths, offsets, and assertion metadata.
    """

    premise: str
    hypothesis: str
    premise_assertion: AssertionMetadata
    hypothesis_assertion: AssertionMetadata
    premise_offset: SpanOffset | None = None
    hypothesis_offset: SpanOffset | None = None
    schema_version: int = NLI_ASSERTION_PAIR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.premise) is not str or not self.premise.strip():
            raise _missing("premise text")
        if type(self.hypothesis) is not str or not self.hypothesis.strip():
            raise _missing("hypothesis text")
        if not isinstance(self.premise_assertion, AssertionMetadata):
            raise _invalid("premise assertion metadata")
        if not isinstance(self.hypothesis_assertion, AssertionMetadata):
            raise _invalid("hypothesis assertion metadata")
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
        if type(self.schema_version) is not int or (
            self.schema_version != NLI_ASSERTION_PAIR_SCHEMA_VERSION
        ):
            raise _invalid("schema_version")

    @property
    def requires_clinician_review(self) -> Literal[True]:
        """Return the literal review requirement for clinical NLI inputs."""

        return True

    @property
    def pair_id(self) -> str:
        """Return a deterministic identifier derived from safe fingerprints."""

        return stable_hash(self._fingerprint_payload())

    @property
    def assertion_metadata(self) -> dict[str, dict[str, Any]]:
        """Return assertion metadata for both sides without source text."""

        return {
            "premise": self.premise_assertion.to_dict(),
            "hypothesis": self.hypothesis_assertion.to_dict(),
        }

    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "premise_hash": hash_text(self.premise),
            "hypothesis_hash": hash_text(self.hypothesis),
            "premise_assertion": self.premise_assertion.to_dict(),
            "hypothesis_assertion": self.hypothesis_assertion.to_dict(),
            "premise_offset": list(self.premise_offset)
            if self.premise_offset is not None
            else None,
            "hypothesis_offset": list(self.hypothesis_offset)
            if self.hypothesis_offset is not None
            else None,
        }

    def to_model_input(self) -> dict[str, Any]:
        """Return local model text together with its assertion sidecars.

        A backend that accepts only a text tuple can use
        :meth:`to_text_pair`, but callers must retain this pair's assertion
        sidecars alongside any NLI result.
        """

        return {
            "premise": self.premise,
            "hypothesis": self.hypothesis,
            "premise_assertion": self.premise_assertion.to_dict(),
            "hypothesis_assertion": self.hypothesis_assertion.to_dict(),
        }

    def to_text_pair(self) -> tuple[str, str]:
        """Return only the raw text tuple for a local backend adapter."""

        return self.premise, self.hypothesis

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic PHI-safe audit representation."""

        return {
            "schema_version": self.schema_version,
            "pair_id": self.pair_id,
            "premise": {
                "text_hash": hash_text(self.premise),
                "text_length": len(self.premise),
                "offset": list(self.premise_offset)
                if self.premise_offset is not None
                else None,
                "assertion": self.premise_assertion.to_dict(),
            },
            "hypothesis": {
                "text_hash": hash_text(self.hypothesis),
                "text_length": len(self.hypothesis),
                "offset": list(self.hypothesis_offset)
                if self.hypothesis_offset is not None
                else None,
                "assertion": self.hypothesis_assertion.to_dict(),
            },
            "requires_clinician_review": True,
            "advisory": NLI_ASSERTION_PAIR_ADVISORY,
        }

    def to_audit_dict(self) -> dict[str, Any]:
        """Return the PHI-safe serialization under an explicit audit name."""

        return self.to_dict()

    def to_json(self) -> str:
        """Serialize the PHI-safe pair representation deterministically."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def __repr__(self) -> str:
        return (
            "NliAssertionPair("
            f"pair_id={self.pair_id!r}, "
            f"premise_hash={hash_text(self.premise)!r}, "
            f"hypothesis_hash={hash_text(self.hypothesis)!r}, "
            f"premise_assertion={self.premise_assertion!r}, "
            f"hypothesis_assertion={self.hypothesis_assertion!r})"
        )


AssertionAwareNliPair = NliAssertionPair
ClinicalNliPair = NliAssertionPair
NliPair = NliAssertionPair


def build_nli_pair(
    premise: object,
    hypothesis: object,
    *,
    premise_assertion: object = _MISSING,
    hypothesis_assertion: object = _MISSING,
    premise_offset: object = _MISSING,
    hypothesis_offset: object = _MISSING,
) -> NliAssertionPair:
    """Construct one assertion-aware clinical NLI pair.

    ``premise`` and ``hypothesis`` may be strings plus explicit assertion
    keyword arguments, or span-like records carrying ``text`` and nested or
    direct assertion fields.  Both sides must provide all three canonical
    axes: ``negation``, ``certainty``/``uncertainty``, and
    ``temporality``/``hypothetical``.  Conflicting redundant fields fail
    closed before the pair is returned.

    No model, filesystem, network, logging, or remote provider is consulted.
    """

    premise_text, premise_mapping = _coerce_text(premise, side_name="premise")
    hypothesis_text, hypothesis_mapping = _coerce_text(
        hypothesis,
        side_name="hypothesis",
    )
    resolved_premise_assertion = _merge_side_assertion(
        premise,
        premise_assertion,
        side_name="premise",
    )
    resolved_hypothesis_assertion = _merge_side_assertion(
        hypothesis,
        hypothesis_assertion,
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

    return NliAssertionPair(
        premise=premise_text,
        hypothesis=hypothesis_text,
        premise_assertion=resolved_premise_assertion,
        hypothesis_assertion=resolved_hypothesis_assertion,
        premise_offset=resolved_premise_offset,
        hypothesis_offset=resolved_hypothesis_offset,
    )


def build_assertion_aware_pair(*args: object, **kwargs: object) -> NliAssertionPair:
    """Alias for :func:`build_nli_pair` with the issue vocabulary."""

    return build_nli_pair(*args, **kwargs)


construct_assertion_aware_pair = build_assertion_aware_pair
build_assertion_aware_nli_pair = build_assertion_aware_pair


def build_nli_pairs(
    pair_inputs: Iterable[object],
) -> tuple[NliAssertionPair, ...]:
    """Build a deterministic tuple from explicit pair input records.

    Each item may already be an :class:`NliAssertionPair`, a mapping with
    ``premise`` and ``hypothesis`` fields (and optional side assertion fields),
    or a two-item sequence whose sides carry their own assertion metadata.
    Input order is preserved; no deduplication or network-backed lookup occurs.
    """

    pairs: list[NliAssertionPair] = []
    for item in pair_inputs:
        if isinstance(item, NliAssertionPair):
            pairs.append(item)
            continue
        if isinstance(item, Mapping):
            premise = item.get("premise", _MISSING)
            hypothesis = item.get("hypothesis", _MISSING)
            if premise is _MISSING or hypothesis is _MISSING:
                raise _missing("pair premise and hypothesis")
            pairs.append(
                build_nli_pair(
                    premise,
                    hypothesis,
                    premise_assertion=item.get("premise_assertion", _MISSING),
                    hypothesis_assertion=item.get(
                        "hypothesis_assertion",
                        _MISSING,
                    ),
                    premise_offset=item.get("premise_offset", _MISSING),
                    hypothesis_offset=item.get("hypothesis_offset", _MISSING),
                )
            )
            continue
        if (
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes, bytearray))
            and len(item) == 2
        ):
            pairs.append(build_nli_pair(item[0], item[1]))
            continue
        raise NliAssertionPairValidationError("pair input is invalid")
    return tuple(pairs)


build_assertion_aware_pairs = build_nli_pairs
construct_assertion_aware_pairs = build_nli_pairs


def validate_nli_pair(value: object) -> NliAssertionPair:
    """Validate and return an assertion-aware pair without exposing its text."""

    if not isinstance(value, NliAssertionPair):
        raise NliAssertionPairValidationError("NLI pair is invalid")
    # Re-run the immutable boundary so callers get the same fail-closed checks
    # even when a subclass or deserialized object is supplied.
    return NliAssertionPair(
        premise=value.premise,
        hypothesis=value.hypothesis,
        premise_assertion=validate_assertion_metadata(
            value.premise_assertion,
            side_name="premise",
        ),
        hypothesis_assertion=validate_assertion_metadata(
            value.hypothesis_assertion,
            side_name="hypothesis",
        ),
        premise_offset=value.premise_offset,
        hypothesis_offset=value.hypothesis_offset,
        schema_version=value.schema_version,
    )


__all__ = [
    "NLI_ASSERTION_PAIR_ADVISORY",
    "CLINICAL_NLI_ASSERTION_PAIR_ADVISORY",
    "NLI_ASSERTION_PAIR_SCHEMA_VERSION",
    "AssertionNegation",
    "AssertionCertainty",
    "AssertionTemporality",
    "SpanOffset",
    "NliAssertionPairError",
    "MissingAssertionMetadataError",
    "InconsistentAssertionMetadataError",
    "NliAssertionPairValidationError",
    "AssertionMetadata",
    "NliAssertionMetadata",
    "ClinicalNliAssertionMetadata",
    "NliAssertionPair",
    "AssertionAwareNliPair",
    "ClinicalNliPair",
    "NliPair",
    "validate_assertion_metadata",
    "build_nli_pair",
    "build_assertion_aware_pair",
    "construct_assertion_aware_pair",
    "build_assertion_aware_nli_pair",
    "build_nli_pairs",
    "build_assertion_aware_pairs",
    "construct_assertion_aware_pairs",
    "validate_nli_pair",
]
