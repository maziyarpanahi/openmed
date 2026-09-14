"""Offline section priors for lexical grounding candidates.

Section context is an advisory grounding signal.  It uses semantic-type
metadata supplied by the caller's vocabulary snapshot to remove implausible
matches, adjust deterministic ranking, and preserve the section and
experiencer that governed the result.  No terminology is downloaded and no
source span text is copied into the returned metadata.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TypeAlias

from ..context import canonical_section_label
from .matcher import ConceptMatch

__all__ = [
    "DEFAULT_SECTION_CONTEXT_CONFIG",
    "DEFAULT_SECTION_CONTEXT_RULES",
    "SECTION_CONTEXT_RULES",
    "SectionContextConfig",
    "SectionContextRule",
    "apply_section_context",
]


_PATIENT = "patient"
_SECTION_FIELDS = (
    "section",
    "section_label",
    "section_name",
    "canonical_section",
    "canonical_label",
    "label",
    "name",
)
_SEMANTIC_TYPE_FIELDS = (
    "semantic_type",
    "semantic_types",
    "semanticType",
    "semanticTypes",
    "concept_type",
    "concept_types",
    "kind",
    "type",
    "types",
    "category",
    "domain",
)
_VALUE_SPLIT_RE = re.compile(r"[,|;]")


def _normalize_key(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} values must be strings")
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = re.sub(r"[^\w]+", "_", normalized, flags=re.UNICODE)
    normalized = normalized.strip("_")
    if not normalized:
        raise ValueError(f"{name} values must not be empty")
    return normalized


def _normalize_section(value: object) -> str | None:
    text = _section_text(value)
    if text is None:
        return None
    canonical = canonical_section_label(text)
    if canonical:
        return canonical
    return _normalize_key(text, name="section")


def _section_text(value: object) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Mapping):
        for field_name in _SECTION_FIELDS:
            candidate = value.get(field_name)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None
    for field_name in _SECTION_FIELDS:
        candidate = getattr(value, field_name, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _normalize_semantic_type(value: object) -> str:
    return _normalize_key(value, name="semantic type")


def _normalize_experiencer(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("experiencer must be a non-empty string when provided")
    return _normalize_key(value, name="experiencer")


@dataclass(frozen=True)
class SectionContextRule:
    """Configurable grounding policy for one canonical note section.

    ``semantic_type_biases`` contains additive score adjustments keyed by the
    semantic type recorded in a concept's metadata.  ``excluded_semantic_types``
    are hard exclusions for this section.  ``default_bias`` applies to a match
    whose type has no explicit bias, while ``non_patient_bias`` is applied when
    the resolved experiencer is not the patient.
    """

    semantic_type_biases: Mapping[str, float] = field(default_factory=dict)
    excluded_semantic_types: frozenset[str] = frozenset()
    default_bias: float = 0.0
    experiencer: str | None = None
    non_patient_bias: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.semantic_type_biases, Mapping):
            raise TypeError("semantic_type_biases must be a mapping")
        biases: dict[str, float] = {}
        for raw_type, raw_bias in self.semantic_type_biases.items():
            semantic_type = _normalize_semantic_type(raw_type)
            bias = float(raw_bias)
            if not math.isfinite(bias):
                raise ValueError("semantic type biases must be finite")
            biases[semantic_type] = bias
        object.__setattr__(self, "semantic_type_biases", MappingProxyType(biases))

        excluded = self.excluded_semantic_types
        if isinstance(excluded, str):
            excluded_values = (excluded,)
        else:
            try:
                excluded_values = tuple(excluded)
            except TypeError as exc:
                raise TypeError(
                    "excluded_semantic_types must be an iterable of strings"
                ) from exc
        object.__setattr__(
            self,
            "excluded_semantic_types",
            frozenset(_normalize_semantic_type(value) for value in excluded_values),
        )

        for field_name in ("default_bias", "non_patient_bias"):
            bias = float(getattr(self, field_name))
            if not math.isfinite(bias):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, bias)
        object.__setattr__(
            self, "experiencer", _normalize_experiencer(self.experiencer)
        )

    @property
    def biases(self) -> Mapping[str, float]:
        """Alias for :attr:`semantic_type_biases` used by policy consumers."""

        return self.semantic_type_biases

    @property
    def exclusions(self) -> frozenset[str]:
        """Alias for :attr:`excluded_semantic_types`."""

        return self.excluded_semantic_types


SectionContextRuleInput: TypeAlias = SectionContextRule | Mapping[str, object]


def _coerce_rule(value: SectionContextRuleInput) -> SectionContextRule:
    if isinstance(value, SectionContextRule):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("section rules must be SectionContextRule or mappings")

    raw_biases = value.get("semantic_type_biases")
    if raw_biases is None:
        raw_biases = value.get("biases", value.get("bias", {}))
    if isinstance(raw_biases, Sequence) and not isinstance(raw_biases, str):
        raw_biases = {
            item: 0.1 for item in raw_biases if isinstance(item, str) and item.strip()
        }
    if raw_biases is None:
        raw_biases = {}

    preferred = value.get("preferred_semantic_types", value.get("preferred_types"))
    if preferred is not None:
        if isinstance(preferred, str):
            preferred_values = (preferred,)
        else:
            preferred_values = tuple(preferred)
        preferred_bias = float(value.get("preferred_bias", 0.1))
        if not isinstance(raw_biases, Mapping):
            raise TypeError("section rule biases must be a mapping")
        raw_biases = {
            **raw_biases,
            **{item: preferred_bias for item in preferred_values},
        }

    raw_exclusions = value.get("excluded_semantic_types")
    if raw_exclusions is None:
        raw_exclusions = value.get(
            "exclusions",
            value.get("exclude", value.get("excluded_types", ())),
        )
    if raw_exclusions is None:
        raw_exclusions = ()
    return SectionContextRule(
        semantic_type_biases=raw_biases,
        excluded_semantic_types=raw_exclusions,
        default_bias=value.get("default_bias", 0.0),
        experiencer=value.get("experiencer"),
        non_patient_bias=value.get("non_patient_bias", value.get("downrank", 0.0)),
    )


def _compile_rules(
    rules: Mapping[str, SectionContextRuleInput],
) -> Mapping[str, SectionContextRule]:
    if not isinstance(rules, Mapping):
        raise TypeError("section rules must be a mapping")
    compiled: dict[str, SectionContextRule] = {}
    for raw_section, raw_rule in rules.items():
        section = _normalize_section(raw_section)
        if section is None:
            raise ValueError("section rule keys must be non-empty labels")
        compiled[section] = _coerce_rule(raw_rule)
    return MappingProxyType(compiled)


# This table is intentionally data-shaped: downstream users can replace it with
# a caller-owned mapping without changing the ranking implementation.
_DEFAULT_RULE_DATA: Mapping[str, Mapping[str, object]] = {
    "allergies": {
        "semantic_type_biases": {
            "allergen": 0.18,
            "allergy": 0.18,
            "substance": 0.08,
        },
        "excluded_semantic_types": ("medication", "drug", "pharmaceutical"),
        "experiencer": "patient",
    },
    "medications": {
        "semantic_type_biases": {
            "medication": 0.18,
            "drug": 0.18,
            "pharmaceutical": 0.15,
        },
        "excluded_semantic_types": ("allergen", "allergy"),
        "experiencer": "patient",
    },
    "problem_list": {
        "semantic_type_biases": {
            "condition": 0.12,
            "disease": 0.12,
            "disorder": 0.12,
        },
        "excluded_semantic_types": ("medication", "drug"),
        "experiencer": "patient",
    },
    "assessment": {
        "semantic_type_biases": {
            "condition": 0.08,
            "disease": 0.08,
            "disorder": 0.08,
        },
        "experiencer": "patient",
    },
    "family_history": {
        "semantic_type_biases": {
            "condition": 0.04,
            "disease": 0.04,
            "disorder": 0.04,
        },
        "default_bias": -0.12,
        "non_patient_bias": -0.05,
        "experiencer": "family",
    },
    "social_history": {"experiencer": "patient"},
}

SECTION_CONTEXT_RULES = _compile_rules(_DEFAULT_RULE_DATA)
DEFAULT_SECTION_CONTEXT_RULES = SECTION_CONTEXT_RULES


@dataclass(frozen=True)
class SectionContextConfig:
    """Immutable, validated section-policy configuration."""

    rules: Mapping[str, SectionContextRuleInput] = field(
        default_factory=lambda: SECTION_CONTEXT_RULES
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", _compile_rules(self.rules))

    def rule_for(self, section: object) -> SectionContextRule | None:
        """Return the configured rule for ``section``, if one exists."""

        normalized = _normalize_section(section)
        if normalized is None:
            return None
        return self.rules.get(normalized)


DEFAULT_SECTION_CONTEXT_CONFIG = SectionContextConfig()


@dataclass(frozen=True)
class _SectionEntry:
    label: str
    start: int | None = None
    end: int | None = None
    match_key: tuple[str, str] | str | None = None


def _offset(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"section {field_name} must be a non-negative integer")
    return value


def _entry_from_value(
    value: object,
    *,
    match_key: tuple[str, str] | str | None = None,
) -> _SectionEntry | None:
    label = _normalize_section(value)
    if label is None:
        return None
    if not isinstance(value, Mapping):
        return _SectionEntry(label=label, match_key=match_key)
    start = _offset(value.get("start", value.get("start_char")), "start")
    end = _offset(value.get("end", value.get("end_char")), "end")
    if (start is None) != (end is None) or (start is not None and end < start):
        raise ValueError("section start and end offsets must be an ordered pair")
    raw_key = value.get("match_key")
    if raw_key is None and "system" in value and "code" in value:
        raw_key = (value["system"], value["code"])
    if raw_key is not None and not isinstance(raw_key, (str, tuple)):
        raise TypeError("section match_key must be a code or (system, code) pair")
    return _SectionEntry(
        label=label,
        start=start,
        end=end,
        match_key=raw_key if raw_key is not None else match_key,
    )


def _coerce_sections(sections: object | None) -> tuple[_SectionEntry, ...]:
    if sections is None:
        return ()
    if isinstance(sections, str):
        entry = _entry_from_value(sections)
        return (entry,) if entry is not None else ()
    if isinstance(sections, Mapping):
        if _section_text(sections) is not None:
            entry = _entry_from_value(sections)
            return (entry,) if entry is not None else ()
        entries: list[_SectionEntry] = []
        for raw_key, value in sections.items():
            key = raw_key if isinstance(raw_key, (str, tuple)) else None
            entry = _entry_from_value(value, match_key=key)
            if entry is None and isinstance(raw_key, str):
                entry = _entry_from_value(raw_key, match_key=key)
            if entry is not None:
                entries.append(entry)
        return tuple(entries)
    if isinstance(sections, (bytes, bytearray)):
        raise TypeError("sections must be a label, mapping, or iterable")
    try:
        values = tuple(sections)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("sections must be a label, mapping, or iterable") from exc
    entries = []
    for value in values:
        entry = _entry_from_value(value)
        if entry is not None:
            entries.append(entry)
    return tuple(entries)


def _match_offsets(match: ConceptMatch) -> tuple[int, int] | None:
    start = match.metadata.get("start", match.metadata.get("start_char"))
    end = match.metadata.get("end", match.metadata.get("end_char"))
    if isinstance(start, int) and not isinstance(start, bool):
        if isinstance(end, int) and not isinstance(end, bool) and start <= end:
            return start, end
    return None


def _entry_for_match(
    match: ConceptMatch,
    entries: Sequence[_SectionEntry],
) -> _SectionEntry | None:
    for entry in entries:
        if entry.match_key is None:
            continue
        if entry.match_key == match.key or entry.match_key == match.code:
            return entry

    offsets = _match_offsets(match)
    if offsets is not None:
        start, end = offsets
        containing = [
            entry
            for entry in entries
            if entry.start is not None
            and entry.end is not None
            and entry.start <= start
            and end <= entry.end
        ]
        if containing:
            return min(
                containing, key=lambda entry: (entry.end - entry.start, entry.label)
            )

    return next(
        (
            entry
            for entry in entries
            if entry.match_key is None and entry.start is None and entry.end is None
        ),
        None,
    )


def _metadata_values(
    metadata: Mapping[str, object], field_name: str
) -> tuple[object, ...]:
    value = metadata.get(field_name)
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item for item in _VALUE_SPLIT_RE.split(value) if item.strip())
    if isinstance(value, Mapping):
        return ()
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return (value,)


def _semantic_types(match: ConceptMatch) -> frozenset[str]:
    values: list[object] = []
    for field_name in _SEMANTIC_TYPE_FIELDS:
        values.extend(_metadata_values(match.metadata, field_name))
    result: set[str] = set()
    for value in values:
        if isinstance(value, str) and value.strip():
            result.add(_normalize_semantic_type(value))
    return frozenset(result)


def _section_bias(
    match: ConceptMatch,
    rule: SectionContextRule | None,
    resolved_experiencer: str | None,
) -> tuple[float, bool]:
    if rule is None:
        return 0.0, False
    semantic_types = _semantic_types(match)
    excluded = bool(semantic_types & rule.excluded_semantic_types)
    matching_biases = [
        rule.semantic_type_biases[semantic_type]
        for semantic_type in semantic_types
        if semantic_type in rule.semantic_type_biases
    ]
    bias = max(matching_biases, default=rule.default_bias)
    if resolved_experiencer is not None and resolved_experiencer != _PATIENT:
        bias += rule.non_patient_bias
    return bias, excluded


def _sort_key(match: ConceptMatch) -> tuple[float, float, float, str, str, str, str]:
    context_score = match.context_score
    if context_score is None:
        context_score = match.score
    return (
        -context_score,
        -match.score,
        -match.section_bias,
        match.system_uri,
        match.code,
        match.display,
        match.matched_term,
    )


def apply_section_context(
    matches: Iterable[ConceptMatch],
    sections: object | None,
    experiencer: str | None = None,
    *,
    rules: Mapping[str, SectionContextRuleInput] | SectionContextConfig | None = None,
    config: SectionContextConfig | None = None,
    drop_excluded: bool = True,
) -> tuple[ConceptMatch, ...]:
    """Apply section-aware semantic-type priors to lexical matches.

    Args:
        matches: Candidate :class:`ConceptMatch` records with vocabulary
            metadata such as ``semantic_type`` or ``semantic_types``.
        sections: A detected section label, or offset-bearing section mappings.
            When several unscoped labels are supplied, the first recognized label
            governs the matches; offset-bearing entries are selected per match.
        experiencer: Optional explicit experiencer.  It overrides a section's
            configured prior, while an omitted value lets a section rule provide
            one (for example, ``family_history`` -> ``family``).
        rules: Optional caller-owned section rule mapping.
        config: Optional :class:`SectionContextConfig`; mutually exclusive with
            ``rules``.
        drop_excluded: Remove candidates whose semantic type is excluded by the
            governing section.  Set to ``False`` to retain them with provenance.

    Returns:
        A deterministic tuple of section-enriched matches.  Only canonical
        section labels, offsets supplied by the caller, concept metadata, and
        concept identifiers are carried forward; raw query text is not stored.
    """

    if not isinstance(drop_excluded, bool):
        raise TypeError("drop_excluded must be a boolean")
    if config is not None and rules is not None:
        raise ValueError("pass either config or rules, not both")
    if isinstance(rules, SectionContextConfig):
        if config is not None:
            raise ValueError("pass either config or rules, not both")
        config = rules
        rules = None
    resolved_config = config or (
        SectionContextConfig(rules)
        if rules is not None
        else DEFAULT_SECTION_CONTEXT_CONFIG
    )
    resolved_experiencer = _normalize_experiencer(experiencer)
    entries = _coerce_sections(sections)

    enriched: list[ConceptMatch] = []
    for match in matches:
        if not isinstance(match, ConceptMatch):
            raise TypeError("matches must contain ConceptMatch objects")
        entry = _entry_for_match(match, entries)
        section = (
            entry.label
            if entry is not None
            else match.section or _normalize_section(match.metadata)
        )
        rule = resolved_config.rule_for(section) if section is not None else None
        match_experiencer = (
            resolved_experiencer
            or (rule.experiencer if rule is not None else None)
            or match.experiencer
            or _normalize_experiencer(match.metadata.get("experiencer"))
        )
        bias, excluded = _section_bias(match, rule, match_experiencer)
        if excluded and drop_excluded:
            continue

        metadata = dict(match.metadata)
        if section is not None:
            metadata["section"] = section
        if match_experiencer is not None:
            metadata["experiencer"] = match_experiencer
            metadata["patient_record_eligible"] = match_experiencer == _PATIENT
            metadata["non_patient"] = match_experiencer != _PATIENT
        metadata["section_bias"] = bias
        if excluded:
            metadata["section_excluded"] = True

        context_score = match.score + bias
        enriched.append(
            replace(
                match,
                section=section,
                experiencer=match_experiencer,
                section_bias=bias,
                context_score=context_score,
                metadata=metadata,
            )
        )

    enriched.sort(key=_sort_key)
    return tuple(enriched)
