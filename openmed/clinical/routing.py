"""Deterministic note-type routing for section-scoped clinical extraction.

The router is deliberately a small rules-only boundary around the existing
document classifier and section detector.  It selects a profile, produces a
section-scoped extraction plan, and records enough non-text provenance for a
caller to audit the decision.  It does not load a model, fetch terminology, or
interpret a finding as a diagnosis.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import copy
from dataclasses import dataclass, fields, is_dataclass, replace
from math import isfinite
from types import MappingProxyType
from typing import Any

from .lab_values import (
    LabValueAttributeMention,
    link_lab_value_attributes,
)
from .medication_sig import (
    MEDICATION_CANDIDATES,
    MedicationCandidate,
    MedicationCandidatePreset,
    MedicationGrounder,
    filter_medication_candidates,
)
from .problem_list import ProblemMention, problem_mentions_from_grounded_terms
from .sections import (
    DOCUMENT_TYPE_CONFIDENCE_THRESHOLD,
    UNKNOWN_DOCUMENT_TYPE,
    SectionSpan,
    classify_document,
    detect_sections,
)

ROUTING_PROVENANCE_KEY = "routing_provenance"
GENERIC_PROFILE_NAME = "generic"
RADIOLOGY_PROFILE_NAME = "radiology"
PATHOLOGY_PROFILE_NAME = "pathology"

ROUTING_STAGE_NAMES = ("medication", "problem_list", "lab_values")
ROUTING_SECTION_LABELS = frozenset(
    {
        "technique",
        "findings",
        "impression",
        "specimen",
        "diagnosis",
        "synoptic",
        "staging",
        "grading",
    }
)

_TARGET_DOCUMENT_TYPES = frozenset({"radiology_report", "pathology_report"})
_STAGE_ALIASES = {
    "medications": "medication",
    "medication": "medication",
    "problem": "problem_list",
    "problems": "problem_list",
    "problem_list": "problem_list",
    "problem-list": "problem_list",
    "lab": "lab_values",
    "labs": "lab_values",
    "lab_value": "lab_values",
    "lab_values": "lab_values",
    "lab-value": "lab_values",
    "lab-values": "lab_values",
}


def _tuple_of_strings(value: Iterable[object], *, field_name: str) -> tuple[str, ...]:
    values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} entries must be non-empty strings")
        normalized = item.strip()
        if normalized not in values:
            values.append(normalized)
    return tuple(values)


def _normalize_stage_name(stage: str) -> str:
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")
    normalized = stage.strip().casefold().replace(" ", "_")
    return _STAGE_ALIASES.get(normalized, normalized)


def _freeze_section_config(
    config: Mapping[object, Iterable[object]],
) -> Mapping[str, tuple[str, ...]]:
    frozen: dict[str, tuple[str, ...]] = {}
    for raw_stage, raw_sections in config.items():
        if not isinstance(raw_stage, str):
            raise ValueError("stage configuration keys must be strings")
        stage = _normalize_stage_name(raw_stage)
        if stage in frozen:
            raise ValueError(f"duplicate stage configuration for {stage!r}")
        frozen[stage] = _tuple_of_strings(
            raw_sections,
            field_name=f"{stage} section configuration",
        )
    return MappingProxyType(frozen)


def _freeze_terms(
    terms: Mapping[object, Iterable[object]],
) -> Mapping[str, tuple[str, ...]]:
    frozen: dict[str, tuple[str, ...]] = {}
    for raw_name, raw_values in terms.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("cue term keys must be non-empty strings")
        frozen[raw_name.strip()] = _tuple_of_strings(
            raw_values,
            field_name=f"{raw_name} cues",
        )
    return MappingProxyType(frozen)


def _freeze_thresholds(
    thresholds: Mapping[object, object],
) -> Mapping[str, float]:
    frozen: dict[str, float] = {}
    for raw_name, raw_value in thresholds.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("threshold keys must be non-empty strings")
        if isinstance(raw_value, bool):
            raise ValueError(f"threshold {raw_name!r} must be numeric")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"threshold {raw_name!r} must be numeric") from exc
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold {raw_name!r} must be between 0 and 1")
        frozen[raw_name.strip()] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class RoutingProvenance(Mapping[str, object]):
    """PHI-free evidence for one profile selection."""

    profile: str
    confidence: float
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.profile, str) or not self.profile.strip():
            raise ValueError("routing provenance profile must be non-empty")
        if isinstance(self.confidence, bool):
            raise ValueError("routing provenance confidence must be numeric")
        confidence = float(self.confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("routing provenance confidence must be between 0 and 1")
        if self.fallback_reason is not None and (
            not isinstance(self.fallback_reason, str)
            or not self.fallback_reason.strip()
        ):
            raise ValueError("fallback_reason must be non-empty when provided")
        object.__setattr__(self, "profile", self.profile.strip())
        object.__setattr__(self, "confidence", round(confidence, 6))

    def __getitem__(self, key: str) -> object:
        if key == "profile":
            return self.profile
        if key == "confidence":
            return self.confidence
        if key == "fallback_reason":
            return self.fallback_reason
        raise KeyError(key)

    def __iter__(self):
        return iter(("profile", "confidence", "fallback_reason"))

    def __len__(self) -> int:
        return 3

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready provenance mapping."""

        return {
            "profile": self.profile,
            "confidence": self.confidence,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class NoteTypeProfile:
    """Immutable contract for a note-type extraction route.

    ``expected_sections`` controls the profile's public section view.  The
    per-stage map can include additional sections when a stage needs a more
    specific scope, such as pathology staging and grading.  A profile with
    ``pass_through=True`` deliberately returns all supplied sections and
    entities in their original order.
    """

    name: str
    document_types: tuple[str, ...]
    expected_sections: tuple[str, ...]
    entity_priorities: tuple[str, ...]
    section_scoped_stage_config: Mapping[str, tuple[str, ...]]
    cue_terms: Mapping[str, tuple[str, ...]]
    thresholds: Mapping[str, float]
    pass_through: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("profile name must be non-empty")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(
            self,
            "document_types",
            _tuple_of_strings(self.document_types, field_name="document_types"),
        )
        object.__setattr__(
            self,
            "expected_sections",
            _tuple_of_strings(self.expected_sections, field_name="expected_sections"),
        )
        object.__setattr__(
            self,
            "entity_priorities",
            _tuple_of_strings(self.entity_priorities, field_name="entity_priorities"),
        )
        object.__setattr__(
            self,
            "section_scoped_stage_config",
            _freeze_section_config(self.section_scoped_stage_config),
        )
        object.__setattr__(self, "cue_terms", _freeze_terms(self.cue_terms))
        object.__setattr__(self, "thresholds", _freeze_thresholds(self.thresholds))

    @property
    def profile_name(self) -> str:
        """Return the stable serialized profile name."""

        return self.name

    @property
    def supported_document_types(self) -> tuple[str, ...]:
        """Return document-type labels handled by this profile."""

        return self.document_types

    @property
    def stage_config(self) -> Mapping[str, tuple[str, ...]]:
        """Backward-friendly alias for the scoped stage configuration."""

        return self.section_scoped_stage_config

    def supports(self, document_type: str) -> bool:
        """Return whether this profile handles ``document_type``."""

        return document_type in self.document_types

    def sections_for_stage(
        self,
        sections: Iterable[Mapping[str, Any]],
        *,
        stage: str | None = None,
    ) -> tuple[SectionSpan, ...]:
        """Select detected sections for the profile or one extraction stage."""

        active = tuple(
            _coerce_section(section, index) for index, section in enumerate(sections)
        )
        if self.pass_through:
            return active

        normalized_stage = _normalize_stage_name(stage) if stage is not None else None
        labels = (
            self.section_scoped_stage_config.get(
                normalized_stage, self.expected_sections
            )
            if normalized_stage is not None
            else self.expected_sections
        )
        allowed = set(labels)
        if "*" in allowed:
            return active
        return tuple(section for section in active if section.label in allowed)

    def scope_sections(
        self,
        sections: Iterable[Mapping[str, Any]],
    ) -> tuple[SectionSpan, ...]:
        """Return the profile's public section scope."""

        return self.sections_for_stage(sections)

    def scope_entities(
        self,
        entities: Iterable[object],
        sections: Iterable[Mapping[str, Any]],
        *,
        stage: str | None = None,
    ) -> list[object]:
        """Keep entity spans contained in the selected sections.

        Generic pass-through keeps entities without offsets because the legacy
        pipeline accepts such caller-owned objects.  A specialized profile
        safely drops an entity whose section cannot be established.
        """

        values = list(entities)
        if self.pass_through:
            return values
        selected = self.sections_for_stage(sections, stage=stage)
        return [entity for entity in values if _entity_in_sections(entity, selected)]

    def provenance(
        self,
        confidence: float,
        *,
        fallback_reason: str | None = None,
    ) -> RoutingProvenance:
        """Build provenance for an already-selected profile."""

        return RoutingProvenance(
            profile=self.name,
            confidence=confidence,
            fallback_reason=fallback_reason,
        )


class GenericProfile(NoteTypeProfile):
    """Pass-through profile used when routing abstains."""

    def __init__(self) -> None:
        super().__init__(
            name=GENERIC_PROFILE_NAME,
            document_types=(),
            expected_sections=(),
            entity_priorities=(),
            section_scoped_stage_config={
                stage: ("*",) for stage in ROUTING_STAGE_NAMES
            },
            cue_terms={},
            thresholds={},
            pass_through=True,
        )


class RadiologyProfile(NoteTypeProfile):
    """Rules-only profile for radiology report extraction."""

    def __init__(self) -> None:
        sections = ("technique", "findings", "impression")
        super().__init__(
            name=RADIOLOGY_PROFILE_NAME,
            document_types=("radiology_report",),
            expected_sections=sections,
            entity_priorities=(
                "finding",
                "laterality",
                "technique",
                "measurement",
                "anatomical_site",
            ),
            section_scoped_stage_config={
                stage: sections for stage in ROUTING_STAGE_NAMES
            },
            cue_terms={
                "laterality": (
                    "left",
                    "right",
                    "bilateral",
                    "both",
                    "lt",
                    "rt",
                ),
                "technique": (
                    "technique",
                    "protocol",
                    "sequence",
                    "contrast",
                    "enhancement",
                ),
            },
            thresholds={"laterality": 0.70, "technique": 0.65},
        )


class PathologyProfile(NoteTypeProfile):
    """Rules-only profile for pathology report extraction."""

    def __init__(self) -> None:
        sections = ("specimen", "diagnosis", "synoptic")
        scoped_sections = sections + ("staging", "grading")
        super().__init__(
            name=PATHOLOGY_PROFILE_NAME,
            document_types=("pathology_report",),
            expected_sections=sections,
            entity_priorities=(
                "specimen",
                "diagnosis",
                "synoptic",
                "staging",
                "grading",
            ),
            section_scoped_stage_config={
                stage: scoped_sections for stage in ROUTING_STAGE_NAMES
            },
            cue_terms={
                "staging": (
                    "stage",
                    "staging",
                    "tnm",
                    "pt",
                    "pn",
                    "pm",
                ),
                "grading": (
                    "grade",
                    "grading",
                    "histologic grade",
                    "gleason",
                ),
            },
            thresholds={"staging": 0.70, "grading": 0.70},
        )


GENERIC_PROFILE = GenericProfile()
RADIOLOGY_PROFILE = RadiologyProfile()
PATHOLOGY_PROFILE = PathologyProfile()

# Explicit aliases make the profile constants discoverable without requiring a
# caller to know whether the surrounding code says "note type" or "document".
GENERIC_NOTE_TYPE_PROFILE = GENERIC_PROFILE
RADIOLOGY_NOTE_TYPE_PROFILE = RADIOLOGY_PROFILE
PATHOLOGY_NOTE_TYPE_PROFILE = PATHOLOGY_PROFILE
RadiologyNoteTypeProfile = RadiologyProfile
PathologyNoteTypeProfile = PathologyProfile


@dataclass(frozen=True)
class RoutingSelection:
    """Profile plus the provenance of the classifier-to-profile decision."""

    profile: NoteTypeProfile
    provenance: RoutingProvenance

    @property
    def profile_name(self) -> str:
        """Return the selected profile name."""

        return self.profile.name

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready routing decision."""

        return {
            "profile": self.profile.name,
            "confidence": self.provenance.confidence,
            "fallback_reason": self.provenance.fallback_reason,
        }


def _classification_value(
    classification: object,
    key: str,
    default: object = None,
) -> object:
    if isinstance(classification, Mapping):
        return classification.get(key, default)
    return getattr(classification, key, default)


def _safe_confidence(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not isfinite(confidence):
        return 0.0
    return min(max(confidence, 0.0), 1.0)


def resolve_profile(classify_document_result: object) -> RoutingSelection:
    """Resolve a classifier result to a profile and audit provenance.

    Unsupported document types, unknown labels, malformed labels, and scores
    below the classifier's documented confidence threshold all select the
    generic pass-through profile.  This makes abstention explicit without
    changing the entities or ordering produced by an existing pipeline.
    """

    if isinstance(classify_document_result, str):
        document_type = classify_document_result.strip()
        confidence = 1.0
    else:
        raw_type = _classification_value(classify_document_result, "type", "")
        document_type = raw_type.strip() if isinstance(raw_type, str) else ""
        confidence = _safe_confidence(
            _classification_value(classify_document_result, "confidence", 0.0)
        )

    if document_type in _TARGET_DOCUMENT_TYPES:
        if confidence < DOCUMENT_TYPE_CONFIDENCE_THRESHOLD:
            reason = "low_confidence"
            profile = GENERIC_PROFILE
        elif document_type == "radiology_report":
            reason = None
            profile = RADIOLOGY_PROFILE
        else:
            reason = None
            profile = PATHOLOGY_PROFILE
    elif not document_type or document_type == UNKNOWN_DOCUMENT_TYPE:
        reason = "unknown_document_type"
        profile = GENERIC_PROFILE
    else:
        reason = "unsupported_document_type"
        profile = GENERIC_PROFILE

    return RoutingSelection(
        profile=profile,
        provenance=profile.provenance(confidence, fallback_reason=reason),
    )


def select_profile(classify_document_result: object) -> NoteTypeProfile:
    """Select the radiology, pathology, or generic profile.

    The function intentionally returns the profile itself.  Call
    :func:`resolve_profile` or :func:`routing_provenance` when the caller also
    needs the decision record.
    """

    return resolve_profile(classify_document_result).profile


def routing_provenance(classify_document_result: object) -> RoutingProvenance:
    """Return PHI-free provenance for a classifier result."""

    return resolve_profile(classify_document_result).provenance


def classify_and_select_profile(text: str) -> RoutingSelection:
    """Classify a note locally and resolve its extraction profile."""

    return resolve_profile(classify_document(text))


def _coerce_section(section: Mapping[str, Any], index: int) -> SectionSpan:
    if not isinstance(section, Mapping):
        raise TypeError(f"section {index} must be a mapping")
    label = section.get("label")
    start = section.get("start")
    end = section.get("end")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"section {index} requires a non-empty label")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end <= start
    ):
        raise ValueError(f"section {index} requires valid start/end offsets")
    metadata = {
        key: value
        for key, value in section.items()
        if key not in {"label", "start", "end"}
    }
    return SectionSpan(label=label.strip(), start=start, end=end, **metadata)


def _entity_offsets(entity: object) -> tuple[int, int] | None:
    candidate = entity
    if not isinstance(candidate, Mapping):
        to_dict = getattr(candidate, "to_dict", None)
        if callable(to_dict):
            mapped = to_dict()
            if isinstance(mapped, Mapping):
                candidate = mapped
    if isinstance(candidate, Mapping):
        start = candidate.get("start")
        end = candidate.get("end")
        if start is None or end is None:
            offset = candidate.get("offset")
            if isinstance(offset, (tuple, list)) and len(offset) == 2:
                start, end = offset
    else:
        start = getattr(candidate, "start", None)
        end = getattr(candidate, "end", None)
        if start is None or end is None:
            offset = getattr(candidate, "offset", None)
            if isinstance(offset, (tuple, list)) and len(offset) == 2:
                start, end = offset
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end < start
    ):
        return None
    return start, end


def _section_content_start(section: SectionSpan) -> int:
    content_start = section.get("content_start")
    if isinstance(content_start, int) and not isinstance(content_start, bool):
        return max(section.start, min(content_start, section.end))
    return section.start


def _entity_in_sections(
    entity: object,
    sections: Iterable[SectionSpan],
) -> bool:
    offsets = _entity_offsets(entity)
    if offsets is None:
        return False
    start, end = offsets
    return any(
        _section_content_start(section) <= start and end <= section.end
        for section in sections
    )


def resolve_profile_sections(
    text: str,
    profile: NoteTypeProfile,
    *,
    sections: Iterable[Mapping[str, Any]] | None = None,
    language: str | None = None,
) -> tuple[SectionSpan, ...]:
    """Detect and scope sections for a profile."""

    if not isinstance(profile, NoteTypeProfile):
        raise TypeError("profile must be a NoteTypeProfile")
    detected = (
        detect_sections(text, language=language)
        if sections is None
        else tuple(sections)
    )
    return profile.scope_sections(detected)


def scope_entities(
    entities: Iterable[object],
    profile: NoteTypeProfile,
    sections: Iterable[Mapping[str, Any]],
    *,
    stage: str | None = None,
) -> list[object]:
    """Apply a profile's section scope to caller-provided entity spans."""

    return profile.scope_entities(entities, sections, stage=stage)


@dataclass(frozen=True)
class ExtractionPlan:
    """Section-scoped inputs and provenance for the existing extractors."""

    profile: NoteTypeProfile
    sections: tuple[SectionSpan, ...]
    stage_sections: Mapping[str, tuple[SectionSpan, ...]]
    medication_entities: tuple[object, ...]
    problem_mentions: tuple[object, ...]
    lab_value_mentions: tuple[object, ...]
    routing_provenance: RoutingProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(
            self,
            "stage_sections",
            MappingProxyType(
                {key: tuple(value) for key, value in self.stage_sections.items()}
            ),
        )
        for field_name in (
            "medication_entities",
            "problem_mentions",
            "lab_value_mentions",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))

    @property
    def medications(self) -> tuple[object, ...]:
        """Alias for medication entity inputs."""

        return self.medication_entities

    @property
    def problems(self) -> tuple[object, ...]:
        """Alias for problem-list entity inputs."""

        return self.problem_mentions

    @property
    def lab_values(self) -> tuple[object, ...]:
        """Alias for lab-value entity inputs."""

        return self.lab_value_mentions

    def sections_for_stage(self, stage: str) -> tuple[SectionSpan, ...]:
        """Return the detected section scope for one extraction stage."""

        return self.stage_sections[_normalize_stage_name(stage)]


def _profile_from_argument(profile_or_classification: object) -> NoteTypeProfile:
    if isinstance(profile_or_classification, NoteTypeProfile):
        return profile_or_classification
    return select_profile(profile_or_classification)


def build_extraction_plan(
    text: str,
    classify_document_result: object | None = None,
    *,
    profile: NoteTypeProfile | None = None,
    sections: Iterable[Mapping[str, Any]] | None = None,
    medication_entities: Iterable[object] = (),
    problem_mentions: Iterable[object] = (),
    lab_value_mentions: Iterable[object] = (),
    language: str | None = None,
) -> ExtractionPlan:
    """Build a routed plan without changing the generic pipeline semantics."""

    classification = (
        classify_document(text)
        if classify_document_result is None
        else classify_document_result
    )
    selection = resolve_profile(classification)
    selected_profile = selection.profile if profile is None else profile
    if not isinstance(selected_profile, NoteTypeProfile):
        raise TypeError("profile must be a NoteTypeProfile")
    if profile is not None and profile.name != selection.profile.name:
        provenance = profile.provenance(
            selection.provenance.confidence,
            fallback_reason="explicit_profile",
        )
    else:
        provenance = selection.provenance

    detected_sections = (
        detect_sections(text, language=language)
        if sections is None
        else tuple(sections)
    )
    coerced_sections = tuple(
        _coerce_section(section, index)
        for index, section in enumerate(detected_sections)
    )
    scoped_sections = selected_profile.scope_sections(coerced_sections)
    stage_sections = {
        stage: selected_profile.sections_for_stage(
            coerced_sections,
            stage=stage,
        )
        for stage in ROUTING_STAGE_NAMES
    }
    return ExtractionPlan(
        profile=selected_profile,
        sections=scoped_sections,
        stage_sections=stage_sections,
        medication_entities=tuple(
            selected_profile.scope_entities(
                medication_entities,
                coerced_sections,
                stage="medication",
            )
        ),
        problem_mentions=tuple(
            selected_profile.scope_entities(
                problem_mentions,
                coerced_sections,
                stage="problem_list",
            )
        ),
        lab_value_mentions=tuple(
            selected_profile.scope_entities(
                lab_value_mentions,
                coerced_sections,
                stage="lab_values",
            )
        ),
        routing_provenance=provenance,
    )


def _scoped_stage_inputs(
    text: str,
    profile_or_classification: object,
    *,
    stage: str,
    sections: Iterable[Mapping[str, Any]] | None,
    language: str | None,
    entities: Iterable[object],
) -> list[object]:
    profile = _profile_from_argument(profile_or_classification)
    detected_sections = (
        detect_sections(text, language=language)
        if sections is None
        else tuple(sections)
    )
    return profile.scope_entities(entities, detected_sections, stage=stage)


def extract_scoped_medication_candidates(
    text: str,
    entities: Iterable[object],
    profile_or_classification: object,
    *,
    sections: Iterable[Mapping[str, Any]] | None = None,
    language: str | None = None,
    preset: str | MedicationCandidatePreset = MEDICATION_CANDIDATES,
    grounder: MedicationGrounder | None = None,
) -> list[MedicationCandidate]:
    """Run medication candidate filtering inside the selected stage scope."""

    scoped = _scoped_stage_inputs(
        text,
        profile_or_classification,
        stage="medication",
        sections=sections,
        language=language,
        entities=entities,
    )
    return filter_medication_candidates(
        text,
        scoped,
        preset=preset,
        grounder=grounder,
    )


def extract_scoped_problem_mentions(
    text: str,
    grounded_terms: Iterable[object],
    profile_or_classification: object,
    *,
    sections: Iterable[Mapping[str, Any]] | None = None,
    language: str | None = None,
) -> list[ProblemMention]:
    """Convert only section-scoped grounded condition terms to problems."""

    scoped = _scoped_stage_inputs(
        text,
        profile_or_classification,
        stage="problem_list",
        sections=sections,
        language=language,
        entities=grounded_terms,
    )
    return problem_mentions_from_grounded_terms(scoped)


def link_scoped_lab_value_attributes(
    text: str,
    mentions: Iterable[LabValueAttributeMention | Mapping[str, object]],
    profile_or_classification: object,
    *,
    sections: Iterable[Mapping[str, Any]] | None = None,
    language: str | None = None,
    max_distance: int = 80,
):
    """Link lab attributes only within sections selected for the lab stage."""

    scoped = _scoped_stage_inputs(
        text,
        profile_or_classification,
        stage="lab_values",
        sections=sections,
        language=language,
        entities=mentions,
    )
    return link_lab_value_attributes(scoped, max_distance=max_distance)


def attach_routing_provenance(
    analysis_result: object,
    classify_document_result: object,
    *,
    selection: RoutingSelection | None = None,
) -> object:
    """Return ``analysis_result`` with routing provenance in its metadata.

    Dataclass results such as :class:`openmed.core.results.AnalyzeResult` are
    copied with their existing type preserved.  Mapping results are copied as
    mappings.  No source text is added to the provenance record.
    """

    resolved = selection or resolve_profile(classify_document_result)
    provenance = resolved.provenance.to_dict()

    if is_dataclass(analysis_result) and not isinstance(analysis_result, type):
        field_names = {item.name for item in fields(analysis_result)}
        if "metadata" in field_names:
            current = getattr(analysis_result, "metadata", None)
            metadata = dict(current) if isinstance(current, Mapping) else {}
            metadata[ROUTING_PROVENANCE_KEY] = provenance
            return replace(analysis_result, metadata=metadata)

    if isinstance(analysis_result, Mapping):
        payload = dict(analysis_result)
        current = payload.get("metadata")
        metadata = dict(current) if isinstance(current, Mapping) else {}
        metadata[ROUTING_PROVENANCE_KEY] = provenance
        payload["metadata"] = metadata
        return payload

    try:
        copied = copy(analysis_result)
        current = getattr(copied, "metadata", None)
        metadata = dict(current) if isinstance(current, Mapping) else {}
        metadata[ROUTING_PROVENANCE_KEY] = provenance
        setattr(copied, "metadata", metadata)
        return copied
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            "analysis_result must be a mapping or expose writable metadata"
        ) from exc


@dataclass(frozen=True)
class RoutedAnalysisResult(Mapping[str, Any]):
    """Analysis result wrapper exposing profile and provenance explicitly."""

    analysis_result: object
    profile: NoteTypeProfile
    sections: tuple[SectionSpan, ...]
    routing_provenance: RoutingProvenance

    @property
    def result(self) -> object:
        """Return the wrapped analysis result."""

        return self.analysis_result

    def to_dict(self) -> dict[str, Any]:
        """Serialize the wrapped result with routing metadata."""

        if hasattr(self.analysis_result, "to_dict"):
            payload = self.analysis_result.to_dict()
        elif isinstance(self.analysis_result, Mapping):
            payload = dict(self.analysis_result)
        else:
            payload = {"result": self.analysis_result}
        current = payload.get("metadata")
        metadata = dict(current) if isinstance(current, Mapping) else {}
        metadata[ROUTING_PROVENANCE_KEY] = self.routing_provenance.to_dict()
        payload["metadata"] = metadata
        payload[ROUTING_PROVENANCE_KEY] = self.routing_provenance.to_dict()
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def route_analysis(
    text: str,
    analysis_result: object | None = None,
    *,
    classify_document_result: object | None = None,
    sections: Iterable[Mapping[str, Any]] | None = None,
    language: str | None = None,
) -> RoutedAnalysisResult:
    """Attach a routing decision and section scope to an analysis result."""

    classification = (
        classify_document(text)
        if classify_document_result is None
        else classify_document_result
    )
    selection = resolve_profile(classification)
    detected = (
        detect_sections(text, language=language)
        if sections is None
        else tuple(sections)
    )
    scoped_sections = selection.profile.scope_sections(detected)
    base_result = {"metadata": {}} if analysis_result is None else analysis_result
    attached = attach_routing_provenance(
        base_result,
        classification,
        selection=selection,
    )
    return RoutedAnalysisResult(
        analysis_result=attached,
        profile=selection.profile,
        sections=scoped_sections,
        routing_provenance=selection.provenance,
    )


annotate_analysis_result = attach_routing_provenance
resolve_note_type_profile = select_profile


__all__ = [
    "GENERIC_NOTE_TYPE_PROFILE",
    "GENERIC_PROFILE",
    "GENERIC_PROFILE_NAME",
    "PATHOLOGY_NOTE_TYPE_PROFILE",
    "PATHOLOGY_PROFILE",
    "PATHOLOGY_PROFILE_NAME",
    "RADIOLOGY_NOTE_TYPE_PROFILE",
    "RADIOLOGY_PROFILE",
    "RADIOLOGY_PROFILE_NAME",
    "ROUTING_PROVENANCE_KEY",
    "ROUTING_SECTION_LABELS",
    "ROUTING_STAGE_NAMES",
    "ExtractionPlan",
    "GenericProfile",
    "LabValueAttributeMention",
    "MedicationCandidate",
    "MedicationCandidatePreset",
    "NoteTypeProfile",
    "PathologyNoteTypeProfile",
    "PathologyProfile",
    "ProblemMention",
    "RadiologyNoteTypeProfile",
    "RadiologyProfile",
    "RoutedAnalysisResult",
    "RoutingProvenance",
    "RoutingSelection",
    "annotate_analysis_result",
    "attach_routing_provenance",
    "build_extraction_plan",
    "classify_and_select_profile",
    "extract_scoped_medication_candidates",
    "extract_scoped_problem_mentions",
    "link_scoped_lab_value_attributes",
    "resolve_note_type_profile",
    "resolve_profile",
    "resolve_profile_sections",
    "route_analysis",
    "routing_provenance",
    "scope_entities",
    "select_profile",
]
