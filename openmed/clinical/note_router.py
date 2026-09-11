"""Section-aware routing for local clinical note extraction.

The router is intentionally a small orchestration boundary around the
existing deterministic section detector.  It validates that supplied sections
tile the source note, selects a profile from section labels, and gives an
injected extractor the original note plus absolute source offsets.  Unknown
sections remain explicit and never inherit a specialized profile by accident.

No model, terminology service, environment state, or network call is required
by this module.  Routing metadata is text-free so it can be placed in an audit
record without copying source note content.
"""

from __future__ import annotations

import inspect
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .routing import PATHOLOGY_PROFILE, RADIOLOGY_PROFILE, NoteTypeProfile
from .sections import SectionSpan, detect_sections, validate_sections

UNKNOWN_PROFILE_NAME = "unknown"
UNKNOWN_SECTION_REASON = "unknown_section"
PROFILE_MISMATCH_REASON = "profile_section_mismatch"
UNKNOWN_PROFILE_REASON = "unknown_profile"
EXTRACTOR_FAILURE_REASON = "extractor_failure"

SpanOffset = tuple[int, int]
SectionExtractor = Callable[..., object]


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _label_key(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("section labels must be strings")
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = re.sub(r"[^\w]+", "_", normalized, flags=re.UNICODE)
    return normalized.strip("_")


def _profile_key(value: object) -> str:
    normalized = _required_text(value, "profile name")
    aliases = {
        "radiology_report": "radiology",
        "pathology_report": "pathology",
    }
    return aliases.get(_label_key(normalized), _label_key(normalized))


def _normalize_labels(value: Iterable[object]) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError("section_labels must be an iterable of strings")
    labels: list[str] = []
    for item in value:
        label = _label_key(item)
        if not label:
            raise ValueError("section labels must not be empty")
        if label not in labels:
            labels.append(label)
    if not labels:
        raise ValueError("section_labels must contain at least one label")
    if "*" in labels and len(labels) != 1:
        raise ValueError("wildcard section profiles cannot mix labels")
    return tuple(labels)


def _safe_optional_offset(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer when provided")
    return value


def _coerce_section(value: object, index: int) -> SectionSpan:
    if not isinstance(value, Mapping):
        raise ValueError(f"section span {index} must be a mapping")
    label = value.get("label")
    start = value.get("start")
    end = value.get("end")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"section span {index} requires a non-empty label")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise ValueError(f"section span {index} requires integer offsets")

    metadata = {
        key: item for key, item in value.items() if key not in {"label", "start", "end"}
    }
    content_start = _safe_optional_offset(
        metadata.get("content_start"),
        field_name="content_start",
    )
    header_start = _safe_optional_offset(
        metadata.get("header_start"),
        field_name="header_start",
    )
    header_end = _safe_optional_offset(
        metadata.get("header_end"),
        field_name="header_end",
    )
    if content_start is not None and not start <= content_start <= end:
        raise ValueError(f"section span {index} has invalid content_start")
    if header_start is not None and not start <= header_start <= end:
        raise ValueError(f"section span {index} has invalid header_start")
    if header_end is not None and not start <= header_end <= end:
        raise ValueError(f"section span {index} has invalid header_end")
    if (
        header_start is not None
        and header_end is not None
        and header_end < header_start
    ):
        raise ValueError(f"section span {index} has invalid header offsets")
    return SectionSpan(label=label.strip(), start=start, end=end, **metadata)


@dataclass(frozen=True, slots=True)
class SectionProfile:
    """Profile label set and optional extractor used for section dispatch.

    ``extractor`` is called with ``(source_text, route)`` by
    :meth:`NoteRouter.extract`.  The route exposes absolute ``start`` and
    ``end`` offsets, allowing an extractor to retain source provenance without
    requiring the router to copy note text into its result.
    """

    name: str
    section_labels: tuple[str, ...]
    extractor: SectionExtractor | None = None
    priority: int = 0

    def __post_init__(self) -> None:
        profile_name = _profile_key(self.name)
        if profile_name == UNKNOWN_PROFILE_NAME:
            raise ValueError("unknown is reserved for the fallback route")
        object.__setattr__(self, "name", profile_name)
        object.__setattr__(
            self, "section_labels", _normalize_labels(self.section_labels)
        )
        if self.extractor is not None and not callable(self.extractor):
            raise TypeError("profile extractor must be callable")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("profile priority must be an integer")

    @property
    def labels(self) -> tuple[str, ...]:
        """Return the normalized labels handled by this profile."""

        return self.section_labels

    def matches(self, label: str) -> bool:
        """Return whether this profile accepts a section label."""

        key = _label_key(label)
        return "*" in self.section_labels or key in self.section_labels


@dataclass(frozen=True, slots=True)
class SectionRoute:
    """PHI-free routing decision for one validated contiguous section."""

    section: SectionSpan
    profile: str
    extractor_name: str | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.section, SectionSpan):
            raise TypeError("section must be a SectionSpan")
        object.__setattr__(self, "profile", _profile_key(self.profile))
        if self.extractor_name is not None:
            object.__setattr__(
                self,
                "extractor_name",
                _profile_key(self.extractor_name),
            )
        if self.reason is not None:
            object.__setattr__(self, "reason", _required_text(self.reason, "reason"))

    @property
    def label(self) -> str:
        """Return the original canonical section label."""

        return self.section.label

    @property
    def start(self) -> int:
        """Return the inclusive source offset."""

        return self.section.start

    @property
    def end(self) -> int:
        """Return the exclusive source offset."""

        return self.section.end

    @property
    def offset(self) -> SpanOffset:
        """Return the absolute half-open source offset pair."""

        return self.start, self.end

    @property
    def offsets(self) -> SpanOffset:
        """Alias for :attr:`offset`."""

        return self.offset

    @property
    def is_unknown(self) -> bool:
        """Return whether this section uses the conservative fallback path."""

        return self.profile == UNKNOWN_PROFILE_NAME

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, text-free routing metadata."""

        payload: dict[str, Any] = {
            "label": self.label,
            "profile": self.profile,
            "start": self.start,
            "end": self.end,
            "offset": list(self.offset),
            "extractor": self.extractor_name,
            "reason": self.reason,
        }
        for key in ("content_start", "header_start", "header_end"):
            value = self.section.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                payload[key] = value
        source = self.section.get("source")
        if isinstance(source, str) and source.strip():
            payload["source"] = source.strip()
        confidence = self.section.get("confidence")
        if isinstance(confidence, int | float) and not isinstance(confidence, bool):
            payload["confidence"] = min(max(float(confidence), 0.0), 1.0)
        return payload


@dataclass(frozen=True, slots=True)
class NoteRoutingResult:
    """Validated section routes with an explicit unknown-section partition."""

    routes: tuple[SectionRoute, ...]
    valid: bool = True
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "routes", tuple(self.routes))
        if self.fallback_reason is not None:
            object.__setattr__(
                self,
                "fallback_reason",
                _required_text(self.fallback_reason, "fallback_reason"),
            )

    @property
    def sections(self) -> tuple[SectionRoute, ...]:
        """Return all section routes in source order."""

        return self.routes

    @property
    def known_routes(self) -> tuple[SectionRoute, ...]:
        """Return routes handled by a named specialized profile."""

        return tuple(route for route in self.routes if not route.is_unknown)

    @property
    def unknown_routes(self) -> tuple[SectionRoute, ...]:
        """Return routes handled by the conservative unknown path."""

        return tuple(route for route in self.routes if route.is_unknown)

    @property
    def unknown_sections(self) -> tuple[SectionRoute, ...]:
        """Alias for :attr:`unknown_routes`."""

        return self.unknown_routes

    @property
    def profiles(self) -> tuple[str, ...]:
        """Return unique profile names in source order."""

        values: list[str] = []
        for route in self.routes:
            if route.profile not in values:
                values.append(route.profile)
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        """Return a safe JSON-ready routing report without source text."""

        return {
            "valid": self.valid,
            "fallback_reason": self.fallback_reason,
            "profiles": list(self.profiles),
            "routed_count": len(self.known_routes),
            "unknown_count": len(self.unknown_routes),
            "routes": [route.to_dict() for route in self.routes],
        }


@dataclass(frozen=True, slots=True)
class SectionInput:
    """One extractor input view with source offsets kept out of reports."""

    source_text: str
    route: SectionRoute

    @property
    def section(self) -> SectionSpan:
        """Return the validated section metadata."""

        return self.route.section

    @property
    def label(self) -> str:
        """Return the section label."""

        return self.route.label

    @property
    def start(self) -> int:
        """Return the inclusive source offset."""

        return self.route.start

    @property
    def end(self) -> int:
        """Return the exclusive source offset."""

        return self.route.end

    @property
    def text(self) -> str:
        """Return the section slice for an in-memory extractor call."""

        return self.source_text[self.route.start : self.route.end]

    @property
    def content_text(self) -> str:
        """Return section content after an optional detected header."""

        content_start = self.section.get("content_start", self.route.start)
        if not isinstance(content_start, int) or isinstance(content_start, bool):
            content_start = self.route.start
        return self.source_text[content_start : self.route.end]

    @property
    def offset(self) -> SpanOffset:
        """Return the absolute section offset."""

        return self.route.offset

    def __getitem__(self, key: str) -> object:
        """Expose section metadata to lightweight one-argument extractors."""

        return self.section[key]


@dataclass(frozen=True, slots=True)
class SectionExtraction:
    """In-memory extractor output anchored to one section route."""

    route: SectionRoute
    value: object

    @property
    def offset(self) -> SpanOffset:
        """Return the source offset associated with this output."""

        return self.route.offset

    @property
    def result(self) -> object:
        """Return the extractor-owned result value."""

        return self.value

    def to_dict(self) -> dict[str, Any]:
        """Return only safe route metadata, excluding extractor-owned values."""

        return {"route": self.route.to_dict()}


@dataclass(frozen=True, slots=True)
class NoteExtractionResult:
    """Extractor outputs plus the text-free routing report."""

    routing: NoteRoutingResult
    extractions: tuple[SectionExtraction, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "extractions", tuple(self.extractions))

    @property
    def results(self) -> tuple[SectionExtraction, ...]:
        """Alias for :attr:`extractions`."""

        return self.extractions

    def to_dict(self) -> dict[str, Any]:
        """Return safe routing metadata without serializing extractor values."""

        payload = self.routing.to_dict()
        payload["extraction_count"] = len(self.extractions)
        payload["extractions"] = [item.to_dict() for item in self.extractions]
        return payload


class SectionRoutingError(RuntimeError):
    """Raised when a configured extractor cannot be safely dispatched."""


def _sections_for_profile(profile: NoteTypeProfile) -> tuple[str, ...]:
    labels = list(profile.expected_sections)
    for configured in profile.section_scoped_stage_config.values():
        for label in configured:
            if label != "*" and label not in labels:
                labels.append(label)
    return tuple(labels)


def _default_profiles() -> tuple[SectionProfile, ...]:
    return (
        SectionProfile(
            name=RADIOLOGY_PROFILE.name,
            section_labels=_sections_for_profile(RADIOLOGY_PROFILE),
        ),
        SectionProfile(
            name=PATHOLOGY_PROFILE.name,
            section_labels=_sections_for_profile(PATHOLOGY_PROFILE),
        ),
    )


def _profile_from_value(
    name: str,
    value: object,
    extractors: Mapping[str, SectionExtractor],
) -> SectionProfile:
    if isinstance(value, SectionProfile):
        extractor = extractors.get(value.name, value.extractor)
        return SectionProfile(
            name=value.name,
            section_labels=value.section_labels,
            extractor=extractor,
            priority=value.priority,
        )
    if isinstance(value, NoteTypeProfile):
        profile_name = _profile_key(name or value.name)
        return SectionProfile(
            name=profile_name,
            section_labels=_sections_for_profile(value),
            extractor=extractors.get(profile_name),
        )
    if callable(value):
        profile_name = _profile_key(name)
        labels = {
            profile.name: profile.section_labels for profile in _default_profiles()
        }.get(profile_name, ("*",))
        return SectionProfile(
            name=profile_name,
            section_labels=labels,
            extractor=value,
        )
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        profile_name = _profile_key(name)
        return SectionProfile(
            name=profile_name,
            section_labels=value,
            extractor=extractors.get(profile_name),
        )
    raise TypeError(
        "profiles must contain SectionProfile, NoteTypeProfile, callables, or label iterables"
    )


def _normalize_profiles(
    profiles: Iterable[SectionProfile | NoteTypeProfile] | Mapping[str, object] | None,
    extractors: Mapping[str, SectionExtractor],
) -> tuple[SectionProfile, ...]:
    if profiles is None:
        values = [
            SectionProfile(
                name=profile.name,
                section_labels=profile.section_labels,
                extractor=extractors.get(profile.name, profile.extractor),
                priority=profile.priority,
            )
            for profile in _default_profiles()
        ]
    elif isinstance(profiles, Mapping):
        values = [
            _profile_from_value(str(name), value, extractors)
            for name, value in profiles.items()
        ]
    else:
        values = []
        for item in profiles:
            if isinstance(item, (SectionProfile, NoteTypeProfile)):
                values.append(_profile_from_value(item.name, item, extractors))
            else:
                raise TypeError("profiles must contain profile objects")

    by_name: dict[str, SectionProfile] = {}
    for profile in values:
        key = _profile_key(profile.name)
        if key == UNKNOWN_PROFILE_NAME:
            raise ValueError("unknown is reserved for the fallback route")
        if key in by_name:
            raise ValueError("profile names must be unique")
        by_name[key] = profile
    return tuple(sorted(by_name.values(), key=lambda item: (-item.priority, item.name)))


def _normalize_extractors(
    extractors: Mapping[str, SectionExtractor] | None,
) -> dict[str, SectionExtractor]:
    if extractors is None:
        return {}
    result: dict[str, SectionExtractor] = {}
    for name, extractor in extractors.items():
        key = _profile_key(name)
        if not callable(extractor):
            raise TypeError("extractors must be callable")
        if key in result:
            raise ValueError("extractor names must be unique")
        result[key] = extractor
    return result


class NoteRouter:
    """Route contiguous note sections to deterministic profile extractors."""

    def __init__(
        self,
        profiles: (
            Iterable[SectionProfile | NoteTypeProfile] | Mapping[str, object] | None
        ) = None,
        *,
        extractors: Mapping[str, SectionExtractor] | None = None,
        unknown_extractor: SectionExtractor | None = None,
    ) -> None:
        normalized_extractors = _normalize_extractors(extractors)
        if unknown_extractor is not None and not callable(unknown_extractor):
            raise TypeError("unknown_extractor must be callable")
        self._extractors = normalized_extractors
        self._unknown_extractor = (
            unknown_extractor
            if unknown_extractor is not None
            else normalized_extractors.get(UNKNOWN_PROFILE_NAME)
        )
        self._profiles = _normalize_profiles(profiles, normalized_extractors)
        self._profiles_by_name = {profile.name: profile for profile in self._profiles}

    @property
    def profiles(self) -> tuple[SectionProfile, ...]:
        """Return the immutable profile snapshot in dispatch order."""

        return self._profiles

    def available_profiles(self) -> tuple[str, ...]:
        """Return registered profile names in deterministic order."""

        return tuple(profile.name for profile in self._profiles)

    def profile_for_section(self, label: str) -> SectionProfile | None:
        """Return the highest-priority profile matching ``label``."""

        return next(
            (profile for profile in self._profiles if profile.matches(label)),
            None,
        )

    def route(
        self,
        text: str,
        *,
        sections: Iterable[Mapping[str, Any]] | None = None,
        profile: str | SectionProfile | NoteTypeProfile | None = None,
        language: str | None = None,
    ) -> NoteRoutingResult:
        """Build a safe route plan for every validated section in ``text``.

        When ``sections`` is omitted, the local rules-first detector supplies a
        complete partition.  Caller-supplied spans are validated as a complete
        non-overlapping partition before any profile is selected.  A section
        that does not match a profile receives ``unknown`` and is never routed
        to a specialized extractor.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        detected = (
            tuple(detect_sections(text, language=language))
            if sections is None
            else tuple(
                _coerce_section(item, index) for index, item in enumerate(sections)
            )
        )
        validate_sections(text, detected)

        selected_profile, unknown_reason = self._requested_profile(profile)
        routes: list[SectionRoute] = []
        for section in detected:
            if selected_profile is not None:
                chosen = (
                    selected_profile
                    if selected_profile.matches(section.label)
                    else None
                )
            elif unknown_reason is not None:
                chosen = None
            else:
                chosen = self.profile_for_section(section.label)
            if chosen is None:
                reason = unknown_reason or (
                    PROFILE_MISMATCH_REASON
                    if selected_profile is not None
                    else UNKNOWN_SECTION_REASON
                )
                routes.append(
                    SectionRoute(
                        section=section,
                        profile=UNKNOWN_PROFILE_NAME,
                        extractor_name=(
                            UNKNOWN_PROFILE_NAME
                            if self._unknown_extractor is not None
                            else None
                        ),
                        reason=reason,
                    )
                )
                continue
            routes.append(
                SectionRoute(
                    section=section,
                    profile=chosen.name,
                    extractor_name=chosen.name,
                )
            )

        return NoteRoutingResult(routes=tuple(routes))

    def extract(
        self,
        text: str,
        *,
        sections: Iterable[Mapping[str, Any]] | None = None,
        profile: str | SectionProfile | NoteTypeProfile | None = None,
        language: str | None = None,
    ) -> NoteExtractionResult:
        """Route sections and invoke their configured local extractors.

        Extractor values remain in memory under ``SectionExtraction.value``.
        The serializable result deliberately includes route metadata only, so
        callers must choose their own PHI-safe output contract for extracted
        content.
        """

        routing = self.route(
            text,
            sections=sections,
            profile=profile,
            language=language,
        )
        selected_profile, _ = self._requested_profile(profile)
        outputs: list[SectionExtraction] = []
        for route in routing.routes:
            extractor = self._extractor_for_route(route, selected_profile)
            if extractor is None:
                continue
            self._assert_local(route, extractor)
            try:
                value = _invoke_extractor(extractor, text, route)
            except Exception:
                raise SectionRoutingError(
                    f"{EXTRACTOR_FAILURE_REASON} for configured profile"
                ) from None
            outputs.append(SectionExtraction(route=route, value=value))
        return NoteExtractionResult(routing=routing, extractions=tuple(outputs))

    def run(self, text: str, **kwargs: Any) -> NoteExtractionResult:
        """Alias for :meth:`extract` for pipeline-style callers."""

        return self.extract(text, **kwargs)

    def _requested_profile(
        self,
        requested: str | SectionProfile | NoteTypeProfile | None,
    ) -> tuple[SectionProfile | None, str | None]:
        if requested is None:
            return None, None
        if isinstance(requested, NoteTypeProfile):
            requested = SectionProfile(
                name=requested.name,
                section_labels=_sections_for_profile(requested),
            )
        if isinstance(requested, SectionProfile):
            selected = self._profiles_by_name.get(requested.name)
            return (selected or requested), None
        key = _profile_key(requested)
        selected = self._profiles_by_name.get(key)
        if selected is None:
            return None, UNKNOWN_PROFILE_REASON
        return selected, None

    def _extractor_for_route(
        self,
        route: SectionRoute,
        selected_profile: SectionProfile | None = None,
    ) -> SectionExtractor | None:
        if route.is_unknown:
            return self._unknown_extractor
        profile = self._profiles_by_name.get(route.profile)
        extractor = self._extractors.get(route.profile)
        if extractor is not None:
            return extractor
        if selected_profile is not None and selected_profile.name == route.profile:
            return selected_profile.extractor
        if profile is not None:
            return profile.extractor
        return None

    @staticmethod
    def _assert_local(route: SectionRoute, extractor: SectionExtractor) -> None:
        if bool(getattr(extractor, "network_egress", False)) or bool(
            getattr(extractor, "allows_network", False)
        ):
            raise SectionRoutingError("configured extractor is not local-only")
        del route


def _invoke_extractor(
    extractor: SectionExtractor,
    text: str,
    route: SectionRoute,
) -> object:
    """Call two-argument extractors while supporting a one-argument view."""

    try:
        signature = inspect.signature(extractor)
    except (TypeError, ValueError):
        return extractor(text, route)

    positional = tuple(
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    )
    if (
        any(
            parameter.kind == parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        or len(positional) >= 2
    ):
        return extractor(text, route)
    if len(positional) == 1:
        return extractor(SectionInput(source_text=text, route=route))
    return extractor()


def route_note_sections(
    text: str,
    *,
    sections: Iterable[Mapping[str, Any]] | None = None,
    profiles: (
        Iterable[SectionProfile | NoteTypeProfile] | Mapping[str, object] | None
    ) = None,
    extractors: Mapping[str, SectionExtractor] | None = None,
    profile: str | SectionProfile | NoteTypeProfile | None = None,
    language: str | None = None,
) -> NoteRoutingResult:
    """Return section routes using a one-shot local :class:`NoteRouter`."""

    return NoteRouter(profiles, extractors=extractors).route(
        text,
        sections=sections,
        profile=profile,
        language=language,
    )


def extract_note_sections(
    text: str,
    *,
    sections: Iterable[Mapping[str, Any]] | None = None,
    profiles: (
        Iterable[SectionProfile | NoteTypeProfile] | Mapping[str, object] | None
    ) = None,
    extractors: Mapping[str, SectionExtractor] | None = None,
    unknown_extractor: SectionExtractor | None = None,
    profile: str | SectionProfile | NoteTypeProfile | None = None,
    language: str | None = None,
) -> NoteExtractionResult:
    """Route and execute configured extractors for a single note."""

    return NoteRouter(
        profiles,
        extractors=extractors,
        unknown_extractor=unknown_extractor,
    ).extract(
        text,
        sections=sections,
        profile=profile,
        language=language,
    )


SectionAwareNoteRouter = NoteRouter
NoteSectionProfile = SectionProfile
RoutedSection = SectionRoute
route_note = route_note_sections
extract_note = extract_note_sections


__all__ = [
    "EXTRACTOR_FAILURE_REASON",
    "NoteExtractionResult",
    "NoteRouter",
    "NoteRoutingResult",
    "NoteSectionProfile",
    "PROFILE_MISMATCH_REASON",
    "RoutedSection",
    "SectionAwareNoteRouter",
    "SectionExtraction",
    "SectionExtractor",
    "SectionInput",
    "SectionProfile",
    "SectionRoute",
    "SectionRoutingError",
    "SpanOffset",
    "UNKNOWN_PROFILE_NAME",
    "UNKNOWN_PROFILE_REASON",
    "UNKNOWN_SECTION_REASON",
    "extract_note",
    "extract_note_sections",
    "route_note",
    "route_note_sections",
]
