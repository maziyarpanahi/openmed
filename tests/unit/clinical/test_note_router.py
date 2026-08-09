"""Synthetic offline tests for section-aware clinical note routing."""

from __future__ import annotations

import pytest

from openmed.clinical.note_router import (
    NoteRouter,
    SectionProfile,
    SectionRoutingError,
)
from openmed.clinical.sections import SectionSpan, detect_sections


def test_default_profiles_dispatch_contiguous_sections_with_absolute_offsets() -> None:
    text = (
        "RADIOLOGY REPORT\n"
        "FINDINGS: Synthetic opacity.\n"
        "IMPRESSION: Synthetic stable finding.\n"
        "MEDICATIONS: Synthetic tablet."
    )
    calls: list[tuple[str, tuple[int, int]]] = []

    def extract(source: str, route) -> dict[str, object]:
        calls.append((route.label, route.offset))
        return {"offset": route.offset, "section": source[route.start : route.end]}

    router = NoteRouter(extractors={"radiology": extract})
    planned = router.route(text)

    assert [route.label for route in planned.routes] == [
        "unsectioned",
        "findings",
        "impression",
        "medications",
    ]
    assert [route.profile for route in planned.routes] == [
        "unknown",
        "radiology",
        "radiology",
        "unknown",
    ]
    findings = planned.routes[1]
    assert findings.offset == (
        text.index("FINDINGS"),
        text.index("IMPRESSION"),
    )
    assert planned.routes[2].offset == (
        text.index("IMPRESSION"),
        text.index("MEDICATIONS"),
    )

    extracted = router.extract(text)
    assert calls == [
        ("findings", findings.offset),
        ("impression", planned.routes[2].offset),
    ]
    assert [item.offset for item in extracted.extractions] == [
        findings.offset,
        planned.routes[2].offset,
    ]
    assert "Synthetic opacity" not in str(extracted.to_dict())


def test_unknown_sections_use_explicit_fallback_extractor() -> None:
    text = "HPI: Synthetic cough.\nFINDINGS: Synthetic clear lungs."
    fallback_calls: list[tuple[int, int]] = []

    def fallback(section) -> str:
        fallback_calls.append(section.offset)
        return section.text

    result = NoteRouter(
        extractors={"radiology": lambda _text, _route: "specialized"},
        unknown_extractor=fallback,
    ).extract(text)

    assert [route.profile for route in result.routing.routes] == [
        "unknown",
        "radiology",
    ]
    assert fallback_calls == [result.routing.routes[0].offset]
    assert result.extractions[0].result == "HPI: Synthetic cough.\n"
    assert result.routing.unknown_sections == (result.routing.routes[0],)


def test_pathology_profile_includes_staging_and_preserves_metadata_offsets() -> None:
    text = (
        "PATHOLOGY REPORT\n"
        "SPECIMEN: Synthetic tissue.\n"
        "STAGING: pT1 N0.\n"
        "DIAGNOSIS: Synthetic benign lesion."
    )

    result = NoteRouter().route(text)

    assert [route.profile for route in result.routes] == [
        "unknown",
        "pathology",
        "pathology",
        "pathology",
    ]
    staging = next(route for route in result.routes if route.label == "staging")
    assert staging.start == text.index("STAGING")
    assert staging.end == text.index("DIAGNOSIS")
    assert staging.to_dict()["offset"] == [staging.start, staging.end]


def test_explicit_profile_mismatch_and_unknown_profile_fail_closed() -> None:
    text = "FINDINGS: Synthetic observation."
    router = NoteRouter()

    mismatch = router.route(text, profile="pathology")
    assert mismatch.routes[0].profile == "unknown"
    assert mismatch.routes[0].reason == "profile_section_mismatch"

    unsupported = router.route(text, profile="unsupported")
    assert unsupported.routes[0].profile == "unknown"
    assert unsupported.routes[0].reason == "unknown_profile"


def test_custom_profiles_require_a_complete_non_overlapping_partition() -> None:
    text = "Synthetic header\nSynthetic body"
    sections = (
        SectionSpan(label="header", start=0, end=text.index("Synthetic body")),
        SectionSpan(label="body", start=text.index("Synthetic body"), end=len(text)),
    )
    router = NoteRouter(
        profiles=(SectionProfile("local", ("body",)),),
    )

    result = router.route(text, sections=sections)

    assert [route.profile for route in result.routes] == ["unknown", "local"]
    assert result.routes[1].offset == (sections[1].start, sections[1].end)

    with pytest.raises(ValueError, match="gap"):
        router.route(
            text,
            sections=(
                SectionSpan(label="header", start=0, end=4),
                SectionSpan(label="body", start=5, end=len(text)),
            ),
        )


def test_extractors_marked_as_network_capable_are_refused() -> None:
    class NetworkExtractor:
        allows_network = True

        def __call__(self, _text: str, _route) -> str:
            return "not used"

    text = "FINDINGS: Synthetic observation."
    router = NoteRouter(extractors={"radiology": NetworkExtractor()})

    with pytest.raises(SectionRoutingError, match="local-only"):
        router.extract(text)


def test_extractor_failure_does_not_echo_source_or_original_exception() -> None:
    text = "FINDINGS: Synthetic observation."
    secret = "synthetic-private-value"

    def failing(_text: str, _route) -> None:
        raise RuntimeError(secret)

    with pytest.raises(SectionRoutingError) as raised:
        NoteRouter(extractors={"radiology": failing}).extract(text)

    assert secret not in str(raised.value)
    assert secret not in repr(raised.value)


def test_route_reports_are_deterministic_and_text_free() -> None:
    text = "Unknown section: Synthetic value."
    router = NoteRouter()

    reports = [router.route(text).to_dict() for _ in range(5)]

    assert reports == [reports[0]] * 5
    assert "Synthetic value" not in str(reports[0])
    assert reports[0]["unknown_count"] == 1


def test_detected_sections_are_the_same_validated_partition_on_repeat() -> None:
    text = "RADIOLOGY REPORT\nFINDINGS: Synthetic observation."
    sections = detect_sections(text)
    router = NoteRouter()

    assert (
        router.route(text, sections=sections).to_dict()
        == router.route(
            text,
            sections=sections,
        ).to_dict()
    )
