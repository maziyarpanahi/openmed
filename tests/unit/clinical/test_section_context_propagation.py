"""Synthetic section-scoped context propagation and grounding traps."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical import RECENT, assert_context, assert_context_axes
from openmed.clinical.exporters.fhir import to_fhir
from openmed.clinical.grounding import VocabLoader, VocabSource, ground
from openmed.clinical.sections import SectionSpan, detect_sections

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "section_context_traps.jsonl"
)


def _fixture_rows() -> tuple[dict, list[dict]]:
    rows = [
        json.loads(line)
        for line in _FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[0], rows[1:]


def _spans(text: str, cases: list[dict]) -> list[dict]:
    spans = []
    for case in cases:
        surface = case["text"]
        start = text.index(surface)
        spans.append(
            {
                "text": surface,
                "start": start,
                "end": start + len(surface),
                "canonical_label": "CONDITION",
            }
        )
    return spans


def test_section_context_trap_suite_blocks_family_and_pmh_leakage() -> None:
    metadata, cases = _fixture_rows()
    correct = 0
    total = 0
    leaks: list[str] = []

    assert metadata["synthetic"] is True
    assert len(cases) >= 8
    assert all(case["synthetic"] is True for case in cases)

    for case in cases:
        text = case["text"]
        contextualized = assert_context(
            text,
            _spans(text, case["spans"]),
            sections=detect_sections(text),
        )
        for expected_span, actual in zip(case["spans"], contextualized, strict=True):
            for axis, expected in expected_span["expected"].items():
                total += 1
                correct += actual[axis] == expected
                if (
                    axis == "experiencer"
                    and expected == "family"
                    and actual[axis] != expected
                ):
                    leaks.append(f"{case['id']}:{actual[axis]}")
                if (
                    axis == "temporality"
                    and expected == "historical"
                    and actual[axis] != expected
                ):
                    leaks.append(f"{case['id']}:{actual[axis]}")
            for axis, source in expected_span["expected_sources"].items():
                assert actual["context_sources"][axis] == source, case["id"]

    assert leaks == []
    assert correct / total >= 0.90


def test_loinc_only_section_metadata_applies_scoped_priors() -> None:
    text = "Cedar syndrome. Amber syndrome."
    second_start = text.index("Amber syndrome")
    spans = _spans(
        text,
        [
            {"text": "Cedar syndrome"},
            {"text": "Amber syndrome"},
        ],
    )
    sections = (
        SectionSpan(
            "external",
            0,
            second_start,
            coding={"system": "http://loinc.org", "code": "10157-6"},
        ),
        SectionSpan(
            "external",
            second_start,
            len(text),
            loinc_code="11348-0",
        ),
    )

    family, historical = assert_context(text, spans, sections=sections)

    assert family["experiencer"] == "family"
    assert family["context_sources"]["experiencer"] == "section"
    assert historical["temporality"] == "historical"
    assert historical["context_sources"]["temporality"] == "section"


def test_explicit_modifier_hits_override_section_prior() -> None:
    assertion = assert_context_axes(
        {"text": "Cobalt syndrome"},
        modifier_hits=("currently",),
        section="PMH",
    )

    assert assertion.temporality == RECENT


def test_omitting_sections_is_byte_identical_to_explicit_none() -> None:
    text = "No evidence of pneumonia."
    start = text.index("pneumonia")
    spans = [{"text": "pneumonia", "start": start, "end": start + 9}]

    omitted = assert_context(text, spans)
    explicit_none = assert_context(text, spans, sections=None)

    serialize = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    assert serialize(omitted) == serialize(explicit_none)
    assert "context_sources" not in omitted[0]
    assert "clinical_context_sources" not in omitted[0]["metadata"]


def test_propagated_context_controls_grounded_fhir_conditions(
    tmp_path: Path,
) -> None:
    text = "Family History:\nCedar syndrome.\nPMH:\nAmber syndrome."
    surfaces = ("Cedar syndrome", "Amber syndrome")
    contextualized = assert_context(
        text,
        _spans(text, [{"text": surface} for surface in surfaces]),
        sections=detect_sections(text),
    )
    vocabulary = tmp_path / "synthetic_icd10cm.jsonl"
    vocabulary.write_text(
        "\n".join(
            json.dumps({"code": f"SYN-{index}", "display": surface})
            for index, surface in enumerate(surfaces, start=1)
        )
        + "\n",
        encoding="utf-8",
    )
    loader = VocabLoader(
        cache_dir=tmp_path / "cache",
        local_only=True,
        registry={
            "icd10cm": VocabSource(system="icd10cm", path=vocabulary),
        },
    )

    grounded = ground(contextualized, systems=("icd10cm",), loader=loader)
    bundle = to_fhir(
        grounded,
        subject_reference="Patient/synthetic",
        document_id="synthetic-section-context",
    )

    assert bundle is not None
    resources = [entry["resource"] for entry in bundle["entry"]]
    assert len(resources) == 1
    assert resources[0]["code"]["coding"][0]["code"] == "SYN-2"
    assert resources[0]["clinicalStatus"]["coding"][0]["code"] == "inactive"
