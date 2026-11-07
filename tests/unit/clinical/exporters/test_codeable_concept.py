"""Grounded FHIR CodeableConcept emission and assertion-context tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical import assert_context
from openmed.clinical.context import ClinicalContextResult
from openmed.clinical.exporters import check_codeable_concept
from openmed.clinical.exporters.fhir import (
    GROUNDED_CODE_PROVENANCE_EXTENSION_URL,
    MEDICAL_DEVICE_ASSIST_EXTENSION_URL,
    to_codeable_concept,
    to_fhir,
)
from openmed.clinical.grounding import Candidate, GroundedSpan, ground
from openmed.clinical.grounding.vocab import VocabLoader, VocabSource
from openmed.eval.golden.loader import list_fixture_paths

_FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "openmed"
    / "eval"
    / "golden"
    / "fixtures"
    / "grounded_codeable_concepts.jsonl"
)


def _cases() -> list[dict]:
    return [
        json.loads(line)
        for line in _FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _span(case: dict) -> GroundedSpan:
    context = case["context"]
    return GroundedSpan(
        text=case["text"],
        start=case["start"],
        end=case["end"],
        canonical_label=case["canonical_label"],
        assertion=ClinicalContextResult(**context),
        candidates=tuple(Candidate(**candidate) for candidate in case["candidates"]),
    )


def _nested_values(extension: dict) -> dict[str, object]:
    values = {}
    for item in extension["extension"]:
        value_key = next(key for key in item if key.startswith("value"))
        values[item["url"]] = item[value_key]
    return values


def test_synthetic_gold_emits_valid_auditable_codeable_concepts() -> None:
    cases = _cases()
    exact = 0
    expected_total = 0

    assert len(cases) >= 20
    for case in cases:
        span = _span(case)
        assert span.end - span.start == len(span.text)
        assert span.to_dict()["assertion"] == case["context"]
        concept = to_codeable_concept(span)

        assert check_codeable_concept(concept) == []
        actual_codings = {
            (coding["system"], coding["code"], coding["display"])
            for coding in concept["coding"]
        }
        expected_codings = {tuple(coding) for coding in case["expected_codings"]}
        assert actual_codings == expected_codings
        exact += len(actual_codings & expected_codings)
        expected_total += len(expected_codings)
        assert len(concept["coding"]) == len(
            {candidate["system"].casefold() for candidate in case["candidates"]}
        )

        for coding in concept["coding"]:
            provenance = next(
                extension
                for extension in coding["extension"]
                if extension["url"] == GROUNDED_CODE_PROVENANCE_EXTENSION_URL
            )
            values = _nested_values(provenance)
            assert {
                "linker",
                "score",
                "matched_alias",
                "vocab_version",
            } <= values.keys()
            assert values["vocab_version"] == coding["version"] == "synthetic-v1"
            assert values["evidence_start"] == span.start
            assert values["evidence_end"] == span.end

        assist = next(
            extension
            for extension in concept["extension"]
            if extension["url"] == MEDICAL_DEVICE_ASSIST_EXTENSION_URL
        )
        assist_values = _nested_values(assist)
        assert assist_values["assist_only"] is True
        assert assist_values["autonomous_decision"] is False
        assert assist_values["evidence_start"] == span.start
        assert assist_values["evidence_end"] == span.end
        assert "human review" in str(assist_values["disclaimer"])

    assert exact / expected_total >= 0.95


def test_context_result_refutes_negation_and_excludes_family_experiencer() -> None:
    cases = {case["id"]: case for case in _cases()}

    negated = to_fhir(
        _span(cases["grounded-cc-018-negated-trap"]),
        subject_reference="Patient/synthetic",
    )
    family = to_fhir(
        _span(cases["grounded-cc-020-family"]),
        subject_reference="Patient/synthetic",
    )

    assert negated is not None
    verification = negated["verificationStatus"]["coding"][0]["code"]
    assert verification == "refuted"
    assert "clinicalStatus" not in negated
    assert family is None


def test_assert_context_metadata_flows_through_grounding_to_fhir(
    tmp_path: Path,
) -> None:
    text = "No evidence of Aster syndrome. Her mother had Beryl syndrome."
    surfaces = ("Aster syndrome", "Beryl syndrome")
    spans = [
        {
            "text": surface,
            "start": text.index(surface),
            "end": text.index(surface) + len(surface),
            "canonical_label": "CONDITION",
        }
        for surface in surfaces
    ]
    contextualized = assert_context(text, spans)
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
        document_id="synthetic-context-flow",
    )

    assert bundle is not None
    resources = [entry["resource"] for entry in bundle["entry"]]
    assert len(resources) == 1
    assert resources[0]["code"]["coding"][0]["code"] == "SYN-1"
    assert resources[0]["verificationStatus"]["coding"][0]["code"] == "refuted"


def test_emitter_keeps_only_first_ranked_candidate_per_system() -> None:
    span = GroundedSpan(
        text="Wren condition",
        start=4,
        end=18,
        candidates=(
            Candidate(
                "ICD10CM",
                "SYN-FIRST",
                "Wren condition",
                0.9,
                source="sparse",
                matched_alias="wren condition",
                vocab_version="synthetic-v1",
            ),
            Candidate(
                "icd10cm",
                "SYN-SECOND",
                "Wren condition",
                0.8,
                source="dense",
                matched_alias="wren condition",
                vocab_version="synthetic-v1",
            ),
        ),
    )

    concept = to_codeable_concept(span)

    assert [coding["code"] for coding in concept["coding"]] == ["SYN-FIRST"]


def test_emitter_rejects_coding_without_nonempty_evidence_offsets() -> None:
    span = GroundedSpan(
        text="Wren condition",
        start=4,
        end=4,
        candidates=(Candidate("ICD10CM", "SYN-W", "Wren condition", 1.0),),
    )

    with pytest.raises(ValueError, match="evidence offsets"):
        to_codeable_concept(span)


def test_fixture_is_not_generic_deidentification_gold() -> None:
    assert all(
        path.name != "grounded_codeable_concepts.jsonl" for path in list_fixture_paths()
    )
