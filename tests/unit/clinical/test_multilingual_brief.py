"""Synthetic full-pipeline Unicode contracts, not clinical language quality."""

import json
from dataclasses import replace

import pytest

from openmed.clinical import build_clinical_brief
from openmed.clinical.brief import STAGES, BriefRefusal
from openmed.clinical.citation_boundaries import (
    CitationBoundaryError,
    build_deidentification_offset_map,
    validate_citation_boundaries,
)
from openmed.core.text_normalize import normalize_for_detection
from tests.fixtures.clinical.multilingual_briefs import (
    SCENARIOS,
    WIRE,
    corpus,
    fixture_context,
    parity_report,
)

CASES = corpus()["cases"]


def test_versioned_corpus_boundary():
    config = corpus()
    assert config["schema_version"] == 1
    assert config["synthetic_only"] is True
    assert config["normalization"] == "NFKC"
    assert {case["case_id"] for case in CASES} == {
        "latin-en",
        "latin-fr",
        "rtl-ar",
        "indic-hi",
        "mixed-en-hi",
    }


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_source_claim_and_raw_offsets_round_trip_without_protected_splits(
    case, monkeypatch
):
    import importlib

    observed = {}
    for module_name, function_name in (
        ("nli_assertion_pairs", "build_nli_pair"),
        ("nli_temporal_pairs", "build_temporal_nli_pair"),
        ("nli_experiencer_pairs", "build_experiencer_nli_pair"),
    ):
        module = importlib.import_module("openmed.clinical." + module_name)
        original_function = getattr(module, function_name)
        records = observed[function_name] = []

        def record(*args, _original=original_function, _records=records, **kwargs):
            _records.append(kwargs)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, function_name, record)
    raw, normalization, value, context, generated = fixture_context(case)
    brief = build_clinical_brief(value, model="extractive", context=context)
    assert brief.refusal_reason is None
    assert brief.summary == generated
    assert brief.to_dict()["stages"] == list(STAGES)
    assert brief.metrics["coverage"]["recall"] == 1.0
    assert brief.envelope["requires_human_review"]
    assert [
        row["premise_assertion"]["negation"] for row in observed["build_nli_pair"]
    ] == ["negated", "affirmed", "affirmed"]
    assert all(
        row["premise_assertion"] == row["hypothesis_assertion"]
        for row in observed["build_nli_pair"]
    )
    assert [
        row["hypothesis_experiencer"] for row in observed["build_experiencer_nli_pair"]
    ] == ["patient", "family", "patient"]
    assert [
        row["hypothesis_status"] for row in observed["build_temporal_nli_pair"]
    ] == ["recent", "recent", "historical"]
    mapping = build_deidentification_offset_map(value)
    assert len(mapping.replacements) == 1
    for citation, ref in zip(brief.citations, context.packet.references):
        post = citation["source_start"], citation["source_end"]
        source = mapping.map_post_span(*post)
        assert source is not None
        assert mapping.map_source_span(*source) == post
        original = normalization.to_original_span(*source)
        expected = brief.summary[citation["output_start"] : citation["output_end"]]
        assert value.deidentified_text[slice(*post)] == expected
        assert normalize_for_detection(raw[slice(*original)])[0] == expected
        assert post == (ref.start, ref.end)
    replacement = mapping.replacements[0]
    protected = replacement.post_start, replacement.post_end
    assert mapping.map_source_span(*mapping.map_post_span(*protected)) == protected
    # Every interior marker boundary is ambiguous, including the boundary
    # adjoining the first clinical claim. No partial replacement is citeable.
    for boundary in range(replacement.post_start + 1, replacement.post_end):
        report = validate_citation_boundaries(
            [
                {
                    "start": boundary,
                    "end": replacement.post_end + 1,
                    "document_digest": mapping.document_digest,
                }
            ],
            mapping,
            raise_on_error=False,
        )
        assert report.accepted_count == 0 and report.rejected_count == 1
    safe = json.dumps(brief.to_dict()) + mapping.to_json() + repr(brief)
    assert value.pii_entities[0].text not in safe
    assert all(sentence not in safe for sentence in case["sentences"])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_governed_conflicts_and_explicit_unsupported_provider(case, scenario):
    _, _, value, context, generated = fixture_context(case, scenario)
    brief = build_clinical_brief(value, model=lambda _: generated, context=context)
    expected = {
        "preserved": None,
        "negation_conflict": BriefRefusal.UNSUPPORTED_CLAIM,
        "family_conflict": BriefRefusal.UNSUPPORTED_CLAIM,
        "temporal_conflict": BriefRefusal.UNSUPPORTED_CLAIM,
        "nli_contradiction": BriefRefusal.NLI_REJECTED,
        "unsupported_provider": BriefRefusal.NLI_UNAVAILABLE,
    }[scenario]
    assert brief.refusal_reason is expected
    if expected:
        assert brief.summary == "" and not brief.citations and not brief.verdicts
    assert [f.negation for f in context.facts] == ["negated", "affirmed", "affirmed"]
    assert context.facts[1].experiencer == "family"
    assert context.facts[2].temporality == "historical"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_identifiers_and_evidence_drift_fail_closed(case):
    _, _, value, context, _ = fixture_context(case)
    identifier = value.pii_entities[0].text
    leaked = build_clinical_brief(value, model=lambda _: identifier, context=context)
    assert leaked.refusal_reason is BriefRefusal.PRIVACY
    assert identifier not in json.dumps(leaked.to_response())
    changed = replace(value, deidentified_text=value.deidentified_text + " changed")
    assert (
        build_clinical_brief(
            changed, model="extractive", context=context
        ).refusal_reason
        is BriefRefusal.INVALID_EVIDENCE
    )
    context = replace(
        context,
        facts=(replace(context.facts[0], negation="affirmed"), *context.facts[1:]),
    )
    assert (
        build_clinical_brief(value, model="extractive", context=context).refusal_reason
        is BriefRefusal.INVALID_EVIDENCE
    )


def test_populated_replacement_aliases_must_agree():
    _, _, value, _, _ = fixture_context(CASES[0])
    value.pii_entities[0].surrogate = "[OTHER]"
    with pytest.raises(CitationBoundaryError):
        build_deidentification_offset_map(value)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_language_pack_presence_never_establishes_clinical_support(case):
    _, _, value, context, _ = fixture_context(case, "unsupported_provider")
    result = build_clinical_brief(
        value, model="extractive", context=context
    ).to_response()
    report = parity_report(case, "unsupported_provider", {"composer": result})
    assert report["language_pack_count"] == len(case["languages"])
    assert report["provider_support"] == "unsupported"
    assert report["reason"] == "nli_unavailable"
    assert report["model_quality"] == "not_evaluated"
    assert report["clinical_language_support"] == "not_established"
    assert report["claim_atomicity"] == (
        "supported" if case["languages"] == ["en"] else "unsupported"
    )
    assert value.original_text not in json.dumps(report)
    assert value.pii_entities[0].text not in json.dumps(report)
    assert not parity_report(
        case,
        "unsupported_provider",
        {"one": result, "two": {**result, "refusal_reason": "invalid_evidence"}},
    )["contract_parity"]


def test_native_wire_packets_come_from_current_public_composer():
    wire = json.loads(WIRE.read_text(encoding="utf-8"))
    assert wire["schema_version"] == 1 and wire["synthetic_only"] is True
    assert len(wire["cases"]) == len(CASES) * 2
    for row in wire["cases"]:
        case = next(c for c in CASES if c["case_id"] == row["case_id"])
        _, _, value, context, _ = fixture_context(case, row["scenario"])
        brief = build_clinical_brief(value, model="extractive", context=context)
        assert row["source"] == value.deidentified_text
        assert row["generator_output"] == brief.summary
        assert row["evaluation_json"] == json.dumps(
            brief.to_response(), sort_keys=True, separators=(",", ":")
        )
