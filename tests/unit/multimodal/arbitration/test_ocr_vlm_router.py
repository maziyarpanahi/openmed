"""Synthetic disagreements, exact source preservation and real admission gates."""

import json
from dataclasses import replace

import pytest

from openmed.multimodal.arbitration.ocr_vlm_router import (
    AgreementPolicy,
    ArbitrationError,
    Disagreement,
    ReadingCandidate,
    ReadingPair,
    ReadingSource,
    RoutingState,
    SemanticType,
    TextNormalization,
    prepare_agreed_values,
    route_ocr_vlm,
)
from openmed.multimodal.evidence import EvidenceNode, NodeKind, PageGeometry
from openmed.multimodal.vision.providers.base import Confidence


def candidate(
    index,
    source,
    text="Synthetic observation",
    *,
    field_id=50,
    box=(100, 100, 200, 120),
    **changes,
):
    return replace(
        ReadingCandidate(
            candidate_id=f"{index:032x}",
            field_id=f"{field_id:032x}",
            source=source,
            text=text,
            semantic_type=SemanticType.TEXT,
            confidence=Confidence(score=0.9),
            evidence=EvidenceNode(
                node_id=f"{index + 1000:032x}",
                kind=NodeKind.OCR if source is ReadingSource.OCR else NodeKind.PIXELS,
                source_digest="a" * 64,
                geometry=PageGeometry(page=0, width=1000, height=1000),
                region=box,
                span=(0, len(text)) if source is ReadingSource.OCR else None,
            ),
        ),
        **changes,
    )


def pair(index=1, texts=("Synthetic observation", "Synthetic observation"), **changes):
    return replace(
        ReadingPair(
            pair_id=f"{index:032x}",
            ocr=candidate(index * 2, ReadingSource.OCR, texts[0], field_id=index + 50),
            vlm=candidate(
                index * 2 + 1, ReadingSource.VLM, texts[1], field_id=index + 50
            ),
        ),
        **changes,
    )


def policy(**changes):
    return replace(
        AgreementPolicy(
            version="synthetic-v1",
            allowed_types=(SemanticType.TEXT,),
            normalization=TextNormalization.EXACT,
            minimum_geometry_iou=0.8,
            minimum_confidence=0.8,
            preferred_source=ReadingSource.OCR,
            maximum_review_items=128,
        ),
        **changes,
    )


def reasons(source, **changes):
    return route_ocr_vlm((source,), policy(**changes)).comparisons[0].reasons


def test_exact_agreement_admits_original_reading_with_both_citations():
    source = pair()
    report = route_ocr_vlm((source,), policy())
    result = report.comparisons[0]
    assert result.state is RoutingState.AGREEMENT
    assert result.reasons == () and report.review_queue == ()
    assert result.selected_candidate_id == source.ocr.candidate_id
    assert result.geometry_iou == 1
    data = json.loads(prepare_agreed_values((source,), policy()).data_json)
    assert data["items"][0]["value"] == source.ocr.text
    assert data["items"][0]["evidence"][0]["span"] == [0, len(source.ocr.text)]
    assert data["items"][0]["evidence"][1]["region"] == [100, 100, 200, 120]


def test_normalization_is_opt_in_and_selected_string_is_never_rewritten():
    source = pair(texts=("  Cafe\u0301\tobservation  ", "Café observation"))
    assert Disagreement.TEXT in reasons(source)
    normalized = policy(normalization=TextNormalization.NFC_WHITESPACE)
    for preferred, expected in (
        (ReadingSource.OCR, source.ocr),
        (ReadingSource.VLM, source.vlm),
    ):
        selected = replace(normalized, preferred_source=preferred)
        exported = prepare_agreed_values((source,), selected)
        row = json.loads(exported.data_json)["items"][0]
        assert row["value"] == expected.text
        assert row["selected_candidate_id"] == expected.candidate_id


@pytest.mark.parametrize(
    "texts",
    [
        ("0.5 mg", "5 mg"),
        ("mg", "Mg"),
        ("01/02", "02/01"),
        ("０.５", "0.5"),
        ("12.0", "12"),
        ("not present", "present"),
    ],
)
def test_numeric_case_date_width_and_negation_conflicts_are_not_normalized_away(texts):
    source = pair(texts=texts)
    setting = policy(normalization=TextNormalization.NFC_WHITESPACE)
    report = route_ocr_vlm((source,), setting)
    assert Disagreement.TEXT in report.comparisons[0].reasons
    assert report.comparisons[0].selected_candidate_id is None
    with pytest.raises(ArbitrationError, match="require review"):
        prepare_agreed_values((source,), setting)


def test_confidence_does_not_break_a_conflict_or_replace_policy_source_preference():
    source = pair(texts=("Synthetic first", "Synthetic other"))
    source = replace(source, vlm=replace(source.vlm, confidence=Confidence(score=1)))
    assert (
        route_ocr_vlm((source,), policy()).comparisons[0].selected_candidate_id is None
    )
    same = pair(vlm=replace(pair().vlm, confidence=Confidence(score=1)))
    assert (
        route_ocr_vlm((same,), policy()).comparisons[0].selected_candidate_id
        == same.ocr.candidate_id
    )


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"source_digest": "b" * 64}, Disagreement.SOURCE),
        (
            {"geometry": PageGeometry(page=1, width=1000, height=1000)},
            Disagreement.PAGE,
        ),
        (
            {"geometry": PageGeometry(page=0, width=2000, height=1000)},
            Disagreement.GEOMETRY,
        ),
        ({"region": (400, 400, 500, 420)}, Disagreement.GEOMETRY),
        ({"region": (150, 100, 250, 120)}, Disagreement.GEOMETRY),
    ],
)
def test_matching_text_cannot_override_wrong_source_page_or_geometry(change, expected):
    source = pair()
    changed = replace(
        source, vlm=replace(source.vlm, evidence=replace(source.vlm.evidence, **change))
    )
    assert expected in reasons(changed)
    with pytest.raises(ArbitrationError):
        prepare_agreed_values((changed,), policy())


def test_iou_threshold_includes_equality_and_requires_positive_overlap():
    source = pair()
    changed = replace(
        source,
        vlm=replace(
            source.vlm,
            evidence=replace(source.vlm.evidence, region=(100, 100, 300, 120)),
        ),
    )
    assert (
        route_ocr_vlm((changed,), policy(minimum_geometry_iou=0.5)).comparisons[0].state
        is RoutingState.AGREEMENT
    )
    assert Disagreement.GEOMETRY in reasons(changed, minimum_geometry_iou=0.5001)
    with pytest.raises(ArbitrationError):
        policy(minimum_geometry_iou=0)


def test_tiny_valid_boxes_do_not_underflow_to_zero_area():
    source = pair()
    box = (0, 0, 1e-200, 1e-200)
    source = replace(
        source,
        ocr=replace(source.ocr, evidence=replace(source.ocr.evidence, region=box)),
        vlm=replace(source.vlm, evidence=replace(source.vlm.evidence, region=box)),
    )
    assert route_ocr_vlm((source,), policy()).comparisons[0].geometry_iou == 1


def test_field_and_semantic_type_mismatches_are_independent_conflicts():
    source = pair()
    changed = replace(
        source,
        vlm=replace(source.vlm, field_id="c" * 32, semantic_type=SemanticType.NUMBER),
    )
    codes = reasons(changed, allowed_types=(SemanticType.TEXT, SemanticType.NUMBER))
    assert Disagreement.FIELD in codes and Disagreement.SEMANTIC_TYPE in codes
    assert Disagreement.TYPE_POLICY not in codes


def test_semantic_permissions_are_required_even_for_matching_high_confidence_text():
    source = pair()
    changed = replace(
        source,
        ocr=replace(source.ocr, semantic_type=SemanticType.DOSE),
        vlm=replace(source.vlm, semantic_type=SemanticType.DOSE),
    )
    assert reasons(changed) == (Disagreement.TYPE_POLICY,)
    assert not reasons(changed, allowed_types=(SemanticType.DOSE,))
    assert reasons(source, allowed_types=()) == (Disagreement.TYPE_POLICY,)


def test_unknown_semantics_and_unknown_confidence_cannot_be_allowed():
    source = pair()
    source = replace(
        source,
        ocr=replace(
            source.ocr, semantic_type=SemanticType.UNKNOWN, confidence=Confidence()
        ),
        vlm=replace(source.vlm, semantic_type=SemanticType.UNKNOWN),
    )
    codes = reasons(source, minimum_confidence=0)
    assert (
        Disagreement.UNKNOWN_TYPE in codes and Disagreement.CONFIDENCE_UNKNOWN in codes
    )
    with pytest.raises(ArbitrationError):
        policy(allowed_types=(SemanticType.UNKNOWN,))


def test_both_confidences_must_meet_the_floor_without_averaging():
    source = pair()
    source = replace(
        source,
        ocr=replace(source.ocr, confidence=Confidence(score=0.79)),
        vlm=replace(source.vlm, confidence=Confidence(score=1)),
    )
    assert Disagreement.CONFIDENCE_LOW in reasons(source)
    assert not reasons(source, minimum_confidence=0.79)


def test_matching_handwriting_always_requires_its_separate_review_gate():
    source = pair()
    source = replace(source, vlm=replace(source.vlm, handwritten=True))
    assert reasons(source) == (Disagreement.HANDWRITING,)
    with pytest.raises(ArbitrationError):
        prepare_agreed_values((source,), policy())


def test_review_queue_is_complete_and_preserves_both_visual_sources():
    sources = (
        pair(1),
        pair(2, texts=("Synthetic A", "Synthetic B")),
        pair(3, texts=("Synthetic C", "Synthetic D")),
    )
    report = route_ocr_vlm(sources, policy(maximum_review_items=2))
    assert [item.pair_id for item in report.review_queue] == [
        sources[1].pair_id,
        sources[2].pair_id,
    ]
    for source, item in zip(sources[1:], report.review_queue):
        assert item.evidence == (source.ocr.evidence, source.vlm.evidence)
        assert item.candidate_ids == (source.ocr.candidate_id, source.vlm.candidate_id)
    with pytest.raises(ArbitrationError, match="capacity exceeded"):
        route_ocr_vlm(sources, policy(maximum_review_items=1))
    with pytest.raises(ArbitrationError, match="require review"):
        prepare_agreed_values(sources, policy())


def test_zero_review_capacity_allows_only_complete_agreements():
    assert route_ocr_vlm((pair(),), policy(maximum_review_items=0)).review_queue == ()
    with pytest.raises(ArbitrationError, match="capacity exceeded"):
        route_ocr_vlm((pair(texts=("First", "Other")),), policy(maximum_review_items=0))


def test_review_packet_blocks_and_retains_source_mapping_without_values():
    source = pair(texts=("Synthetic original", "Synthetic disputed"))
    report = route_ocr_vlm((source,), policy())
    packet = report.to_review_packet()
    assert len(packet.findings) == 1 and len(packet.citations) == 2
    assert all(gate.blocking and not gate.passed for gate in packet.gate_results)
    assert {citation.citation_id for citation in packet.citations} == set(
        report.review_citation_map
    )
    assert set(report.review_citation_map.values()) == {
        source.ocr.evidence,
        source.vlm.evidence,
    }
    assert set(json.loads(report.to_json())["review_citation_ids"].values()) == {
        source.ocr.evidence.node_id,
        source.vlm.evidence.node_id,
    }
    for text in (source.ocr.text, source.vlm.text):
        assert text not in repr(source) + repr(report) + repr(packet) + report.to_json()


def test_agreement_report_and_export_repr_never_include_protected_strings():
    source = pair(texts=('Synthetic "quoted" \n value', 'Synthetic "quoted" \n value'))
    exported = prepare_agreed_values((source,), policy())
    assert json.loads(exported.data_json)["items"][0]["value"] == source.ocr.text
    assert (
        "Synthetic" not in repr(source.ocr) + repr(exported) + exported.to_report_json()
    )


def test_current_inputs_are_recomputed_instead_of_trusting_old_agreements():
    source = pair()
    old = route_ocr_vlm((source,), policy())
    changed = replace(
        source, vlm=replace(source.vlm, text="Synthetic different observation")
    )
    current = route_ocr_vlm((changed,), policy())
    assert current.input_digest != old.input_digest
    assert current.comparisons[0].selected_candidate_id is None
    with pytest.raises(ArbitrationError):
        prepare_agreed_values((changed,), policy())
    assert (
        route_ocr_vlm((source,), policy(version="synthetic-v2")).input_digest
        != old.input_digest
    )
    assert (
        route_ocr_vlm(
            (source,), policy(preferred_source=ReadingSource.VLM)
        ).policy_digest
        != old.policy_digest
    )


def test_determinism_and_batch_order_do_not_change_individual_decisions():
    sources = (pair(1), pair(2, texts=("Synthetic first", "Synthetic other")))
    first = route_ocr_vlm(sources, policy())
    assert first.to_json() == route_ocr_vlm(sources, policy()).to_json()
    assert (
        tuple(reversed(first.comparisons))
        == route_ocr_vlm(tuple(reversed(sources)), policy()).comparisons
    )


def test_duplicate_pairs_fields_candidates_or_visual_ids_are_rejected():
    first, second = pair(1), pair(2)
    malformed = (
        (first, first),
        (first, replace(second, ocr=replace(second.ocr, field_id=first.ocr.field_id))),
        (
            first,
            replace(
                second, ocr=replace(second.ocr, candidate_id=first.ocr.candidate_id)
            ),
        ),
        (
            first,
            replace(
                second,
                ocr=replace(
                    second.ocr,
                    evidence=replace(
                        second.ocr.evidence, node_id=first.ocr.evidence.node_id
                    ),
                ),
            ),
        ),
    )
    for batch in malformed:
        with pytest.raises(ArbitrationError):
            route_ocr_vlm(batch, policy())


def test_input_bounds_and_fixed_exceptions():
    with pytest.raises(ArbitrationError):
        route_ocr_vlm((), policy())
    with pytest.raises(ArbitrationError):
        route_ocr_vlm(tuple(pair(i) for i in range(1, 130)), policy())
    with pytest.raises(ArbitrationError):
        route_ocr_vlm(
            tuple(pair(i, texts=("x" * 4096, "x" * 4096)) for i in range(1, 34)),
            policy(),
        )
    for text in ("", " ", "Synthetic" * 1000, "Synthetic\ud800"):
        with pytest.raises(ArbitrationError) as raised:
            replace(pair().vlm, text=text)
        assert "Synthetic" not in str(raised.value)
        assert raised.value.__context__ is None
    for changes in (
        {"maximum_review_items": True},
        {"minimum_geometry_iou": float("nan")},
        {"minimum_confidence": float("inf")},
        {"allowed_types": (SemanticType.TEXT, SemanticType.TEXT)},
    ):
        with pytest.raises(ArbitrationError):
            policy(**changes)


def test_candidate_contracts_do_not_silently_relabel_sources_or_spans():
    source = pair()
    with pytest.raises(ArbitrationError):
        replace(source.ocr, text="short")
    with pytest.raises(ArbitrationError):
        replace(source.ocr, source=ReadingSource.VLM)
    with pytest.raises(ArbitrationError):
        replace(source, ocr=source.vlm, vlm=source.ocr)
    with pytest.raises(ArbitrationError):
        replace(source.vlm, handwritten=1)
