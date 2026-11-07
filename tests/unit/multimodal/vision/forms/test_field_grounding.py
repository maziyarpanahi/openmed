"""Synthetic spatial associations, explicit abstentions and protected exports."""

import json
from dataclasses import replace

import pytest

from openmed.multimodal.evidence import (
    EvidenceNode,
    NodeKind,
    PageGeometry,
    SpatialUnit,
)
from openmed.multimodal.vision.forms.field_grounding import (
    ControlState,
    FormAssociationReview,
    FormGroundingError,
    FormLabel,
    FormSection,
    FormValue,
    GroundingPolicy,
    GroundingState,
    LabelRelation,
    ValueKind,
    VisualForm,
    export_grounded_fields,
    ground_form_fields,
)
from openmed.multimodal.vision.handwriting.uncertainty import (
    HandwritingCandidate,
    HandwritingReview,
    ReviewAction,
    TranscriptionAlternative,
)
from openmed.multimodal.vision.providers.base import Confidence


def node(index, box, *, source="a" * 64, page=0):
    return EvidenceNode(
        node_id=f"{index:032x}",
        kind=NodeKind.PIXELS,
        source_digest=source,
        geometry=PageGeometry(page=page, width=1000, height=1000),
        region=box,
    )


def policy(**changes):
    return replace(
        GroundingPolicy(
            version="synthetic-v1",
            maximum_gap=0.2,
            minimum_axis_overlap=0.5,
            minimum_score=0.1,
            ambiguity_margin=0.05,
        ),
        **changes,
    )


def label(
    index=2,
    box=(100, 100, 200, 120),
    *,
    section_id=1,
    relation=LabelRelation.RIGHT,
    text="Synthetic label",
):
    return FormLabel(
        evidence=node(index, box),
        text=text,
        section_id=f"{section_id:032x}",
        relation=relation,
        accepts=(ValueKind.TEXT, ValueKind.HANDWRITING),
        confidence=Confidence(score=0.8),
    )


def value(index=3, box=(230, 100, 300, 120), *, text="Synthetic value"):
    return FormValue(
        evidence=node(index, box),
        kind=ValueKind.TEXT,
        text=text,
        confidence=Confidence(score=0.9),
    )


def form(**changes):
    return replace(
        VisualForm(
            sections=(FormSection(node(1, (0, 0, 1000, 1000))),),
            labels=(label(),),
            values=(value(),),
        ),
        **changes,
    )


def reviews(report, choices=None, **changes):
    choices = choices or [item.associations[0] for item in report.values]
    return tuple(
        replace(
            FormAssociationReview(
                value_id=item.value_id,
                association_id=choice.association_id,
                grounding_digest=report.grounding_digest,
                review_id=f"{i + 500:032x}",
                reviewer_id="d" * 32,
                decision_digest="b" * 64,
                reviewer_confirmed=True,
            ),
            **changes,
        )
        for i, (item, choice) in enumerate(zip(report.values, choices))
    )


def test_geometry_proposal_still_requires_explicit_review_before_export():
    source = form()
    report = ground_form_fields(source, policy())
    item = report.values[0]
    assert item.state is GroundingState.REVIEW_REQUIRED
    assert item.associations[0].label_id == source.labels[0].evidence.node_id
    assert item.associations[0].geometry_score == 0.85
    assert item.associations[0].confidence == 0.8
    with pytest.raises(FormGroundingError):
        export_grounded_fields(source, policy(), ())
    with pytest.raises(FormGroundingError, match="explicit association review"):
        export_grounded_fields(
            source, policy(), reviews(report, reviewer_confirmed=False)
        )
    payload = json.loads(
        export_grounded_fields(source, policy(), reviews(report)).data_json
    )
    assert payload["fields"][0]["value"] == "Synthetic value"


@pytest.mark.parametrize(
    ("relation", "box"),
    [
        (LabelRelation.RIGHT, (230, 100, 300, 120)),
        (LabelRelation.LEFT, (50, 100, 80, 120)),
        (LabelRelation.BELOW, (100, 150, 200, 180)),
    ],
)
def test_explicit_relations_use_directional_gap_and_axis_alignment(relation, box):
    source = form(labels=(label(relation=relation),), values=(value(box=box),))
    result = ground_form_fields(source, policy()).values[0]
    assert result.state is GroundingState.REVIEW_REQUIRED
    assert result.associations[0].relation is relation
    assert result.associations[0].geometry_score > 0.7


@pytest.mark.parametrize(
    "box",
    [
        (50, 100, 80, 120),
        (230, 400, 300, 420),
        (800, 100, 850, 120),
        (150, 100, 250, 120),
    ],
)
def test_wrong_direction_row_large_gap_and_overlap_do_not_create_links(box):
    report = ground_form_fields(form(values=(value(box=box),)), policy())
    assert report.values[0].state is GroundingState.ABSTAINED
    assert report.values[0].associations == ()


def test_form_associations_do_not_depend_on_ocr_input_order():
    labels = (label(), label(4, (100, 300, 200, 320)))
    values = (value(), value(5, (230, 300, 300, 320)))
    first = ground_form_fields(form(labels=labels, values=values), policy())
    second = ground_form_fields(
        form(labels=tuple(reversed(labels)), values=tuple(reversed(values))), policy()
    )
    mapping = lambda report: {
        item.value_id: item.associations for item in report.values
    }
    assert mapping(first) == mapping(second)


def test_repeated_labels_remain_inside_their_declared_sections():
    sections = (
        FormSection(node(1, (0, 0, 1000, 200))),
        FormSection(node(6, (0, 250, 1000, 450))),
    )
    labels = (label(), label(4, (100, 300, 200, 320), section_id=6))
    values = (value(), value(5, (230, 300, 300, 320)))
    source = form(sections=sections, labels=labels, values=values)
    report = ground_form_fields(source, policy())
    assert [item.associations[0].section_id for item in report.values] == [
        f"{i:032x}" for i in (1, 6)
    ]
    assert all(len(item.associations) == 1 for item in report.values)


def test_column_distance_prevents_linking_to_an_earlier_column():
    source = form(
        labels=(label(), label(4, (600, 100, 700, 120))),
        values=(value(), value(5, (730, 100, 800, 120))),
    )
    report = ground_form_fields(source, policy())
    assert [item.associations[0].label_id for item in report.values] == [
        f"{i:032x}" for i in (2, 4)
    ]
    assert all(len(item.associations) == 1 for item in report.values)


def test_same_page_index_in_another_source_cannot_supply_a_label():
    source = form(
        sections=(
            FormSection(node(1, (0, 0, 1000, 1000))),
            FormSection(node(6, (0, 0, 1000, 1000), source="e" * 64)),
        ),
        values=(
            replace(value(), evidence=node(3, (230, 100, 300, 120), source="e" * 64)),
        ),
    )
    result = ground_form_fields(source, policy()).values[0]
    assert result.state is GroundingState.ABSTAINED
    assert result.associations == ()


def test_value_crossing_section_boundary_is_not_assigned_to_nearest_section():
    source = form(sections=(FormSection(node(1, (0, 0, 220, 200))),))
    result = ground_form_fields(source, policy()).values[0]
    assert "section_missing" in result.reasons
    assert result.associations == ()


def test_overlapping_section_membership_abstains_and_preserves_both_associations():
    source = form(
        sections=(
            FormSection(node(1, (0, 0, 1000, 1000))),
            FormSection(node(6, (50, 50, 500, 500))),
        ),
        labels=(label(), label(4, section_id=6)),
    )
    report = ground_form_fields(source, policy())
    assert report.values[0].state is GroundingState.ABSTAINED
    assert "section_ambiguous" in report.values[0].reasons
    assert len(report.values[0].associations) == 2


def test_ties_are_not_broken_by_id_but_reviewer_can_select_a_listed_alternative():
    source = form(labels=(label(), label(4)))
    report = ground_form_fields(source, policy())
    item = report.values[0]
    assert item.state is GroundingState.ABSTAINED
    assert "association_ambiguous" in item.reasons
    assert len(item.associations) == 2
    chosen = item.associations[1]
    exported = export_grounded_fields(source, policy(), reviews(report, (chosen,)))
    assert (
        json.loads(exported.data_json)["fields"][0]["association"]["association_id"]
        == chosen.association_id
    )


def test_competing_values_require_distinct_reviewed_fields():
    source = form(
        labels=(label(box=(100, 100, 180, 120)), label(4, (120, 100, 200, 120))),
        values=(value(box=(220, 100, 240, 120)), value(5, (245, 100, 265, 120))),
    )
    report = ground_form_fields(source, policy())
    assert all("field_competition" in item.reasons for item in report.values)
    with pytest.raises(FormGroundingError, match="conflicting"):
        export_grounded_fields(source, policy(), reviews(report))
    choices = [
        next(link for link in item.associations if link.label_id == f"{label_id:032x}")
        for item, label_id in zip(report.values, (2, 4))
    ]
    assert (
        len(
            json.loads(
                export_grounded_fields(
                    source, policy(), reviews(report, choices)
                ).data_json
            )["fields"]
        )
        == 2
    )


@pytest.mark.parametrize("kind", [ValueKind.CHECKBOX, ValueKind.RADIO])
@pytest.mark.parametrize("state", list(ControlState))
def test_control_states_are_explicit_and_unknown_is_never_unchecked(kind, state):
    field_label = replace(
        label(box=(250, 100, 350, 120), relation=LabelRelation.LEFT), accepts=(kind,)
    )
    control = FormValue(
        evidence=node(3, (220, 100, 240, 120)), kind=kind, control_state=state
    )
    source = form(labels=(field_label,), values=(control,))
    report = ground_form_fields(source, policy())
    if state is ControlState.UNKNOWN:
        assert report.values[0].state is GroundingState.ABSTAINED
        with pytest.raises(FormGroundingError, match="control state"):
            export_grounded_fields(source, policy(), reviews(report))
    else:
        payload = json.loads(
            export_grounded_fields(source, policy(), reviews(report)).data_json
        )
        assert payload["fields"][0]["value"] is (state is ControlState.CHECKED)


def test_value_kind_mismatch_cannot_become_a_candidate():
    checkbox = FormValue(
        evidence=value().evidence,
        kind=ValueKind.CHECKBOX,
        control_state=ControlState.CHECKED,
    )
    report = ground_form_fields(form(values=(checkbox,)), policy())
    assert report.values[0].associations == ()


def test_unknown_recognition_confidence_is_separate_from_geometric_score():
    source = form(labels=(replace(label(), confidence=Confidence()),))
    link = ground_form_fields(source, policy()).values[0].associations[0]
    assert link.geometry_score == 0.85
    assert link.confidence is None


def test_pixel_and_point_scale_changes_preserve_geometric_associations():
    original = form()

    def scaled(evidence):
        return replace(
            evidence,
            geometry=PageGeometry(
                page=0, width=100, height=100, unit=SpatialUnit.POINT
            ),
            region=tuple(value / 10 for value in evidence.region),
        )

    smaller = form(
        sections=(FormSection(scaled(original.sections[0].evidence)),),
        labels=(
            replace(original.labels[0], evidence=scaled(original.labels[0].evidence)),
        ),
        values=(
            replace(original.values[0], evidence=scaled(original.values[0].evidence)),
        ),
    )
    first, second = (
        ground_form_fields(original, policy()),
        ground_form_fields(smaller, policy()),
    )
    assert first.values[0].associations == second.values[0].associations
    assert first.grounding_digest != second.grounding_digest


@pytest.mark.parametrize("change", ["value", "label", "bounds", "policy", "section"])
def test_changed_form_or_policy_invalidates_association_approval(change):
    source, selected_policy = form(), policy()
    decisions = reviews(ground_form_fields(source, selected_policy))
    if change == "value":
        source = replace(
            source, values=(replace(source.values[0], text="Changed synthetic value"),)
        )
    elif change == "label":
        source = replace(
            source, labels=(replace(source.labels[0], text="Changed synthetic label"),)
        )
    elif change == "bounds":
        source = replace(
            source,
            values=(replace(source.values[0], evidence=node(3, (240, 100, 300, 120))),),
        )
    elif change == "policy":
        selected_policy = policy(ambiguity_margin=0.1)
    else:
        source = replace(source, sections=(FormSection(node(1, (0, 0, 900, 900))),))
    with pytest.raises(FormGroundingError, match="current explicit"):
        export_grounded_fields(source, selected_policy, decisions)


def handwriting_form():
    evidence = value().evidence
    candidate = HandwritingCandidate(
        candidate_id="f" * 32,
        evidence=evidence,
        alternatives=(
            TranscriptionAlternative(
                alternative_id="1" * 32, text="0.5 mg", confidence=Confidence(score=0.7)
            ),
            TranscriptionAlternative(
                alternative_id="2" * 32, text="5 mg", confidence=Confidence(score=0.95)
            ),
        ),
    )
    source = form(
        values=(
            FormValue(
                evidence=evidence, kind=ValueKind.HANDWRITING, handwriting=candidate
            ),
        )
    )
    decision = HandwritingReview(
        review_id="e" * 32,
        reviewer_id="d" * 32,
        decision_digest="c" * 64,
        candidate_id=candidate.candidate_id,
        candidate_digest=candidate.digest,
        action=ReviewAction.SELECT,
        alternative_id="1" * 32,
        reviewer_confirmed=True,
    )
    return source, decision


def test_handwritten_field_requires_both_exact_transcription_and_association_review():
    source, transcription = handwriting_form()
    report = ground_form_fields(source, policy())
    with pytest.raises(FormGroundingError):
        export_grounded_fields(source, policy(), reviews(report))
    with pytest.raises(FormGroundingError, match="does not bind"):
        export_grounded_fields(
            source, policy(), reviews(report), handwriting_reviews=(transcription,)
        )
    choices = reviews(report, handwriting_review_digest=transcription.digest)
    exported = export_grounded_fields(
        source, policy(), choices, handwriting_reviews=(transcription,)
    )
    field = json.loads(exported.data_json)["fields"][0]
    assert field["value"] == "0.5 mg"
    assert field["handwriting_review"]["confidence"] == 0.7
    assert field["handwriting_review"]["evidence"]["region"] == [230, 100, 300, 120]
    changed = replace(transcription, alternative_id="2" * 32)
    with pytest.raises(FormGroundingError, match="does not bind"):
        export_grounded_fields(
            source, policy(), choices, handwriting_reviews=(changed,)
        )


@pytest.mark.parametrize("defect", ["stale", "unconfirmed", "rejected"])
def test_invalid_handwriting_decision_stays_blocked_even_with_form_approval(defect):
    source, decision = handwriting_form()
    if defect == "stale":
        decision = replace(decision, candidate_digest="a" * 64)
    elif defect == "unconfirmed":
        decision = replace(decision, reviewer_confirmed=False)
    else:
        decision = replace(decision, action=ReviewAction.REJECT, alternative_id=None)
    choices = reviews(
        ground_form_fields(source, policy()), handwriting_review_digest=decision.digest
    )
    with pytest.raises(
        FormGroundingError, match="accepted current handwriting"
    ) as raised:
        export_grounded_fields(
            source, policy(), choices, handwriting_reviews=(decision,)
        )
    assert raised.value.__context__ is None


def test_default_review_packet_is_blocked_and_references_all_source_evidence():
    report = ground_form_fields(form(), policy())
    packet = report.to_review_packet()
    assert packet.review_status == "blocked"
    assert {item.citation_id for item in packet.citations} == set(
        report.review_citation_map
    )
    assert set(report.review_citation_map.values()) == set(report.evidence)
    mapping = json.loads(report.to_json())["review_citation_ids"]
    assert set(mapping.values()) == {node.node_id for node in report.evidence}
    assert "Synthetic label" not in packet.to_json()
    assert "Synthetic value" not in packet.to_json()
    assert not packet.to_dict()["privacy"]["protected_text_available"]


def test_protected_data_and_all_original_boxes_survive_only_in_export_payload():
    source = form()
    report = ground_form_fields(source, policy())
    exported = export_grounded_fields(source, policy(), reviews(report))
    row = json.loads(exported.data_json)["fields"][0]
    assert row["label"] == "Synthetic label"
    assert row["value"] == "Synthetic value"
    assert [entry["region"] for entry in row["evidence"]] == [
        [0, 0, 1000, 1000],
        [100, 100, 200, 120],
        [230, 100, 300, 120],
    ]
    diagnostics = (
        repr(source)
        + repr(source.labels[0])
        + repr(source.values[0])
        + repr(exported)
        + report.to_json()
        + exported.to_report_json()
    )
    assert "Synthetic label" not in diagnostics and "Synthetic value" not in diagnostics


def test_duplicate_or_unknown_choices_and_partial_batches_cannot_export():
    source = form(
        labels=(label(), label(4, (100, 300, 200, 320))),
        values=(value(), value(5, (230, 300, 300, 320))),
    )
    report = ground_form_fields(source, policy())
    decisions = reviews(report)
    for invalid in (
        decisions[:1],
        (decisions[0], decisions[0]),
        (decisions[0], replace(decisions[1], reviewer_confirmed=False)),
        (decisions[0], replace(decisions[1], association_id="f" * 32)),
        (decisions[0], replace(decisions[1], review_id=decisions[0].review_id)),
    ):
        with pytest.raises(FormGroundingError):
            export_grounded_fields(source, policy(), invalid)


def test_resource_and_malformed_input_bounds_fail_closed():
    with pytest.raises(FormGroundingError):
        form(labels=tuple(label(i + 100) for i in range(129)))
    with pytest.raises(FormGroundingError):
        form(labels=tuple(label(i + 100, text="x" * 4096) for i in range(65)))
    with pytest.raises(FormGroundingError):
        form(values=(value(), value()))
    with pytest.raises(FormGroundingError):
        form(labels=(label(section_id=100),))
    with pytest.raises(FormGroundingError):
        policy(maximum_gap=float("nan"))
    with pytest.raises(FormGroundingError):
        policy(maximum_gap=0)
    with pytest.raises(FormGroundingError):
        form(
            labels=(
                replace(
                    label(),
                    evidence=replace(
                        label().evidence,
                        geometry=PageGeometry(page=0, width=500, height=500),
                    ),
                ),
            )
        )


def test_tiny_source_extents_fail_with_controlled_normalization_error():
    source = form(
        labels=(label(box=(0, 0, 1e-323, 1e-323)),),
        values=(value(box=(2e-323, 0, 3e-323, 1e-323)),),
    )
    with pytest.raises(FormGroundingError, match="cannot be normalized"):
        ground_form_fields(source, policy())
