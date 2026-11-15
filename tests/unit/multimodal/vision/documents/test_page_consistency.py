"""Synthetic packet identity, missing-page, layout and provenance negative controls."""

import json
import socket
from dataclasses import replace

import pytest

from openmed.multimodal.evidence.fused_graph import EvidenceNode, NodeKind, PageGeometry
from openmed.multimodal.privacy.mask_lineage import (
    CoordinateSpace,
    Rectangle,
    TransformStep,
    propagate_masks,
)
from openmed.multimodal.vision.documents.page_consistency import (
    ConsistencyCode as Code,
)
from openmed.multimodal.vision.documents.page_consistency import (
    ConsistencyPolicy,
    GroupBoundary,
    IdentityKind,
    PageConsistencyError,
    PageFinding,
    anchor_from_span,
    assess_page_consistency,
    assess_pdf_page_consistency,
    capture_page,
    continuation_from_span,
    identity_from_span,
    numbering_from_span,
    pdf_page_sources,
    require_consistent_group,
)

KEY = b"synthetic-packet-key-not-a-secret-2805"
NS = "1" * 32
POLICY = ConsistencyPolicy(identity_namespace=NS)


def source_page(index):
    return EvidenceNode(
        node_id=f"{1000 + index * 100:032x}",
        kind=NodeKind.PAGE,
        source_digest="a" * 64,
        geometry=PageGeometry(page=index, width=100, height=100),
    )


def observe(
    index=0,
    *,
    patient="SYNTHETIC-ID-A",
    current=None,
    total=2,
    key=KEY,
    source=None,
    anchor="Synthetic form",
    anchor_x=10,
    incoming=None,
    outgoing=None,
    secondary=(),
    pixels=None,
):
    source = source or source_page(index)
    current = index + 1 if current is None else current
    entries = []
    if patient is not None:
        entries.append(("id", patient))
    entries += [("secondary", value) for _, value in secondary]
    if anchor is not None:
        entries.append(("anchor", anchor))
    if current is not False:
        entries.append(("number", f"{current}/{total}"))
    if incoming is not None:
        entries.append(("in", incoming))
    if outgoing is not None:
        entries.append(("out", outgoing))
    text = "|".join(value for _, value in entries)
    identities, anchors, numbers, ins, outs = [], [], [], [], []
    start, secondary_index = 0, 0
    for slot, (kind, value) in enumerate(entries):
        region = (anchor_x, 10, anchor_x + 10, 20) if kind == "anchor" else (1, 1, 5, 5)
        evidence = EvidenceNode(
            node_id=f"{1001 + index * 100 + slot:032x}",
            kind=NodeKind.OCR,
            source_digest=source.source_digest,
            geometry=source.geometry,
            region=region,
            span=(start, start + len(value)),
        )
        if kind in ("id", "secondary"):
            field = (
                IdentityKind.PATIENT_ID
                if kind == "id"
                else secondary[secondary_index][0]
            )
            if kind == "secondary":
                secondary_index += 1
            identities.append(
                identity_from_span(
                    text, evidence, key=key, kind=field, namespace_ref=NS
                )
            )
        elif kind == "anchor":
            anchors.append(anchor_from_span(text, evidence, key=key))
        elif kind == "number":
            numbers.append(numbering_from_span(text, evidence))
        elif kind == "in":
            ins.append(continuation_from_span(text, evidence, key=key))
        else:
            outs.append(continuation_from_span(text, evidence, key=key))
        start += len(value) + 1
    return capture_page(
        source,
        bytes([index + 1]) * 12 if pixels is None else pixels,
        (2, 2),
        identities=tuple(identities),
        anchors=tuple(anchors),
        numbering=numbers[0] if numbers else None,
        incoming=ins[0] if ins else None,
        outgoing=outs[0] if outs else None,
    )


def codes(report):
    return {finding.code for group in report.groups for finding in group.findings}


def test_coherent_actual_spans_and_pixels_form_one_complete_group():
    pages = (observe(), observe(1))
    report = assess_page_consistency(pages, policy=POLICY)
    assert report.whole_packet_consistent and not codes(report)
    assert report.groups[0].packet_indexes == (0, 1)
    assert require_consistent_group(pages, 0, policy=POLICY) == pages
    assert report.to_json() == assess_page_consistency(pages, policy=POLICY).to_json()
    assert (
        pages[0].identities[0].token.value_digest
        == pages[1].identities[0].token.value_digest
    )
    assert pages[0].pixels_digest != pages[1].pixels_digest


def test_mixed_patients_split_complete_groups_without_rewriting_source_indexes():
    pages = (observe(total=1), observe(1, patient="SYNTHETIC-ID-B", current=1, total=1))
    report = assess_page_consistency(pages, policy=POLICY)
    assert not report.whole_packet_consistent and not codes(report)
    assert [group.packet_indexes for group in report.groups] == [(0,), (1,)]
    assert report.groups[1].boundary is GroupBoundary.IDENTITY_CHANGE
    selected = require_consistent_group(pages, 1, policy=POLICY)
    assert selected[0] is pages[1] and selected[0].source.geometry.page == 1


def test_missing_identity_cannot_bridge_conflicting_patient_pages():
    pages = (
        observe(total=1),
        observe(1, patient=None, current=1, total=1),
        observe(2, patient="SYNTHETIC-ID-B", current=1, total=1),
    )
    report = assess_page_consistency(pages, policy=POLICY)
    assert len(report.groups) == 3
    assert report.groups[1].boundary is GroupBoundary.IDENTITY_UNCERTAIN
    assert Code.IDENTITY_UNKNOWN in codes(report)
    with pytest.raises(PageConsistencyError):
        require_consistent_group(pages, 1, policy=POLICY)


def test_complete_group_secondary_identity_conflicts_survive_missing_middle_values():
    pages = (
        observe(total=3, secondary=((IdentityKind.DATE_OF_BIRTH, "SYNTHETIC-DATE-A"),)),
        observe(1, total=3),
        observe(
            2, total=3, secondary=((IdentityKind.DATE_OF_BIRTH, "SYNTHETIC-DATE-B"),)
        ),
    )
    report = assess_page_consistency(pages, policy=POLICY)
    assert Code.IDENTITY_CONFLICT in codes(report)
    assert report.groups[0].quarantined
    assert report.groups[0].findings[0].page_refs == tuple(
        page.source.node_id for page in pages
    )


def test_multiple_identifiers_on_one_page_are_ambiguous():
    page = observe(total=1, secondary=((IdentityKind.PATIENT_ID, "SYNTHETIC-ID-B"),))
    report = assess_page_consistency((page,), policy=POLICY)
    assert {Code.IDENTITY_UNKNOWN, Code.IDENTITY_CONFLICT} <= codes(report)


def test_noncontiguous_identity_is_not_silently_reordered_or_joined():
    pages = (
        observe(total=1),
        observe(1, patient="SYNTHETIC-ID-B", current=1, total=1),
        observe(2, current=1, total=1),
    )
    report = assess_page_consistency(pages, policy=POLICY)
    assert [group.packet_indexes for group in report.groups] == [(0,), (1,), (2,)]
    assert report.groups[0].quarantined and report.groups[2].quarantined
    assert Code.NONCONTIGUOUS_IDENTITY in codes(report)


def test_identical_rendered_pixels_flag_every_occurrence_even_with_different_identifiers():
    pixels = bytes(range(12))
    pages = (
        observe(total=1, pixels=pixels),
        observe(1, patient="SYNTHETIC-ID-B", current=1, total=1, pixels=pixels),
    )
    report = assess_page_consistency(pages, policy=POLICY)
    assert all(group.quarantined for group in report.groups)
    assert all(
        Code.DUPLICATE_PAGE in {finding.code for finding in group.findings}
        for group in report.groups
    )


@pytest.mark.parametrize(
    "numbers,totals,expected",
    [
        ((1, 3), (3, 3), Code.NUMBER_SEQUENCE),
        ((2, 1), (2, 2), Code.NUMBER_SEQUENCE),
        ((1, 1), (2, 2), Code.NUMBER_SEQUENCE),
        ((1, 2), (2, 3), Code.NUMBER_TOTAL),
        ((2, 3), (3, 3), Code.INCOMPLETE_GROUP),
        ((1, 2), (3, 3), Code.INCOMPLETE_GROUP),
    ],
)
def test_missing_reversed_repeated_and_incomplete_numbering(numbers, totals, expected):
    pages = tuple(
        observe(i, current=number, total=total)
        for i, (number, total) in enumerate(zip(numbers, totals, strict=True))
    )
    assert expected in codes(assess_page_consistency(pages, policy=POLICY))


def test_unknown_counter_is_not_treated_as_complete_packet():
    assert Code.NUMBER_UNKNOWN in codes(
        assess_page_consistency((observe(current=False), observe(1)), policy=POLICY)
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"anchor": None}, Code.LAYOUT_UNKNOWN),
        ({"anchor": "Different synthetic form"}, Code.LAYOUT_CHANGED),
        ({"anchor_x": 20}, Code.LAYOUT_CHANGED),
    ],
)
def test_missing_changed_and_moved_template_anchors_quarantine(kwargs, expected):
    assert expected in codes(
        assess_page_consistency((observe(), observe(1, **kwargs)), policy=POLICY)
    )


def test_layout_comparison_rejects_accumulated_drift_beyond_pairwise_tolerance():
    pages = tuple(
        observe(index, total=3, anchor_x=10 + 1.5 * index) for index in range(3)
    )
    assert Code.LAYOUT_CHANGED in codes(assess_page_consistency(pages, policy=POLICY))
    assert assess_page_consistency(
        (observe(), observe(1, anchor_x=11)), policy=POLICY
    ).whole_packet_consistent


def test_matching_continuation_cues_need_both_endpoints_and_same_patient():
    good = (
        observe(outgoing="synthetic-section"),
        observe(1, incoming="synthetic-section"),
    )
    assert assess_page_consistency(good, policy=POLICY).whole_packet_consistent
    mismatch = (good[0], observe(1, incoming="different-section"))
    assert Code.CONTINUATION_CONFLICT in codes(
        assess_page_consistency(mismatch, policy=POLICY)
    )
    missing = (good[0], observe(1))
    assert Code.CONTINUATION_MISSING in codes(
        assess_page_consistency(missing, policy=POLICY)
    )
    cross = (
        good[0],
        observe(1, patient="SYNTHETIC-ID-B", incoming="synthetic-section"),
    )
    assert Code.CONTINUATION_CROSSES_IDENTITY in codes(
        assess_page_consistency(cross, policy=POLICY)
    )


@pytest.mark.parametrize("kwargs", [{"incoming": "orphan"}, {"outgoing": "missing"}])
def test_packet_boundary_continuation_is_incomplete(kwargs):
    assert Code.CONTINUATION_MISSING in codes(
        assess_page_consistency((observe(total=1, **kwargs),), policy=POLICY)
    )


def test_keys_namespaces_and_token_purposes_cannot_be_mixed_silently():
    first, second = observe(), observe(1, key=b"other-synthetic-packet-key-00000000")
    assert Code.KEY_SCOPE_MISMATCH in codes(
        assess_page_consistency((first, second), policy=POLICY)
    )
    assert Code.IDENTITY_UNKNOWN in codes(
        assess_page_consistency(
            (first,), policy=replace(POLICY, identity_namespace="2" * 32)
        )
    )
    with pytest.raises(PageConsistencyError):
        replace(first, anchors=(first.identities[0].token,))
    with pytest.raises(PageConsistencyError):
        replace(first, incoming=first.anchors[0])


def test_source_page_geometry_and_duplicate_source_locations_are_validated():
    first, second = observe(), observe(1)
    with pytest.raises(PageConsistencyError):
        replace(first, identities=second.identities)
    with pytest.raises(PageConsistencyError):
        assess_page_consistency((first, first), policy=POLICY)
    alias = replace(first, source=replace(first.source, node_id="f" * 32))
    with pytest.raises(PageConsistencyError):
        assess_page_consistency((first, alias), policy=POLICY)


def test_mask_coordinates_survive_group_selection_and_rejected_coverage_stays_held():
    page = observe(total=1)
    source = CoordinateSpace(
        space_ref="a" * 32,
        source_digest=page.source.source_digest,
        geometry=page.source.geometry,
    )
    target = replace(
        source, space_ref="b" * 32, geometry=replace(source.geometry, width=2, height=2)
    )
    mask = EvidenceNode(
        node_id="c" * 32,
        kind=NodeKind.PIXELS,
        source_digest=source.source_digest,
        geometry=source.geometry,
        region=(20, 20, 40, 40),
    )
    lineage = propagate_masks(
        source, (mask,), (TransformStep.scale(source, target, resampling_margin=0),)
    )
    bound = replace(page, mask_lineage=lineage)
    (selected,) = require_consistent_group((bound,), 0, policy=POLICY)
    assert selected.mask_lineage is lineage
    assert selected.mask_lineage.source_nodes[0] is mask
    malformed_space = replace(source)
    object.__setattr__(malformed_space, "space_ref", "SYNTHETIC-RAW-MASK-REFERENCE")
    with pytest.raises(PageConsistencyError) as caught:
        capture_page(
            page.source,
            bytes(12),
            (2, 2),
            mask_lineage=replace(lineage, source=malformed_space),
        )
    assert caught.value.__context__ is None
    cropped = replace(
        source,
        space_ref="d" * 32,
        geometry=replace(source.geometry, width=10, height=10),
    )
    rejected = propagate_masks(
        source, (mask,), (TransformStep.crop(source, cropped, Rectangle(0, 0, 10, 10)),)
    )
    held = replace(page, mask_lineage=rejected)
    assert Code.MASK_REJECTED in codes(assess_page_consistency((held,), policy=POLICY))
    with pytest.raises(PageConsistencyError):
        replace(page, mask_lineage=replace(rejected, failure=None))


def synthetic_pdf():
    return b"""%PDF-1.7
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 /MediaBox [0 0 100 100] >> endobj
3 0 obj << /Type /Page /Parent 2 0 R >> endobj
4 0 obj << /Type /Page /Parent 2 0 R >> endobj
trailer << /Root 1 0 R >>
%%EOF
"""


def test_real_pdf_page_tree_is_bound_to_complete_ordered_observations():
    source = synthetic_pdf()
    nodes = pdf_page_sources(source)
    pages = tuple(observe(index, source=node) for index, node in enumerate(nodes))
    assert assess_pdf_page_consistency(
        source, pages, policy=POLICY
    ).whole_packet_consistent
    for bad_source, bad_pages in (
        (source + b"\n", pages),
        (source, pages[:1]),
        (source, pages[::-1]),
    ):
        with pytest.raises(PageConsistencyError):
            assess_pdf_page_consistency(bad_source, bad_pages, policy=POLICY)
    with pytest.raises(PageConsistencyError):
        pdf_page_sources(source, max_pages=1)
    with pytest.raises(PageConsistencyError):
        pdf_page_sources(source.replace(b"/Count 2", b"/Count 3"))


def test_protected_values_keys_logs_and_exception_context_are_not_retained(
    caplog, monkeypatch
):
    def network(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket, "create_connection", network)
    sentinel = "SYNTHETIC-PROTECTED-TEXT-2805"
    pages = (observe(patient=sentinel), observe(1, patient=sentinel))
    report = assess_page_consistency(pages, policy=POLICY)
    serialized = report.to_json() + repr(pages) + repr(report) + caplog.text
    assert sentinel not in serialized and KEY.decode() not in serialized
    assert json.loads(report.to_json())["review_required"] is True
    bad = replace(pages[0].identities[0].token.evidence, span=(0, len(sentinel) + 1))
    with pytest.raises(PageConsistencyError) as caught:
        identity_from_span(
            sentinel, bad, key=KEY, kind=IdentityKind.PATIENT_ID, namespace_ref=NS
        )
    assert caught.value.__context__ is None
    assert str(caught.value) == "page_consistency_rejected"


def test_invalid_budgets_pixels_and_policy_do_not_return_partial_groups():
    with pytest.raises(PageConsistencyError):
        capture_page(source_page(0), b"short", (2, 2))
    with pytest.raises(PageConsistencyError):
        capture_page(source_page(0), b"", (65535, 65535))
    with pytest.raises(PageConsistencyError):
        ConsistencyPolicy(identity_namespace=NS, anchor_tolerance=float("nan"))
    with pytest.raises(PageConsistencyError):
        assess_page_consistency(
            (observe(), observe(1)), policy=replace(POLICY, max_pages=1)
        )
    with pytest.raises(PageConsistencyError):
        assess_page_consistency((), policy=POLICY)


def test_guard_reassesses_observations_instead_of_accepting_an_edited_result():
    pages = (observe(), observe(1, anchor="different"))
    report = assess_page_consistency(pages, policy=POLICY)
    forged = replace(report, groups=(replace(report.groups[0], findings=()),))
    assert forged.whole_packet_consistent
    with pytest.raises(PageConsistencyError):
        require_consistent_group(pages, 0, policy=POLICY)


def test_report_constructors_and_serializers_reject_free_text_and_foreign_references():
    report = assess_page_consistency((observe(total=1),), policy=POLICY)
    with pytest.raises(PageConsistencyError):
        PageFinding("SYNTHETIC-RAW-TEXT", report.groups[0].page_refs)
    with pytest.raises(PageConsistencyError):
        replace(
            report.groups[0],
            findings=(PageFinding(Code.IDENTITY_UNKNOWN, ("f" * 32,)),),
        )
    with pytest.raises(PageConsistencyError):
        replace(report, groups=(replace(report.groups[0], packet_indexes=(1,)),))
    finding = PageFinding(Code.IDENTITY_UNKNOWN, report.groups[0].page_refs)
    held = replace(report, groups=(replace(report.groups[0], findings=(finding,)),))
    object.__setattr__(finding, "code", "SYNTHETIC-RAW-TEXT")
    with pytest.raises(PageConsistencyError):
        held.to_json()


def test_identity_capture_is_case_sensitive_and_key_scoped_with_conservative_normalization():
    first = observe(patient="SYNTHETIC-ID-A", total=1).identities[0].token
    padded = observe(patient=" SYNTHETIC-ID-A ", total=1).identities[0].token
    lower = observe(patient="synthetic-id-a", total=1).identities[0].token
    changed_key = (
        observe(
            patient="SYNTHETIC-ID-A",
            total=1,
            key=b"changed-synthetic-packet-key-000000",
        )
        .identities[0]
        .token
    )
    assert first.value_digest == padded.value_digest
    assert first.value_digest != lower.value_digest
    assert first.scope_digest != changed_key.scope_digest
    assert first.value_digest != changed_key.value_digest
