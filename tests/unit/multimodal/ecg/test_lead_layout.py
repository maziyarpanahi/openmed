"""Synthetic structural lead and display/timing mismatch controls."""

from dataclasses import replace

import pytest

from openmed.multimodal.ecg.lead_layout import (
    STANDARD_LEADS,
    AcquisitionKind,
    LeadFindingCode,
    LeadLayout,
    LeadLayoutError,
    LeadObservation,
    LeadSlot,
    SlotRole,
    assess_lead_layout,
)


def observations(layout, *, image=False):
    return tuple(
        LeadObservation(
            label=slot.lead,
            row=slot.row,
            column=slot.column,
            start_seconds=slot.timing_group * 2.5,
            duration_seconds=2.5,
            sample_rate_hz=None if image else 500,
            sample_count=None if image else 1250,
        )
        for slot in layout.slots
    )


def assess(items, layout=None, **kwargs):
    return assess_lead_layout(
        tuple(items),
        layout or LeadLayout.digital(),
        source_digest="a" * 64,
        acquisition=kwargs.pop("acquisition", AcquisitionKind.DIGITAL),
        **kwargs,
    )


def codes(report):
    return {finding.code for finding in report.findings}


def test_canonical_digital_leads_are_structurally_aligned():
    layout = LeadLayout.digital()
    report = assess(observations(layout), layout)
    assert not report.requires_review
    assert report.expected_leads == STANDARD_LEADS
    assert '"requires_review":false' in report.to_json()


def test_known_case_and_whitespace_variants_are_normalized():
    layout = LeadLayout.digital(("I", "aVR"))
    items = observations(layout)
    report = assess(
        (replace(items[0], label=" i "), replace(items[1], label="AVR")), layout
    )
    assert not report.requires_review


def test_swapped_labels_are_found_before_channel_reordering():
    layout = LeadLayout.digital()
    items = list(observations(layout))
    items[0], items[1] = replace(items[0], label="II"), replace(items[1], label="I")
    report = assess(items, layout)
    assert codes(report) == {LeadFindingCode.WRONG_LEAD}
    assert {f.channels for f in report.findings} == {(0,), (1,)}


def test_duplicate_lead_is_not_mistaken_for_a_complete_expected_set():
    items = list(observations(LeadLayout.digital()))
    items[1] = replace(items[1], label="I")
    assert codes(assess(items)) == {
        LeadFindingCode.WRONG_LEAD,
        LeadFindingCode.DUPLICATE_LEAD,
    }


def test_absent_and_duplicate_positions_require_review():
    items = observations(LeadLayout.digital())
    assert LeadFindingCode.MISSING_SLOT in codes(assess(items[:-1]))
    assert LeadFindingCode.DUPLICATE_SLOT in codes(assess(items + items[:1]))
    assert LeadFindingCode.MISSING_SLOT in codes(assess(()))


def test_undeclared_positions_do_not_count_as_expected_leads():
    items = list(observations(LeadLayout.digital()))
    items[0] = replace(items[0], row=20)
    assert codes(assess(items)) == {
        LeadFindingCode.MISSING_SLOT,
        LeadFindingCode.UNDECLARED_SLOT,
    }


def test_unknown_label_is_never_echoed_in_report_or_repr(caplog):
    item = LeadObservation(
        label="synthetic-private-label", row=0, start_seconds=0, duration_seconds=1
    )
    report = assess((item,), LeadLayout.digital(("II",)))
    assert LeadFindingCode.UNKNOWN_LEAD in codes(report)
    assert "synthetic-private-label" not in report.to_json()
    assert "synthetic-private-label" not in repr(item)
    assert caplog.records == []


@pytest.mark.parametrize(
    "change,expected",
    [
        (dict(start_seconds=0.1), LeadFindingCode.TIMING_MISMATCH),
        (dict(duration_seconds=3), LeadFindingCode.TIMING_MISMATCH),
        (dict(sample_rate_hz=250, sample_count=625), LeadFindingCode.SAMPLING_MISMATCH),
        (dict(sample_count=1200), LeadFindingCode.SAMPLING_MISMATCH),
        (
            dict(start_seconds=None, duration_seconds=None),
            LeadFindingCode.TIMING_UNKNOWN,
        ),
        (
            dict(sample_rate_hz=None, sample_count=None),
            LeadFindingCode.SAMPLING_UNKNOWN,
        ),
    ],
)
def test_digital_timing_and_sampling_are_not_inferred(change, expected):
    items = list(observations(LeadLayout.digital()))
    items[-1] = replace(items[-1], **change)
    assert expected in codes(assess(items))


def test_three_by_four_uses_declared_column_time_groups():
    layout = LeadLayout.three_by_four()
    items = observations(layout, image=True)
    report = assess(
        items,
        layout,
        acquisition=AcquisitionKind.IMAGE_DERIVED,
        image_labels_confirmed=True,
    )
    assert not report.requires_review
    # Same column, different row: a shifted window is an alignment defect.
    changed = list(items)
    changed[4] = replace(changed[4], start_seconds=0.1)
    assert LeadFindingCode.TIMING_MISMATCH in codes(
        assess(
            changed,
            layout,
            acquisition=AcquisitionKind.IMAGE_DERIVED,
            image_labels_confirmed=True,
        )
    )


def test_image_labels_remain_unverified_until_explicitly_confirmed():
    layout = LeadLayout.three_by_four()
    report = assess(
        observations(layout, image=True),
        layout,
        acquisition=AcquisitionKind.IMAGE_DERIVED,
    )
    assert codes(report) == {LeadFindingCode.LABEL_REVIEW}


def test_explicit_rhythm_strip_is_not_an_accidental_duplicate():
    layout = LeadLayout(
        rows=2,
        columns=1,
        expected_leads=("II",),
        slots=(
            LeadSlot(lead="II", row=0, column=0),
            LeadSlot(lead="II", row=1, column=0, timing_group=1, role=SlotRole.RHYTHM),
        ),
    )
    report = assess(observations(layout), layout)
    assert not report.requires_review


def test_expected_single_lead_does_not_require_all_twelve():
    layout = LeadLayout.digital(("II",))
    assert not assess(observations(layout), layout).requires_review


@pytest.mark.parametrize(
    "change",
    [
        dict(rows=0),
        dict(columns=True),
        dict(expected_leads=("I", "I")),
        dict(expected_leads=("synthetic-private-label",)),
        dict(slots=()),
        dict(rows=1),
    ],
)
def test_invalid_layout_declarations_fail_closed(change):
    with pytest.raises(LeadLayoutError, match="^invalid ECG lead layout$"):
        replace(LeadLayout.digital(), **change)


@pytest.mark.parametrize(
    "change",
    [
        dict(start_seconds=float("nan"), duration_seconds=1),
        dict(sample_rate_hz=float("inf"), sample_count=1),
        dict(duration_seconds=0),
        dict(start_seconds=True),
        dict(row=-1),
        dict(row=10**1000),
        dict(label="x" * 65),
    ],
)
def test_invalid_observations_fail_without_values(change):
    original = observations(LeadLayout.digital())[0]
    with pytest.raises(LeadLayoutError) as caught:
        replace(original, **change)
    assert str(caught.value) == "invalid ECG lead layout"


def test_observation_bound_is_checked_before_analysis():
    original = observations(LeadLayout.digital())[0]
    with pytest.raises(LeadLayoutError):
        assess((original,) * 65)


def test_findings_are_deterministic_and_non_diagnostic():
    layout = LeadLayout.digital()
    items = observations(layout)[:-1]
    assert assess(items, layout).to_json() == assess(items, layout).to_json()
    assert "Non-diagnostic" in assess(items, layout).to_json()
