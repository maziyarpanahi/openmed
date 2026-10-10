"""Synthetic non-ECG acquisition and privacy negative controls."""

import dataclasses
import json
from pathlib import Path

import pytest

from openmed.multimodal.physiological_waveforms import (
    CHANNEL_CONSTRAINTS,
    ChannelKind,
    ChannelQuality,
    QualityState,
    RecordingQuality,
    WaveformChannel,
    WaveformContractError,
    WaveformProvenance,
    evaluate_recording,
)

FIXTURES = json.loads(
    (
        Path(__file__).parents[2] / "fixtures/multimodal/physiological_waveforms.json"
    ).read_text()
)


def channel(row, **changes):
    fields = {
        key: value
        for key, value in row.items()
        if key not in {"fixture", "expected", "source_sha256"}
    }
    fields["provenance"] = WaveformProvenance(row["source_sha256"])
    return WaveformChannel(**(fields | changes))


@pytest.mark.parametrize(
    "row", FIXTURES, ids=lambda row: row["kind"] + ":" + row["fixture"]
)
def test_shared_quality_fixtures(row):
    item = channel(row)
    report = evaluate_recording((item,))
    assert report.channels[0].to_dict() == row["expected"]
    assert report.to_json() == evaluate_recording((item,)).to_json()
    assert "not vital-sign" in item.non_diagnostic_notice
    assert report.non_diagnostic_notice == item.non_diagnostic_notice
    with pytest.raises(WaveformContractError, match="^reviewer_confirmation_required$"):
        report.require_reviewer_confirmation()
    report.require_reviewer_confirmation(confirmed=True)


@pytest.mark.parametrize("kind", list(ChannelKind))
def test_units_ranges_and_rate_boundaries(kind):
    row = next(
        row for row in FIXTURES if row["kind"] == kind and row["fixture"] == "pass"
    )
    limits = CHANNEL_CONSTRAINTS[kind]
    for rate in (limits.minimum_rate_hz, limits.maximum_rate_hz):
        channel(row, sample_rate_hz=rate, offsets_seconds=[i / rate for i in range(40)])
    for rate in (
        0,
        -1,
        limits.minimum_rate_hz / 2,
        limits.maximum_rate_hz + 1,
        float("nan"),
        float("inf"),
        True,
        10**1000,
    ):
        with pytest.raises(WaveformContractError, match="^invalid_sampling_rate$"):
            channel(row, sample_rate_hz=rate)
    with pytest.raises(WaveformContractError, match="^invalid_unit$"):
        channel(row, unit="mV")
    for low, high in (
        (limits.minimum - 1, limits.maximum),
        (limits.minimum, limits.maximum + 1),
        (1, 1),
        (float("nan"), 1),
        (True, 1),
    ):
        with pytest.raises(WaveformContractError, match="^invalid_acquisition_range$"):
            channel(row, acquisition_minimum=low, acquisition_maximum=high)
    for sample in (
        limits.minimum - 1,
        limits.maximum + 1,
        float("nan"),
        float("inf"),
        True,
        "synthetic-private-payload",
    ):
        with pytest.raises(WaveformContractError, match="^invalid_sample$"):
            channel(row, samples=[sample] + row["samples"][1:])


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"kind": "synthetic-private-payload"}, "unknown_kind"),
        ({"kind": "ecg"}, "unknown_kind"),
        ({"kind": []}, "unknown_kind"),
        ({"unit": "synthetic-private-payload"}, "invalid_unit"),
        ({"provenance": "synthetic-private-payload"}, "invalid_provenance"),
        ({"samples": []}, "invalid_sample_shape"),
        ({"samples": [0.5]}, "invalid_sample_shape"),
        ({"offsets_seconds": [0]}, "invalid_sample_shape"),
        ({"offsets_seconds": [0] * 40}, "invalid_timing"),
        ({"offsets_seconds": [i / 20 for i in range(40)]}, "invalid_timing"),
        (
            {"offsets_seconds": [i / 10 + (1 if i > 20 else 0) for i in range(40)]},
            "invalid_timing",
        ),
        ({"offsets_seconds": [-1] + [i / 10 for i in range(1, 40)]}, "invalid_timing"),
        ({"offsets_seconds": [float("inf")] * 40}, "invalid_timing"),
        ({"offsets_seconds": [True] * 40}, "invalid_timing"),
    ],
)
def test_controlled_rejections(changes, code):
    with pytest.raises(WaveformContractError) as error:
        channel(FIXTURES[0], **changes)
    assert str(error.value) == code
    assert error.value.__cause__ is None


def test_input_snapshot_and_protected_representations():
    row = FIXTURES[0]
    samples = row["samples"].copy()
    offsets = row["offsets_seconds"].copy()
    item = channel(row, samples=samples, offsets_seconds=offsets)
    samples[0] = 1
    offsets[0] = 99
    assert item.samples[0] != 1
    assert item.offsets_seconds[0] != 99
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.unit = "mV"
    assert repr(item) == "WaveformChannel(kind=ppg, sample_count=40)"
    assert repr(item.provenance) == "WaveformProvenance()"
    payload = evaluate_recording((item,)).to_json()
    assert row["source_sha256"] not in payload
    assert set(json.loads(payload)) == {"channel_count", "codes", "channels"}
    assert set(json.loads(payload)["channels"][0]) == {
        "kind",
        "state",
        "sample_count",
        "dropout_count",
        "saturation_count",
        "flatline_count",
        "motion_proxy_count",
        "codes",
    }


@pytest.mark.parametrize(
    "digest", ["synthetic-private-payload", "A" * 64, "a" * 63, None, []]
)
def test_provenance_rejects_free_text(digest):
    with pytest.raises(WaveformContractError, match="^invalid_provenance$"):
        WaveformProvenance(digest)


def test_reports_refuse_uncontrolled_content():
    valid = evaluate_recording((channel(FIXTURES[0]),)).channels[0]
    for changes in (
        {"codes": ("synthetic-private-payload",)},
        {"sample_count": "synthetic-private-payload"},
        {"kind": "synthetic-private-payload"},
        {"state": "alarm"},
    ):
        with pytest.raises(WaveformContractError, match="^invalid_quality_report$"):
            dataclasses.replace(valid, **changes)
    with pytest.raises(WaveformContractError, match="^invalid_quality_report$"):
        RecordingQuality(("synthetic-private-payload",))
    assert {field.name for field in dataclasses.fields(ChannelQuality)} == {
        "kind",
        "state",
        "sample_count",
        "dropout_count",
        "saturation_count",
        "flatline_count",
        "motion_proxy_count",
        "codes",
    }
    assert set(QualityState) == {
        QualityState.PASS,
        QualityState.LIMITED_USE,
        QualityState.REVIEW,
    }


def test_recording_shape_and_resource_limits():
    item = channel(FIXTURES[0])
    for channels in ((), (item,) * 65, "synthetic-private-payload"):
        with pytest.raises(WaveformContractError, match="^invalid_recording$"):
            evaluate_recording(channels)
    with pytest.raises(WaveformContractError, match="^invalid_channel$"):
        evaluate_recording((item, "synthetic-private-payload"))
    large = channel(
        FIXTURES[0],
        samples=[0.5] * 20_000,
        offsets_seconds=[i / 10 for i in range(20_000)],
    )
    with pytest.raises(WaveformContractError, match="^resource_limit$"):
        evaluate_recording((large,) * 51)


def test_dropout_breaks_constant_runs_and_never_invents_motion():
    values = [0.5] * 10 + [None] + [0.5] * 9 + [None] + [0.5] * 9
    item = channel(
        FIXTURES[0],
        samples=values,
        offsets_seconds=[i / 10 for i in range(len(values))],
    )
    result = evaluate_recording((item,)).channels[0]
    assert result.flatline_count == 0
    assert result.motion_proxy_count == 0
    assert result.state == QualityState.LIMITED_USE
    for confirmed in (1, "yes", None):
        with pytest.raises(WaveformContractError):
            evaluate_recording((item,)).require_reviewer_confirmation(
                confirmed=confirmed
            )
