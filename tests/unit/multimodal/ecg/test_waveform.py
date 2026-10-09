"""Synthetic waveform vectors; no patient records, models, or network access."""

import json
import socket
import traceback
from dataclasses import FrozenInstanceError

import pytest

from openmed.multimodal.ecg.waveform import (
    ECG_SCHEMA_VERSION,
    NON_DIAGNOSTIC_NOTICE,
    EcgLead,
    EcgProvenance,
    EcgWaveform,
    WaveformValidationError,
)


def provenance():
    return EcgProvenance("array", "a" * 64)


def lead(name="II", samples=(0.0, 1.0, None), **kwargs):
    return EcgLead.from_samples(name, samples, unit="mV", **kwargs)


def waveform(**kwargs):
    fields = dict(leads=(lead(),), sample_rate_hz=250, provenance=provenance())
    fields.update(kwargs)
    return EcgWaveform(**fields)


@pytest.mark.parametrize(
    "unit,values",
    [
        ("V", (0, 0.001)),
        ("mV", (0, 1)),
        ("uV", (0, 1000)),
        ("µV", (0, 1000)),
        ("μV", (0, 1000)),
    ],
)
def test_physical_units(unit, values):
    result = EcgLead.from_samples(" ii ", values, unit=unit)
    assert result.lead_id == "II"
    assert result.samples_mv == (0.0, 1.0)
    assert result.valid_mask == (True, True)


def test_adc_calibration_and_mask():
    result = EcgLead.from_samples(
        "avr", (10, 210, None), unit="adc", gain_counts_per_mv=200, baseline_counts=10
    )
    assert result.lead_id == "aVR"
    assert result.samples_mv == (0, 1, None)
    assert result.valid_mask == (True, True, False)


def test_lead_order_regular_timing_and_determinism(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("waveform normalization attempted network access")

    monkeypatch.setattr(socket, "socket", no_network)
    first = waveform(
        leads=[lead("V2"), lead("I"), lead("aVF")],
        start_offset_seconds=1,
        sample_offsets_seconds=[1, 1.004, 1.008],
    )
    second = waveform(
        leads=[lead("aVF"), lead("V2"), lead("I")], start_offset_seconds=1
    )
    assert first == second
    assert tuple(item.lead_id for item in first.leads) == ("I", "aVF", "V2")
    assert first.sample_offsets_seconds == (1, 1.004, 1.008)
    assert first.to_json() == second.to_json()
    report = json.loads(first.to_json())
    assert report["schema_version"] == ECG_SCHEMA_VERSION
    assert report["notice"] == NON_DIAGNOSTIC_NOTICE
    assert report["review_required"] is True
    assert report["leads"][0] == {"lead_id": "I", "missing_count": 1}


def test_mask_and_samples_are_copied_without_interpolation():
    samples = [2, None, -3]
    mask = [True, False, True]
    result = EcgLead.from_samples("I", samples, unit="mV", valid_mask=mask)
    samples[0] = 99
    mask[0] = False
    assert result.samples_mv == (2, None, -3)
    assert result.valid_mask == (True, False, True)
    with pytest.raises(FrozenInstanceError):
        result.lead_id = "II"


@pytest.mark.parametrize(
    "samples,mask",
    [
        ((1, None), (True,)),
        ((1,), (1,)),
        ((None,), (True,)),
        ((1,), (False,)),
        ((float("nan"),), (False,)),
    ],
)
def test_ambiguous_missingness_rejected(samples, mask):
    with pytest.raises(WaveformValidationError):
        EcgLead.from_samples("I", samples, unit="mV", valid_mask=mask)


@pytest.mark.parametrize(
    "sample", [True, "synthetic-secret", float("inf"), float("nan"), 10**1000]
)
def test_invalid_numeric_samples_do_not_echo_values(sample):
    with pytest.raises(WaveformValidationError) as error:
        lead(samples=(sample,))
    assert "sample_invalid" in str(error.value)
    assert "synthetic-secret" not in str(error.value)
    assert NON_DIAGNOSTIC_NOTICE in str(error.value)
    assert error.value.__context__ is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unit": "unknown"},
        {"unit": "adc"},
        {"unit": "adc", "gain_counts_per_mv": 0, "baseline_counts": 0},
        {"unit": "adc", "gain_counts_per_mv": -2, "baseline_counts": 0},
        {"unit": "adc", "gain_counts_per_mv": 2},
        {"unit": "mV", "gain_counts_per_mv": 2},
        {"unit": "mV", "baseline_counts": 0},
    ],
)
def test_unit_and_gain_ambiguity(kwargs):
    with pytest.raises(WaveformValidationError):
        EcgLead.from_samples("I", (1,), **kwargs)


@pytest.mark.parametrize("values", [(), "synthetic-secret", (1,) * 1_000_001])
def test_sample_resource_and_shape_limits(values):
    with pytest.raises(WaveformValidationError):
        lead(samples=values)


@pytest.mark.parametrize(
    "rate",
    [0, -1, True, float("nan"), float("inf"), 100_001, 1e-310, "synthetic-secret"],
)
def test_invalid_rates(rate):
    with pytest.raises(WaveformValidationError):
        waveform(sample_rate_hz=rate)


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"leads": ()}, "leads_invalid"),
        ({"leads": (lead("I"), lead("i"))}, "lead_duplicate"),
        ({"leads": (lead("I"), lead("II", (1,)))}, "alignment_invalid"),
        ({"leads": ("synthetic-secret",)}, "leads_invalid"),
        ({"provenance": "synthetic-secret"}, "provenance_invalid"),
        ({"start_offset_seconds": -1}, "timing_limit"),
        ({"start_offset_seconds": 1_700_000_000}, "timing_limit"),
        ({"sample_offsets_seconds": (0, 0.005, 0.008)}, "timing_ambiguous"),
        ({"sample_offsets_seconds": (0, 0.004)}, "timing_invalid"),
        ({"sample_offsets_seconds": (0, float("nan"), 0.008)}, "timing_invalid"),
        ({"sample_offsets_seconds": (0, 0, 0)}, "timing_ambiguous"),
    ],
)
def test_waveform_validation(kwargs, reason):
    with pytest.raises(WaveformValidationError, match=reason):
        waveform(**kwargs)


def test_privacy_errors_reports_repr_and_logs(caplog):
    sentinel = "SYNTHETIC-PRIVATE-HEADER-Ω"
    constructors = [
        lambda: lead(sentinel),
        lambda: EcgProvenance(sentinel, "a" * 64),
        lambda: EcgProvenance("array", sentinel),
        lambda: waveform(start_offset_seconds=sentinel),
        lambda: EcgLead.from_samples("I", (1,), unit=sentinel),
    ]
    for construct in constructors:
        with pytest.raises(WaveformValidationError) as error:
            construct()
        rendered = "".join(traceback.format_exception(error.value))
        assert sentinel not in rendered
        assert error.value.__context__ is None
    result = waveform(leads=(lead(samples=(123.456789, None, 0)),))
    for rendered in (
        repr(result),
        repr(result.leads[0]),
        result.to_json(),
        caplog.text,
    ):
        assert "123.456789" not in rendered
        assert sentinel not in rendered
    assert "samples_mv" not in result.to_report()
    assert "sample_offsets_seconds" not in result.to_report()


@pytest.mark.parametrize("confirmed", [False, None, 1, "true"])
def test_reviewer_confirmation_is_explicit(confirmed):
    with pytest.raises(WaveformValidationError, match="review_required"):
        waveform().require_reviewer_confirmation(confirmed=confirmed)
    waveform().require_reviewer_confirmation(confirmed=True)
    assert waveform().to_report()["review_required"] is True


def test_conversion_overflow_rejected():
    with pytest.raises(WaveformValidationError, match="sample_invalid"):
        EcgLead.from_samples("I", (1e308,), unit="V")
    with pytest.raises(WaveformValidationError, match="sample_invalid"):
        EcgLead.from_samples(
            "I", (1e308,), unit="adc", baseline_counts=-1e308, gain_counts_per_mv=1
        )


def test_canonical_constructor_and_all_missing_lead():
    result = EcgLead("V1", (None, None), (False, False))
    assert result.samples_mv == (None, None)
    assert waveform(leads=(result,)).to_report()["leads"] == [
        {"lead_id": "V1", "missing_count": 2}
    ]
    with pytest.raises(WaveformValidationError):
        EcgLead("V1", (float("nan"),), (True,))


def test_aggregate_resource_limit():
    large = lead("I", (0,) * 500_001)
    other = lead("II", (0,) * 500_001)
    with pytest.raises(WaveformValidationError, match="resource_limit"):
        waveform(leads=(large, other))
