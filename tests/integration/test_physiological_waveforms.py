"""Offline mixed-kind acquisition-to-review contract integration."""

import pytest

from openmed.multimodal.physiological_waveforms import (
    CHANNEL_CONSTRAINTS,
    ChannelKind,
    WaveformChannel,
    WaveformContractError,
    WaveformProvenance,
    evaluate_recording,
)


@pytest.mark.integration
def test_mixed_rates_preserve_independent_clocks_and_review_boundary():
    channels = []
    for kind in ChannelKind:
        limits = CHANNEL_CONSTRAINTS[kind]
        rate = limits.minimum_rate_hz
        span = limits.maximum - limits.minimum
        channels.append(
            WaveformChannel(
                kind=kind,
                unit=limits.unit,
                sample_rate_hz=rate,
                acquisition_minimum=limits.minimum,
                acquisition_maximum=limits.maximum,
                samples=tuple(
                    limits.minimum + span * (0.45 if i % 2 else 0.55) for i in range(40)
                ),
                offsets_seconds=tuple(2 + i / rate for i in range(40)),
                provenance=WaveformProvenance("b" * 64),
            )
        )
    report = evaluate_recording(tuple(channels))
    assert [item.kind for item in report.channels] == list(ChannelKind)
    assert {item.state.value for item in report.channels} == {"pass"}
    with pytest.raises(WaveformContractError, match="reviewer_confirmation_required"):
        report.require_reviewer_confirmation()
    report.require_reviewer_confirmation(confirmed=True)
    assert "b" * 64 not in report.to_json()
    assert "non_diagnostic" in report.to_json()
