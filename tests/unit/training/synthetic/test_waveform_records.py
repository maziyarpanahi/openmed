"""Declared waveform stimulus, timing, determinism and bounds controls."""

import random

import pytest

from openmed.training.synthetic.waveform_records import (
    NOTICE,
    WaveformConfig,
    generate_waveform_record,
)


def test_multilead_generation_has_aligned_sample_and_event_truth():
    record = generate_waveform_record(
        WaveformConfig(duration_seconds=2, leads=("II", "I"))
    )
    assert record.config.leads == ("I", "II")
    assert all(len(samples) == 500 for _lead, samples in record.samples)
    assert all(index == round(seconds * 250) for _kind, index, seconds in record.events)
    assert record.to_dict()["synthetic"] is True
    assert record.to_dict()["non_diagnostic_notice"] == NOTICE


def test_identical_seed_and_lead_permutations_have_identical_digest():
    first = generate_waveform_record(
        WaveformConfig(
            seed=7, rhythm="irregular", noise_amplitude_mv=0.05, leads=("I", "II")
        )
    )
    second = generate_waveform_record(
        WaveformConfig(
            seed=7, rhythm="irregular", noise_amplitude_mv=0.05, leads=("II", "I")
        )
    )
    assert first.report() == second.report()
    assert first.samples == second.samples


def test_pause_is_recorded_as_generator_event_and_changes_signal():
    regular = generate_waveform_record(WaveformConfig(duration_seconds=5))
    paused = generate_waveform_record(
        WaveformConfig(duration_seconds=5, rhythm="pause")
    )
    events = [event for event in paused.events if event[0] == "template_pause"]
    assert len(events) == 1
    index = events[0][1]
    assert regular.samples[0][1][index] > paused.samples[0][1][index] + 0.9


def test_noise_and_irregular_timing_are_configurable_and_seeded():
    regular = generate_waveform_record(WaveformConfig(seed=1))
    irregular = generate_waveform_record(WaveformConfig(seed=1, rhythm="irregular"))
    noisy = generate_waveform_record(WaveformConfig(seed=1, noise_amplitude_mv=0.1))
    assert irregular.events != regular.events
    assert noisy.samples != regular.samples


def test_annotations_can_be_disabled_without_changing_waveform():
    first = generate_waveform_record(WaveformConfig(annotations=False))
    second = generate_waveform_record()
    assert first.events == ()
    assert first.samples == second.samples


def test_generation_does_not_mutate_global_random_state():
    before = random.getstate()
    generate_waveform_record()
    assert random.getstate() == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_seconds": 100},
        {"sample_rate_hz": True},
        {"noise_amplitude_mv": float("nan")},
        {"rhythm": "diagnosis"},
        {"leads": ("I", "I")},
        {"pulses_per_minute": 0},
    ],
)
def test_invalid_parameters_fail_before_generation(kwargs):
    with pytest.raises(ValueError):
        WaveformConfig(**kwargs)


def test_metadata_report_contains_no_signal_payload():
    assert "signals" not in generate_waveform_record().report()
    assert "samples" not in generate_waveform_record().report()


def test_constructor_rejects_conflicting_event_truth_and_disabled_annotations():
    from dataclasses import replace

    record = generate_waveform_record()
    with pytest.raises(ValueError):
        replace(record, events=(("template_pulse", 0, 4.0),))
    with pytest.raises(ValueError):
        replace(record, config=WaveformConfig(annotations=False))


def test_constructor_does_not_consume_unbounded_iterables():
    from dataclasses import replace

    record = generate_waveform_record()

    def forbidden():
        raise AssertionError("iterator must not be consumed")
        yield 0

    with pytest.raises(ValueError):
        replace(record, samples=forbidden())
    with pytest.raises(ValueError):
        replace(record, events=forbidden())
    with pytest.raises(ValueError):
        replace(record, samples=(("I", forbidden()), ("II", record.samples[1][1])))
