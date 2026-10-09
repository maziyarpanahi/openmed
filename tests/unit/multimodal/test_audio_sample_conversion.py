"""Offline synthetic controls for bounded local sample conversion."""

import json
from dataclasses import asdict, replace
from fractions import Fraction

import numpy as np
import pytest

from openmed.multimodal.asr_audio_profile import AsrAudioProfile
from openmed.multimodal.audio_resample_plan import RoundingPolicy, plan_audio_resampling
from openmed.multimodal.audio_sample_conversion import (
    AudioConversionError,
    ChannelPolicy,
    ConversionBudget,
    convert_audio_samples,
)
from openmed.multimodal.wav_metadata import WavMetadata


def contracts(
    samples, source_rate=48000, target_rate=16000, rounding=RoundingPolicy.NEAREST_EVEN
):
    frames, channels = samples.shape
    metadata = WavMetadata(
        1,
        channels,
        source_rate,
        16,
        frames * channels * 2,
        frames,
        frames / source_rate,
    )
    plan = plan_audio_resampling(source_rate, target_rate, frames, rounding=rounding)
    profile = AsrAudioProfile("synthetic-local", (1,), (target_rate,), (1, 2), (16,))
    return metadata, plan, profile


def convert(samples, *, chunk=8192, policy=ChannelPolicy.PRESERVE, **kwargs):
    source_rate = kwargs.pop("source_rate", 48000)
    target_rate = kwargs.pop("target_rate", 16000)
    rounding = kwargs.pop("rounding", RoundingPolicy.NEAREST_EVEN)
    return convert_audio_samples(
        (samples[i : i + chunk] for i in range(0, len(samples), chunk)),
        *contracts(samples, source_rate, target_rate, rounding),
        channel_policy=policy,
        **kwargs,
    )


def pcm(result):
    return result.samples_for_reviewed_handoff(reviewer_confirmed=True)


@pytest.mark.parametrize("rounding", list(RoundingPolicy))
@pytest.mark.parametrize(
    "source,target", [(48000, 16000), (44100, 16000), (8000, 16000), (16000, 16000)]
)
def test_silence_counts_duration_and_lineage(source, target, rounding):
    samples = np.zeros((101, 1))
    metadata, plan, _ = contracts(samples, source, target, rounding)
    with convert(
        samples, source_rate=source, target_rate=target, rounding=rounding
    ) as result:
        assert pcm(result).shape == (plan.target_frames, 1)
        assert not pcm(result).any()
        assert abs(result.report.duration_error_seconds) <= 1 / target
        assert result.report.source_position(1) == Fraction(source, target)
        assert result.report.source_interval_seconds(0, plan.target_frames)[
            1
        ] <= Fraction(metadata.frame_count, source)
        assert "non-diagnostic" in result.report.notice


@pytest.mark.parametrize("signal", ["tone", "impulse", "silence"])
def test_chunk_boundary_continuity(signal):
    samples = np.zeros((1401, 2))
    if signal == "tone":
        samples[:, 0] = 0.4 * np.sin(2 * np.pi * 1000 * np.arange(1401) / 44100)
        samples[:, 1] = -0.2 * np.sin(2 * np.pi * 1500 * np.arange(1401) / 44100)
    if signal == "impulse":
        samples[700] = [0.5, -0.5]
    with convert(samples, chunk=1401, source_rate=44100) as whole:
        for size in (1, 17, 700):
            with convert(samples, chunk=size, source_rate=44100) as split:
                np.testing.assert_array_equal(pcm(whole), pcm(split))
                assert whole.report == split.report


def test_anti_alias_rejects_above_nyquist_and_preserves_low_tone():
    t = np.arange(4800) / 48000
    rms = []
    for frequency in (1000, 12000):
        with convert((0.5 * np.sin(2 * np.pi * frequency * t))[:, None]) as result:
            x = pcm(result)[100:-100, 0].astype(float) / 32767
            rms.append(float(np.sqrt(np.mean(x**2))))
    assert 0.34 < rms[0] < 0.36
    assert rms[1] < 0.002


def test_explicit_channel_policy_and_pcm_endpoints():
    samples = np.array([[1.0, -1.0], [0.5, -0.5], [0.0, 0.0]])
    with convert(samples, source_rate=16000) as result:
        np.testing.assert_array_equal(
            pcm(result), [[32767, -32768], [16384, -16384], [0, 0]]
        )
    with convert(samples, source_rate=16000, policy=ChannelPolicy.MEAN_MONO) as result:
        assert pcm(result).shape == (3, 1)
        assert not pcm(result).any()
    with (
        convert(samples, source_rate=16000) as a,
        convert(samples, source_rate=16000, policy=ChannelPolicy.MEAN_MONO) as b,
    ):
        assert a.report.transform_digest != b.report.transform_digest


@pytest.mark.parametrize(
    "value,code",
    [
        (float("nan"), "sample_invalid"),
        (float("inf"), "sample_invalid"),
        (1.01, "input_clipping"),
        (-1.01, "input_clipping"),
    ],
)
def test_invalid_sample_codes(value, code):
    with pytest.raises(AudioConversionError, match=f"^conversion_{code}$"):
        convert(np.array([[0.0], [value]]))


@pytest.mark.parametrize(
    "samples", [np.zeros((4, 1), dtype=np.int16), np.zeros((4, 1), dtype=object)]
)
def test_unsupported_decoded_formats(samples):
    with pytest.raises(AudioConversionError, match="chunk_invalid"):
        convert(samples)


@pytest.mark.parametrize(
    "change",
    [
        dict(format_code=6),
        dict(bit_depth=64),
        dict(channels=65),
        dict(duration_seconds=float("nan")),
        dict(data_byte_count=5),
    ],
)
def test_invalid_metadata(change):
    samples = np.zeros((12, 1))
    metadata, plan, profile = contracts(samples)
    with pytest.raises(AudioConversionError):
        convert_audio_samples(
            [samples],
            replace(metadata, **change),
            plan,
            profile,
            channel_policy=ChannelPolicy.PRESERVE,
        )


def test_forged_plan_and_invalid_policy():
    samples = np.zeros((12, 1))
    metadata, plan, profile = contracts(samples)
    with pytest.raises(AudioConversionError, match="plan_mismatch"):
        convert_audio_samples(
            [samples],
            metadata,
            replace(plan, target_frames=5),
            profile,
            channel_policy=ChannelPolicy.PRESERVE,
        )
    with pytest.raises(AudioConversionError, match="contract_invalid"):
        convert_audio_samples(
            [samples], metadata, plan, profile, channel_policy="mean_mono"
        )


@pytest.mark.parametrize(
    "budget",
    [ConversionBudget(max_buffer_bytes=1), ConversionBudget(max_filter_operations=1)],
)
def test_impossible_budget_does_not_even_consume_source(budget):
    samples = np.zeros((12, 1))

    def chunks():
        pytest.fail("source consumed before admission")
        yield samples

    with pytest.raises(AudioConversionError, match="budget_exceeded"):
        convert_audio_samples(
            chunks(),
            *contracts(samples),
            channel_policy=ChannelPolicy.PRESERVE,
            budget=budget,
        )


def test_chunk_bounds_and_missing_frames():
    samples = np.zeros((12, 1))
    for chunks in (
        [samples[:-1]],
        [samples, samples],
        [samples[:, 0]],
        [np.zeros((12, 2))],
    ):
        with pytest.raises(AudioConversionError):
            convert_audio_samples(
                chunks, *contracts(samples), channel_policy=ChannelPolicy.PRESERVE
            )
    with pytest.raises(AudioConversionError, match="chunk_invalid"):
        convert(samples, budget=ConversionBudget(max_chunk_frames=1))


def test_profile_transform_and_duration_guards():
    samples = np.zeros((12, 2))
    metadata, plan, profile = contracts(samples)
    profiles = [
        replace(profile, allow_resample=False),
        replace(profile, max_duration_seconds=0.0001),
        replace(profile, bit_depths=(32,)),
        replace(profile, channel_counts=(1,), allow_downmix=False),
    ]
    for candidate in profiles:
        with pytest.raises(AudioConversionError):
            convert_audio_samples(
                [samples],
                metadata,
                plan,
                candidate,
                channel_policy=ChannelPolicy.MEAN_MONO,
            )


def test_rounding_cannot_violate_output_duration():
    samples = np.zeros((5, 1))
    metadata, plan, profile = contracts(samples, 48000, 16000, RoundingPolicy.CEILING)
    profile = replace(profile, max_duration_seconds=metadata.duration_seconds)
    with pytest.raises(AudioConversionError, match="target_incompatible"):
        convert_audio_samples(
            [samples], metadata, plan, profile, channel_policy=ChannelPolicy.PRESERVE
        )


@pytest.mark.parametrize("cancel_at", [1, 5, 45, 110])
def test_cancel_releases_all_owned_buffers(monkeypatch, cancel_at):
    allocated = []
    zeros = np.zeros

    def tracked(*args, **kwargs):
        buffer = zeros(*args, **kwargs)
        allocated.append(buffer)
        return buffer

    samples = np.full((100, 1), 0.25)
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= cancel_at

    monkeypatch.setattr(np, "zeros", tracked)
    with pytest.raises(AudioConversionError, match="cancelled"):
        convert(samples, cancelled=cancelled)
    assert all(not buffer.any() for buffer in allocated)


def test_output_clipping_is_rejected_not_saturated():
    samples = np.zeros((200, 1))
    samples[70:130] = 1
    with pytest.raises(AudioConversionError, match="output_clipping"):
        convert(samples)


def test_output_lifetime_privacy_and_offsets():
    samples = np.full((100, 1), 0.1234567)
    result = convert(samples)
    with pytest.raises(AudioConversionError, match="review_required"):
        result.samples_for_reviewed_handoff(reviewer_confirmed=False)
    view = pcm(result)
    report = json.dumps(asdict(result.report))
    assert "0.1234567" not in report
    assert "samples" not in report
    assert ".1234567" not in repr(result)
    assert result.report.source_interval_seconds(2, 3) == (
        Fraction(2, 16000),
        Fraction(3, 16000),
    )
    for offsets in [(-1, 1), (2, 1), (0, 1000), (True, 1)]:
        with pytest.raises(AudioConversionError):
            result.report.source_interval_seconds(*offsets)
    result.close()
    result.close()
    assert not view.any()
    with pytest.raises(AudioConversionError, match="closed"):
        pcm(result)


def test_iterator_error_is_sanitized_and_source_unchanged():
    samples = np.full((12, 1), 0.25)

    def chunks():
        yield samples[:6]
        raise RuntimeError("synthetic private source/path")

    with pytest.raises(AudioConversionError) as error:
        convert_audio_samples(
            chunks(), *contracts(samples), channel_policy=ChannelPolicy.PRESERVE
        )
    assert str(error.value) == "conversion_failed"
    assert error.value.__suppress_context__
    assert (samples == 0.25).all()


def test_optional_backend_missing_is_controlled(monkeypatch):
    import builtins

    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("synthetic private package path")
        return original(name, *args, **kwargs)

    samples = np.zeros((12, 1))
    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(AudioConversionError, match="^conversion_backend_missing$"):
        convert(samples)


@pytest.mark.parametrize("boundary", ["cancelled", "chunks"])
def test_source_bearing_errors_cannot_masquerade_as_conversion_errors(boundary):
    samples = np.zeros((12, 1))

    def failure():
        raise AudioConversionError("synthetic private source/path")

    def chunks():
        failure()
        yield samples

    with pytest.raises(AudioConversionError, match="^conversion_failed$"):
        convert_audio_samples(
            chunks() if boundary == "chunks" else [samples],
            *contracts(samples),
            channel_policy=ChannelPolicy.PRESERVE,
            cancelled=failure if boundary == "cancelled" else lambda: False,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_supported_decoded_depths_and_exact_storage_budget(dtype):
    samples = np.full((12, 2), 0.25, dtype=dtype)
    taps = 2 * 107 + 1  # ceil(32 / (0.9 * 16000/48000))
    required = 8 * 12 * 2 + 2 * 4 * 2 + 8 * taps
    with convert(samples, budget=ConversionBudget(max_buffer_bytes=required)) as result:
        assert (pcm(result) > 0).all()
    with pytest.raises(AudioConversionError, match="budget_exceeded"):
        convert(samples, budget=ConversionBudget(max_buffer_bytes=required - 1))
