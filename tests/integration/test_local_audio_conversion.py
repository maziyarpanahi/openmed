"""Synthetic provider-boundary checks; no real inference or network calls."""

import numpy as np
import pytest

from openmed.multimodal.asr_audio_profile import MONO_16K_PCM_PROFILE
from openmed.multimodal.audio_resample_plan import plan_audio_resampling
from openmed.multimodal.audio_sample_conversion import (
    AudioConversionError,
    ChannelPolicy,
    convert_audio_samples,
)
from openmed.multimodal.wav_metadata import WavMetadata


@pytest.mark.integration
@pytest.mark.parametrize("bad", [False, True])
def test_no_provider_handoff_until_complete_validation_and_review(bad):
    samples = np.zeros((4800, 2))
    if bad:
        samples[-1, 1] = float("nan")
    metadata = WavMetadata(1, 2, 48000, 16, 19200, 4800, 0.1)
    plan = plan_audio_resampling(48000, 16000, 4800)
    calls = []

    def run():
        with convert_audio_samples(
            [samples[:2400], samples[2400:]],
            metadata,
            plan,
            MONO_16K_PCM_PROFILE,
            channel_policy=ChannelPolicy.MEAN_MONO,
        ) as converted:
            with pytest.raises(AudioConversionError, match="review_required"):
                converted.samples_for_reviewed_handoff(reviewer_confirmed=False)
            calls.append(
                converted.samples_for_reviewed_handoff(reviewer_confirmed=True).shape
            )
            assert converted.report.source_position(1600) == 4800

    if bad:
        with pytest.raises(AudioConversionError, match="sample_invalid"):
            run()
        assert calls == []
    else:
        run()
        assert calls == [(1600, 1)]
