"""Offline fixture handoff to existing audio preflight contracts."""

import io

import pytest

from openmed.multimodal.asr_audio_profile import (
    MONO_16K_PCM_PROFILE,
    check_asr_compatibility,
)
from openmed.multimodal.wav_metadata import read_wav_metadata
from openmed.training.synthetic.speech_render import (
    GoldPhiSpan,
    ModelFreeSynthesizer,
    ScriptedTurn,
    render_dialogue,
)


@pytest.mark.integration
def test_synthetic_dialogue_handoff_without_provider_qualification():
    wav = io.BytesIO()
    manifest = render_dialogue(
        (
            ScriptedTurn(
                "patient", "en", "Synthetic One speaks", (GoldPhiSpan(0, 13),)
            ),
            ScriptedTurn(
                "reviewer", "fr", "Synthetic Deux répond", (GoldPhiSpan(0, 14),)
            ),
        ),
        ModelFreeSynthesizer(),
        wav,
    ).to_dict()
    metadata = read_wav_metadata(wav.getvalue())
    report = check_asr_compatibility(metadata, MONO_16K_PCM_PROFILE)
    assert report.is_compatible
    assert metadata.frame_count == manifest["frame_count"]
    assert all(turn["phi"] for turn in manifest["turns"])
    assert manifest["reviewer_confirmation_required"] is True
