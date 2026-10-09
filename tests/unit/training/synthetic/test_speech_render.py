"""Synthetic offline renderer controls; all identifiers are invented."""

from __future__ import annotations

import hashlib
import io
import json
import socket
import struct
import wave
from dataclasses import replace

import pytest

from openmed.multimodal.asr_audio_profile import (
    MONO_16K_PCM_PROFILE,
    AsrAudioProfile,
    check_asr_compatibility,
)
from openmed.multimodal.wav_metadata import read_wav_metadata
from openmed.training.synthetic.speech_render import (
    Augmentations,
    GoldPhiSpan,
    ModelFreeSynthesizer,
    ScriptedTurn,
    SpeechRenderError,
    SynthesizedChunk,
    SynthesizerDeclaration,
    WordAlignment,
    render_dialogue,
)


def scripts():
    return (
        ScriptedTurn("patient", "en", "I am Synthetic Ada.", (GoldPhiSpan(5, 18),)),
        ScriptedTurn("reviewer", "fr", "Bonjour Élise Test.", (GoldPhiSpan(8, 18),)),
    )


def render(turns=None, provider=None, **kwargs):
    wav = io.BytesIO()
    output = io.StringIO()
    manifest = render_dialogue(
        turns or scripts(),
        provider or ModelFreeSynthesizer(),
        wav,
        manifest_output=output,
        **kwargs,
    )
    assert output.getvalue() == manifest.to_json()
    assert json.loads(output.getvalue()) == manifest.to_dict()
    with wave.open(io.BytesIO(wav.getvalue()), "rb") as reader:
        pcm = reader.readframes(reader.getnframes())
    return wav.getvalue(), pcm, manifest.to_dict()


def test_deterministic_two_speaker_phi_sample_truth_and_word_fragments():
    first = render()
    assert first == render()
    _, pcm, manifest = first
    provider = ModelFreeSynthesizer()
    for script, turn in zip(scripts(), manifest["turns"]):
        assert turn["start_seconds"] == turn["start_sample"] / 16000
        assert turn["end_seconds"] == turn["end_sample"] / 16000
        assert turn["overlapping_turns"] == []
        for phi in turn["phi"]:
            text = script.text[phi["char_start"] : phi["char_end"]]
            expected = provider.synthesize(
                text,
                speaker_role=script.speaker_role,
                language=script.language,
                sample_rate_hz=16000,
            ).pcm16
            actual = pcm[2 * phi["start_sample"] : 2 * phi["end_sample"]]
            assert actual == expected
            assert phi["source_pcm_sha256"] == hashlib.sha256(actual).hexdigest()
            assert phi["rendered_pcm_sha256"] == phi["source_pcm_sha256"]
        # Punctuation immediately after a PHI span splits a script token. Both
        # exact fragment intervals are retained instead of estimating timing.
        last_word = turn["words"][-1]
        assert len(last_word["fragments"]) == 2
        assert (
            last_word["fragments"][0]["end_sample"] + 160
            == last_word["fragments"][1]["start_sample"]
        )


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000])
def test_wav_metadata_matches_declared_asr_profile(rate):
    wav, pcm, manifest = render(sample_rate_hz=rate)
    metadata = read_wav_metadata(wav)
    profile = replace(MONO_16K_PCM_PROFILE, sample_rates_hz=(rate,))
    assert isinstance(profile, AsrAudioProfile)
    assert check_asr_compatibility(metadata, profile).is_compatible
    assert metadata.frame_count == len(pcm) // 2 == manifest["frame_count"]
    assert metadata.sample_rate_hz == manifest["sample_rate_hz"] == rate
    assert metadata.bit_depth == 16
    assert manifest["wav_sha256"] == hashlib.sha256(wav).hexdigest()


@pytest.mark.parametrize(
    "license_value",
    [
        None,
        "",
        "unknown",
        "LicenseRef-private",
        "GPL-3.0-only",
        "MPL-2.0",
        "MIT OR GPL-3.0-only",
    ],
)
@pytest.mark.parametrize("field", ["engine_license", "voice_license"])
def test_license_negative_controls_rejected_before_synthesis(license_value, field):
    class NeverCalled(ModelFreeSynthesizer):
        declaration = replace(
            ModelFreeSynthesizer.declaration, **{field: license_value}
        )

        def synthesize(self, *args, **kwargs):
            pytest.fail("Rejected adapter was called")

    wav = io.BytesIO()
    with pytest.raises(SpeechRenderError, match="speech_license_rejected"):
        render_dialogue(scripts(), NeverCalled(), wav)
    assert wav.getvalue() == b""


@pytest.mark.parametrize("field", ["offline", "synthetic_voice"])
def test_cloud_and_cloned_voice_declarations_rejected(field):
    provider = ModelFreeSynthesizer()
    provider.declaration = replace(provider.declaration, **{field: False})
    with pytest.raises(SpeechRenderError, match="speech_adapter_policy_rejected"):
        render(provider=provider)


def test_missing_declaration_is_rejected():
    with pytest.raises(SpeechRenderError, match="speech_declaration_invalid"):
        render(provider=object())


def test_privacy_network_negative_control_and_safety_notice(monkeypatch):
    def fail_network(*args, **kwargs):
        pytest.fail("Network was called")

    monkeypatch.setattr(socket, "socket", fail_network)
    monkeypatch.setattr(socket, "create_connection", fail_network)
    _, _, manifest = render()
    encoded = json.dumps(manifest)
    for script in scripts():
        for source in (script.text, script.speaker_role, script.language):
            assert f'"{source}"' not in encoded
        for span in script.phi_spans:
            assert script.text[span.start : span.end] not in encoded
    assert "pcm16" not in encoded
    assert "transcript" not in encoded
    assert manifest["synthetic_fixture"] is True
    assert manifest["reviewer_confirmation_required"] is True
    assert "non-diagnostic" in manifest["non_diagnostic_notice"]
    assert "Synthetic Ada" not in repr(scripts())


@pytest.mark.parametrize(
    "text,spans",
    [
        ("Identifiant 00123456789.", (GoldPhiSpan(12, 23),)),
        ("ID Тест00042.", (GoldPhiSpan(3, 12),)),
        ("ID 合成人物42。", (GoldPhiSpan(3, 9),)),
        ("رقم ٠٠١٢٣٤.", (GoldPhiSpan(4, 10),)),
        ("A B", (GoldPhiSpan(0, 1), GoldPhiSpan(2, 3))),
        ("AB", (GoldPhiSpan(0, 1), GoldPhiSpan(1, 2))),
    ],
)
def test_multilingual_identifier_and_adjacent_span_integrity(text, spans):
    turn = ScriptedTurn("synthetic", "und", text, spans)
    _, pcm, manifest = render((turn,), chunk_gap_samples=7)
    assert len(manifest["turns"][0]["phi"]) == len(spans)
    for source, truth in zip(spans, manifest["turns"][0]["phi"]):
        expected = (
            ModelFreeSynthesizer()
            .synthesize(
                text[source.start : source.end],
                speaker_role="synthetic",
                language="und",
                sample_rate_hz=16000,
            )
            .pcm16
        )
        assert pcm[truth["start_sample"] * 2 : truth["end_sample"] * 2] == expected


@pytest.mark.parametrize(
    "spans",
    [
        (GoldPhiSpan(-1, 2),),
        (GoldPhiSpan(2, 2),),
        (GoldPhiSpan(0, 999),),
        (GoldPhiSpan(1, 4), GoldPhiSpan(3, 5)),
        (GoldPhiSpan(5, 6), GoldPhiSpan(0, 1)),
        (GoldPhiSpan(True, 2),),
        (GoldPhiSpan(1, 2),),
    ],
)
def test_invalid_phi_spans_fail_before_provider_call(spans):
    class NeverCalled(ModelFreeSynthesizer):
        def synthesize(self, *args, **kwargs):
            pytest.fail("Invalid script was synthesized")

    with pytest.raises(SpeechRenderError, match="speech_phi_span_invalid"):
        render((ScriptedTurn("a", "en", "A BC DE", spans),), NeverCalled())


@pytest.mark.parametrize(
    "options",
    [
        Augmentations(noise_amplitude=100, seed=12),
        Augmentations(gain=2),
        Augmentations(overlap_samples=100),
        Augmentations(band_limit=True),
        Augmentations(
            seed=72, noise_amplitude=30, gain=1.5, overlap_samples=250, band_limit=True
        ),
    ],
)
def test_seeded_augmentations_and_timing_effects(options):
    base_wav, base_pcm, base = render()
    wav, pcm, manifest = render(augmentations=options)
    assert (wav, pcm, manifest) == render(augmentations=options)
    assert wav != base_wav
    assert manifest["augmentations"]["labels"]
    assert (
        manifest["augmentations"]["amplitude_transforms_preserve_frame_count"] is True
    )
    for old, new in zip(base["turns"], manifest["turns"]):
        shift = new["timeline_shift_samples"]
        assert new["start_sample"] == old["start_sample"] + shift
        for old_phi, new_phi in zip(old["phi"], new["phi"]):
            assert new_phi["start_sample"] == old_phi["start_sample"] + shift
            assert new_phi["end_sample"] == old_phi["end_sample"] + shift
            assert new_phi["source_pcm_sha256"] == old_phi["source_pcm_sha256"]
            halo = 2 if options.band_limit else 0
            assert new_phi["affected_start_sample"] == max(
                0, new_phi["start_sample"] - halo
            )
    if options.overlap_samples:
        assert (
            manifest["frame_count"]
            == base["frame_count"] - 1600 - options.overlap_samples
        )
        assert manifest["turns"][0]["overlapping_turns"] == [1]
        assert manifest["turns"][1]["overlapping_turns"] == [0]
    else:
        assert len(pcm) == len(base_pcm)


def test_noise_seed_changes_signal_but_preserves_truth():
    _, pcm1, m1 = render(augmentations=Augmentations(seed=3, noise_amplitude=400))
    _, pcm2, m2 = render(augmentations=Augmentations(seed=4, noise_amplitude=400))
    assert pcm1 != pcm2
    assert (
        m1["turns"][0]["phi"][0]["start_sample"]
        == m2["turns"][0]["phi"][0]["start_sample"]
    )


def test_gain_clips_with_measured_count():
    class Loud(ModelFreeSynthesizer):
        def synthesize(self, text, **kwargs):
            chunk = super().synthesize(text, **kwargs)
            return replace(
                chunk, pcm16=struct.pack("<h", 30000) * (len(chunk.pcm16) // 2)
            )

    _, pcm, manifest = render(provider=Loud(), augmentations=Augmentations(gain=2))
    assert manifest["augmentations"]["clipped_samples"] > 0
    assert max(value[0] for value in struct.iter_unpack("<h", pcm)) == 32767


@pytest.mark.parametrize(
    "result",
    [
        SynthesizedChunk(b"x", ()),
        SynthesizedChunk(b"", ()),
        SynthesizedChunk(b"\0\0", ()),
        SynthesizedChunk(b"\0\0", (WordAlignment(0, 999, 0, 1),)),
        SynthesizedChunk(b"\0\0", (WordAlignment(0, 1, -1, 1),)),
        SynthesizedChunk(b"\0\0", (WordAlignment(0, 1, 0, 99),)),
        SynthesizedChunk(b"\0\0", (WordAlignment(0, 1, 0, 0),)),
    ],
)
def test_provider_chunk_and_alignment_negative_controls(result):
    class Broken(ModelFreeSynthesizer):
        def synthesize(self, *args, **kwargs):
            return result

    with pytest.raises(SpeechRenderError, match="speech_(chunk|alignment)_invalid"):
        render((ScriptedTurn("test", "en", "A"),), Broken())


def test_provider_and_output_errors_do_not_retain_private_context():
    class Broken(ModelFreeSynthesizer):
        def synthesize(self, *args, **kwargs):
            raise RuntimeError("private synthetic transcript /secret/path")

    class BrokenOutput(io.BytesIO):
        def write(self, *args, **kwargs):
            raise RuntimeError("private path")

    for provider, output in (
        (Broken(), io.BytesIO()),
        (ModelFreeSynthesizer(), BrokenOutput()),
    ):
        with pytest.raises(SpeechRenderError) as caught:
            render_dialogue(scripts(), provider, output)
        assert caught.value.__context__ is None
        assert caught.value.__cause__ is None
        assert "private" not in str(caught.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate_hz": True},
        {"sample_rate_hz": 0},
        {"max_frames": 0},
        {"chunk_gap_samples": -1},
        {"turn_gap_samples": -1},
        {"augmentations": Augmentations(gain=float("nan"))},
        {"augmentations": Augmentations(gain=0)},
        {"augmentations": Augmentations(noise_amplitude=-1)},
        {"augmentations": Augmentations(overlap_samples=-1)},
        {"augmentations": Augmentations(seed=True)},
    ],
)
def test_option_negative_controls(kwargs):
    with pytest.raises(SpeechRenderError, match="speech_options_invalid"):
        render(**kwargs)


def test_bounded_render_leaves_output_empty():
    output = io.BytesIO()
    with pytest.raises(SpeechRenderError):
        render_dialogue(scripts(), ModelFreeSynthesizer(), output, max_frames=10)
    assert output.getvalue() == b""


def test_overlap_requires_two_distinct_speakers_and_valid_duration():
    turn = ScriptedTurn("test", "en", "A")
    with pytest.raises(SpeechRenderError, match="speech_overlap_invalid"):
        render((turn, turn), augmentations=Augmentations(overlap_samples=1))
    with pytest.raises(SpeechRenderError, match="speech_overlap_invalid"):
        render(augmentations=Augmentations(overlap_samples=90000))


def test_license_normalization_and_manifest_immutability():
    provider = ModelFreeSynthesizer()
    provider.declaration = SynthesizerDeclaration("mit", "bsd-3-clause")
    result = render_dialogue(scripts(), provider, io.BytesIO())
    changed = result.to_dict()
    changed["turns"].clear()
    assert len(result.to_dict()["turns"]) == 2
    assert result.to_dict()["engine_license"] == "MIT"


def test_actual_overlap_mix_and_accumulated_timeline_shift():
    class Fixed(ModelFreeSynthesizer):
        def synthesize(self, text, **kwargs):
            value = {"a": 100, "b": 200, "c": -50}[kwargs["speaker_role"]]
            return SynthesizedChunk(
                struct.pack("<h", value) * 10,
                (WordAlignment(0, len(text), 0, 10),),
            )

    turns = tuple(
        ScriptedTurn(role, "en", "A", (GoldPhiSpan(0, 1),)) for role in ("a", "b", "c")
    )
    _, pcm, manifest = render(
        turns,
        Fixed(),
        turn_gap_samples=4,
        augmentations=Augmentations(overlap_samples=3),
    )
    assert [sample[0] for sample in struct.iter_unpack("<h", pcm)] == [100] * 7 + [
        300
    ] * 3 + [200] * 4 + [150] * 3 + [-50] * 7
    assert [turn["timeline_shift_samples"] for turn in manifest["turns"]] == [
        0,
        -7,
        -14,
    ]
    assert manifest["sequential_frame_count"] == 38
    assert manifest["frame_count_delta"] == -14


def test_bandpass_impulse_support_is_measured_and_zero_delay():
    class Impulse(ModelFreeSynthesizer):
        def synthesize(self, text, **kwargs):
            return SynthesizedChunk(
                struct.pack("<7h", 0, 0, 0, 400, 0, 0, 0),
                (WordAlignment(0, 1, 3, 4),),
            )

    _, pcm, manifest = render(
        (ScriptedTurn("a", "en", "A"),),
        Impulse(),
        augmentations=Augmentations(band_limit=True),
    )
    assert [sample[0] for sample in struct.iter_unpack("<h", pcm)] == [
        0,
        -100,
        0,
        200,
        0,
        -100,
        0,
    ]
    word = manifest["turns"][0]["words"][0]
    assert (word["start_sample"], word["end_sample"]) == (3, 4)
    assert (word["affected_start_sample"], word["affected_end_sample"]) == (1, 6)


def test_provider_word_alignments_are_used_without_proportional_estimates():
    class Uneven(ModelFreeSynthesizer):
        def synthesize(self, text, **kwargs):
            return SynthesizedChunk(
                struct.pack("<9h", 1, 2, 3, 4, 5, 6, 7, 8, 9),
                (WordAlignment(0, 1, 0, 1), WordAlignment(2, 3, 2, 9)),
            )

    _, _, manifest = render((ScriptedTurn("a", "en", "A B"),), Uneven())
    assert [
        (word["start_sample"], word["end_sample"])
        for word in manifest["turns"][0]["words"]
    ] == [(0, 1), (2, 9)]


@pytest.mark.parametrize("field", ["text", "language", "speaker_role"])
def test_unencodable_source_is_rejected_before_synthesis(field):
    class NeverCalled(ModelFreeSynthesizer):
        def synthesize(self, *args, **kwargs):
            pytest.fail("Unencodable source was synthesized")

    turn = replace(scripts()[0], **{field: "synthetic\ud800"})
    with pytest.raises(SpeechRenderError, match="speech_script_invalid") as caught:
        render((turn,), NeverCalled())
    assert caught.value.__context__ is None
