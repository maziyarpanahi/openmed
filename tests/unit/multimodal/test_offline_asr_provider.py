"""Safety boundaries for streaming ASR, without weights or optional runtimes."""

from __future__ import annotations

import json
import threading
from dataclasses import replace

import pytest

from openmed.mlx.models.whisper import WhisperAssets, WhisperError
from openmed.multimodal.asr_hallucination import (
    VoiceActivityInterval,
    VoiceActivityKind,
    VoiceActivityTimeline,
)
from openmed.multimodal.runtime.thermal_gate import (
    MemoryPressure,
    PlatformObservation,
    ThermalState,
)
from openmed.multimodal.streaming import providers as module
from openmed.multimodal.streaming.privacy.withholding_buffer import (
    PhiRedactionSpan,
    PhiResolution,
    PhiScreenDeclaration,
    PhiScreenResult,
    SegmentPhiDecision,
)
from openmed.multimodal.streaming.speech.providers import AudioChunk


class Screen:
    declaration = PhiScreenDeclaration(artifact_digest="a" * 64)

    def __init__(self):
        self.fail = False
        self.requests = []

    def screen(self, request):
        self.requests.append(request.request_digest)
        if self.fail:
            raise RuntimeError("Synthetic Patient Secret")
        return PhiScreenResult(
            request_digest=request.request_digest,
            decisions=tuple(
                SegmentPhiDecision(
                    segment_ref=s.segment_ref,
                    revision=s.revision,
                    resolution=PhiResolution.RESOLVED,
                    redactions=(PhiRedactionSpan(0, len(s.text)),),
                )
                for s in request.segments
            ),
        )


class Worker:
    versions = {"mlx-whisper": "test"}

    def __init__(self, assets):
        self.closed = False
        self.text = "Synthetic Patient Secret"
        self.probability = 0.99
        self.entered = threading.Event()
        self.resume = None

    def decode(self, pcm, *, timeout):
        self.entered.set()
        if self.resume is not None:
            assert self.resume.wait(5)
        return {
            "text": self.text,
            "language": "en",
            "language_probability": self.probability,
            "words": [
                {
                    "word": " " + word,
                    "start": i * 0.15,
                    "end": (i + 1) * 0.15,
                    "probability": 0.9,
                }
                for i, word in enumerate(self.text.split())
            ],
            "no_speech_probability": 0.01,
            "avg_logprob": -0.1,
            "compression_ratio": 1.0,
            "token_count": 4,
            "network_attempts": 0,
            "peak_rss_bytes": 100,
            "peak_device_bytes": 100,
        }

    def close(self):
        self.closed = True
        if self.resume is not None:
            self.resume.set()


@pytest.fixture
def assets(tmp_path):
    config = {
        "n_mels": 128,
        "n_audio_ctx": 1500,
        "n_audio_state": 1280,
        "n_audio_head": 20,
        "n_audio_layer": 32,
        "n_vocab": 51866,
        "n_text_ctx": 448,
        "n_text_state": 1280,
        "n_text_head": 20,
        "n_text_layer": 4,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "weights.safetensors").write_bytes(b"synthetic manifest fixture only")
    return WhisperAssets(tmp_path, "b" * 40, b"caller supplied license snapshot")


@pytest.fixture
def stream(monkeypatch, assets):
    monkeypatch.setattr(module, "WhisperProcess", Worker)
    observation = [
        PlatformObservation(
            observed_at=0.0,
            thermal=ThermalState.NOMINAL,
            memory_pressure=MemoryPressure.NORMAL,
            available_bytes=10 * 1024**3,
        )
    ]
    screen = Screen()
    session = module.OfflineWhisperSession(
        assets,
        screen,
        languages=("en",),
        observe=lambda: observation[0],
        clock=lambda: 0.0,
    )
    yield session, screen, observation
    session.cancel()


def chunk(sequence, milliseconds=500):
    return AudioChunk(
        sequence, b"\x01\x00" * (16 * milliseconds), 16_000, 1, milliseconds
    )


def timeline(milliseconds=1000):
    return VoiceActivityTimeline(
        milliseconds,
        (VoiceActivityInterval(0, milliseconds, VoiceActivityKind.SPEECH),),
    )


def finalize_window(session, sequence=0):
    session.push_chunk(chunk(sequence))
    partial = session.partial()
    assert partial.reason == "partial_withheld" and not partial.released
    session.push_chunk(chunk(sequence + 1))
    return session.end_utterance(timeline())


def test_partials_private_final_requires_screen_review_and_timing_preserved(stream):
    session, screen, _ = stream
    update = finalize_window(session)
    assert update.reason == "context_or_phi_withheld" and not update.released
    assert not screen.requests
    final = session.finish()
    assert final.reason == "finalized" and len(final.released) == 1
    segment = final.released[0]
    with pytest.raises(module.OfflineAsrError, match="review_required"):
        segment.reviewed_text()
    assert "Synthetic" not in segment.reviewed_text(reviewer_confirmed=True)
    assert segment.speaker == "unknown"
    assert all(t.token == "" for t in segment.token_times)
    assert segment.token_times[0].start_ms == 0
    assert "Synthetic" not in repr(final) + repr(segment) + json.dumps(
        segment.to_dict()
    ) + json.dumps(session.report())
    assert session.report()["buffered_bytes"] == 0 and session._worker.closed
    assert not any(session._window)


def test_context_screen_receives_all_finalized_utterances_and_retains_timing(stream):
    session, screen, _ = stream
    finalize_window(session)
    released = finalize_window(session, 2).released
    assert len(released) == 1 and len(screen.requests) == 1
    last = session.finish().released[0]
    assert last.token_times[0].start_ms == 1000


def test_stability_needs_growing_audio_not_duplicate_decode(stream):
    session, _, _ = stream
    session.push_chunk(chunk(0))
    session.partial()
    update = session.end_utterance(timeline(500))
    assert not update.released and session.report()["closed"]
    assert session.report()["buffered_bytes"] == 0


def test_revised_text_does_not_release(stream):
    session, _, _ = stream
    session.push_chunk(chunk(0))
    session.partial()
    session._worker.text = "Changed synthetic content"
    session.push_chunk(chunk(1))
    assert not session.end_utterance(timeline()).released
    assert session.report()["closed"]


def test_screener_exception_is_value_free_and_drops_all_pending(stream):
    session, screen, _ = stream
    finalize_window(session)
    screen.fail = True
    result = session.finish()
    assert result.reason == "final_withheld" and not result.released
    assert session.report()["pending_segments"] == 0
    assert "Synthetic" not in repr(result) + json.dumps(session.report())


def test_missing_or_non_speech_evidence_cannot_release(stream):
    session, _, _ = stream
    session.push_chunk(chunk(0))
    session.partial()
    session.push_chunk(chunk(1))
    silent = VoiceActivityTimeline(
        1000, (VoiceActivityInterval(0, 1000, VoiceActivityKind.SILENCE),)
    )
    assert not session.end_utterance(silent).released
    assert session.report()["closed"] and not any(session._window)


def test_resource_degradation_cancels_and_overwrites_audio(stream):
    session, _, observations = stream
    session.push_chunk(chunk(0))
    observations[0] = replace(observations[0], thermal=ThermalState.SERIOUS)
    with pytest.raises(module.OfflineAsrError, match="offline_asr_input_rejected"):
        session.push_chunk(chunk(1))
    assert session._worker.closed and not any(session._window)


def test_pcm_length_sequence_and_rate_are_enforced(stream):
    session, _, _ = stream
    bad = AudioChunk(0, b"synthetic", 16_000, 1, 100)
    with pytest.raises(module.OfflineAsrError):
        session.push_chunk(bad)
    assert session.report()["closed"] and not any(session._window)


def test_cancel_interrupts_inflight_decoder_without_releasing_late_result(stream):
    session, _, _ = stream
    session.push_chunk(chunk(0))
    session._worker.resume = threading.Event()
    outputs = []
    thread = threading.Thread(target=lambda: outputs.append(session.partial()))
    thread.start()
    assert session._worker.entered.wait(5)
    session.cancel()
    thread.join(5)
    assert not thread.is_alive()
    assert outputs[0].reason == "cancelled" and not outputs[0].released
    assert session._worker.closed and not any(session._window)


def test_retention_expiry_is_terminal_even_when_idle(stream):
    session, _, _ = stream
    session.push_chunk(chunk(0))
    session._expire()
    assert session.report()["reason"] == "retention_expired"
    assert not any(session._window) and session._worker.closed


def test_pending_text_expires_and_old_audio_timer_cannot_cancel_new_window(stream):
    session, _, _ = stream
    generation = session._audio_generation
    finalize_window(session)
    session._expire_window(generation)
    assert not session.report()["closed"]
    assert session._privacy_timer is not None
    session._expire_privacy(session._privacy_generation)
    assert session.report()["pending_segments"] == 0
    assert session.report()["reason"] == "retention_expired"


def test_asset_revision_license_identity_and_mutation(assets):
    changed = WhisperAssets(
        assets.directory, assets.revision, b"different terms snapshot"
    )
    assert changed.fingerprint != assets.fingerprint
    assets.verify()
    (assets.directory / "weights.safetensors").write_bytes(b"changed synthetic weights")
    with pytest.raises(WhisperError):
        assets.verify()


def test_executable_config_is_rejected(assets):
    path = assets.directory / "config.json"
    config = json.loads(path.read_text())
    config["auto_map"] = {"model": "untrusted.py"}
    path.write_text(json.dumps(config))
    with pytest.raises(WhisperError):
        WhisperAssets(assets.directory, assets.revision, assets.license_snapshot)
