"""Opt-in, real-device qualification of the local ASR streaming boundary.

The default suite never downloads weights, renders speech or imports MLX. Set
OPENMED_ASR_PROFILE=1 and OPENMED_ASR_MODEL_DIR to an existing pinned directory.
Run under an OS network-deny policy for the offline qualification receipt.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import threading
import time
import wave
from collections import defaultdict
from pathlib import Path

import pytest

from openmed.eval.transcript_accuracy import score_transcript
from openmed.mlx.models.whisper import WhisperAssets
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
from openmed.multimodal.streaming.privacy.withholding_buffer import (
    PhiRedactionSpan,
    PhiResolution,
    PhiScreenDeclaration,
    PhiScreenResult,
    SegmentPhiDecision,
    WithholdingPolicy,
)
from openmed.multimodal.streaming.providers import (
    OfflineAsrError,
    OfflineAsrPolicy,
    OfflineWhisperSession,
)
from openmed.multimodal.streaming.speech.providers import AudioChunk
from tests.fixtures.multimodal.synthetic_speech import SPEECH_CASES

pytestmark = [pytest.mark.integration, pytest.mark.slow]
PROFILE = Path(__file__).parents[2] / "fixtures/multimodal/offline_asr_profile.json"


class FixtureScreen:
    """Known-fixture whole-utterance suppression, never a production PHI detector.

    The profile measures whether a real ASR output can escape a screen decision,
    including token timings and previews. It does not measure detector recall.
    """

    declaration = PhiScreenDeclaration(
        artifact_digest=hashlib.sha256(
            b"synthetic-full-utterance-screen-v1"
        ).hexdigest()
    )
    suppress = False

    def screen(self, request):
        return PhiScreenResult(
            request_digest=request.request_digest,
            decisions=tuple(
                SegmentPhiDecision(
                    segment_ref=segment.segment_ref,
                    revision=segment.revision,
                    resolution=PhiResolution.RESOLVED,
                    redactions=(PhiRedactionSpan(0, len(segment.text)),)
                    if self.suppress
                    else (),
                )
                for segment in request.segments
            ),
        )


def _mac_observer(states):
    # Read the platform's actual thermal state and VM-pressure level. Missing
    # observations fail the profile rather than being recorded as nominal.
    ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    pointer_send = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc)
    )
    process = pointer_send(
        objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo")
    )
    integer_send = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc)
    )

    def observe():
        thermal = ThermalState(
            integer_send(process, objc.sel_registerName(b"thermalState"))
        )
        pressure = int(
            subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                text=True,
            )
        )
        memory = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
        page_size = int(re.search(r"page size of (\d+) bytes", memory).group(1))
        pages = {
            name: int(value)
            for name, value in re.findall(r"(Pages [a-z ]+):\s+(\d+)\.", memory)
        }
        # Same reclaimable-page definition used for macOS available RAM; this
        # is headroom, not a claim that inactive pages contain no private data.
        available = (pages["Pages free"] + pages["Pages inactive"]) * page_size
        states.append(int(thermal))
        return PlatformObservation(
            observed_at=time.monotonic(),
            thermal=thermal,
            memory_pressure={
                1: MemoryPressure.NORMAL,
                2: MemoryPressure.WARNING,
                4: MemoryPressure.CRITICAL,
            }[pressure],
            available_bytes=available,
        )

    return observe


def _render(tmp_path):
    rendered = []
    for index, case in enumerate(SPEECH_CASES):
        path = tmp_path / f"synthetic-{index}.wav"
        subprocess.run(
            [
                "/usr/bin/say",
                "-v",
                case.voice,
                "-r",
                "145",
                "-o",
                str(path),
                "--file-format=WAVE",
                "--data-format=LEI16@16000",
            ],
            input=case.reference,
            text=True,
            check=True,
            capture_output=True,
        )
        with wave.open(str(path), "rb") as audio:
            assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
            assert audio.getframerate() == 16_000
            pcm = audio.readframes(audio.getnframes())
        path.unlink()
        # Round to whole milliseconds and add enough trailing silence for
        # growing-window stability, without changing the spoken source.
        pcm += b"\0" * ((-len(pcm)) % 32)
        rendered.append(pcm + b"\0" * (800 * 32))
    return rendered


def _feed(session, pcm, sequence):
    for offset in range(0, len(pcm), 16_000):
        block = pcm[offset : offset + 16_000]
        session.push_chunk(AudioChunk(sequence, block, 16_000, 1, len(block) // 32))
        sequence += 1
    return sequence


def _p95(values):
    return sorted(values)[math.ceil(len(values) * 0.95) - 1]


def test_streaming_asr_provider(tmp_path, monkeypatch):
    if os.environ.get("OPENMED_ASR_PROFILE") != "1":
        pytest.skip("optional local ASR device profile disabled")
    assert platform.system() == "Darwin" and platform.machine() == "arm64"
    directory = Path(os.environ["OPENMED_ASR_MODEL_DIR"])
    profile_bytes = PROFILE.read_bytes()
    profile = json.loads(profile_bytes)
    assets = WhisperAssets(
        directory,
        profile["revision"],
        (directory / "UPSTREAM_MODEL_CARD.md").read_bytes(),
    )
    with (directory / "weights.safetensors").open("rb") as handle:
        assert (
            hashlib.file_digest(handle, "sha256").hexdigest()
            == profile["weight_sha256"]
        )
    rendered = _render(tmp_path)
    network_attempts = []

    def offline(*args, **kwargs):
        network_attempts.append(True)
        raise RuntimeError("network_disabled")

    monkeypatch.setattr(socket, "create_connection", offline)
    monkeypatch.setattr(socket, "getaddrinfo", offline)
    monkeypatch.setattr(socket.socket, "connect", offline)
    monkeypatch.setattr(socket.socket, "connect_ex", offline)
    states = []
    observe = _mac_observer(states)
    screen = FixtureScreen()
    policy = OfflineAsrPolicy(
        max_session_ms=600_000, max_operations=4096, decode_timeout_ms=45_000
    )
    latencies, factors, scores, hypotheses = (
        [],
        [],
        defaultdict(list),
        defaultdict(list),
    )
    durations = []
    released_count = identifier_leaks = unknown_speakers = identifier_cases = 0
    started = time.monotonic()
    with OfflineWhisperSession(
        assets,
        screen,
        languages=tuple(profile["languages"]),
        observe=observe,
        policy=policy,
        withholding_policy=WithholdingPolicy(
            max_latency_ms=60_000, trailing_segments=0
        ),
    ) as session:
        sequence = completed = 0
        while (
            completed < len(SPEECH_CASES) * profile["repeats"]
            or time.monotonic() - started < profile["minimum_sustained_seconds"]
        ):
            index = completed % len(SPEECH_CASES)
            case, pcm = SPEECH_CASES[index], rendered[index]
            screen.suppress = case.contains_identifier
            initial, tail = pcm[: -400 * 32], pcm[-400 * 32 :]
            sequence = _feed(session, initial, sequence)
            before = time.monotonic()
            partial = session.partial()
            elapsed = time.monotonic() - before
            latencies.append(elapsed)
            factors.append(elapsed / (len(initial) / 32_000))
            assert partial.reason == "partial_withheld" and not partial.released, (
                session.report()
            )
            sequence = _feed(session, tail, sequence)
            duration = len(pcm) // 32
            # Exact synthetic rendered speech/silence intervals, not a claim to
            # qualify an automatic VAD on real recordings.
            activity = VoiceActivityTimeline(
                duration,
                (
                    VoiceActivityInterval(0, duration - 800, VoiceActivityKind.SPEECH),
                    VoiceActivityInterval(
                        duration - 800, duration, VoiceActivityKind.SILENCE
                    ),
                ),
            )
            before = time.monotonic()
            update = session.end_utterance(activity)
            elapsed = time.monotonic() - before
            latencies.append(elapsed)
            factors.append(elapsed / (duration / 1000))
            assert len(update.released) == 1, session.report()
            segment = update.released[0]
            assert segment.language.language == case.language
            assert segment.token_times and all(
                t.token == "" for t in segment.token_times
            )
            unknown_speakers += segment.speaker == "unknown"
            output = segment.reviewed_text(reviewer_confirmed=True)
            if case.contains_identifier:
                identifier_cases += 1
                identifier_leaks += bool(output.replace("[REDACTED]", "").strip())
            else:
                score = score_transcript(case.reference, output, language=case.language)
                scores[case.language].append(score.words)
                hypotheses[index].append(hashlib.sha256(output.encode()).hexdigest())
            released_count += 1
            completed += 1
            durations.append(duration)
        assert session.finish().reason == "finalized"
        report = session.report()
        assert (
            report["closed"]
            and report["buffered_bytes"] == 0
            and not report["cleanup_failed"]
        )

    # Real-process cancellation and resource refusal use the same adapter and
    # model. The serious thermal observation is explicitly injected, separate
    # from the physical states recorded above.
    with OfflineWhisperSession(
        assets,
        screen,
        languages=tuple(profile["languages"]),
        observe=observe,
        policy=policy,
    ) as cancelled:
        _feed(cancelled, rendered[0], 0)
        outputs = []
        worker = threading.Thread(target=lambda: outputs.append(cancelled.partial()))
        worker.start()
        # Wait for the adapter to dispatch its native call, with a harness
        # deadline to fail a hung test rather than hiding an implementation bug.
        deadline = time.monotonic() + 5
        while not cancelled._busy and worker.is_alive() and time.monotonic() < deadline:
            time.sleep(0.005)
        before = time.monotonic()
        cancelled.cancel()
        worker.join(5)
        cancel_seconds = time.monotonic() - before
        assert not worker.is_alive() and not any(cancelled._window)
        assert not cancelled.report()["cleanup_failed"]
        assert outputs and not outputs[0].released
    with OfflineWhisperSession(
        assets,
        screen,
        languages=tuple(profile["languages"]),
        observe=observe,
        policy=policy,
    ) as degraded:
        _feed(degraded, rendered[0][:16_000], 0)
        degraded._observe = lambda: PlatformObservation(
            observed_at=time.monotonic(),
            thermal=ThermalState.SERIOUS,
            memory_pressure=MemoryPressure.NORMAL,
            available_bytes=10 * 1024**3,
        )
        with pytest.raises(OfflineAsrError):
            _feed(degraded, rendered[0][:16_000], 1)
        assert degraded.report()["closed"] and not any(degraded._window)
        assert not degraded.report()["cleanup_failed"]

    words = [row for rows in scores.values() for row in rows]
    # Reproduce a host that imports Hub before the worker's environment flags
    # are set. This separate script is re-imported by multiprocessing spawn.
    probe = tmp_path / "preloaded_hub.py"
    probe.write_text("""from pathlib import Path
import sys
from huggingface_hub import constants
from mlx_whisper import load_models
from openmed.mlx.models.whisper import WhisperAssets, WhisperProcess

constants.HF_HUB_OFFLINE = False
original_load = load_models.load_model

def checked_load(*args, **kwargs):
    assert constants.HF_HUB_OFFLINE is True
    return original_load(*args, **kwargs)

load_models.load_model = checked_load
if __name__ == "__main__":
    root = Path(sys.argv[1])
    assets = WhisperAssets(root, sys.argv[2], (root / "UPSTREAM_MODEL_CARD.md").read_bytes())
    worker = WhisperProcess(assets)
    worker.close()
""")
    cached_mode = subprocess.run(
        [sys.executable, str(probe), str(directory), assets.revision],
        capture_output=True,
        timeout=120,
    )
    assert cached_mode.returncode == 0, "preloaded_hub_offline_mode_failed"
    wer = sum(row.errors for row in words) / sum(row.reference for row in words)
    per_language = {
        language: sum(row.errors for row in rows) / sum(row.reference for row in rows)
        for language, rows in scores.items()
    }
    receipt = {
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "model_fingerprint": assets.fingerprint,
        "model_revision": assets.revision,
        "license_snapshot_sha256": assets.license_digest,
        "hardware": platform.machine(),
        "os": platform.mac_ver()[0],
        "python": platform.python_version(),
        "utterances": completed,
        "synthetic_audio_seconds": sum(durations) / 1000,
        "elapsed_seconds": time.monotonic() - started,
        "word_error_rate": wer,
        "per_language_word_error_rate": per_language,
        "identifier_cases": identifier_cases,
        "identifier_leaks": identifier_leaks,
        "unknown_speaker_fraction": unknown_speakers / completed,
        "release_fraction": released_count / completed,
        "first_decode_seconds": latencies[0],
        "warm_decode_p95_seconds": _p95(latencies[1:]),
        "warm_decode_p95_real_time_factor": _p95(factors[1:]),
        "physical_thermal_states": sorted(set(states)),
        "injected_serious_thermal_cleanup": True,
        "preloaded_hub_offline_enforced": True,
        "cancel_seconds": cancel_seconds,
        "network_attempts": len(network_attempts),
        "repeat_exact_match_fraction": sum(
            len(set(values)) == 1 for values in hypotheses.values()
        )
        / len(hypotheses),
        "runtime": report,
    }
    destination = os.environ.get("OPENMED_ASR_PROFILE_REPORT")
    if destination:
        Path(destination).write_text(json.dumps(receipt, indent=2) + "\n")
    assert wer <= profile["maximum_aggregate_clean_wer"]
    assert all(
        value <= profile["maximum_per_language_clean_wer"]
        for value in per_language.values()
    )
    assert receipt["release_fraction"] >= profile["minimum_clean_release_fraction"]
    assert identifier_leaks <= profile["maximum_identifier_leaks"]
    assert (
        receipt["unknown_speaker_fraction"]
        >= profile["minimum_unknown_speaker_fraction"]
    )
    assert latencies[0] <= profile["maximum_first_decode_seconds"]
    assert (
        receipt["warm_decode_p95_seconds"] <= profile["maximum_warm_decode_p95_seconds"]
    )
    assert (
        receipt["warm_decode_p95_real_time_factor"]
        <= profile["maximum_warm_decode_p95_real_time_factor"]
    )
    assert (
        report["peak_worker_rss_bytes"] / 1024**3 <= profile["maximum_worker_rss_gib"]
    )
    assert report["peak_device_bytes"] / 1024**3 <= profile["maximum_device_gib"]
    assert cancel_seconds <= profile["maximum_cancel_seconds"]
    assert len(network_attempts) <= profile["maximum_network_attempts"]
