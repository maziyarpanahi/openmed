"""Normalize a short local PCM WAV to metadata-free 16 kHz mono in memory."""

from __future__ import annotations

import io
import math
import struct
import wave

from openmed.multimodal.asr_audio_profile import (
    AsrAudioProfile,
    AsrCompatibility,
    check_asr_compatibility,
)
from openmed.multimodal.audio_resample_plan import plan_audio_resampling
from openmed.multimodal.wav_metadata import read_wav_metadata

from .decision_media import _admit, _need

_PROFILE = AsrAudioProfile(
    "decision_d1", (1,), (16000,), (1,), (16,), max_duration_seconds=30
)
_RATES = frozenset({8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000})


def normalize_wav(data: bytes, digest: str) -> bytes:
    """Check the capture length and format before allocating or transforming PCM."""
    meta = read_wav_metadata(data)
    _need(0 < meta.frame_count <= 30 * meta.sample_rate_hz, "decision_audio_duration")
    _need(
        meta.channels in {1, 2}
        and meta.sample_rate_hz in _RATES
        and meta.bit_depth == 16
        and meta.format_code == 1,
        "decision_audio_format",
    )
    report = check_asr_compatibility(meta, _PROFILE)
    _need(
        report.compatibility
        in {
            AsrCompatibility.COMPATIBLE,
            AsrCompatibility.RESAMPLE,
            AsrCompatibility.DOWNMIX,
        },
        "decision_audio_format",
    )
    _admit(data, "audio/wav", digest, duration_seconds=meta.duration_seconds)
    _need(
        struct.unpack_from("<I", data, 4)[0] + 8 == len(data), "decision_audio_length"
    )
    # Only a single PCM data chunk can contribute samples. Discard known
    # non-audio annotations; reject alternate audio containers or truncated padding.
    pos = 12
    pcm = None
    seen_fmt = False
    while pos < len(data):
        _need(pos + 8 <= len(data), "decision_audio_length")
        tag, n = struct.unpack_from("<4sI", data, pos)
        start = pos + 8
        pos = start + n + (n % 2)
        _need(pos <= len(data), "decision_audio_length")
        if tag == b"fmt ":
            _need(
                not seen_fmt and pcm is None and n in {16, 18}, "decision_audio_format"
            )
            seen_fmt = True
        elif tag == b"data":
            _need(
                seen_fmt
                and pcm is None
                and n == meta.data_byte_count == meta.frame_count * 2 * meta.channels,
                "decision_audio_length",
            )
            pcm = memoryview(data)[start : start + n]
        else:
            _need(
                tag in {b"JUNK", b"bext", b"iXML"}
                or (tag == b"LIST" and n >= 4 and data[start : start + 4] == b"INFO"),
                "decision_audio_chunk",
            )
    _need(pcm is not None, "decision_audio_length")
    assert pcm is not None
    if meta.sample_rate_hz == 16000 and meta.channels == 1:
        samples = pcm.tobytes()
    else:
        import numpy as np

        arr = np.frombuffer(pcm, dtype="<i2").reshape(-1, meta.channels).mean(axis=1)
        if meta.sample_rate_hz != 16000:
            from scipy.signal import resample_poly

            plan = plan_audio_resampling(meta.sample_rate_hz, 16000, meta.frame_count)
            _need(0 < plan.target_frames <= 480000, "decision_audio_duration")
            gcd = math.gcd(16000, meta.sample_rate_hz)
            arr = resample_poly(arr, 16000 // gcd, meta.sample_rate_hz // gcd)[
                : plan.target_frames
            ]
            _need(len(arr) == plan.target_frames, "decision_audio_resample")
        samples = np.rint(arr).clip(-32768, 32767).astype("<i2").tobytes()
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        writer.writeframes(samples)
    return out.getvalue()
