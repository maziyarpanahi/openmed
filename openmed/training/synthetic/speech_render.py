"""Offline synthetic dialogue rendering with sample-exact PHI chunk truth."""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
import struct
import wave
from dataclasses import asdict, dataclass, field
from typing import Any, BinaryIO, Protocol, TextIO

from .spdx_identifier import normalize_spdx_identifier

__all__ = [
    "Augmentations",
    "GoldPhiSpan",
    "ModelFreeSynthesizer",
    "ScriptedTurn",
    "SpeechRenderError",
    "SpeechRenderManifest",
    "SpeechSynthesizer",
    "SynthesizedChunk",
    "SynthesizerDeclaration",
    "WordAlignment",
    "render_dialogue",
]

# Match the in-process dependency policy; do not infer permissive terms from
# arbitrary declarations or accept custom LicenseRef expressions.
_LICENSES = frozenset(
    {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "0BSD"}
)
_WORDS = re.compile(r"\S+")
_NOTICE = "Synthetic test audio only; non-diagnostic; no clinical validation."


class SpeechRenderError(ValueError):
    """Controlled, value-free rendering failure."""


@dataclass(frozen=True)
class GoldPhiSpan:
    """Gold PHI interval in Python Unicode character offsets, end exclusive."""

    start: int
    end: int


@dataclass(frozen=True)
class ScriptedTurn:
    """Caller-supplied synthetic turn; source values are excluded from repr."""

    speaker_role: str = field(repr=False)
    language: str = field(repr=False)
    text: str = field(repr=False)
    phi_spans: tuple[GoldPhiSpan, ...] = ()


@dataclass(frozen=True)
class SynthesizerDeclaration:
    """Adapter declarations checked before any synthesis call.

    Args:
        engine_license: Permissive SPDX identifier for the engine.
        voice_license: Permissive SPDX identifier for the voice assets.
        offline: True only for local execution without network access.
        synthetic_voice: True only for voices that do not clone or imitate people.
    """

    engine_license: str = field(repr=False)
    voice_license: str = field(repr=False)
    offline: bool = True
    synthetic_voice: bool = True


@dataclass(frozen=True)
class WordAlignment:
    """One whitespace token's character and sample bounds within a chunk."""

    char_start: int
    char_end: int
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class SynthesizedChunk:
    """Mono little-endian signed PCM16 and measured local word alignments."""

    pcm16: bytes = field(repr=False)
    words: tuple[WordAlignment, ...]


class SpeechSynthesizer(Protocol):
    """Injected local engine with caller-reviewed licensing and word alignment.

    Adapters must not perform network IO, clone voices, log source text, or
    retain payloads. The renderer validates declarations, not engine behavior.
    """

    declaration: SynthesizerDeclaration

    def synthesize(
        self, text: str, *, speaker_role: str, language: str, sample_rate_hz: int
    ) -> SynthesizedChunk:
        """Return PCM16 and exact whitespace-token alignments for this chunk."""
        ...


class ModelFreeSynthesizer:
    """Deterministic waveform fixture for plumbing, never intelligible speech.

    This engine has no model or voice assets. Its waveforms cannot qualify an
    ASR provider or establish clinical quality.
    """

    declaration = SynthesizerDeclaration("Apache-2.0", "Apache-2.0")

    def synthesize(
        self, text: str, *, speaker_role: str, language: str, sample_rate_hz: int
    ) -> SynthesizedChunk:
        """Generate deterministic token waveforms with construction-time truth."""
        pcm = bytearray()
        words = []
        for match in _WORDS.finditer(text):
            start = len(pcm) // 2
            length = max(1, len(match.group()) * sample_rate_hz // 100)
            digest = hashlib.sha256(
                json.dumps([speaker_role, language, match.group()]).encode()
            ).digest()
            for index in range(length):
                value = (digest[index % len(digest)] - 128) * 32
                pcm.extend(struct.pack("<h", value))
            words.append(
                WordAlignment(match.start(), match.end(), start, start + length)
            )
        return SynthesizedChunk(bytes(pcm), tuple(words))


@dataclass(frozen=True)
class Augmentations:
    """Seeded amplitude noise, gain, turn overlap and fixed FIR band limiting.

    Args:
        seed: Integer seed for an isolated PRNG.
        noise_amplitude: Uniform integer PCM noise bound (0 disables noise).
        gain: Linear gain before clipping to PCM16.
        overlap_samples: Later turns start this far before the preceding end.
        band_limit: Apply centered five-tap bandpass [-1, 0, 2, 0, -1] / 4.
    """

    seed: int = 0
    noise_amplitude: int = 0
    gain: float = 1.0
    overlap_samples: int = 0
    band_limit: bool = False


@dataclass(frozen=True)
class SpeechRenderManifest:
    """Content-free immutable manifest with deterministic serialization."""

    _json: str = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible manifest dictionary."""
        return json.loads(self._json)

    def to_json(self) -> str:
        """Return canonical JSON without transcript or audio payloads."""
        return self._json


def _digest(value: bytes | str) -> str:
    return hashlib.sha256(
        value.encode("utf-8") if isinstance(value, str) else value
    ).hexdigest()


def _integer(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _validate_turn(turn: ScriptedTurn) -> None:
    if not isinstance(turn, ScriptedTurn):
        raise SpeechRenderError("speech_script_invalid")
    if any(
        type(value) is not str or not value
        for value in (turn.text, turn.language, turn.speaker_role)
    ):
        raise SpeechRenderError("speech_script_invalid")
    if (
        len(turn.text) > 100_000
        or not turn.text.strip()
        or type(turn.phi_spans) is not tuple
    ):
        raise SpeechRenderError("speech_script_invalid")
    end = 0
    for span in turn.phi_spans:
        if (
            not isinstance(span, GoldPhiSpan)
            or not _integer(span.start, end, len(turn.text) - 1)
            or not _integer(span.end, span.start + 1, len(turn.text))
            or not turn.text[span.start : span.end].strip()
        ):
            raise SpeechRenderError("speech_phi_span_invalid")
        end = span.end


def _licenses(synthesizer: SpeechSynthesizer) -> tuple[str, str]:
    try:
        declaration = synthesizer.declaration
    except Exception:
        pass
    else:
        if isinstance(declaration, SynthesizerDeclaration):
            engine = normalize_spdx_identifier(declaration.engine_license).normalized
            voice = normalize_spdx_identifier(declaration.voice_license).normalized
            if engine not in _LICENSES or voice not in _LICENSES:
                raise SpeechRenderError("speech_license_rejected")
            if (
                declaration.offline is not True
                or declaration.synthetic_voice is not True
            ):
                raise SpeechRenderError("speech_adapter_policy_rejected")
            return engine, voice
    raise SpeechRenderError("speech_declaration_invalid")


def _synthesize(
    synthesizer: SpeechSynthesizer,
    text: str,
    turn: ScriptedTurn,
    rate: int,
    remaining: int,
) -> SynthesizedChunk:
    # Raise outside the exception handler so a provider's PHI-bearing exception
    # is not retained in the public exception context or cause.
    try:
        result = synthesizer.synthesize(
            text,
            speaker_role=turn.speaker_role,
            language=turn.language,
            sample_rate_hz=rate,
        )
    except Exception:
        pass
    else:
        if (
            not isinstance(result, SynthesizedChunk)
            or type(result.pcm16) is not bytes
            or len(result.pcm16) % 2
            or not 0 < len(result.pcm16) // 2 <= remaining
            or type(result.words) is not tuple
        ):
            raise SpeechRenderError("speech_chunk_invalid")
        tokens = list(_WORDS.finditer(text))
        if len(tokens) != len(result.words):
            raise SpeechRenderError("speech_alignment_invalid")
        end = 0
        for token, word in zip(tokens, result.words):
            if (
                not isinstance(word, WordAlignment)
                or type(word.char_start) is not int
                or type(word.char_end) is not int
                or (word.char_start, word.char_end) != (token.start(), token.end())
                or not _integer(word.start_sample, end, len(result.pcm16) // 2 - 1)
                or not _integer(
                    word.end_sample, word.start_sample + 1, len(result.pcm16) // 2
                )
            ):
                raise SpeechRenderError("speech_alignment_invalid")
            end = word.end_sample
        return result
    raise SpeechRenderError("speech_synthesis_failed")


def _interval(start: int, end: int, rate: int, **extra: Any) -> dict[str, Any]:
    return {
        **extra,
        "start_sample": start,
        "end_sample": end,
        "start_seconds": start / rate,
        "end_seconds": end / rate,
    }


def _render_turn(
    turn: ScriptedTurn,
    synthesizer: SpeechSynthesizer,
    rate: int,
    gap: int,
    remaining: int,
) -> tuple[bytes, list[dict[str, Any]], list[dict[str, Any]]]:
    pcm = bytearray()
    fragments: list[dict[str, Any]] = []
    phi: list[dict[str, Any]] = []
    ranges: list[tuple[int, int, int | None]] = []
    cursor = 0
    for index, span in enumerate(turn.phi_spans):
        if cursor < span.start:
            ranges.append((cursor, span.start, None))
        ranges.append((span.start, span.end, index))
        cursor = span.end
    if cursor < len(turn.text):
        ranges.append((cursor, len(turn.text), None))
    for char_start, char_end, phi_index in ranges:
        text = turn.text[char_start:char_end]
        if not text.strip():
            continue
        if pcm:
            pcm.extend(b"\0\0" * gap)
        start = len(pcm) // 2
        chunk = _synthesize(synthesizer, text, turn, rate, remaining - start)
        pcm.extend(chunk.pcm16)
        for word in chunk.words:
            fragments.append(
                _interval(
                    start + word.start_sample,
                    start + word.end_sample,
                    rate,
                    char_start=char_start + word.char_start,
                    char_end=char_start + word.char_end,
                )
            )
        if phi_index is not None:
            phi.append(
                _interval(
                    start,
                    len(pcm) // 2,
                    rate,
                    phi_index=phi_index,
                    char_start=char_start,
                    char_end=char_end,
                    text_sha256=_digest(text),
                    source_pcm_sha256=_digest(chunk.pcm16),
                )
            )
    # A PHI boundary can split a whitespace token (e.g. Ada in "Ada,").
    # Keep the true fragment intervals, and envelope them under the original
    # script's token. Never estimate intra-word timings by character lengths.
    words = []
    fragment_cursor = 0
    for index, token in enumerate(_WORDS.finditer(turn.text)):
        parts = []
        while (
            fragment_cursor < len(fragments)
            and fragments[fragment_cursor]["char_start"] < token.end()
        ):
            part = fragments[fragment_cursor]
            if part["char_end"] > token.start():
                parts.append(part)
            fragment_cursor += 1
        words.append(
            _interval(
                parts[0]["start_sample"],
                parts[-1]["end_sample"],
                rate,
                word_index=index,
                char_start=token.start(),
                char_end=token.end(),
                text_sha256=_digest(token.group()),
                fragments=parts,
            )
        )
    return bytes(pcm), words, phi


def _augment(samples: list[int], options: Augmentations) -> tuple[bytes, int]:
    rng = random.Random(options.seed)
    values = [
        value * options.gain
        + rng.randint(-options.noise_amplitude, options.noise_amplitude)
        for value in samples
    ]
    if options.band_limit:
        # Centered FIR: no timestamp delay, support expands two frames each way.
        values = [
            (
                2 * value
                - (values[index - 2] if index >= 2 else 0)
                - (values[index + 2] if index + 2 < len(values) else 0)
            )
            / 4
            for index, value in enumerate(values)
        ]
    clipped = sum(value < -32768 or value > 32767 for value in values)
    return b"".join(
        struct.pack("<h", max(-32768, min(32767, round(value)))) for value in values
    ), clipped


def render_dialogue(
    turns: tuple[ScriptedTurn, ...],
    synthesizer: SpeechSynthesizer,
    wav_output: BinaryIO,
    *,
    manifest_output: TextIO | None = None,
    sample_rate_hz: int = 16_000,
    chunk_gap_samples: int = 160,
    turn_gap_samples: int = 1_600,
    augmentations: Augmentations = Augmentations(),
    max_frames: int = 9_600_000,
) -> SpeechRenderManifest:
    """Write synthetic mono PCM16 WAV and content-free timing truth locally.

    PHI spans are synthesized separately; no full-turn waveform is substituted
    for their exact chunks. All scripts and declarations are checked before
    synthesis. Output streams are caller-owned; no paths or temp files are used.

    Args:
        turns: One to 100 synthetic scripted turns with sorted disjoint PHI spans.
        synthesizer: Injected offline adapter with permissive declarations.
        wav_output: Binary destination for the WAV, starting at its current position.
        manifest_output: Optional text destination for canonical manifest JSON.
        sample_rate_hz: Declared sample rate between 8,000 and 48,000 Hz.
        chunk_gap_samples: Silence between nonempty separately synthesized chunks.
        turn_gap_samples: Silence between turns when overlap is disabled.
        augmentations: Reproducible labelled augmentation parameters.
        max_frames: Bound on both total synthesis and final timeline (at most 9.6M).

    Returns:
        Manifest containing source and rendered digests, timings and safety notices.

    Raises:
        SpeechRenderError: Controlled validation, provider or output failure code.
    """
    engine, voice = _licenses(synthesizer)
    if type(turns) is not tuple or not 1 <= len(turns) <= 100:
        raise SpeechRenderError("speech_script_invalid")
    for turn in turns:
        _validate_turn(turn)
    options = augmentations
    if (
        not _integer(sample_rate_hz, 8_000, 48_000)
        or not _integer(max_frames, 1, 9_600_000)
        or not _integer(chunk_gap_samples, 0, max_frames)
        or not _integer(turn_gap_samples, 0, max_frames)
        or not isinstance(options, Augmentations)
        or not _integer(options.seed, 0, 2**64 - 1)
        or not _integer(options.noise_amplitude, 0, 32767)
        or type(options.gain) not in (int, float)
        or not 0 < options.gain <= 16
        or not math.isfinite(options.gain)
        or not _integer(options.overlap_samples, 0, max_frames)
        or type(options.band_limit) is not bool
    ):
        raise SpeechRenderError("speech_options_invalid")
    samples: list[int] = []
    records = []
    total_synthesized = 0
    previous_start = 0
    previous_end = 0
    nominal_start = 0
    previous_speaker = None
    for index, turn in enumerate(turns):
        if index and options.overlap_samples:
            if (
                turn.speaker_role == previous_speaker
                or options.overlap_samples > previous_end - previous_start
            ):
                raise SpeechRenderError("speech_overlap_invalid")
            start = previous_end - options.overlap_samples
        else:
            start = previous_end + (turn_gap_samples if index else 0)
        pcm, words, phi = _render_turn(
            turn,
            synthesizer,
            sample_rate_hz,
            chunk_gap_samples,
            min(max_frames - total_synthesized, max_frames - start),
        )
        length = len(pcm) // 2
        total_synthesized += length
        end = start + length
        if end > max_frames:
            raise SpeechRenderError("speech_frame_limit")
        samples.extend([0] * max(0, end - len(samples)))
        for offset, (value,) in enumerate(struct.iter_unpack("<h", pcm)):
            samples[start + offset] += value
        for interval in words + phi:
            for part in [interval] + interval.get("fragments", []):
                part.update(
                    _interval(
                        part["start_sample"] + start,
                        part["end_sample"] + start,
                        sample_rate_hz,
                    )
                )
        records.append(
            _interval(
                start,
                end,
                sample_rate_hz,
                turn_index=index,
                speaker_sha256=_digest(turn.speaker_role),
                language_sha256=_digest(turn.language),
                text_sha256=_digest(turn.text),
                source_pcm_sha256=_digest(pcm),
                timeline_shift_samples=start - nominal_start,
                words=words,
                phi=phi,
            )
        )
        nominal_start += length + turn_gap_samples
        previous_start, previous_end, previous_speaker = start, end, turn.speaker_role
    pcm, clipped = _augment(samples, options)
    halo = 2 if options.band_limit else 0
    for record in records:
        for interval in [record] + record["words"] + record["phi"]:
            start, end = interval["start_sample"], interval["end_sample"]
            interval["rendered_pcm_sha256"] = _digest(pcm[start * 2 : end * 2])
            interval["affected_start_sample"] = max(0, start - halo)
            interval["affected_end_sample"] = min(len(samples), end + halo)
        record["overlapping_turns"] = [
            other["turn_index"]
            for other in records
            if other is not record
            and other["start_sample"] < record["end_sample"]
            and other["end_sample"] > record["start_sample"]
        ]
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm)
    wav_bytes = output.getvalue()
    manifest = SpeechRenderManifest(
        json.dumps(
            {
                "schema_version": 1,
                "sample_rate_hz": sample_rate_hz,
                "channels": 1,
                "bit_depth": 16,
                "frame_count": len(samples),
                "sequential_frame_count": nominal_start - turn_gap_samples,
                "frame_count_delta": len(samples) - (nominal_start - turn_gap_samples),
                "wav_sha256": _digest(wav_bytes),
                "pcm_sha256": _digest(pcm),
                "engine_license": engine,
                "voice_license": voice,
                "chunk_gap_samples": chunk_gap_samples,
                "turn_gap_samples": turn_gap_samples,
                "augmentations": {
                    **asdict(options),
                    "labels": [
                        label
                        for enabled, label in (
                            (options.noise_amplitude, "uniform_noise"),
                            (options.gain != 1, "gain"),
                            (options.overlap_samples, "speaker_overlap"),
                            (options.band_limit, "fir_bandpass_5tap"),
                        )
                        if enabled
                    ],
                    "clipped_samples": clipped,
                    "support_expansion_samples": halo,
                    "amplitude_transforms_preserve_frame_count": True,
                },
                "turns": records,
                "synthetic_fixture": True,
                "non_diagnostic_notice": _NOTICE,
                "reviewer_confirmation_required": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    try:
        written = wav_output.write(wav_bytes)
        if written != len(wav_bytes):
            raise OSError
        if manifest_output is not None and manifest_output.write(
            manifest.to_json()
        ) != len(manifest.to_json()):
            raise OSError
    except Exception:
        pass
    else:
        return manifest
    raise SpeechRenderError("speech_output_failed")
