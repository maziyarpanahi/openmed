"""Mute finalized speech identifiers in one retained, aligned local PCM clip."""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right
from dataclasses import asdict, dataclass
from enum import Enum

from ...audio_resample_plan import plan_audio_resampling
from ..transcript.revision_ledger import REVISION_SCHEMA_VERSION, TranscriptToken
from .audio_retention import AudioRetentionSession
from .withholding_buffer import RedactionMapping, ReleasedSpeechSegment

_MAX_FRAMES = 16 * 1024 * 1024
_ZERO = bytes(8192)
_DIGEST = re.compile(r"[a-f0-9]{64}\Z")
_REF = re.compile(r"[a-f0-9]{32}\Z")


class AudioMutingError(ValueError):
    """Controlled refusal; no speech, samples, paths or identifiers."""


class TimingEvidence(str, Enum):
    """Explicit host evidence about a complete utterance and its source clip."""

    TOKENS = "reliable_tokens"
    SEGMENT = "reliable_whole_segment"
    UNCERTAIN = "uncertain"


def _require(condition: bool) -> None:
    if not condition:
        raise AudioMutingError("audio_muting_unverified")


def _int(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _digest(value: object) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def screened_segment_digest(release: ReleasedSpeechSegment) -> str:
    """Bind the final release mappings and lineage at trusted screening handoff."""
    _require(type(release) is ReleasedSpeechSegment)
    _require(
        type(release.lineage_json) is str
        and len(release.lineage_json) <= 2_097_152
        and release.lineage_json.isascii()
    )
    _require(type(release.mappings) is tuple and len(release.mappings) <= 8193)
    _require(type(release.text) is str and len(release.text) <= 131072)
    _require(
        type(release.segment_ref) is str
        and _REF.fullmatch(release.segment_ref) is not None
    )
    _require(_int(release.revision, 1, 4096))
    _require(_digest(release.request_digest) and _digest(release.output_digest))
    for row in release.mappings:
        _require(type(row) is RedactionMapping and type(row.redacted) is bool)
        _require(
            all(
                _int(i, 0, 131072)
                for i in (
                    row.source_start,
                    row.source_end,
                    row.output_start,
                    row.output_end,
                )
            )
        )
    bound = release.to_dict()
    bound["lineage_sha256"] = hashlib.sha256(
        release.lineage_json.encode("ascii")
    ).hexdigest()
    return hashlib.sha256(
        json.dumps(bound, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class RetainedClipAlignment:
    """Trusted alignment for one complete utterance in interleaved S16LE PCM.

    Capture these digests and coordinates at the trusted recording/screening
    boundary. The target-rate clock begins at stream_start_ms, with no codec
    delay, trimming or gaps. Unresolved offsets require a refusal upstream.
    """

    session_ref: str
    source_sha256: str
    screened_sha256: str
    source_rate_hz: int
    target_rate_hz: int
    source_frames: int
    channels: int
    stream_start_ms: int
    timing: TimingEvidence

    def __post_init__(self) -> None:
        _require(
            type(self.session_ref) is str
            and _REF.fullmatch(self.session_ref) is not None
        )
        _require(_digest(self.source_sha256) and _digest(self.screened_sha256))
        _require(
            _int(self.source_rate_hz, 1, 384000)
            and _int(self.target_rate_hz, 1, 384000)
        )
        _require(_int(self.source_frames, 1, _MAX_FRAMES) and _int(self.channels, 1, 8))
        _require(self.source_frames * self.channels * 2 <= 64 * 1024 * 1024)
        _require(
            _int(self.stream_start_ms, 0, 86400000)
            and type(self.timing) is TimingEvidence
        )


@dataclass(frozen=True, slots=True)
class MutedAudioReport:
    """Source-frame ranges [start, end), counts and digests; no audio or transcript."""

    ranges: tuple[tuple[int, int], ...]
    span_count: int
    muted_frames: int
    source_sha256: str
    output_sha256: str
    screened_sha256: str

    def to_json(self) -> str:
        """Serialize value-free evidence for this derivative."""
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)


def _ranges(
    clip: RetainedClipAlignment, release: ReleasedSpeechSegment, padding_ms: int
) -> tuple[int, tuple[tuple[int, int], ...]]:
    _require(clip.timing is not TimingEvidence.UNCERTAIN)
    _require(screened_segment_digest(release) == clip.screened_sha256)
    _require(hashlib.sha256(release.text.encode()).hexdigest() == release.output_digest)
    lineage = json.loads(release.lineage_json)
    _require(
        type(lineage) is dict
        and lineage.get("schema_version") == REVISION_SCHEMA_VERSION
    )
    _require(
        lineage.get("finalized") is True
        and lineage.get("session_ref") == clip.session_ref
    )
    _require(
        lineage.get("segment_ref") == release.segment_ref
        and type(lineage.get("revision")) is int
        and lineage["revision"] == release.revision
    )
    length = lineage["length"]
    _require(_int(length, 1, 65536))
    spans = []
    source_at = output_at = 0
    for row in release.mappings:
        _require(source_at == row.source_start < row.source_end <= length)
        _require(output_at == row.output_start < row.output_end <= len(release.text))
        if row.redacted:
            _require(release.text[output_at : row.output_end] == "[REDACTED]")
            spans.append((source_at, row.source_end))
        else:
            _require(row.source_end - source_at == row.output_end - output_at)
        source_at, output_at = row.source_end, row.output_end
    _require(source_at == length and output_at == len(release.text))
    if clip.timing is TimingEvidence.SEGMENT:
        return len(spans), ((0, clip.source_frames),) if spans else ()

    records = lineage["tokens"]
    _require(type(records) is list and 0 < len(records) <= 4096)
    tokens = [TranscriptToken(**row) for row in records]
    _require(len({t.token_ref for t in tokens}) == len(tokens))
    plan = plan_audio_resampling(
        clip.source_rate_hz, clip.target_rate_hz, clip.source_frames
    )
    target_frames = plan.target_frames
    _require(target_frames > 0)

    def target(ms: int, end: bool) -> int:
        rel = ms - clip.stream_start_ms
        _require(rel >= 0)
        return (rel * clip.target_rate_hz + (999 if end else 0)) // 1000

    for i, t in enumerate(tokens):
        _require(
            t.end <= length
            and clip.stream_start_ms <= t.start_ms < t.end_ms
            and target(t.end_ms, True) <= target_frames
        )
        if i:
            left = tokens[i - 1]
            _require(
                left.end <= t.start
                and left.start_ms <= t.start_ms
                and left.end_ms <= t.end_ms
            )
    ends = [t.end for t in tokens]
    padding = (padding_ms * clip.source_rate_hz + 999) // 1000
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        i = bisect_right(ends, start)
        _require(i < len(tokens) and tokens[i].start <= start)
        first = last = tokens[i]
        while last.end < end:
            i += 1
            _require(i < len(tokens) and tokens[i].start == last.end)
            last = tokens[i]
        first_frame = (
            target(first.start_ms, False) * clip.source_rate_hz // clip.target_rate_hz
        )
        end_frame = -(
            -target(last.end_ms, True) * clip.source_rate_hz // clip.target_rate_hz
        )
        _require(0 <= first_frame < end_frame and first_frame < clip.source_frames)
        # A rounded-up target frame can extend beyond the last real source frame.
        lo, hi = (
            max(0, first_frame - padding),
            min(clip.source_frames, end_frame + padding),
        )
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return len(spans), tuple(merged)


def mute_retained_clip(
    session: AudioRetentionSession,
    source_ref: str,
    destination_ref: str,
    clip: RetainedClipAlignment,
    release: ReleasedSpeechSegment,
    *,
    padding_ms: int,
) -> MutedAudioReport:
    """Create a separately inventoried muted buffer under the source deadline.

    Use only a trusted complete final screening of this exact single-utterance
    clip. The caller owns consent and the quality of acoustic/screening evidence.
    Never pass provisional text or an arbitrary subset of a longer recording.
    The output remains inside the session and is never an export authorization.
    """
    _require(type(clip) is RetainedClipAlignment and _int(padding_ms, 0, 5000))
    try:
        span_count, ranges = _ranges(clip, release, padding_ms)
    except (
        ValueError,
        KeyError,
        AttributeError,
        TypeError,
        OverflowError,
        RecursionError,
    ):
        pass
    else:

        def render(source: memoryview, output: memoryview) -> MutedAudioReport:
            _require(hashlib.sha256(source).hexdigest() == clip.source_sha256)
            # The new buffer starts zeroed. Only copy ranges outside all PHI.
            stride = clip.channels * 2
            cursor = 0
            for lo, hi in ranges:
                a, b = lo * stride, hi * stride
                output[cursor:a] = source[cursor:a]
                _require(output[cursor:a] == source[cursor:a])
                for offset in range(a, b, len(_ZERO)):
                    stop = min(b, offset + len(_ZERO))
                    _require(output[offset:stop] == _ZERO[: stop - offset])
                cursor = b
            output[cursor:] = source[cursor:]
            _require(output[cursor:] == source[cursor:])
            return MutedAudioReport(
                ranges,
                span_count,
                sum(hi - lo for lo, hi in ranges),
                clip.source_sha256,
                hashlib.sha256(output).hexdigest(),
                clip.screened_sha256,
            )

        return session._derive_buffer(
            clip.session_ref,
            source_ref,
            destination_ref,
            clip.source_frames * clip.channels * 2,
            render,
        )
    raise AudioMutingError("audio_muting_unverified")
