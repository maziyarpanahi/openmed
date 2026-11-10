"""Flag ASR hypotheses that are not supported by speech or that loop.

Local encoder-decoder speech models can emit fluent text over silence, noise
or music and can fall into repeated-phrase loops. This module checks fixed
provider outputs against an injected voice-activity timeline and
provider-reported statistics under a versioned threshold profile. It runs no
ASR, no voice-activity detection and no window stitching.

Hypothesis text and token identifiers are read only to count repetition
units. They are never stored in a decision, a report, a ``repr`` or an error:
records carry segment identifiers, half-open millisecond offsets, closed
reason codes, numeric scores and SHA-256 digests only.

A segment whose outcome is not ``supported`` is a non-final review candidate.
A later draft or PHI-withholding stage must call :func:`require_draft_eligible`
before citing a segment, which keeps flagged and under-evidenced segments out
of drafts until a reviewer or a later revision resolves them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .abstention import AbstentionReason, AbstentionRecord, AbstentionStage
from .audio_windows import MAX_AUDIO_DURATION_MS
from .provider_result import ProviderResultEnvelope, ProviderResultOutcome

ASR_HALLUCINATION_SCHEMA_VERSION: Final[str] = "openmed.multimodal.asr_hallucination.v1"
MAX_VOICE_ACTIVITY_INTERVALS: Final[int] = 100_000
MAX_HYPOTHESIS_SEGMENTS: Final[int] = 100_000
MAX_SEGMENT_UNIT_COUNT: Final[int] = 100_000
MAX_HYPOTHESIS_CHARS: Final[int] = 1_000_000
MAX_TOKEN_ID: Final[int] = (1 << 32) - 1

FLAG_REASON_CODES: Final[tuple[str, ...]] = (
    "speech_coverage_low",
    "silence_dominant",
    "noise_dominant",
    "music_dominant",
    "no_speech_probability_high",
    "avg_logprob_low",
    "compression_ratio_high",
    "ngram_loop",
    "repeated_ngram_fraction_high",
    "runaway_length",
)
INSUFFICIENT_EVIDENCE_REASON_CODES: Final[tuple[str, ...]] = (
    "vad_coverage_missing",
    "no_speech_probability_missing",
    "avg_logprob_missing",
    "compression_ratio_missing",
    "token_units_missing",
)
HALLUCINATION_REASON_CODES: Final[tuple[str, ...]] = (
    FLAG_REASON_CODES + INSUFFICIENT_EVIDENCE_REASON_CODES
)

_IDENTIFIER_RE: Final = re.compile(r"[a-z0-9](?:[a-z0-9_.-]{0,62}[a-z0-9])?\Z")
_VERSION_RE: Final = re.compile(r"(?:0|[1-9]\d{0,5})(?:\.(?:0|[1-9]\d{0,5})){2}\Z")
_DIGEST_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_WORD_RE: Final = re.compile(r"\w+")
_FLAG_SET: Final = frozenset(FLAG_REASON_CODES)
_INSUFFICIENT_SET: Final = frozenset(INSUFFICIENT_EVIDENCE_REASON_CODES)
_REASON_ORDER: Final = MappingProxyType(
    {code: index for index, code in enumerate(HALLUCINATION_REASON_CODES)}
)
_UNIT_KINDS: Final = frozenset({"token", "word", "none"})
_MIN_AVG_LOGPROB: Final[float] = -100.0
_MAX_RATE: Final[float] = 1_000.0
_MAX_THRESHOLD_COUNT: Final[int] = 1_000_000
_SCORE_DIGITS: Final[int] = 6

_THRESHOLD_FIELDS: Final = (
    "profile_id",
    "version",
    "min_vad_coverage",
    "min_speech_ratio",
    "max_no_speech_probability",
    "min_avg_logprob",
    "max_compression_ratio",
    "loop_max_ngram",
    "min_loop_repeats",
    "min_unigram_loop_repeats",
    "repetition_ngram",
    "min_repetition_units",
    "max_repeated_ngram_fraction",
    "max_units_per_speech_second",
    "max_segment_units",
)
_DECISION_FIELDS: Final = (
    "schema_version",
    "segment_id",
    "start_ms",
    "end_ms",
    "outcome",
    "reason_codes",
    "final_eligible",
    "review_required",
    "unit_kind",
    "unit_count",
    "vad_coverage",
    "speech_ratio",
    "no_speech_probability",
    "avg_logprob",
    "compression_ratio",
    "loop_ngram_size",
    "loop_repeats",
    "repeated_ngram_fraction",
    "units_per_speech_second",
    "thresholds_version",
    "thresholds_digest",
)
_REPORT_FIELDS: Final = (
    "schema_version",
    "thresholds_id",
    "thresholds_version",
    "thresholds_digest",
    "timeline_digest",
    "provider_input_digest",
    "provider_output_digest",
    "decisions",
)


def _activity_kind(value: object) -> VoiceActivityKind:
    if isinstance(value, VoiceActivityKind):
        return value
    if type(value) is str:
        try:
            return VoiceActivityKind(value)
        except ValueError:
            pass
    raise AsrHallucinationError("asr_vad_kind_unsupported")


def _identifier(value: object, category: str) -> None:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise AsrHallucinationError(category)


def _optional_stat(
    value: object, minimum: float, maximum: float, *, low_open: bool
) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float):
        raise AsrHallucinationError("asr_stat_invalid")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or number > maximum or number < minimum:
        raise AsrHallucinationError("asr_stat_invalid")
    if low_open and number == minimum:
        raise AsrHallucinationError("asr_stat_invalid")
    return number


def _threshold_float(value: object, category: str) -> float:
    if type(value) not in (int, float):
        raise AsrHallucinationError(category)
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise AsrHallucinationError(category)
    return number


def _score(value: float) -> float:
    return round(value, _SCORE_DIGITS)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


class AsrHallucinationError(ValueError):
    """Value-free failure with a stable ``category`` code.

    The message equals the category. Submitted values, hypothesis text and
    token identifiers are never included.
    """

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


class VoiceActivityKind(str, Enum):
    """Closed, provider-neutral voice-activity labels.

    Values:
        SPEECH: Human speech was detected.
        SILENCE: No meaningful signal was detected.
        NOISE: Non-speech, non-music sound was detected.
        MUSIC: Music was detected.
    """

    SPEECH = "speech"
    SILENCE = "silence"
    NOISE = "noise"
    MUSIC = "music"


_NON_SPEECH_KINDS: Final = (
    (VoiceActivityKind.SILENCE, "silence_dominant"),
    (VoiceActivityKind.NOISE, "noise_dominant"),
    (VoiceActivityKind.MUSIC, "music_dominant"),
)


class HallucinationOutcome(str, Enum):
    """Closed outcome vocabulary for one hypothesis segment.

    Values:
        SUPPORTED: Every check ran and none fired; the segment is final
            eligible.
        FLAGGED: At least one check fired; the segment needs review.
        INSUFFICIENT_EVIDENCE: No check fired, but required evidence was
            missing, so the segment still needs review. This is never a pass.
    """

    SUPPORTED = "supported"
    FLAGGED = "flagged"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class VoiceActivityInterval:
    """One labeled half-open ``[start_ms, end_ms)`` voice-activity interval.

    Attributes:
        start_ms: Inclusive start in whole milliseconds.
        end_ms: Exclusive end in whole milliseconds; greater than ``start_ms``.
        kind: Voice-activity label for the interval.
    """

    start_ms: int
    end_ms: int
    kind: VoiceActivityKind

    def __post_init__(self) -> None:
        for value in (self.start_ms, self.end_ms):
            if type(value) is not int or not 0 <= value <= MAX_AUDIO_DURATION_MS:
                raise AsrHallucinationError("asr_vad_interval_invalid")
        if self.end_ms <= self.start_ms:
            raise AsrHallucinationError("asr_vad_interval_invalid")
        object.__setattr__(self, "kind", _activity_kind(self.kind))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary of offsets and the label."""

        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "kind": self.kind.value,
        }


@dataclass(frozen=True, slots=True)
class VoiceActivityTimeline:
    """Injected voice-activity timeline from any upstream detector.

    Intervals are sorted, non-overlapping and within ``duration_ms``. Gaps are
    allowed and count as unlabeled time, which lowers voice-activity coverage
    instead of counting as speech.

    Attributes:
        duration_ms: Declared audio duration in whole milliseconds.
        intervals: Labeled intervals as a tuple.
    """

    duration_ms: int
    intervals: tuple[VoiceActivityInterval, ...]

    def __post_init__(self) -> None:
        if (
            type(self.duration_ms) is not int
            or not 0 <= self.duration_ms <= MAX_AUDIO_DURATION_MS
        ):
            raise AsrHallucinationError("asr_vad_duration_invalid")
        if type(self.intervals) is not tuple:
            raise AsrHallucinationError("asr_vad_intervals_invalid")
        if len(self.intervals) > MAX_VOICE_ACTIVITY_INTERVALS:
            raise AsrHallucinationError("asr_vad_interval_limit_exceeded")
        previous_end = 0
        for interval in self.intervals:
            if not isinstance(interval, VoiceActivityInterval):
                raise AsrHallucinationError("asr_vad_intervals_invalid")
            if interval.start_ms < previous_end:
                raise AsrHallucinationError("asr_vad_intervals_unordered")
            if interval.end_ms > self.duration_ms:
                raise AsrHallucinationError("asr_vad_interval_out_of_range")
            previous_end = interval.end_ms

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary of offsets and labels."""

        return {
            "duration_ms": self.duration_ms,
            "intervals": [interval.to_dict() for interval in self.intervals],
        }

    @property
    def digest(self) -> str:
        """Return the SHA-256 hex digest of the canonical timeline JSON."""

        return _sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class AsrSegmentStatistics:
    """Provider-reported statistics for one hypothesis segment.

    Every field is optional because providers differ. A missing value is
    reported as an explicit ``*_missing`` reason, never treated as a pass.

    Attributes:
        no_speech_probability: Provider probability, from 0 through 1, that
            the segment contains no speech.
        avg_logprob: Mean token log-probability, from -100 through 0.
        compression_ratio: Provider-computed text compression ratio, greater
            than 0 and at most 1000.
    """

    no_speech_probability: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "no_speech_probability",
            _optional_stat(self.no_speech_probability, 0.0, 1.0, low_open=False),
        )
        object.__setattr__(
            self,
            "avg_logprob",
            _optional_stat(self.avg_logprob, _MIN_AVG_LOGPROB, 0.0, low_open=False),
        )
        object.__setattr__(
            self,
            "compression_ratio",
            _optional_stat(self.compression_ratio, 0.0, _MAX_RATE, low_open=True),
        )


@dataclass(frozen=True, slots=True)
class AsrHypothesisSegment:
    """One fixed ASR hypothesis segment supplied by a provider boundary.

    ``text`` and ``token_ids`` are used only to count repetition units and
    are excluded from ``repr``. When both are given, ``token_ids`` wins; text
    is split into case-folded ``\\w+`` words.

    Attributes:
        segment_id: Opaque lowercase identifier, unique within one check.
        start_ms: Inclusive segment start in whole milliseconds.
        end_ms: Exclusive segment end; greater than ``start_ms``.
        statistics: Provider statistics, or ``None`` when none were reported.
        text: Hypothesis text, or ``None``. Never serialized.
        token_ids: Provider token identifiers, or ``None``. Never serialized.
    """

    segment_id: str
    start_ms: int
    end_ms: int
    statistics: AsrSegmentStatistics | None = None
    text: str | None = None
    token_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _identifier(self.segment_id, "asr_segment_id_invalid")
        for value in (self.start_ms, self.end_ms):
            if type(value) is not int or not 0 <= value <= MAX_AUDIO_DURATION_MS:
                raise AsrHallucinationError("asr_segment_bounds_invalid")
        if self.end_ms <= self.start_ms:
            raise AsrHallucinationError("asr_segment_bounds_invalid")
        if self.statistics is not None and not isinstance(
            self.statistics, AsrSegmentStatistics
        ):
            raise AsrHallucinationError("asr_segment_statistics_invalid")
        if self.text is not None and (
            type(self.text) is not str or len(self.text) > MAX_HYPOTHESIS_CHARS
        ):
            raise AsrHallucinationError("asr_segment_text_invalid")
        if self.token_ids is not None:
            if (
                type(self.token_ids) is not tuple
                or len(self.token_ids) > MAX_SEGMENT_UNIT_COUNT
            ):
                raise AsrHallucinationError("asr_segment_tokens_invalid")
            for token in self.token_ids:
                if type(token) is not int or not 0 <= token <= MAX_TOKEN_ID:
                    raise AsrHallucinationError("asr_segment_tokens_invalid")

    def __repr__(self) -> str:
        return (
            f"AsrHypothesisSegment(segment_id={self.segment_id!r}, "
            f"start_ms={self.start_ms}, end_ms={self.end_ms}, "
            f"statistics={self.statistics!r}, "
            f"text={'<withheld>' if self.text is not None else None}, "
            f"token_ids={'<withheld>' if self.token_ids is not None else None})"
        )


@dataclass(frozen=True, slots=True)
class HallucinationThresholds:
    """Versioned, digest-bound thresholds for the hallucination check.

    Defaults are conservative starting points, not clinically validated
    cut-offs. A deployment that changes any value must also change
    ``version``; the ``digest`` binds every decision to the exact values used.

    Comparisons: a ratio, probability or rate fires when it is strictly past
    its bound, so a value exactly at the bound passes. A loop fires when the
    consecutive repeat count reaches its minimum.

    Attributes:
        profile_id: Lowercase bounded identifier for the profile.
        version: ``MAJOR.MINOR.PATCH`` version of the threshold values.
        min_vad_coverage: Minimum labeled fraction of the segment.
        min_speech_ratio: Minimum speech fraction of the segment.
        max_no_speech_probability: Highest accepted no-speech probability.
        min_avg_logprob: Lowest accepted mean token log-probability.
        max_compression_ratio: Highest accepted compression ratio.
        loop_max_ngram: Largest n-gram size scanned for tandem loops.
        min_loop_repeats: Consecutive repeats that make an n-gram (n >= 2) a
            loop.
        min_unigram_loop_repeats: Consecutive repeats that make a single unit
            a loop.
        repetition_ngram: N-gram size for the repeated-fraction score.
        min_repetition_units: Fewest units before the repeated fraction is
            scored.
        max_repeated_ngram_fraction: Highest accepted repeated n-gram share.
        max_units_per_speech_second: Highest accepted units per second of
            detected speech.
        max_segment_units: Highest accepted unit count for one segment.
    """

    profile_id: str
    version: str
    min_vad_coverage: float = 0.95
    min_speech_ratio: float = 0.3
    max_no_speech_probability: float = 0.6
    min_avg_logprob: float = -1.0
    max_compression_ratio: float = 2.4
    loop_max_ngram: int = 6
    min_loop_repeats: int = 4
    min_unigram_loop_repeats: int = 6
    repetition_ngram: int = 3
    min_repetition_units: int = 12
    max_repeated_ngram_fraction: float = 0.5
    max_units_per_speech_second: float = 8.0
    max_segment_units: int = 600

    def __post_init__(self) -> None:
        _identifier(self.profile_id, "asr_thresholds_id_invalid")
        if type(self.version) is not str or _VERSION_RE.fullmatch(self.version) is None:
            raise AsrHallucinationError("asr_thresholds_version_invalid")
        for name in (
            "min_vad_coverage",
            "min_speech_ratio",
            "max_no_speech_probability",
            "max_repeated_ngram_fraction",
        ):
            value = _threshold_float(
                getattr(self, name), "asr_thresholds_ratio_invalid"
            )
            if not 0.0 <= value <= 1.0:
                raise AsrHallucinationError("asr_thresholds_ratio_invalid")
            object.__setattr__(self, name, value)
        logprob = _threshold_float(
            self.min_avg_logprob, "asr_thresholds_logprob_invalid"
        )
        if not _MIN_AVG_LOGPROB <= logprob <= 0.0:
            raise AsrHallucinationError("asr_thresholds_logprob_invalid")
        object.__setattr__(self, "min_avg_logprob", logprob)
        for name in ("max_compression_ratio", "max_units_per_speech_second"):
            value = _threshold_float(getattr(self, name), "asr_thresholds_rate_invalid")
            if not 0.0 < value <= _MAX_RATE:
                raise AsrHallucinationError("asr_thresholds_rate_invalid")
            object.__setattr__(self, name, value)
        for name, minimum in (
            ("loop_max_ngram", 1),
            ("min_loop_repeats", 2),
            ("min_unigram_loop_repeats", 2),
            ("repetition_ngram", 1),
            ("min_repetition_units", 1),
            ("max_segment_units", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= _MAX_THRESHOLD_COUNT:
                raise AsrHallucinationError("asr_thresholds_count_invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return the threshold values in declared field order."""

        return {name: getattr(self, name) for name in _THRESHOLD_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with sorted keys for byte-identical payloads."""

        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        """Return the SHA-256 hex digest of :meth:`to_json`."""

        return _sha256_json(self.to_dict())


DEFAULT_HALLUCINATION_THRESHOLDS: Final[HallucinationThresholds] = (
    HallucinationThresholds(
        profile_id="openmed-asr-hallucination-default",
        version="1.0.0",
    )
)
"""Default threshold profile; values are documented, not clinically validated."""


@dataclass(frozen=True, slots=True)
class HallucinationDecision:
    """Value-free decision for one hypothesis segment.

    ``final_eligible`` is true only for a ``supported`` outcome and
    ``review_required`` is its inverse. Construction rejects any combination
    in which the outcome, reason codes and eligibility disagree, so a record
    cannot claim eligibility its reasons do not support.

    Attributes:
        segment_id: Opaque segment identifier.
        start_ms: Inclusive segment start in milliseconds.
        end_ms: Exclusive segment end in milliseconds.
        outcome: Closed outcome for the segment.
        reason_codes: Reason codes in :data:`HALLUCINATION_REASON_CODES` order.
        unit_kind: ``token``, ``word`` or ``none``.
        unit_count: Number of repetition units, or ``None`` when absent.
        vad_coverage: Labeled fraction of the segment, or ``None``.
        speech_ratio: Speech fraction of the segment, or ``None``.
        no_speech_probability: Provider value, or ``None``.
        avg_logprob: Provider value, or ``None``.
        compression_ratio: Provider value, or ``None``.
        loop_ngram_size: N-gram size of the strongest loop, or ``None``.
        loop_repeats: Consecutive repeats of that loop, or ``None``.
        repeated_ngram_fraction: Repeated n-gram share, or ``None``.
        units_per_speech_second: Units per second of speech, or ``None``.
        thresholds_version: Version of the threshold profile used.
        thresholds_digest: Digest of the threshold profile used.
        schema_version: Serialization contract identifier.
    """

    segment_id: str
    start_ms: int
    end_ms: int
    outcome: HallucinationOutcome
    reason_codes: tuple[str, ...]
    unit_kind: str
    unit_count: int | None
    vad_coverage: float | None
    speech_ratio: float | None
    no_speech_probability: float | None
    avg_logprob: float | None
    compression_ratio: float | None
    loop_ngram_size: int | None
    loop_repeats: int | None
    repeated_ngram_fraction: float | None
    units_per_speech_second: float | None
    thresholds_version: str
    thresholds_digest: str
    schema_version: str = ASR_HALLUCINATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ASR_HALLUCINATION_SCHEMA_VERSION:
            raise AsrHallucinationError("asr_decision_schema_unsupported")
        _identifier(self.segment_id, "asr_segment_id_invalid")
        try:
            outcome = HallucinationOutcome(self.outcome)
        except ValueError:
            raise AsrHallucinationError("asr_decision_inconsistent") from None
        codes = self.reason_codes
        if type(codes) is not tuple or any(
            type(code) is not str or code not in _REASON_ORDER for code in codes
        ):
            raise AsrHallucinationError("asr_decision_inconsistent")
        if list(codes) != sorted(set(codes), key=_REASON_ORDER.__getitem__):
            raise AsrHallucinationError("asr_decision_inconsistent")
        if outcome is not _outcome_for(codes):
            raise AsrHallucinationError("asr_decision_inconsistent")
        if self.unit_kind not in _UNIT_KINDS:
            raise AsrHallucinationError("asr_decision_inconsistent")
        if (
            type(self.thresholds_digest) is not str
            or _DIGEST_RE.fullmatch(self.thresholds_digest) is None
        ):
            raise AsrHallucinationError("asr_decision_inconsistent")
        object.__setattr__(self, "outcome", outcome)

    @property
    def final_eligible(self) -> bool:
        """Return whether the segment may become cited draft evidence."""

        return self.outcome is HallucinationOutcome.SUPPORTED

    @property
    def review_required(self) -> bool:
        """Return whether a reviewer or later revision must resolve it."""

        return not self.final_eligible

    def abstention(self) -> AbstentionRecord | None:
        """Return a post-process abstention for a non-final segment.

        Returns:
            ``None`` for a supported segment; otherwise an
            :class:`~openmed.multimodal.abstention.AbstentionRecord` with stage
            ``post_process`` and reason ``low_quality``.
        """

        if self.final_eligible:
            return None
        return AbstentionRecord(
            stage=AbstentionStage.POST_PROCESS,
            reason=AbstentionReason.LOW_QUALITY,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the fixed-shape, value-free dictionary in field order."""

        values: dict[str, Any] = {
            "schema_version": self.schema_version,
            "segment_id": self.segment_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "outcome": self.outcome.value,
            "reason_codes": list(self.reason_codes),
            "final_eligible": self.final_eligible,
            "review_required": self.review_required,
            "unit_kind": self.unit_kind,
            "unit_count": self.unit_count,
            "vad_coverage": self.vad_coverage,
            "speech_ratio": self.speech_ratio,
            "no_speech_probability": self.no_speech_probability,
            "avg_logprob": self.avg_logprob,
            "compression_ratio": self.compression_ratio,
            "loop_ngram_size": self.loop_ngram_size,
            "loop_repeats": self.loop_repeats,
            "repeated_ngram_fraction": self.repeated_ngram_fraction,
            "units_per_speech_second": self.units_per_speech_second,
            "thresholds_version": self.thresholds_version,
            "thresholds_digest": self.thresholds_digest,
        }
        return {name: values[name] for name in _DECISION_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with sorted keys for byte-identical payloads."""

        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class HallucinationReport:
    """Deterministic report for one check over fixed provider outputs.

    Attributes:
        thresholds_id: Identifier of the threshold profile used.
        thresholds_version: Version of the threshold profile used.
        thresholds_digest: Digest of the threshold profile used.
        timeline_digest: Digest of the voice-activity timeline, or ``None``.
        provider_input_digest: Provider input digest, or ``None``.
        provider_output_digest: Provider output digest, or ``None``.
        decisions: One decision per segment, in input order.
        schema_version: Serialization contract identifier.
    """

    thresholds_id: str
    thresholds_version: str
    thresholds_digest: str
    timeline_digest: str | None
    provider_input_digest: str | None
    provider_output_digest: str | None
    decisions: tuple[HallucinationDecision, ...]
    schema_version: str = ASR_HALLUCINATION_SCHEMA_VERSION

    @property
    def eligible_segment_ids(self) -> tuple[str, ...]:
        """Return identifiers of final-eligible segments, in input order."""

        return tuple(d.segment_id for d in self.decisions if d.final_eligible)

    @property
    def review_segment_ids(self) -> tuple[str, ...]:
        """Return identifiers of segments that need review, in input order."""

        return tuple(d.segment_id for d in self.decisions if d.review_required)

    @property
    def outcome_counts(self) -> dict[str, int]:
        """Return the number of decisions per outcome, in a fixed order."""

        counts = {outcome.value: 0 for outcome in HallucinationOutcome}
        for decision in self.decisions:
            counts[decision.outcome.value] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Return the fixed-shape, value-free dictionary in field order."""

        values: dict[str, Any] = {
            "schema_version": self.schema_version,
            "thresholds_id": self.thresholds_id,
            "thresholds_version": self.thresholds_version,
            "thresholds_digest": self.thresholds_digest,
            "timeline_digest": self.timeline_digest,
            "provider_input_digest": self.provider_input_digest,
            "provider_output_digest": self.provider_output_digest,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }
        return {name: values[name] for name in _REPORT_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with sorted keys for byte-identical payloads."""

        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        """Return the SHA-256 hex digest of :meth:`to_json`."""

        return _sha256_json(self.to_dict())


def check_asr_hallucinations(
    segments: Sequence[AsrHypothesisSegment],
    timeline: VoiceActivityTimeline | None,
    *,
    thresholds: HallucinationThresholds = DEFAULT_HALLUCINATION_THRESHOLDS,
    provider_result: ProviderResultEnvelope | None = None,
) -> HallucinationReport:
    """Check fixed ASR hypothesis segments for unsupported or looping text.

    Each segment is compared with the speech coverage of the injected
    voice-activity timeline, the provider statistics, tandem n-gram loops,
    the repeated n-gram share and its unit rate per second of speech. A
    fired check gives ``flagged``; otherwise missing evidence gives
    ``insufficient_evidence``; only a segment with complete evidence and no
    fired check is ``supported``.

    Args:
        segments: Hypothesis segments with unique identifiers.
        timeline: Voice-activity timeline, or ``None`` when no detector ran.
            ``None`` makes every segment report ``vad_coverage_missing``.
        thresholds: Versioned threshold profile.
        provider_result: Optional successful provider envelope whose input
            and output digests are copied into the report.

    Returns:
        A :class:`HallucinationReport` with one decision per segment.

    Raises:
        AsrHallucinationError: If an argument has the wrong type, segment
            identifiers repeat, a segment has too many units, or the provider
            result is not a success.
    """

    if not isinstance(thresholds, HallucinationThresholds):
        raise AsrHallucinationError("asr_thresholds_type_invalid")
    if timeline is not None and not isinstance(timeline, VoiceActivityTimeline):
        raise AsrHallucinationError("asr_timeline_type_invalid")
    if provider_result is not None:
        if not isinstance(provider_result, ProviderResultEnvelope):
            raise AsrHallucinationError("asr_provider_result_type_invalid")
        if provider_result.outcome is not ProviderResultOutcome.SUCCESS:
            raise AsrHallucinationError("asr_provider_result_not_success")
    if isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence):
        raise AsrHallucinationError("asr_segments_type_invalid")
    if len(segments) > MAX_HYPOTHESIS_SEGMENTS:
        raise AsrHallucinationError("asr_segment_limit_exceeded")
    seen: set[str] = set()
    for segment in segments:
        if not isinstance(segment, AsrHypothesisSegment):
            raise AsrHallucinationError("asr_segment_type_invalid")
        if segment.segment_id in seen:
            raise AsrHallucinationError("asr_segment_id_duplicate")
        seen.add(segment.segment_id)

    ends = (
        [interval.end_ms for interval in timeline.intervals]
        if timeline is not None
        else []
    )
    digest = thresholds.digest
    decisions = tuple(
        _decide(segment, timeline, ends, thresholds, digest) for segment in segments
    )
    return HallucinationReport(
        thresholds_id=thresholds.profile_id,
        thresholds_version=thresholds.version,
        thresholds_digest=digest,
        timeline_digest=None if timeline is None else timeline.digest,
        provider_input_digest=(
            None if provider_result is None else provider_result.input_digest
        ),
        provider_output_digest=(
            None if provider_result is None else provider_result.output_digest
        ),
        decisions=decisions,
    )


def require_draft_eligible(
    report: HallucinationReport,
    segment_ids: Iterable[str],
    *,
    thresholds: HallucinationThresholds = DEFAULT_HALLUCINATION_THRESHOLDS,
) -> tuple[str, ...]:
    """Gate that a draft or PHI-withholding stage must call before citing.

    The report must have been produced under exactly ``thresholds`` (matched
    by digest), every requested segment must be in the report, and every one
    must be final eligible. Flagged and insufficient-evidence segments are
    refused until a reviewer or a later revision replaces the report.

    Args:
        report: Report from :func:`check_asr_hallucinations`.
        segment_ids: Segment identifiers a later stage intends to cite.
        thresholds: The threshold profile the caller expects.

    Returns:
        The requested identifiers, in the given order.

    Raises:
        AsrHallucinationError: ``asr_report_type_invalid``,
            ``asr_thresholds_type_invalid``, ``asr_threshold_profile_mismatch``,
            ``asr_segment_ids_invalid``, ``asr_segment_unknown`` or
            ``asr_segment_review_required``.
    """

    if not isinstance(report, HallucinationReport):
        raise AsrHallucinationError("asr_report_type_invalid")
    if not isinstance(thresholds, HallucinationThresholds):
        raise AsrHallucinationError("asr_thresholds_type_invalid")
    expected = thresholds.digest
    if report.thresholds_digest != expected or any(
        decision.thresholds_digest != expected for decision in report.decisions
    ):
        raise AsrHallucinationError("asr_threshold_profile_mismatch")
    if isinstance(segment_ids, (str, bytes)):
        raise AsrHallucinationError("asr_segment_ids_invalid")
    try:
        requested = tuple(segment_ids)
    except TypeError:
        raise AsrHallucinationError("asr_segment_ids_invalid") from None
    by_id = {decision.segment_id: decision for decision in report.decisions}
    for segment_id in requested:
        if type(segment_id) is not str:
            raise AsrHallucinationError("asr_segment_ids_invalid")
        decision = by_id.get(segment_id)
        if decision is None:
            raise AsrHallucinationError("asr_segment_unknown")
        if not decision.final_eligible:
            raise AsrHallucinationError("asr_segment_review_required")
    return requested


def _decide(
    segment: AsrHypothesisSegment,
    timeline: VoiceActivityTimeline | None,
    ends: list[int],
    thresholds: HallucinationThresholds,
    thresholds_digest: str,
) -> HallucinationDecision:
    codes: set[str] = set()
    segment_ms = segment.end_ms - segment.start_ms

    vad_coverage: float | None = None
    speech_ratio: float | None = None
    speech_ms: int | None = None
    if timeline is None:
        codes.add("vad_coverage_missing")
    else:
        per_kind = _overlap_by_kind(timeline, ends, segment.start_ms, segment.end_ms)
        labeled_ms = sum(per_kind.values())
        speech_ms = per_kind[VoiceActivityKind.SPEECH]
        vad_coverage = _score(labeled_ms / segment_ms)
        speech_ratio = _score(speech_ms / segment_ms)
        if labeled_ms / segment_ms < thresholds.min_vad_coverage:
            codes.add("vad_coverage_missing")
            speech_ms = None
        elif speech_ms / segment_ms < thresholds.min_speech_ratio:
            codes.add("speech_coverage_low")
            dominant_ms = 0
            dominant_code = None
            for kind, code in _NON_SPEECH_KINDS:
                if per_kind[kind] > dominant_ms:
                    dominant_ms, dominant_code = per_kind[kind], code
            if dominant_code is not None:
                codes.add(dominant_code)

    stats = segment.statistics or AsrSegmentStatistics()
    if stats.no_speech_probability is None:
        codes.add("no_speech_probability_missing")
    elif stats.no_speech_probability > thresholds.max_no_speech_probability:
        codes.add("no_speech_probability_high")
    if stats.avg_logprob is None:
        codes.add("avg_logprob_missing")
    elif stats.avg_logprob < thresholds.min_avg_logprob:
        codes.add("avg_logprob_low")
    if stats.compression_ratio is None:
        codes.add("compression_ratio_missing")
    elif stats.compression_ratio > thresholds.max_compression_ratio:
        codes.add("compression_ratio_high")

    units, unit_kind = _units(segment)
    unit_count: int | None = None
    loop_size: int | None = None
    loop_repeats: int | None = None
    repeated_fraction: float | None = None
    rate: float | None = None
    if units is None:
        codes.add("token_units_missing")
    else:
        unit_count = len(units)
        loop_size, loop_repeats, looped = _strongest_loop(units, thresholds)
        if looped:
            codes.add("ngram_loop")
        if unit_count >= thresholds.min_repetition_units:
            fraction = _repeated_fraction(units, thresholds.repetition_ngram)
            repeated_fraction = _score(fraction)
            if fraction > thresholds.max_repeated_ngram_fraction:
                codes.add("repeated_ngram_fraction_high")
        if unit_count > thresholds.max_segment_units:
            codes.add("runaway_length")
        if speech_ms:
            raw_rate = unit_count * 1000 / speech_ms
            rate = _score(raw_rate)
            if raw_rate > thresholds.max_units_per_speech_second:
                codes.add("runaway_length")

    reason_codes = tuple(sorted(codes, key=_REASON_ORDER.__getitem__))
    return HallucinationDecision(
        segment_id=segment.segment_id,
        start_ms=segment.start_ms,
        end_ms=segment.end_ms,
        outcome=_outcome_for(reason_codes),
        reason_codes=reason_codes,
        unit_kind=unit_kind,
        unit_count=unit_count,
        vad_coverage=vad_coverage,
        speech_ratio=speech_ratio,
        no_speech_probability=stats.no_speech_probability,
        avg_logprob=stats.avg_logprob,
        compression_ratio=stats.compression_ratio,
        loop_ngram_size=loop_size,
        loop_repeats=loop_repeats,
        repeated_ngram_fraction=repeated_fraction,
        units_per_speech_second=rate,
        thresholds_version=thresholds.version,
        thresholds_digest=thresholds_digest,
    )


def _overlap_by_kind(
    timeline: VoiceActivityTimeline, ends: list[int], start_ms: int, end_ms: int
) -> dict[VoiceActivityKind, int]:
    totals = {kind: 0 for kind in VoiceActivityKind}
    index = bisect_right(ends, start_ms)
    intervals = timeline.intervals
    while index < len(intervals) and intervals[index].start_ms < end_ms:
        interval = intervals[index]
        overlap = min(end_ms, interval.end_ms) - max(start_ms, interval.start_ms)
        if overlap > 0:
            totals[interval.kind] += overlap
        index += 1
    return totals


def _units(segment: AsrHypothesisSegment) -> tuple[tuple[Any, ...] | None, str]:
    if segment.token_ids is not None:
        return segment.token_ids, "token"
    if segment.text is not None:
        words = tuple(_WORD_RE.findall(segment.text.casefold()))
        if len(words) > MAX_SEGMENT_UNIT_COUNT:
            raise AsrHallucinationError("asr_segment_unit_limit_exceeded")
        return words, "word"
    return None, "none"


def _max_tandem_repeats(units: tuple[Any, ...], size: int) -> int:
    """Return the most consecutive copies of any ``size``-gram, in O(n)."""

    if len(units) < size:
        return 0
    best_run = run = 0
    for index in range(len(units) - size):
        if units[index] == units[index + size]:
            run += 1
            if run > best_run:
                best_run = run
        else:
            run = 0
    return best_run // size + 1


def _strongest_loop(
    units: tuple[Any, ...], thresholds: HallucinationThresholds
) -> tuple[int | None, int | None, bool]:
    best: tuple[int, int] | None = None
    best_looped: tuple[int, int] | None = None
    for size in range(1, thresholds.loop_max_ngram + 1):
        repeats = _max_tandem_repeats(units, size)
        if repeats < 2:
            continue
        minimum = (
            thresholds.min_unigram_loop_repeats
            if size == 1
            else thresholds.min_loop_repeats
        )
        candidate = (size, repeats)
        if best is None or repeats * size > best[0] * best[1]:
            best = candidate
        if repeats >= minimum and (
            best_looped is None or repeats * size > best_looped[0] * best_looped[1]
        ):
            best_looped = candidate
    chosen = best_looped or best
    if chosen is None:
        return None, None, False
    return chosen[0], chosen[1], best_looped is not None


def _repeated_fraction(units: tuple[Any, ...], size: int) -> float:
    total = len(units) - size + 1
    if total <= 0:
        return 0.0
    unique = {units[index : index + size] for index in range(total)}
    return (total - len(unique)) / total


def _outcome_for(codes: Iterable[str]) -> HallucinationOutcome:
    present = frozenset(codes)
    if present & _FLAG_SET:
        return HallucinationOutcome.FLAGGED
    if present & _INSUFFICIENT_SET:
        return HallucinationOutcome.INSUFFICIENT_EVIDENCE
    return HallucinationOutcome.SUPPORTED


__all__ = [
    "ASR_HALLUCINATION_SCHEMA_VERSION",
    "DEFAULT_HALLUCINATION_THRESHOLDS",
    "FLAG_REASON_CODES",
    "HALLUCINATION_REASON_CODES",
    "INSUFFICIENT_EVIDENCE_REASON_CODES",
    "MAX_HYPOTHESIS_CHARS",
    "MAX_HYPOTHESIS_SEGMENTS",
    "MAX_SEGMENT_UNIT_COUNT",
    "MAX_TOKEN_ID",
    "MAX_VOICE_ACTIVITY_INTERVALS",
    "AsrHallucinationError",
    "AsrHypothesisSegment",
    "AsrSegmentStatistics",
    "HallucinationDecision",
    "HallucinationOutcome",
    "HallucinationReport",
    "HallucinationThresholds",
    "VoiceActivityInterval",
    "VoiceActivityKind",
    "VoiceActivityTimeline",
    "check_asr_hallucinations",
    "require_draft_eligible",
]
