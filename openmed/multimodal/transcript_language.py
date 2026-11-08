"""Fail-closed language routing for fixed, finalized local transcripts.

This boundary does not implement ASR or install/qualify PHI packs. The caller
supplies a local token language identifier and detectors for installed packs.
Only explicitly reviewed, detector-processed text can leave the boundary.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from openmed.core.lang_id_codemix import TokenLanguageIdentifier
from openmed.core.language_pack import LanguagePack
from openmed.core.language_router import LanguageIdentifier, LanguagePrediction

NON_DIAGNOSTIC_NOTICE = (
    "Non-diagnostic transcript; explicit reviewer confirmation required."
)
_TAG = re.compile(r"([a-zA-Z]{2})(?:-([a-zA-Z]{4}))?(?:-([a-zA-Z]{2}|[0-9]{3}))?\Z")
_TAG_SCRIPTS = frozenset(
    "Arab Armn Beng Cyrl Deva Ethi Geor Grek Gujr Guru Hang Hani Hans Hant "
    "Hebr Jpan Kana Khmr Knda Kore Laoo Latn Mlym Mymr Orya Sinh Taml Telu "
    "Thaa Thai Tibt".split()
)
LanguageStatus = Literal["supported", "mixed", "unsupported", "uncertain"]


class TranscriptLanguageError(ValueError):
    """A controlled, value-free failure at the transcript boundary."""


def _tag(value: str) -> str:
    match = _TAG.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise TranscriptLanguageError("invalid_language_tag")
    language, script, region = match.groups()
    if script and script.title() not in _TAG_SCRIPTS:
        raise TranscriptLanguageError("invalid_language_tag")
    return "-".join(
        part
        for part in (
            language.lower(),
            script.title() if script else None,
            region.upper() if region else None,
        )
        if part
    )


def _probability(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise TranscriptLanguageError("invalid_confidence")


def _bucket(value: float | None, threshold: float) -> str:
    if value is None:
        return "missing"
    if value < threshold:
        return "low"
    return "high" if value >= 0.9 else "accepted"


@dataclass(frozen=True, slots=True)
class TranscriptPHISpan:
    """A detector's half-open Unicode code-point offsets, without PHI text.

    Args:
        start: Inclusive local character offset.
        end: Exclusive local character offset.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int:
            raise TranscriptLanguageError("invalid_detector_span")
        if not 0 <= self.start < self.end:
            raise TranscriptLanguageError("invalid_detector_span")


@dataclass(frozen=True, slots=True)
class TranscriptLanguageRun:
    """Content-free routing metadata addressing the original transcript.

    Args:
        start: Inclusive source code-point offset.
        end: Exclusive source code-point offset.
        tag: Canonical BCP 47 language tag from text evidence.
        confidence_bucket: Lowest token confidence bucket in this run.
    """

    start: int
    end: int
    tag: str
    confidence_bucket: str


@dataclass(frozen=True, slots=True)
class TranscriptLanguageDecision:
    """A segment decision with safe metadata and protected reviewed output.

    Use :meth:`reviewed_text` for both release and draft intake. Supported and
    mixed classify language coverage, not clinical validity or PHI recall.
    """

    segment_index: int
    status: LanguageStatus
    reason_code: str
    provider_tag: str | None
    provider_confidence_bucket: str
    runs: tuple[TranscriptLanguageRun, ...] = ()
    phi_spans: tuple[TranscriptPHISpan, ...] = ()
    notice: str = NON_DIAGNOSTIC_NOTICE
    _review_text: str | None = field(default=None, repr=False)

    def reviewed_text(self, *, reviewer_confirmed: bool = False) -> str:
        """Return protected redacted text for release/drafts after review.

        Args:
            reviewer_confirmed: Explicit confirmation of this segment's output.

        Raises:
            TranscriptLanguageError: If withheld or confirmation is absent.
        """
        if self._review_text is None:
            raise TranscriptLanguageError("segment_withheld")
        if reviewer_confirmed is not True:
            raise TranscriptLanguageError("review_required")
        return self._review_text

    def to_dict(self) -> dict[str, object]:
        """Return only controlled codes, tags, buckets, counts and offsets."""
        return {
            "segment_index": self.segment_index,
            "status": self.status,
            "reason_code": self.reason_code,
            "provider_tag": self.provider_tag,
            "provider_confidence_bucket": self.provider_confidence_bucket,
            "runs": [
                {
                    "start": run.start,
                    "end": run.end,
                    "tag": run.tag,
                    "confidence_bucket": run.confidence_bucket,
                }
                for run in self.runs
            ],
            "phi_span_count": len(self.phi_spans),
        }


Detector = Callable[[str], Sequence[TranscriptPHISpan]]


class TranscriptLanguageRouter:
    """Combine spoken-language confidence, token LID and installed PHI packs."""

    def __init__(
        self,
        *,
        installed_packs: Sequence[LanguagePack],
        detectors: Mapping[str, Detector],
        text_identifier: LanguageIdentifier,
        candidate_languages: Sequence[str],
        confidence_threshold: float = 0.8,
    ) -> None:
        """Snapshot installed metadata and explicitly local detector callbacks.

        Args:
            installed_packs: Actually installed packs, not catalog availability.
            detectors: One offline detector per installed pack code. Adapt an
                existing detector's entities to offset-only TranscriptPHISpan.
            text_identifier: Caller-supplied on-device token LID; no implicit
                script/pack-priority fallback is used as language evidence.
            candidate_languages: Text LID languages, including uninstalled ones
                to avoid biasing identification to installed PHI packs.
            confidence_threshold: Minimum provider and lexical-token confidence.
        """
        _probability(confidence_threshold)
        if confidence_threshold == 0:
            raise TranscriptLanguageError("invalid_confidence_threshold")
        codes = [pack.code for pack in installed_packs]
        if len(set(codes)) != len(codes) or set(codes) != set(detectors):
            raise TranscriptLanguageError("invalid_installed_detectors")
        if not all(callable(detector) for detector in detectors.values()):
            raise TranscriptLanguageError("invalid_installed_detectors")
        candidates = tuple(
            sorted({_tag(code).split("-")[0] for code in candidate_languages})
        )
        if not candidates or not isinstance(text_identifier, LanguageIdentifier):
            raise TranscriptLanguageError("invalid_text_identifier")
        self._detectors = dict(detectors)
        self._identifier = text_identifier
        self._candidates = candidates
        self._threshold = confidence_threshold

    def route(
        self,
        *,
        segment_index: int,
        text: str,
        provider_language: LanguagePrediction | None,
        finalized: bool = True,
    ) -> TranscriptLanguageDecision:
        """Route one fixed hypothesis, withholding any failed gate atomically.

        Args:
            segment_index: Opaque nonnegative segment number; never a patient ID.
            text: Protected transcript, retained only within this call.
            provider_language: Fixed dominant-language hypothesis. Regional tags
                use the same primary-language pack; missing evidence withholds.
            finalized: False withholds partial transcripts without detector calls.

        Returns:
            A content-free decision with review-gated redacted text, if eligible.
        """
        if type(segment_index) is not int or segment_index < 0:
            raise TranscriptLanguageError("invalid_segment_index")
        if not isinstance(text, str) or len(text) > 65_536:
            raise TranscriptLanguageError("invalid_transcript")
        tag = None
        confidence = None
        if provider_language is not None:
            if not isinstance(provider_language, LanguagePrediction):
                raise TranscriptLanguageError("invalid_provider_language")
            tag = _tag(provider_language.language)
            confidence = provider_language.confidence
            _probability(confidence)
        bucket = _bucket(confidence, self._threshold)

        def withheld(
            reason: str,
            status: LanguageStatus = "uncertain",
            runs: tuple[TranscriptLanguageRun, ...] = (),
        ) -> TranscriptLanguageDecision:
            return TranscriptLanguageDecision(
                segment_index, status, reason, tag, bucket, runs
            )

        if finalized is not True:
            return withheld("segment_not_finalized")
        if confidence is None or tag is None:
            return withheld("provider_language_missing")
        if confidence < self._threshold:
            return withheld("provider_confidence_low")
        try:
            runs, languages = self._text_runs(text, tag)
        except Exception:
            # No exception from a caller's local backend is copied to diagnostics.
            return withheld("text_identifier_failed")
        if not runs:
            return withheld("text_language_uncertain")
        if any(run.confidence_bucket in {"low", "missing"} for run in runs):
            return withheld("text_confidence_low", runs=runs)
        maximum = max(languages.values())
        dominant = {code for code, count in languages.items() if count == maximum}
        if tag.split("-")[0] not in dominant:
            return withheld("language_disagreement", runs=runs)
        if any(run.tag.split("-")[0] not in self._detectors for run in runs):
            return withheld("phi_pack_unavailable", "unsupported", runs)
        spans: list[TranscriptPHISpan] = []
        try:
            for run in runs:
                found = tuple(
                    self._detectors[run.tag.split("-")[0]](text[run.start : run.end])
                )
                for span in found:
                    if (
                        not isinstance(span, TranscriptPHISpan)
                        or span.end > run.end - run.start
                    ):
                        return withheld("detector_result_invalid", runs=runs)
                    spans.append(
                        TranscriptPHISpan(run.start + span.start, run.start + span.end)
                    )
        except Exception:
            return withheld("detector_failed", runs=runs)
        merged: list[TranscriptPHISpan] = []
        for span in sorted(spans, key=lambda item: (item.start, item.end)):
            if merged and span.start <= merged[-1].end:
                previous = merged.pop()
                span = TranscriptPHISpan(previous.start, max(previous.end, span.end))
            merged.append(span)
        output = list(text)
        for span in merged:
            output[span.start : span.end] = "█" * (span.end - span.start)
        status: LanguageStatus = "mixed" if len(languages) > 1 else "supported"
        return TranscriptLanguageDecision(
            segment_index,
            status,
            "language_routed",
            tag,
            bucket,
            runs,
            tuple(merged),
            _review_text="".join(output),
        )

    def _text_runs(
        self, text: str, provider_tag: str
    ) -> tuple[tuple[TranscriptLanguageRun, ...], Counter[str]]:
        tokens = TokenLanguageIdentifier().identify(text)
        evidence: list[tuple[int, str, float]] = []
        counts: Counter[str] = Counter()
        candidates = tuple(sorted(set(self._candidates) | {provider_tag.split("-")[0]}))
        for token in tokens:
            surface = text[token.start : token.end]
            if not any(character.isalpha() for character in surface):
                continue
            prediction = self._identifier.identify(surface, candidates)
            if prediction is None:
                return (), counts
            tag = _tag(prediction.language)
            _probability(prediction.confidence)
            counts[tag.split("-")[0]] += 1
            evidence.append((token.start, tag, prediction.confidence))
        if not evidence:
            return (), counts
        # Neutral tokens and whitespace stay with the preceding lexical run;
        # a leading neutral prefix stays with the first run. Thus all source
        # characters reach a detector, with no gaps or offset normalization.
        runs: list[TranscriptLanguageRun] = []
        start, tag, confidence = 0, evidence[0][1], evidence[0][2]
        for boundary, next_tag, next_confidence in evidence[1:]:
            if next_tag != tag:
                runs.append(
                    TranscriptLanguageRun(
                        start, boundary, tag, _bucket(confidence, self._threshold)
                    )
                )
                start, tag, confidence = boundary, next_tag, next_confidence
            else:
                confidence = min(confidence, next_confidence)
        runs.append(
            TranscriptLanguageRun(
                start, len(text), tag, _bucket(confidence, self._threshold)
            )
        )
        return tuple(runs), counts


def transcript_language_report(
    decisions: Sequence[TranscriptLanguageDecision],
) -> dict[str, dict[str, int]]:
    """Aggregate only controlled tags, confidence buckets and decision counts.

    Args:
        decisions: Router decisions; protected text is never accessed.
    """
    statuses = Counter(decision.status for decision in decisions)
    reasons = Counter(decision.reason_code for decision in decisions)
    tags = Counter(run.tag for decision in decisions for run in decision.runs)
    provider_tags = Counter(
        decision.provider_tag for decision in decisions if decision.provider_tag
    )
    buckets = Counter(decision.provider_confidence_bucket for decision in decisions)
    return {
        "status_counts": dict(sorted(statuses.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "tag_counts": dict(sorted(tags.items())),
        "provider_tag_counts": dict(sorted(provider_tags.items())),
        "confidence_bucket_counts": dict(sorted(buckets.items())),
    }
