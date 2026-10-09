"""Local, audio-free endpoint state for caller-owned streaming speech frames."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

ENDPOINT_NOTICE = (
    "Non-diagnostic speech segmentation only; no speaker identity, consent or "
    "clinical meaning. Consequential use requires explicit reviewer confirmation."
)


class Activity(str, Enum):
    """Detector decisions; uncertainty is never promoted to confirmed speech."""

    SPEECH = "speech"
    SILENCE = "silence"
    UNCERTAIN = "uncertain"


class LocalActivityDetector(Protocol):
    """Injected offline detector; must release any retained audio on reset.

    The caller is responsible for verifying the implementation's offline and
    ownership declarations. No model qualification is implied by this protocol.
    """

    requires_network: bool

    def detect(self, samples: tuple[float, ...]) -> Activity:
        """Classify one bounded mono frame without logging or persisting audio."""
        ...

    def reset(self) -> None:
        """Discard detector-owned audio and temporal state."""
        ...


class EndpointError(ValueError):
    """Controlled, value-free endpoint failure."""


@dataclass(frozen=True, slots=True)
class EndpointEvent:
    """Audio-free event with half-open source sample bounds.

    Start events have equal bounds; end events span the complete utterance,
    including bounded hangover. Gap and uncertainty events span affected input.
    """

    kind: str
    start_sample: int
    end_sample: int
    reason: str

    @property
    def notice(self) -> str:
        """Return the notice bound to every segmentation event."""
        return ENDPOINT_NOTICE

    @property
    def reviewer_confirmation_required(self) -> bool:
        """Require explicit confirmation before consequential downstream use."""
        return True


def _integer(value: int, minimum: int, maximum: int) -> bool:
    return type(value) is int and minimum <= value <= maximum


class SpeechEndpointAdapter:
    """Bound utterances without buffering audio or invoking ASR.

    Args:
        detector: Caller-supplied local activity detector with reset support.
        hangover_samples: Consecutive non-speech samples before ending speech.
        max_utterance_samples: Hard duration cap, including trailing silence.
        max_frame_samples: Input frame cap, no larger than the utterance cap.

    All frames use one caller-declared sample clock and mono normalized samples.
    The detector classifies the entire frame; endpoint cuts may occur inside it.
    Feed consistent detector frames for reproducibility. Instances are serial,
    single-stream objects. Neither input audio nor event history is retained.
    """

    def __init__(
        self,
        detector: LocalActivityDetector,
        *,
        hangover_samples: int,
        max_utterance_samples: int,
        max_frame_samples: int = 4096,
    ) -> None:
        if not (
            _integer(max_utterance_samples, 1, 10_000_000)
            and _integer(hangover_samples, 0, max_utterance_samples)
            and _integer(max_frame_samples, 1, min(65_536, max_utterance_samples))
        ):
            raise EndpointError("endpoint_invalid_limits")
        if getattr(detector, "requires_network", None) is not False:
            raise EndpointError("endpoint_detector_not_local")
        self._detector = detector
        self._hangover = hangover_samples
        self._maximum = max_utterance_samples
        self._frame_limit = max_frame_samples
        self._start: int | None = None
        self._expected: int | None = None
        self._silence = 0
        self._cancelled = False

    @property
    def buffered_sample_count(self) -> int:
        """Return zero: audio remains caller-owned and is never buffered."""
        return 0

    def push(
        self, start_sample: int, samples: tuple[float, ...]
    ) -> tuple[EndpointEvent, ...]:
        """Classify a bounded frame and return ordered offset-only events.

        Args:
            start_sample: Absolute source index, monotonic within a stream.
            samples: Immutable mono frame of finite samples in [-1, 1].

        Returns:
            Speech start/end, uncertainty and forward-gap events. Continuous
            speech is split exactly at the duration cap with no missing samples.

        Raises:
            EndpointError: For invalid frames, overlaps, cancellation or detector
                failure. Detector failures discard state and require reset.
        """
        if self._cancelled:
            raise EndpointError("endpoint_cancelled")
        if not (
            type(samples) is tuple
            and 1 <= len(samples) <= self._frame_limit
            and _integer(start_sample, 0, 2**63 - 1 - len(samples))
            and all(
                type(value) in (int, float)
                and -1 <= value <= 1
                and math.isfinite(value)
                for value in samples
            )
        ):
            raise EndpointError("endpoint_invalid_frame")
        if self._expected is not None and start_sample < self._expected:
            raise EndpointError("endpoint_overlapping_frame")
        try:
            if self._detector.requires_network is not False:
                raise EndpointError("endpoint_detector_not_local")
            if self._expected is not None and start_sample > self._expected:
                self._detector.reset()
            activity = self._detector.detect(samples)
            if type(activity) is not Activity:
                raise EndpointError("endpoint_invalid_activity")
        except Exception:
            self._cancelled = True
            self._clear()
            try:
                self._detector.reset()
            except Exception:
                pass
            raise EndpointError("endpoint_detector_failure") from None

        events: list[EndpointEvent] = []
        if self._expected is not None and start_sample > self._expected:
            self._end(self._expected, "discontinuity", events)
            events.append(
                EndpointEvent("discontinuity", self._expected, start_sample, "gap")
            )
        end = start_sample + len(samples)
        if activity is Activity.UNCERTAIN:
            events.append(
                EndpointEvent("uncertain_activity", start_sample, end, "uncertain")
            )
        cursor = start_sample
        while cursor < end:
            if self._start is None:
                if activity is not Activity.SPEECH:
                    break
                self._start = cursor
                events.append(EndpointEvent("speech_start", cursor, cursor, "detected"))
            if activity is Activity.SPEECH:
                self._silence = 0
            limit = self._start + self._maximum
            reason = "maximum_duration"
            if activity is not Activity.SPEECH:
                silence_limit = cursor + max(0, self._hangover - self._silence)
                if silence_limit <= limit:
                    limit = silence_limit
                    reason = "hangover"
            cut = min(end, limit)
            if activity is not Activity.SPEECH:
                self._silence += cut - cursor
            cursor = cut
            if cut == limit:
                self._end(cut, reason, events)
            else:
                break
        self._expected = end
        return tuple(events)

    def finish(self) -> tuple[EndpointEvent, ...]:
        """End a stream, release detector state, and permit a new source clock."""
        if self._cancelled:
            raise EndpointError("endpoint_cancelled")
        return self._release("stream_end", cancelled=False)

    def reset(self) -> tuple[EndpointEvent, ...]:
        """Discard detector state and end active speech; reopen after cancellation."""
        return self._release("reset", cancelled=False)

    def cancel(self) -> tuple[EndpointEvent, ...]:
        """Discard detector state and refuse frames until explicit reset."""
        return self._release("cancelled", cancelled=True)

    def _end(self, end: int, reason: str, events: list[EndpointEvent]) -> None:
        if self._start is not None:
            events.append(EndpointEvent("speech_end", self._start, end, reason))
        self._start = None
        self._silence = 0

    def _clear(self) -> None:
        self._start = None
        self._expected = None
        self._silence = 0

    def _release(self, reason: str, *, cancelled: bool) -> tuple[EndpointEvent, ...]:
        events: list[EndpointEvent] = []
        self._end(self._expected or 0, reason, events)
        self._clear()
        self._cancelled = True
        try:
            self._detector.reset()
        except Exception:
            raise EndpointError("endpoint_detector_reset_failure") from None
        self._cancelled = cancelled
        return tuple(events)


__all__ = [
    "Activity",
    "ENDPOINT_NOTICE",
    "EndpointError",
    "EndpointEvent",
    "LocalActivityDetector",
    "SpeechEndpointAdapter",
]
