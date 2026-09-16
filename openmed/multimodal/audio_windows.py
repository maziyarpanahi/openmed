"""Deterministic half-open window planning for offline streaming audio.

Windows are planned from declared integer-millisecond metadata before any
audio is read or decoded. Nothing here touches samples, transcripts, speaker
labels, filenames, or clinical text, so a plan carries offsets and counts only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

AUDIO_WINDOW_SCHEMA_VERSION: Final[str] = "openmed.multimodal.audio_windows.v1"
MAX_AUDIO_DURATION_MS: Final[int] = 86_400_000
MAX_AUDIO_WINDOW_MS: Final[int] = 3_600_000
MAX_AUDIO_WINDOW_COUNT: Final[int] = 100_000

_WINDOW_FIELDS = ("window_id", "window_index", "start_ms", "end_ms", "overlap_ms")
_PLAN_FIELDS = (
    "schema_version",
    "duration_ms",
    "window_ms",
    "overlap_ms",
    "stride_ms",
    "min_tail_ms",
    "tail_policy",
    "covered_ms",
    "windows",
)


class TailPolicy(str, Enum):
    """Closed set of rules for a final window shorter than ``min_tail_ms``.

    Values:
        KEEP: Emit the short tail as its own window.
        MERGE: Extend the previous window to the end of the input.
        DROP: Discard the tail, leaving the declared duration partly uncovered.
    """

    KEEP = "keep"
    MERGE = "merge"
    DROP = "drop"


class AudioWindowError(ValueError):
    """Value-free failure raised for unusable audio window parameters."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """One half-open ``[start_ms, end_ms)`` window of declared audio."""

    window_id: str
    window_index: int
    start_ms: int
    end_ms: int
    overlap_ms: int

    @property
    def duration_ms(self) -> int:
        """Return the window length in milliseconds."""

        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in declared field order."""

        values: dict[str, Any] = {
            "window_id": self.window_id,
            "window_index": self.window_index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "overlap_ms": self.overlap_ms,
        }
        return {field: values[field] for field in _WINDOW_FIELDS}


@dataclass(frozen=True, slots=True)
class AudioWindowPlan:
    """Reproducible window plan for one declared audio duration."""

    duration_ms: int
    window_ms: int
    overlap_ms: int
    stride_ms: int
    min_tail_ms: int
    tail_policy: TailPolicy
    covered_ms: int
    windows: tuple[AudioWindow, ...]
    schema_version: str = AUDIO_WINDOW_SCHEMA_VERSION

    @property
    def window_count(self) -> int:
        """Return the number of planned windows."""

        return len(self.windows)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in declared field order."""

        values: dict[str, Any] = {
            "schema_version": self.schema_version,
            "duration_ms": self.duration_ms,
            "window_ms": self.window_ms,
            "overlap_ms": self.overlap_ms,
            "stride_ms": self.stride_ms,
            "min_tail_ms": self.min_tail_ms,
            "tail_policy": self.tail_policy.value,
            "covered_ms": self.covered_ms,
            "windows": [window.to_dict() for window in self.windows],
        }
        return {field: values[field] for field in _PLAN_FIELDS}

    def to_json(self) -> str:
        """Return compact JSON with sorted keys for byte-identical payloads."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def plan_audio_windows(
    duration_ms: int,
    *,
    window_ms: int,
    overlap_ms: int = 0,
    min_tail_ms: int = 0,
    tail_policy: TailPolicy | str = TailPolicy.KEEP,
) -> AudioWindowPlan:
    """Plan half-open windows over a declared audio duration.

    Windows start at zero and advance by ``window_ms - overlap_ms``. The last
    window is clipped to the declared duration, and planning stops as soon as a
    window reaches the end, so no window is ever contained in its predecessor.

    A final window shorter than ``min_tail_ms`` is resolved by ``tail_policy``.
    The rule is not applied when the plan holds a single window, because that
    window is the whole input rather than a tail.

    Args:
        duration_ms: Declared total duration in whole milliseconds.
        window_ms: Window length in whole milliseconds.
        overlap_ms: Overlap between consecutive windows; must be smaller than
            ``window_ms``.
        min_tail_ms: Shortest acceptable final window; ``0`` disables the rule.
        tail_policy: How to resolve a tail shorter than ``min_tail_ms``.

    Returns:
        An :class:`AudioWindowPlan` covering the duration according to the
        selected tail policy.

    Raises:
        AudioWindowError: If any parameter is out of range, non-integer,
            boolean, or would produce more than ``MAX_AUDIO_WINDOW_COUNT``
            windows.
    """

    _validate_bounds(duration_ms, "duration", MAX_AUDIO_DURATION_MS)
    _validate_bounds(window_ms, "window", MAX_AUDIO_WINDOW_MS)
    _validate_bounds(overlap_ms, "overlap", MAX_AUDIO_WINDOW_MS)
    _validate_bounds(min_tail_ms, "min_tail", MAX_AUDIO_WINDOW_MS)
    policy = _parse_policy(tail_policy)
    if window_ms == 0:
        raise AudioWindowError("audio_window_size_invalid")
    if overlap_ms >= window_ms:
        raise AudioWindowError("audio_overlap_not_less_than_window")
    if min_tail_ms > window_ms:
        raise AudioWindowError("audio_min_tail_exceeds_window")

    stride_ms = window_ms - overlap_ms
    planned = -(-duration_ms // stride_ms) if duration_ms else 0
    if planned > MAX_AUDIO_WINDOW_COUNT:
        raise AudioWindowError("audio_window_count_limit_exceeded")

    bounds = _window_bounds(duration_ms, window_ms, stride_ms)
    bounds = _apply_tail_policy(bounds, duration_ms, min_tail_ms, policy)
    windows = _build_windows(bounds)
    return AudioWindowPlan(
        duration_ms=duration_ms,
        window_ms=window_ms,
        overlap_ms=overlap_ms,
        stride_ms=stride_ms,
        min_tail_ms=min_tail_ms,
        tail_policy=policy,
        covered_ms=windows[-1].end_ms if windows else 0,
        windows=windows,
    )


def _window_bounds(
    duration_ms: int, window_ms: int, stride_ms: int
) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < duration_ms:
        end = min(start + window_ms, duration_ms)
        bounds.append((start, end))
        if end == duration_ms:
            break
        start += stride_ms
    return bounds


def _apply_tail_policy(
    bounds: list[tuple[int, int]],
    duration_ms: int,
    min_tail_ms: int,
    policy: TailPolicy,
) -> list[tuple[int, int]]:
    if min_tail_ms == 0 or len(bounds) < 2:
        return bounds
    start, end = bounds[-1]
    if end - start >= min_tail_ms:
        return bounds
    if policy is TailPolicy.KEEP:
        return bounds
    trimmed = bounds[:-1]
    if policy is TailPolicy.MERGE:
        previous_start, _ = trimmed[-1]
        trimmed[-1] = (previous_start, duration_ms)
    return trimmed


def _build_windows(bounds: list[tuple[int, int]]) -> tuple[AudioWindow, ...]:
    windows: list[AudioWindow] = []
    previous_end = 0
    for index, (start, end) in enumerate(bounds):
        windows.append(
            AudioWindow(
                window_id=f"w{index:06d}",
                window_index=index,
                start_ms=start,
                end_ms=end,
                overlap_ms=max(0, previous_end - start) if index else 0,
            )
        )
        previous_end = end
    return tuple(windows)


def _validate_bounds(value: Any, name: str, maximum: int) -> None:
    if type(value) is not int:
        raise AudioWindowError(f"audio_{name}_not_an_integer")
    if value < 0 or value > maximum:
        raise AudioWindowError(f"audio_{name}_out_of_range")


def _parse_policy(value: Any) -> TailPolicy:
    if isinstance(value, TailPolicy):
        return value
    if type(value) is str:
        try:
            return TailPolicy(value)
        except ValueError:
            pass
    raise AudioWindowError("audio_tail_policy_unsupported")


__all__ = [
    "AUDIO_WINDOW_SCHEMA_VERSION",
    "MAX_AUDIO_DURATION_MS",
    "MAX_AUDIO_WINDOW_COUNT",
    "MAX_AUDIO_WINDOW_MS",
    "AudioWindow",
    "AudioWindowError",
    "AudioWindowPlan",
    "TailPolicy",
    "plan_audio_windows",
]
