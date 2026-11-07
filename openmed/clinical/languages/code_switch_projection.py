"""Source-aligned code-switch windows and deterministic entity reconciliation."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice

from openmed.core.language_router import LanguageRouter, LanguageRun

_LANGUAGE = re.compile(r"[a-z]{2,3}")
_LABEL = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _probability(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("invalid code-switch confidence")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("invalid code-switch confidence")


@dataclass(frozen=True, slots=True)
class LanguageWindow:
    """An overlapping language hypothesis in original character coordinates."""

    start: int
    end: int
    language: str
    confidence: float

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end
        ):
            raise ValueError("invalid code-switch window offsets")
        if not isinstance(self.language, str) or not _LANGUAGE.fullmatch(self.language):
            raise ValueError("invalid code-switch language")
        _probability(self.confidence)
        object.__setattr__(
            self, "confidence", 0.0 if self.confidence == 0 else float(self.confidence)
        )

    @property
    def window_id(self) -> str:
        """Return a deterministic metadata-derived window reference."""
        value = f"{self.start}:{self.end}:{self.language}:{self.confidence}"
        return hashlib.sha256(value.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class WindowEntity:
    """A local detector span with a window binding and evidence fingerprint."""

    window_id: str
    start: int
    end: int
    label: str
    confidence: float
    evidence_digest: str

    def __post_init__(self) -> None:
        for digest in (self.window_id, self.evidence_digest):
            if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
                raise ValueError("invalid code-switch evidence reference")
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end
        ):
            raise ValueError("invalid local code-switch entity offsets")
        if not isinstance(self.label, str) or not _LABEL.fullmatch(self.label):
            raise ValueError("invalid code-switch entity label")
        _probability(self.confidence)
        object.__setattr__(
            self, "confidence", 0.0 if self.confidence == 0 else float(self.confidence)
        )


def code_switch_windows(
    text: str,
    *,
    router: LanguageRouter | None = None,
    overlap: int = 16,
    max_characters: int = 1000000,
    max_windows: int = 10000,
) -> tuple[LanguageWindow, ...]:
    """Route language runs and add bounded overlap without transforming source text.

    The default router is deterministic and import-light. Its language confidence
    remains a routing hint, separate from clinical or identifier recall.
    """
    for value, upper, minimum in (
        (overlap, 1024, 0),
        (max_characters, 10000000, 1),
        (max_windows, 10000, 1),
    ):
        if type(value) is not int or not minimum <= value <= upper:
            raise ValueError("invalid code-switch window limit")
    if not isinstance(text, str) or len(text) > max_characters:
        raise ValueError("invalid code-switch source length")
    if not text:
        return ()
    runs = tuple(islice((router or LanguageRouter()).route_runs(text), max_windows + 1))
    if len(runs) > max_windows:
        raise ValueError("code-switch window limit exceeded")
    windows = set()
    for run in runs:
        if (
            not isinstance(run, LanguageRun)
            or type(run.start) is not int
            or type(run.end) is not int
        ):
            raise ValueError("invalid code-switch language run")
        if not 0 <= run.start < run.end <= len(text):
            raise ValueError("code-switch language run exceeds source")
        windows.add(
            LanguageWindow(
                max(0, run.start - overlap),
                min(len(text), run.end + overlap),
                run.language,
                run.confidence,
            )
        )
    return tuple(
        sorted(windows, key=lambda w: (w.start, w.end, w.language, w.confidence))
    )


def reconcile_code_switch_entities(
    windows: Iterable[LanguageWindow],
    candidates: Iterable[WindowEntity],
    *,
    text_length: int,
    max_candidates: int = 10000,
) -> dict[str, object]:
    """Project local spans, deduplicate exact matches, and preserve overlap conflicts.

    Different labels or boundaries that overlap require review; no clinical
    winner is inferred from a language confidence or a detector score.
    """
    if type(text_length) is not int or not 0 <= text_length <= 10000000:
        raise ValueError("invalid code-switch text length")
    if type(max_candidates) is not int or not 1 <= max_candidates <= 10000:
        raise ValueError("invalid code-switch candidate limit")
    bounded_windows = tuple(islice(windows, 10001))
    if len(bounded_windows) > 10000 or any(
        not isinstance(w, LanguageWindow) for w in bounded_windows
    ):
        raise ValueError("invalid code-switch windows")
    if any(w.end > text_length for w in bounded_windows):
        raise ValueError("code-switch window exceeds source")
    indexed = {w.window_id: w for w in bounded_windows}
    grouped = {}
    for index, candidate in enumerate(candidates):
        if index >= max_candidates:
            raise ValueError("code-switch candidate limit exceeded")
        if not isinstance(candidate, WindowEntity):
            raise ValueError("invalid code-switch candidate")
        window = indexed.get(candidate.window_id)
        if window is None:
            raise ValueError("code-switch candidate has unknown window")
        if candidate.end > window.end - window.start:
            raise ValueError("code-switch candidate exceeds window")
        key = (
            window.start + candidate.start,
            window.start + candidate.end,
            candidate.label,
        )
        row = grouped.setdefault(
            key, {"confidence": 0.0, "evidence": set(), "languages": set()}
        )
        row["confidence"] = max(row["confidence"], candidate.confidence)
        row["evidence"].add(candidate.evidence_digest)
        row["languages"].add(window.language)
    keys = sorted(grouped)
    conflicts = set()
    # Sweep only currently overlapping spans; a budget bounds ambiguous dense inputs.
    comparisons = 0
    for i, left in enumerate(keys):
        for j in range(i + 1, len(keys)):
            right = keys[j]
            if right[0] >= left[1]:
                break
            comparisons += 1
            if comparisons > 100000:
                raise ValueError("code-switch conflict comparison limit exceeded")
            conflicts.add(left)
            conflicts.add(right)
    entities = []
    for key in keys:
        row = grouped[key]
        entities.append(
            {
                "start": key[0],
                "end": key[1],
                "label": key[2],
                "confidence": row["confidence"],
                "languages": sorted(row["languages"]),
                "evidence_digests": sorted(row["evidence"]),
                "status": "needs_review" if key in conflicts else "candidate",
            }
        )
    return {
        "schema_version": "1.0.0",
        "entities": entities,
        "conflict_count": len(conflicts),
        "needs_review": bool(conflicts),
    }
