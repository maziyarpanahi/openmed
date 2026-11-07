"""Conservative overlap reconciliation for fixed local ASR window results.

Only identical, explicitly timed overlap tokens are coalesced. Ambiguous source
alternatives stay protected and provisional; no textual similarity heuristic may
remove a medication, repeated statement or negation. Existing revision and PHI
withholding contracts own finalization and release.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openmed.multimodal.audio_windows import AudioWindowPlan, plan_audio_windows
from openmed.multimodal.streaming.privacy.withholding_buffer import (
    LocalPhiScreener,
    ReleasedSpeechSegment,
    WithholdingBuffer,
    WithholdingPolicy,
)
from openmed.multimodal.streaming.speech.providers import (
    PartialHypothesis,
    ProviderProvenance,
    TokenTime,
)

from .revision_ledger import RevisionKind, TranscriptToken

_REF = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class StitchingError(ValueError):
    """Fixed contract refusal without protected source or transcript values."""


def _require(condition: bool) -> None:
    if not condition:
        raise StitchingError("invalid_asr_window_reconciliation")


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _text(value: str) -> None:
    _require(type(value) is str and len(value) <= 65_536)
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise StitchingError("invalid_asr_window_reconciliation") from None


class SeamReason(str, Enum):
    """Closed reasons that prevent automatic transcript finalization."""

    MISSING_WINDOW = "missing_window"
    OUT_OF_ORDER = "out_of_order_window"
    DUPLICATE_WINDOW = "duplicate_window"
    INCOMPLETE = "incomplete_provider_coverage"
    PROVISIONAL = "provisional_provider_result"
    UNALIGNED = "uncovered_transcript_characters"
    TIMING = "ambiguous_token_timing"
    CONFLICT = "conflicting_overlap"
    BOUNDARY = "ambiguous_text_boundary"
    UNCOVERED_TAIL = "uncovered_audio_tail"


@dataclass(frozen=True, slots=True, kw_only=True)
class AlignedWindow:
    """Protected provider output with explicit source and character alignment.

    Token times are stream-relative milliseconds, never window-relative. Each
    character interval points to the exact token text in the original hypothesis;
    uncovered non-whitespace characters force review rather than being dropped.
    Coverage completeness is a caller-governed provider declaration, not proof of
    speech-recognition accuracy. No ASR model is invoked by this module.
    """

    window_index: int
    source_digest: str
    provider: ProviderProvenance
    hypothesis: PartialHypothesis = field(repr=False)
    character_offsets: tuple[tuple[int, int], ...]
    coverage_complete: bool

    def __post_init__(self) -> None:
        _require(type(self.window_index) is int and 0 <= self.window_index < 64)
        _require(
            type(self.source_digest) is str
            and _DIGEST.fullmatch(self.source_digest) is not None
        )
        _require(
            type(self.provider) is ProviderProvenance
            and not self.provider.requires_network
        )
        _require(type(self.hypothesis) is PartialHypothesis)
        _text(self.hypothesis.text)
        _require(
            type(self.hypothesis.token_times) is tuple
            and len(self.hypothesis.token_times) <= 4096
        )
        _require(all(type(token) is TokenTime for token in self.hypothesis.token_times))
        _require(
            type(self.character_offsets) is tuple
            and len(self.character_offsets) == len(self.hypothesis.token_times)
        )
        _require(type(self.coverage_complete) is bool)
        end = 0
        for token, offsets in zip(self.hypothesis.token_times, self.character_offsets):
            _require(
                type(offsets) is tuple
                and len(offsets) == 2
                and all(type(value) is int for value in offsets)
            )
            start, stop = offsets
            _require(end <= start < stop <= len(self.hypothesis.text))
            _require(self.hypothesis.text[start:stop] == token.token)
            end = stop


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptEdit:
    """Existing ledger operation with fragment-local token coordinates."""

    kind: RevisionKind
    expected_revision: int
    position: int
    text: str = field(repr=False)
    tokens: tuple[TranscriptToken, ...]

    def to_dict(self) -> dict[str, Any]:
        """Describe offsets and opaque token references without transcript text."""
        return {
            "kind": self.kind.value,
            "expected_revision": self.expected_revision,
            "position": self.position,
            "characters": len(self.text),
            "tokens": [
                {
                    "token_ref": token.token_ref,
                    "start": token.start,
                    "end": token.end,
                    "start_ms": token.start_ms,
                    "end_ms": token.end_ms,
                }
                for token in self.tokens
            ],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ReconciledWindows:
    """Protected source alternatives and a deterministic provisional edit plan.

    Only ``to_json`` is an audit report. Original hypotheses and edit text stay
    protected. ``release_with`` recomputes this plan and routes it through a fresh
    existing PHI-withholding buffer; it cannot silently revise a finalized ledger.
    """

    segment_ref: str
    review_required: bool
    reasons: tuple[SeamReason, ...]
    gaps: tuple[tuple[int, int], ...]
    operations: tuple[TranscriptEdit, ...] = field(repr=False)
    alternatives: tuple[AlignedWindow, ...] = field(repr=False)
    _plan: AudioWindowPlan = field(repr=False)
    _session_ref: str = field(repr=False)
    _source_digest: str = field(repr=False)
    _provider: ProviderProvenance = field(repr=False)

    def to_json(self) -> str:
        """Serialize controlled codes, offsets and opaque identities, never text."""
        return json.dumps(
            {
                "schema_version": "openmed.asr_window_stitching.v1",
                "segment_ref": self.segment_ref,
                "review_required": self.review_required,
                "reasons": [reason.value for reason in self.reasons],
                "gaps_ms": self.gaps,
                "received_windows": len(self.alternatives),
                "expected_windows": len(self._plan.windows),
                "operations": [operation.to_dict() for operation in self.operations],
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def release_with(
        self,
        screener: LocalPhiScreener,
        *,
        policy: WithholdingPolicy | None = None,
        clock: Callable[[], float] | None = None,
    ) -> tuple[ReleasedSpeechSegment, ...]:
        """Apply exact edits and release only through local PHI screening.

        Args:
            screener: Trusted locally qualified PHI screener; no network fallback.
            policy: Existing holdback limits, including retention and character caps.
            clock: Optional injected monotonic clock for controlled offline evaluation.

        Returns:
            Existing protected release objects, or an empty tuple for ambiguous,
            incomplete, provisional, expired, over-budget or PHI-unresolved input.
            The temporary buffer closes after the call. Caller-owned source and
            returned text remain caller-owned and are not erased by that close.
        """
        expected = reconcile_windows(
            self._plan,
            self.alternatives,
            session_ref=self._session_ref,
            source_digest=self._source_digest,
            provider=self._provider,
        )
        _require(self == expected)
        target = (
            WithholdingBuffer(self._session_ref, screener, policy=policy)
            if clock is None
            else WithholdingBuffer(
                self._session_ref, screener, policy=policy, clock=clock
            )
        )
        with target:
            if not target.append(self.segment_ref, ""):
                return ()
            for operation in self.operations:
                if operation.kind is RevisionKind.FINALIZE:
                    applied = target.finalize(
                        self.segment_ref, expected_revision=operation.expected_revision
                    )
                else:
                    applied = target.edit(
                        self.segment_ref,
                        operation.kind,
                        operation.position,
                        operation.position,
                        operation.text,
                        operation.tokens,
                        expected_revision=operation.expected_revision,
                    )
                if not applied:
                    return ()
            return target.finish()


def _window_reasons(window: AlignedWindow, plan: AudioWindowPlan) -> set[SeamReason]:
    reasons: set[SeamReason] = set()
    source = plan.windows[window.window_index]
    hypothesis = window.hypothesis
    if not window.coverage_complete:
        reasons.add(SeamReason.INCOMPLETE)
    if not hypothesis.is_final:
        reasons.add(SeamReason.PROVISIONAL)
    end = 0
    previous_start = previous_end = -1
    seen: set[tuple[int, int]] = set()
    for index, (token, (start, stop)) in enumerate(
        zip(hypothesis.token_times, window.character_offsets)
    ):
        _require(token.token_index == index)
        _require(source.start_ms <= token.start_ms <= token.end_ms <= source.end_ms)
        gap = hypothesis.text[end:start]
        if gap and not gap.isspace():
            reasons.add(SeamReason.UNALIGNED)
        if (
            token.start_ms == token.end_ms
            or (token.start_ms, token.end_ms) in seen
            or token.start_ms < previous_start
            or token.end_ms < previous_end
        ):
            reasons.add(SeamReason.TIMING)
        seen.add((token.start_ms, token.end_ms))
        previous_start, previous_end = token.start_ms, token.end_ms
        end = stop
    if hypothesis.text[end:] and not hypothesis.text[end:].isspace():
        reasons.add(SeamReason.UNALIGNED)
    return reasons


def _tokens(
    window: AlignedWindow, segment_ref: str, first: int, cut: int
) -> tuple[TranscriptToken, ...]:
    return tuple(
        TranscriptToken(
            token_ref=_hash(
                [
                    segment_ref,
                    window.window_index,
                    window.hypothesis.revision,
                    token.token_index,
                    token.start_ms,
                    token.end_ms,
                    start,
                    stop,
                ]
            )[:32],
            start=start - cut,
            end=stop - cut,
            start_ms=token.start_ms,
            end_ms=token.end_ms,
        )
        for token, (start, stop) in zip(
            window.hypothesis.token_times[first:], window.character_offsets[first:]
        )
    )


def _seam(
    previous: AlignedWindow, current: AlignedWindow, plan: AudioWindowPlan
) -> tuple[int, int] | None:
    start = plan.windows[current.window_index].start_ms
    end = plan.windows[previous.window_index].end_ms
    left = [
        token
        for token in previous.hypothesis.token_times
        if token.start_ms < end and token.end_ms > start
    ]
    right = [
        token
        for token in current.hypothesis.token_times
        if token.start_ms < end and token.end_ms > start
    ]
    identity = lambda token: (token.start_ms, token.end_ms, token.token)
    if [identity(token) for token in left] != [identity(token) for token in right]:
        return None
    # An overlap must be a complete previous suffix and incoming prefix. Never
    # delete a repeated phrase found somewhere else in either source window.
    if left and tuple(left) != previous.hypothesis.token_times[-len(left) :]:
        return None
    if right and tuple(right) != current.hypothesis.token_times[: len(right)]:
        return None
    return len(right), current.character_offsets[len(right) - 1][1] if right else 0


def reconcile_windows(
    plan: AudioWindowPlan,
    windows: tuple[AlignedWindow, ...],
    *,
    session_ref: str,
    source_digest: str,
    provider: ProviderProvenance,
) -> ReconciledWindows:
    """Reconcile bounded fixed outputs without ASR, fuzzy deletion or PHI release.

    Args:
        plan: Canonical planner output, limited here to 64 windows.
        windows: Received order, including protected alternatives and explicit alignment.
        session_ref: Opaque 32-hex session identity, never a patient identifier.
        source_digest: Exact audio source digest shared by every window.
        provider: Exact offline provider/model declaration shared by every window.

    Returns:
        A deterministic edit plan. Missing, reordered, nonfinal, conflicting or
        incompletely aligned evidence stays provisional and preserves all original
        alternatives. Timing equality is deliberately exact: uncertain or shifted
        alignments need external review, never a confidence-only deletion rule.

    Raises:
        StitchingError: Invalid shapes, source/provider mismatch or exceeded bounds.
    """
    _require(type(plan) is AudioWindowPlan and 1 <= len(plan.windows) <= 64)
    _require(
        plan
        == plan_audio_windows(
            plan.duration_ms,
            window_ms=plan.window_ms,
            overlap_ms=plan.overlap_ms,
            min_tail_ms=plan.min_tail_ms,
            tail_policy=plan.tail_policy,
        )
    )
    _require(type(session_ref) is str and _REF.fullmatch(session_ref) is not None)
    _require(
        type(source_digest) is str and _DIGEST.fullmatch(source_digest) is not None
    )
    _require(type(provider) is ProviderProvenance and not provider.requires_network)
    _require(type(windows) is tuple and len(windows) <= 64)
    _require(all(type(window) is AlignedWindow for window in windows))
    _require(sum(len(window.hypothesis.text) for window in windows) <= 65_536)
    _require(sum(len(window.hypothesis.token_times) for window in windows) <= 4096)
    _require(
        all(
            window.window_index < len(plan.windows)
            and window.source_digest == source_digest
            and window.provider == provider
            for window in windows
        )
    )
    segment_ref = _hash(
        [
            "asr-window-reconciliation-v1",
            session_ref,
            source_digest,
            provider.to_dict(),
            plan.to_dict(),
        ]
    )[:32]
    reasons: set[SeamReason] = set()
    gaps: set[tuple[int, int]] = set()
    seen: set[int] = set()
    individual = []
    for position, window in enumerate(windows):
        if window.window_index in seen:
            reasons.add(SeamReason.DUPLICATE_WINDOW)
        seen.add(window.window_index)
        if window.window_index != position:
            reasons.add(SeamReason.OUT_OF_ORDER)
            expected = plan.windows[min(position, len(plan.windows) - 1)]
            gaps.add((expected.start_ms, expected.end_ms))
        local = _window_reasons(window, plan)
        reasons.update(local)
        individual.append(local)
    for source in plan.windows:
        if source.window_index not in seen:
            reasons.add(SeamReason.MISSING_WINDOW)
            gaps.add((source.start_ms, source.end_ms))
    if plan.covered_ms < plan.duration_ms:
        reasons.add(SeamReason.UNCOVERED_TAIL)
        gaps.add((plan.covered_ms, plan.duration_ms))
    operations: list[TranscriptEdit] = []
    draft = ""
    for position, window in enumerate(windows):
        if window.window_index != position or individual[position]:
            break
        first = cut = 0
        if position:
            seam = _seam(windows[position - 1], window, plan)
            if seam is None:
                reasons.add(SeamReason.CONFLICT)
                break
            first, cut = seam
        suffix = window.hypothesis.text[cut:]
        if draft and suffix and draft[-1].isalnum() and suffix[0].isalnum():
            reasons.add(SeamReason.BOUNDARY)
            break
        if suffix:
            operations.append(
                TranscriptEdit(
                    kind=RevisionKind.INSERT,
                    expected_revision=len(operations),
                    position=len(draft),
                    text=suffix,
                    tokens=_tokens(window, segment_ref, first, cut),
                )
            )
            draft += suffix
    if not reasons:
        operations.append(
            TranscriptEdit(
                kind=RevisionKind.FINALIZE,
                expected_revision=len(operations),
                position=len(draft),
                text="",
                tokens=(),
            )
        )
    return ReconciledWindows(
        segment_ref=segment_ref,
        review_required=bool(reasons),
        reasons=tuple(sorted(reasons, key=lambda value: value.value)),
        gaps=tuple(sorted(gaps)),
        operations=tuple(operations),
        alternatives=windows,
        _plan=plan,
        _session_ref=session_ref,
        _source_digest=source_digest,
        _provider=provider,
    )
