"""Bounded clock translation for actual local multimodal evidence references."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from fractions import Fraction
from math import ceil, floor

from ..audio_windows import AudioWindowPlan, plan_audio_windows
from ..ecg.evidence_windows import EcgEvidenceWindows
from ..ecg.interval_candidates import CandidateState
from ..ecg.signal_quality import waveform_fingerprint
from ..ecg.waveform import EcgWaveform
from ..evidence.fused_graph import EvidenceNode, NodeKind
from ..streaming.speech.providers import AudioChunk
from ..streaming.transcript.revision_ledger import (
    RevisionLedger,
    RevisionLedgerError,
    SpanReference,
)

CLOCK_VERSION = "openmed.multimodal.clock-alignment.v1"
ALIGNMENT_NOTICE = (
    "Temporal alignment for human review only; not clinical verification or "
    "authenticated acquisition. Bounds depend on qualified host clock and timing "
    "declarations. No automatic clinical decision or evidence approval."
)
_LIMIT = 9_000_000_000_000_000_000
_BILLION = 1_000_000_000
_MILLION = 1_000_000


class ClockAlignmentError(ValueError):
    """Fixed declaration error, without timestamps, content, paths or input values."""


def _require(condition: bool) -> None:
    if not condition:
        raise ClockAlignmentError("invalid multimodal clock declaration")


def _integer(value: object, minimum: int = 0, maximum: int = _LIMIT) -> None:
    _require(type(value) is int and minimum <= value <= maximum)  # type: ignore[operator]


def _digest(value: object) -> None:
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def _ref(value: object) -> None:
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{32}", value) is not None)


def _encode(value: object) -> list[int]:
    _require(type(value) is Fraction)
    assert isinstance(value, Fraction)
    return [value.numerator, value.denominator]


def _json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_encode
    )


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


class BreakKind(str, Enum):
    """Recorded exclusions; none can be bridged by fitting across it."""

    PAUSE = "pause"
    DROPPED = "dropped-chunks"
    WALL_STEP = "wall-clock-step"
    RESET = "clock-reset"


class ClockReview(str, Enum):
    """Controlled reasons for withholding cross-modal translation."""

    ANCHORS = "insufficient-anchors"
    INCONSISTENT = "inconsistent-clock-anchors"
    COVERAGE = "clock-coverage-missing"
    DISCONTINUITY = "clock-discontinuity"
    SPARSE = "anchor-gap-exceeds-policy"
    EPOCH = "clock-epoch-mismatch"
    DECLARATION = "clock-declaration-mismatch"
    WALL_DOMAIN = "wall-reference-mismatch"
    TIMING = "source-timing-missing"
    BINDING = "source-binding-mismatch"
    REFERENCE = "invalid-source-reference"
    UNFINALIZED = "transcript-not-finalized"
    EVIDENCE = "source-requires-review"
    UNCERTAINTY = "alignment-uncertainty-exceeds-policy"
    NO_OVERLAP = "evidence-does-not-overlap"
    POSSIBLE_OVERLAP = "overlap-is-uncertain"
    CHANGED = "source-changed-during-alignment"


class AlignmentState(str, Enum):
    """Temporal agreement never implies clinical evidence approval."""

    ALIGNED = "aligned-for-review"
    REVIEW = "review-required"


class Modality(str, Enum):
    """Actual source contract adapted into its declared clock domain."""

    AUDIO = "audio"
    TRANSCRIPT = "transcript"
    ECG = "ecg"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class ClockAnchor:
    """Host-observed monotonic/wall correspondence with an absolute error bound.

    Args:
        monotonic_ns: Nominal source-clock nanoseconds within one reset epoch.
        wall_ns: Nanoseconds in the explicitly declared shared wall reference.
        uncertainty_ns: Qualified total anchor error, including capture latency.
    """

    monotonic_ns: int = field(repr=False)
    wall_ns: int = field(repr=False)
    uncertainty_ns: int = field(repr=False)

    def __post_init__(self) -> None:
        _integer(self.monotonic_ns)
        _integer(self.wall_ns, -_LIMIT)
        _integer(self.uncertainty_ns, maximum=10**12)


@dataclass(frozen=True, slots=True)
class ClockBreak:
    """Inclusive excluded monotonic interval, or a point when start equals end.

    A counter that actually goes backwards needs a new ClockEpoch identifier.
    This declaration does not stitch reset counters into a continuous timeline.
    """

    kind: BreakKind
    start_ns: int = field(repr=False)
    end_ns: int = field(repr=False)

    def __post_init__(self) -> None:
        _require(type(self.kind) is BreakKind)
        _integer(self.start_ns)
        _integer(self.end_ns, self.start_ns)


@dataclass(frozen=True, slots=True, kw_only=True)
class ClockEpoch:
    """One nominal monotonic clock epoch, with explicit qualified model bounds.

    Args:
        clock_ref: Opaque clock identity, never a device/patient name.
        epoch_ref: Unique reset epoch; unrelated counters must not reuse it.
        wall_domain_digest: Shared wall-reference definition/calibration digest.
        calibration_digest: Host qualification evidence for these clock assumptions.
        anchors: Up to 64 strictly increasing monotonic observations, not sorted here.
        maximum_rate_error_ppb: Allowed affine slope deviation from one, in ppb.
        nonlinearity_ns: Absolute deviation from the affine model anywhere between
            anchors, independently qualified by the host, not inferred from a fit.
        breaks: Up to 64 ordered, nonoverlapping discontinuities/exclusions.

    A feasible fit is conditional evidence, not proof of undisclosed drift,
    accurate wall time or authenticated acquisition. No clock or network is read.
    """

    clock_ref: str
    epoch_ref: str
    wall_domain_digest: str
    calibration_digest: str
    anchors: tuple[ClockAnchor, ...] = field(repr=False)
    maximum_rate_error_ppb: int = field(repr=False)
    nonlinearity_ns: int = field(repr=False)
    breaks: tuple[ClockBreak, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _ref(self.clock_ref)
        _ref(self.epoch_ref)
        _digest(self.wall_domain_digest)
        _digest(self.calibration_digest)
        _integer(self.maximum_rate_error_ppb, maximum=_BILLION - 1)
        _integer(self.nonlinearity_ns, maximum=10**12)
        _require(type(self.anchors) is tuple and len(self.anchors) <= 64)
        _require(all(type(item) is ClockAnchor for item in self.anchors))
        _require(
            all(
                a.monotonic_ns < b.monotonic_ns
                for a, b in zip(self.anchors, self.anchors[1:])
            )
        )
        _require(type(self.breaks) is tuple and len(self.breaks) <= 64)
        _require(all(type(item) is ClockBreak for item in self.breaks))
        _require(
            all(a.end_ns < b.start_ns for a, b in zip(self.breaks, self.breaks[1:]))
        )
        _require(
            not any(
                gap.start_ns <= anchor.monotonic_ns <= gap.end_ns
                for gap in self.breaks
                for anchor in self.anchors
            )
        )

    @property
    def digest(self) -> str:
        """Bind all observations, discontinuities and declared model assumptions."""
        return _hash({"algorithm": CLOCK_VERSION, "epoch": asdict(self)})


@dataclass(frozen=True, slots=True, kw_only=True)
class AlignmentPolicy:
    """Host-qualified limits; no universal or clinically recommended defaults.

    Args:
        version: Opaque version for replay, never clinical free text.
        evidence_digest: Digest of the intended-task policy evaluation.
        maximum_uncertainty_ns: Maximum full endpoint envelope width, including
            source/target clocks and both references' timing uncertainty.
        maximum_anchor_gap_ns: Maximum adjacent bracketing gap used for translation.
    """

    version: str
    evidence_digest: str
    maximum_uncertainty_ns: int
    maximum_anchor_gap_ns: int

    def __post_init__(self) -> None:
        _require(
            type(self.version) is str
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.version)
            is not None
        )
        _digest(self.evidence_digest)
        _integer(self.maximum_uncertainty_ns, maximum=10**12)
        _integer(self.maximum_anchor_gap_ns, 1)

    @property
    def digest(self) -> str:
        """Bind the complete translation policy and algorithm version."""
        return _hash({"algorithm": CLOCK_VERSION, "policy": asdict(self)})


@dataclass(frozen=True, slots=True)
class NanosecondBound:
    """Protected inclusive nanosecond envelope, always rounded outwards."""

    lower: int = field(repr=False)
    upper: int = field(repr=False)

    def __post_init__(self) -> None:
        _integer(self.lower, -_LIMIT)
        _integer(self.upper, self.lower)


_Vertex = tuple[Fraction, Fraction]


def _clip(
    vertices: tuple[_Vertex, ...], x: Fraction, y: Fraction, limit: Fraction
) -> tuple[_Vertex, ...]:
    """Intersect a convex (possibly degenerate) polygon with x*a+y*b <= limit."""
    if not vertices:
        return ()
    result: list[_Vertex] = []
    previous = vertices[-1]
    previous_d = x * previous[0] + y * previous[1] - limit
    for current in vertices:
        current_d = x * current[0] + y * current[1] - limit
        if (previous_d <= 0) != (current_d <= 0):
            fraction = previous_d / (previous_d - current_d)
            result.append(
                (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                )
            )
        if current_d <= 0:
            result.append(current)
        previous, previous_d = current, current_d
    return tuple(dict.fromkeys(result))


@dataclass(frozen=True, slots=True)
class ClockFit:
    """Protected exact feasible affine family for one uninterrupted anchor group."""

    anchors: tuple[ClockAnchor, ...] = field(repr=False)
    vertices: tuple[_Vertex, ...] = field(repr=False)
    nonlinearity_ns: int = field(repr=False)
    reason: ClockReview | None

    @property
    def drift_estimate_ppb(self) -> Fraction | None:
        """Centroid-of-vertices slope estimate, not a calibrated probability."""
        if not self.vertices:
            return None
        total = sum((a for a, _ in self.vertices), Fraction(0))
        return (total / len(self.vertices) - 1) * _BILLION

    @property
    def drift_bounds_ppb(self) -> tuple[Fraction, Fraction] | None:
        """Full feasible slope interval, rather than uncertainty from fit residuals."""
        if not self.vertices:
            return None
        return (
            (min(a for a, _ in self.vertices) - 1) * _BILLION,
            (max(a for a, _ in self.vertices) - 1) * _BILLION,
        )


@dataclass(frozen=True, slots=True)
class ClockCalibration:
    """Local calibration evidence; protected fits retain no clinical payload."""

    epoch: ClockEpoch = field(repr=False)
    fits: tuple[ClockFit, ...] = field(repr=False)

    def to_report_json(self) -> str:
        """Report only model identity, counts and controlled fit findings."""
        return _json(
            {
                "algorithm": CLOCK_VERSION,
                "clock_digest": self.epoch.digest,
                "fit_count": len(self.fits),
                "anchor_count": len(self.epoch.anchors),
                "break_count": len(self.epoch.breaks),
                "findings": [
                    item.reason for item in self.fits if item.reason is not None
                ],
                "notice": ALIGNMENT_NOTICE,
            }
        )


def calibrate_clock(epoch: ClockEpoch) -> ClockCalibration:
    """Fit every uninterrupted anchor group using exact affine constraints.

    Args:
        epoch: Actual observations and explicit model qualification declarations.

    Returns:
        All feasible slope/intercept polygons or controlled review findings. A
        discontinuity divides groups before fitting; no observations are dropped.

    Raises:
        ClockAlignmentError: For an invalid epoch declaration.
    """
    _require(type(epoch) is ClockEpoch)
    groups: list[list[ClockAnchor]] = [[] for _ in range(len(epoch.breaks) + 1)]
    for anchor in epoch.anchors:
        groups[sum(gap.end_ns < anchor.monotonic_ns for gap in epoch.breaks)].append(
            anchor
        )
    fits = []
    drift = Fraction(epoch.maximum_rate_error_ppb, _BILLION)
    for group in groups:
        anchors = tuple(group)
        if len(anchors) < 2:
            fits.append(
                ClockFit(anchors, (), epoch.nonlinearity_ns, ClockReview.ANCHORS)
            )
            continue
        origin = anchors[0]
        error = Fraction(origin.uncertainty_ns + epoch.nonlinearity_ns)
        vertices = tuple(
            dict.fromkeys(
                (
                    (1 - drift, -error),
                    (1 + drift, -error),
                    (1 + drift, error),
                    (1 - drift, error),
                )
            )
        )
        for anchor in anchors:
            x = Fraction(anchor.monotonic_ns - origin.monotonic_ns)
            y = Fraction(anchor.wall_ns - origin.wall_ns)
            error = Fraction(anchor.uncertainty_ns + epoch.nonlinearity_ns)
            vertices = _clip(vertices, x, Fraction(1), y + error)
            vertices = _clip(vertices, -x, Fraction(-1), -y + error)
        fits.append(
            ClockFit(
                anchors,
                vertices,
                epoch.nonlinearity_ns,
                None if vertices else ClockReview.INCONSISTENT,
            )
        )
    return ClockCalibration(epoch, tuple(fits))


@dataclass(frozen=True, slots=True, kw_only=True)
class ClockBinding:
    """Declared continuous asset-time to monotonic-clock offset and timing error.

    Args:
        source_digest: Exact source identity expected by the modality adapter.
        clock_ref: Clock identity, matching the calibration.
        epoch_ref: Reset epoch, never inferred from matching numeric timestamps.
        evidence_digest: Host's timing/provenance manifest digest.
        local_start_ns: Start of the continuous asset-time interval covered.
        local_end_ns: End of that interval, exclusive for evidence windows.
        monotonic_start_ns: Clock reading corresponding to local_start_ns.
        timing_uncertainty_ns: Qualified boundary uncertainty for the source,
            including token alignment or capture timing, not just clock error.

    The mapping has nominal unit slope. Paused/compressed media needs separate
    bindings, not inferred continuity. Source rates are nominal; oscillator drift
    is handled by the clock model. Declarations do not authenticate provenance.
    """

    source_digest: str
    clock_ref: str
    epoch_ref: str
    evidence_digest: str
    local_start_ns: int = field(repr=False)
    local_end_ns: int = field(repr=False)
    monotonic_start_ns: int = field(repr=False)
    timing_uncertainty_ns: int = field(repr=False)

    def __post_init__(self) -> None:
        _digest(self.source_digest)
        _digest(self.evidence_digest)
        _ref(self.clock_ref)
        _ref(self.epoch_ref)
        _integer(self.local_start_ns)
        _integer(self.local_end_ns, self.local_start_ns + 1)
        _integer(self.monotonic_start_ns)
        _integer(self.timing_uncertainty_ns, maximum=10**12)
        _require(
            self.monotonic_start_ns + self.local_end_ns - self.local_start_ns <= _LIMIT
        )


@dataclass(frozen=True, slots=True)
class AudioPlanEvidence:
    """Exact planned window over a host-declared audio source; no audio decoding."""

    plan: AudioWindowPlan = field(repr=False)
    window_id: str = field(repr=False)
    binding: ClockBinding = field(repr=False)


@dataclass(frozen=True, slots=True)
class AudioChunkEvidence:
    """Encoded chunk with explicit local start; sequence numbers are not clocks.

    local_start_ns=None means timing is unknown. Duration remains the provider's
    declared duration; neither encoding nor duration is inferred from byte count.
    binding.source_digest must equal audio_chunk_digest(chunk).
    """

    chunk: AudioChunk = field(repr=False)
    local_start_ns: int | None = field(repr=False)
    binding: ClockBinding = field(repr=False)


def audio_chunk_digest(chunk: AudioChunk) -> str:
    """Bind exact encoded bytes and their typed declaration without logging either."""
    _require(type(chunk) is AudioChunk)
    return _hash(
        {
            "payload_digest": hashlib.sha256(chunk.payload).hexdigest(),
            "declaration": chunk.to_dict(),
        }
    )


def transcript_source_digest(ledger: RevisionLedger) -> str:
    """Identify a ledger by its opaque session/segment, not by clinical text."""
    _require(type(ledger) is RevisionLedger)
    state = ledger.to_dict()
    return _hash(
        {"session_ref": state["session_ref"], "segment_ref": state["segment_ref"]}
    )


@dataclass(frozen=True, slots=True)
class TranscriptEvidence:
    """Registered ledger citation, freshly resolved for each alignment attempt."""

    ledger: RevisionLedger = field(repr=False)
    reference: SpanReference = field(repr=False)
    binding: ClockBinding = field(repr=False)


@dataclass(frozen=True, slots=True)
class EcgEvidence:
    """Exact ECG evidence-window member and current original waveform."""

    windows: EcgEvidenceWindows = field(repr=False)
    waveform: EcgWaveform = field(repr=False)
    window_digest: str = field(repr=False)
    binding: ClockBinding = field(repr=False)


@dataclass(frozen=True, slots=True)
class ImageEvidence:
    """Source image/OCR geometry plus explicit host-declared capture/exposure time.

    Unknown times require review. A page/frame ordinal never supplies timing.
    The timing manifest and clock binding are host evidence, not authenticated
    pixel capture. No pixels or OCR values are loaded by this adapter.
    """

    node: EvidenceNode = field(repr=False)
    local_start_ns: int | None = field(repr=False)
    local_end_ns: int | None = field(repr=False)
    capture_digest: str = field(repr=False)
    binding: ClockBinding = field(repr=False)


TimedEvidence = (
    AudioPlanEvidence
    | AudioChunkEvidence
    | TranscriptEvidence
    | EcgEvidence
    | ImageEvidence
)


@dataclass(frozen=True, slots=True)
class _Snapshot:
    modality: Modality
    digest: str
    binding: ClockBinding = field(repr=False)
    start: NanosecondBound | None = field(repr=False)
    end: NanosecondBound | None = field(repr=False)
    reasons: tuple[ClockReview, ...]


def _bound_snapshot(
    modality: Modality,
    binding: ClockBinding,
    identity: object,
    start: Fraction | int | None,
    end: Fraction | int | None,
    reasons: tuple[ClockReview, ...] = (),
) -> _Snapshot:
    _require(type(binding) is ClockBinding)
    digest = _hash(
        {
            "modality": modality,
            "identity": identity,
            "binding": asdict(binding),
            "start": start,
            "end": end,
            "reasons": reasons,
        }
    )
    if reasons:
        return _Snapshot(modality, digest, binding, None, None, reasons)
    if start is None or end is None:
        return _Snapshot(modality, digest, binding, None, None, (ClockReview.TIMING,))
    uncertainty = binding.timing_uncertainty_ns
    if not (
        binding.local_start_ns
        <= start - uncertainty
        < end + uncertainty
        <= binding.local_end_ns
    ):
        return _Snapshot(modality, digest, binding, None, None, (ClockReview.BINDING,))
    offset = binding.monotonic_start_ns - binding.local_start_ns
    first = NanosecondBound(
        floor(start - uncertainty) + offset, ceil(start + uncertainty) + offset
    )
    last = NanosecondBound(
        floor(end - uncertainty) + offset, ceil(end + uncertainty) + offset
    )
    if first.upper >= last.lower:
        return _Snapshot(
            modality, digest, binding, None, None, (ClockReview.UNCERTAINTY,)
        )
    return _Snapshot(modality, digest, binding, first, last, ())


def _snapshot(evidence: TimedEvidence) -> _Snapshot:
    _require(
        type(evidence)
        in (
            AudioPlanEvidence,
            AudioChunkEvidence,
            TranscriptEvidence,
            EcgEvidence,
            ImageEvidence,
        )
    )
    binding = evidence.binding
    _require(type(binding) is ClockBinding)
    if type(evidence) is AudioPlanEvidence:
        plan = evidence.plan
        _require(type(plan) is AudioWindowPlan and type(evidence.window_id) is str)
        # Regenerate the actual planner output, rather than trusting editable DTO fields.
        expected = plan_audio_windows(
            plan.duration_ms,
            window_ms=plan.window_ms,
            overlap_ms=plan.overlap_ms,
            min_tail_ms=plan.min_tail_ms,
            tail_policy=plan.tail_policy,
        )
        _require(plan == expected)
        window = next(
            (item for item in plan.windows if item.window_id == evidence.window_id),
            None,
        )
        reasons = () if window is not None else (ClockReview.REFERENCE,)
        return _bound_snapshot(
            Modality.AUDIO,
            binding,
            {"plan": plan.to_dict(), "window_id": evidence.window_id},
            window.start_ms * _MILLION if window else None,
            window.end_ms * _MILLION if window else None,
            reasons,
        )
    if type(evidence) is AudioChunkEvidence:
        _require(type(evidence.chunk) is AudioChunk)
        if evidence.local_start_ns is not None:
            _integer(evidence.local_start_ns)
        identity = audio_chunk_digest(evidence.chunk)
        reasons = () if identity == binding.source_digest else (ClockReview.BINDING,)
        return _bound_snapshot(
            Modality.AUDIO,
            binding,
            identity,
            evidence.local_start_ns,
            None
            if evidence.local_start_ns is None
            else evidence.local_start_ns + evidence.chunk.duration_ms * _MILLION,
            reasons,
        )
    if type(evidence) is TranscriptEvidence:
        _require(
            type(evidence.ledger) is RevisionLedger
            and type(evidence.reference) is SpanReference
        )
        transcript_identity = {
            "ledger": evidence.ledger.to_dict(),
            "reference": asdict(evidence.reference),
        }
        mapped = None
        try:
            mapped = evidence.ledger.remap_span(evidence.reference)
        except RevisionLedgerError:
            pass
        reasons = ()
        if binding.source_digest != transcript_source_digest(evidence.ledger):
            reasons = (ClockReview.BINDING,)
        elif mapped is None or not mapped.valid:
            reasons = (ClockReview.REFERENCE,)
        elif not mapped.finalized:
            reasons = (ClockReview.UNFINALIZED,)
        return _bound_snapshot(
            Modality.TRANSCRIPT,
            binding,
            transcript_identity,
            mapped.start_ms * _MILLION
            if mapped is not None and mapped.start_ms is not None
            else None,
            mapped.end_ms * _MILLION
            if mapped is not None and mapped.end_ms is not None
            else None,
            reasons,
        )
    if type(evidence) is EcgEvidence:
        _require(
            type(evidence.windows) is EcgEvidenceWindows
            and type(evidence.waveform) is EcgWaveform
        )
        _digest(evidence.window_digest)
        windows, waveform = evidence.windows, evidence.waveform
        ecg_window = next(
            (item for item in windows.windows if item.digest == evidence.window_digest),
            None,
        )
        candidate = next(
            (
                item
                for item in windows.candidates.measurements
                if ecg_window is not None
                and _hash(asdict(item)) == ecg_window.candidate_digest
                and item.kind is ecg_window.kind
            ),
            None,
        )
        reasons = ()
        if (
            binding.source_digest != waveform.provenance.source_digest
            or waveform_fingerprint(waveform) != windows.candidates.waveform_digest
            or windows.candidates.source_digest != waveform.provenance.source_digest
        ):
            reasons = (ClockReview.BINDING,)
        elif ecg_window is None or candidate is None:
            reasons = (ClockReview.REFERENCE,)
        elif (
            ecg_window.state is CandidateState.ABSTAINED
            or candidate.state is CandidateState.ABSTAINED
            or not windows.candidates.quality.measurement_allowed
        ):
            reasons = (ClockReview.EVIDENCE,)
        rate = Fraction(str(waveform.sample_rate_hz))
        origin = Fraction(str(waveform.start_offset_seconds)) * _BILLION
        return _bound_snapshot(
            Modality.ECG,
            binding,
            {"bundle": windows.digest, "window": evidence.window_digest},
            origin + ecg_window.start_sample * _BILLION / rate if ecg_window else None,
            origin + ecg_window.end_sample * _BILLION / rate if ecg_window else None,
            reasons,
        )
    assert type(evidence) is ImageEvidence
    _require(
        type(evidence.node) is EvidenceNode
        and evidence.node.kind in (NodeKind.PAGE, NodeKind.PIXELS, NodeKind.OCR)
    )
    _digest(evidence.capture_digest)
    _require((evidence.local_start_ns is None) == (evidence.local_end_ns is None))
    if evidence.local_start_ns is not None:
        _integer(evidence.local_start_ns)
        _integer(evidence.local_end_ns, evidence.local_start_ns + 1)
    reasons = (
        ()
        if binding.source_digest == evidence.node.source_digest
        else (ClockReview.BINDING,)
    )
    return _bound_snapshot(
        Modality.IMAGE,
        binding,
        {"node": asdict(evidence.node), "capture": evidence.capture_digest},
        evidence.local_start_ns,
        evidence.local_end_ns,
        reasons,
    )


def _fit_for(
    calibration: ClockCalibration, start: int, end: int, policy: AlignmentPolicy
) -> tuple[ClockFit | None, ClockReview | None]:
    if any(
        start <= gap.end_ns and gap.start_ns <= end for gap in calibration.epoch.breaks
    ):
        return None, ClockReview.DISCONTINUITY
    fit = next(
        (
            item
            for item in calibration.fits
            if item.anchors
            and item.anchors[0].monotonic_ns
            <= start
            <= end
            <= item.anchors[-1].monotonic_ns
        ),
        None,
    )
    if fit is None:
        reason = (
            ClockReview.ANCHORS
            if len(calibration.epoch.anchors) < 2
            else ClockReview.COVERAGE
        )
        return None, reason
    if fit.reason is not None:
        return None, fit.reason
    if any(
        b.monotonic_ns - a.monotonic_ns > policy.maximum_anchor_gap_ns
        for a, b in zip(fit.anchors, fit.anchors[1:])
        if a.monotonic_ns < end and b.monotonic_ns > start
    ):
        return None, ClockReview.SPARSE
    return fit, None


def _forward(fit: ClockFit, point: NanosecondBound) -> tuple[Fraction, Fraction]:
    origin = fit.anchors[0]
    lower = min(a * (point.lower - origin.monotonic_ns) + b for a, b in fit.vertices)
    upper = max(a * (point.upper - origin.monotonic_ns) + b for a, b in fit.vertices)
    return (
        lower + origin.wall_ns - fit.nonlinearity_ns,
        upper + origin.wall_ns + fit.nonlinearity_ns,
    )


def _inverse(fit: ClockFit, wall: tuple[Fraction, Fraction]) -> NanosecondBound | None:
    origin = fit.anchors[0]
    lower = min(
        (wall[0] - origin.wall_ns - b - fit.nonlinearity_ns) / a
        for a, b in fit.vertices
    )
    upper = max(
        (wall[1] - origin.wall_ns - b + fit.nonlinearity_ns) / a
        for a, b in fit.vertices
    )
    first, last = floor(lower) + origin.monotonic_ns, ceil(upper) + origin.monotonic_ns
    if first < -_LIMIT or last > _LIMIT:
        return None
    return NanosecondBound(first, last)


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Temporal translation into the target monotonic clock, never asset indices.

    The guaranteed overlap is a half-open interval inside both evidence windows.
    Translated endpoints are inclusive uncertainty envelopes. Review states
    withhold all translated timestamps. No clinical payload or wall time is kept.
    """

    state: AlignmentState
    reasons: tuple[ClockReview, ...]
    source_reference_digest: str
    target_reference_digest: str
    source_clock_digest: str
    target_clock_digest: str
    policy_digest: str
    target_start_ns: NanosecondBound | None = field(repr=False)
    target_end_ns: NanosecondBound | None = field(repr=False)
    guaranteed_overlap_ns: tuple[int, int] | None = field(repr=False)
    uncertainty_ns: int | None = field(repr=False)

    @property
    def digest(self) -> str:
        """Bind the complete protected alignment and its provenance."""
        return _hash({"algorithm": CLOCK_VERSION, "result": asdict(self)})

    def to_report_json(self) -> str:
        """Return decisions and hashes only, with no timestamps or signal/text."""
        return _json(
            {
                "algorithm": CLOCK_VERSION,
                "state": self.state,
                "reasons": self.reasons,
                "source_reference_digest": self.source_reference_digest,
                "target_reference_digest": self.target_reference_digest,
                "source_clock_digest": self.source_clock_digest,
                "target_clock_digest": self.target_clock_digest,
                "policy_digest": self.policy_digest,
                "result_digest": self.digest,
                "notice": ALIGNMENT_NOTICE,
            }
        )


def align_evidence(
    source: TimedEvidence,
    target: TimedEvidence,
    source_clock: ClockEpoch,
    target_clock: ClockEpoch,
    policy: AlignmentPolicy,
) -> AlignmentResult:
    """Translate actual source evidence to a target clock with conservative bounds.

    Args:
        source: Actual audio, registered transcript, ECG or image evidence adapter.
        target: Actual target reference; selects the intended uninterrupted segment.
        source_clock: Source monotonic epoch and qualified anchor/model evidence.
        target_clock: Target epoch in the same declared wall-reference domain.
        policy: Maximum uncertainty and bracketing-anchor gap for the intended task.

    Returns:
        Protected target-clock endpoints and guaranteed evidence overlap only when
        all declarations, current references, coverage and uncertainty gates pass.
        Otherwise review-required with controlled reasons and no translated times.

    Raises:
        ClockAlignmentError: For invalid declarations. Source-contract validators
            may also refuse malformed typed inputs with their value-free errors.

    All calculations are local and deterministic. No source is sampled, decoded,
    persisted or sent to a model/service, and no clinical evidence is approved.
    """
    _require(type(source_clock) is ClockEpoch and type(target_clock) is ClockEpoch)
    _require(type(policy) is AlignmentPolicy)
    first, second = _snapshot(source), _snapshot(target)

    def result(
        reasons: tuple[ClockReview, ...],
        start: NanosecondBound | None = None,
        end: NanosecondBound | None = None,
        overlap: tuple[int, int] | None = None,
        uncertainty: int | None = None,
    ) -> AlignmentResult:
        return AlignmentResult(
            AlignmentState.REVIEW if reasons else AlignmentState.ALIGNED,
            tuple(dict.fromkeys(reasons)),
            first.digest,
            second.digest,
            source_clock.digest,
            target_clock.digest,
            policy.digest,
            start,
            end,
            overlap,
            uncertainty,
        )

    reasons = first.reasons + second.reasons
    if reasons:
        return result(reasons)
    for snapshot, epoch in ((first, source_clock), (second, target_clock)):
        if (snapshot.binding.clock_ref, snapshot.binding.epoch_ref) != (
            epoch.clock_ref,
            epoch.epoch_ref,
        ):
            return result((ClockReview.EPOCH,))
    assert first.start is not None and first.end is not None
    assert second.start is not None and second.end is not None
    if (source_clock.clock_ref, source_clock.epoch_ref) == (
        target_clock.clock_ref,
        target_clock.epoch_ref,
    ):
        if source_clock.digest != target_clock.digest:
            return result((ClockReview.DECLARATION,))
        if any(
            start.lower <= gap.end_ns and gap.start_ns <= end.upper
            for start, end in ((first.start, first.end), (second.start, second.end))
            for gap in source_clock.breaks
        ):
            return result((ClockReview.DISCONTINUITY,))
        # Already in the identical nominal monotonic domain; no wall-time claim.
        start, end = first.start, first.end
    else:
        if source_clock.wall_domain_digest != target_clock.wall_domain_digest:
            return result((ClockReview.WALL_DOMAIN,))
        source_calibration, target_calibration = (
            calibrate_clock(source_clock),
            calibrate_clock(target_clock),
        )
        source_fit, source_reason = _fit_for(
            source_calibration, first.start.lower, first.end.upper, policy
        )
        target_fit, target_reason = _fit_for(
            target_calibration, second.start.lower, second.end.upper, policy
        )
        if source_reason or target_reason:
            return result(
                tuple(
                    reason
                    for reason in (source_reason, target_reason)
                    if reason is not None
                )
            )
        assert source_fit is not None and target_fit is not None
        mapped_start = _inverse(target_fit, _forward(source_fit, first.start))
        mapped_end = _inverse(target_fit, _forward(source_fit, first.end))
        if mapped_start is None or mapped_end is None:
            return result((ClockReview.COVERAGE,))
        start, end = mapped_start, mapped_end
        translated_fit, reason = _fit_for(
            target_calibration, start.lower, end.upper, policy
        )
        if reason is not None:
            return result((reason,))
        if translated_fit is not target_fit:
            return result((ClockReview.DISCONTINUITY,))
    uncertainty = max(start.upper - start.lower, end.upper - end.lower) + max(
        second.start.upper - second.start.lower, second.end.upper - second.end.lower
    )
    if uncertainty > policy.maximum_uncertainty_ns:
        return result((ClockReview.UNCERTAINTY,))
    overlap = (max(start.upper, second.start.upper), min(end.lower, second.end.lower))
    if overlap[0] >= overlap[1]:
        possible = max(start.lower, second.start.lower) < min(
            end.upper, second.end.upper
        )
        return result(
            (ClockReview.POSSIBLE_OVERLAP if possible else ClockReview.NO_OVERLAP,)
        )
    if _snapshot(source) != first or _snapshot(target) != second:
        return result((ClockReview.CHANGED,))
    return result((), start, end, overlap, uncertainty)
