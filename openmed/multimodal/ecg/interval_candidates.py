"""Conservative, local ECG timing candidates for human review, without diagnosis."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from statistics import median

from .signal_quality import (
    AcquisitionEvidence,
    SignalQualityPolicy,
    SignalQualityReport,
    run_gated_measurement,
)
from .waveform import EcgLead, EcgWaveform

ALGORITHM_VERSION = "openmed.ecg.excursion-intervals.v1"
REVIEW_NOTICE = (
    "Non-diagnostic timing candidates for human review only. Agreement is not "
    "clinical accuracy or a calibrated probability. No disease labels, QT "
    "correction, treatment recommendations or automatic clinical decisions."
)


class IntervalCandidateError(ValueError):
    """Fixed, value-free declaration error."""


def _require(condition: bool) -> None:
    if not condition:
        raise IntervalCandidateError("invalid ECG interval policy or input")


def _number(value: object, *, positive: bool = True) -> None:
    _require(type(value) in (int, float))
    _require(0 < value <= 1e9 if positive else 0 <= value <= 1e9)  # type: ignore[operator]


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class SampleRange:
    """Inclusive sample-domain limits, qualified at the policy's exact rate.

    Args:
        minimum: Positive lower limit.
        maximum: Upper limit, at most one million samples.
    """

    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        _require(type(self.minimum) is int and type(self.maximum) is int)
        _require(0 < self.minimum <= self.maximum <= 1_000_000)

    def contains(self, value: float) -> bool:
        """Whether a value falls inside this declared range, including endpoints."""
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True, slots=True, kw_only=True)
class IntervalPolicy:
    """Explicit detector settings; no clinical defaults or qualification implied.

    Args:
        version: Opaque policy identifier, never patient/device free text.
        evidence_digest: SHA-256 of the host's evaluation evidence.
        sample_rate_hz: Exact sampling rate for these sample-domain settings.
        boundary_low_mv: Lower excursion threshold above the record median.
        boundary_high_mv: Upper threshold used to measure boundary sensitivity.
        secondary_peak_mv: Minimum absolute P/T candidate excursion amplitude.
        qrs_peak_mv: Minimum absolute ventricular candidate amplitude.
        qrs_slope_mv_per_sample: Minimum ventricular candidate adjacent change.
        quiet_samples: Required baseline samples on each side of every excursion.
        join_samples: Maximum internal quiet gap joined into one excursion.
        baseline_fraction: Required fraction within the low threshold, at least half.
        qrs_width: Allowed full excursion duration, in samples.
        secondary_width: Allowed P/T full excursion duration, in samples.
        rr_spacing: Allowed ventricular peak spacing, in samples; no beat filling.
        p_search: Lookback distances from QRS onset for a complete P excursion.
        t_search: Forward distances from QRS offset for a complete T excursion.
        boundary_uncertainty_samples: Maximum threshold/plateau uncertainty.
        alignment_samples: Maximum cross-lead ventricular peak displacement.
        minimum_beats: Required detected beats per lead, between 2 and 256.
        minimum_leads: Required agreeing leads, between 2 and 12.
        rate_spread_bpm: Allowed full envelope of rate candidates across all beats/leads.
        interval_spread_ms: Allowed full envelope for each PR/QRS/QT candidate.

    This narrow detector requires separated, baseline-returning excursions with
    a dominant, sufficiently steep ventricular complex. The host must evaluate
    these assumptions and all thresholds for its acquisition and intended task.
    """

    version: str
    evidence_digest: str
    sample_rate_hz: float
    boundary_low_mv: float
    boundary_high_mv: float
    secondary_peak_mv: float
    qrs_peak_mv: float
    qrs_slope_mv_per_sample: float
    quiet_samples: int
    join_samples: int
    baseline_fraction: float
    qrs_width: SampleRange
    secondary_width: SampleRange
    rr_spacing: SampleRange
    p_search: SampleRange
    t_search: SampleRange
    boundary_uncertainty_samples: int
    alignment_samples: int
    minimum_beats: int
    minimum_leads: int
    rate_spread_bpm: float
    interval_spread_ms: float

    def __post_init__(self) -> None:
        _require(
            type(self.version) is str
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.version)
            is not None
        )
        _require(
            type(self.evidence_digest) is str
            and re.fullmatch(r"[0-9a-f]{64}", self.evidence_digest) is not None
        )
        for value in (
            self.sample_rate_hz,
            self.boundary_low_mv,
            self.boundary_high_mv,
            self.secondary_peak_mv,
            self.qrs_peak_mv,
            self.qrs_slope_mv_per_sample,
            self.baseline_fraction,
            self.rate_spread_bpm,
            self.interval_spread_ms,
        ):
            _number(value)
        _require(self.sample_rate_hz <= 100_000)
        _require(
            self.boundary_low_mv
            < self.boundary_high_mv
            < self.secondary_peak_mv
            < self.qrs_peak_mv
        )
        _require(0.5 <= self.baseline_fraction <= 1)
        for value in (
            self.quiet_samples,
            self.boundary_uncertainty_samples,
            self.alignment_samples,
        ):
            _require(type(value) is int and 1 <= value <= 1_000_000)
        _require(
            type(self.join_samples) is int
            and 0 <= self.join_samples < self.quiet_samples
        )
        _require(type(self.minimum_beats) is int and 2 <= self.minimum_beats <= 256)
        _require(type(self.minimum_leads) is int and 2 <= self.minimum_leads <= 12)
        for limits in (
            self.qrs_width,
            self.secondary_width,
            self.rr_spacing,
            self.p_search,
            self.t_search,
        ):
            _require(type(limits) is SampleRange)
        _require(self.qrs_width.maximum < self.rr_spacing.minimum)
        _require(self.p_search.minimum >= self.quiet_samples)
        _require(self.t_search.minimum >= self.quiet_samples)

    @property
    def digest(self) -> str:
        """Bind every setting, declared evaluation evidence and algorithm version."""
        return _hash({"algorithm": ALGORITHM_VERSION, "policy": asdict(self)})


class CandidateState(str, Enum):
    """Neither state permits automatic clinical use."""

    REVIEW = "review-candidate"
    ABSTAINED = "abstained"


class TimingKind(str, Enum):
    """Uncorrected measurements, with no diagnostic categories."""

    RATE = "rate-bpm"
    PR = "pr-ms"
    QRS = "qrs-ms"
    QT = "qt-ms"


class Abstention(str, Enum):
    """Controlled reasons; no sample values or acquisition free text."""

    MISSING = "missing-samples"
    BASELINE = "baseline-unresolved"
    LIMIT = "excursion-limit"
    QRS = "ventricular-boundaries-unresolved"
    BEATS = "insufficient-beats"
    SPACING = "ventricular-spacing-outside-policy"
    LEADS = "insufficient-leads"
    ALIGNMENT = "lead-beat-disagreement"
    P = "p-boundaries-unresolved"
    T = "t-boundaries-unresolved"
    SPREAD = "measurement-disagreement"


@dataclass(frozen=True, slots=True)
class SampleBound:
    """Protected inclusive sample-index uncertainty; not a clinical error bound."""

    lower: int = field(repr=False)
    upper: int = field(repr=False)


@dataclass(frozen=True, slots=True)
class WaveBoundary:
    """Original-grid excursion boundaries and dominant absolute peak candidate."""

    onset: SampleBound = field(repr=False)
    peak: SampleBound = field(repr=False)
    offset: SampleBound = field(repr=False)
    window_start: int = field(repr=False)
    window_end: int = field(repr=False)


@dataclass(frozen=True, slots=True)
class BeatEvidence:
    """Lead-local wave candidates; absent P/T boundaries are never interpolated."""

    qrs: WaveBoundary = field(repr=False)
    p: WaveBoundary | None = field(repr=False)
    t: WaveBoundary | None = field(repr=False)


@dataclass(frozen=True, slots=True)
class LeadIntervals:
    """Protected original-grid fiducials and controlled lead abstentions."""

    lead: str
    beats: tuple[BeatEvidence, ...] = field(repr=False)
    reasons: tuple[Abstention, ...]


@dataclass(frozen=True, slots=True)
class TimingCandidate:
    """Protected measurement plus threshold/lead agreement, never an accuracy claim.

    Bounds span all included beats and leads, including sample/threshold sensitivity.
    The agreement score is 1 - envelope width / the host's allowed spread. It is
    not a probability. Abstention contains no selected measurement or confidence.
    """

    kind: TimingKind
    state: CandidateState
    leads: tuple[str, ...]
    reasons: tuple[Abstention, ...]
    value: float | None = field(repr=False)
    lower: float | None = field(repr=False)
    upper: float | None = field(repr=False)
    agreement_score: float | None = field(repr=False)
    observations: int


@dataclass(frozen=True, slots=True)
class IntervalCandidates:
    """Review-only timing and provenance; no signal samples retained or exported.

    Measurements and fiducials are protected in-memory fields. Only
    to_report_json() is intended for value-free audit output. Sample i refers to
    source-relative time start_offset_seconds + i / sample_rate_hz.
    """

    source_digest: str
    waveform_digest: str
    policy_digest: str
    quality: SignalQualityReport = field(repr=False)
    sample_rate_hz: float = field(repr=False)
    start_offset_seconds: float = field(repr=False)
    sample_count: int = field(repr=False)
    leads: tuple[LeadIntervals, ...] = field(repr=False)
    measurements: tuple[TimingCandidate, ...] = field(repr=False)

    @property
    def digest(self) -> str:
        """Bind the full protected result and provenance, without authenticating it."""
        return _hash({"algorithm": ALGORITHM_VERSION, "result": asdict(self)})

    def to_report_json(self) -> str:
        """Serialize controlled decisions and hashes, omitting measurement values."""
        return json.dumps(
            {
                "algorithm": ALGORITHM_VERSION,
                "source_digest": self.source_digest,
                "waveform_digest": self.waveform_digest,
                "policy_digest": self.policy_digest,
                "quality_policy_digest": self.quality.policy_digest,
                "acquisition_digest": self.quality.acquisition_digest,
                "quality_state": self.quality.state,
                "result_digest": self.digest,
                "measurements": [
                    {
                        "kind": item.kind,
                        "state": item.state,
                        "leads": item.leads,
                        "reasons": item.reasons,
                        "observations": item.observations,
                    }
                    for item in self.measurements
                ],
                "notice": REVIEW_NOTICE,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def _runs(values: list[float], policy: IntervalPolicy) -> list[tuple[int, int]] | None:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    end = 0
    for index, value in enumerate(values):
        if abs(value) > policy.boundary_low_mv:
            if start is None:
                start = index
            elif index - end > policy.join_samples:
                runs.append((start, end))
                if len(runs) > 4096:
                    return None
                start = index
            end = index + 1
    if start is not None:
        runs.append((start, end))
    return runs if len(runs) <= 4096 else None


def _boundary(
    values: list[float],
    run: tuple[int, int],
    width: SampleRange,
    policy: IntervalPolicy,
) -> WaveBoundary | None:
    start, end = run
    quiet = policy.quiet_samples
    if start < quiet or end + quiet > len(values) or not width.contains(end - start):
        return None
    if any(
        abs(v) > policy.boundary_low_mv
        for v in values[start - quiet : start] + values[end : end + quiet]
    ):
        return None
    high = [i for i in range(start, end) if abs(values[i]) > policy.boundary_high_mv]
    if not high:
        return None
    maximum = max(abs(values[i]) for i in range(start, end))
    peaks = [i for i in range(start, end) if abs(values[i]) == maximum]
    # A separated equal maximum is ambiguous, not a tie resolved by array order.
    if peaks[-1] - peaks[0] + 1 != len(peaks):
        return None
    onset = SampleBound(start - 1, high[0])
    offset = SampleBound(high[-1], end)
    peak = SampleBound(peaks[0], peaks[-1])
    if any(
        bound.upper - bound.lower > policy.boundary_uncertainty_samples
        for bound in (onset, offset, peak)
    ):
        return None
    return WaveBoundary(onset, peak, offset, start - quiet, end + quiet)


def _secondary(
    values: list[float],
    runs: list[tuple[int, int]],
    left: int,
    right: int,
    policy: IntervalPolicy,
) -> WaveBoundary | None:
    if left < 0 or right >= len(values) or left >= right:
        return None
    # Inspect every intersecting excursion: never select the strongest of rivals.
    intersecting = [(a, b) for a, b in runs if a <= right and b > left]
    if len(intersecting) != 1:
        return None
    start, end = intersecting[0]
    if start - policy.quiet_samples < left or end + policy.quiet_samples > right + 1:
        return None
    amplitude = max(abs(v) for v in values[start:end])
    if not policy.secondary_peak_mv <= amplitude < policy.qrs_peak_mv:
        return None
    return _boundary(values, (start, end), policy.secondary_width, policy)


def _lead(lead: EcgLead, policy: IntervalPolicy) -> LeadIntervals:
    def refuse(reason: Abstention) -> LeadIntervals:
        return LeadIntervals(lead.lead_id, (), (reason,))

    if any(value is None for value in lead.samples_mv):
        return refuse(Abstention.MISSING)
    samples = [value for value in lead.samples_mv if value is not None]
    # Numerical overflow is rejected before arithmetic; these are resource bounds.
    if any(abs(value) > 1e9 for value in samples):
        return refuse(Abstention.BASELINE)
    baseline = median(samples)
    values = [value - baseline for value in samples]
    if (
        sum(abs(value) <= policy.boundary_low_mv for value in values) / len(values)
        < policy.baseline_fraction
    ):
        return refuse(Abstention.BASELINE)
    runs = _runs(values, policy)
    if runs is None:
        return refuse(Abstention.LIMIT)
    ventricular: list[WaveBoundary] = []
    for start, end in runs:
        if max(abs(v) for v in values[start:end]) < policy.qrs_peak_mv:
            continue
        # Large but insufficiently steep/ambiguous excursions are not discarded.
        slope = max(
            abs(values[i] - values[i - 1])
            for i in range(max(1, start), end + (end < len(values)))
        )
        boundary = _boundary(values, (start, end), policy.qrs_width, policy)
        if slope < policy.qrs_slope_mv_per_sample or boundary is None:
            return refuse(Abstention.QRS)
        ventricular.append(boundary)
        if len(ventricular) > 256:
            return refuse(Abstention.LIMIT)
    if len(ventricular) < policy.minimum_beats:
        return refuse(Abstention.BEATS)
    for first, second in zip(ventricular, ventricular[1:]):
        if not (
            policy.rr_spacing.contains(second.peak.lower - first.peak.upper)
            and policy.rr_spacing.contains(second.peak.upper - first.peak.lower)
        ):
            return refuse(Abstention.SPACING)
    beats = []
    for qrs in ventricular:
        # Use the outer boundary for a conservative search region. P/T are actual
        # observed excursions, not expected offsets substituted for measurements.
        p = _secondary(
            values,
            runs,
            qrs.onset.lower - policy.p_search.maximum,
            qrs.onset.lower - policy.p_search.minimum,
            policy,
        )
        t = _secondary(
            values,
            runs,
            qrs.offset.upper + policy.t_search.minimum,
            qrs.offset.upper + policy.t_search.maximum,
            policy,
        )
        beats.append(BeatEvidence(qrs, p, t))
    # One observed excursion cannot supply two different wave identities. Hold
    # every competing assignment instead of resolving it by loop/beat order.
    assigned = [
        (index, name, boundary)
        for index, beat in enumerate(beats)
        for name, boundary in (("p", beat.p), ("t", beat.t))
        if boundary is not None
    ]
    ambiguous: set[tuple[int, str]] = set()
    for position, (index, name, boundary) in enumerate(assigned):
        for other_index, other_name, other in assigned[position + 1 :]:
            if max(boundary.onset.lower, other.onset.lower) <= min(
                boundary.offset.upper, other.offset.upper
            ):
                ambiguous.add((index, name))
                ambiguous.add((other_index, other_name))
    for index, name in sorted(ambiguous):
        beats[index] = (
            replace(beats[index], p=None)
            if name == "p"
            else replace(beats[index], t=None)
        )
    return LeadIntervals(lead.lead_id, tuple(beats), ())


def _aligned(
    leads: tuple[LeadIntervals, ...], policy: IntervalPolicy
) -> tuple[Abstention, ...]:
    reasons = {reason for lead in leads for reason in lead.reasons}
    if len(leads) < policy.minimum_leads:
        reasons.add(Abstention.LEADS)
    if len({len(lead.beats) for lead in leads}) != 1:
        reasons.add(Abstention.ALIGNMENT)
    if reasons:
        return tuple(sorted(reasons, key=lambda reason: reason.value))
    for beats in zip(*(lead.beats for lead in leads)):
        if (
            max(beat.qrs.peak.upper for beat in beats)
            - min(beat.qrs.peak.lower for beat in beats)
            > policy.alignment_samples
        ):
            return (Abstention.ALIGNMENT,)
    return ()


def _candidate(
    kind: TimingKind,
    leads: tuple[LeadIntervals, ...],
    policy: IntervalPolicy,
    common: tuple[Abstention, ...],
) -> TimingCandidate:
    ids = tuple(lead.lead for lead in leads)

    def refuse(reasons: tuple[Abstention, ...]) -> TimingCandidate:
        return TimingCandidate(
            kind, CandidateState.ABSTAINED, ids, reasons, None, None, None, None, 0
        )

    if common:
        return refuse(common)
    ranges: list[tuple[float, float]] = []
    for lead in leads:
        if kind is TimingKind.RATE:
            for first, second in zip(lead.beats, lead.beats[1:]):
                ranges.append(
                    (
                        60
                        * policy.sample_rate_hz
                        / (second.qrs.peak.upper - first.qrs.peak.lower),
                        60
                        * policy.sample_rate_hz
                        / (second.qrs.peak.lower - first.qrs.peak.upper),
                    )
                )
            continue
        for beat in lead.beats:
            start, end = beat.qrs.onset, beat.qrs.offset
            if kind is TimingKind.PR:
                if beat.p is None:
                    return refuse((Abstention.P,))
                start, end = beat.p.onset, beat.qrs.onset
            elif kind is TimingKind.QT:
                if beat.t is None:
                    return refuse((Abstention.T,))
                end = beat.t.offset
            ranges.append(
                (
                    (end.lower - start.upper) * 1000 / policy.sample_rate_hz,
                    (end.upper - start.lower) * 1000 / policy.sample_rate_hz,
                )
            )
    lower = min(pair[0] for pair in ranges)
    upper = max(pair[1] for pair in ranges)
    allowed = (
        policy.rate_spread_bpm if kind is TimingKind.RATE else policy.interval_spread_ms
    )
    if lower <= 0 or upper - lower > allowed:
        return refuse((Abstention.SPREAD,))
    return TimingCandidate(
        kind,
        CandidateState.REVIEW,
        ids,
        (),
        median((a + b) / 2 for a, b in ranges),
        lower,
        upper,
        1 - (upper - lower) / allowed,
        len(ranges),
    )


def extract_interval_candidates(
    waveform: EcgWaveform,
    quality_policy: SignalQualityPolicy,
    acquisition: AcquisitionEvidence,
    interval_policy: IntervalPolicy,
    *,
    reviewer_confirmed: bool = False,
) -> IntervalCandidates:
    """Freshly gate source quality, then derive review-only timing from samples.

    Args:
        waveform: Original normalized immutable source; missingness is preserved.
        quality_policy: Independently chosen required leads and acquisition limits.
        acquisition: Bound original layout, calibration and waveform evidence.
        interval_policy: Explicit local detector and agreement settings.
        reviewer_confirmed: Explicit permission to measure this acquisition now.

    Returns:
        Protected per-lead fiducials and four review candidates or abstentions.
        No filtering, interpolation, model download, disk write or diagnosis occurs.

    Raises:
        IntervalCandidateError: For invalid declarations or a policy rate mismatch.
        SignalQualityError: When fresh quality or reviewer confirmation blocks work.
    """
    _require(type(waveform) is EcgWaveform)
    _require(type(quality_policy) is SignalQualityPolicy)
    _require(type(acquisition) is AcquisitionEvidence)
    _require(type(interval_policy) is IntervalPolicy)
    _require(interval_policy.sample_rate_hz == waveform.sample_rate_hz)

    def measure(selected: EcgWaveform) -> tuple[LeadIntervals, ...]:
        return tuple(_lead(lead, interval_policy) for lead in selected.leads)

    measured = run_gated_measurement(
        waveform,
        quality_policy,
        acquisition,
        measure,
        reviewer_confirmed=reviewer_confirmed,
    )
    leads = measured.value
    common = _aligned(leads, interval_policy)
    return IntervalCandidates(
        waveform.provenance.source_digest,
        measured.quality.waveform_digest,
        interval_policy.digest,
        measured.quality,
        waveform.sample_rate_hz,
        waveform.start_offset_seconds,
        len(waveform.leads[0].samples_mv),
        leads,
        tuple(_candidate(kind, leads, interval_policy, common) for kind in TimingKind),
    )
