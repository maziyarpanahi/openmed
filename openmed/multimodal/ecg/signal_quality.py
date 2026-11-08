"""Offline ECG acquisition gates with explicit policies, without diagnosis."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from statistics import median
from typing import ClassVar, Generic, TypeVar

from openmed.multimodal.notices import NOTICE_CATALOG, MultimodalNotice, NoticeKind

from .lead_layout import (
    STANDARD_LEADS,
    AcquisitionKind,
    LeadLayout,
    LeadObservation,
    SlotRole,
    assess_lead_layout,
)
from .waveform import NON_DIAGNOSTIC_NOTICE, EcgLead, EcgWaveform

QUALITY_VERSION = "openmed.ecg.signal-quality.v1"
_NUMERIC_BOUND = 1e12  # Arithmetic resource bound, not a physiological threshold.
_T = TypeVar("_T")


class SignalQualityError(ValueError):
    """Controlled error with no samples, source labels, paths or callback details."""


class QualityState(str, Enum):
    """Policy outcome; only PASS permits the guarded measurement callback."""

    PASS = "pass"
    LIMITED_USE = "limited-use"
    REVIEW = "review"
    REACQUIRE = "reacquire"


_RANK = {state: index for index, state in enumerate(QualityState)}


def _require(condition: bool) -> None:
    if not condition:
        raise SignalQualityError("invalid ECG quality declaration")


def _finite(value: object, *, positive: bool = False) -> None:
    _require(type(value) in (int, float))
    _require(-_NUMERIC_BOUND <= value <= _NUMERIC_BOUND)  # type: ignore[operator]
    if positive:
        _require(value > 0)  # type: ignore[operator]


def _digest(value: str) -> None:
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _worst(states: list[QualityState]) -> QualityState:
    return max(states, key=_RANK.__getitem__, default=QualityState.PASS)


@dataclass(frozen=True, slots=True)
class QualityLimits:
    """Upper bounds for one metric; equality remains in the less severe state.

    Args:
        limited_above: Values strictly above this require restricted review use.
        reacquire_above: Values strictly above this require reacquisition review.
    """

    limited_above: float
    reacquire_above: float

    def __post_init__(self) -> None:
        _finite(self.limited_above)
        _finite(self.reacquire_above)
        _require(0 <= self.limited_above <= self.reacquire_above)

    def classify(self, value: float) -> QualityState:
        """Classify a finite, nonnegative measurement using both explicit bounds."""
        _require(type(value) in (int, float) and 0 <= value < math.inf)
        if value > self.reacquire_above:
            return QualityState.REACQUIRE
        if value > self.limited_above:
            return QualityState.LIMITED_USE
        return QualityState.PASS


@dataclass(frozen=True, slots=True, kw_only=True)
class SignalQualityPolicy:
    """Versioned caller thresholds; no clinical defaults or qualification claim.

    Args:
        version: Bounded opaque policy version, never patient or device free text.
        evidence_digest: SHA-256 of the caller's policy evaluation evidence.
        required_leads: Independently selected leads required for measurement.
        sample_rate_hz: Exact rate for which these sample-domain proxies were evaluated.
        minimum_duration_seconds: Minimum acquisition duration for this policy.
        block_samples: Nonoverlapping proxy window size, at least three samples.
        minimum_block_valid_fraction: Minimum observed fraction in every proxy block.
        flatline_tolerance_mv: Maximum full range within a flatline candidate run.
        rail_tolerance_mv: Distance inside declared calibrated rails counted as clipping.
        missing_fraction: Missing sample fraction limits.
        gap_seconds: Longest missing run duration limits.
        clipped_fraction: Rail occupancy among observed samples limits.
        flatline_seconds: Longest bounded-range run duration limits.
        baseline_span_mv: Range of block medians limits; a drift proxy only.
        roughness_mv: Maximum block median absolute second difference limits.
        gain_relative_error: Maximum deviation from independently expected gain limits.
    """

    version: str
    evidence_digest: str
    required_leads: tuple[str, ...]
    sample_rate_hz: float
    minimum_duration_seconds: float
    block_samples: int
    minimum_block_valid_fraction: float
    flatline_tolerance_mv: float
    rail_tolerance_mv: float
    missing_fraction: QualityLimits
    gap_seconds: QualityLimits
    clipped_fraction: QualityLimits
    flatline_seconds: QualityLimits
    baseline_span_mv: QualityLimits
    roughness_mv: QualityLimits
    gain_relative_error: QualityLimits

    def __post_init__(self) -> None:
        _require(
            type(self.version) is str
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.version)
            is not None
        )
        _digest(self.evidence_digest)
        _require(
            type(self.required_leads) is tuple and 1 <= len(self.required_leads) <= 12
        )
        _require(
            all(
                type(lead) is str and lead in STANDARD_LEADS
                for lead in self.required_leads
            )
        )
        _require(len(set(self.required_leads)) == len(self.required_leads))
        _finite(self.sample_rate_hz, positive=True)
        _require(self.sample_rate_hz <= 100_000)
        _finite(self.minimum_duration_seconds, positive=True)
        _require(self.minimum_duration_seconds <= 86400)
        _require(
            type(self.block_samples) is int and 3 <= self.block_samples <= 1_000_000
        )
        _finite(self.minimum_block_valid_fraction, positive=True)
        _require(self.minimum_block_valid_fraction <= 1)
        for value in (self.flatline_tolerance_mv, self.rail_tolerance_mv):
            _finite(value)
            _require(value >= 0)
        for name in _METRICS:
            _require(type(getattr(self, name)) is QualityLimits)
        for limits in (self.missing_fraction, self.clipped_fraction):
            _require(limits.reacquire_above <= 1)

    @property
    def digest(self) -> str:
        """Bind every policy field and the algorithm version for replay."""
        return _hash({"algorithm": QUALITY_VERSION, "policy": asdict(self)})


_METRICS = (
    "missing_fraction",
    "gap_seconds",
    "clipped_fraction",
    "flatline_seconds",
    "baseline_span_mv",
    "roughness_mv",
    "gain_relative_error",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationSegment:
    """Explicit acquisition calibration over a half-open sample interval.

    Args:
        start_sample: First sample covered, inclusive.
        end_sample: Last covered boundary, exclusive; at most one million.
        gain_counts_per_mv: Actual acquisition gain, or None when unverified.
        lower_rail_mv: Actual lower acquisition rail after calibration, or None.
        upper_rail_mv: Actual upper acquisition rail after calibration, or None.

    Rails must come from acquisition metadata, never observed signal extrema.
    None is represented as REVIEW, not silently inferred or defaulted.
    """

    start_sample: int
    end_sample: int
    gain_counts_per_mv: float | None = field(repr=False)
    lower_rail_mv: float | None = field(repr=False)
    upper_rail_mv: float | None = field(repr=False)

    def __post_init__(self) -> None:
        _require(type(self.start_sample) is int and type(self.end_sample) is int)
        _require(0 <= self.start_sample < self.end_sample <= 1_000_000)
        if self.gain_counts_per_mv is not None:
            _finite(self.gain_counts_per_mv, positive=True)
            _require(self.gain_counts_per_mv >= 1 / _NUMERIC_BOUND)
        _require((self.lower_rail_mv is None) == (self.upper_rail_mv is None))
        if self.lower_rail_mv is not None:
            _finite(self.lower_rail_mv)
            _finite(self.upper_rail_mv)
            assert self.upper_rail_mv is not None
            _require(self.lower_rail_mv < self.upper_rail_mv)


@dataclass(frozen=True, slots=True, kw_only=True)
class LeadCalibration:
    """Expected gain and observed segments for one canonical lead.

    Args:
        lead: Canonical lead identifier.
        expected_gain_counts_per_mv: Independent reference gain, or None.
        segments: Up to 4096 ordered segments; incomplete coverage requires review.
    """

    lead: str
    expected_gain_counts_per_mv: float | None = field(repr=False)
    segments: tuple[CalibrationSegment, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _require(type(self.lead) is str and self.lead in STANDARD_LEADS)
        if self.expected_gain_counts_per_mv is not None:
            _finite(self.expected_gain_counts_per_mv, positive=True)
            _require(self.expected_gain_counts_per_mv >= 1 / _NUMERIC_BOUND)
        _require(type(self.segments) is tuple and len(self.segments) <= 4096)
        _require(all(type(segment) is CalibrationSegment for segment in self.segments))


@dataclass(frozen=True, slots=True, kw_only=True)
class AcquisitionEvidence:
    """Bound original layout observations and calibration, retained only in memory.

    Args:
        source_digest: Original source digest, matching the waveform provenance.
        waveform_digest: Fingerprint of the normalized waveform at evidence capture.
        layout: Independently expected original channel/display layout.
        observations: Original positions and labels, before canonical reordering.
        acquisition: Digital or image-derived acquisition.
        calibrations: At most one per lead; absent calibration produces review.
        image_labels_confirmed: Explicit caller assertion of image label review.

    This binding detects accidental reuse, not forged metadata or physical wiring.
    """

    source_digest: str
    waveform_digest: str
    layout: LeadLayout = field(repr=False)
    observations: tuple[LeadObservation, ...] = field(repr=False)
    acquisition: AcquisitionKind
    calibrations: tuple[LeadCalibration, ...] = field(repr=False)
    image_labels_confirmed: bool = False

    def __post_init__(self) -> None:
        _digest(self.source_digest)
        _digest(self.waveform_digest)
        _require(type(self.layout) is LeadLayout)
        _require(type(self.observations) is tuple and len(self.observations) <= 64)
        _require(all(type(item) is LeadObservation for item in self.observations))
        _require(type(self.acquisition) is AcquisitionKind)
        _require(type(self.image_labels_confirmed) is bool)
        _require(type(self.calibrations) is tuple and len(self.calibrations) <= 12)
        _require(all(type(item) is LeadCalibration for item in self.calibrations))
        _require(
            len({item.lead for item in self.calibrations}) == len(self.calibrations)
        )


def waveform_fingerprint(waveform: EcgWaveform) -> str:
    """Hash normalized samples, missingness, lead identity, timing and provenance.

    Args:
        waveform: Immutable canonical waveform; original source digest is also bound.

    Returns:
        SHA-256 for protected lineage, never a de-identification guarantee.
    """
    _require(type(waveform) is EcgWaveform)
    digest = hashlib.sha256()
    digest.update(
        _hash(
            {
                "version": QUALITY_VERSION,
                "source": asdict(waveform.provenance),
                "rate": waveform.sample_rate_hz,
                "offset": waveform.start_offset_seconds,
                "leads": [lead.lead_id for lead in waveform.leads],
                "count": len(waveform.leads[0].samples_mv),
            }
        ).encode()
    )
    for lead in waveform.leads:
        for sample in lead.samples_mv:
            digest.update(
                b"\x00" if sample is None else b"\x01" + struct.pack("!d", sample)
            )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QualityFinding:
    """Controlled metric/evidence code and resulting non-diagnostic state."""

    code: str
    state: QualityState


@dataclass(frozen=True, slots=True)
class LeadQuality:
    """Per-lead aggregate metrics; None means evidence could not be measured."""

    lead: str
    required: bool
    state: QualityState
    missing_fraction: float
    gap_seconds: float
    clipped_fraction: float | None
    flatline_seconds: float | None
    baseline_span_mv: float | None
    roughness_mv: float | None
    gain_relative_error: float | None
    findings: tuple[QualityFinding, ...]


@dataclass(frozen=True, slots=True)
class SignalQualityReport:
    """Bound aggregate evidence; overall state covers required leads and structure.

    No raw samples, observed labels, gain/rail values or source paths are exported.
    Reports are evidence only; the measurement wrapper always reassesses inputs.
    """

    source_digest: str
    waveform_digest: str
    acquisition_digest: str
    policy_digest: str
    policy_version: str
    state: QualityState
    leads: tuple[LeadQuality, ...]
    findings: tuple[QualityFinding, ...]

    @property
    def measurement_allowed(self) -> bool:
        """Whether required quality gates pass; human confirmation is still needed."""
        return self.state is QualityState.PASS

    def to_json(self) -> str:
        """Return deterministic aggregate JSON with the non-diagnostic boundary."""
        return json.dumps(
            {
                **asdict(self),
                "schema_version": QUALITY_VERSION,
                "measurement_allowed": self.measurement_allowed,
                "notice": NON_DIAGNOSTIC_NOTICE,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def _longest_flatline(samples: tuple[float | None, ...], tolerance: float) -> int:
    # Sliding min/max queues avoid chaining adjacent tiny differences into a drift.
    minima: deque[tuple[int, float]] = deque()
    maxima: deque[tuple[int, float]] = deque()
    left = longest = 0
    for index, value in enumerate(samples):
        if value is None:
            minima.clear()
            maxima.clear()
            left = index + 1
            continue
        while minima and minima[-1][1] >= value:
            minima.pop()
        while maxima and maxima[-1][1] <= value:
            maxima.pop()
        minima.append((index, value))
        maxima.append((index, value))
        while maxima[0][1] - minima[0][1] > tolerance:
            left = min(minima[0][0], maxima[0][0]) + 1
            while minima and minima[0][0] < left:
                minima.popleft()
            while maxima and maxima[0][0] < left:
                maxima.popleft()
        longest = max(longest, index - left + 1)
    return longest


def _proxies(
    lead: EcgLead, policy: SignalQualityPolicy
) -> tuple[float | None, float | None]:
    baselines: list[float] = []
    noise: list[float] = []
    samples = lead.samples_mv
    for start in range(0, len(samples), policy.block_samples):
        block = samples[start : start + policy.block_samples]
        observed = [value for value in block if value is not None]
        # A short trailing block is still examined: never hide a corrupted tail.
        if (
            len(observed) < 3
            or len(observed) / len(block) < policy.minimum_block_valid_fraction
        ):
            return None, None
        baselines.append(float(median(observed)))
        differences = [
            abs(c - 2 * b + a)
            for a, b, c in zip(block, block[1:], block[2:])
            if a is not None and b is not None and c is not None
        ]
        if not differences:
            return None, None
        noise.append(float(median(differences)))
    return (
        max(baselines) - min(baselines) if len(baselines) >= 2 else None,
        max(noise) if noise else None,
    )


def _calibration_metrics(
    lead: EcgLead,
    calibration: LeadCalibration | None,
    policy: SignalQualityPolicy,
) -> tuple[float | None, float | None]:
    if calibration is None or not calibration.segments:
        return None, None
    cursor = clipped = observed = 0
    gain_error: float | None = 0.0
    rails_known = True
    for segment in calibration.segments:
        if segment.start_sample != cursor or segment.end_sample > len(lead.samples_mv):
            return None, None
        cursor = segment.end_sample
        actual, expected = (
            segment.gain_counts_per_mv,
            calibration.expected_gain_counts_per_mv,
        )
        if actual is None or expected is None:
            gain_error = None
        elif gain_error is not None:
            gain_error = max(gain_error, abs(actual / expected - 1))
        low, high = segment.lower_rail_mv, segment.upper_rail_mv
        if low is None or high is None or 2 * policy.rail_tolerance_mv >= high - low:
            rails_known = False
            continue
        for value in lead.samples_mv[segment.start_sample : segment.end_sample]:
            if value is not None:
                observed += 1
                if (
                    value <= low + policy.rail_tolerance_mv
                    or value >= high - policy.rail_tolerance_mv
                ):
                    clipped += 1
    if cursor != len(lead.samples_mv):
        return None, None
    return (clipped / observed if rails_known and observed else None, gain_error)


def _assess_lead(
    lead: EcgLead, policy: SignalQualityPolicy, calibration: LeadCalibration | None
) -> LeadQuality:
    count = len(lead.samples_mv)
    missing = lead.valid_mask.count(False)
    run = gap = 0
    for valid in lead.valid_mask:
        run = 0 if valid else run + 1
        gap = max(gap, run)
    findings: list[QualityFinding] = []
    values: dict[str, float | None] = {name: None for name in _METRICS}
    values.update(
        missing_fraction=missing / count, gap_seconds=gap / policy.sample_rate_hz
    )
    if missing == count:
        findings.append(QualityFinding("no_observed_samples", QualityState.REACQUIRE))
    elif any(
        value is not None and abs(value) > _NUMERIC_BOUND for value in lead.samples_mv
    ):
        findings.append(QualityFinding("numeric_range_unverified", QualityState.REVIEW))
    else:
        values["flatline_seconds"] = (
            _longest_flatline(lead.samples_mv, policy.flatline_tolerance_mv)
            / policy.sample_rate_hz
        )
        values["baseline_span_mv"], values["roughness_mv"] = _proxies(lead, policy)
        values["clipped_fraction"], values["gain_relative_error"] = (
            _calibration_metrics(lead, calibration, policy)
        )
    for name in _METRICS:
        value = values[name]
        state = (
            QualityState.REVIEW
            if value is None
            else getattr(policy, name).classify(value)
        )
        if state is not QualityState.PASS:
            findings.append(
                QualityFinding(f"{name}_unverified" if value is None else name, state)
            )
    return LeadQuality(
        lead=lead.lead_id,
        required=lead.lead_id in policy.required_leads,
        state=_worst([item.state for item in findings]),
        missing_fraction=missing / count,
        gap_seconds=gap / policy.sample_rate_hz,
        clipped_fraction=values["clipped_fraction"],
        flatline_seconds=values["flatline_seconds"],
        baseline_span_mv=values["baseline_span_mv"],
        roughness_mv=values["roughness_mv"],
        gain_relative_error=values["gain_relative_error"],
        findings=tuple(findings),
    )


def assess_signal_quality(
    waveform: EcgWaveform,
    policy: SignalQualityPolicy,
    evidence: AcquisitionEvidence,
) -> SignalQualityReport:
    """Freshly assess structural, timing, calibration and per-lead signal gates.

    Args:
        waveform: Canonical regular aligned samples in millivolts.
        policy: Explicit thresholds and the rate at which they were evaluated.
        evidence: Original layout and full-record calibration bound to these samples.

    Returns:
        Aggregate findings. Unknown evidence requires review; severe measured
        defects require reacquisition review. Neither state triggers an action.
    """
    _require(type(waveform) is EcgWaveform and type(policy) is SignalQualityPolicy)
    _require(type(evidence) is AcquisitionEvidence)
    fingerprint = waveform_fingerprint(waveform)
    findings: list[QualityFinding] = []
    if (
        evidence.source_digest != waveform.provenance.source_digest
        or evidence.waveform_digest != fingerprint
    ):
        findings.append(
            QualityFinding("acquisition_binding_mismatch", QualityState.REVIEW)
        )
    layout_report = assess_lead_layout(
        evidence.observations,
        evidence.layout,
        source_digest=waveform.provenance.source_digest,
        acquisition=evidence.acquisition,
        image_labels_confirmed=evidence.image_labels_confirmed,
    )
    if layout_report.requires_review:
        findings.append(QualityFinding("lead_layout_unverified", QualityState.REVIEW))
    present = {lead.lead_id for lead in waveform.leads}
    if not set(policy.required_leads) <= present:
        findings.append(QualityFinding("required_lead_missing", QualityState.REVIEW))
    if present != set(evidence.layout.expected_leads) or any(
        item.lead not in present for item in evidence.calibrations
    ):
        findings.append(
            QualityFinding("acquisition_leads_mismatch", QualityState.REVIEW)
        )
    count = len(waveform.leads[0].samples_mv)
    duration = count / waveform.sample_rate_hz
    primary = {
        (slot.row, slot.column)
        for slot in evidence.layout.slots
        if slot.role is SlotRole.PRIMARY
    }
    for observation in evidence.observations:
        if (observation.row, observation.column) not in primary:
            continue
        if (
            observation.sample_count != count
            or observation.sample_rate_hz != waveform.sample_rate_hz
            or observation.start_seconds is None
            or observation.duration_seconds is None
            or not math.isclose(
                observation.start_seconds,
                waveform.start_offset_seconds,
                rel_tol=0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                observation.duration_seconds, duration, rel_tol=0, abs_tol=1e-9
            )
        ):
            findings.append(
                QualityFinding("waveform_timing_mismatch", QualityState.REVIEW)
            )
            break
    if waveform.sample_rate_hz != policy.sample_rate_hz:
        findings.append(QualityFinding("policy_rate_mismatch", QualityState.REVIEW))
    if duration < policy.minimum_duration_seconds:
        findings.append(QualityFinding("duration_insufficient", QualityState.REVIEW))
    calibrations = {item.lead: item for item in evidence.calibrations}
    # Durations are always measured using the actual acquisition rate, even when
    # a policy-rate mismatch already blocks downstream use.
    measured_policy = replace(policy, sample_rate_hz=waveform.sample_rate_hz)
    leads = tuple(
        _assess_lead(lead, measured_policy, calibrations.get(lead.lead_id))
        for lead in waveform.leads
    )
    state = _worst(
        [item.state for item in findings]
        + [item.state for item in leads if item.required]
    )
    return SignalQualityReport(
        waveform.provenance.source_digest,
        fingerprint,
        _hash(asdict(evidence)),
        policy.digest,
        policy.version,
        state,
        leads,
        tuple(findings),
    )


@dataclass(frozen=True, slots=True)
class GatedMeasurement(Generic[_T]):
    """Protected callback value and the newly generated acquisition gate report."""

    review_notice: ClassVar[MultimodalNotice] = NOTICE_CATALOG[NoticeKind.MEASUREMENT]

    value: _T = field(repr=False)
    quality: SignalQualityReport


def run_gated_measurement(
    waveform: EcgWaveform,
    policy: SignalQualityPolicy,
    evidence: AcquisitionEvidence,
    measure: Callable[[EcgWaveform], _T],
    *,
    reviewer_confirmed: bool = False,
) -> GatedMeasurement[_T]:
    """Invoke measurement only after fresh gates and explicit reviewer confirmation.

    Args:
        waveform: Original immutable normalized input.
        policy: Explicit required leads and quality policy.
        evidence: Bound acquisition evidence, reassessed on every call.
        measure: Local callback, receiving only required leads. Not a sandbox.
        reviewer_confirmed: Explicit approval for this invocation; not qualification.

    Returns:
        Protected measurement result with its report; no report can bypass gating.

    Raises:
        SignalQualityError: On blocked quality, absent confirmation or callback failure.
    """
    report = assess_signal_quality(waveform, policy, evidence)
    if not report.measurement_allowed:
        raise SignalQualityError("ECG measurement blocked by acquisition quality")
    if reviewer_confirmed is not True:
        raise SignalQualityError(
            "ECG measurement requires explicit reviewer confirmation"
        )
    _require(callable(measure))
    selected = EcgWaveform(
        leads=tuple(
            lead for lead in waveform.leads if lead.lead_id in policy.required_leads
        ),
        sample_rate_hz=waveform.sample_rate_hz,
        provenance=waveform.provenance,
        start_offset_seconds=waveform.start_offset_seconds,
    )
    try:
        return GatedMeasurement(measure(selected), report)
    except Exception:
        pass
    # Raise outside the handler so even exception context cannot expose callback data.
    raise SignalQualityError("ECG measurement callback failed")
