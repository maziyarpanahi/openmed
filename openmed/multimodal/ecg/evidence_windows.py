"""Source-bound ECG review windows, with metadata-only serialization."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum

from .interval_candidates import (
    ALGORITHM_VERSION,
    REVIEW_NOTICE,
    Abstention,
    CandidateState,
    IntervalCandidates,
    IntervalPolicy,
    TimingCandidate,
    TimingKind,
    WaveBoundary,
    extract_interval_candidates,
)
from .lead_layout import STANDARD_LEADS
from .signal_quality import (
    QUALITY_VERSION,
    AcquisitionEvidence,
    SignalQualityPolicy,
    waveform_fingerprint,
)
from .waveform import EcgWaveform

WINDOW_SCHEMA = "openmed.ecg.evidence-windows.v1"
_MAX_WINDOWS = 12 * 256 * 4
_MAX_METADATA_CHARS = 16_000_000


class EvidenceWindowError(ValueError):
    """Controlled error that never includes source data or serialized input."""


def _require(condition: bool) -> None:
    if not condition:
        raise EvidenceWindowError("invalid ECG evidence window declaration or binding")


def _integer(value: object, lower: int, upper: int) -> None:
    _require(type(value) is int and lower <= value <= upper)  # type: ignore[operator]


def _digest(value: object) -> None:
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class WindowPolicy:
    """Explicit context and resource bounds, never a clinical qualification.

    Args:
        version: Opaque policy version, never patient/device free text.
        evidence_digest: SHA-256 of the host's policy evaluation evidence.
        padding_samples: Extra original samples on each side of a complete window.
        maximum_window_samples: Maximum length of any one requested window.
        maximum_total_samples: Sum of all window lengths, including overlap.

    Padding cannot be silently clipped. Exceeding a bound refuses the whole
    window set rather than truncating measurement evidence.
    """

    version: str
    evidence_digest: str
    padding_samples: int
    maximum_window_samples: int
    maximum_total_samples: int

    def __post_init__(self) -> None:
        _require(
            type(self.version) is str
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.version)
            is not None
        )
        _digest(self.evidence_digest)
        _integer(self.padding_samples, 0, 1_000_000)
        _integer(self.maximum_window_samples, 1, 1_000_000)
        _integer(self.maximum_total_samples, 1, 8_000_000)
        _require(self.maximum_window_samples <= self.maximum_total_samples)

    @property
    def digest(self) -> str:
        """Bind every setting and the window schema for reproducible review."""
        return _hash({"schema": WINDOW_SCHEMA, "policy": asdict(self)})


class FiducialKind(str, Enum):
    """Observed point identities; a peak is never substituted for an onset."""

    P_ONSET = "p-onset"
    P_PEAK = "p-peak"
    P_OFFSET = "p-offset"
    QRS_ONSET = "qrs-onset"
    QRS_PEAK = "qrs-peak"
    QRS_OFFSET = "qrs-offset"
    T_ONSET = "t-onset"
    T_PEAK = "t-peak"
    T_OFFSET = "t-offset"


@dataclass(frozen=True, slots=True)
class WindowFiducial:
    """Inclusive original-grid bracket tied to a lead-local beat index."""

    kind: FiducialKind
    beat_index: int = field(repr=False)
    lower_sample: int = field(repr=False)
    upper_sample: int = field(repr=False)

    def __post_init__(self) -> None:
        _require(type(self.kind) is FiducialKind)
        _integer(self.beat_index, 0, 255)
        _integer(self.lower_sample, 0, 999_999)
        _integer(self.upper_sample, self.lower_sample, 999_999)


@dataclass(frozen=True, slots=True)
class MeasurementWindow:
    """One complete source window, linked to an aggregate candidate and context.

    Bounds are half-open original sample indices. Fiducials are never rebased to
    a sliced array. An abstained aggregate can still have individual windows
    available for review; this does not upgrade its state or approve a result.
    """

    context_digest: str
    candidate_digest: str
    lead: str
    kind: TimingKind
    state: CandidateState
    reasons: tuple[Abstention, ...]
    observation_index: int = field(repr=False)
    start_sample: int = field(repr=False)
    end_sample: int = field(repr=False)
    source_sample_count: int = field(repr=False)
    fiducials: tuple[WindowFiducial, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _digest(self.context_digest)
        _digest(self.candidate_digest)
        _require(type(self.lead) is str and self.lead in STANDARD_LEADS)
        _require(type(self.kind) is TimingKind and type(self.state) is CandidateState)
        _require(
            type(self.reasons) is tuple
            and all(type(item) is Abstention for item in self.reasons)
        )
        _require(bool(self.reasons) == (self.state is CandidateState.ABSTAINED))
        _integer(self.observation_index, 0, 255)
        _integer(self.source_sample_count, 1, 1_000_000)
        _integer(self.start_sample, 0, self.source_sample_count - 1)
        _integer(self.end_sample, self.start_sample + 1, self.source_sample_count)
        _require(type(self.fiducials) is tuple and len(self.fiducials) in (3, 6))
        _require(all(type(item) is WindowFiducial for item in self.fiducials))
        _require(
            len({(item.kind, item.beat_index) for item in self.fiducials})
            == len(self.fiducials)
        )
        _require(
            all(
                self.start_sample
                <= item.lower_sample
                <= item.upper_sample
                < self.end_sample
                for item in self.fiducials
            )
        )

    @property
    def digest(self) -> str:
        """Bind the complete descriptor, candidate and parent source context."""
        return _hash(asdict(self))

    def _metadata(self) -> dict:
        return {"window_digest": self.digest, **asdict(self)}


@dataclass(frozen=True, slots=True)
class WindowSamples:
    """Protected original samples for a local viewer; no serialization method."""

    window_digest: str
    lead: str
    samples_mv: tuple[float | None, ...] = field(repr=False)
    source_start_sample: int = field(repr=False)
    sample_rate_hz: float = field(repr=False)
    start_offset_seconds: float = field(repr=False)


@dataclass(frozen=True, slots=True)
class EcgEvidenceWindows:
    """Review evidence, with protected payload and deterministic metadata replay.

    Construct with build_evidence_windows, or restore with the original source
    and policies. Metadata and hashes are not an authenticated clinical review.
    No approved clinical EvidencePacket is fabricated, and real ECG data is
    never labelled synthetic to cross that separate boundary.
    """

    candidates: IntervalCandidates = field(repr=False)
    window_policy: WindowPolicy = field(repr=False)
    upstream_transform_digests: tuple[str, ...] = field(repr=False)
    transform_digest: str
    context_digest: str
    windows: tuple[MeasurementWindow, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _require(type(self.candidates) is IntervalCandidates)
        _require(type(self.window_policy) is WindowPolicy)
        _transforms(self.upstream_transform_digests)
        _digest(self.transform_digest)
        _digest(self.context_digest)
        _require(type(self.windows) is tuple and len(self.windows) <= _MAX_WINDOWS)
        _require(all(type(item) is MeasurementWindow for item in self.windows))
        _require(
            all(
                item.context_digest == self.context_digest
                and item.source_sample_count == self.candidates.sample_count
                for item in self.windows
            )
        )
        _require(len({item.digest for item in self.windows}) == len(self.windows))

    def _metadata(self) -> dict:
        quality = self.candidates.quality
        return {
            "schema": WINDOW_SCHEMA,
            "algorithm": ALGORITHM_VERSION,
            "source_digest": self.candidates.source_digest,
            "waveform_digest": self.candidates.waveform_digest,
            "candidate_result_digest": self.candidates.digest,
            "interval_policy_digest": self.candidates.policy_digest,
            "window_policy_digest": self.window_policy.digest,
            "upstream_transform_digests": self.upstream_transform_digests,
            "transform_digest": self.transform_digest,
            "context_digest": self.context_digest,
            "source_grid": {
                "sample_count": self.candidates.sample_count,
                "sample_rate_hz": self.candidates.sample_rate_hz,
                "start_offset_seconds": self.candidates.start_offset_seconds,
            },
            "quality": {
                "algorithm": QUALITY_VERSION,
                "policy_digest": quality.policy_digest,
                "acquisition_digest": quality.acquisition_digest,
                "state": quality.state,
                "findings": [asdict(item) for item in quality.findings],
                "leads": [
                    {
                        "lead": item.lead,
                        "required": item.required,
                        "state": item.state,
                        "findings": [asdict(finding) for finding in item.findings],
                    }
                    for item in quality.leads
                ],
            },
            "candidates": [
                {
                    "kind": item.kind,
                    "state": item.state,
                    "leads": item.leads,
                    "reasons": item.reasons,
                    "candidate_digest": _hash(asdict(item)),
                }
                for item in self.candidates.measurements
            ],
            "windows": [item._metadata() for item in self.windows],
            "notice": REVIEW_NOTICE,
        }

    @property
    def digest(self) -> str:
        """Bind all metadata and the complete protected candidate-result digest."""
        return _hash(self._metadata())

    def to_metadata_json(self) -> str:
        """Serialize protected source offsets/fiducials and hashes, never samples.

        This metadata includes timing and must not be used as an audit log. It
        omits measurement values and retains abstentions without review approval.
        Use to_report_json for controlled counts-only operational output.
        """
        result = _json({"bundle_digest": self.digest, **self._metadata()})
        _require(len(result) <= _MAX_METADATA_CHARS)
        return result

    def to_report_json(self) -> str:
        """Emit hashes, controlled states and window counts; omit timing and signal."""
        return _json(
            {
                "schema": WINDOW_SCHEMA,
                "bundle_digest": self.digest,
                "source_digest": self.candidates.source_digest,
                "waveform_digest": self.candidates.waveform_digest,
                "transform_digest": self.transform_digest,
                "window_count": len(self.windows),
                "candidates": [
                    {
                        "kind": candidate.kind,
                        "state": candidate.state,
                        "reasons": candidate.reasons,
                        "window_count": sum(
                            item.kind is candidate.kind for item in self.windows
                        ),
                    }
                    for candidate in self.candidates.measurements
                ],
                "notice": REVIEW_NOTICE,
            }
        )

    def samples_for(
        self,
        window_digest: str,
        waveform: EcgWaveform,
        *,
        reviewer_confirmed: bool = False,
    ) -> WindowSamples:
        """Read only the referenced lead's original samples after fresh source binding.

        Args:
            window_digest: Exact member of this locally built/restored window set.
            waveform: Immutable original acquisition, including optional channels.
            reviewer_confirmed: Explicit permission to expose samples to a local viewer.

        Returns:
            Protected original normalized mV values and their source-relative grid.

        Raises:
            EvidenceWindowError: For absent confirmation, stale source or unknown window.

        This is not authentication, a viewer sandbox or secure memory erasure.
        """
        _require(reviewer_confirmed is True)
        _digest(window_digest)
        _require(type(waveform) is EcgWaveform)
        _require(waveform_fingerprint(waveform) == self.candidates.waveform_digest)
        window = next(
            (item for item in self.windows if item.digest == window_digest), None
        )
        _require(window is not None)
        assert window is not None
        lead = next(
            (item for item in waveform.leads if item.lead_id == window.lead), None
        )
        _require(lead is not None)
        assert lead is not None
        _require(window.source_sample_count == len(lead.samples_mv))
        return WindowSamples(
            window.digest,
            window.lead,
            lead.samples_mv[window.start_sample : window.end_sample],
            window.start_sample,
            waveform.sample_rate_hz,
            waveform.start_offset_seconds
            + window.start_sample / waveform.sample_rate_hz,
        )


def _transforms(digests: tuple[str, ...]) -> None:
    _require(type(digests) is tuple and len(digests) <= 32)
    for item in digests:
        _digest(item)


def _fiducials(
    beat: int, wave: str, boundary: WaveBoundary
) -> tuple[WindowFiducial, ...]:
    return tuple(
        WindowFiducial(FiducialKind(f"{wave}-{point}"), beat, bound.lower, bound.upper)
        for point, bound in (
            ("onset", boundary.onset),
            ("peak", boundary.peak),
            ("offset", boundary.offset),
        )
    )


def _window(
    context: str,
    candidate: TimingCandidate,
    lead: str,
    observation: int,
    waves: tuple[tuple[int, str, WaveBoundary], ...],
    count: int,
    policy: WindowPolicy,
) -> MeasurementWindow:
    start = min(item.window_start for _, _, item in waves) - policy.padding_samples
    end = max(item.window_end for _, _, item in waves) + policy.padding_samples
    _require(0 <= start < end <= count)
    _require(end - start <= policy.maximum_window_samples)
    return MeasurementWindow(
        context,
        _hash(asdict(candidate)),
        lead,
        candidate.kind,
        candidate.state,
        candidate.reasons,
        observation,
        start,
        end,
        count,
        tuple(
            fiducial
            for beat, wave, boundary in waves
            for fiducial in _fiducials(beat, wave, boundary)
        ),
    )


def build_evidence_windows(
    waveform: EcgWaveform,
    quality_policy: SignalQualityPolicy,
    acquisition: AcquisitionEvidence,
    interval_policy: IntervalPolicy,
    window_policy: WindowPolicy,
    *,
    upstream_transform_digests: tuple[str, ...],
    reviewer_confirmed: bool = False,
) -> EcgEvidenceWindows:
    """Extract fresh candidates and attach complete original-grid lead windows.

    Args:
        waveform: Original normalized source with explicit missingness and offsets.
        quality_policy: Independently chosen acquisition and required-lead policy.
        acquisition: Bound layout/calibration evidence, reassessed on every call.
        interval_policy: Actual local timing extractor settings.
        window_policy: Context padding and total resource limits.
        upstream_transform_digests: Ordered host-provided preprocessing provenance;
            explicit empty tuple declares no upstream transforms. Never inferred.
        reviewer_confirmed: Explicit permission to measure this acquisition now.

    Returns:
        Review windows and protected candidates, including any abstentions. Missing
        P/T fiducials have no corresponding PR/QT window. Rate windows retain both
        consecutive ventricular complexes, not just a single anchor.

    Raises:
        EvidenceWindowError: For invalid provenance, incomplete padding or resource limits.
        SignalQualityError: When fresh acquisition quality or confirmation blocks work.
        IntervalCandidateError: For an invalid detector declaration.
    """
    _require(type(window_policy) is WindowPolicy)
    _transforms(upstream_transform_digests)
    candidates = extract_interval_candidates(
        waveform,
        quality_policy,
        acquisition,
        interval_policy,
        reviewer_confirmed=reviewer_confirmed,
    )
    transform = _hash(
        {
            "upstream": upstream_transform_digests,
            "waveform_digest": candidates.waveform_digest,
            "acquisition_digest": candidates.quality.acquisition_digest,
            "detector": ALGORITHM_VERSION,
            "detector_policy": candidates.policy_digest,
            "detection_centering": "record-median",
            "window_samples": "original-normalized-mv-no-additional-filter",
            "window_schema": WINDOW_SCHEMA,
            "window_policy": window_policy.digest,
        }
    )
    context = _hash(
        {
            "candidates": candidates.digest,
            "transform": transform,
            "quality_policy": candidates.quality.policy_digest,
        }
    )
    by_kind = {item.kind: item for item in candidates.measurements}
    windows: list[MeasurementWindow] = []
    total = 0
    for lead in candidates.leads:
        for index, beat in enumerate(lead.beats):
            qrs = ((index, "qrs", beat.qrs),)
            selections: list[
                tuple[TimingKind, tuple[tuple[int, str, WaveBoundary], ...]]
            ] = [(TimingKind.QRS, qrs)]
            if beat.p is not None:
                selections.append((TimingKind.PR, ((index, "p", beat.p), *qrs)))
            if beat.t is not None:
                selections.append((TimingKind.QT, (*qrs, (index, "t", beat.t))))
            if index + 1 < len(lead.beats):
                selections.append(
                    (
                        TimingKind.RATE,
                        (*qrs, (index + 1, "qrs", lead.beats[index + 1].qrs)),
                    )
                )
            for kind, waves in selections:
                window = _window(
                    context,
                    by_kind[kind],
                    lead.lead,
                    index,
                    waves,
                    candidates.sample_count,
                    window_policy,
                )
                total += window.end_sample - window.start_sample
                _require(
                    total <= window_policy.maximum_total_samples
                    and len(windows) < _MAX_WINDOWS
                )
                windows.append(window)
    return EcgEvidenceWindows(
        candidates,
        window_policy,
        upstream_transform_digests,
        transform,
        context,
        tuple(windows),
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("duplicate metadata key")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite metadata number")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite metadata number")
    return result


def _typed_equal(actual: object, expected: object) -> bool:
    # Do not allow True == 1 or 1.0 == 1 for integer sample offsets.
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_typed_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_typed_equal(a, b) for a, b in zip(actual, expected))
        )
    if type(expected) is float:
        return type(actual) in (int, float) and actual == expected
    return type(actual) is type(expected) and actual == expected


def restore_evidence_windows(
    metadata_json: str,
    waveform: EcgWaveform,
    quality_policy: SignalQualityPolicy,
    acquisition: AcquisitionEvidence,
    interval_policy: IntervalPolicy,
    window_policy: WindowPolicy,
    *,
    upstream_transform_digests: tuple[str, ...],
    reviewer_confirmed: bool = False,
) -> EcgEvidenceWindows:
    """Restore only metadata matching freshly regenerated source-bound evidence.

    Args:
        metadata_json: Bounded JSON from to_metadata_json; key order/whitespace may vary.
        waveform: Original local source, required for replay; never read from a path.
        quality_policy: Current acquisition policy for fresh quality assessment.
        acquisition: Current bound original acquisition evidence.
        interval_policy: Expected actual timing detector policy.
        window_policy: Expected context/resource policy.
        upstream_transform_digests: Expected ordered upstream provenance, not from JSON.
        reviewer_confirmed: Explicit permission for fresh measurement on restoration.

    Returns:
        Freshly computed windows; no untrusted JSON DTO can upgrade an abstention.

    Raises:
        EvidenceWindowError: On malformed, stale, duplicate-key or tampered metadata.
        SignalQualityError: When current source quality or confirmation blocks replay.
        IntervalCandidateError: For invalid detector declarations.

    Restoration never loads serialized samples, paths, filters or executable code.
    Hashes and caller declarations are not authenticated acquisition/review receipts.
    """
    _require(type(metadata_json) is str and len(metadata_json) <= _MAX_METADATA_CHARS)
    parsed = None
    try:
        parsed = json.loads(
            metadata_json,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (ValueError, TypeError, RecursionError):
        pass
    # Outside exception context so malformed sensitive input cannot leak via traceback.
    _require(type(parsed) is dict)
    result = build_evidence_windows(
        waveform,
        quality_policy,
        acquisition,
        interval_policy,
        window_policy,
        upstream_transform_digests=upstream_transform_digests,
        reviewer_confirmed=reviewer_confirmed,
    )
    expected = json.loads(result.to_metadata_json())
    try:
        matches = _typed_equal(parsed, expected)
    except (ValueError, TypeError, RecursionError, OverflowError):
        matches = False
    _require(matches)
    return result
