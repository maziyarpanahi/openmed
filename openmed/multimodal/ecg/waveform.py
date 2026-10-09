"""Canonical ECG samples and content-free validation, without interpretation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Final

__all__ = [
    "ECG_SCHEMA_VERSION",
    "NON_DIAGNOSTIC_NOTICE",
    "EcgLead",
    "EcgProvenance",
    "EcgWaveform",
    "WaveformValidationError",
]

ECG_SCHEMA_VERSION: Final = "openmed.multimodal.ecg.waveform.v1"
NON_DIAGNOSTIC_NOTICE: Final = (
    "Non-diagnostic ECG input normalization only. No disease interpretation or "
    "clinical validation. Explicit reviewer confirmation is required before "
    "consequential use; never automatically trigger clinical decisions."
)
_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
_LEAD_ALIASES = {lead.lower(): lead for lead in _LEADS}
_SOURCE_FORMATS = frozenset({"array", "dicom", "edf", "wfdb", "hl7-aecg"})
_MAX_SAMPLES = 1_000_000
_MAX_DURATION_SECONDS = 86_400


class WaveformValidationError(ValueError):
    """Static actionable reason and non-diagnostic boundary; no input values."""


def _reject(reason: str) -> None:
    raise WaveformValidationError(f"{reason}. {NON_DIAGNOSTIC_NOTICE}")


def _number(value: object, reason: str) -> float:
    if type(value) not in (int, float):
        _reject(reason)
    # Avoid native overflow errors, including their exception context.
    try:
        result = float(value)
    except (ValueError, OverflowError):
        pass
    else:
        if math.isfinite(result):
            return result
    _reject(reason)


def _sequence(value: object, reason: str) -> tuple:
    if type(value) not in (list, tuple) or not 0 < len(value) <= _MAX_SAMPLES:
        _reject(reason)
    return tuple(value)


@dataclass(frozen=True, slots=True)
class EcgProvenance:
    """Source format and caller-computed SHA-256; never paths or patient headers.

    Args:
        source_format: One of array, dicom, edf, wfdb, or hl7-aecg.
        source_digest: Lowercase SHA-256 of the caller's original source bytes.
    """

    source_format: str
    source_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.source_format) is not str
            or self.source_format not in _SOURCE_FORMATS
        ):
            _reject("source_format_unsupported: supply a supported source format")
        if (
            type(self.source_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.source_digest) is None
        ):
            _reject("source_digest_invalid: supply a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class EcgLead:
    """Immutable canonical lead with millivolt samples and a validity mask.

    Args:
        lead_id: Standard lead identifier, case-insensitive, with whitespace allowed.
        samples_mv: Finite millivolt samples; missing samples must be None.
        valid_mask: Booleans, true for observed samples and false for missing samples.

    Use from_samples for unit conversion or ADC calibration. Samples are sensitive
    in-memory data, excluded from repr and reports; this is not de-identification.
    """

    lead_id: str
    samples_mv: tuple[float | None, ...] = field(repr=False)
    valid_mask: tuple[bool, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.lead_id) is not str:
            _reject("lead_unsupported: supply a standard ECG lead identifier")
        lead = _LEAD_ALIASES.get(self.lead_id.strip().lower())
        if lead is None:
            _reject("lead_unsupported: supply a standard ECG lead identifier")
        samples = _sequence(self.samples_mv, "samples_invalid: supply bounded samples")
        mask = _sequence(
            self.valid_mask, "mask_invalid: supply a boolean validity mask"
        )
        if len(samples) != len(mask) or any(type(flag) is not bool for flag in mask):
            _reject("mask_invalid: match sample length with boolean flags")
        normalized = []
        for sample, valid in zip(samples, mask):
            if not valid:
                if sample is not None:
                    _reject("missingness_ambiguous: use None for each masked sample")
                normalized.append(None)
            else:
                normalized.append(
                    _number(sample, "sample_invalid: observed samples must be finite")
                )
        object.__setattr__(self, "lead_id", lead)
        object.__setattr__(self, "samples_mv", tuple(normalized))
        object.__setattr__(self, "valid_mask", mask)

    @classmethod
    def from_samples(
        cls,
        lead_id: str,
        samples: list[float | None] | tuple[float | None, ...],
        *,
        unit: str,
        valid_mask: list[bool] | tuple[bool, ...] | None = None,
        gain_counts_per_mv: float | None = None,
        baseline_counts: float | None = None,
    ) -> EcgLead:
        """Convert explicit physical units or calibrated ADC counts to millivolts.

        Args:
            lead_id: Standard lead identifier.
            samples: Finite numbers or explicitly missing None samples.
            unit: V, mV, uV, µV, μV, or adc; spelling is case-sensitive.
            valid_mask: Optional exact booleans; omitted mask derives from None.
            gain_counts_per_mv: Positive gain required only for adc.
            baseline_counts: Finite zero level required only for adc.

        Returns:
            An immutable canonical lead, preserving sample order and missingness.

        Raises:
            WaveformValidationError: For unsupported or ambiguous input.
        """
        values = _sequence(samples, "samples_invalid: supply bounded samples")
        scales = {"V": 1000.0, "mV": 1.0, "uV": 0.001, "µV": 0.001, "μV": 0.001}
        if type(unit) is not str or unit not in (*scales, "adc"):
            _reject("unit_unsupported: specify V, mV, uV, or calibrated adc")
        if unit == "adc":
            gain = _number(gain_counts_per_mv, "gain_invalid: supply positive ADC gain")
            baseline = _number(
                baseline_counts, "baseline_invalid: supply ADC zero level"
            )
            if gain <= 0:
                _reject("gain_invalid: supply positive ADC gain")
        else:
            if gain_counts_per_mv is not None or baseline_counts is not None:
                _reject("calibration_ambiguous: calibration is accepted only for adc")
            gain, baseline = 1.0, 0.0
        converted = []
        for value in values:
            if value is None:
                converted.append(None)
            else:
                number = _number(
                    value, "sample_invalid: observed samples must be finite"
                )
                converted.append(
                    (number - baseline) / gain
                    if unit == "adc"
                    else number * scales[unit]
                )
        mask = (
            tuple(value is not None for value in values)
            if valid_mask is None
            else valid_mask
        )
        return cls(lead_id, tuple(converted), mask)


@dataclass(frozen=True, slots=True)
class EcgWaveform:
    """Aligned regular ECG input, with relative seconds and canonical lead order.

    Args:
        leads: One to twelve unique leads sharing sample count and timing.
        sample_rate_hz: Explicit finite positive rate, at most 100,000 Hz.
        provenance: Digest-only source provenance.
        start_offset_seconds: Relative offset within the source, never wall-clock time.
        sample_offsets_seconds: Optional explicit regular relative sample times.
            They must agree with start_offset_seconds + index / sample_rate_hz.

    No interpolation, resampling, lead derivation, diagnosis, or clinical decision
    is performed. Raw samples and per-sample offsets are omitted from repr/reports.
    """

    leads: tuple[EcgLead, ...] = field(repr=False)
    sample_rate_hz: float
    provenance: EcgProvenance
    start_offset_seconds: float = field(default=0.0, repr=False)
    sample_offsets_seconds: tuple[float, ...] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.leads) not in (list, tuple) or not 1 <= len(self.leads) <= 12:
            _reject("leads_invalid: supply one to twelve standard leads")
        if any(type(lead) is not EcgLead for lead in self.leads):
            _reject("leads_invalid: supply validated canonical leads")
        if len({lead.lead_id for lead in self.leads}) != len(self.leads):
            _reject("lead_duplicate: supply each canonical lead only once")
        count = len(self.leads[0].samples_mv)
        if any(len(lead.samples_mv) != count for lead in self.leads):
            _reject("alignment_invalid: leads must share sample count and timing")
        if count * len(self.leads) > _MAX_SAMPLES:
            _reject("resource_limit: split input into smaller aligned records")
        if type(self.provenance) is not EcgProvenance:
            _reject("provenance_invalid: supply validated digest-only provenance")
        rate = _number(self.sample_rate_hz, "rate_invalid: supply finite positive Hz")
        if not 0 < rate <= 100_000:
            _reject("rate_invalid: supply positive Hz at most 100000")
        start = _number(
            self.start_offset_seconds, "timing_invalid: supply relative seconds"
        )
        if (
            not 0 <= start <= _MAX_DURATION_SECONDS
            or count / rate > _MAX_DURATION_SECONDS
        ):
            _reject("timing_limit: use a bounded relative recording window")
        # At extremely low rates, even a single-sample record has excessive duration.
        offsets = tuple(start + index / rate for index in range(count))
        if offsets[-1] > _MAX_DURATION_SECONDS:
            _reject("timing_limit: use a bounded relative recording window")
        if self.sample_offsets_seconds is not None:
            supplied = _sequence(
                self.sample_offsets_seconds, "timing_invalid: supply sample offsets"
            )
            if len(supplied) != count:
                _reject("timing_invalid: match sample offsets to sample count")
            for actual, expected in zip(supplied, offsets):
                actual = _number(
                    actual, "timing_invalid: supply finite relative seconds"
                )
                if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9 / rate):
                    _reject(
                        "timing_ambiguous: offsets must match the regular rate grid"
                    )
        object.__setattr__(
            self,
            "leads",
            tuple(sorted(self.leads, key=lambda lead: _LEADS.index(lead.lead_id))),
        )
        object.__setattr__(self, "sample_rate_hz", rate)
        object.__setattr__(self, "start_offset_seconds", start)
        object.__setattr__(self, "sample_offsets_seconds", offsets)

    @property
    def notice(self) -> str:
        """Return the mandatory non-diagnostic boundary attached to this record."""
        return NON_DIAGNOSTIC_NOTICE

    def require_reviewer_confirmation(self, *, confirmed: bool = False) -> None:
        """Fail closed unless the caller explicitly confirms consequential use.

        Confirmation records a caller decision, not clinical qualification. Every
        downstream consequential output must retain the notice and its own gate.
        """
        if confirmed is not True:
            _reject("review_required: obtain explicit reviewer confirmation")

    def to_report(self) -> dict[str, object]:
        """Return content-free metadata; never samples, times, or source headers."""
        return {
            "schema_version": ECG_SCHEMA_VERSION,
            "notice": self.notice,
            "review_required": True,
            "source_format": self.provenance.source_format,
            "source_digest": self.provenance.source_digest,
            "unit": "mV",
            "sample_rate_hz": self.sample_rate_hz,
            "sample_count": len(self.leads[0].samples_mv),
            "leads": [
                {"lead_id": lead.lead_id, "missing_count": lead.valid_mask.count(False)}
                for lead in self.leads
            ],
        }

    def to_json(self) -> str:
        """Serialize only the content-free report, deterministically."""
        return json.dumps(
            self.to_report(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
