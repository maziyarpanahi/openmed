"""Structural ECG lead/layout review, without diagnosing waveform content."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .waveform import _LEAD_ALIASES, _LEADS, NON_DIAGNOSTIC_NOTICE

LEAD_LAYOUT_VERSION = "openmed.ecg.lead-layout.v1"
STANDARD_LEADS = _LEADS
_MAX_CHANNELS = 64


class LeadLayoutError(ValueError):
    """Value-free malformed declaration or observation error."""


class AcquisitionKind(str, Enum):
    """Origin of lead declarations; image labels require explicit confirmation."""

    DIGITAL = "digital"
    IMAGE_DERIVED = "image_derived"


class SlotRole(str, Enum):
    """Primary lead versus an explicitly declared repeated rhythm strip."""

    PRIMARY = "primary"
    RHYTHM = "rhythm_strip"


class LeadFindingCode(str, Enum):
    """Controlled structural findings that require human review."""

    UNKNOWN_LEAD = "unknown_lead"
    MISSING_SLOT = "missing_slot"
    UNDECLARED_SLOT = "undeclared_slot"
    DUPLICATE_SLOT = "duplicate_slot"
    WRONG_LEAD = "lead_order_mismatch"
    DUPLICATE_LEAD = "duplicate_primary_lead"
    TIMING_UNKNOWN = "timing_unverified"
    TIMING_MISMATCH = "timing_misaligned"
    SAMPLING_UNKNOWN = "sampling_unverified"
    SAMPLING_MISMATCH = "sampling_misaligned"
    LABEL_REVIEW = "image_labels_unconfirmed"


def _require(condition: bool) -> None:
    if not condition:
        raise LeadLayoutError("invalid ECG lead layout")


def _integer(value: object, maximum: int) -> None:
    _require(type(value) is int and 0 <= value <= maximum)


def _number(value: object, maximum: float) -> None:
    _require((type(value) is float or type(value) is int) and 0 <= value <= maximum)


def _canonical(name: str) -> str | None:
    return _LEAD_ALIASES.get(name.strip().lower())


@dataclass(frozen=True, slots=True, kw_only=True)
class LeadSlot:
    """Expected display position and timing group for a canonical ECG lead.

    Args:
        lead: Canonical lead name, without patient labels.
        row: Zero-based display row.
        column: Zero-based display column.
        timing_group: Slots in this group must share acquisition timing and sampling.
        role: Primary lead, or explicitly repeated rhythm strip.
    """

    lead: str
    row: int
    column: int
    timing_group: int = 0
    role: SlotRole = SlotRole.PRIMARY

    def __post_init__(self) -> None:
        _require(type(self.lead) is str and self.lead in STANDARD_LEADS)
        for value in (self.row, self.column, self.timing_group):
            _integer(value, _MAX_CHANNELS - 1)
        _require(type(self.role) is SlotRole)


@dataclass(frozen=True, slots=True, kw_only=True)
class LeadLayout:
    """Explicit expected lead set and layout, independent of observed labels.

    Args:
        rows: Declared display height, one to 64 rows.
        columns: Declared display width, one to 64 columns.
        expected_leads: Unique canonical primary leads required by this acquisition.
        slots: Unique occupied positions; primary slots must match the expected set.
    """

    rows: int
    columns: int
    expected_leads: tuple[str, ...]
    slots: tuple[LeadSlot, ...]

    def __post_init__(self) -> None:
        _integer(self.rows, _MAX_CHANNELS)
        _integer(self.columns, _MAX_CHANNELS)
        _require(self.rows > 0 and self.columns > 0)
        _require(
            type(self.expected_leads) is tuple and 1 <= len(self.expected_leads) <= 12
        )
        _require(
            all(
                type(lead) is str and lead in STANDARD_LEADS
                for lead in self.expected_leads
            )
        )
        _require(len(set(self.expected_leads)) == len(self.expected_leads))
        _require(type(self.slots) is tuple and 1 <= len(self.slots) <= _MAX_CHANNELS)
        _require(all(type(slot) is LeadSlot for slot in self.slots))
        _require(
            len({(slot.row, slot.column) for slot in self.slots}) == len(self.slots)
        )
        _require(
            all(
                slot.row < self.rows and slot.column < self.columns
                for slot in self.slots
            )
        )
        primary = [slot.lead for slot in self.slots if slot.role is SlotRole.PRIMARY]
        _require(
            len(primary) == len(self.expected_leads)
            and set(primary) == set(self.expected_leads)
        )
        _require(all(slot.lead in self.expected_leads for slot in self.slots))
        object.__setattr__(
            self, "slots", tuple(sorted(self.slots, key=lambda s: (s.row, s.column)))
        )
        object.__setattr__(
            self,
            "expected_leads",
            tuple(lead for lead in STANDARD_LEADS if lead in self.expected_leads),
        )

    @classmethod
    def digital(cls, expected_leads: tuple[str, ...] = STANDARD_LEADS) -> LeadLayout:
        """Declare simultaneous digital channels in the supplied expected order."""
        _require(type(expected_leads) is tuple and 1 <= len(expected_leads) <= 12)
        return cls(
            rows=len(expected_leads),
            columns=1,
            expected_leads=expected_leads,
            slots=tuple(
                LeadSlot(lead=lead, row=i, column=0)
                for i, lead in enumerate(expected_leads)
            ),
        )

    @classmethod
    def three_by_four(cls) -> LeadLayout:
        """Declare standard 3×4 display order with one timing group per column.

        Each column may show a different time interval; its three rows must align.
        Use a custom declaration for simultaneous or nonstandard display windows.
        """
        rows = (
            ("I", "aVR", "V1", "V4"),
            ("II", "aVL", "V2", "V5"),
            ("III", "aVF", "V3", "V6"),
        )
        return cls(
            rows=3,
            columns=4,
            expected_leads=STANDARD_LEADS,
            slots=tuple(
                LeadSlot(lead=lead, row=r, column=c, timing_group=c)
                for r, row in enumerate(rows)
                for c, lead in enumerate(row)
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LeadObservation:
    """One observed label and acquisition window before canonical reordering.

    Args:
        label: Protected raw label, excluded from representations and reports.
        row: Observed zero-based display row or digital channel index.
        column: Observed zero-based display column, zero for digital channels.
        start_seconds: Relative source offset; None means unverified timing.
        duration_seconds: Declared positive window duration, or None.
        sample_rate_hz: Positive digital sample rate, or None for undigitized images.
        sample_count: Positive digital sample count, or None.
    """

    label: str = field(repr=False)
    row: int
    column: int = 0
    start_seconds: float | None = None
    duration_seconds: float | None = None
    sample_rate_hz: float | None = None
    sample_count: int | None = None

    def __post_init__(self) -> None:
        _require(type(self.label) is str and 0 < len(self.label) <= 64)
        for value in (self.row, self.column):
            _integer(value, _MAX_CHANNELS - 1)
        _require((self.start_seconds is None) == (self.duration_seconds is None))
        if self.start_seconds is not None:
            _number(self.start_seconds, 86400)
            _number(self.duration_seconds, 86400)
            assert self.duration_seconds is not None
            _require(
                self.duration_seconds > 0
                and self.start_seconds + self.duration_seconds <= 86400
            )
        _require((self.sample_rate_hz is None) == (self.sample_count is None))
        if self.sample_rate_hz is not None:
            _number(self.sample_rate_hz, 100_000)
            _integer(self.sample_count, 1_000_000)
            assert self.sample_count is not None
            _require(self.sample_rate_hz > 0 and self.sample_count > 0)


@dataclass(frozen=True, slots=True)
class LeadFinding:
    """Controlled code and observed channel indices, with no raw labels."""

    code: LeadFindingCode
    channels: tuple[int, ...]
    slot: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        _require(type(self.code) is LeadFindingCode)
        _require(type(self.channels) is tuple and len(self.channels) <= _MAX_CHANNELS)
        for channel in self.channels:
            _integer(channel, _MAX_CHANNELS - 1)
        _require(len(set(self.channels)) == len(self.channels))
        if self.slot is not None:
            _require(type(self.slot) is tuple and len(self.slot) == 2)
            for coordinate in self.slot:
                _integer(coordinate, _MAX_CHANNELS - 1)


@dataclass(frozen=True, slots=True)
class LeadLayoutReport:
    """Structural review findings; never a diagnostic or electrode-placement claim."""

    source_digest: str
    acquisition: AcquisitionKind
    observed_count: int
    expected_leads: tuple[str, ...]
    findings: tuple[LeadFinding, ...]

    def __post_init__(self) -> None:
        _require(
            type(self.source_digest) is str
            and re.fullmatch(r"[0-9a-f]{64}", self.source_digest) is not None
        )
        _require(type(self.acquisition) is AcquisitionKind)
        _integer(self.observed_count, _MAX_CHANNELS)
        _require(
            type(self.expected_leads) is tuple and 1 <= len(self.expected_leads) <= 12
        )
        _require(
            all(
                type(lead) is str and lead in STANDARD_LEADS
                for lead in self.expected_leads
            )
        )
        _require(
            type(self.findings) is tuple and len(self.findings) <= _MAX_CHANNELS * 10
        )
        _require(all(type(finding) is LeadFinding for finding in self.findings))
        _require(
            all(
                channel < self.observed_count
                for finding in self.findings
                for channel in finding.channels
            )
        )

    @property
    def requires_review(self) -> bool:
        """Whether any structural or unverified-label/timing finding remains."""
        return bool(self.findings)

    def to_json(self) -> str:
        """Return deterministic, value-free JSON without observed labels or samples."""
        result: dict[str, Any] = asdict(self)
        result.update(
            schema_version=LEAD_LAYOUT_VERSION,
            requires_review=self.requires_review,
            notice=NON_DIAGNOSTIC_NOTICE,
        )
        return json.dumps(result, sort_keys=True, separators=(",", ":"))


def assess_lead_layout(
    observations: tuple[LeadObservation, ...],
    layout: LeadLayout,
    *,
    source_digest: str,
    acquisition: AcquisitionKind,
    image_labels_confirmed: bool = False,
) -> LeadLayoutReport:
    """Compare observed lead identity, order and timing with an explicit declaration.

    Args:
        observations: Original display/channel positions, including missing/unknown labels.
        layout: Independent expected layout; never infer it from observations.
        source_digest: SHA-256 binding the review to the source bytes.
        acquisition: Digital input or image-derived observations.
        image_labels_confirmed: Explicit caller confirmation of image label review.

    Returns:
        A bounded report of structural anomalies and unverified evidence. Timing
        comparisons use a fixed numerical tolerance of one nanosecond, not a
        clinical safety threshold. Physical electrode swaps require other evidence.
    """
    _require(type(observations) is tuple and len(observations) <= _MAX_CHANNELS)
    _require(all(type(item) is LeadObservation for item in observations))
    _require(type(layout) is LeadLayout and type(acquisition) is AcquisitionKind)
    _require(type(image_labels_confirmed) is bool)
    _require(
        type(source_digest) is str
        and re.fullmatch(r"[0-9a-f]{64}", source_digest) is not None
    )
    slots = {(slot.row, slot.column): slot for slot in layout.slots}
    positions: dict[tuple[int, int], list[int]] = defaultdict(list)
    leads: dict[str, list[int]] = defaultdict(list)
    groups: dict[int, list[int]] = defaultdict(list)
    findings: list[LeadFinding] = []
    for index, item in enumerate(observations):
        position = (item.row, item.column)
        positions[position].append(index)
        canonical = _canonical(item.label)
        if canonical is None:
            findings.append(
                LeadFinding(LeadFindingCode.UNKNOWN_LEAD, (index,), position)
            )
        expected = slots.get(position)
        if expected is None:
            findings.append(
                LeadFinding(LeadFindingCode.UNDECLARED_SLOT, (index,), position)
            )
        else:
            if canonical is not None and canonical != expected.lead:
                findings.append(
                    LeadFinding(LeadFindingCode.WRONG_LEAD, (index,), position)
                )
            if canonical is not None and expected.role is SlotRole.PRIMARY:
                leads[canonical].append(index)
            groups[expected.timing_group].append(index)
        if item.start_seconds is None:
            findings.append(
                LeadFinding(LeadFindingCode.TIMING_UNKNOWN, (index,), position)
            )
        if item.sample_rate_hz is None:
            if acquisition is AcquisitionKind.DIGITAL:
                findings.append(
                    LeadFinding(LeadFindingCode.SAMPLING_UNKNOWN, (index,), position)
                )
        elif item.duration_seconds is not None:
            assert item.sample_count is not None
            if not math.isclose(
                item.sample_count / item.sample_rate_hz,
                item.duration_seconds,
                rel_tol=0,
                abs_tol=1e-9,
            ):
                findings.append(
                    LeadFinding(LeadFindingCode.SAMPLING_MISMATCH, (index,), position)
                )
    for position in sorted(slots):
        indices = positions.get(position, [])
        if not indices:
            findings.append(LeadFinding(LeadFindingCode.MISSING_SLOT, (), position))
    for position, indices in sorted(positions.items()):
        if len(indices) > 1:
            findings.append(
                LeadFinding(LeadFindingCode.DUPLICATE_SLOT, tuple(indices), position)
            )
    for indices in leads.values():
        if len(indices) > 1:
            findings.append(LeadFinding(LeadFindingCode.DUPLICATE_LEAD, tuple(indices)))
    for indices in groups.values():
        timed = [i for i in indices if observations[i].start_seconds is not None]
        if timed:
            first = observations[timed[0]]
            if any(
                not math.isclose(
                    getattr(observations[i], field),
                    getattr(first, field),
                    rel_tol=0,
                    abs_tol=1e-9,
                )
                for i in timed[1:]
                for field in ("start_seconds", "duration_seconds")
            ):
                findings.append(
                    LeadFinding(LeadFindingCode.TIMING_MISMATCH, tuple(timed))
                )
        sampled = [i for i in indices if observations[i].sample_rate_hz is not None]
        if sampled:
            first = observations[sampled[0]]
            if any(
                observations[i].sample_rate_hz != first.sample_rate_hz
                or observations[i].sample_count != first.sample_count
                for i in sampled[1:]
            ):
                findings.append(
                    LeadFinding(LeadFindingCode.SAMPLING_MISMATCH, tuple(sampled))
                )
    if acquisition is AcquisitionKind.IMAGE_DERIVED and not image_labels_confirmed:
        findings.append(
            LeadFinding(LeadFindingCode.LABEL_REVIEW, tuple(range(len(observations))))
        )
    return LeadLayoutReport(
        source_digest,
        acquisition,
        len(observations),
        layout.expected_leads,
        tuple(
            sorted(
                findings,
                key=lambda finding: (
                    finding.code,
                    finding.channels,
                    finding.slot or (-1, -1),
                ),
            )
        ),
    )
