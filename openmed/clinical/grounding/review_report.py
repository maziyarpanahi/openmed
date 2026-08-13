"""Human-reviewable code-to-span reports for grounded clinical concepts.

The report is intentionally built from already-redacted grounding spans. It
contains only the span surface supplied by the caller, offsets, terminology
metadata, and confidence fields; it never receives or stores the surrounding
source document. The reverse index is the same canonical URI/code index used by
the FHIR CodeableConcept exporter.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openmed.clinical.exporters.codeable_concept import (
    SYSTEM_URI,
    build_reverse_index,
    to_codeable_concept,
)

from .calibrate import (
    ACCEPT_BAND,
    DEFAULT_ACCEPT_THRESHOLD,
    UNCERTAIN_BAND,
    GroundingConfidenceCalibrator,
)
from .calibration import GroundingCalibrator
from .types import GROUNDING_CONFIDENCE_BANDS, Candidate, GroundedSpan

REVIEW_REPORT_SCHEMA_VERSION = 1
GROUNDING_REVIEW_REPORT_ARTIFACT = "openmed.grounding.review_report"
GROUNDING_REVIEW_REPORT_ADVISORY = (
    "Grounding assignments are assistive outputs and require human verification; "
    "this report is not a clinical decision or coding authorization."
)


@dataclass(frozen=True)
class GroundingReviewEntry:
    """One assigned code paired with its source span and confidence."""

    span_text: str
    start: int
    end: int
    system: str
    code: str
    display: str
    raw_score: float
    calibrated_confidence: float | None
    band: str
    system_uri: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.span_text, str):
            raise TypeError("review span text must be a string")
        if type(self.start) is not int or self.start < 0:
            raise ValueError("review span start must be a non-negative integer")
        if type(self.end) is not int or self.end < self.start:
            raise ValueError("review span end must be at or after start")
        raw_score = _bounded_probability(self.raw_score, "raw_score")
        calibrated = (
            None
            if self.calibrated_confidence is None
            else _bounded_probability(
                self.calibrated_confidence,
                "calibrated_confidence",
            )
        )
        band = str(self.band).strip().lower()
        if band not in GROUNDING_CONFIDENCE_BANDS:
            raise ValueError("band must be 'accept' or 'uncertain'")
        if not self.system or not self.code:
            raise ValueError("review entries require a system and code")
        object.__setattr__(self, "raw_score", raw_score)
        object.__setattr__(self, "calibrated_confidence", calibrated)
        object.__setattr__(self, "band", band)
        if self.system_uri is None:
            object.__setattr__(self, "system_uri", _system_uri(self.system))

    @property
    def source_text(self) -> str:
        """Alias for the already-redacted span surface."""

        return self.span_text

    @property
    def source_span(self) -> str:
        """Alias matching the terminology used by the issue."""

        return self.span_text

    @property
    def confidence(self) -> float | None:
        """Alias for the calibrated confidence value."""

        return self.calibrated_confidence

    @property
    def accepted(self) -> bool:
        """Return whether the link is in the accept band."""

        return self.band == ACCEPT_BAND

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-ready review entry."""

        return {
            "span_text": self.span_text,
            "source_span": self.span_text,
            "start": self.start,
            "end": self.end,
            "offsets": [self.start, self.end],
            "system": self.system,
            "system_uri": self.system_uri,
            "code": self.code,
            "display": self.display,
            "raw_score": self.raw_score,
            "calibrated_confidence": self.calibrated_confidence,
            "band": self.band,
            "accepted": self.accepted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GroundingReviewEntry":
        """Build an entry from :meth:`to_dict` output."""

        offsets = payload.get("offsets") or ()
        start = payload.get("start", offsets[0] if len(offsets) > 0 else None)
        end = payload.get("end", offsets[1] if len(offsets) > 1 else None)
        if start is None or end is None:
            raise ValueError("review entry requires start and end offsets")
        return cls(
            span_text=str(payload.get("span_text", payload.get("source_span", ""))),
            start=int(start),
            end=int(end),
            system=str(payload["system"]),
            code=str(payload["code"]),
            display=str(payload.get("display", "")),
            raw_score=float(payload["raw_score"]),
            calibrated_confidence=(
                None
                if payload.get("calibrated_confidence") is None
                else float(payload["calibrated_confidence"])
            ),
            band=str(payload["band"]),
            system_uri=(
                None
                if payload.get("system_uri") is None
                else str(payload["system_uri"])
            ),
        )


@dataclass(frozen=True)
class GroundingReviewReport:
    """Deterministic JSON and Markdown artifact for grounding review."""

    entries: tuple[GroundingReviewEntry, ...]
    reverse_index: Mapping[tuple[str, str], tuple[tuple[int, int], ...]] = field(
        default_factory=dict
    )
    generated_at: str | None = None
    schema_version: int = REVIEW_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if any(not isinstance(entry, GroundingReviewEntry) for entry in entries):
            raise TypeError("entries must contain GroundingReviewEntry values")
        index = {
            (str(system), str(code)): tuple(
                (int(start), int(end)) for start, end in offsets
            )
            for (system, code), offsets in self.reverse_index.items()
        }
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "reverse_index", index)

    def __len__(self) -> int:
        """Return the number of assigned-code entries."""

        return len(self.entries)

    def __iter__(self):
        """Iterate over assigned-code entries."""

        return iter(self.entries)

    def to_dict(self) -> dict[str, Any]:
        """Return the report payload with a JSON-safe reverse index."""

        return {
            "schema_version": self.schema_version,
            "artifact_type": GROUNDING_REVIEW_REPORT_ARTIFACT,
            "generated_at": self.generated_at,
            "entry_count": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
            "reverse_index": [
                {
                    "system_uri": system,
                    "code": code,
                    "offsets": [list(offset) for offset in offsets],
                }
                for (system, code), offsets in sorted(self.reverse_index.items())
            ],
            "advisory": GROUNDING_REVIEW_REPORT_ADVISORY,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GroundingReviewReport":
        """Build a report from :meth:`to_dict` output."""

        raw_index = payload.get("reverse_index", ())
        index: dict[tuple[str, str], tuple[tuple[int, int], ...]] = {}
        if isinstance(raw_index, Mapping):
            for key, offsets in raw_index.items():
                system, separator, code = str(key).partition("|")
                if not separator:
                    continue
                index[(system, code)] = tuple(tuple(offset) for offset in offsets)
        else:
            for row in raw_index:
                if not isinstance(row, Mapping):
                    continue
                key = (str(row["system_uri"]), str(row["code"]))
                index[key] = tuple(tuple(offset) for offset in row["offsets"])
        return cls(
            entries=tuple(
                GroundingReviewEntry.from_dict(entry)
                for entry in payload.get("entries", ())
            ),
            reverse_index=index,
            generated_at=payload.get("generated_at"),
            schema_version=int(
                payload.get("schema_version", REVIEW_REPORT_SCHEMA_VERSION)
            ),
        )

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report as deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON to *path*."""

        output_path = Path(path)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Serialize the report as a reviewer-friendly Markdown table."""

        lines = [
            "# Grounding Review Report",
            "",
            GROUNDING_REVIEW_REPORT_ADVISORY,
            "",
            "| Source span | Offsets | System | Code | Display | Raw score | "
            "Calibrated confidence | Band |",
            "|---|---|---|---|---|---:|---:|---|",
        ]
        for entry in self.entries:
            confidence = (
                "—"
                if entry.calibrated_confidence is None
                else f"{entry.calibrated_confidence:.6f}"
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        _markdown_cell(entry.span_text),
                        f"{entry.start}:{entry.end}",
                        _markdown_cell(entry.system),
                        _markdown_cell(entry.code),
                        _markdown_cell(entry.display),
                        f"{entry.raw_score:.6f}",
                        confidence,
                        entry.band,
                    )
                )
                + " |"
            )
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write the Markdown report to *path*."""

        output_path = Path(path)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path


def build_review_report(
    grounded_spans: Iterable[GroundedSpan],
    calibrator: GroundingConfidenceCalibrator | GroundingCalibrator | None = None,
    *,
    threshold: float | None = None,
    label: str | None = None,
    include_uncertain: bool = True,
    generated_at: str | None = None,
) -> GroundingReviewReport:
    """Build a code-to-span report from already-redacted grounded spans.

    ``calibrator`` is optional when spans were calibrated before report
    construction. When supplied, every assigned candidate is calibrated so
    multi-vocabulary spans receive the correct vocabulary-specific confidence.
    Uncertain links are retained by default for human review.
    """

    spans = tuple(grounded_spans)
    review_spans = tuple(span for span in spans if not span.abstained)
    reverse_index_raw = build_reverse_index(review_spans)
    reverse_index = {key: tuple(offsets) for key, offsets in reverse_index_raw.items()}
    confidence_calibrator = _as_confidence_calibrator(calibrator, threshold)
    entries: list[GroundingReviewEntry] = []

    for span in review_spans:
        if not span.candidates:
            continue
        concept = to_codeable_concept(span)
        candidates_by_key = {
            (_system_uri(candidate.system), candidate.code): candidate
            for candidate in span.candidates
        }
        for coding in concept.get("coding", ()):
            key = (str(coding["system"]), str(coding["code"]))
            candidate = candidates_by_key[key]
            confidence, band = _candidate_confidence(
                span,
                candidate,
                confidence_calibrator,
                label=label,
                threshold=threshold,
            )
            if not include_uncertain and band == UNCERTAIN_BAND:
                continue
            entries.append(
                GroundingReviewEntry(
                    span_text=span.text,
                    start=span.start,
                    end=span.end,
                    system=candidate.system,
                    system_uri=key[0],
                    code=candidate.code,
                    display=candidate.display,
                    raw_score=float(candidate.score),
                    calibrated_confidence=confidence,
                    band=band,
                )
            )

    return GroundingReviewReport(
        entries=tuple(entries),
        reverse_index=reverse_index,
        generated_at=generated_at,
    )


def build_grounding_review_report(*args: Any, **kwargs: Any) -> GroundingReviewReport:
    """Explicitly named alias for :func:`build_review_report`."""

    return build_review_report(*args, **kwargs)


def build_code_span_review_report(
    *args: Any,
    **kwargs: Any,
) -> GroundingReviewReport:
    """Alias naming the report's code-to-span review purpose."""

    return build_review_report(*args, **kwargs)


def _as_confidence_calibrator(
    calibrator: GroundingConfidenceCalibrator | GroundingCalibrator | None,
    threshold: float | None,
) -> GroundingConfidenceCalibrator | None:
    if isinstance(calibrator, GroundingConfidenceCalibrator):
        return calibrator
    if isinstance(calibrator, GroundingCalibrator):
        return GroundingConfidenceCalibrator(
            model=calibrator,
            threshold=(DEFAULT_ACCEPT_THRESHOLD if threshold is None else threshold),
        )
    if calibrator is not None:
        raise TypeError("calibrator must be a grounding calibrator")
    return None


def _candidate_confidence(
    span: GroundedSpan,
    candidate: Candidate,
    calibrator: GroundingConfidenceCalibrator | None,
    *,
    label: str | None,
    threshold: float | None,
) -> tuple[float | None, str]:
    if calibrator is not None:
        result = calibrator.classify(
            candidate.system,
            candidate.score,
            label=label or span.canonical_label,
            threshold=threshold,
        )
        return result.calibrated_confidence, result.band

    metadata_result = _metadata_candidate_result(span, candidate)
    if metadata_result is not None:
        confidence = metadata_result.get("calibrated_confidence")
        band = metadata_result.get("band")
        if confidence is not None and band in GROUNDING_CONFIDENCE_BANDS:
            return float(confidence), str(band)

    is_top_candidate = span.candidates and candidate == span.candidates[0]
    if is_top_candidate and span.calibrated_confidence is not None:
        confidence = float(span.calibrated_confidence)
        band = span.confidence_band
        if band in GROUNDING_CONFIDENCE_BANDS:
            return confidence, band
        return confidence, _band_from_values(confidence, DEFAULT_ACCEPT_THRESHOLD)

    # An uncalibrated span can still be rendered for review, but the raw score
    # is explicitly marked as the provisional confidence fallback.
    raw_score = float(candidate.score)
    return raw_score, _band_from_values(raw_score, DEFAULT_ACCEPT_THRESHOLD)


def _metadata_candidate_result(
    span: GroundedSpan,
    candidate: Candidate,
) -> Mapping[str, Any] | None:
    calibration = span.metadata.get("grounding_confidence_calibration")
    if not isinstance(calibration, Mapping):
        return None
    candidates = calibration.get("candidates")
    if not isinstance(candidates, Iterable) or isinstance(candidates, (str, bytes)):
        return None
    system_uri = _system_uri(candidate.system)
    for row in candidates:
        if not isinstance(row, Mapping):
            continue
        row_system = str(row.get("system", ""))
        if str(row.get("code", "")) == candidate.code and (
            row_system == candidate.system or row_system == system_uri
        ):
            return row
    return None


def _system_uri(system: str) -> str:
    try:
        return SYSTEM_URI[str(system).upper()]
    except KeyError:
        raise ValueError(f"unknown grounding system {system!r}") from None


def _band_from_values(confidence: float, threshold: float) -> str:
    bounded_confidence = _bounded_probability(confidence, "confidence")
    bounded_threshold = _bounded_probability(threshold, "threshold")
    return ACCEPT_BAND if bounded_confidence >= bounded_threshold else UNCERTAIN_BAND


def _bounded_probability(value: Any, name: str) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return probability


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


review_report = build_review_report


__all__ = [
    "GROUNDING_REVIEW_REPORT_ADVISORY",
    "GROUNDING_REVIEW_REPORT_ARTIFACT",
    "REVIEW_REPORT_SCHEMA_VERSION",
    "GroundingReviewEntry",
    "GroundingReviewReport",
    "build_code_span_review_report",
    "build_grounding_review_report",
    "build_review_report",
    "review_report",
]
