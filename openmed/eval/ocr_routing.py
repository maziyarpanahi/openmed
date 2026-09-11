"""Offline evaluation for OCR-aware clinical document routing.

OCR changes whitespace, punctuation, and individual characters without
changing the document family.  This module evaluates the existing local
document classifier and routing profiles against a small synthetic corpus.  A
standard-library sequence alignment projects section offsets from OCR text
back to canonical text, so the report can score routing and boundary behavior
without storing source text in an artifact.

The public reports contain fixture identifiers, labels, offsets, counts,
confidence values, and domain-separated hashes only.  The default fixtures
are invented strings and are kept in memory; callers supplying their own
fixtures should apply the same synthetic/offline policy before running an
evaluation.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from math import isfinite
from pathlib import Path
from typing import Any

from openmed.clinical.routing import GENERIC_PROFILE_NAME, resolve_profile
from openmed.clinical.sections import (
    UNSECTIONED_SECTION,
    classify_document,
    detect_sections,
)

OCR_ROUTING_SUITE = "ocr_document_routing"
OCR_ROUTING_SCHEMA_VERSION = 1
OCR_ROUTING_FIXTURE_VERSION = "openmed.ocr_document_routing.synthetic.v1"

OCR_DOCUMENT_FAMILIES: tuple[str, ...] = (
    "radiology_report",
    "pathology_report",
    "progress_note",
    "discharge_summary",
    "operative_note",
    "consult_note",
    "unknown",
)
OCR_ROUTING_PROFILES: tuple[str, ...] = ("generic", "pathology", "radiology")

_DOCUMENT_TYPE_ALIASES = {
    "radiology": "radiology_report",
    "radiology_report": "radiology_report",
    "pathology": "pathology_report",
    "pathology_report": "pathology_report",
    "progress": "progress_note",
    "progress_note": "progress_note",
    "discharge": "discharge_summary",
    "discharge_summary": "discharge_summary",
    "operative": "operative_note",
    "operative_note": "operative_note",
    "consult": "consult_note",
    "consult_note": "consult_note",
    "unknown": "unknown",
}
_SPECIALIZED_PROFILES = {
    "radiology_report": "radiology",
    "pathology_report": "pathology",
}


def _normalized_document_type(value: object, *, default: str = "unknown") -> str:
    """Normalize a document-family value without exposing caller data."""

    if not isinstance(value, str) or not value.strip():
        return default
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    return _DOCUMENT_TYPE_ALIASES.get(normalized, normalized)


def _expected_profile(document_type: str) -> str:
    return _SPECIALIZED_PROFILES.get(document_type, GENERIC_PROFILE_NAME)


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a rounded rate, treating an empty comparison as complete."""

    if denominator == 0:
        return 1.0
    return round(numerator / denominator, 6)


def _safe_float(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not isfinite(parsed):
        return default
    return round(min(max(parsed, 0.0), 1.0), 6)


def _classification_value(
    classification: object,
    key: str,
    default: object = None,
) -> object:
    if isinstance(classification, Mapping):
        return classification.get(key, default)
    return getattr(classification, key, default)


def _sha256_digest(*parts: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"openmed.ocr-routing.v1\0")
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class OcrRoutingSection:
    """A canonical, half-open section range used by the OCR eval."""

    label: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("section label must be a non-empty string")
        if (
            not isinstance(self.start, int)
            or isinstance(self.start, bool)
            or not isinstance(self.end, int)
            or isinstance(self.end, bool)
            or self.start < 0
            or self.end <= self.start
        ):
            raise ValueError("section offsets must be a non-empty half-open range")
        object.__setattr__(self, "label", self.label.strip())

    def to_dict(self) -> dict[str, object]:
        """Return the label and offsets without any source substring."""

        return {"label": self.label, "start": self.start, "end": self.end}


ExpectedSection = OcrRoutingSection


def _coerce_section(raw: object, *, field_name: str, index: int) -> OcrRoutingSection:
    """Normalize section-like inputs using only safe structural diagnostics."""

    if isinstance(raw, OcrRoutingSection):
        return raw

    label: object = None
    start: object = None
    end: object = None
    if isinstance(raw, Mapping):
        label = raw.get("label", raw.get("name"))
        start = raw.get("start")
        end = raw.get("end")
        offset = raw.get("offset")
        if (start is None or end is None) and isinstance(offset, (tuple, list)):
            if len(offset) == 2:
                start, end = offset
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        if len(raw) == 3:
            label, start, end = raw
    else:
        label = getattr(raw, "label", getattr(raw, "name", None))
        start = getattr(raw, "start", None)
        end = getattr(raw, "end", None)

    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{field_name} section {index} has no label")
    try:
        return OcrRoutingSection(label, start, end)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} section {index} has invalid offsets") from exc


def _section_signature(section: object) -> tuple[str, int, int]:
    normalized = _coerce_section(section, field_name="section", index=0)
    return normalized.label, normalized.start, normalized.end


@dataclass(frozen=True)
class OcrRoutingFixture:
    """One synthetic canonical/OCR document pair.

    ``canonical_text`` is the reference coordinate space and ``ocr_text`` is
    the text supplied to the local classifier and section detector.  If
    ``gold_sections`` is omitted, the committed local section detector is used
    once on the canonical fixture to create section targets.  The fixture's
    ``to_dict`` method deliberately excludes both text values.

    ``text`` is accepted as a compatibility alias for ``canonical_text`` and
    is populated with the same in-memory value after construction.
    """

    fixture_id: str
    document_family: str
    canonical_text: str | None = None
    ocr_text: str | None = None
    gold_sections: Sequence[object] | None = None
    expected_document_type: str | None = None
    expected_profile: str | None = None
    language: str = "en"
    expect_fallback: bool | None = None
    text: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.fixture_id, str) or not self.fixture_id.strip():
            raise ValueError("fixture_id must be a non-empty string")
        canonical = self.canonical_text
        if canonical is None:
            canonical = self.text
        elif self.text is not None and self.text != canonical:
            raise ValueError("canonical_text and text must agree")
        if not isinstance(canonical, str) or not canonical:
            raise ValueError("canonical_text must be a non-empty string")
        ocr = canonical if self.ocr_text is None else self.ocr_text
        if not isinstance(ocr, str) or not ocr:
            raise ValueError("ocr_text must be a non-empty string")
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("language must be a non-empty string")

        family = _normalized_document_type(self.document_family)
        expected_type = _normalized_document_type(self.expected_document_type or family)
        expected_profile = self.expected_profile or _expected_profile(expected_type)
        if not isinstance(expected_profile, str) or not expected_profile.strip():
            raise ValueError("expected_profile must be a non-empty string")
        expected_profile = expected_profile.strip()

        if self.gold_sections is None:
            detected = detect_sections(canonical, include_unsectioned=False)
            sections = tuple(
                OcrRoutingSection(span.label, span.start, span.end)
                for span in detected
                if span.label != UNSECTIONED_SECTION
            )
        else:
            sections = tuple(
                _coerce_section(
                    section,
                    field_name="gold_sections",
                    index=index,
                )
                for index, section in enumerate(self.gold_sections)
            )
        for index, section in enumerate(sections):
            if section.end > len(canonical):
                raise ValueError(
                    f"gold_sections section {index} exceeds fixture length"
                )

        fallback = (
            expected_profile == GENERIC_PROFILE_NAME
            if self.expect_fallback is None
            else bool(self.expect_fallback)
        )
        object.__setattr__(self, "fixture_id", self.fixture_id.strip())
        object.__setattr__(self, "document_family", family)
        object.__setattr__(self, "canonical_text", canonical)
        object.__setattr__(self, "ocr_text", ocr)
        object.__setattr__(self, "gold_sections", sections)
        object.__setattr__(self, "expected_document_type", expected_type)
        object.__setattr__(self, "expected_profile", expected_profile)
        object.__setattr__(self, "language", self.language.strip())
        object.__setattr__(self, "expect_fallback", fallback)
        object.__setattr__(self, "text", canonical)

    @property
    def canonical_length(self) -> int:
        """Return the canonical document length."""

        return len(self.canonical_text or "")

    @property
    def ocr_length(self) -> int:
        """Return the OCR document length."""

        return len(self.ocr_text or "")

    @property
    def family(self) -> str:
        """Return the normalized document-family label."""

        return self.document_family

    @property
    def fixture_digest(self) -> str:
        """Return a domain-separated digest of the in-memory pair."""

        return _sha256_digest(self.canonical_text or "", self.ocr_text or "")

    def to_dict(self) -> dict[str, object]:
        """Return a privacy-safe fixture manifest without source text."""

        return {
            "fixture_id": self.fixture_id,
            "document_family": self.document_family,
            "expected_document_type": self.expected_document_type,
            "expected_profile": self.expected_profile,
            "expect_fallback": self.expect_fallback,
            "language": self.language,
            "canonical_length": self.canonical_length,
            "ocr_length": self.ocr_length,
            "fixture_digest": self.fixture_digest,
            "gold_sections": [section.to_dict() for section in self.gold_sections],
            "synthetic": True,
        }


OcrRoutingCase = OcrRoutingFixture


@dataclass(frozen=True)
class OffsetProjection:
    """Monotonic OCR-source to canonical-target boundary mapping."""

    source_length: int
    target_length: int
    boundaries: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.source_length < 0 or self.target_length < 0:
            raise ValueError("projection lengths must be non-negative")
        if len(self.boundaries) != self.source_length + 1:
            raise ValueError("projection must include one boundary per source offset")
        previous = -1
        for boundary in self.boundaries:
            if (
                not isinstance(boundary, int)
                or isinstance(boundary, bool)
                or boundary < 0
                or boundary > self.target_length
                or boundary < previous
            ):
                raise ValueError("projection boundaries must be bounded and monotonic")
            previous = boundary

    @property
    def source_to_target(self) -> tuple[int, ...]:
        """Return the boundary map under its explicit direction name."""

        return self.boundaries

    def project_offset(self, offset: int) -> int:
        """Project one source boundary into canonical coordinates."""

        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            or offset > self.source_length
        ):
            raise ValueError("source offset is outside the projection")
        return self.boundaries[offset]

    def project_span(self, start: int, end: int) -> tuple[int, int]:
        """Project a non-empty source span into a canonical half-open range."""

        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end <= start
            or end > self.source_length
        ):
            raise ValueError("source span is outside the projection")
        return self.project_offset(start), self.project_offset(end)

    def project_offsets(self, start: int, end: int) -> tuple[int, int]:
        """Alias for :meth:`project_span`."""

        return self.project_span(start, end)

    def to_dict(self) -> dict[str, object]:
        """Return projection metadata without the full boundary list."""

        return {
            "source_length": self.source_length,
            "target_length": self.target_length,
            "boundary_digest": _sha256_digest(
                json.dumps(self.boundaries, separators=(",", ":"))
            ),
        }


def build_offset_projection(source_text: str, target_text: str) -> OffsetProjection:
    """Build a deterministic character-boundary map from OCR to canonical text.

    ``source_text`` is the OCR coordinate space and ``target_text`` is the
    canonical coordinate space.  Equal runs retain exact offsets. Insertions,
    deletions, and replacements are mapped monotonically across the affected
    target interval, which is sufficient for section-boundary projection and
    keeps the implementation dependency-free and offline.
    """

    if not isinstance(source_text, str) or not isinstance(target_text, str):
        raise TypeError("projection inputs must be strings")

    source_length = len(source_text)
    target_length = len(target_text)
    boundaries: list[int | None] = [None] * (source_length + 1)
    boundaries[0] = 0

    for tag, source_start, source_end, target_start, target_end in SequenceMatcher(
        None,
        source_text,
        target_text,
        autojunk=False,
    ).get_opcodes():
        source_width = source_end - source_start
        target_width = target_end - target_start
        if tag == "equal":
            for index in range(source_width + 1):
                boundaries[source_start + index] = target_start + index
        elif tag == "replace":
            for index in range(source_width + 1):
                projected = target_start + (
                    (index * target_width + source_width // 2) // source_width
                    if source_width
                    else 0
                )
                boundaries[source_start + index] = projected
        elif tag == "delete":
            boundaries[source_start] = target_end
        elif tag == "insert":
            boundaries[source_start] = target_start

    # SequenceMatcher emits contiguous opcodes, but filling defensively keeps
    # the public class safe if the alignment implementation changes later.
    last = 0
    for index, boundary in enumerate(boundaries):
        if boundary is None:
            boundaries[index] = last
        else:
            last = boundary
    last = target_length
    for index in range(len(boundaries) - 1, -1, -1):
        boundary = boundaries[index]
        if boundary is None:
            boundaries[index] = last
        else:
            last = boundary

    monotonic: list[int] = []
    previous = 0
    for boundary in boundaries:
        bounded = min(max(int(boundary or 0), 0), target_length)
        previous = max(previous, bounded)
        monotonic.append(previous)
    monotonic[-1] = target_length
    return OffsetProjection(source_length, target_length, tuple(monotonic))


project_offsets = build_offset_projection
project_span_offsets = build_offset_projection


@dataclass(frozen=True)
class OffsetProjectionScore:
    """Exact and set-based scores for projected section boundaries."""

    predicted_count: int
    gold_count: int
    exact_matches: int
    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "predicted_count": self.predicted_count,
            "gold_count": self.gold_count,
            "exact_matches": self.exact_matches,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def __getitem__(self, key: str) -> int | float:
        return self.to_dict()[key]


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 1.0
    return round(2.0 * precision * recall / (precision + recall), 6)


def score_offset_projection(
    predicted_sections: Iterable[object],
    gold_sections: Iterable[object],
    *,
    projection: OffsetProjection | None = None,
) -> OffsetProjectionScore:
    """Score predicted section ranges after optional OCR→canonical mapping.

    Duplicate ranges are counted with multiset semantics.  This makes a
    duplicated section visible as a precision loss instead of allowing a set
    conversion to hide it.
    """

    predicted: list[OcrRoutingSection] = []
    for index, raw in enumerate(predicted_sections):
        section = _coerce_section(raw, field_name="predicted_sections", index=index)
        if projection is not None:
            start, end = projection.project_span(section.start, section.end)
            section = OcrRoutingSection(section.label, start, end)
        predicted.append(section)
    gold = tuple(
        _coerce_section(raw, field_name="gold_sections", index=index)
        for index, raw in enumerate(gold_sections)
    )

    predicted_counter = Counter(_section_signature(section) for section in predicted)
    gold_counter = Counter(_section_signature(section) for section in gold)
    exact_matches = sum((predicted_counter & gold_counter).values())
    accuracy = _safe_ratio(exact_matches, max(len(predicted), len(gold)))
    precision = _safe_ratio(exact_matches, len(predicted))
    recall = _safe_ratio(exact_matches, len(gold))
    return OffsetProjectionScore(
        predicted_count=len(predicted),
        gold_count=len(gold),
        exact_matches=exact_matches,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
    )


@dataclass(frozen=True)
class OcrRoutingCaseResult:
    """PHI-free outcome for one OCR routing fixture."""

    fixture_id: str
    document_family: str
    expected_document_type: str
    predicted_document_type: str
    expected_profile: str
    predicted_profile: str
    classifier_confidence: float
    expected_fallback: bool
    observed_fallback: bool
    fallback_safe: bool
    route_correct: bool
    profile_correct: bool
    offset_score: OffsetProjectionScore
    gold_sections: tuple[OcrRoutingSection, ...]
    projected_sections: tuple[OcrRoutingSection, ...]
    fixture_digest: str
    classifier_error: str | None = None
    detector_error: str | None = None

    @property
    def offset_projection_correct(self) -> bool:
        """Return whether every predicted/gold section range matched exactly."""

        return self.offset_score.accuracy == 1.0

    @property
    def fallback_correct(self) -> bool:
        """Return whether fallback was observed exactly when expected."""

        return self.expected_fallback == self.observed_fallback

    def to_dict(self) -> dict[str, object]:
        """Return a source-text-free case result."""

        return {
            "fixture_id": self.fixture_id,
            "document_family": self.document_family,
            "expected_document_type": self.expected_document_type,
            "predicted_document_type": self.predicted_document_type,
            "expected_profile": self.expected_profile,
            "predicted_profile": self.predicted_profile,
            "classifier_confidence": self.classifier_confidence,
            "expected_fallback": self.expected_fallback,
            "observed_fallback": self.observed_fallback,
            "fallback_safe": self.fallback_safe,
            "fallback_correct": self.fallback_correct,
            "route_correct": self.route_correct,
            "profile_correct": self.profile_correct,
            "offset_projection_correct": self.offset_projection_correct,
            "offset_score": self.offset_score.to_dict(),
            "gold_sections": [section.to_dict() for section in self.gold_sections],
            "projected_sections": [
                section.to_dict() for section in self.projected_sections
            ],
            "fixture_digest": self.fixture_digest,
            "classifier_error": self.classifier_error,
            "detector_error": self.detector_error,
        }


@dataclass(frozen=True)
class OcrRoutingFailure:
    """Safe diagnostic for one failed route, projection, or fallback check."""

    fixture_id: str
    category: str
    reason: str
    evidence: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "fixture_id": self.fixture_id,
            "category": self.category,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class OcrRoutingMetrics:
    """Aggregate route, offset, and fallback metrics."""

    fixture_count: int
    route_selection_correct: int
    route_selection_accuracy: float
    profile_correct: int
    profile_accuracy: float
    fallback_cases: int
    fallback_observed: int
    fallback_safe: int
    safe_fallback_rate: float
    gold_section_count: int
    projected_section_count: int
    exact_offset_matches: int
    offset_projection_accuracy: float
    offset_projection_precision: float
    offset_projection_recall: float
    offset_projection_f1: float
    failed_case_count: int

    @property
    def route_correct(self) -> int:
        """Compatibility alias for the route-selection count."""

        return self.route_selection_correct

    @property
    def route_accuracy(self) -> float:
        """Compatibility alias for route-selection accuracy."""

        return self.route_selection_accuracy

    @property
    def fallback_safety_rate(self) -> float:
        """Compatibility alias for the safe-fallback rate."""

        return self.safe_fallback_rate

    def to_dict(self) -> dict[str, int | float]:
        return {
            "fixture_count": self.fixture_count,
            "route_selection_correct": self.route_selection_correct,
            "route_selection_accuracy": self.route_selection_accuracy,
            "route_correct": self.route_selection_correct,
            "route_accuracy": self.route_selection_accuracy,
            "profile_correct": self.profile_correct,
            "profile_accuracy": self.profile_accuracy,
            "fallback_cases": self.fallback_cases,
            "fallback_observed": self.fallback_observed,
            "fallback_safe": self.fallback_safe,
            "safe_fallback_rate": self.safe_fallback_rate,
            "fallback_safety_rate": self.safe_fallback_rate,
            "gold_section_count": self.gold_section_count,
            "projected_section_count": self.projected_section_count,
            "exact_offset_matches": self.exact_offset_matches,
            "offset_projection_accuracy": self.offset_projection_accuracy,
            "offset_projection_precision": self.offset_projection_precision,
            "offset_projection_recall": self.offset_projection_recall,
            "offset_projection_f1": self.offset_projection_f1,
            "failed_case_count": self.failed_case_count,
        }

    def __getitem__(self, key: str) -> int | float:
        return self.to_dict()[key]


@dataclass(frozen=True)
class OcrRoutingReport:
    """Deterministic, privacy-safe aggregate OCR routing report."""

    metrics: OcrRoutingMetrics
    cases: tuple[OcrRoutingCaseResult, ...]
    failures: tuple[OcrRoutingFailure, ...]
    passed: bool
    schema_version: int = OCR_ROUTING_SCHEMA_VERSION
    suite: str = OCR_ROUTING_SUITE

    @property
    def case_results(self) -> tuple[OcrRoutingCaseResult, ...]:
        """Return per-fixture outcomes under the explicit result name."""

        return self.cases

    def failure_reasons(self) -> tuple[str, ...]:
        """Return sorted distinct failure categories and reasons."""

        return tuple(
            sorted(
                {f"{failure.category}:{failure.reason}" for failure in self.failures}
            )
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable report with no source text."""

        return {
            "suite": self.suite,
            "schema_version": self.schema_version,
            "metrics": self.metrics.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
            "failures": [failure.to_dict() for failure in self.failures],
            "passed": self.passed,
            "synthetic": True,
            "offline": True,
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report as byte-stable JSON."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write a source-text-free JSON report to ``path``."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render a deterministic summary containing labels and offsets only."""

        metrics = self.metrics.to_dict()
        lines = [
            "# OCR Document-Routing Evaluation",
            "",
            "| Field | Value |",
            "| --- | ---: |",
            f"| Suite | `{self.suite}` |",
            f"| Fixtures | `{metrics['fixture_count']}` |",
            f"| Route accuracy | `{metrics['route_selection_accuracy']:.4f}` |",
            f"| Profile accuracy | `{metrics['profile_accuracy']:.4f}` |",
            f"| Offset projection accuracy | "
            f"`{metrics['offset_projection_accuracy']:.4f}` |",
            f"| Safe fallback rate | `{metrics['safe_fallback_rate']:.4f}` |",
            f"| Verdict | `{'pass' if self.passed else 'fail'}` |",
            "",
            "All fixtures are synthetic and all scoring is local/offline. Reports "
            "contain identifiers, labels, offsets, counts, confidences, and hashes; "
            "they do not contain source text.",
            "",
            "## Cases",
            "",
            "| Fixture | Family | Expected type | Predicted type | Profile | "
            "Fallback | Sections | Offset F1 | Status |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
        for case in self.cases:
            status = (
                "pass"
                if (
                    case.route_correct
                    and case.profile_correct
                    and case.fallback_correct
                    and case.fallback_safe
                    and case.offset_projection_correct
                )
                else "fail"
            )
            lines.append(
                f"| `{case.fixture_id}` | `{case.document_family}` | "
                f"`{case.expected_document_type}` | "
                f"`{case.predicted_document_type}` | `{case.predicted_profile}` | "
                f"`{'yes' if case.observed_fallback else 'no'}` | "
                f"{case.offset_score.exact_matches}/{case.offset_score.gold_count} | "
                f"{case.offset_score.f1:.4f} | `{status}` |"
            )

        lines.extend(["", "## Failures", ""])
        if not self.failures:
            lines.append("No failures.")
        else:
            lines.extend(
                f"- `{failure.fixture_id}` — `{failure.category}`: `{failure.reason}`"
                for failure in self.failures
            )
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write the source-text-free Markdown summary to ``path``."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path


def _default_fixture(
    fixture_id: str,
    document_family: str,
    canonical_text: str,
    ocr_text: str,
) -> OcrRoutingFixture:
    return OcrRoutingFixture(
        fixture_id=fixture_id,
        document_family=document_family,
        canonical_text=canonical_text,
        ocr_text=ocr_text,
    )


def default_ocr_routing_fixtures() -> tuple[OcrRoutingFixture, ...]:
    """Return the committed synthetic OCR corpus for common note families."""

    return (
        _default_fixture(
            "radiology-basic",
            "radiology_report",
            "Radiology Report\n\nTechnique: CT chest without contrast.\n"
            "Findings: Zeta opacity in lung.\n"
            "Impression: Zeta sample finding.",
            "Radiology Report\n\nTechnique: CT  chest without contrast.\n"
            "Findings: Zeta opac1ty in lung.\n"
            "Impression: Zeta sample finding.",
        ),
        _default_fixture(
            "pathology-basic",
            "pathology_report",
            "Pathology Report\n\nSpecimen: Synthetic block.\n"
            "Diagnosis: benign sample.\n"
            "Synoptic: Stage pT1; Grade 1.",
            "Pathology Report\n\nSpecimen: Synthetic  block.\n"
            "Diagnosis: ben1gn sample.\n"
            "Synoptic: Stage pT1; Grade 1.",
        ),
        _default_fixture(
            "progress-basic",
            "progress_note",
            "Progress Note\n\nSubjective: Synthetic report.\n"
            "Objective: Synthetic measure.\n"
            "Assessment and Plan: Stable sample.\n"
            "Medications: Synthetic therapy.",
            "Progress Note\n\nSubjective: Synthetic report.\n"
            "Objective: Synthetic measure.\n"
            "Assessment and Plan: Stable samp1e.\n"
            "Medications: Synthetic therapy.",
        ),
        _default_fixture(
            "discharge-basic",
            "discharge_summary",
            "Discharge Summary\n\nHospital Course: Synthetic course.\n"
            "Impression: Sample summary.\n"
            "Assessment and Plan: Follow-up sample.",
            "Discharge Summary\n\nHospital  Course: Synthetic course.\n"
            "Impression: Sample summary.\n"
            "Assessment and Plan: Follow-up samp1e.",
        ),
        _default_fixture(
            "operative-basic",
            "operative_note",
            "Operative Note\n\nPreoperative Diagnosis: Synthetic indication.\n"
            "Findings: Sample procedure.\n"
            "Assessment and Plan: Synthetic follow-up.",
            "Operative Note\n\nPreoperative Diagnosis: Synthetic indication.\n"
            "Findings: Sample procedure.\n"
            "Assessment and Plan: Synthetic follow-up. ",
        ),
        _default_fixture(
            "consult-basic",
            "consult_note",
            "Consultation Note\n\nHistory of Present Illness: Synthetic question.\n"
            "Assessment: Sample assessment.\n"
            "Plan: Sample follow-up.",
            "Consultation Note\n\nHistory of Present Illness: Synthetic quest1on.\n"
            "Assessment: Sample assessment.\n"
            "Plan: Sample follow-up.",
        ),
        _default_fixture(
            "unknown-fallback",
            "unknown",
            "General Note\n\nAssessment and Plan: Synthetic content.",
            "General N0te\n\nAssessment and Plan: Synthetic content.",
        ),
    )


load_ocr_routing_fixtures = default_ocr_routing_fixtures


def _fallback_is_safe(profile: object, detected_sections: Sequence[object]) -> bool:
    """Check that generic fallback preserves sections and offset-bearing rows."""

    profile_name = getattr(profile, "name", None)
    if profile_name != GENERIC_PROFILE_NAME:
        return False
    try:
        raw_sections = [
            section.to_dict() if isinstance(section, OcrRoutingSection) else section
            for section in detected_sections
        ]
        scoped_sections = profile.scope_sections(raw_sections)
        original_signatures = tuple(
            _section_signature(section) for section in detected_sections
        )
        scoped_signatures = tuple(
            _section_signature(section) for section in scoped_sections
        )
        if scoped_signatures != original_signatures:
            return False
        probe_entities = [
            {
                "start": section.start,
                "end": min(section.end, section.start + 1),
                "label": "SYNTHETIC",
            }
            for section in (
                _coerce_section(raw, field_name="detected_sections", index=index)
                for index, raw in enumerate(raw_sections)
            )
        ]
        return len(profile.scope_entities(probe_entities, raw_sections)) == len(
            probe_entities
        )
    except (TypeError, ValueError, AttributeError):
        return False


def _normalize_detector_sections(value: object) -> tuple[OcrRoutingSection, ...]:
    if value is None:
        return ()
    try:
        return tuple(
            _coerce_section(raw, field_name="detected_sections", index=index)
            for index, raw in enumerate(value)  # type: ignore[arg-type]
        )
    except TypeError as exc:
        raise ValueError("section detector did not return an iterable") from exc


def _run_section_detector(
    detector: Callable[..., object],
    text: str,
    language: str,
) -> object:
    """Call a detector with the public language hook or a one-argument seam."""

    try:
        return detector(text, language=language)
    except TypeError as first_error:
        try:
            return detector(text)
        except TypeError:
            raise first_error


def _case_failures(case: OcrRoutingCaseResult) -> list[OcrRoutingFailure]:
    failures: list[OcrRoutingFailure] = []
    if case.classifier_error is not None:
        failures.append(
            OcrRoutingFailure(
                case.fixture_id,
                "route_selection",
                "classifier_error",
                {"error_type": case.classifier_error},
            )
        )
    if not case.route_correct:
        failures.append(
            OcrRoutingFailure(
                case.fixture_id,
                "route_selection",
                "document_type_mismatch",
                {
                    "expected": case.expected_document_type,
                    "predicted": case.predicted_document_type,
                },
            )
        )
    if not case.profile_correct:
        failures.append(
            OcrRoutingFailure(
                case.fixture_id,
                "route_selection",
                "profile_mismatch",
                {
                    "expected": case.expected_profile,
                    "predicted": case.predicted_profile,
                },
            )
        )
    if not case.fallback_correct:
        failures.append(
            OcrRoutingFailure(
                case.fixture_id,
                "safe_fallback",
                "fallback_decision_mismatch",
                {
                    "expected": case.expected_fallback,
                    "observed": case.observed_fallback,
                },
            )
        )
    if case.expected_fallback and not case.fallback_safe:
        failures.append(
            OcrRoutingFailure(
                case.fixture_id,
                "safe_fallback",
                "generic_profile_did_not_pass_through",
                {"predicted_profile": case.predicted_profile},
            )
        )
    if case.detector_error is not None:
        failures.append(
            OcrRoutingFailure(
                case.fixture_id,
                "offset_projection",
                "section_detector_error",
                {"error_type": case.detector_error},
            )
        )
    if not case.offset_projection_correct:
        failures.append(
            OcrRoutingFailure(
                case.fixture_id,
                "offset_projection",
                "section_offsets_mismatch",
                {
                    "expected_sections": [
                        section.to_dict() for section in case.gold_sections
                    ],
                    "projected_sections": [
                        section.to_dict() for section in case.projected_sections
                    ],
                },
            )
        )
    return failures


def run_ocr_routing_eval(
    fixtures: Iterable[OcrRoutingFixture] | None = None,
    *,
    classifier: Callable[[str], object] | None = None,
    section_detector: Callable[..., object] | None = None,
    min_route_accuracy: float = 1.0,
    min_offset_projection_accuracy: float = 1.0,
    min_safe_fallback_rate: float = 1.0,
) -> OcrRoutingReport:
    """Run the deterministic OCR routing evaluation locally.

    Args:
        fixtures: Synthetic fixture pairs. Defaults to the committed corpus.
        classifier: Optional local classifier seam. The default is
            :func:`openmed.clinical.sections.classify_document`.
        section_detector: Optional local section detector seam. The default is
            :func:`openmed.clinical.sections.detect_sections`.
        min_route_accuracy: Required document-type route accuracy.
        min_offset_projection_accuracy: Required exact projected-section rate.
        min_safe_fallback_rate: Required safe pass-through rate for expected
            generic fallbacks.

    Returns:
        A deterministic report whose artifacts never include fixture source
        text. Classifier and detector exceptions are represented by exception
        type names so an accidental raw-text exception message cannot enter a
        report.
    """

    thresholds = (
        min_route_accuracy,
        min_offset_projection_accuracy,
        min_safe_fallback_rate,
    )
    if any(
        isinstance(value, bool)
        or not isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for value in thresholds
    ):
        raise ValueError("evaluation thresholds must be between 0 and 1")

    active_fixtures = tuple(
        default_ocr_routing_fixtures() if fixtures is None else fixtures
    )
    if not active_fixtures:
        raise ValueError("OCR routing evaluation requires at least one fixture")
    if any(not isinstance(fixture, OcrRoutingFixture) for fixture in active_fixtures):
        raise TypeError("fixtures must contain OcrRoutingFixture values")
    fixture_ids = [fixture.fixture_id for fixture in active_fixtures]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("fixture_id values must be unique")

    classify = classifier or classify_document
    detect = section_detector or detect_sections
    cases: list[OcrRoutingCaseResult] = []

    for fixture in active_fixtures:
        classification_error: str | None = None
        detector_error: str | None = None
        try:
            classification = classify(fixture.ocr_text or "")
        except Exception as exc:  # pragma: no cover - exercised by callers
            classification = {"type": "unknown", "confidence": 0.0}
            classification_error = type(exc).__name__

        predicted_document_type = _normalized_document_type(
            _classification_value(classification, "type", classification)
        )
        confidence = _safe_float(
            _classification_value(classification, "confidence", 0.0)
        )
        selection = resolve_profile(classification)
        predicted_profile = selection.profile_name
        observed_fallback = selection.provenance.fallback_reason is not None

        try:
            detected = _normalize_detector_sections(
                _run_section_detector(detect, fixture.ocr_text or "", fixture.language)
            )
            projection = build_offset_projection(
                fixture.ocr_text or "", fixture.canonical_text or ""
            )
            predicted_sections = tuple(
                section for section in detected if section.label != UNSECTIONED_SECTION
            )
            projected_sections = tuple(
                OcrRoutingSection(
                    section.label,
                    *projection.project_span(section.start, section.end),
                )
                for section in predicted_sections
            )
            offset_score = score_offset_projection(
                predicted_sections,
                fixture.gold_sections or (),
                projection=projection,
            )
        except Exception as exc:  # pragma: no cover - exercised by callers
            detector_error = type(exc).__name__
            projected_sections = ()
            offset_score = score_offset_projection((), fixture.gold_sections or ())
            detected = ()

        fallback_safe = (
            _fallback_is_safe(selection.profile, detected)
            if observed_fallback
            else not fixture.expect_fallback
        )
        case = OcrRoutingCaseResult(
            fixture_id=fixture.fixture_id,
            document_family=fixture.document_family,
            expected_document_type=fixture.expected_document_type or "unknown",
            predicted_document_type=predicted_document_type,
            expected_profile=fixture.expected_profile or GENERIC_PROFILE_NAME,
            predicted_profile=predicted_profile,
            classifier_confidence=confidence,
            expected_fallback=bool(fixture.expect_fallback),
            observed_fallback=observed_fallback,
            fallback_safe=fallback_safe,
            route_correct=(
                predicted_document_type == (fixture.expected_document_type or "unknown")
            ),
            profile_correct=(
                predicted_profile == (fixture.expected_profile or GENERIC_PROFILE_NAME)
            ),
            offset_score=offset_score,
            gold_sections=tuple(fixture.gold_sections or ()),
            projected_sections=projected_sections,
            fixture_digest=fixture.fixture_digest,
            classifier_error=classification_error,
            detector_error=detector_error,
        )
        cases.append(case)

    failures = tuple(failure for case in cases for failure in _case_failures(case))
    route_correct = sum(case.route_correct for case in cases)
    profile_correct = sum(case.profile_correct for case in cases)
    fallback_cases = sum(case.expected_fallback for case in cases)
    fallback_observed = sum(case.observed_fallback for case in cases)
    fallback_safe = sum(case.fallback_safe for case in cases if case.expected_fallback)
    gold_section_count = sum(case.offset_score.gold_count for case in cases)
    projected_section_count = sum(case.offset_score.predicted_count for case in cases)
    exact_offset_matches = sum(case.offset_score.exact_matches for case in cases)
    offset_precision = _safe_ratio(exact_offset_matches, projected_section_count)
    offset_recall = _safe_ratio(exact_offset_matches, gold_section_count)
    metrics = OcrRoutingMetrics(
        fixture_count=len(cases),
        route_selection_correct=route_correct,
        route_selection_accuracy=_safe_ratio(route_correct, len(cases)),
        profile_correct=profile_correct,
        profile_accuracy=_safe_ratio(profile_correct, len(cases)),
        fallback_cases=fallback_cases,
        fallback_observed=fallback_observed,
        fallback_safe=fallback_safe,
        safe_fallback_rate=_safe_ratio(fallback_safe, fallback_cases),
        gold_section_count=gold_section_count,
        projected_section_count=projected_section_count,
        exact_offset_matches=exact_offset_matches,
        offset_projection_accuracy=_safe_ratio(
            exact_offset_matches,
            max(gold_section_count, projected_section_count),
        ),
        offset_projection_precision=offset_precision,
        offset_projection_recall=offset_recall,
        offset_projection_f1=_f1(offset_precision, offset_recall),
        failed_case_count=len({failure.fixture_id for failure in failures}),
    )
    passed = (
        bool(cases)
        and metrics.route_selection_accuracy >= float(min_route_accuracy)
        and metrics.offset_projection_accuracy >= float(min_offset_projection_accuracy)
        and metrics.safe_fallback_rate >= float(min_safe_fallback_rate)
        and not failures
    )
    return OcrRoutingReport(
        metrics=metrics, cases=tuple(cases), failures=failures, passed=passed
    )


run_ocr_routing = run_ocr_routing_eval
score_ocr_routing = run_ocr_routing_eval


def assert_ocr_routing_gate(
    fixtures: Iterable[OcrRoutingFixture] | None = None,
    **kwargs: object,
) -> OcrRoutingReport:
    """Return a passing report or raise with only safe fixture diagnostics."""

    report = run_ocr_routing_eval(fixtures, **kwargs)
    if not report.passed:
        details = ", ".join(
            f"{failure.fixture_id}:{failure.category}:{failure.reason}"
            for failure in report.failures
        )
        raise AssertionError(f"OCR routing gate failed: {details or 'threshold'}")
    return report


def ocr_routing_metadata() -> dict[str, object]:
    """Return metadata suitable for an offline eval manifest."""

    return {
        "suite": OCR_ROUTING_SUITE,
        "schema_version": OCR_ROUTING_SCHEMA_VERSION,
        "fixture_version": OCR_ROUTING_FIXTURE_VERSION,
        "fixture_count": len(default_ocr_routing_fixtures()),
        "synthetic": True,
        "offline": True,
        "mandatory_network": False,
        "reports_include_source_text": False,
    }


__all__ = [
    "OCR_DOCUMENT_FAMILIES",
    "OCR_ROUTING_FIXTURE_VERSION",
    "OCR_ROUTING_PROFILES",
    "OCR_ROUTING_SCHEMA_VERSION",
    "OCR_ROUTING_SUITE",
    "ExpectedSection",
    "OffsetProjection",
    "OffsetProjectionScore",
    "OcrRoutingCase",
    "OcrRoutingCaseResult",
    "OcrRoutingFailure",
    "OcrRoutingFixture",
    "OcrRoutingMetrics",
    "OcrRoutingReport",
    "build_offset_projection",
    "default_ocr_routing_fixtures",
    "assert_ocr_routing_gate",
    "load_ocr_routing_fixtures",
    "ocr_routing_metadata",
    "project_offsets",
    "project_span_offsets",
    "run_ocr_routing",
    "run_ocr_routing_eval",
    "score_ocr_routing",
    "score_offset_projection",
]
