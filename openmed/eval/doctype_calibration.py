"""Deterministic, counts-only document-type calibration reports.

The report boundary accepts only canonical document-type labels and numeric
confidence scores. Source text, fixture identifiers, model names, and
arbitrary metadata are never copied into the serialized artifact.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.clinical.sections.doctype import (
    DOCUMENT_TYPES,
    UNKNOWN_DOCUMENT_TYPE,
)
from openmed.core.audit import stable_hash

DOCTYPE_CALIBRATION_ARTIFACT = "openmed.eval.doctype_calibration"
DOCTYPE_CALIBRATION_SCHEMA_VERSION = 1
DEFAULT_CALIBRATION_BINS = 10
DEFAULT_ABSTENTION_THRESHOLDS = (0.0, 0.5, 0.7, 0.9)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_DESCRIPTOR_DEPTH = 8
_MAX_DESCRIPTOR_ITEMS = 256
_MAX_DESCRIPTOR_TEXT = 4096
_MAX_SAMPLES = 1_000_000
_MAX_BINS = 100
_MAX_THRESHOLDS = 100
_MAX_JSON_INDENT = 8
_EXPECTED_TYPES = frozenset(DOCUMENT_TYPES)
_PREDICTED_TYPES = _EXPECTED_TYPES | {UNKNOWN_DOCUMENT_TYPE}


@dataclass(frozen=True)
class DocumentTypeCalibrationSample:
    """One synthetic document-type prediction used only for aggregation.

    Args:
        expected_type: Canonical synthetic gold document type.
        predicted_type: Canonical predicted type or ``unknown``.
        confidence: Finite classifier confidence in the inclusive range
            ``[0, 1]``.
    """

    expected_type: str
    predicted_type: str
    confidence: float

    def __post_init__(self) -> None:
        if (
            type(self.expected_type) is not str
            or self.expected_type not in _EXPECTED_TYPES
        ):
            raise ValueError("expected_type must be a canonical document type")
        if (
            type(self.predicted_type) is not str
            or self.predicted_type not in _PREDICTED_TYPES
        ):
            raise ValueError("predicted_type must be a canonical document type")
        confidence = _confidence(self.confidence)
        object.__setattr__(self, "confidence", confidence)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DocumentTypeCalibrationSample:
        """Build a sample from an allowlisted plain-dictionary projection.

        Extra keys are ignored so note text, fixture identifiers, and arbitrary
        metadata cannot cross the report boundary.
        """

        if type(value) is not dict:
            raise TypeError("calibration samples must be plain dictionaries")
        return cls(
            expected_type=value.get("expected_type"),
            predicted_type=value.get("predicted_type"),
            confidence=value.get("confidence"),
        )

    def to_fingerprint_dict(self) -> dict[str, str | float]:
        """Return the bounded prediction fields used by the fixture digest."""

        return {
            "confidence": self.confidence,
            "expected_type": self.expected_type,
            "predicted_type": self.predicted_type,
        }


@dataclass(frozen=True)
class DocumentTypeCalibrationBin:
    """Aggregate reliability values for one confidence interval."""

    lower_bound: float
    upper_bound: float
    sample_count: int
    correct_count: int
    mean_confidence: float | None
    accuracy: float | None
    absolute_gap: float | None

    def __post_init__(self) -> None:
        _bounded_rate(self.lower_bound, field_name="bin lower bound")
        _bounded_rate(self.upper_bound, field_name="bin upper bound")
        if self.lower_bound >= self.upper_bound:
            raise ValueError("calibration bin bounds must be increasing")
        _aggregate_count(self.sample_count, field_name="bin sample count")
        _aggregate_count(self.correct_count, field_name="bin correct count")
        if self.correct_count > self.sample_count:
            raise ValueError("bin correct count must not exceed sample count")
        if self.sample_count == 0:
            if any(
                value is not None
                for value in (self.mean_confidence, self.accuracy, self.absolute_gap)
            ):
                raise ValueError("empty calibration bins must use null rates")
        else:
            _bounded_rate(self.mean_confidence, field_name="bin mean confidence")
            _bounded_rate(self.accuracy, field_name="bin accuracy")
            _bounded_rate(self.absolute_gap, field_name="bin absolute gap")

    def to_dict(self) -> dict[str, int | float | None]:
        """Return a deterministic JSON-ready aggregate."""

        return {
            "absolute_gap": self.absolute_gap,
            "accuracy": self.accuracy,
            "correct_count": self.correct_count,
            "lower_bound": self.lower_bound,
            "mean_confidence": self.mean_confidence,
            "sample_count": self.sample_count,
            "upper_bound": self.upper_bound,
        }


@dataclass(frozen=True)
class DocumentTypeAbstentionMetrics:
    """Aggregate accuracy and abstention at one confidence threshold."""

    threshold: float
    abstained_count: int
    abstention_rate: float
    retained_count: int
    retained_correct_count: int
    retained_accuracy: float | None

    def __post_init__(self) -> None:
        _bounded_rate(self.threshold, field_name="abstention threshold")
        _aggregate_count(self.abstained_count, field_name="abstained count")
        _bounded_rate(self.abstention_rate, field_name="abstention rate")
        _aggregate_count(self.retained_count, field_name="retained count")
        _aggregate_count(
            self.retained_correct_count,
            field_name="retained correct count",
        )
        if self.retained_correct_count > self.retained_count:
            raise ValueError("retained correct count must not exceed retained count")
        if self.retained_count:
            _bounded_rate(self.retained_accuracy, field_name="retained accuracy")
        elif self.retained_accuracy is not None:
            raise ValueError("empty retained sets must use a null accuracy")

    def to_dict(self) -> dict[str, int | float | None]:
        """Return a deterministic JSON-ready aggregate."""

        return {
            "abstained_count": self.abstained_count,
            "abstention_rate": self.abstention_rate,
            "retained_accuracy": self.retained_accuracy,
            "retained_correct_count": self.retained_correct_count,
            "retained_count": self.retained_count,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class DocumentTypeSupport:
    """Gold support and prediction counts for one canonical document type."""

    document_type: str
    support: int
    predicted_count: int
    correct_count: int

    def __post_init__(self) -> None:
        if type(self.document_type) is not str or self.document_type not in (
            _EXPECTED_TYPES
        ):
            raise ValueError("support rows require a canonical document type")
        _aggregate_count(self.support, field_name="type support")
        _aggregate_count(self.predicted_count, field_name="type prediction count")
        _aggregate_count(self.correct_count, field_name="type correct count")
        if self.correct_count > min(self.support, self.predicted_count):
            raise ValueError("type correct count exceeds its support")

    def to_dict(self) -> dict[str, str | int]:
        """Return a deterministic JSON-ready aggregate."""

        return {
            "correct_count": self.correct_count,
            "document_type": self.document_type,
            "predicted_count": self.predicted_count,
            "support": self.support,
        }


@dataclass(frozen=True)
class DocumentTypeCalibrationReport:
    """Counts-only document-type confidence calibration evidence."""

    model_fingerprint: str
    fixture_fingerprint: str
    sample_count: int
    correct_count: int
    unknown_prediction_count: int
    accuracy: float
    expected_calibration_error: float
    bins: tuple[DocumentTypeCalibrationBin, ...]
    abstention: tuple[DocumentTypeAbstentionMetrics, ...]
    per_type_support: tuple[DocumentTypeSupport, ...]

    def __post_init__(self) -> None:
        if type(self.model_fingerprint) is not str or not _DIGEST_RE.fullmatch(
            self.model_fingerprint
        ):
            raise ValueError("model_fingerprint must be a SHA-256 fingerprint")
        if type(self.fixture_fingerprint) is not str or not _DIGEST_RE.fullmatch(
            self.fixture_fingerprint
        ):
            raise ValueError("fixture_fingerprint must be a SHA-256 fingerprint")
        _aggregate_count(self.sample_count, field_name="sample count")
        if self.sample_count == 0:
            raise ValueError("calibration reports require at least one sample")
        _aggregate_count(self.correct_count, field_name="correct count")
        _aggregate_count(
            self.unknown_prediction_count,
            field_name="unknown prediction count",
        )
        if self.correct_count > self.sample_count:
            raise ValueError("correct count must not exceed sample count")
        if self.unknown_prediction_count > self.sample_count:
            raise ValueError("unknown prediction count must not exceed sample count")
        _bounded_rate(self.accuracy, field_name="accuracy")
        _bounded_rate(
            self.expected_calibration_error,
            field_name="expected calibration error",
        )
        if type(self.bins) is not tuple or not self.bins:
            raise TypeError("bins must be a non-empty tuple")
        if len(self.bins) > _MAX_BINS or not all(
            type(row) is DocumentTypeCalibrationBin for row in self.bins
        ):
            raise ValueError("bins contain unsupported calibration rows")
        if type(self.abstention) is not tuple or not self.abstention:
            raise TypeError("abstention must be a non-empty tuple")
        if len(self.abstention) > _MAX_THRESHOLDS or not all(
            type(row) is DocumentTypeAbstentionMetrics for row in self.abstention
        ):
            raise ValueError("abstention contains unsupported rows")
        if type(self.per_type_support) is not tuple or not all(
            type(row) is DocumentTypeSupport for row in self.per_type_support
        ):
            raise TypeError("per_type_support must contain support rows")
        if tuple(row.document_type for row in self.per_type_support) != tuple(
            sorted(DOCUMENT_TYPES)
        ):
            raise ValueError("per_type_support must cover each canonical type")
        self._validate_aggregates()

    def _validate_aggregates(self) -> None:
        if sum(row.sample_count for row in self.bins) != self.sample_count:
            raise ValueError("calibration bin counts do not match sample count")
        if sum(row.correct_count for row in self.bins) != self.correct_count:
            raise ValueError("calibration bin counts do not match correct count")
        if self.accuracy != _ratio(self.correct_count, self.sample_count):
            raise ValueError("accuracy does not match aggregate counts")
        if sum(row.support for row in self.per_type_support) != self.sample_count:
            raise ValueError("per-type support does not match sample count")
        if (
            sum(row.correct_count for row in self.per_type_support)
            != self.correct_count
        ):
            raise ValueError("per-type support does not match correct count")
        predicted_count = sum(row.predicted_count for row in self.per_type_support)
        if predicted_count + self.unknown_prediction_count != self.sample_count:
            raise ValueError("prediction counts do not match sample count")
        thresholds = tuple(row.threshold for row in self.abstention)
        if thresholds != tuple(sorted(set(thresholds))):
            raise ValueError("abstention thresholds must be sorted and unique")
        for row in self.abstention:
            if row.abstained_count + row.retained_count != self.sample_count:
                raise ValueError("abstention counts do not match sample count")
            if row.abstention_rate != _ratio(row.abstained_count, self.sample_count):
                raise ValueError("abstention rate does not match aggregate counts")

    def to_dict(self) -> dict[str, Any]:
        """Return the allowlisted counts-only report payload."""

        return {
            "abstention": [row.to_dict() for row in self.abstention],
            "accuracy": self.accuracy,
            "artifact_type": DOCTYPE_CALIBRATION_ARTIFACT,
            "bins": [row.to_dict() for row in self.bins],
            "correct_count": self.correct_count,
            "expected_calibration_error": self.expected_calibration_error,
            "fixture_fingerprint": self.fixture_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "num_bins": len(self.bins),
            "per_type_support": [row.to_dict() for row in self.per_type_support],
            "sample_count": self.sample_count,
            "schema_version": DOCTYPE_CALIBRATION_SCHEMA_VERSION,
            "unknown_prediction_count": self.unknown_prediction_count,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to deterministic JSON."""

        if (
            type(indent) is not int
            or type(indent) is bool
            or not (0 <= indent <= _MAX_JSON_INDENT)
        ):
            raise ValueError("JSON indent must be an integer from 0 to 8")
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render the report as deterministic counts-only Markdown."""

        lines = [
            "# Document-type calibration report",
            "",
            "## Provenance",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Model fingerprint | `{self.model_fingerprint}` |",
            f"| Fixture fingerprint | `{self.fixture_fingerprint}` |",
            f"| Schema version | {DOCTYPE_CALIBRATION_SCHEMA_VERSION} |",
            "",
            "## Summary",
            "",
            "| Samples | Correct | Unknown predictions | Accuracy | ECE |",
            "|---:|---:|---:|---:|---:|",
            (
                f"| {self.sample_count} | {self.correct_count} | "
                f"{self.unknown_prediction_count} | {_format_rate(self.accuracy)} | "
                f"{_format_rate(self.expected_calibration_error)} |"
            ),
            "",
            "## Calibration bins",
            "",
            "| Lower | Upper | Samples | Correct | Mean confidence | Accuracy | Gap |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in self.bins:
            lines.append(
                f"| {_format_rate(row.lower_bound)} | "
                f"{_format_rate(row.upper_bound)} | {row.sample_count} | "
                f"{row.correct_count} | {_format_rate(row.mean_confidence)} | "
                f"{_format_rate(row.accuracy)} | "
                f"{_format_rate(row.absolute_gap)} |"
            )
        lines.extend(
            [
                "",
                "## Abstention",
                "",
                (
                    "| Threshold | Abstained | Abstention rate | Retained | "
                    "Retained correct | Retained accuracy |"
                ),
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in self.abstention:
            lines.append(
                f"| {_format_rate(row.threshold)} | {row.abstained_count} | "
                f"{_format_rate(row.abstention_rate)} | {row.retained_count} | "
                f"{row.retained_correct_count} | "
                f"{_format_rate(row.retained_accuracy)} |"
            )
        lines.extend(
            [
                "",
                "## Per-type support",
                "",
                "| Document type | Gold support | Predicted | Correct |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in self.per_type_support:
            lines.append(
                f"| `{row.document_type}` | {row.support} | "
                f"{row.predicted_count} | {row.correct_count} |"
            )
        return "\n".join(lines) + "\n"

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON without exposing a failing path."""

        return _write_report(path, self.to_json(indent=indent) + "\n")

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown without exposing a failing path."""

        return _write_report(path, self.to_markdown())


def fingerprint_doctype_model(model: Any) -> str:
    """Return a stable opaque fingerprint for bounded model metadata.

    A precomputed ``sha256:`` fingerprint is accepted unchanged. All other
    descriptors are normalized into bounded JSON-compatible data and hashed in
    memory; descriptor values are never included in a report or exception.
    """

    if type(model) is str and _DIGEST_RE.fullmatch(model):
        return model
    normalized = _normalize_descriptor(model)
    return stable_hash(
        {
            "artifact_type": DOCTYPE_CALIBRATION_ARTIFACT,
            "model": normalized,
            "schema_version": DOCTYPE_CALIBRATION_SCHEMA_VERSION,
        }
    )


def fingerprint_doctype_fixtures(
    samples: Iterable[DocumentTypeCalibrationSample | Mapping[str, Any]],
) -> str:
    """Return an order-independent fingerprint of normalized scored labels."""

    normalized = _normalize_samples(samples)
    return _fixture_fingerprint(normalized)


def build_doctype_calibration_report(
    samples: Iterable[DocumentTypeCalibrationSample | Mapping[str, Any]],
    *,
    model: Any,
    num_bins: int = DEFAULT_CALIBRATION_BINS,
    abstention_thresholds: Iterable[float] = DEFAULT_ABSTENTION_THRESHOLDS,
) -> DocumentTypeCalibrationReport:
    """Compute deterministic calibration and abstention evidence.

    ``unknown`` predictions always count as abstentions. At each requested
    threshold, predictions with confidence strictly below that threshold also
    abstain. The returned object contains only canonical type names, aggregate
    counts/rates, and opaque model/fixture fingerprints.

    Args:
        samples: Synthetic gold/prediction/confidence records. Plain dictionary
            inputs are projected onto the three required keys; all extra keys
            are ignored.
        model: Bounded JSON-compatible model metadata or a precomputed SHA-256
            fingerprint. The descriptor itself is never rendered.
        num_bins: Number of equal-width confidence bins in ``[0, 1]``.
        abstention_thresholds: Confidence thresholds to evaluate. Input order
            and duplicates do not affect the report.

    Returns:
        A deterministic counts-only calibration report.
    """

    normalized = _normalize_samples(samples)
    bins = _calibration_bins(normalized, _num_bins(num_bins))
    correct_count = sum(_is_correct(sample) for sample in normalized)
    sample_count = len(normalized)
    ece = sum(
        (row.sample_count / sample_count) * (row.absolute_gap or 0.0) for row in bins
    )
    return DocumentTypeCalibrationReport(
        model_fingerprint=fingerprint_doctype_model(model),
        fixture_fingerprint=_fixture_fingerprint(normalized),
        sample_count=sample_count,
        correct_count=correct_count,
        unknown_prediction_count=sum(
            sample.predicted_type == UNKNOWN_DOCUMENT_TYPE for sample in normalized
        ),
        accuracy=_ratio(correct_count, sample_count),
        expected_calibration_error=_rounded(ece),
        bins=bins,
        abstention=_abstention_metrics(
            normalized,
            _thresholds(abstention_thresholds),
        ),
        per_type_support=_per_type_support(normalized),
    )


def render_doctype_calibration_json(
    report: DocumentTypeCalibrationReport,
    *,
    indent: int = 2,
) -> str:
    """Render a document-type calibration report as deterministic JSON."""

    return _report(report).to_json(indent=indent)


def render_doctype_calibration_markdown(
    report: DocumentTypeCalibrationReport,
) -> str:
    """Render a document-type calibration report as deterministic Markdown."""

    return _report(report).to_markdown()


def _normalize_samples(
    samples: Iterable[DocumentTypeCalibrationSample | Mapping[str, Any]],
) -> tuple[DocumentTypeCalibrationSample, ...]:
    try:
        iterator = iter(samples)
    except TypeError:
        raise TypeError("samples must be an iterable of calibration records") from None
    except Exception:
        raise RuntimeError("failed to read calibration samples") from None

    normalized: list[DocumentTypeCalibrationSample] = []
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            break
        except Exception:
            raise RuntimeError("failed to read calibration samples") from None
        if len(normalized) >= _MAX_SAMPLES:
            raise ValueError("calibration sample count exceeds the supported limit")
        if type(item) is DocumentTypeCalibrationSample:
            normalized.append(item)
        elif type(item) is dict:
            normalized.append(DocumentTypeCalibrationSample.from_mapping(item))
        else:
            raise TypeError("calibration records must use the supported sample schema")
    if not normalized:
        raise ValueError("at least one calibration sample is required")
    normalized.sort(
        key=lambda sample: (
            sample.expected_type,
            sample.predicted_type,
            sample.confidence,
        )
    )
    return tuple(normalized)


def _fixture_fingerprint(
    samples: tuple[DocumentTypeCalibrationSample, ...],
) -> str:
    records = [sample.to_fingerprint_dict() for sample in samples]
    records.sort(key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))
    return stable_hash(
        {
            "artifact_type": DOCTYPE_CALIBRATION_ARTIFACT,
            "document_types": sorted(DOCUMENT_TYPES),
            "samples": records,
            "schema_version": DOCTYPE_CALIBRATION_SCHEMA_VERSION,
        }
    )


def _calibration_bins(
    samples: tuple[DocumentTypeCalibrationSample, ...],
    num_bins: int,
) -> tuple[DocumentTypeCalibrationBin, ...]:
    confidence_sums = [0.0] * num_bins
    sample_counts = [0] * num_bins
    correct_counts = [0] * num_bins
    for sample in samples:
        index = min(int(sample.confidence * num_bins), num_bins - 1)
        confidence_sums[index] += sample.confidence
        sample_counts[index] += 1
        correct_counts[index] += _is_correct(sample)

    rows: list[DocumentTypeCalibrationBin] = []
    for index in range(num_bins):
        count = sample_counts[index]
        mean_confidence = _rounded(confidence_sums[index] / count) if count else None
        accuracy = _ratio(correct_counts[index], count) if count else None
        gap = (
            _rounded(abs(accuracy - mean_confidence))
            if accuracy is not None and mean_confidence is not None
            else None
        )
        rows.append(
            DocumentTypeCalibrationBin(
                lower_bound=_rounded(index / num_bins),
                upper_bound=_rounded((index + 1) / num_bins),
                sample_count=count,
                correct_count=correct_counts[index],
                mean_confidence=mean_confidence,
                accuracy=accuracy,
                absolute_gap=gap,
            )
        )
    return tuple(rows)


def _abstention_metrics(
    samples: tuple[DocumentTypeCalibrationSample, ...],
    thresholds: tuple[float, ...],
) -> tuple[DocumentTypeAbstentionMetrics, ...]:
    rows: list[DocumentTypeAbstentionMetrics] = []
    for threshold in thresholds:
        retained = [
            sample
            for sample in samples
            if sample.predicted_type != UNKNOWN_DOCUMENT_TYPE
            and sample.confidence >= threshold
        ]
        retained_correct = sum(_is_correct(sample) for sample in retained)
        abstained_count = len(samples) - len(retained)
        rows.append(
            DocumentTypeAbstentionMetrics(
                threshold=threshold,
                abstained_count=abstained_count,
                abstention_rate=_ratio(abstained_count, len(samples)),
                retained_count=len(retained),
                retained_correct_count=retained_correct,
                retained_accuracy=(
                    _ratio(retained_correct, len(retained)) if retained else None
                ),
            )
        )
    return tuple(rows)


def _per_type_support(
    samples: tuple[DocumentTypeCalibrationSample, ...],
) -> tuple[DocumentTypeSupport, ...]:
    return tuple(
        DocumentTypeSupport(
            document_type=document_type,
            support=sum(sample.expected_type == document_type for sample in samples),
            predicted_count=sum(
                sample.predicted_type == document_type for sample in samples
            ),
            correct_count=sum(
                sample.expected_type == document_type
                and sample.predicted_type == document_type
                for sample in samples
            ),
        )
        for document_type in sorted(DOCUMENT_TYPES)
    )


def _thresholds(values: Iterable[float]) -> tuple[float, ...]:
    try:
        iterator = iter(values)
    except TypeError:
        raise TypeError("abstention_thresholds must be an iterable") from None
    except Exception:
        raise RuntimeError("failed to read abstention thresholds") from None
    normalized: list[float] = []
    while True:
        try:
            value = next(iterator)
        except StopIteration:
            break
        except Exception:
            raise RuntimeError("failed to read abstention thresholds") from None
        if len(normalized) >= _MAX_THRESHOLDS:
            raise ValueError("abstention threshold count exceeds the supported limit")
        normalized.append(_confidence(value))
    if not normalized:
        raise ValueError("at least one abstention threshold is required")
    return tuple(sorted(set(normalized)))


def _num_bins(value: Any) -> int:
    if type(value) is not int or type(value) is bool or not (1 <= value <= _MAX_BINS):
        raise ValueError("num_bins must be an integer from 1 to 100")
    return value


def _confidence(value: Any) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        raise ValueError("confidence values must be finite numbers from 0 to 1")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(
            "confidence values must be finite numbers from 0 to 1"
        ) from None
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("confidence values must be finite numbers from 0 to 1")
    return 0.0 if result == 0.0 else result


def _normalize_descriptor(
    value: Any,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Any:
    if _seen is None:
        _seen = set()
    if _depth > _MAX_DESCRIPTOR_DEPTH:
        raise ValueError("model metadata exceeds the supported nesting limit")
    if type(value) is dict:
        if id(value) in _seen:
            raise ValueError("model metadata must not contain cycles")
        if len(value) > _MAX_DESCRIPTOR_ITEMS:
            raise ValueError("model metadata exceeds the supported item limit")
        keys = list(value)
        if any(type(key) is not str or len(key) > _MAX_DESCRIPTOR_TEXT for key in keys):
            raise ValueError("model metadata keys must be bounded strings")
        _seen.add(id(value))
        result = {
            key: _normalize_descriptor(
                value[key],
                _depth=_depth + 1,
                _seen=_seen,
            )
            for key in sorted(keys)
        }
        _seen.remove(id(value))
        return result
    if type(value) in (list, tuple):
        if id(value) in _seen:
            raise ValueError("model metadata must not contain cycles")
        if len(value) > _MAX_DESCRIPTOR_ITEMS:
            raise ValueError("model metadata exceeds the supported item limit")
        _seen.add(id(value))
        result = [
            _normalize_descriptor(item, _depth=_depth + 1, _seen=_seen)
            for item in value
        ]
        _seen.remove(id(value))
        return result
    if type(value) is str:
        if len(value) > _MAX_DESCRIPTOR_TEXT:
            raise ValueError("model metadata text exceeds the supported limit")
        return value
    if type(value) is bool or value is None:
        return value
    if type(value) is int:
        if abs(value) > _MAX_SAMPLES:
            raise ValueError("model metadata integer exceeds the supported limit")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("model metadata numbers must be finite")
        return value
    raise TypeError("model metadata must contain only JSON-compatible values")


def _is_correct(sample: DocumentTypeCalibrationSample) -> int:
    return int(sample.expected_type == sample.predicted_type)


def _ratio(numerator: int, denominator: int) -> float:
    return _rounded(numerator / denominator) if denominator else 0.0


def _rounded(value: float) -> float:
    return round(value, 12)


def _aggregate_count(value: Any, *, field_name: str) -> None:
    if (
        type(value) is not int
        or type(value) is bool
        or not (0 <= value <= _MAX_SAMPLES)
    ):
        raise ValueError(f"{field_name} must be a bounded non-negative integer")


def _bounded_rate(value: Any, *, field_name: str) -> None:
    if type(value) not in (int, float) or type(value) is bool:
        raise ValueError(f"{field_name} must be a finite number from 0 to 1")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field_name} must be a finite number from 0 to 1")


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _report(value: Any) -> DocumentTypeCalibrationReport:
    if type(value) is not DocumentTypeCalibrationReport:
        raise TypeError("report must be a DocumentTypeCalibrationReport")
    return value


def _write_report(path: str | Path, content: str) -> Path:
    if type(path) is str:
        try:
            output_path = Path(path)
        except (OSError, ValueError):
            raise ValueError("invalid document-type report output path") from None
    elif isinstance(path, Path):
        output_path = path
    else:
        raise TypeError("report output path must be a string or Path")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
    except (OSError, ValueError):
        raise OSError("failed to write document-type calibration report") from None
    return output_path


__all__ = [
    "DEFAULT_ABSTENTION_THRESHOLDS",
    "DEFAULT_CALIBRATION_BINS",
    "DOCTYPE_CALIBRATION_ARTIFACT",
    "DOCTYPE_CALIBRATION_SCHEMA_VERSION",
    "DocumentTypeAbstentionMetrics",
    "DocumentTypeCalibrationBin",
    "DocumentTypeCalibrationReport",
    "DocumentTypeCalibrationSample",
    "DocumentTypeSupport",
    "build_doctype_calibration_report",
    "fingerprint_doctype_fixtures",
    "fingerprint_doctype_model",
    "render_doctype_calibration_json",
    "render_doctype_calibration_markdown",
]
