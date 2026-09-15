"""PHI-safe error slices for clinical natural-language inference evaluation.

The report consumes only opaque fixture identifiers and already-computed NLI
labels.  It deliberately has no premise, hypothesis, example, or model
inference field, so rendering the report cannot copy source text into an
evaluation artifact.  The five slices are fixed to keep comparisons stable
across runs and callers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from openmed.core.audit import stable_hash

NLI_ERROR_SLICE_ARTIFACT = "openmed.eval.nli_error_slices"
NLI_ERROR_SLICE_SCHEMA_VERSION = 1

CLINICAL_NLI_PHENOMENA: tuple[str, ...] = (
    "negation",
    "temporality",
    "experiencer",
    "numbers",
    "medication_status",
)
PREDEFINED_CLINICAL_PHENOMENA = CLINICAL_NLI_PHENOMENA
NLI_LABELS: tuple[str, ...] = ("entailment", "contradiction", "neutral")
ABSTENTION_LABEL = "abstain"
NLI_CONFUSION_LABELS: tuple[str, ...] = NLI_LABELS + (ABSTENTION_LABEL,)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_MAX_COUNT = 1_000_000_000
_MISSING = object()

_PHENOMENON_ALIASES = {
    "medication": "medication_status",
    "medication-status": "medication_status",
    "number": "numbers",
    "numeric": "numbers",
    "numerical": "numbers",
}
_LABEL_ALIASES = {
    "abstained": ABSTENTION_LABEL,
    "abstention": ABSTENTION_LABEL,
    "contradict": "contradiction",
    "contradicted": "contradiction",
    "entail": "entailment",
    "entailed": "entailment",
}

__all__ = [
    "ABSTENTION_LABEL",
    "CLINICAL_NLI_PHENOMENA",
    "NLI_CONFUSION_LABELS",
    "NLI_ERROR_SLICE_ARTIFACT",
    "NLI_ERROR_SLICE_SCHEMA_VERSION",
    "NLI_LABELS",
    "NLIErrorSlice",
    "NLIErrorSliceCase",
    "NLIErrorSliceProvenance",
    "NLIErrorSliceRecord",
    "NLIErrorSliceReport",
    "PREDEFINED_CLINICAL_PHENOMENA",
    "build_error_slice_report",
    "build_nli_error_slice_report",
    "build_nli_error_slices_report",
    "evaluate_nli_error_slices",
    "render_nli_error_slice_report_json",
    "render_nli_error_slice_report_markdown",
    "render_nli_error_slices_json",
    "render_nli_error_slices_markdown",
    "write_nli_error_slice_report",
]


def _safe_identifier(value: Any, *, field_name: str) -> str:
    """Validate an opaque identifier without echoing the supplied value."""

    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be an opaque identifier")
    return value


def _safe_optional_identifier(value: Any, *, default: str) -> str:
    """Return a safe identifier or a fixed fallback without exposing input."""

    if type(value) is str and _IDENTIFIER_RE.fullmatch(value) is not None:
        return value
    return default


def _normalise_phenomenon(value: Any) -> str:
    if type(value) is not str:
        raise ValueError("clinical phenomena must use the predefined names")
    normalized = value.strip().lower().replace(" ", "_")
    normalized = _PHENOMENON_ALIASES.get(normalized, normalized)
    if normalized not in CLINICAL_NLI_PHENOMENA:
        raise ValueError("clinical phenomena must use the predefined names")
    return normalized


def _normalise_label(value: Any, *, allow_abstention: bool = False) -> str:
    if type(value) is not str:
        raise ValueError("NLI labels must use the predefined names")
    normalized = value.strip().lower().replace(" ", "_")
    normalized = _LABEL_ALIASES.get(normalized, normalized)
    allowed = NLI_CONFUSION_LABELS if allow_abstention else NLI_LABELS
    if normalized not in allowed:
        raise ValueError("NLI labels must use the predefined names")
    return normalized


def _safe_count(value: Any) -> int:
    if type(value) is not int or value < 0 or value > _MAX_COUNT:
        raise ValueError("report counts must be bounded non-negative integers")
    return value


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _safe_digest(value: Any, *, namespace: str) -> str | None:
    """Return a digest, hashing arbitrary model metadata instead of storing it."""

    if value is None:
        return None
    if type(value) is str and _DIGEST_RE.fullmatch(value):
        return value
    if type(value) is str:
        return stable_hash({"namespace": namespace, "value": value})
    return stable_hash({"namespace": namespace, "value": "unavailable"})


def _plain_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _mapping_value(
    payload: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    default: Any = _MISSING,
) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    if default is not _MISSING:
        return default
    raise ValueError("NLI error-slice case is missing a required field")


def _normalise_phenomena(value: Any) -> tuple[str, ...]:
    if type(value) is str:
        raw_values: Iterable[Any] = (value,)
    elif isinstance(value, Mapping) or value is None:
        raise ValueError("a case must declare one or more clinical phenomena")
    else:
        try:
            raw_values = tuple(value)
        except TypeError as exc:
            raise ValueError(
                "a case must declare one or more clinical phenomena"
            ) from exc

    normalized_values = {_normalise_phenomenon(item) for item in raw_values}
    if not normalized_values:
        raise ValueError("a case must declare one or more clinical phenomena")
    return tuple(
        phenomenon
        for phenomenon in CLINICAL_NLI_PHENOMENA
        if phenomenon in normalized_values
    )


@dataclass(frozen=True)
class NLIErrorSliceCase:
    """One PHI-free scored fixture assigned to one or more clinical slices.

    Only the fixture identifier and label outcomes are accepted.  Callers must
    compute NLI outcomes before creating this record; premise and hypothesis
    text are intentionally not part of the input contract.
    """

    fixture_id: str
    phenomena: tuple[str, ...]
    gold_label: str
    predicted_label: str | None = None
    abstained: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fixture_id",
            _safe_identifier(self.fixture_id, field_name="fixture_id"),
        )
        object.__setattr__(self, "phenomena", _normalise_phenomena(self.phenomena))
        object.__setattr__(
            self,
            "gold_label",
            _normalise_label(self.gold_label),
        )

        predicted = self.predicted_label
        if predicted is not None:
            predicted = _normalise_label(predicted, allow_abstention=True)
        abstained = self.abstained
        if abstained is None:
            abstained = predicted is None or predicted == ABSTENTION_LABEL
        elif type(abstained) is not bool:
            raise ValueError("abstained must be a boolean")
        if predicted == ABSTENTION_LABEL:
            abstained = True
            predicted = None
        if abstained and predicted is not None:
            raise ValueError("abstained cases cannot carry a predicted label")
        if not abstained and predicted is None:
            raise ValueError("scored cases require a predicted label")

        object.__setattr__(self, "predicted_label", predicted)
        object.__setattr__(self, "abstained", abstained)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "NLIErrorSliceCase":
        """Build a case from a JSON-ready mapping without reading raw text."""

        if type(payload) is not dict:
            raise TypeError("NLI error-slice cases must be plain dictionaries")
        fixture_id = _mapping_value(payload, ("fixture_id", "id"))
        phenomena = _mapping_value(
            payload,
            ("phenomena", "phenomenon", "slice", "slices"),
        )
        gold_label = _mapping_value(payload, ("gold_label", "gold", "label"))
        predicted_label = _mapping_value(
            payload,
            ("predicted_label", "prediction", "predicted"),
            default=None,
        )
        abstained = payload.get("abstained")
        if abstained is None:
            abstained = predicted_label is None or (
                type(predicted_label) is str
                and _normalise_label(predicted_label, allow_abstention=True)
                == ABSTENTION_LABEL
            )
        return cls(
            fixture_id=fixture_id,
            phenomena=phenomena,
            gold_label=gold_label,
            predicted_label=predicted_label,
            abstained=abstained,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the safe, label-only case representation."""

        return {
            "abstained": self.abstained,
            "fixture_id": self.fixture_id,
            "gold_label": self.gold_label,
            "phenomena": list(self.phenomena),
            "predicted_label": (
                ABSTENTION_LABEL if self.abstained else self.predicted_label
            ),
        }


NLIErrorSliceRecord = NLIErrorSliceCase


@dataclass(frozen=True)
class NLIErrorSliceProvenance:
    """Content-free provenance attached to an error-slice artifact."""

    fixture_set_id: str
    fixture_set_digest: str
    model_digest: str | None = None
    evaluator: str = NLI_ERROR_SLICE_ARTIFACT
    raw_text_suppressed: bool = True
    examples_suppressed: bool = True
    network_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fixture_set_id",
            _safe_optional_identifier(self.fixture_set_id, default="provided"),
        )
        object.__setattr__(
            self,
            "fixture_set_digest",
            _safe_digest(
                self.fixture_set_digest,
                namespace="nli-error-slice-fixture-set",
            )
            or stable_hash({"namespace": "nli-error-slice-fixture-set"}),
        )
        object.__setattr__(
            self,
            "model_digest",
            _safe_digest(self.model_digest, namespace="nli-error-slice-model"),
        )
        object.__setattr__(self, "evaluator", NLI_ERROR_SLICE_ARTIFACT)
        object.__setattr__(self, "raw_text_suppressed", True)
        object.__setattr__(self, "examples_suppressed", True)
        object.__setattr__(self, "network_required", False)

    def to_dict(self) -> dict[str, Any]:
        """Return safe provenance flags and content digests."""

        return {
            "evaluator": self.evaluator,
            "examples_suppressed": self.examples_suppressed,
            "fixture_set_digest": self.fixture_set_digest,
            "fixture_set_id": self.fixture_set_id,
            "model_digest": self.model_digest,
            "network_required": self.network_required,
            "raw_text_suppressed": self.raw_text_suppressed,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "NLIErrorSliceProvenance":
        """Rebuild provenance while discarding non-allowlisted metadata."""

        if not isinstance(payload, Mapping):
            raise TypeError("provenance must be a mapping")
        return cls(
            fixture_set_id=payload.get("fixture_set_id", "provided"),
            fixture_set_digest=payload.get("fixture_set_digest", ""),
            model_digest=payload.get("model_digest"),
        )


def _empty_confusion_matrix() -> dict[str, dict[str, int]]:
    return {
        gold: {predicted: 0 for predicted in NLI_CONFUSION_LABELS}
        for gold in NLI_LABELS
    }


def _normalise_confusion_matrix(
    value: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, int]]:
    if not isinstance(value, Mapping):
        raise TypeError("confusion_matrix must be a mapping")
    matrix = _empty_confusion_matrix()
    for gold, row in value.items():
        if gold not in NLI_LABELS:
            raise ValueError("confusion_matrix contains an unsupported gold label")
        if not isinstance(row, Mapping):
            raise TypeError("confusion_matrix rows must be mappings")
        for predicted, count in row.items():
            if predicted not in NLI_CONFUSION_LABELS:
                raise ValueError("confusion_matrix contains an unsupported prediction")
            matrix[gold][predicted] = _safe_count(count)
    return MappingProxyType(
        {gold: MappingProxyType(dict(row)) for gold, row in matrix.items()}
    )


@dataclass(frozen=True)
class NLIErrorSlice:
    """Aggregate counts for one predefined clinical phenomenon."""

    name: str
    fixture_ids: tuple[str, ...]
    confusion_matrix: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        name = _normalise_phenomenon(self.name)
        identifiers = tuple(
            sorted(
                _safe_identifier(item, field_name="fixture_id")
                for item in self.fixture_ids
            )
        )
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("a slice cannot contain duplicate fixture identifiers")
        matrix = _normalise_confusion_matrix(self.confusion_matrix)
        if sum(sum(row.values()) for row in matrix.values()) != len(identifiers):
            raise ValueError("confusion counts must match the fixture identifiers")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "fixture_ids", identifiers)
        object.__setattr__(self, "confusion_matrix", matrix)

    @property
    def fixture_count(self) -> int:
        """Return the number of fixture assignments in this slice."""

        return len(self.fixture_ids)

    @property
    def total_count(self) -> int:
        """Return the number of scored or abstained assignments."""

        return self.fixture_count

    @property
    def abstained_count(self) -> int:
        """Return the number of abstained assignments."""

        return sum(self.confusion_matrix[gold][ABSTENTION_LABEL] for gold in NLI_LABELS)

    @property
    def scored_count(self) -> int:
        """Return the number of non-abstained assignments."""

        return self.total_count - self.abstained_count

    @property
    def correct_count(self) -> int:
        """Return the diagonal count over non-abstained NLI labels."""

        return sum(self.confusion_matrix[label][label] for label in NLI_LABELS)

    @property
    def error_count(self) -> int:
        """Return non-abstained assignments outside the NLI diagonal."""

        return self.scored_count - self.correct_count

    @property
    def abstention_coverage(self) -> float:
        """Return the fraction of assignments with a usable prediction."""

        return _rate(self.scored_count, self.total_count)

    @property
    def abstention_rate(self) -> float:
        """Return the fraction of assignments that abstained."""

        return _rate(self.abstained_count, self.total_count)

    @property
    def accuracy(self) -> float:
        """Return accuracy over non-abstained assignments."""

        return _rate(self.correct_count, self.scored_count)

    def to_dict(self) -> dict[str, Any]:
        """Return only fixture identifiers and aggregate slice evidence."""

        counts = {
            "abstained": self.abstained_count,
            "correct": self.correct_count,
            "errors": self.error_count,
            "scored": self.scored_count,
            "total": self.total_count,
        }
        return {
            "abstention": {
                "abstained": self.abstained_count,
                "coverage": self.abstention_coverage,
                "rate": self.abstention_rate,
                "scored": self.scored_count,
                "total": self.total_count,
            },
            "accuracy": self.accuracy,
            "abstained_count": self.abstained_count,
            "abstention_coverage": self.abstention_coverage,
            "abstention_rate": self.abstention_rate,
            "confusion_matrix": {
                gold: dict(self.confusion_matrix[gold]) for gold in NLI_LABELS
            },
            "counts": counts,
            "fixture_count": self.fixture_count,
            "fixture_ids": list(self.fixture_ids),
            "name": self.name,
            "scored_count": self.scored_count,
        }

    @classmethod
    def from_mapping(
        cls,
        name: str,
        payload: Mapping[str, Any],
    ) -> "NLIErrorSlice":
        """Rebuild a slice from a serialized report mapping."""

        if not isinstance(payload, Mapping):
            raise TypeError("slice payloads must be mappings")
        fixture_ids = payload.get("fixture_ids", ())
        if isinstance(fixture_ids, str) or not isinstance(fixture_ids, Iterable):
            raise ValueError("slice fixture_ids must be a sequence")
        return cls(
            name=name,
            fixture_ids=tuple(fixture_ids),
            confusion_matrix=payload.get("confusion_matrix", {}),
        )

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class NLIErrorSliceReport:
    """Deterministic, counts-only report across all clinical NLI slices."""

    slices: Mapping[str, NLIErrorSlice]
    provenance: NLIErrorSliceProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.slices, Mapping):
            raise TypeError("slices must be a mapping")
        unknown = set(self.slices) - set(CLINICAL_NLI_PHENOMENA)
        if unknown:
            raise ValueError("report contains an unsupported clinical slice")
        normalized: dict[str, NLIErrorSlice] = {}
        for name in CLINICAL_NLI_PHENOMENA:
            slice_report = self.slices.get(name)
            if slice_report is None:
                slice_report = NLIErrorSlice(
                    name=name,
                    fixture_ids=(),
                    confusion_matrix=_empty_confusion_matrix(),
                )
            if not isinstance(slice_report, NLIErrorSlice):
                raise TypeError("slices must contain NLIErrorSlice values")
            normalized[name] = slice_report
        if not isinstance(self.provenance, NLIErrorSliceProvenance):
            raise TypeError("provenance must be NLIErrorSliceProvenance")
        object.__setattr__(self, "slices", MappingProxyType(normalized))

    @property
    def fixture_count(self) -> int:
        """Return the unique fixture count across all slice assignments."""

        return len(
            {
                fixture_id
                for slice_report in self.slices.values()
                for fixture_id in slice_report.fixture_ids
            }
        )

    @property
    def summary(self) -> Mapping[str, int | float]:
        """Return aggregate assignment counts and abstention coverage."""

        total = sum(slice_report.total_count for slice_report in self.slices.values())
        scored = sum(slice_report.scored_count for slice_report in self.slices.values())
        abstained = sum(
            slice_report.abstained_count for slice_report in self.slices.values()
        )
        correct = sum(
            slice_report.correct_count for slice_report in self.slices.values()
        )
        return {
            "abstained_count": abstained,
            "abstention_coverage": _rate(scored, total),
            "abstention_rate": _rate(abstained, total),
            "correct_count": correct,
            "error_count": scored - correct,
            "slice_assignment_count": total,
            "scored_count": scored,
            "unique_fixture_count": self.fixture_count,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic artifact with no examples or raw text."""

        return {
            "artifact_type": NLI_ERROR_SLICE_ARTIFACT,
            "fixture_count": self.fixture_count,
            "provenance": self.provenance.to_dict(),
            "schema_version": NLI_ERROR_SLICE_SCHEMA_VERSION,
            "slices": {
                name: self.slices[name].to_dict() for name in CLINICAL_NLI_PHENOMENA
            },
            "summary": dict(self.summary),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NLIErrorSliceReport":
        """Rebuild a report from :meth:`to_dict` output."""

        if not isinstance(payload, Mapping):
            raise TypeError("NLI error-slice reports must be mappings")
        if payload.get("artifact_type") != NLI_ERROR_SLICE_ARTIFACT:
            raise ValueError("unsupported NLI error-slice report artifact")
        if payload.get("schema_version") != NLI_ERROR_SLICE_SCHEMA_VERSION:
            raise ValueError("unsupported NLI error-slice report schema")
        raw_slices = payload.get("slices")
        if not isinstance(raw_slices, Mapping):
            raise ValueError("NLI error-slice reports require slice mappings")
        provenance = NLIErrorSliceProvenance.from_mapping(
            _plain_mapping(payload.get("provenance", {}), field_name="provenance")
        )
        slices = {
            name: NLIErrorSlice.from_mapping(
                name,
                raw_slices[name],
            )
            for name in CLINICAL_NLI_PHENOMENA
            if name in raw_slices
        }
        return cls(slices=slices, provenance=provenance)

    @classmethod
    def read_json(cls, path: str | Path) -> "NLIErrorSliceReport":
        """Read a JSON report without any network or model access."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("NLI error-slice report JSON must contain an object")
        return cls.from_dict(payload)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize a stable JSON report."""

        if type(indent) is not int or indent < 0 or indent > 8:
            raise ValueError("JSON indentation must be an integer from 0 through 8")
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write a stable JSON report to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render deterministic, counts-only Markdown for human review."""

        lines = [
            "# Clinical NLI Error-Slice Report",
            "",
            "This evaluation artifact reports aggregate slice failures only. "
            "Premise, hypothesis, examples, and raw clinical text are suppressed.",
            "",
            "## Provenance",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Fixture set | `{self.provenance.fixture_set_id}` |",
            f"| Fixture-set digest | `{self.provenance.fixture_set_digest}` |",
            f"| Model digest | `{self.provenance.model_digest or 'not-recorded'}` |",
            "| Network required | no |",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Unique fixtures | {self.fixture_count} |",
            f"| Slice assignments | {self.summary['slice_assignment_count']} |",
            f"| Scored | {self.summary['scored_count']} |",
            f"| Abstained | {self.summary['abstained_count']} |",
            f"| Abstention coverage | {_format_rate(self.summary['abstention_coverage'])} |",
            f"| Abstention rate | {_format_rate(self.summary['abstention_rate'])} |",
            "",
            "## Slices",
            "",
        ]
        for name in CLINICAL_NLI_PHENOMENA:
            slice_report = self.slices[name]
            lines.extend(
                [
                    f"### {name}",
                    "",
                    "| Metric | Value |",
                    "|---|---:|",
                    f"| Fixtures | {slice_report.fixture_count} |",
                    f"| Scored | {slice_report.scored_count} |",
                    f"| Abstained | {slice_report.abstained_count} |",
                    f"| Abstention coverage | {_format_rate(slice_report.abstention_coverage)} |",
                    f"| Abstention rate | {_format_rate(slice_report.abstention_rate)} |",
                    f"| Accuracy (scored) | {_format_rate(slice_report.accuracy)} |",
                    f"| Fixture IDs | {_format_fixture_ids(slice_report.fixture_ids)} |",
                    "",
                    "| Gold \\ Predicted | Entailment | Contradiction | Neutral | Abstain |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for gold in NLI_LABELS:
                row = slice_report.confusion_matrix[gold]
                lines.append(
                    f"| {gold} | {row['entailment']} | "
                    f"{row['contradiction']} | {row['neutral']} | {row['abstain']} |"
                )
            lines.append("")
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write the Markdown report to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def _format_rate(value: Any) -> str:
    return f"{float(value):.4f}"


def _format_fixture_ids(fixture_ids: Iterable[str]) -> str:
    identifiers = tuple(fixture_ids)
    if not identifiers:
        return "none"
    return ", ".join(f"`{fixture_id}`" for fixture_id in identifiers)


def _coerce_case(value: Any) -> NLIErrorSliceCase:
    if isinstance(value, NLIErrorSliceCase):
        return value
    if type(value) is dict:
        return NLIErrorSliceCase.from_mapping(value)
    raise TypeError("NLI error-slice records must be cases or plain dictionaries")


def _coerce_provenance_value(
    provenance: Mapping[str, Any] | None,
    names: tuple[str, ...],
) -> Any:
    if provenance is None:
        return None
    for name in names:
        if name in provenance:
            return provenance[name]
    return None


def build_nli_error_slice_report(
    records: Iterable[NLIErrorSliceCase | Mapping[str, Any]] | Mapping[str, Any],
    *,
    fixture_set_id: str = "provided",
    model_id: str | None = None,
    model_digest: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> NLIErrorSliceReport:
    """Build a deterministic clinical NLI error-slice report.

    Args:
        records: Cases containing fixture IDs, clinical slice names, gold NLI
            labels, predicted labels, and an optional abstention flag. A single
            plain dictionary is accepted for convenience.
        fixture_set_id: Opaque identifier retained for evidence provenance.
        model_id: Optional model identifier. It is hashed before serialization.
        model_digest: Optional precomputed ``sha256:`` digest.
        provenance: Optional mapping whose model keys are limited to
            ``model_id``/``model_digest`` and whose fixture-set key is
            ``fixture_set_id``. Other values are ignored.

    Returns:
        A report with all five predefined slices, aggregate confusion matrices,
        abstention coverage, and no source-text examples.
    """

    if isinstance(records, NLIErrorSliceCase):
        raw_records: Iterable[Any] = (records,)
    elif type(records) is dict:
        raw_records: Iterable[Any] = (records,)
    else:
        if isinstance(records, (str, bytes)):
            raise TypeError("records must be an iterable of NLI error-slice cases")
        try:
            raw_records = tuple(records)
        except TypeError as exc:
            raise TypeError(
                "records must be an iterable of NLI error-slice cases"
            ) from exc

    cases = tuple(
        sorted(
            (_coerce_case(item) for item in raw_records),
            key=lambda item: item.fixture_id,
        )
    )
    fixture_ids = [case.fixture_id for case in cases]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("fixture identifiers must be unique across input cases")

    matrix_by_slice = {
        name: _empty_confusion_matrix() for name in CLINICAL_NLI_PHENOMENA
    }
    ids_by_slice = {name: [] for name in CLINICAL_NLI_PHENOMENA}
    for case in cases:
        predicted = ABSTENTION_LABEL if case.abstained else case.predicted_label
        for phenomenon in case.phenomena:
            ids_by_slice[phenomenon].append(case.fixture_id)
            matrix_by_slice[phenomenon][case.gold_label][predicted] += 1  # type: ignore[index]

    slices = {
        name: NLIErrorSlice(
            name=name,
            fixture_ids=tuple(ids_by_slice[name]),
            confusion_matrix=matrix_by_slice[name],
        )
        for name in CLINICAL_NLI_PHENOMENA
    }

    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    raw_provenance = provenance
    selected_fixture_set_id = fixture_set_id
    if selected_fixture_set_id == "provided" and raw_provenance is not None:
        selected_fixture_set_id = (
            _coerce_provenance_value(
                raw_provenance,
                ("fixture_set_id",),
            )
            or selected_fixture_set_id
        )

    selected_model = model_digest
    if selected_model is None and raw_provenance is not None:
        selected_model = _coerce_provenance_value(
            raw_provenance,
            ("model_digest", "model_id", "model"),
        )
    if selected_model is None:
        selected_model = model_id

    fixture_set_digest = stable_hash(
        {
            "artifact": NLI_ERROR_SLICE_ARTIFACT,
            "cases": [case.to_dict() for case in cases],
        }
    )
    report_provenance = NLIErrorSliceProvenance(
        fixture_set_id=selected_fixture_set_id,
        fixture_set_digest=fixture_set_digest,
        model_digest=_safe_digest(selected_model, namespace="nli-error-slice-model"),
    )
    return NLIErrorSliceReport(slices=slices, provenance=report_provenance)


def _as_report(
    value: NLIErrorSliceReport
    | Iterable[NLIErrorSliceCase | Mapping[str, Any]]
    | Mapping[str, Any],
) -> NLIErrorSliceReport:
    if isinstance(value, NLIErrorSliceReport):
        return value
    if (
        isinstance(value, Mapping)
        and value.get("artifact_type") == NLI_ERROR_SLICE_ARTIFACT
    ):
        return NLIErrorSliceReport.from_dict(value)
    return build_nli_error_slice_report(value)


def render_nli_error_slice_report_json(
    report: NLIErrorSliceReport
    | Iterable[NLIErrorSliceCase | Mapping[str, Any]]
    | Mapping[str, Any],
    *,
    indent: int = 2,
) -> str:
    """Render a report or report input records as deterministic JSON."""

    return _as_report(report).to_json(indent=indent)


def render_nli_error_slice_report_markdown(
    report: NLIErrorSliceReport
    | Iterable[NLIErrorSliceCase | Mapping[str, Any]]
    | Mapping[str, Any],
) -> str:
    """Render a report or report input records as deterministic Markdown."""

    return _as_report(report).to_markdown()


def write_nli_error_slice_report(
    report: NLIErrorSliceReport,
    path: str | Path,
    *,
    format: str = "json",
    indent: int = 2,
) -> Path:
    """Write a JSON or Markdown report using an explicit safe format."""

    if format == "json":
        return report.write_json(path, indent=indent)
    if format in {"markdown", "md"}:
        return report.write_markdown(path)
    raise ValueError("report format must be json or markdown")


build_error_slice_report = build_nli_error_slice_report
build_nli_error_slices_report = build_nli_error_slice_report
evaluate_nli_error_slices = build_nli_error_slice_report
render_nli_error_slices_json = render_nli_error_slice_report_json
render_nli_error_slices_markdown = render_nli_error_slice_report_markdown
