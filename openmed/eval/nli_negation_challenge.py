"""Deterministic synthetic negation challenge for clinical NLI.

The challenge is deliberately backend-neutral.  Callers supply predictions from
an already-available local model, or pass precomputed labels, while this module
provides the synthetic cases and the safety-focused aggregation.  No model is
loaded and no network access is performed here.

Reports contain only bounded counts, controlled pattern names, gate values, and
content digests.  Premises, hypotheses, case identifiers, predictor output,
and predictor exceptions never cross the report boundary.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

NLI_NEGATION_CHALLENGE = "nli_negation_challenge"
NLI_NEGATION_CHALLENGE_SCHEMA_VERSION = "openmed.eval.nli_negation_challenge.v1"
NLI_NEGATION_FIXTURE_SCHEMA_VERSION = 1

NEGATION_PATTERNS: tuple[str, ...] = (
    "simple",
    "nested",
    "double",
    "section_scoped",
)
NLI_LABELS: tuple[str, ...] = (
    "entailment",
    "contradiction",
    "neutral",
    "abstention",
)
DEFAULT_MAX_FALSE_ENTAILMENT_RATE = 0.0
MAX_NEGATION_CASES = 256
MAX_CASE_TEXT_LENGTH = 2_048
MAX_CASE_ID_LENGTH = 64

_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")
_SECTION_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")
_LABEL_ALIASES = {
    "abstain": "abstention",
    "abstained": "abstention",
    "abstention": "abstention",
    "contradict": "contradiction",
    "contradicted": "contradiction",
    "contradiction": "contradiction",
    "entail": "entailment",
    "entailed": "entailment",
    "entailment": "entailment",
    "neutral": "neutral",
}

NliPredictor: TypeAlias = Callable[..., Any]
NliPredictionInput: TypeAlias = Sequence[Any] | Mapping[str, Any] | Iterable[Any]


class NliNegationChallengeError(ValueError):
    """Raised when a challenge case, prediction, or report input is invalid."""


class NliNegationGateError(AssertionError):
    """Raised when the clinical NLI negation gate fails."""


@dataclass(frozen=True)
class NliNegationCase:
    """One synthetic premise-hypothesis pair in the negation challenge.

    Args:
        case_id: Stable, non-sensitive identifier used only for joining
            precomputed predictions.  It is never rendered in a report.
        pattern: One of ``simple``, ``nested``, ``double``, or
            ``section_scoped``.
        premise: Synthetic clinical premise shown to the predictor.
        hypothesis: Synthetic hypothesis shown to the predictor.
        gold_label: Expected NLI label: ``entailment``, ``contradiction``, or
            ``neutral``.
        negation_depth: Number of negation operators represented by the case.
        section: Controlled section name for section-scoped cases.
        synthetic: Safety declaration for the fixture.
        contains_real_phi: Safety declaration for the fixture.

    The text fields are available to a local predictor because they are the
    challenge input.  Evaluation reports intentionally omit them.
    """

    case_id: str
    pattern: str
    premise: str
    hypothesis: str
    gold_label: str
    negation_depth: int = 1
    section: str | None = None
    synthetic: bool = True
    contains_real_phi: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize the controlled fixture metadata."""

        case_id = _normalize_identifier(self.case_id, _CASE_ID_RE)
        pattern = _normalize_pattern(self.pattern)
        premise = _require_text(self.premise)
        hypothesis = _require_text(self.hypothesis)
        gold_label = _normalize_label(self.gold_label, allow_abstention=False)

        if isinstance(self.negation_depth, bool) or not isinstance(
            self.negation_depth, int
        ):
            raise NliNegationChallengeError("negation depth must be a positive integer")
        if not 1 <= self.negation_depth <= len(NEGATION_PATTERNS):
            raise NliNegationChallengeError("negation depth is outside the safe range")
        if self.synthetic is not True:
            raise NliNegationChallengeError("negation fixtures must be synthetic")
        if self.contains_real_phi is not False:
            raise NliNegationChallengeError(
                "negation fixtures must declare that they contain no real PHI"
            )

        section = _normalize_optional_identifier(self.section, _SECTION_RE)
        if pattern == "section_scoped" and section is None:
            raise NliNegationChallengeError(
                "section-scoped negation fixtures require a section"
            )
        if pattern != "section_scoped" and section is not None:
            raise NliNegationChallengeError(
                "only section-scoped negation fixtures may name a section"
            )

        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "premise", premise)
        object.__setattr__(self, "hypothesis", hypothesis)
        object.__setattr__(self, "gold_label", gold_label)
        object.__setattr__(self, "section", section)

    @property
    def requires_non_entailment(self) -> bool:
        """Return whether predicting entailment is a false entailment."""

        return self.gold_label != "entailment"

    @property
    def expected_entailment(self) -> bool:
        """Return whether the pair's gold label is entailment."""

        return self.gold_label == "entailment"

    def to_mapping(self) -> dict[str, Any]:
        """Return the explicit synthetic fixture representation.

        This method is for local fixture exchange.  It is not used for report
        serialization because it necessarily includes the challenge text.
        """

        return {
            "case_id": self.case_id,
            "contains_real_phi": False,
            "gold_label": self.gold_label,
            "hypothesis": self.hypothesis,
            "negation_depth": self.negation_depth,
            "pattern": self.pattern,
            "phi_free": True,
            "premise": self.premise,
            "section": self.section,
            "synthetic": True,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NliNegationCase":
        """Build a validated synthetic case from a JSON-like mapping.

        The loader accepts ``id``/``case_id``, ``category``/``pattern``, and
        ``label``/``gold_label`` aliases so local fixture manifests can remain
        small.  Safety flags may be top-level or nested under ``metadata``.
        """

        if not isinstance(value, Mapping):
            raise NliNegationChallengeError("negation fixture rows must be objects")

        metadata = value.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise NliNegationChallengeError(
                "negation fixture metadata must be an object"
            )

        synthetic = _first_present(value, metadata, "synthetic")
        if synthetic is not True:
            raise NliNegationChallengeError(
                "negation fixtures must declare synthetic=true"
            )
        contains_real_phi = _first_present(value, metadata, "contains_real_phi")
        if contains_real_phi is not False:
            raise NliNegationChallengeError(
                "negation fixtures must declare contains_real_phi=false"
            )
        phi_free = _first_present(value, metadata, "phi_free")
        if phi_free is not True:
            raise NliNegationChallengeError(
                "negation fixtures must declare phi_free=true"
            )

        case_id = _first_present(value, metadata, "case_id", "id")
        pattern = _first_present(value, metadata, "pattern", "category")
        gold_label = _first_present(
            value,
            metadata,
            "gold_label",
            "expected_label",
            "label",
        )
        section = _first_present(value, metadata, "section")
        return cls(
            case_id=case_id,
            pattern=pattern,
            premise=value.get("premise"),
            hypothesis=value.get("hypothesis"),
            gold_label=gold_label,
            negation_depth=value.get("negation_depth", 1),
            section=section,
            synthetic=True,
            contains_real_phi=False,
        )


NliNegationPair = NliNegationCase
NliNegationFixture = NliNegationCase


@dataclass(frozen=True)
class NliNegationPatternMetrics:
    """Aggregate accuracy and false-entailment counts for one pattern."""

    pattern: str
    case_count: int
    correct_count: int
    abstention_count: int
    non_entailment_case_count: int
    false_entailment_count: int

    @property
    def aggregate_accuracy(self) -> float:
        """Return accuracy over all cases, counting abstention as incorrect."""

        return _rate(self.correct_count, self.case_count)

    @property
    def accuracy(self) -> float:
        """Alias for :attr:`aggregate_accuracy`."""

        return self.aggregate_accuracy

    @property
    def false_entailment_rate(self) -> float:
        """Return false entailments over non-entailment target cases."""

        return _rate(self.false_entailment_count, self.non_entailment_case_count)

    def to_dict(self) -> dict[str, int | float | str]:
        """Return deterministic aggregate metrics without source text."""

        return {
            "abstention_count": self.abstention_count,
            "accuracy": self.accuracy,
            "aggregate_accuracy": self.aggregate_accuracy,
            "case_count": self.case_count,
            "correct_count": self.correct_count,
            "false_entailment_count": self.false_entailment_count,
            "false_entailment_rate": self.false_entailment_rate,
            "non_entailment_case_count": self.non_entailment_case_count,
            "pattern": self.pattern,
        }


@dataclass(frozen=True)
class NliNegationGateResult:
    """Separate accuracy and false-entailment gate decisions."""

    aggregate_accuracy: float
    minimum_aggregate_accuracy: float | None
    aggregate_accuracy_gate_passed: bool
    false_entailment_rate: float
    max_false_entailment_rate: float
    false_entailment_gate_passed: bool
    passed: bool

    @property
    def accuracy_gate_passed(self) -> bool:
        """Return the aggregate-accuracy gate decision."""

        return self.aggregate_accuracy_gate_passed

    @property
    def false_entailment_passed(self) -> bool:
        """Return the dedicated false-entailment gate decision."""

        return self.false_entailment_gate_passed

    @property
    def gate_passed(self) -> bool:
        """Return the combined gate decision."""

        return self.passed

    def to_dict(self) -> dict[str, bool | float | None]:
        """Return the two independently visible gate decisions."""

        return {
            "aggregate_accuracy": self.aggregate_accuracy,
            "aggregate_accuracy_gate_passed": self.aggregate_accuracy_gate_passed,
            "false_entailment_gate_passed": self.false_entailment_gate_passed,
            "false_entailment_rate": self.false_entailment_rate,
            "max_false_entailment_rate": self.max_false_entailment_rate,
            "minimum_aggregate_accuracy": self.minimum_aggregate_accuracy,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class NliNegationReport:
    """Privacy-safe result for one synthetic clinical NLI run."""

    fixture_set_hash: str
    prediction_hash: str
    case_count: int
    correct_count: int
    abstention_count: int
    aggregate_accuracy: float
    false_entailment_count: int
    false_entailment_case_count: int
    false_entailment_rate: float
    by_pattern: Mapping[str, NliNegationPatternMetrics]
    gate: NliNegationGateResult
    schema_version: str = NLI_NEGATION_CHALLENGE_SCHEMA_VERSION
    challenge: str = NLI_NEGATION_CHALLENGE

    def __post_init__(self) -> None:
        """Freeze per-pattern metrics and validate the report shape."""

        if self.schema_version != NLI_NEGATION_CHALLENGE_SCHEMA_VERSION:
            raise NliNegationChallengeError("unsupported negation report schema")
        if self.challenge != NLI_NEGATION_CHALLENGE:
            raise NliNegationChallengeError("unsupported negation report challenge")
        if not isinstance(self.by_pattern, Mapping):
            raise NliNegationChallengeError(
                "negation report patterns must be an object"
            )
        missing = set(NEGATION_PATTERNS) - set(self.by_pattern)
        if missing:
            raise NliNegationChallengeError("negation report patterns are incomplete")
        ordered = {pattern: self.by_pattern[pattern] for pattern in NEGATION_PATTERNS}
        if any(
            not isinstance(metrics, NliNegationPatternMetrics)
            for metrics in ordered.values()
        ):
            raise NliNegationChallengeError(
                "negation report pattern metrics are invalid"
            )
        object.__setattr__(self, "by_pattern", MappingProxyType(ordered))

    @property
    def accuracy(self) -> float:
        """Alias for aggregate NLI accuracy."""

        return self.aggregate_accuracy

    @property
    def fixture_count(self) -> int:
        """Alias for the number of evaluated challenge cases."""

        return self.case_count

    @property
    def false_entailment_denominator(self) -> int:
        """Return the non-entailment target count used by the gate."""

        return self.false_entailment_case_count

    @property
    def aggregate_accuracy_gate_passed(self) -> bool:
        """Return the aggregate-accuracy gate decision."""

        return self.gate.aggregate_accuracy_gate_passed

    @property
    def false_entailment_gate_passed(self) -> bool:
        """Return the dedicated false-entailment gate decision."""

        return self.gate.false_entailment_gate_passed

    @property
    def gate_passed(self) -> bool:
        """Return the combined gate decision."""

        return self.gate.passed

    @property
    def passed(self) -> bool:
        """Return the combined gate decision."""

        return self.gate.passed

    def to_dict(self) -> dict[str, Any]:
        """Return an aggregate-only, JSON-ready report payload."""

        return {
            "abstention_count": self.abstention_count,
            "accuracy": self.accuracy,
            "aggregate_accuracy": self.aggregate_accuracy,
            "by_pattern": {
                pattern: self.by_pattern[pattern].to_dict()
                for pattern in NEGATION_PATTERNS
            },
            "case_count": self.case_count,
            "challenge": self.challenge,
            "correct_count": self.correct_count,
            "fixture_count": self.fixture_count,
            "false_entailment_case_count": self.false_entailment_case_count,
            "false_entailment_denominator": self.false_entailment_denominator,
            "false_entailment_count": self.false_entailment_count,
            "false_entailment_gate_passed": self.false_entailment_gate_passed,
            "false_entailment_rate": self.false_entailment_rate,
            "fixture_set_hash": self.fixture_set_hash,
            "gate": self.gate.to_dict(),
            "aggregate_accuracy_gate_passed": self.aggregate_accuracy_gate_passed,
            "gate_passed": self.gate_passed,
            "prediction_hash": self.prediction_hash,
            "passed": self.passed,
            "schema_version": self.schema_version,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the aggregate report to deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON to a local path."""

        try:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                self.to_json(indent=indent) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError, UnicodeError) as exc:
            del exc
            raise NliNegationChallengeError(
                "unable to write the local negation report"
            ) from None
        return output_path

    def to_markdown(self) -> str:
        """Render a deterministic aggregate-only Markdown report."""

        gate = self.gate
        lines = [
            "# Clinical NLI Negation Challenge",
            "",
            "| Measure | Value |",
            "|---|---:|",
            f"| Cases | {self.case_count} |",
            f"| Aggregate accuracy | {_format_rate(self.aggregate_accuracy)} |",
            f"| Abstentions | {self.abstention_count} |",
            (
                "| False entailments | "
                f"{self.false_entailment_count}/{self.false_entailment_case_count} |"
            ),
            (f"| False-entailment rate | {_format_rate(self.false_entailment_rate)} |"),
            (
                "| False-entailment gate | "
                f"{'passed' if gate.false_entailment_gate_passed else 'failed'} |"
            ),
            (
                "| Aggregate-accuracy gate | "
                f"{'passed' if gate.aggregate_accuracy_gate_passed else 'failed'} |"
            ),
            f"| Combined gate | {'passed' if gate.passed else 'failed'} |",
            "",
            "## Pattern metrics",
            "",
            (
                "| Pattern | Cases | Correct | Accuracy | Non-entailment "
                "cases | False entailments | False-entailment rate |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for pattern in NEGATION_PATTERNS:
            metrics = self.by_pattern[pattern]
            lines.append(
                f"| `{pattern}` | {metrics.case_count} | {metrics.correct_count} | "
                f"{_format_rate(metrics.aggregate_accuracy)} | "
                f"{metrics.non_entailment_case_count} | "
                f"{metrics.false_entailment_count} | "
                f"{_format_rate(metrics.false_entailment_rate)} |"
            )
        lines.extend(
            [
                "",
                "The report is aggregate-only; source text and case identifiers "
                "are intentionally omitted.",
                "",
            ]
        )
        return "\n".join(lines)

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to a local path."""

        try:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(self.to_markdown(), encoding="utf-8")
        except (OSError, TypeError, ValueError, UnicodeError) as exc:
            del exc
            raise NliNegationChallengeError(
                "unable to write the local negation report"
            ) from None
        return output_path

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access to the serialized report."""

        return self.to_dict()[key]


def default_nli_negation_cases() -> tuple[NliNegationCase, ...]:
    """Return the stable, synthetic four-pattern challenge corpus."""

    return DEFAULT_NLI_NEGATION_CASES


def build_nli_negation_challenge() -> tuple[NliNegationCase, ...]:
    """Return challenge cases for a local predictor or fixture export."""

    return default_nli_negation_cases()


def load_nli_negation_fixtures(
    path: str | Path | None = None,
) -> tuple[NliNegationCase, ...]:
    """Load local synthetic cases, or return the built-in corpus.

    JSON files may contain a list of cases or an object with a ``cases`` or
    ``fixtures`` list.  JSONL files contain one case object per non-empty line.
    The loader is local-only and requires the explicit ``synthetic=true`` flag.

    Args:
        path: Optional local JSON/JSONL path.  When omitted, the built-in
            challenge is returned.

    Returns:
        Validated cases in source order.

    Raises:
        NliNegationChallengeError: If the local file is malformed or unsafe.
    """

    if path is None:
        return default_nli_negation_cases()
    try:
        fixture_path = Path(path)
        raw_text = fixture_path.read_text(encoding="utf-8")
    except (OSError, TypeError, ValueError, UnicodeError) as exc:
        del exc
        raise NliNegationChallengeError(
            "unable to read local negation fixtures"
        ) from None

    try:
        if fixture_path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
        else:
            payload = json.loads(raw_text)
            if isinstance(payload, list):
                rows = payload
            elif isinstance(payload, Mapping):
                rows = payload.get("cases", payload.get("fixtures"))
            else:
                rows = None
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        del exc
        raise NliNegationChallengeError(
            "unable to parse local negation fixtures"
        ) from None

    if not isinstance(rows, list) or not rows:
        raise NliNegationChallengeError(
            "local negation fixtures must contain at least one case"
        )
    return _coerce_cases(rows)


def run_nli_negation_challenge(
    predictions: NliPredictionInput | NliPredictor | None = None,
    *,
    runner: NliPredictor | None = None,
    predictor: NliPredictor | None = None,
    cases: Iterable[NliNegationCase | Mapping[str, Any]] | None = None,
    max_false_entailment_rate: float = DEFAULT_MAX_FALSE_ENTAILMENT_RATE,
    minimum_aggregate_accuracy: float | None = None,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
) -> NliNegationReport:
    """Score synthetic negation cases with an independent safety gate.

    Args:
        predictions: An ordered iterable of labels, a mapping keyed by case
            identifier, or a callable predictor.  A callable passed here is
            treated as ``runner`` for convenient positional use.
        runner: Local predictor.  It may accept ``(premise, hypothesis)`` or a
            single :class:`NliNegationCase`.  It must return a label or a
            mapping/object containing a ``label`` field.
        predictor: Alias for ``runner``.
        cases: Optional validated cases.  The default corpus is used when
            omitted.
        max_false_entailment_rate: Inclusive ceiling for predictions of
            entailment on contradiction/neutral targets.  The default is zero.
        minimum_aggregate_accuracy: Optional independent accuracy floor.  If
            omitted, aggregate accuracy is measured but does not gate the run.
        output_json: Optional local JSON report path.
        output_markdown: Optional local Markdown report path.

    Returns:
        A deterministic aggregate-only report.  Abstentions count as incorrect
        for aggregate accuracy and do not count as false entailments.

    Raises:
        NliNegationChallengeError: If the input contract is invalid or the
            predictor fails.  Predictor exception details are deliberately
            suppressed.
    """

    if runner is not None and predictor is not None:
        raise NliNegationChallengeError("provide only one local NLI predictor")
    selected_runner = runner if runner is not None else predictor
    if callable(predictions):
        if selected_runner is not None:
            raise NliNegationChallengeError(
                "provide predictions or one local NLI predictor"
            )
        selected_runner = predictions
        predictions = None
    if selected_runner is not None and not callable(selected_runner):
        raise NliNegationChallengeError("local NLI predictor must be callable")
    if selected_runner is None and predictions is None:
        raise NliNegationChallengeError(
            "provide precomputed NLI predictions or a local predictor"
        )
    if selected_runner is not None and predictions is not None:
        raise NliNegationChallengeError(
            "provide predictions or a local NLI predictor, not both"
        )

    maximum_false_entailment_rate = _validate_rate(
        max_false_entailment_rate,
        "false-entailment rate ceiling",
    )
    minimum_accuracy = _validate_optional_rate(
        minimum_aggregate_accuracy,
        "aggregate accuracy floor",
    )
    resolved_cases = _coerce_cases(cases)
    resolved_predictions = _resolve_predictions(
        resolved_cases,
        predictions,
        selected_runner,
    )

    counts = {
        pattern: {
            "case_count": 0,
            "correct_count": 0,
            "abstention_count": 0,
            "non_entailment_case_count": 0,
            "false_entailment_count": 0,
        }
        for pattern in NEGATION_PATTERNS
    }
    correct_count = 0
    abstention_count = 0
    false_entailment_count = 0
    false_entailment_case_count = 0
    for case, prediction in zip(resolved_cases, resolved_predictions):
        pattern_counts = counts[case.pattern]
        pattern_counts["case_count"] += 1
        if prediction == case.gold_label:
            correct_count += 1
            pattern_counts["correct_count"] += 1
        if prediction == "abstention":
            abstention_count += 1
            pattern_counts["abstention_count"] += 1
        if case.requires_non_entailment:
            false_entailment_case_count += 1
            pattern_counts["non_entailment_case_count"] += 1
            if prediction == "entailment":
                false_entailment_count += 1
                pattern_counts["false_entailment_count"] += 1

    case_count = len(resolved_cases)
    aggregate_accuracy = _rate(correct_count, case_count)
    false_entailment_rate = _rate(
        false_entailment_count,
        false_entailment_case_count,
    )
    aggregate_accuracy_gate_passed = (
        minimum_accuracy is None or aggregate_accuracy >= minimum_accuracy
    )
    false_entailment_gate_passed = (
        false_entailment_rate <= maximum_false_entailment_rate
    )
    gate = NliNegationGateResult(
        aggregate_accuracy=aggregate_accuracy,
        minimum_aggregate_accuracy=minimum_accuracy,
        aggregate_accuracy_gate_passed=aggregate_accuracy_gate_passed,
        false_entailment_rate=false_entailment_rate,
        max_false_entailment_rate=maximum_false_entailment_rate,
        false_entailment_gate_passed=false_entailment_gate_passed,
        passed=aggregate_accuracy_gate_passed and false_entailment_gate_passed,
    )
    report = NliNegationReport(
        fixture_set_hash=_fixture_set_hash(resolved_cases),
        prediction_hash=_prediction_hash(resolved_cases, resolved_predictions),
        case_count=case_count,
        correct_count=correct_count,
        abstention_count=abstention_count,
        aggregate_accuracy=aggregate_accuracy,
        false_entailment_count=false_entailment_count,
        false_entailment_case_count=false_entailment_case_count,
        false_entailment_rate=false_entailment_rate,
        by_pattern={
            pattern: NliNegationPatternMetrics(
                pattern=pattern,
                **counts[pattern],
            )
            for pattern in NEGATION_PATTERNS
        },
        gate=gate,
    )
    if output_json is not None:
        report.write_json(output_json)
    if output_markdown is not None:
        report.write_markdown(output_markdown)
    return report


evaluate_nli_negation_challenge = run_nli_negation_challenge


def check_false_entailment_gate(
    report: NliNegationReport,
    *,
    max_false_entailment_rate: float | None = None,
) -> bool:
    """Return only the dedicated false-entailment gate decision.

    This helper deliberately ignores aggregate accuracy so a caller can retain
    the two release signals independently.
    """

    _require_report(report)
    ceiling = (
        report.gate.max_false_entailment_rate
        if max_false_entailment_rate is None
        else _validate_rate(
            max_false_entailment_rate,
            "false-entailment rate ceiling",
        )
    )
    return report.false_entailment_rate <= ceiling


def assert_false_entailment_gate(
    report: NliNegationReport,
    *,
    max_false_entailment_rate: float | None = None,
) -> NliNegationReport:
    """Return *report* or raise if the dedicated false-entailment gate fails."""

    if not check_false_entailment_gate(
        report,
        max_false_entailment_rate=max_false_entailment_rate,
    ):
        raise NliNegationGateError("clinical NLI false-entailment gate failed")
    return report


def assert_nli_negation_gate(
    report: NliNegationReport,
    *,
    max_false_entailment_rate: float | None = None,
    minimum_aggregate_accuracy: float | None = None,
) -> NliNegationReport:
    """Return *report* or raise when either configured gate fails."""

    _require_report(report)
    false_entailment_passed = check_false_entailment_gate(
        report,
        max_false_entailment_rate=max_false_entailment_rate,
    )
    accuracy_floor = (
        report.gate.minimum_aggregate_accuracy
        if minimum_aggregate_accuracy is None
        else _validate_rate(
            minimum_aggregate_accuracy,
            "aggregate accuracy floor",
        )
    )
    accuracy_passed = (
        accuracy_floor is None or report.aggregate_accuracy >= accuracy_floor
    )
    if not false_entailment_passed or not accuracy_passed:
        raise NliNegationGateError("clinical NLI negation gate failed")
    return report


def render_nli_negation_report_json(
    report: NliNegationReport,
    *,
    indent: int = 2,
) -> str:
    """Render a validated report as deterministic JSON."""

    _require_report(report)
    return report.to_json(indent=indent)


def render_nli_negation_report_markdown(report: NliNegationReport) -> str:
    """Render a validated report as aggregate-only Markdown."""

    _require_report(report)
    return report.to_markdown()


def write_nli_negation_report(
    report: NliNegationReport,
    path: str | Path,
    *,
    format: str = "json",
    indent: int = 2,
) -> Path:
    """Write a JSON or Markdown aggregate report to a local path."""

    _require_report(report)
    if format == "json":
        return report.write_json(path, indent=indent)
    if format in {"markdown", "md"}:
        return report.write_markdown(path)
    raise NliNegationChallengeError("report format must be json or markdown")


def _coerce_cases(
    cases: Iterable[NliNegationCase | Mapping[str, Any]] | None,
) -> tuple[NliNegationCase, ...]:
    if cases is None:
        resolved = list(default_nli_negation_cases())
    elif isinstance(cases, NliNegationCase):
        resolved = [cases]
    else:
        try:
            resolved = [
                case
                if isinstance(case, NliNegationCase)
                else NliNegationCase.from_mapping(case)
                for case in cases
            ]
        except (TypeError, ValueError, NliNegationChallengeError) as exc:
            if isinstance(exc, NliNegationChallengeError):
                raise
            raise NliNegationChallengeError("negation cases must be iterable") from None
    if not resolved:
        raise NliNegationChallengeError("negation challenge requires at least one case")
    if len(resolved) > MAX_NEGATION_CASES:
        raise NliNegationChallengeError("negation challenge exceeds the case limit")
    case_ids = [case.case_id for case in resolved]
    if len(case_ids) != len(set(case_ids)):
        raise NliNegationChallengeError("negation case identifiers must be unique")
    return tuple(resolved)


def _resolve_predictions(
    cases: Sequence[NliNegationCase],
    predictions: NliPredictionInput | None,
    runner: NliPredictor | None,
) -> tuple[str, ...]:
    if runner is not None:
        resolved: list[str] = []
        for case in cases:
            try:
                output = _invoke_runner(runner, case)
                resolved.append(_normalize_prediction(output))
            except Exception as exc:
                del exc
                raise NliNegationChallengeError("local NLI predictor failed") from None
        return tuple(resolved)

    if isinstance(predictions, Mapping):
        expected_ids = {case.case_id for case in cases}
        if set(predictions) != expected_ids:
            raise NliNegationChallengeError(
                "prediction mapping must contain exactly the challenge cases"
            )
        try:
            return tuple(
                _normalize_prediction(predictions[case.case_id]) for case in cases
            )
        except NliNegationChallengeError:
            raise
        except Exception as exc:
            del exc
            raise NliNegationChallengeError("prediction mapping is invalid") from None

    try:
        values = list(predictions) if predictions is not None else []
    except Exception as exc:
        del exc
        raise NliNegationChallengeError("predictions must be iterable") from None
    if len(values) != len(cases):
        raise NliNegationChallengeError(
            "prediction count must match the challenge case count"
        )
    return tuple(_normalize_prediction(value) for value in values)


def _invoke_runner(runner: NliPredictor, case: NliNegationCase) -> Any:
    """Invoke a local predictor without exposing its exception details."""

    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return runner(case.premise, case.hypothesis)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        parameter.kind == parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if not has_varargs and len(positional) <= 1:
        return runner(case)
    return runner(case.premise, case.hypothesis)


def _normalize_prediction(value: Any) -> str:
    if value is None:
        return "abstention"
    if isinstance(value, Mapping):
        if "label" in value:
            value = value["label"]
        elif "prediction" in value:
            value = value["prediction"]
        else:
            raise NliNegationChallengeError("NLI prediction objects require a label")
    elif not isinstance(value, str) and hasattr(value, "label"):
        value = getattr(value, "label")
    return _normalize_label(value, allow_abstention=True)


def _normalize_label(value: Any, *, allow_abstention: bool) -> str:
    if not isinstance(value, str):
        raise NliNegationChallengeError("NLI labels must be strings")
    normalized = " ".join(value.strip().lower().split())
    label = _LABEL_ALIASES.get(normalized)
    if label is None or (label == "abstention" and not allow_abstention):
        raise NliNegationChallengeError("unsupported NLI label")
    return label


def _normalize_pattern(value: Any) -> str:
    if not isinstance(value, str):
        raise NliNegationChallengeError("negation pattern must be a string")
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in NEGATION_PATTERNS:
        raise NliNegationChallengeError("unsupported negation pattern")
    return normalized


def _normalize_identifier(value: Any, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str):
        raise NliNegationChallengeError("negation identifiers must be strings")
    normalized = value.strip().lower()
    if (
        not normalized
        or len(normalized) > MAX_CASE_ID_LENGTH
        or not pattern.fullmatch(normalized)
    ):
        raise NliNegationChallengeError("negation identifier is invalid")
    return normalized


def _normalize_optional_identifier(
    value: Any,
    pattern: re.Pattern[str],
) -> str | None:
    if value is None:
        return None
    return _normalize_identifier(value, pattern)


def _require_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NliNegationChallengeError("negation premise and hypothesis are required")
    if len(value) > MAX_CASE_TEXT_LENGTH:
        raise NliNegationChallengeError("negation case text exceeds the safe limit")
    return value


def _first_present(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    *keys: str,
) -> Any:
    for source in (primary, secondary):
        for key in keys:
            if key in source:
                return source[key]
    return None


def _validate_rate(value: Any, field_name: str) -> float:
    del field_name
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise NliNegationChallengeError("gate thresholds must be finite numbers")
    if not 0 <= value <= 1:
        raise NliNegationChallengeError("gate thresholds must be between zero and one")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise NliNegationChallengeError("gate thresholds must be between zero and one")
    return normalized


def _validate_optional_rate(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _validate_rate(value, field_name)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _fixture_set_hash(cases: Sequence[NliNegationCase]) -> str:
    canonical = [
        case.to_mapping()
        for case in sorted(cases, key=lambda item: (item.pattern, item.case_id))
    ]
    return _digest(
        {
            "schema_version": NLI_NEGATION_FIXTURE_SCHEMA_VERSION,
            "cases": canonical,
        }
    )


def _prediction_hash(
    cases: Sequence[NliNegationCase],
    predictions: Sequence[str],
) -> str:
    canonical = sorted(
        zip(
            (case.case_id for case in cases),
            predictions,
        ),
        key=lambda item: item[0],
    )
    return _digest(
        {
            "challenge": NLI_NEGATION_CHALLENGE,
            "predictions": [
                {"case_id": case_id, "label": label} for case_id, label in canonical
            ],
        }
    )


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _format_rate(value: float) -> str:
    return f"{value:.6f}"


def _require_report(report: Any) -> None:
    if not isinstance(report, NliNegationReport):
        raise NliNegationChallengeError("expected a clinical NLI negation report")


DEFAULT_NLI_NEGATION_CASES: tuple[NliNegationCase, ...] = (
    NliNegationCase(
        case_id="simple-contradiction",
        pattern="simple",
        premise="Condition alpha is absent.",
        hypothesis="Condition alpha is present.",
        gold_label="contradiction",
        negation_depth=1,
    ),
    NliNegationCase(
        case_id="simple-entailment",
        pattern="simple",
        premise="Condition alpha is absent.",
        hypothesis="Condition alpha is absent.",
        gold_label="entailment",
        negation_depth=1,
    ),
    NliNegationCase(
        case_id="nested-neutral",
        pattern="nested",
        premise="There is no indication that condition beta is not present.",
        hypothesis="Condition beta is present.",
        gold_label="neutral",
        negation_depth=2,
    ),
    NliNegationCase(
        case_id="double-contradiction",
        pattern="double",
        premise="Condition gamma is not absent.",
        hypothesis="Condition gamma is absent.",
        gold_label="contradiction",
        negation_depth=2,
    ),
    NliNegationCase(
        case_id="double-entailment",
        pattern="double",
        premise="Condition gamma is not absent.",
        hypothesis="Condition gamma is present.",
        gold_label="entailment",
        negation_depth=2,
    ),
    NliNegationCase(
        case_id="section-scoped-contradiction",
        pattern="section_scoped",
        premise=(
            "NEGATIVE FINDINGS:\n"
            "Condition delta is absent.\n"
            "ACTIVE PROBLEMS:\n"
            "Condition epsilon is present."
        ),
        hypothesis="Condition delta is present.",
        gold_label="contradiction",
        negation_depth=1,
        section="negative_findings",
    ),
    NliNegationCase(
        case_id="section-scoped-entailment",
        pattern="section_scoped",
        premise=(
            "NEGATIVE FINDINGS:\n"
            "Condition delta is absent.\n"
            "ACTIVE PROBLEMS:\n"
            "Condition epsilon is present."
        ),
        hypothesis="Condition epsilon is present.",
        gold_label="entailment",
        negation_depth=1,
        section="negative_findings",
    ),
)


__all__ = [
    "DEFAULT_MAX_FALSE_ENTAILMENT_RATE",
    "DEFAULT_NLI_NEGATION_CASES",
    "MAX_CASE_ID_LENGTH",
    "MAX_CASE_TEXT_LENGTH",
    "MAX_NEGATION_CASES",
    "NEGATION_PATTERNS",
    "NLI_LABELS",
    "NLI_NEGATION_CHALLENGE",
    "NLI_NEGATION_CHALLENGE_SCHEMA_VERSION",
    "NLI_NEGATION_FIXTURE_SCHEMA_VERSION",
    "NliNegationCase",
    "NliNegationChallengeError",
    "NliNegationFixture",
    "NliNegationGateError",
    "NliNegationGateResult",
    "NliNegationPair",
    "NliNegationPatternMetrics",
    "NliNegationReport",
    "NliPredictor",
    "assert_false_entailment_gate",
    "assert_nli_negation_gate",
    "build_nli_negation_challenge",
    "check_false_entailment_gate",
    "default_nli_negation_cases",
    "evaluate_nli_negation_challenge",
    "load_nli_negation_fixtures",
    "render_nli_negation_report_json",
    "render_nli_negation_report_markdown",
    "run_nli_negation_challenge",
    "write_nli_negation_report",
]
