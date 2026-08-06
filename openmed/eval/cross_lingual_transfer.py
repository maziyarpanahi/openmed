"""Offline cross-lingual transfer evaluation over language fixtures.

The evaluator runs the same target-language fixtures once for every
source-language calibration context.  It therefore exposes both the full
source-to-target matrix and the loss in target performance relative to the
source-language baseline.  Reports contain aggregate counts, rates, labels,
and hashes-safe metadata only; fixture text and predictions are never
serialized.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.quality_gates import validate_entity_spans
from openmed.eval.harness import BenchmarkFixture, ModelRunner, default_model_runner
from openmed.eval.metrics import (
    EvalSpan,
    compute_character_recall,
    compute_leakage_rate,
    compute_relaxed_span_f1,
    normalize_eval_spans,
)

CROSS_LINGUAL_TRANSFER_SCHEMA_VERSION = 1
CROSS_LINGUAL_TRANSFER_ARTIFACT_TYPE = "openmed.eval.cross_lingual_transfer"
DEFAULT_CROSS_LINGUAL_TRANSFER_SUITE = "cross-lingual-transfer"


@dataclass(frozen=True, slots=True)
class LabelTransferMetrics:
    """Aggregate leakage, recall, and relaxed F1 for one entity label."""

    label: str
    leakage: float
    recall: float
    f1: float
    leaked_chars: int
    covered_chars: int
    total_chars: int
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def leakage_rate(self) -> float:
        """Return the leakage-rate alias used by other eval reports."""

        return self.leakage

    def to_dict(self) -> dict[str, float | int | str]:
        """Return aggregate label metrics without fixture content."""

        return {
            "covered_chars": self.covered_chars,
            "f1": self.f1,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "label": self.label,
            "leakage": self.leakage,
            "leakage_rate": self.leakage,
            "recall": self.recall,
            "total_chars": self.total_chars,
            "true_positives": self.true_positives,
            "leaked_chars": self.leaked_chars,
        }

    def __getitem__(self, key: str) -> float | int | str:
        """Allow dictionary-style access used by eval consumers."""

        return self.to_dict()[key]


@dataclass(frozen=True, slots=True)
class TransferMatrixCell:
    """Metrics for one source-language to target-language matrix cell."""

    source_language: str
    target_language: str
    leakage: float
    recall: float
    f1: float
    fixture_count: int
    leaked_chars: int
    covered_chars: int
    total_chars: int
    by_label: Mapping[str, LabelTransferMetrics] = field(default_factory=dict)
    zero_shot: bool = False

    @property
    def leakage_rate(self) -> float:
        """Return the leakage-rate alias used by benchmark reports."""

        return self.leakage

    @property
    def held_out(self) -> bool:
        """Return whether this cell evaluates a held-out target language."""

        return self.zero_shot

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, aggregate-only cell payload."""

        return {
            "by_label": {
                label: self.by_label[label].to_dict() for label in sorted(self.by_label)
            },
            "covered_chars": self.covered_chars,
            "f1": self.f1,
            "fixture_count": self.fixture_count,
            "leakage": self.leakage,
            "leakage_rate": self.leakage,
            "leaked_chars": self.leaked_chars,
            "recall": self.recall,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "total_chars": self.total_chars,
            "zero_shot": self.zero_shot,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow field and label lookup on a matrix cell."""

        payload = self.to_dict()
        if key in payload:
            return payload[key]
        return self.by_label[key]


@dataclass(frozen=True, slots=True)
class LabelTransferGap:
    """Signed target-minus-source performance gap for one label."""

    label: str
    source_language: str
    target_language: str
    source_performance: float
    target_performance: float
    gap: float
    source_recall: float
    target_recall: float
    source_leakage: float
    target_leakage: float
    leakage_gap: float
    source_f1: float
    target_f1: float
    f1_gap: float

    @property
    def transfer_gap(self) -> float:
        """Return the signed target-minus-source recall gap."""

        return self.gap

    def to_dict(self) -> dict[str, float | str]:
        """Return a deterministic label-level gap payload."""

        return {
            "f1_gap": self.f1_gap,
            "gap": self.gap,
            "label": self.label,
            "leakage_gap": self.leakage_gap,
            "source_f1": self.source_f1,
            "source_language": self.source_language,
            "source_leakage": self.source_leakage,
            "source_performance": self.source_performance,
            "source_recall": self.source_recall,
            "target_f1": self.target_f1,
            "target_language": self.target_language,
            "target_leakage": self.target_leakage,
            "target_performance": self.target_performance,
            "target_recall": self.target_recall,
            "transfer_gap": self.gap,
        }

    def __getitem__(self, key: str) -> float | str:
        """Allow dictionary-style access to a label gap."""

        return self.to_dict()[key]


@dataclass(frozen=True, slots=True)
class TransferGapMetrics:
    """Signed transfer gaps for one source and target language pair."""

    source_language: str
    target_language: str
    source_performance: float
    target_performance: float
    gap: float
    source_recall: float
    target_recall: float
    leakage_gap: float
    source_leakage: float
    target_leakage: float
    f1_gap: float
    source_f1: float
    target_f1: float
    fixture_count: int
    by_label: Mapping[str, LabelTransferGap] = field(default_factory=dict)

    @property
    def transfer_gap(self) -> float:
        """Return the signed target-minus-source recall gap."""

        return self.gap

    @property
    def recall_gap(self) -> float:
        """Return the signed recall gap."""

        return self.gap

    @property
    def gap_by_label(self) -> Mapping[str, LabelTransferGap]:
        """Return label-level gaps under a descriptive alias."""

        return self.by_label

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic gap payload without raw fixture data."""

        return {
            "by_label": {
                label: self.by_label[label].to_dict() for label in sorted(self.by_label)
            },
            "f1_gap": self.f1_gap,
            "gap": self.gap,
            "leakage_gap": self.leakage_gap,
            "fixture_count": self.fixture_count,
            "source_f1": self.source_f1,
            "source_language": self.source_language,
            "source_leakage": self.source_leakage,
            "source_performance": self.source_performance,
            "source_recall": self.source_recall,
            "target_f1": self.target_f1,
            "target_language": self.target_language,
            "target_leakage": self.target_leakage,
            "target_performance": self.target_performance,
            "target_recall": self.target_recall,
            "transfer_gap": self.gap,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to gap fields."""

        return self.to_dict()[key]


@dataclass(frozen=True, slots=True)
class LanguageTransferCandidate:
    """A target language ranked as a candidate for dedicated training."""

    rank: int
    target_language: str
    transfer_gap: float
    worst_transfer_gap: float
    deficit: float
    source_count: int
    by_label: Mapping[str, float] = field(default_factory=dict)

    @property
    def language(self) -> str:
        """Return the candidate language code."""

        return self.target_language

    @property
    def average_transfer_gap(self) -> float:
        """Return the mean signed target-minus-source recall gap."""

        return self.transfer_gap

    @property
    def average_gap(self) -> float:
        """Return the mean signed target-minus-source recall gap."""

        return self.transfer_gap

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic candidate-ranking row."""

        return {
            "average_gap": self.transfer_gap,
            "average_transfer_gap": self.transfer_gap,
            "by_label": {
                label: self.by_label[label] for label in sorted(self.by_label)
            },
            "deficit": self.deficit,
            "language": self.target_language,
            "rank": self.rank,
            "source_count": self.source_count,
            "target_language": self.target_language,
            "transfer_gap": self.transfer_gap,
            "worst_transfer_gap": self.worst_transfer_gap,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to ranking rows."""

        return self.to_dict()[key]


@dataclass(frozen=True, slots=True)
class CrossLingualTransferReport:
    """Full source-to-target transfer matrix and training-candidate ranking."""

    suite: str
    model_name: str
    device: str
    languages: tuple[str, ...]
    labels: tuple[str, ...]
    fixture_count: int
    matrix: Mapping[str, Mapping[str, TransferMatrixCell]]
    transfer_gaps: Mapping[str, Mapping[str, TransferGapMetrics]]
    ranked_candidates: tuple[LanguageTransferCandidate, ...]
    provenance: str = "synthetic-offline"
    schema_version: int = CROSS_LINGUAL_TRANSFER_SCHEMA_VERSION

    @property
    def full_matrix(self) -> Mapping[str, Mapping[str, TransferMatrixCell]]:
        """Return the complete source-to-target matrix."""

        return self.matrix

    @property
    def transfer_matrix(self) -> Mapping[str, Mapping[str, TransferMatrixCell]]:
        """Return the complete source-to-target matrix under an explicit alias."""

        return self.matrix

    @property
    def gaps(self) -> Mapping[str, Mapping[str, TransferGapMetrics]]:
        """Return source-target gap metrics under a concise alias."""

        return self.transfer_gaps

    @property
    def training_candidates(self) -> tuple[LanguageTransferCandidate, ...]:
        """Return languages ranked for possible dedicated training."""

        return self.ranked_candidates

    @property
    def ranking(self) -> tuple[LanguageTransferCandidate, ...]:
        """Return the deterministic candidate ranking."""

        return self.ranked_candidates

    @property
    def ranked_languages(self) -> tuple[str, ...]:
        """Return ranked language codes without their evidence rows."""

        return tuple(candidate.target_language for candidate in self.ranked_candidates)

    @property
    def candidate_languages(self) -> tuple[str, ...]:
        """Return ranked language codes without their evidence rows."""

        return self.ranked_languages

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, PHI-free report payload."""

        return {
            "artifact_type": CROSS_LINGUAL_TRANSFER_ARTIFACT_TYPE,
            "device": self.device,
            "fixture_count": self.fixture_count,
            "labels": list(self.labels),
            "languages": list(self.languages),
            "matrix": {
                source: {
                    target: self.matrix[source][target].to_dict()
                    for target in self.languages
                }
                for source in self.languages
            },
            "model_name": self.model_name,
            "provenance": self.provenance,
            "ranked_candidates": [
                candidate.to_dict() for candidate in self.ranked_candidates
            ],
            "schema_version": self.schema_version,
            "suite": self.suite,
            "transfer_gaps": {
                source: {
                    target: self.transfer_gaps[source][target].to_dict()
                    for target in self.languages
                }
                for source in self.languages
            },
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report deterministically as JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON evidence to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render the matrix and candidate ranking without fixture content."""

        lines = [
            "# Cross-Lingual Transfer Evaluation",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Suite | `{_markdown_cell(self.suite)}` |",
            f"| Model | `{_markdown_cell(self.model_name)}` |",
            f"| Device | `{_markdown_cell(self.device)}` |",
            f"| Fixtures | {self.fixture_count} |",
            f"| Languages | {len(self.languages)} |",
            "",
            "## Source-to-target recall matrix",
            "",
            _markdown_row(("Source \\ Target", *self.languages)),
            _markdown_row(("---", *("---:" for _ in self.languages))),
        ]
        for source in self.languages:
            values = [f"`{_markdown_cell(source)}`"]
            values.extend(
                _format_cell(self.matrix[source][target]) for target in self.languages
            )
            lines.append(_markdown_row(values))

        lines.extend(
            [
                "",
                "## Transfer-gap ranking",
                "",
                "Signed gaps are target recall minus source-language recall; "
                "more negative gaps indicate larger training candidates.",
                "",
                _markdown_row(
                    (
                        "Rank",
                        "Target",
                        "Mean gap",
                        "Worst gap",
                        "Deficit",
                    )
                ),
                _markdown_row(("---:", "---", "---:", "---:", "---:")),
            ]
        )
        if self.ranked_candidates:
            for candidate in self.ranked_candidates:
                lines.append(
                    _markdown_row(
                        (
                            str(candidate.rank),
                            f"`{_markdown_cell(candidate.target_language)}`",
                            _format_rate(candidate.transfer_gap),
                            _format_rate(candidate.worst_transfer_gap),
                            _format_rate(candidate.deficit),
                        )
                    )
                )
        else:
            lines.append(_markdown_row(("0", "`none`", "0.000", "0.000", "0.000")))
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown evidence to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def model_card_evidence(self) -> dict[str, Any]:
        """Return aggregate transfer evidence suitable for a model card."""

        return {
            "artifact_type": CROSS_LINGUAL_TRANSFER_ARTIFACT_TYPE,
            "candidate_languages": list(self.ranked_languages),
            "ranked_candidates": [
                candidate.to_dict() for candidate in self.ranked_candidates
            ],
            "schema_version": self.schema_version,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to report fields."""

        return self.to_dict()[key]


@dataclass
class _Counts:
    """Mutable aggregate counters used while evaluating one matrix cell."""

    leaked_chars: int = 0
    covered_chars: int = 0
    total_chars: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def add(self, *, leakage: Any, recall: Any, f1: Any) -> None:
        """Add one fixture's aggregate metric counts."""

        self.leaked_chars += int(leakage.leaked_chars)
        self.covered_chars += int(recall.numerator)
        self.total_chars += int(leakage.total_chars)
        self.true_positives += int(f1.true_positives)
        self.false_positives += int(f1.false_positives)
        self.false_negatives += int(f1.false_negatives)


@dataclass
class _Aggregate:
    """Mutable aggregate for a source-target cell."""

    fixture_count: int = 0
    overall: _Counts = field(default_factory=_Counts)
    by_label: defaultdict[str, _Counts] = field(
        default_factory=lambda: defaultdict(_Counts)
    )

    def add(
        self,
        fixture: BenchmarkFixture,
        predicted_spans: Sequence[EvalSpan],
    ) -> None:
        """Add one target-language fixture to the aggregate."""

        leakage = compute_leakage_rate(
            fixture.gold_spans,
            predicted_spans,
            default_language=fixture.language,
            source_text=fixture.text,
        )
        recall = compute_character_recall(
            fixture.gold_spans,
            predicted_spans,
            default_language=fixture.language,
            source_text=fixture.text,
        )
        f1 = compute_relaxed_span_f1(
            fixture.gold_spans,
            predicted_spans,
            default_language=fixture.language,
            source_text=fixture.text,
        )
        self.fixture_count += 1
        self.overall.add(leakage=leakage, recall=recall, f1=f1)

        labels = sorted(
            {span.label for span in fixture.gold_spans}
            | {span.label for span in predicted_spans}
        )
        for label in labels:
            gold = [span for span in fixture.gold_spans if span.label == label]
            predicted = [span for span in predicted_spans if span.label == label]
            label_leakage = compute_leakage_rate(
                gold,
                predicted,
                default_language=fixture.language,
                source_text=fixture.text,
            )
            label_recall = compute_character_recall(
                gold,
                predicted,
                default_language=fixture.language,
                source_text=fixture.text,
            )
            label_f1 = compute_relaxed_span_f1(
                gold,
                predicted,
                default_language=fixture.language,
                source_text=fixture.text,
            )
            self.by_label[label].add(
                leakage=label_leakage,
                recall=label_recall,
                f1=label_f1,
            )

    def cell(
        self,
        *,
        source_language: str,
        target_language: str,
    ) -> TransferMatrixCell:
        """Build an immutable matrix cell from the aggregate counters."""

        return TransferMatrixCell(
            source_language=source_language,
            target_language=target_language,
            leakage=_leakage(self.overall),
            recall=_recall(self.overall),
            f1=_f1(self.overall),
            fixture_count=self.fixture_count,
            leaked_chars=self.overall.leaked_chars,
            covered_chars=self.overall.covered_chars,
            total_chars=self.overall.total_chars,
            by_label={
                label: _label_metrics(label, self.by_label[label])
                for label in sorted(self.by_label)
            },
            zero_shot=source_language != target_language,
        )


def cross_lingual_transfer_report(
    model: str | ModelRunner,
    fixtures: Sequence[BenchmarkFixture | Mapping[str, Any]],
    *,
    runner: ModelRunner | None = None,
    suite: str = DEFAULT_CROSS_LINGUAL_TRANSFER_SUITE,
    device: str = "cpu",
    languages: Sequence[str] | None = None,
) -> CrossLingualTransferReport:
    """Evaluate every source-language calibration against every target.

    The runner receives a copy of each target fixture with ``source_language``,
    ``target_language``, ``train_language``, ``eval_language``, and
    ``zero_shot`` metadata.  Diagonal cells are the in-language source
    baselines.  Off-diagonal cells are held-out target evaluations, and their
    signed transfer gaps are target recall minus the corresponding source
    diagonal recall, reported overall and per label.

    Args:
        model: Model identifier, or a model-runner callable.
        fixtures: Per-language benchmark fixtures or JSON-ready mappings.
        runner: Optional runner used when ``model`` is a string identifier.
        suite: Stable suite name included in the report.
        device: Device tag passed to the runner and span metrics.
        languages: Optional explicit language matrix order. Codes are
            normalized and sorted for deterministic output.

    Raises:
        ValueError: If no fixtures are supplied, fixture ids are duplicated,
            or a language code is empty.
    """

    resolved_fixtures = _coerce_fixtures(fixtures)
    all_languages = _resolve_languages(resolved_fixtures, languages)
    selected = tuple(
        fixture for fixture in resolved_fixtures if fixture.language in all_languages
    )
    if not selected:
        raise ValueError("cross-lingual transfer requires at least one fixture")

    model_name, model_runner = _resolve_model_runner(model, runner)
    fixtures_by_language: defaultdict[str, list[BenchmarkFixture]] = defaultdict(list)
    for fixture in selected:
        fixtures_by_language[fixture.language].append(fixture)
    for target in fixtures_by_language:
        fixtures_by_language[target].sort(key=lambda fixture: fixture.fixture_id)

    matrix: dict[str, dict[str, TransferMatrixCell]] = {}
    for source_language in all_languages:
        row: dict[str, TransferMatrixCell] = {}
        for target_language in all_languages:
            aggregate = _evaluate_cell(
                fixtures_by_language.get(target_language, ()),
                source_language=source_language,
                target_language=target_language,
                model_name=model_name,
                model_runner=model_runner,
                device=device,
            )
            row[target_language] = aggregate.cell(
                source_language=source_language,
                target_language=target_language,
            )
        matrix[source_language] = row

    transfer_gaps: dict[str, dict[str, TransferGapMetrics]] = {}
    for source_language in all_languages:
        source_baseline = matrix[source_language][source_language]
        transfer_gaps[source_language] = {}
        for target_language in all_languages:
            transfer_gaps[source_language][target_language] = _gap_metrics(
                source_baseline,
                matrix[source_language][target_language],
            )

    labels = tuple(
        sorted(
            {
                label
                for row in matrix.values()
                for cell in row.values()
                for label in cell.by_label
            }
        )
    )
    ranked_candidates = _rank_candidates(all_languages, transfer_gaps, labels)
    return CrossLingualTransferReport(
        suite=str(suite),
        model_name=model_name,
        device=str(device),
        languages=all_languages,
        labels=labels,
        fixture_count=len(selected),
        matrix=matrix,
        transfer_gaps=transfer_gaps,
        ranked_candidates=ranked_candidates,
    )


def evaluate_cross_lingual_transfer(
    model: str | ModelRunner,
    fixtures: Sequence[BenchmarkFixture | Mapping[str, Any]],
    **kwargs: Any,
) -> CrossLingualTransferReport:
    """Alias for :func:`cross_lingual_transfer_report`."""

    return cross_lingual_transfer_report(model, fixtures, **kwargs)


def run_cross_lingual_transfer_eval(
    model: str | ModelRunner,
    fixtures: Sequence[BenchmarkFixture | Mapping[str, Any]],
    **kwargs: Any,
) -> CrossLingualTransferReport:
    """Run the offline transfer evaluator with a descriptive name."""

    return cross_lingual_transfer_report(model, fixtures, **kwargs)


def _coerce_fixtures(
    fixtures: Sequence[BenchmarkFixture | Mapping[str, Any]],
) -> tuple[BenchmarkFixture, ...]:
    if not fixtures:
        raise ValueError("cross-lingual transfer requires at least one fixture")
    resolved: list[BenchmarkFixture] = []
    seen_ids: set[str] = set()
    for item in fixtures:
        fixture = (
            item
            if isinstance(item, BenchmarkFixture)
            else BenchmarkFixture.from_mapping(item)
        )
        fixture_id = str(fixture.fixture_id).strip()
        if not fixture_id:
            raise ValueError("cross-lingual transfer fixtures require an id")
        if fixture_id in seen_ids:
            raise ValueError(
                f"duplicate cross-lingual transfer fixture id: {fixture_id}"
            )
        seen_ids.add(fixture_id)
        language = _normalize_language(fixture.language)
        resolved.append(replace(fixture, fixture_id=fixture_id, language=language))
    return tuple(resolved)


def _resolve_languages(
    fixtures: Sequence[BenchmarkFixture],
    languages: Sequence[str] | None,
) -> tuple[str, ...]:
    present = {_normalize_language(fixture.language) for fixture in fixtures}
    requested = (
        present
        if languages is None
        else {_normalize_language(language) for language in languages}
    )
    if not requested:
        raise ValueError("cross-lingual transfer requires at least one language")
    return tuple(sorted(requested))


def _normalize_language(language: Any) -> str:
    value = str(language).strip().lower().replace("_", "-")
    if not value:
        raise ValueError("cross-lingual transfer language codes cannot be empty")
    return value


def _resolve_model_runner(
    model: str | ModelRunner,
    runner: ModelRunner | None,
) -> tuple[str, ModelRunner]:
    if runner is not None:
        return str(model), runner
    if not isinstance(model, str) and callable(model):
        name = getattr(model, "__name__", model.__class__.__name__)
        return str(name), model
    return str(model), default_model_runner


def _evaluate_cell(
    fixtures: Sequence[BenchmarkFixture],
    *,
    source_language: str,
    target_language: str,
    model_name: str,
    model_runner: ModelRunner,
    device: str,
) -> _Aggregate:
    aggregate = _Aggregate()
    for fixture in fixtures:
        routed_fixture = _routed_fixture(
            fixture,
            source_language=source_language,
            target_language=target_language,
        )
        raw_predictions = list(model_runner(routed_fixture, model_name, device))
        predicted_spans = tuple(
            normalize_eval_spans(
                raw_predictions,
                default_language=target_language,
                default_device=device,
                source_text=fixture.text,
            )
        )
        validate_entity_spans(
            [span.to_entity() for span in predicted_spans],
            fixture.text,
        )
        aggregate.add(fixture, predicted_spans)
    return aggregate


def _routed_fixture(
    fixture: BenchmarkFixture,
    *,
    source_language: str,
    target_language: str,
) -> BenchmarkFixture:
    metadata = dict(fixture.metadata)
    zero_shot = source_language != target_language
    metadata.update(
        {
            "eval_language": target_language,
            "held_out_language": target_language if zero_shot else None,
            "source_language": source_language,
            "target_language": target_language,
            "calibration_language": source_language,
            "calibration_languages": [source_language],
            "train_language": source_language,
            "train_languages": [source_language],
            "training_language": source_language,
            "training_languages": [source_language],
            "zero_shot": zero_shot,
        }
    )
    return replace(fixture, language=target_language, metadata=metadata)


def _gap_metrics(
    source_baseline: TransferMatrixCell,
    target_cell: TransferMatrixCell,
) -> TransferGapMetrics:
    by_label: dict[str, LabelTransferGap] = {}
    for label in sorted(set(source_baseline.by_label) & set(target_cell.by_label)):
        source = source_baseline.by_label[label]
        target = target_cell.by_label[label]
        by_label[label] = LabelTransferGap(
            label=label,
            source_language=source_baseline.source_language,
            target_language=target_cell.target_language,
            source_performance=source.recall,
            target_performance=target.recall,
            gap=target.recall - source.recall,
            source_recall=source.recall,
            target_recall=target.recall,
            source_leakage=source.leakage,
            target_leakage=target.leakage,
            leakage_gap=target.leakage - source.leakage,
            source_f1=source.f1,
            target_f1=target.f1,
            f1_gap=target.f1 - source.f1,
        )
    return TransferGapMetrics(
        source_language=source_baseline.source_language,
        target_language=target_cell.target_language,
        source_performance=source_baseline.recall,
        target_performance=target_cell.recall,
        gap=target_cell.recall - source_baseline.recall,
        source_recall=source_baseline.recall,
        target_recall=target_cell.recall,
        leakage_gap=target_cell.leakage - source_baseline.leakage,
        source_leakage=source_baseline.leakage,
        target_leakage=target_cell.leakage,
        f1_gap=target_cell.f1 - source_baseline.f1,
        source_f1=source_baseline.f1,
        target_f1=target_cell.f1,
        fixture_count=target_cell.fixture_count,
        by_label=by_label,
    )


def _rank_candidates(
    languages: Sequence[str],
    transfer_gaps: Mapping[str, Mapping[str, TransferGapMetrics]],
    labels: Sequence[str],
) -> tuple[LanguageTransferCandidate, ...]:
    candidates: list[LanguageTransferCandidate] = []
    for target_language in languages:
        gaps = [
            transfer_gaps[source_language][target_language]
            for source_language in languages
            if source_language != target_language
        ]
        if not gaps:
            continue
        average_gap = sum(item.gap for item in gaps) / len(gaps)
        worst_gap = min(item.gap for item in gaps)
        if worst_gap >= 0.0:
            continue
        by_label = {
            label: sum(gap.by_label[label].gap for gap in gaps if label in gap.by_label)
            / sum(1 for gap in gaps if label in gap.by_label)
            for label in labels
            if any(label in gap.by_label for gap in gaps)
        }
        candidates.append(
            LanguageTransferCandidate(
                rank=0,
                target_language=target_language,
                transfer_gap=average_gap,
                worst_transfer_gap=worst_gap,
                deficit=-worst_gap,
                source_count=len(gaps),
                by_label=by_label,
            )
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            item.transfer_gap,
            item.worst_transfer_gap,
            item.target_language,
        ),
    )
    return tuple(replace(item, rank=index) for index, item in enumerate(ranked, 1))


def _label_metrics(label: str, counts: _Counts) -> LabelTransferMetrics:
    return LabelTransferMetrics(
        label=label,
        leakage=_leakage(counts),
        recall=_recall(counts),
        f1=_f1(counts),
        leaked_chars=counts.leaked_chars,
        covered_chars=counts.covered_chars,
        total_chars=counts.total_chars,
        true_positives=counts.true_positives,
        false_positives=counts.false_positives,
        false_negatives=counts.false_negatives,
    )


def _leakage(counts: _Counts) -> float:
    return _safe_rate(counts.leaked_chars, counts.total_chars, 0.0)


def _recall(counts: _Counts) -> float:
    return _safe_rate(counts.covered_chars, counts.total_chars, 1.0)


def _f1(counts: _Counts) -> float:
    predicted = counts.true_positives + counts.false_positives
    gold = counts.true_positives + counts.false_negatives
    precision = _safe_rate(counts.true_positives, predicted, 1.0)
    recall = _safe_rate(counts.true_positives, gold, 1.0)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _safe_rate(numerator: int, denominator: int, zero_denominator: float) -> float:
    if denominator == 0:
        return zero_denominator
    return numerator / denominator


def _format_cell(cell: TransferMatrixCell) -> str:
    return f"{_format_rate(cell.recall)} recall / {_format_rate(cell.leakage)} leak"


def _format_rate(value: float) -> str:
    return f"{float(value):.3f}"


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_row(values: Sequence[Any]) -> str:
    return "| " + " | ".join(_markdown_cell(value) for value in values) + " |"


__all__ = [
    "CROSS_LINGUAL_TRANSFER_ARTIFACT_TYPE",
    "CROSS_LINGUAL_TRANSFER_SCHEMA_VERSION",
    "DEFAULT_CROSS_LINGUAL_TRANSFER_SUITE",
    "CrossLingualTransferReport",
    "LanguageCandidate",
    "LabelTransferGap",
    "LabelTransferMetrics",
    "LanguageTransferCandidate",
    "TransferGapMetrics",
    "TransferMatrixCell",
    "TransferMatrixReport",
    "cross_lingual_transfer_report",
    "cross_lingual_transfer_eval",
    "evaluate_cross_lingual_transfer",
    "run_cross_lingual_transfer_eval",
]


LanguageCandidate = LanguageTransferCandidate
TransferMatrixReport = CrossLingualTransferReport
cross_lingual_transfer_eval = cross_lingual_transfer_report
