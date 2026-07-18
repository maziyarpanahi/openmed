"""Fairness and transfer metrics for de-identification leakage."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.core.quality_gates import validate_entity_spans
from openmed.eval.golden import load_benchmark_fixtures
from openmed.eval.harness import (
    BenchmarkFixture,
    ModelRunner,
    default_model_runner,
)
from openmed.eval.metrics import (
    EvalSpan,
    bootstrap_ci,
    compute_character_recall,
    compute_leakage_rate,
    normalize_eval_spans,
)
from openmed.eval.suites import GOLDEN, load_suite_fixtures, validate_suite_name

UNSPECIFIED_GROUP = "unspecified"
TRANSFER_MATRIX_SCHEMA_VERSION = 1
DEFAULT_ZERO_SHOT_LEAKAGE_FLOOR = 0.005

_GROUP_METADATA_KEYS = ("group", "demographic_group", "surrogate_group")


@dataclass(frozen=True)
class FairnessGroupMetrics:
    """Leakage and recall for one demographic surrogate group."""

    leakage_rate: float
    recall: float
    leaked_chars: int
    covered_chars: int
    total_chars: int
    span_count: int

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-ready mapping."""
        return {
            "leakage_rate": self.leakage_rate,
            "recall": self.recall,
            "leaked_chars": self.leaked_chars,
            "covered_chars": self.covered_chars,
            "total_chars": self.total_chars,
            "span_count": self.span_count,
        }

    def __getitem__(self, key: str) -> int | float:
        return self.to_dict()[key]


@dataclass(frozen=True)
class FairnessReport:
    """Fairness report over gold PHI groups for one model and suite."""

    suite: str
    model_name: str
    fixture_count: int
    per_group: dict[str, FairnessGroupMetrics]
    leakage_disparity: float
    worst_group_leakage: float
    worst_group: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping."""
        return {
            "suite": self.suite,
            "model_name": self.model_name,
            "fixture_count": self.fixture_count,
            "per_group": {
                group: metrics.to_dict()
                for group, metrics in sorted(self.per_group.items())
            },
            "leakage_disparity": self.leakage_disparity,
            "worst_group_leakage": self.worst_group_leakage,
            "worst_group": self.worst_group,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class TransferMatrixCell:
    """Leakage and recall for one source-language to target-language cell."""

    source_language: str
    target_language: str
    leakage_rate: float
    recall: float
    leaked_chars: int
    covered_chars: int
    total_chars: int
    fixture_count: int
    zero_shot: bool

    def to_dict(self) -> dict[str, bool | float | int | str]:
        """Return a PHI-free JSON-ready mapping."""
        return {
            "source_language": self.source_language,
            "target_language": self.target_language,
            "leakage_rate": self.leakage_rate,
            "recall": self.recall,
            "leaked_chars": self.leaked_chars,
            "covered_chars": self.covered_chars,
            "total_chars": self.total_chars,
            "fixture_count": self.fixture_count,
            "zero_shot": self.zero_shot,
        }

    def __getitem__(self, key: str) -> bool | float | int | str:
        return self.to_dict()[key]


@dataclass(frozen=True)
class TransferGapMetrics:
    """Transfer-gap statistic and bootstrap confidence interval for a language."""

    target_language: str
    in_language_leakage: float
    zero_shot_leakage: float
    transfer_gap: float
    confidence_interval: Mapping[str, Any]
    fixture_count: int
    zero_shot_source_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping."""
        return {
            "target_language": self.target_language,
            "in_language_leakage": self.in_language_leakage,
            "zero_shot_leakage": self.zero_shot_leakage,
            "transfer_gap": self.transfer_gap,
            "confidence_interval": _plain(self.confidence_interval),
            "fixture_count": self.fixture_count,
            "zero_shot_source_count": self.zero_shot_source_count,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class TransferDeficiency:
    """Worst zero-shot leakage-floor breach for one target language."""

    target_language: str
    source_language: str
    leakage_rate: float
    leakage_floor: float
    excess: float
    recall: float
    leaked_chars: int
    total_chars: int
    rank: int = 0

    def with_rank(self, rank: int) -> "TransferDeficiency":
        """Return a copy with a stable one-based report rank."""
        return replace(self, rank=rank)

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a PHI-free JSON-ready mapping."""
        return {
            "rank": self.rank,
            "target_language": self.target_language,
            "source_language": self.source_language,
            "leakage_rate": self.leakage_rate,
            "leakage_floor": self.leakage_floor,
            "excess": self.excess,
            "recall": self.recall,
            "leaked_chars": self.leaked_chars,
            "total_chars": self.total_chars,
        }

    def __getitem__(self, key: str) -> float | int | str:
        return self.to_dict()[key]


@dataclass(frozen=True)
class TransferMatrixReport:
    """Cross-lingual zero-shot transfer report over supported languages."""

    suite: str
    model_name: str
    device: str
    languages: tuple[str, ...]
    fixture_count: int
    matrix: Mapping[str, Mapping[str, TransferMatrixCell]]
    transfer_gaps: Mapping[str, TransferGapMetrics]
    deficiencies: tuple[TransferDeficiency, ...]
    leakage_floors: Mapping[str, float]
    ci_resamples: int
    ci_alpha: float
    ci_seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, PHI-free report payload."""
        languages = tuple(self.languages)
        return {
            "schema_version": TRANSFER_MATRIX_SCHEMA_VERSION,
            "artifact_type": "openmed.cross_lingual_transfer_matrix",
            "suite": self.suite,
            "model_name": self.model_name,
            "device": self.device,
            "fixture_count": self.fixture_count,
            "languages": list(languages),
            "ci": {
                "n_resamples": self.ci_resamples,
                "alpha": self.ci_alpha,
                "seed": self.ci_seed,
            },
            "leakage_floors": {
                language: float(self.leakage_floors.get(language, 0.0))
                for language in languages
            },
            "matrix": {
                source: {
                    target: self.matrix[source][target].to_dict()
                    for target in languages
                }
                for source in languages
            },
            "transfer_gaps": {
                language: self.transfer_gaps[language].to_dict()
                for language in languages
            },
            "deficiencies": [deficiency.to_dict() for deficiency in self.deficiencies],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the transfer report as deterministic JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render a byte-stable Markdown transfer-matrix report."""
        lines = [
            "# Cross-Lingual Transfer Matrix",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Suite | `{self.suite}` |",
            f"| Model | `{self.model_name}` |",
            f"| Device | `{self.device}` |",
            f"| Fixtures | {self.fixture_count} |",
            f"| Languages | {len(self.languages)} |",
            f"| Bootstrap Resamples | {self.ci_resamples} |",
            f"| Bootstrap Seed | {self.ci_seed} |",
            "",
            "## Leakage Matrix",
            "",
        ]
        header = ["Source \\ Target", *self.languages]
        lines.extend([_markdown_row(header), _markdown_row(["---"] * len(header))])
        for source in self.languages:
            lines.append(
                _markdown_row(
                    [
                        f"`{source}`",
                        *[
                            _format_cell(self.matrix[source][target])
                            for target in self.languages
                        ],
                    ]
                )
            )

        lines.extend(
            [
                "",
                "## Transfer Gaps",
                "",
                _markdown_row(
                    [
                        "Target",
                        "In-language",
                        "Zero-shot",
                        "Gap",
                        "95% CI",
                    ]
                ),
                _markdown_row(["---", "---:", "---:", "---:", "---"]),
            ]
        )
        for language in self.languages:
            gap = self.transfer_gaps[language]
            ci = gap.confidence_interval
            lines.append(
                _markdown_row(
                    [
                        f"`{language}`",
                        _format_float(gap.in_language_leakage),
                        _format_float(gap.zero_shot_leakage),
                        _format_float(gap.transfer_gap),
                        (
                            f"[{_format_float(ci.get('lower', 0.0))}, "
                            f"{_format_float(ci.get('upper', 0.0))}]"
                        ),
                    ]
                )
            )

        lines.extend(
            [
                "",
                "## Deficiencies",
                "",
                _markdown_row(
                    [
                        "Rank",
                        "Target",
                        "Source",
                        "Leakage",
                        "Floor",
                        "Excess",
                    ]
                ),
                _markdown_row(["---:", "---", "---", "---:", "---:", "---:"]),
            ]
        )
        if self.deficiencies:
            for deficiency in self.deficiencies:
                lines.append(
                    _markdown_row(
                        [
                            str(deficiency.rank),
                            f"`{deficiency.target_language}`",
                            f"`{deficiency.source_language}`",
                            _format_float(deficiency.leakage_rate),
                            _format_float(deficiency.leakage_floor),
                            _format_float(deficiency.excess),
                        ]
                    )
                )
        else:
            lines.append(
                _markdown_row(["0", "`none`", "`none`", "0.000", "0.000", "0.000"])
            )
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def model_card_evidence(self) -> dict[str, Any]:
        """Return compact aggregate evidence suitable for model cards."""
        return {
            "artifact_type": "openmed.cross_lingual_transfer_matrix",
            "schema_version": TRANSFER_MATRIX_SCHEMA_VERSION,
            "languages": list(self.languages),
            "deficiency_count": len(self.deficiencies),
            "deficiencies": [deficiency.to_dict() for deficiency in self.deficiencies],
            "transfer_gaps": {
                language: self.transfer_gaps[language].to_dict()
                for language in self.languages
            },
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass
class _GroupCounts:
    leaked_chars: int = 0
    covered_chars: int = 0
    total_chars: int = 0
    span_count: int = 0


@dataclass(frozen=True)
class _TransferDocumentCounts:
    fixture_id: str
    leaked_chars: int
    covered_chars: int
    total_chars: int


def fairness_report(
    model: str | ModelRunner,
    suite: str | Sequence[BenchmarkFixture | Mapping[str, Any]],
    *,
    runner: ModelRunner | None = None,
    device: str = "cpu",
    suite_kwargs: Mapping[str, Any] | None = None,
) -> FairnessReport:
    """Run a model and report leakage/recall by gold-span group.

    Args:
        model: Model identifier, or a runner callable with the harness
            ``ModelRunner`` signature.
        suite: Named suite such as ``"golden"``, or concrete benchmark
            fixtures/mappings.
        runner: Optional runner to use when ``model`` is a string identifier.
        device: Device tag passed to the model runner and prediction
            normalization.
        suite_kwargs: Optional keyword arguments for named suite loaders.

    Returns:
        A fairness report with per-group leakage, recall, leakage disparity,
        and worst-group leakage.
    """
    suite_name, fixtures = _load_fairness_fixtures(suite, suite_kwargs=suite_kwargs)
    model_name, model_runner = _resolve_model_runner(model, runner)
    counts: defaultdict[str, _GroupCounts] = defaultdict(_GroupCounts)

    for fixture in fixtures:
        predicted_spans = _predict_fixture(
            fixture,
            model_name=model_name,
            model_runner=model_runner,
            device=device,
        )
        for group, gold_spans in _gold_spans_by_group(fixture.gold_spans).items():
            leakage = compute_leakage_rate(
                gold_spans,
                predicted_spans,
                default_language=fixture.language,
                default_device=device,
                source_text=fixture.text,
            )
            recall = compute_character_recall(
                gold_spans,
                predicted_spans,
                default_language=fixture.language,
                default_device=device,
                source_text=fixture.text,
            )
            group_counts = counts[group]
            group_counts.leaked_chars += leakage.leaked_chars
            group_counts.covered_chars += int(recall.numerator)
            group_counts.total_chars += leakage.total_chars
            group_counts.span_count += len(gold_spans)

    per_group = {
        group: _group_metrics(group_counts)
        for group, group_counts in sorted(counts.items())
    }
    leakage_rates = [metrics.leakage_rate for metrics in per_group.values()]
    worst_group = _worst_group(per_group)
    worst_group_leakage = (
        per_group[worst_group].leakage_rate if worst_group is not None else 0.0
    )

    return FairnessReport(
        suite=suite_name,
        model_name=model_name,
        fixture_count=len(fixtures),
        per_group=per_group,
        leakage_disparity=(
            max(leakage_rates) - min(leakage_rates) if leakage_rates else 0.0
        ),
        worst_group_leakage=worst_group_leakage,
        worst_group=worst_group,
    )


def cross_lingual_transfer_report(
    model: str | ModelRunner,
    suite: str | Sequence[BenchmarkFixture | Mapping[str, Any]],
    *,
    runner: ModelRunner | None = None,
    device: str = "cpu",
    suite_kwargs: Mapping[str, Any] | None = None,
    languages: Sequence[str] | None = None,
    leakage_floors: Mapping[str, float] | None = None,
    ci_resamples: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
) -> TransferMatrixReport:
    """Evaluate cross-lingual zero-shot transfer across supported languages.

    The source language is exposed to the runner through fixture metadata as
    ``source_language`` and ``calibration_language``. The target language is the
    fixture language. Source-target pairs with different languages are treated
    as zero-shot transfer cells.
    """
    suite_name, fixtures = _load_fairness_fixtures(suite, suite_kwargs=suite_kwargs)
    resolved_languages = _resolve_languages(languages)
    floors = _resolve_leakage_floors(resolved_languages, leakage_floors)
    model_name, model_runner = _resolve_model_runner(model, runner)
    fixtures_by_language = _fixtures_by_language(fixtures)
    docs_by_pair: dict[tuple[str, str], list[_TransferDocumentCounts]] = {}
    matrix: dict[str, dict[str, TransferMatrixCell]] = {}

    for source_language in resolved_languages:
        row: dict[str, TransferMatrixCell] = {}
        for target_language in resolved_languages:
            documents = _evaluate_transfer_cell(
                fixtures_by_language.get(target_language, ()),
                source_language=source_language,
                target_language=target_language,
                model_name=model_name,
                model_runner=model_runner,
                device=device,
            )
            docs_by_pair[(source_language, target_language)] = documents
            row[target_language] = _cell_from_documents(
                source_language,
                target_language,
                documents,
            )
        matrix[source_language] = row

    transfer_gaps = {
        target_language: _transfer_gap(
            target_language,
            resolved_languages,
            docs_by_pair,
            ci_resamples=ci_resamples,
            ci_alpha=ci_alpha,
            ci_seed=ci_seed,
        )
        for target_language in resolved_languages
    }
    deficiencies = _rank_deficiencies(resolved_languages, matrix, floors)

    return TransferMatrixReport(
        suite=suite_name,
        model_name=model_name,
        device=device,
        languages=resolved_languages,
        fixture_count=len(fixtures),
        matrix=matrix,
        transfer_gaps=transfer_gaps,
        deficiencies=deficiencies,
        leakage_floors=floors,
        ci_resamples=ci_resamples,
        ci_alpha=ci_alpha,
        ci_seed=ci_seed,
    )


def _load_fairness_fixtures(
    suite: str | Sequence[BenchmarkFixture | Mapping[str, Any]],
    *,
    suite_kwargs: Mapping[str, Any] | None,
) -> tuple[str, list[BenchmarkFixture]]:
    if not isinstance(suite, str):
        return "custom", [_coerce_fixture(item) for item in suite]

    suite_name = validate_suite_name(suite)
    kwargs = dict(suite_kwargs or {})
    if suite_name == GOLDEN:
        return suite_name, load_benchmark_fixtures(**kwargs)
    return suite_name, load_suite_fixtures(suite_name, **kwargs)


def _coerce_fixture(
    fixture: BenchmarkFixture | Mapping[str, Any],
) -> BenchmarkFixture:
    if isinstance(fixture, BenchmarkFixture):
        return fixture
    return BenchmarkFixture.from_mapping(fixture)


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


def _predict_fixture(
    fixture: BenchmarkFixture,
    *,
    model_name: str,
    model_runner: ModelRunner,
    device: str,
) -> tuple[EvalSpan, ...]:
    raw_predictions = list(model_runner(fixture, model_name, device))
    predicted_spans = tuple(
        normalize_eval_spans(
            raw_predictions,
            default_language=fixture.language,
            default_device=device,
            source_text=fixture.text,
        )
    )
    validate_entity_spans(
        [span.to_entity() for span in predicted_spans],
        fixture.text,
    )
    return predicted_spans


def _gold_spans_by_group(spans: Sequence[EvalSpan]) -> dict[str, list[EvalSpan]]:
    grouped: defaultdict[str, list[EvalSpan]] = defaultdict(list)
    for span in spans:
        grouped[_group_for_span(span)].append(span)
    return dict(grouped)


def _group_for_span(span: EvalSpan) -> str:
    for key in _GROUP_METADATA_KEYS:
        value = span.metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return UNSPECIFIED_GROUP


def _group_metrics(counts: _GroupCounts) -> FairnessGroupMetrics:
    leakage_rate = _safe_rate(counts.leaked_chars, counts.total_chars, 0.0)
    recall = _safe_rate(counts.covered_chars, counts.total_chars, 1.0)
    return FairnessGroupMetrics(
        leakage_rate=leakage_rate,
        recall=recall,
        leaked_chars=counts.leaked_chars,
        covered_chars=counts.covered_chars,
        total_chars=counts.total_chars,
        span_count=counts.span_count,
    )


def _worst_group(
    per_group: Mapping[str, FairnessGroupMetrics],
) -> str | None:
    if not per_group:
        return None
    return max(per_group, key=lambda group: per_group[group].leakage_rate)


def _safe_rate(
    numerator: int | float,
    denominator: int | float,
    zero_denominator: float,
) -> float:
    if denominator == 0:
        return zero_denominator
    return float(numerator) / float(denominator)


def _resolve_languages(languages: Sequence[str] | None) -> tuple[str, ...]:
    source = languages if languages is not None else sorted(SUPPORTED_LANGUAGES)
    resolved = tuple(
        sorted({str(language).strip() for language in source if str(language).strip()})
    )
    if not resolved:
        raise ValueError("at least one transfer language is required")
    return resolved


def _resolve_leakage_floors(
    languages: Sequence[str],
    leakage_floors: Mapping[str, float] | None,
) -> dict[str, float]:
    floors = {language: DEFAULT_ZERO_SHOT_LEAKAGE_FLOOR for language in languages}
    for language, floor in (leakage_floors or {}).items():
        key = str(language).strip()
        if key in floors:
            floors[key] = float(floor)
    return floors


def _fixtures_by_language(
    fixtures: Sequence[BenchmarkFixture],
) -> dict[str, list[BenchmarkFixture]]:
    grouped: defaultdict[str, list[BenchmarkFixture]] = defaultdict(list)
    for fixture in fixtures:
        grouped[fixture.language].append(fixture)
    return {
        language: sorted(rows, key=lambda item: item.fixture_id)
        for language, rows in grouped.items()
    }


def _evaluate_transfer_cell(
    fixtures: Sequence[BenchmarkFixture],
    *,
    source_language: str,
    target_language: str,
    model_name: str,
    model_runner: ModelRunner,
    device: str,
) -> list[_TransferDocumentCounts]:
    documents: list[_TransferDocumentCounts] = []
    for fixture in fixtures:
        transfer_fixture = _transfer_fixture(
            fixture,
            source_language=source_language,
            target_language=target_language,
        )
        predicted_spans = _predict_fixture(
            transfer_fixture,
            model_name=model_name,
            model_runner=model_runner,
            device=device,
        )
        leakage = compute_leakage_rate(
            fixture.gold_spans,
            predicted_spans,
            default_language=target_language,
            default_device=device,
            source_text=fixture.text,
        )
        recall = compute_character_recall(
            fixture.gold_spans,
            predicted_spans,
            default_language=target_language,
            default_device=device,
            source_text=fixture.text,
        )
        documents.append(
            _TransferDocumentCounts(
                fixture_id=fixture.fixture_id,
                leaked_chars=leakage.leaked_chars,
                covered_chars=int(recall.numerator),
                total_chars=leakage.total_chars,
            )
        )
    return documents


def _transfer_fixture(
    fixture: BenchmarkFixture,
    *,
    source_language: str,
    target_language: str,
) -> BenchmarkFixture:
    metadata = dict(fixture.metadata)
    metadata.update(
        {
            "source_language": source_language,
            "target_language": target_language,
            "calibration_language": source_language,
            "calibration_languages": [source_language],
            "zero_shot": source_language != target_language,
        }
    )
    if source_language != target_language:
        metadata["held_out_language"] = target_language
    return replace(fixture, metadata=metadata)


def _cell_from_documents(
    source_language: str,
    target_language: str,
    documents: Sequence[_TransferDocumentCounts],
) -> TransferMatrixCell:
    leaked_chars = sum(document.leaked_chars for document in documents)
    covered_chars = sum(document.covered_chars for document in documents)
    total_chars = sum(document.total_chars for document in documents)
    return TransferMatrixCell(
        source_language=source_language,
        target_language=target_language,
        leakage_rate=_safe_rate(leaked_chars, total_chars, 0.0),
        recall=_safe_rate(covered_chars, total_chars, 1.0),
        leaked_chars=leaked_chars,
        covered_chars=covered_chars,
        total_chars=total_chars,
        fixture_count=len(documents),
        zero_shot=source_language != target_language,
    )


def _transfer_gap(
    target_language: str,
    languages: Sequence[str],
    docs_by_pair: Mapping[tuple[str, str], Sequence[_TransferDocumentCounts]],
    *,
    ci_resamples: int,
    ci_alpha: float,
    ci_seed: int,
) -> TransferGapMetrics:
    fixture_ids = sorted(
        {
            document.fixture_id
            for document in docs_by_pair.get((target_language, target_language), ())
        }
    )
    point = _transfer_gap_for_ids(
        fixture_ids,
        target_language,
        languages,
        docs_by_pair,
    )
    interval = bootstrap_ci(
        fixture_ids,
        lambda sample_ids: _transfer_gap_for_ids(
            sample_ids,
            target_language,
            languages,
            docs_by_pair,
        ),
        n_resamples=ci_resamples,
        alpha=ci_alpha,
        seed=_stable_target_seed(ci_seed, target_language),
    ).to_dict()
    interval["point"] = point
    return TransferGapMetrics(
        target_language=target_language,
        in_language_leakage=_leakage_for_documents(
            docs_by_pair.get((target_language, target_language), ())
        ),
        zero_shot_leakage=_zero_shot_leakage(
            target_language,
            languages,
            docs_by_pair,
        ),
        transfer_gap=point,
        confidence_interval=interval,
        fixture_count=len(fixture_ids),
        zero_shot_source_count=len(
            [language for language in languages if language != target_language]
        ),
    )


def _transfer_gap_for_ids(
    fixture_ids: Sequence[str],
    target_language: str,
    languages: Sequence[str],
    docs_by_pair: Mapping[tuple[str, str], Sequence[_TransferDocumentCounts]],
) -> float:
    in_language = _documents_for_ids(
        docs_by_pair.get((target_language, target_language), ()),
        fixture_ids,
    )
    zero_shot: list[_TransferDocumentCounts] = []
    for source_language in languages:
        if source_language == target_language:
            continue
        zero_shot.extend(
            _documents_for_ids(
                docs_by_pair.get((source_language, target_language), ()),
                fixture_ids,
            )
        )
    return _leakage_for_documents(zero_shot) - _leakage_for_documents(in_language)


def _documents_for_ids(
    documents: Sequence[_TransferDocumentCounts],
    fixture_ids: Sequence[str],
) -> list[_TransferDocumentCounts]:
    by_id = {document.fixture_id: document for document in documents}
    return [by_id[fixture_id] for fixture_id in fixture_ids if fixture_id in by_id]


def _zero_shot_leakage(
    target_language: str,
    languages: Sequence[str],
    docs_by_pair: Mapping[tuple[str, str], Sequence[_TransferDocumentCounts]],
) -> float:
    documents: list[_TransferDocumentCounts] = []
    for source_language in languages:
        if source_language == target_language:
            continue
        documents.extend(docs_by_pair.get((source_language, target_language), ()))
    return _leakage_for_documents(documents)


def _leakage_for_documents(
    documents: Sequence[_TransferDocumentCounts],
) -> float:
    leaked_chars = sum(document.leaked_chars for document in documents)
    total_chars = sum(document.total_chars for document in documents)
    return _safe_rate(leaked_chars, total_chars, 0.0)


def _stable_target_seed(seed: int, target_language: str) -> int:
    digest = hashlib.sha256(f"{seed}:{target_language}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _rank_deficiencies(
    languages: Sequence[str],
    matrix: Mapping[str, Mapping[str, TransferMatrixCell]],
    leakage_floors: Mapping[str, float],
) -> tuple[TransferDeficiency, ...]:
    deficiencies: list[TransferDeficiency] = []
    for target_language in languages:
        floor = float(
            leakage_floors.get(target_language, DEFAULT_ZERO_SHOT_LEAKAGE_FLOOR)
        )
        violations = [
            matrix[source_language][target_language]
            for source_language in languages
            if source_language != target_language
            and matrix[source_language][target_language].leakage_rate > floor
        ]
        if not violations:
            continue
        worst = sorted(
            violations,
            key=lambda cell: (
                -(cell.leakage_rate - floor),
                -cell.leakage_rate,
                cell.source_language,
            ),
        )[0]
        deficiencies.append(
            TransferDeficiency(
                target_language=target_language,
                source_language=worst.source_language,
                leakage_rate=worst.leakage_rate,
                leakage_floor=floor,
                excess=worst.leakage_rate - floor,
                recall=worst.recall,
                leaked_chars=worst.leaked_chars,
                total_chars=worst.total_chars,
            )
        )

    ranked = sorted(
        deficiencies,
        key=lambda item: (
            -item.excess,
            -item.leakage_rate,
            item.target_language,
            item.source_language,
        ),
    )
    return tuple(
        deficiency.with_rank(index + 1) for index, deficiency in enumerate(ranked)
    )


def _format_cell(cell: TransferMatrixCell) -> str:
    return f"{cell.leaked_chars}/{cell.total_chars} ({cell.leakage_rate:.3f})"


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _markdown_row(values: Sequence[str]) -> str:
    return "| " + " | ".join(values) + " |"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "DEFAULT_ZERO_SHOT_LEAKAGE_FLOOR",
    "TRANSFER_MATRIX_SCHEMA_VERSION",
    "UNSPECIFIED_GROUP",
    "FairnessGroupMetrics",
    "FairnessReport",
    "TransferDeficiency",
    "TransferGapMetrics",
    "TransferMatrixCell",
    "TransferMatrixReport",
    "cross_lingual_transfer_report",
    "fairness_report",
]
