"""Offline family-aware cross-lingual adapter evaluation.

The report compares an untargeted multilingual baseline, donor-adapter
zero-shot inference, and a target-adapted path on committed synthetic gold.
Only aggregate span metrics, language codes, and transfer deltas are emitted;
fixture text and predicted spans never enter report artifacts.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.quality_gates import validate_entity_spans
from openmed.eval.golden import load_benchmark_fixtures
from openmed.eval.harness import BenchmarkFixture, ModelRunner, default_model_runner
from openmed.eval.metrics import (
    EvalSpan,
    compute_relaxed_span_f1,
    normalize_eval_spans,
)
from openmed.training.adapters.config import (
    DEFAULT_FAMILY_TRANSFER_CONFIG,
    FamilyTransferConfig,
    TransferEdge,
    normalize_language_code,
)

FAMILY_TRANSFER_SCHEMA_VERSION = 1
FAMILY_TRANSFER_ARTIFACT_TYPE = "openmed.cross_lingual_family_transfer"
DEFAULT_ADAPTED_TARGET_F1_FLOOR = 0.80
FAMILY_TRANSFER_FIXTURES_PATH = (
    Path(__file__).resolve().parent
    / "golden"
    / "fixtures"
    / "family_transfer_gold.jsonl"
)

_BASELINE = "baseline"
_ZERO_SHOT = "zero_shot"
_ADAPTED = "adapted"
_DONOR = "donor"
_TARGET = "target"


@dataclass(frozen=True, slots=True)
class FamilyTransferModeMetrics:
    """Aggregate relaxed span metrics for one language and transfer mode."""

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    fixture_count: int

    def to_dict(self) -> dict[str, float | int]:
        """Return aggregate metrics without fixture text or spans."""

        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "fixture_count": self.fixture_count,
        }

    def __getitem__(self, key: str) -> float | int:
        return self.to_dict()[key]


@dataclass(frozen=True, slots=True)
class FamilyTransferComparison:
    """Baseline, zero-shot, adapted, and non-regression metrics for one edge."""

    family: str
    donor_language: str
    target_language: str
    donor_baseline: FamilyTransferModeMetrics
    donor_adapted: FamilyTransferModeMetrics
    target_baseline: FamilyTransferModeMetrics
    target_zero_shot: FamilyTransferModeMetrics
    target_adapted: FamilyTransferModeMetrics
    target_f1_floor: float
    donor_non_regression_tolerance: float

    @property
    def donor_delta(self) -> float:
        """Return post-adaptation donor F1 minus donor baseline F1."""

        return self.donor_adapted.f1 - self.donor_baseline.f1

    @property
    def donor_non_regression(self) -> bool:
        """Return whether donor F1 remains within the allowed tolerance."""

        return (
            self.donor_adapted.f1 + self.donor_non_regression_tolerance
            >= self.donor_baseline.f1
        )

    @property
    def adapted_target_passed(self) -> bool:
        """Return whether adapted target F1 meets its configured floor."""

        return self.target_adapted.f1 >= self.target_f1_floor

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic aggregate comparison payload."""

        return {
            "family": self.family,
            "donor": self.donor_language,
            "target": self.target_language,
            "baseline": {
                "donor": self.donor_baseline.to_dict(),
                "target": self.target_baseline.to_dict(),
            },
            "zero_shot": {
                "donor": self.donor_baseline.to_dict(),
                "target": self.target_zero_shot.to_dict(),
            },
            "adapted": {
                "donor": self.donor_adapted.to_dict(),
                "target": self.target_adapted.to_dict(),
            },
            "deltas": {
                "donor_to_target": {
                    "baseline": self.target_baseline.f1 - self.donor_baseline.f1,
                    "zero_shot": self.target_zero_shot.f1 - self.donor_baseline.f1,
                    "adapted": self.target_adapted.f1 - self.donor_adapted.f1,
                },
                "target_zero_shot_vs_baseline": self.target_zero_shot.f1
                - self.target_baseline.f1,
                "target_adapted_vs_baseline": self.target_adapted.f1
                - self.target_baseline.f1,
                "target_adapted_vs_zero_shot": self.target_adapted.f1
                - self.target_zero_shot.f1,
                "donor_adapted_vs_baseline": self.donor_delta,
            },
            "target_f1_floor": self.target_f1_floor,
            "adapted_target_passed": self.adapted_target_passed,
            "donor_non_regression_tolerance": self.donor_non_regression_tolerance,
            "donor_non_regression": self.donor_non_regression,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True, slots=True)
class FamilyTransferReport:
    """Per-family donor-to-target comparison report over synthetic gold."""

    suite: str
    model_name: str
    device: str
    fixture_count: int
    comparisons: tuple[FamilyTransferComparison, ...]

    @property
    def passed(self) -> bool:
        """Return whether all target floors and donor non-regression checks pass."""

        return bool(self.comparisons) and all(
            comparison.adapted_target_passed
            and comparison.donor_non_regression
            and comparison.target_zero_shot.f1 > comparison.target_baseline.f1
            for comparison in self.comparisons
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic family-grouped aggregate evidence."""

        families: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for comparison in self.comparisons:
            families[comparison.family].append(comparison.to_dict())
        return {
            "schema_version": FAMILY_TRANSFER_SCHEMA_VERSION,
            "artifact_type": FAMILY_TRANSFER_ARTIFACT_TYPE,
            "suite": self.suite,
            "model_name": self.model_name,
            "device": self.device,
            "fixture_count": self.fixture_count,
            "summary": {
                "family_count": len(families),
                "transfer_count": len(self.comparisons),
                "all_adapted_targets_passed": bool(self.comparisons)
                and all(item.adapted_target_passed for item in self.comparisons),
                "all_donors_non_regressed": bool(self.comparisons)
                and all(item.donor_non_regression for item in self.comparisons),
                "all_zero_shot_improved": bool(self.comparisons)
                and all(
                    item.target_zero_shot.f1 > item.target_baseline.f1
                    for item in self.comparisons
                ),
                "passed": self.passed,
            },
            "families": {family: families[family] for family in sorted(families)},
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize deterministic aggregate evidence as JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the family-transfer JSON report to ``path``."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render aggregate family-transfer metrics as stable Markdown."""

        lines = [
            "# Cross-Lingual Family Transfer",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Suite | `{self.suite}` |",
            f"| Model | `{self.model_name}` |",
            f"| Device | `{self.device}` |",
            f"| Fixtures | {self.fixture_count} |",
            f"| Result | {'passed' if self.passed else 'failed'} |",
            "",
            "| Family | Donor | Target | Donor baseline F1 | "
            "Donor adapted F1 | Donor delta | Target baseline F1 | "
            "Target zero-shot F1 | Target adapted F1 | Zero-shot gain | "
            "Adapted gain | Donor non-regression |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for comparison in self.comparisons:
            lines.append(
                "| "
                f"`{comparison.family}` | `{comparison.donor_language}` | "
                f"`{comparison.target_language}` | "
                f"{_format_metric(comparison.donor_baseline.f1)} | "
                f"{_format_metric(comparison.donor_adapted.f1)} | "
                f"{_format_delta(comparison.donor_delta)} | "
                f"{_format_metric(comparison.target_baseline.f1)} | "
                f"{_format_metric(comparison.target_zero_shot.f1)} | "
                f"{_format_metric(comparison.target_adapted.f1)} | "
                f"{_format_delta(comparison.target_zero_shot.f1 - comparison.target_baseline.f1)} | "
                f"{_format_delta(comparison.target_adapted.f1 - comparison.target_baseline.f1)} | "
                f"{'yes' if comparison.donor_non_regression else 'no'} |"
            )
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write the family-transfer Markdown report to ``path``."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def model_card_evidence(self) -> dict[str, Any]:
        """Return the aggregate report under a model-card evidence key."""

        return {"cross_lingual_family_transfer": self.to_dict()}

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True, order=True, slots=True)
class _TransferPair:
    family: str
    donor_language: str
    target_language: str


def load_family_transfer_fixtures(
    path: str | Path = FAMILY_TRANSFER_FIXTURES_PATH,
    *,
    config: FamilyTransferConfig = DEFAULT_FAMILY_TRANSFER_CONFIG,
) -> tuple[BenchmarkFixture, ...]:
    """Load and validate committed synthetic donor and target fixtures.

    Args:
        path: Local JSONL family-transfer gold path.
        config: Offline family taxonomy and donor graph used for validation.

    Returns:
        Validated fixtures in file order.

    Raises:
        ValueError: If fixtures are non-synthetic, incomplete, or inconsistent
            with the configured family-transfer graph.
    """

    fixtures = tuple(load_benchmark_fixtures(path))
    _group_family_transfer_fixtures(fixtures, config=config)
    return fixtures


def cross_lingual_family_transfer_report(
    model: str | ModelRunner,
    fixtures: Sequence[BenchmarkFixture | Mapping[str, Any]] | None = None,
    *,
    runner: ModelRunner | None = None,
    suite: str = "family-transfer-gold",
    device: str = "cpu",
    config: FamilyTransferConfig = DEFAULT_FAMILY_TRANSFER_CONFIG,
    donor_non_regression_tolerance: float = 0.0,
) -> FamilyTransferReport:
    """Evaluate baseline, zero-shot, and adapted family-transfer paths.

    The runner receives a copy of each fixture with aggregate-safe routing
    metadata: ``transfer_mode`` (``baseline``, ``zero_shot``, or ``adapted``),
    ``evaluation_role`` (``donor`` or ``target``), ``family``,
    ``donor_language``, ``target_language``, and ``adapter_language``. This
    keeps the runner contract offline and model-agnostic.

    Args:
        model: Model identifier or a harness-compatible runner callable.
        fixtures: Synthetic family-transfer fixtures. The bundled fixture is
            used when omitted.
        runner: Optional runner when ``model`` is a string identifier.
        suite: Stable suite name included in report metadata.
        device: Device tag forwarded to the runner.
        config: Offline family taxonomy and donor graph.
        donor_non_regression_tolerance: Maximum tolerated donor F1 reduction.

    Returns:
        Aggregate per-family donor/target metrics and transfer deltas.
    """

    tolerance = _validate_tolerance(donor_non_regression_tolerance)
    resolved_fixtures = (
        load_family_transfer_fixtures(config=config)
        if fixtures is None
        else tuple(_coerce_fixture(item) for item in fixtures)
    )
    grouped = _group_family_transfer_fixtures(resolved_fixtures, config=config)
    model_name, model_runner = _resolve_model_runner(model, runner)
    comparisons: list[FamilyTransferComparison] = []

    for pair in sorted(grouped):
        donor_fixtures = grouped[pair][_DONOR]
        target_fixtures = grouped[pair][_TARGET]
        donor_baseline = _score_mode(
            donor_fixtures,
            pair=pair,
            mode=_BASELINE,
            role=_DONOR,
            model_name=model_name,
            model_runner=model_runner,
            device=device,
        )
        donor_adapted = _score_mode(
            donor_fixtures,
            pair=pair,
            mode=_ADAPTED,
            role=_DONOR,
            model_name=model_name,
            model_runner=model_runner,
            device=device,
        )
        target_baseline = _score_mode(
            target_fixtures,
            pair=pair,
            mode=_BASELINE,
            role=_TARGET,
            model_name=model_name,
            model_runner=model_runner,
            device=device,
        )
        target_zero_shot = _score_mode(
            target_fixtures,
            pair=pair,
            mode=_ZERO_SHOT,
            role=_TARGET,
            model_name=model_name,
            model_runner=model_runner,
            device=device,
        )
        target_adapted = _score_mode(
            target_fixtures,
            pair=pair,
            mode=_ADAPTED,
            role=_TARGET,
            model_name=model_name,
            model_runner=model_runner,
            device=device,
        )
        edge = _transfer_edge(pair, config)
        comparisons.append(
            FamilyTransferComparison(
                family=pair.family,
                donor_language=pair.donor_language,
                target_language=pair.target_language,
                donor_baseline=donor_baseline,
                donor_adapted=donor_adapted,
                target_baseline=target_baseline,
                target_zero_shot=target_zero_shot,
                target_adapted=target_adapted,
                target_f1_floor=(
                    edge.expected_f1_floor
                    if edge.expected_f1_floor is not None
                    else DEFAULT_ADAPTED_TARGET_F1_FLOOR
                ),
                donor_non_regression_tolerance=tolerance,
            )
        )

    return FamilyTransferReport(
        suite=_require_text(suite, "suite"),
        model_name=model_name,
        device=_require_text(device, "device"),
        fixture_count=len(resolved_fixtures),
        comparisons=tuple(comparisons),
    )


def _coerce_fixture(
    fixture: BenchmarkFixture | Mapping[str, Any],
) -> BenchmarkFixture:
    if isinstance(fixture, BenchmarkFixture):
        return fixture
    return BenchmarkFixture.from_mapping(fixture)


def _group_family_transfer_fixtures(
    fixtures: Sequence[BenchmarkFixture],
    *,
    config: FamilyTransferConfig,
) -> dict[_TransferPair, dict[str, tuple[BenchmarkFixture, ...]]]:
    if not fixtures:
        raise ValueError("family-transfer evaluation requires synthetic fixtures")

    seen_ids: set[str] = set()
    grouped: defaultdict[_TransferPair, defaultdict[str, list[BenchmarkFixture]]]
    grouped = defaultdict(lambda: defaultdict(list))
    for fixture in fixtures:
        if fixture.fixture_id in seen_ids:
            raise ValueError(
                f"duplicate family-transfer fixture id {fixture.fixture_id!r}"
            )
        seen_ids.add(fixture.fixture_id)
        pair, role = _fixture_pair(fixture, config=config)
        grouped[pair][role].append(fixture)

    validated: dict[_TransferPair, dict[str, tuple[BenchmarkFixture, ...]]] = {}
    for pair in sorted(grouped):
        roles = grouped[pair]
        missing = [role for role in (_DONOR, _TARGET) if not roles.get(role)]
        if missing:
            raise ValueError(
                f"{pair.donor_language}->{pair.target_language}: missing "
                f"{', '.join(missing)} family-transfer fixtures"
            )
        validated[pair] = {
            _DONOR: tuple(roles[_DONOR]),
            _TARGET: tuple(roles[_TARGET]),
        }
    return validated


def _fixture_pair(
    fixture: BenchmarkFixture,
    *,
    config: FamilyTransferConfig,
) -> tuple[_TransferPair, str]:
    if fixture.metadata.get("synthetic") is not True:
        raise ValueError(
            f"family-transfer fixture {fixture.fixture_id!r} must be synthetic"
        )
    raw_transfer = fixture.metadata.get("family_transfer")
    if not isinstance(raw_transfer, Mapping):
        raise ValueError(
            f"family-transfer fixture {fixture.fixture_id!r} requires metadata.family_transfer"
        )

    family = _require_text(raw_transfer.get("family"), "family").casefold()
    donor = normalize_language_code(
        _require_text(raw_transfer.get("donor_language"), "donor_language")
    )
    target = normalize_language_code(
        _require_text(raw_transfer.get("target_language"), "target_language")
    )
    role = _require_text(raw_transfer.get("role"), "role").casefold()
    if role not in {_DONOR, _TARGET}:
        raise ValueError("family-transfer fixture role must be donor or target")

    pair = _TransferPair(family, donor, target)
    edge = _transfer_edge(pair, config)
    if edge.family_id != family:
        raise ValueError(
            f"{donor}->{target}: fixture family {family!r} does not match "
            f"configured family {edge.family_id!r}"
        )
    expected_language = donor if role == _DONOR else target
    fixture_language = normalize_language_code(fixture.language)
    if fixture_language != expected_language:
        raise ValueError(
            f"{fixture.fixture_id}: {role} fixture language must be "
            f"{expected_language!r}, got {fixture_language!r}"
        )
    if not fixture.gold_spans:
        raise ValueError(
            f"family-transfer fixture {fixture.fixture_id!r} requires gold spans"
        )
    return pair, role


def _transfer_edge(
    pair: _TransferPair,
    config: FamilyTransferConfig,
) -> TransferEdge:
    for edge in config.donor_edges_for(pair.target_language):
        if edge.donor_language == pair.donor_language:
            return edge
    raise ValueError(
        f"{pair.donor_language}->{pair.target_language} is not present in the "
        "family-transfer graph"
    )


def _score_mode(
    fixtures: Sequence[BenchmarkFixture],
    *,
    pair: _TransferPair,
    mode: str,
    role: str,
    model_name: str,
    model_runner: ModelRunner,
    device: str,
) -> FamilyTransferModeMetrics:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for fixture in fixtures:
        routed_fixture = _routed_fixture(fixture, pair=pair, mode=mode, role=role)
        predicted_spans = _predict_fixture(
            routed_fixture,
            model_name=model_name,
            model_runner=model_runner,
            device=device,
        )
        metrics = compute_relaxed_span_f1(
            fixture.gold_spans,
            predicted_spans,
            default_language=fixture.language,
            default_device=device,
            source_text=fixture.text,
        )
        true_positives += metrics.true_positives
        false_positives += metrics.false_positives
        false_negatives += metrics.false_negatives
    return _mode_metrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        fixture_count=len(fixtures),
    )


def _routed_fixture(
    fixture: BenchmarkFixture,
    *,
    pair: _TransferPair,
    mode: str,
    role: str,
) -> BenchmarkFixture:
    evaluation_language = (
        pair.donor_language if role == _DONOR else pair.target_language
    )
    source_language = (
        pair.donor_language
        if role == _DONOR or mode in {_ZERO_SHOT, _ADAPTED}
        else pair.target_language
    )
    adapter_language: str | None
    if mode == _BASELINE and role == _TARGET:
        adapter_language = None
    elif role == _DONOR or mode == _ZERO_SHOT:
        adapter_language = pair.donor_language
    else:
        adapter_language = pair.target_language

    metadata = dict(fixture.metadata)
    metadata.update(
        {
            "family": pair.family,
            "donor_language": pair.donor_language,
            "target_language": pair.target_language,
            "evaluation_language": evaluation_language,
            "evaluation_role": role,
            "transfer_mode": mode,
            "source_language": source_language,
            "adapter_language": adapter_language,
            "zero_shot": role == _TARGET and mode == _ZERO_SHOT,
            "adapted": mode == _ADAPTED,
        }
    )
    if role == _TARGET and mode == _ZERO_SHOT:
        metadata["held_out_language"] = pair.target_language
    return replace(fixture, metadata=metadata)


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


def _mode_metrics(
    *,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    fixture_count: int,
) -> FamilyTransferModeMetrics:
    predicted_count = true_positives + false_positives
    gold_count = true_positives + false_negatives
    precision = _safe_rate(true_positives, predicted_count, default=1.0)
    recall = _safe_rate(true_positives, gold_count, default=1.0)
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return FamilyTransferModeMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        fixture_count=fixture_count,
    )


def _resolve_model_runner(
    model: str | ModelRunner,
    runner: ModelRunner | None,
) -> tuple[str, ModelRunner]:
    if runner is not None:
        return str(model), runner
    if not isinstance(model, str) and callable(model):
        return str(getattr(model, "__name__", model.__class__.__name__)), model
    return str(model), default_model_runner


def _validate_tolerance(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("donor_non_regression_tolerance must be a number")
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "donor_non_regression_tolerance must be a finite non-negative number"
        )
    return tolerance


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _safe_rate(numerator: int, denominator: int, *, default: float) -> float:
    return numerator / denominator if denominator else default


def _format_metric(value: float) -> str:
    return f"{value:.3f}"


def _format_delta(value: float) -> str:
    return f"{value:+.3f}"


__all__ = [
    "DEFAULT_ADAPTED_TARGET_F1_FLOOR",
    "FAMILY_TRANSFER_ARTIFACT_TYPE",
    "FAMILY_TRANSFER_FIXTURES_PATH",
    "FAMILY_TRANSFER_SCHEMA_VERSION",
    "FamilyTransferComparison",
    "FamilyTransferModeMetrics",
    "FamilyTransferReport",
    "cross_lingual_family_transfer_report",
    "load_family_transfer_fixtures",
]
