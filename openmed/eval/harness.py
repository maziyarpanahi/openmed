"""Suite runner for OpenMed benchmark fixtures."""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from openmed.core.audit import AuditSignature, stable_hash
from openmed.core.quality_gates import validate_entity_spans
from openmed.core.safety_sweep import hashed_span_surface
from openmed.eval.cache import build_report_key, hash_fixture_set, load_or_compute
from openmed.eval.calibrate import load_calibration_thresholds
from openmed.eval.metrics import (
    EvalSpan,
    RelationTriple,
    compute_confidence_intervals,
    compute_latency_summary,
    compute_metrics_bundle,
    compute_relation_confidence_intervals,
    compute_relation_metrics_bundle,
    compute_resource_metrics,
    expected_calibration_error,
    normalize_eval_spans,
    normalize_relation_triples,
    reliability_bins,
)
from openmed.eval.report import BenchmarkReport

if TYPE_CHECKING:
    from openmed.eval.attacks.reid import SideChannelProbeResult

ModelRunner = Callable[["BenchmarkFixture", str, str], Iterable[Any]]
RelationModelRunner = Callable[[Any, str, str], Iterable[Any]]
_SIGNATURE_ALGORITHM = "HMAC-SHA256"
_DEFAULT_FEDERATED_SIGNING_KEY = "openmed-federated-eval-local-key"
DEFAULT_CONTEXT_MULTILINGUAL_FIXTURE = (
    Path(__file__).resolve().parent
    / "golden"
    / "fixtures"
    / "context_multilingual.jsonl"
)


@dataclass(frozen=True)
class BenchmarkFixture:
    """One benchmark document with gold PHI spans."""

    fixture_id: str
    text: str
    gold_spans: tuple[EvalSpan, ...]
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BenchmarkFixture":
        """Build a fixture from a JSON-ready mapping."""
        text = str(data.get("text", ""))
        language = str(data.get("language") or data.get("lang") or "en")
        fixture_id = str(data.get("id") or data.get("fixture_id") or "fixture")
        gold_spans = tuple(
            normalize_eval_spans(
                data.get("gold_spans") or data.get("entities") or [],
                default_language=language,
                source_text=text,
            )
        )
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {"value": metadata}
        return cls(
            fixture_id=fixture_id,
            text=text,
            gold_spans=gold_spans,
            language=language,
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class FixtureResult:
    """Predictions and timing for one benchmark fixture."""

    fixture_id: str
    predicted_spans: tuple[EvalSpan, ...]
    latency_ms: float


@dataclass(frozen=True)
class RelationFixtureResult:
    """Predicted relation triples and timing for one relation fixture."""

    fixture_id: str
    predicted_relations: tuple[RelationTriple, ...]
    latency_ms: float


@dataclass(frozen=True)
class FederatedDetectorSpec:
    """Python subprocess detector entry point for federated evaluation."""

    script_path: str | Path
    python_executable: str | Path = sys.executable
    timeout_s: float = 10.0
    read_roots: tuple[str | Path, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryLeakageFinding:
    """Raw-PHI egress evidence keyed by source offsets and hashes only."""

    fixture_id: str
    sink: str
    artifact: str
    start: int
    end: int
    label: str
    length: int
    text_hash: str
    byte_offsets: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "sink": self.sink,
            "artifact": self.artifact,
            "start": int(self.start),
            "end": int(self.end),
            "label": self.label,
            "length": int(self.length),
            "text_hash": self.text_hash,
            "byte_offsets": [int(offset) for offset in self.byte_offsets],
        }


@dataclass(frozen=True)
class BoundaryLeakageResult:
    """Boundary leakage rate for detector stdout/stderr/files."""

    rate: float
    leaked_bytes: int
    total_phi_bytes: int
    findings: tuple[BoundaryLeakageFinding, ...] = ()
    emitted_bytes_by_sink: Mapping[str, int] = field(default_factory=dict)

    def to_metric(self) -> dict[str, Any]:
        return {
            "rate": float(self.rate),
            "leaked_bytes": int(self.leaked_bytes),
            "total_phi_bytes": int(self.total_phi_bytes),
            "findings": [finding.to_dict() for finding in self.findings],
            "emitted_bytes_by_sink": {
                str(key): int(value)
                for key, value in sorted(self.emitted_bytes_by_sink.items())
            },
        }


@dataclass(frozen=True)
class SandboxViolation:
    """Sandbox policy violation reported without raw host paths."""

    fixture_id: str
    kind: str
    event: str
    operation: str
    path_hash: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "fixture_id": self.fixture_id,
            "kind": self.kind,
            "event": self.event,
            "operation": self.operation,
            "detail": self.detail,
        }
        if self.path_hash is not None:
            payload["path_hash"] = self.path_hash
        return payload


@dataclass
class FederatedEvalReport:
    """Signed federated detector-boundary report."""

    suite: str
    detector_name: str
    fixture_count: int
    boundary_leakage: BoundaryLeakageResult
    side_channel: SideChannelProbeResult
    sandbox_violations: tuple[SandboxViolation, ...]
    resource_accounting: Mapping[str, Any]
    gate_passed: bool
    generated_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    repro_hash: str = ""
    signature: AuditSignature | None = None

    def __post_init__(self) -> None:
        self.sandbox_violations = tuple(self.sandbox_violations)
        if not self.repro_hash:
            self.repro_hash = self.recompute_repro_hash()

    def _payload(
        self,
        *,
        include_repro_hash: bool,
        include_signature: bool,
        include_resource_accounting: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "suite": self.suite,
            "detector_name": self.detector_name,
            "fixture_count": int(self.fixture_count),
            "boundary_leakage": self.boundary_leakage.to_metric(),
            "side_channel": self.side_channel.to_metric(),
            "sandbox_violations": [
                violation.to_dict() for violation in self.sandbox_violations
            ],
            "gate_passed": bool(self.gate_passed),
            "generated_at": self.generated_at,
            "metadata": _plain(self.metadata),
        }
        if include_resource_accounting:
            payload["resource_accounting"] = _plain(self.resource_accounting)
        if include_repro_hash:
            payload["repro_hash"] = self.repro_hash
        if include_signature:
            payload["signature"] = (
                self.signature.to_dict() if self.signature is not None else None
            )
        return payload

    def recompute_repro_hash(self) -> str:
        """Hash deterministic leakage evidence, excluding volatile resources."""
        return stable_hash(
            self._payload(
                include_repro_hash=False,
                include_signature=False,
                include_resource_accounting=False,
            )
        )

    def sign(
        self,
        key: bytes | str,
        *,
        key_id: str = "federated-eval",
    ) -> "FederatedEvalReport":
        """Sign the complete report payload and return ``self``."""
        self.repro_hash = self.recompute_repro_hash()
        message = _canonical_json(
            self._payload(include_repro_hash=True, include_signature=False)
        ).encode("utf-8")
        self.signature = AuditSignature(
            key_id=key_id,
            algorithm=_SIGNATURE_ALGORITHM,
            value=hmac.new(_key_bytes(key), message, hashlib.sha256).hexdigest(),
        )
        return self

    def verify(self, key: bytes | str) -> bool:
        """Verify the report signature and deterministic evidence hash."""
        if self.recompute_repro_hash() != self.repro_hash:
            return False
        if self.signature is None or self.signature.algorithm != _SIGNATURE_ALGORITHM:
            return False
        message = _canonical_json(
            self._payload(include_repro_hash=True, include_signature=False)
        ).encode("utf-8")
        expected = hmac.new(_key_bytes(key), message, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, self.signature.value)

    def to_dict(self) -> dict[str, Any]:
        return self._payload(include_repro_hash=True, include_signature=True)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_benchmark_report(self) -> BenchmarkReport:
        """Expose federated boundary metrics through the benchmark report shape."""
        return BenchmarkReport(
            suite=self.suite,
            model_name=self.detector_name,
            device="federated-subprocess",
            fixture_count=self.fixture_count,
            generated_at=self.generated_at,
            metrics={
                "boundary_leakage": self.boundary_leakage.to_metric(),
                "federated_boundary_leakage_rate": self.boundary_leakage.rate,
                "side_channel": self.side_channel.to_metric(),
                "sandbox_violation_count": len(self.sandbox_violations),
            },
            metadata={
                **dict(self.metadata),
                "federated_eval": True,
                "federated_gate_passed": self.gate_passed,
                "federated_repro_hash": self.repro_hash,
            },
        )


@dataclass(frozen=True)
class _CapturedArtifact:
    sink: str
    artifact: str
    content: bytes


@dataclass(frozen=True)
class _FederatedFixtureRun:
    fixture_id: str
    predicted_spans: tuple[EvalSpan, ...]
    timing_records: tuple[dict[str, Any], ...]
    artifacts: tuple[_CapturedArtifact, ...]
    sandbox_violations: tuple[SandboxViolation, ...]
    elapsed_ms: float
    exit_code: int


def load_fixtures(path: str | Path) -> list[BenchmarkFixture]:
    """Load benchmark fixtures from a JSON or JSONL file.

    Accepted top-level shapes are either a list of fixture objects or a mapping
    containing a ``fixtures`` list. JSONL files contain one fixture object per
    non-empty line.
    """
    fixture_path = Path(path)
    if fixture_path.suffix.lower() == ".jsonl":
        fixtures = [
            BenchmarkFixture.from_mapping(json.loads(line))
            for line in fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _validate_unique_fixture_ids(fixtures)
        return fixtures

    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    rows = raw.get("fixtures") if isinstance(raw, Mapping) else raw
    if not isinstance(rows, list):
        raise ValueError(
            "benchmark fixture JSON must be a list or contain a fixtures list"
        )
    fixtures = [BenchmarkFixture.from_mapping(row) for row in rows]
    _validate_unique_fixture_ids(fixtures)
    return fixtures


def load_context_multilingual_fixtures(
    path: str | Path = DEFAULT_CONTEXT_MULTILINGUAL_FIXTURE,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    """Load synthetic multilingual ConText assertion fixtures."""

    fixture_path = Path(path)
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or rows[0].get("kind") != "meta":
        raise ValueError("context multilingual fixture must start with a meta row")
    fixtures = tuple(row for row in rows[1:] if row.get("kind") != "meta")
    case_ids = [str(row.get("case_id", "")) for row in fixtures]
    if any(not case_id for case_id in case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("context multilingual fixtures require unique case_id values")
    return rows[0], fixtures


def run_context_multilingual_eval(
    path: str | Path = DEFAULT_CONTEXT_MULTILINGUAL_FIXTURE,
    *,
    generated_at: str | None = None,
) -> BenchmarkReport:
    """Score deterministic multilingual ConText axes on synthetic fixtures."""

    from openmed.clinical.context import (
        CERTAINTY_VALUES,
        NEGATION_VALUES,
        TEMPORALITY_VALUES,
        clinical_context_lexicon_stats,
        resolve_span_context,
    )

    meta, fixtures = load_context_multilingual_fixtures(path)
    labels_by_axis = {
        "negation": NEGATION_VALUES,
        "temporality": TEMPORALITY_VALUES,
        "uncertainty": CERTAINTY_VALUES,
    }
    expected_by_language: dict[str, dict[str, list[str]]] = {}
    predicted_by_language: dict[str, dict[str, list[str]]] = {}

    for row in fixtures:
        language = str(row.get("language") or "en")
        span = _context_fixture_span(row)
        context = resolve_span_context(span, language=language)
        expected = row.get("expected")
        if not isinstance(expected, Mapping):
            raise ValueError(
                f"context fixture {row.get('case_id')} lacks expected axes"
            )

        axis_predictions = {
            "negation": context.negation,
            "temporality": context.temporality,
            "uncertainty": context.certainty,
        }
        axis_expected = {
            "negation": str(expected["negation"]),
            "temporality": str(expected["temporality"]),
            "uncertainty": str(expected["certainty"]),
        }
        language_expected = expected_by_language.setdefault(language, {})
        language_predicted = predicted_by_language.setdefault(language, {})
        for axis in labels_by_axis:
            language_expected.setdefault(axis, []).append(axis_expected[axis])
            language_predicted.setdefault(axis, []).append(axis_predictions[axis])

    macro_f1 = {
        language: {
            axis: _macro_f1(
                expected_by_language[language][axis],
                predicted_by_language[language][axis],
                labels_by_axis[axis],
            )
            for axis in labels_by_axis
        }
        for language in sorted(expected_by_language)
    }
    thresholds = {"negation": 0.90, "temporality": 0.85, "uncertainty": 0.85}
    gate_passed = all(
        macro_f1[language][axis] >= thresholds[axis]
        for language in macro_f1
        for axis in thresholds
    )
    metrics = {
        "context_macro_f1": macro_f1,
        "context_thresholds": thresholds,
        "context_gate_passed": gate_passed,
        "context_lexicon_coverage": clinical_context_lexicon_stats(),
    }
    return BenchmarkReport(
        suite="context_multilingual",
        model_name="deterministic-context",
        device="local",
        fixture_count=len(fixtures),
        metrics=metrics,
        generated_at=generated_at,
        metadata={
            "fixture_ids": [str(row["case_id"]) for row in fixtures],
            "languages": sorted(expected_by_language),
            "parent_issue": "OM-724",
            "synthetic": bool(meta.get("synthetic")),
        },
    )


def _context_fixture_span(row: Mapping[str, Any]) -> dict[str, Any]:
    text = str(row.get("text", ""))
    target = row.get("target")
    if not isinstance(target, Mapping):
        raise ValueError(f"context fixture {row.get('case_id')} lacks target")
    target_text = str(target.get("text") or "")
    if not target_text:
        raise ValueError(f"context fixture {row.get('case_id')} has empty target")
    start = text.find(target_text)
    if start == -1:
        raise ValueError(
            f"context fixture {row.get('case_id')} target is absent from text"
        )
    return {
        "text": target_text,
        "context": text,
        "start": start,
        "end": start + len(target_text),
    }


def _macro_f1(
    expected: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
) -> float:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted labels must be the same length")
    scores = [
        _label_f1(expected, predicted, label)
        for label in labels
        if label in expected or label in predicted
    ]
    if not scores:
        return 1.0
    return sum(scores) / len(scores)


def _label_f1(expected: Sequence[str], predicted: Sequence[str], label: str) -> float:
    true_positive = sum(
        1
        for gold, guess in zip(expected, predicted)
        if gold == label and guess == label
    )
    false_positive = sum(
        1
        for gold, guess in zip(expected, predicted)
        if gold != label and guess == label
    )
    false_negative = sum(
        1
        for gold, guess in zip(expected, predicted)
        if gold == label and guess != label
    )
    if true_positive == 0:
        return 0.0 if false_positive or false_negative else 1.0
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2 * precision * recall / (precision + recall)


def default_model_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
    *,
    loader: Any | None = None,
) -> Iterable[Any]:
    """Run a fixture through the existing PII runtime."""
    from openmed.core.pii import extract_pii

    result = extract_pii(
        fixture.text,
        model_name=model_name,
        lang=fixture.language,
        loader=loader,
    )
    for entity in result.entities:
        metadata = dict(entity.metadata or {})
        metadata.setdefault("device", device)
        entity.metadata = metadata
    return result.entities


def run_benchmark(
    fixtures: Sequence[BenchmarkFixture],
    *,
    suite: str,
    model_name: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    confidence_intervals: bool = False,
    ci_resamples: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
    calibration: bool = False,
    calibration_bins: int = 10,
    abstention_thresholds: Any | None = None,
    abstention_thresholds_path: str | Path | None = None,
    abstention_confidence_threshold: float = 0.0,
    abstention_target_risk: float | None = None,
    abstention_confidence_level: float | None = None,
    abstention_bootstrap_resamples: int = 0,
    abstention_seed: int = 0,
    cache_dir: str | Path | None = None,
    cache_code_hash: str | None = None,
) -> BenchmarkReport:
    """Run *model_name* over fixtures and return a benchmark report.

    When ``confidence_intervals`` is enabled (off by default to keep fast runs
    cheap), a non-parametric bootstrap over documents attaches a
    ``confidence_interval`` payload to the leakage, character recall, and exact
    and relaxed span F1 metrics. The bootstrap is deterministic for a fixed
    ``ci_seed``. Passing ``cache_dir`` opts into a local filesystem cache keyed
    by model, suite, device, fixture-set hash, and eval code hash.
    """
    _validate_unique_fixture_ids(fixtures)
    active_abstention_thresholds = _resolve_abstention_thresholds(
        abstention_thresholds,
        abstention_thresholds_path,
    )
    if cache_dir is not None:
        effective_code_hash = _abstention_cache_hash(
            cache_code_hash,
            thresholds=active_abstention_thresholds,
            thresholds_path=abstention_thresholds_path,
            confidence_threshold=abstention_confidence_threshold,
            target_risk=abstention_target_risk,
            confidence_level=abstention_confidence_level,
            bootstrap_resamples=abstention_bootstrap_resamples,
            seed=abstention_seed,
        )
        report_key = build_report_key(
            model_name=model_name,
            suite=suite,
            fixture_set_hash=hash_fixture_set(fixtures),
            code_hash=effective_code_hash,
            device=device,
        )
        return load_or_compute(
            report_key,
            lambda: run_benchmark(
                fixtures,
                suite=suite,
                model_name=model_name,
                device=device,
                runner=runner,
                generated_at=generated_at,
                metadata=metadata,
                confidence_intervals=confidence_intervals,
                ci_resamples=ci_resamples,
                ci_alpha=ci_alpha,
                ci_seed=ci_seed,
                calibration=calibration,
                calibration_bins=calibration_bins,
                abstention_thresholds=active_abstention_thresholds,
                abstention_confidence_threshold=abstention_confidence_threshold,
                abstention_target_risk=abstention_target_risk,
                abstention_confidence_level=abstention_confidence_level,
                abstention_bootstrap_resamples=abstention_bootstrap_resamples,
                abstention_seed=abstention_seed,
            ),
            cache_dir=cache_dir,
        )

    model_runner = runner or _shared_default_model_runner()
    results: list[FixtureResult] = []
    peak_rss_start = _peak_rss_bytes()

    for fixture in fixtures:
        started = time.perf_counter()
        raw_predictions = list(model_runner(fixture, model_name, device))
        latency_ms = (time.perf_counter() - started) * 1000.0
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
        results.append(
            FixtureResult(
                fixture_id=fixture.fixture_id,
                predicted_spans=predicted_spans,
                latency_ms=latency_ms,
            )
        )

    gold_spans, predicted_spans, corpus_text = _corpus_coordinates(fixtures, results)
    peak_rss_end = _peak_rss_bytes()
    rss_values = [
        value for value in (peak_rss_start, peak_rss_end) if value is not None
    ]
    peak_rss = max(rss_values) if rss_values else None
    metrics = compute_metrics_bundle(
        gold_spans,
        predicted_spans,
        latencies_ms=[result.latency_ms for result in results[1:]],
        cold_start_ms=(results[0].latency_ms if results else None),
        peak_rss_bytes=peak_rss,
        abstention_thresholds=active_abstention_thresholds,
        abstention_confidence_threshold=abstention_confidence_threshold,
        abstention_model_id=model_name,
        abstention_target_risk=abstention_target_risk,
        abstention_confidence_level=abstention_confidence_level,
        abstention_bootstrap_resamples=abstention_bootstrap_resamples,
        abstention_seed=abstention_seed,
        default_device=device,
        source_text=corpus_text,
    )
    if confidence_intervals:
        metrics = _attach_confidence_intervals(
            metrics,
            fixtures,
            results,
            device=device,
            n_resamples=ci_resamples,
            alpha=ci_alpha,
            seed=ci_seed,
        )
    if calibration:
        metrics = _attach_calibration_metrics(
            metrics,
            gold_spans,
            predicted_spans,
            n_bins=calibration_bins,
        )

    report_metadata = dict(metadata or {})
    report_metadata.setdefault(
        "fixture_ids", [fixture.fixture_id for fixture in fixtures]
    )
    return BenchmarkReport(
        suite=suite,
        model_name=model_name,
        device=device,
        fixture_count=len(fixtures),
        metrics=metrics,
        generated_at=generated_at,
        metadata=report_metadata,
    )


def run_relation_benchmark(
    fixtures: Sequence[Any],
    *,
    suite: str,
    model_name: str,
    runner: RelationModelRunner,
    device: str = "cpu",
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    ci_resamples: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
) -> BenchmarkReport:
    """Run a relation-extraction model over DrugProt-style relation fixtures."""
    _validate_unique_fixture_ids(fixtures)
    results: list[RelationFixtureResult] = []
    peak_rss_start = _peak_rss_bytes()

    for fixture in fixtures:
        fixture_id = str(getattr(fixture, "fixture_id"))
        text = str(getattr(fixture, "text", ""))
        started = time.perf_counter()
        raw_predictions = list(runner(fixture, model_name, device))
        latency_ms = (time.perf_counter() - started) * 1000.0
        predicted_relations = normalize_relation_triples(raw_predictions)
        for relation in predicted_relations:
            _validate_relation_offsets(relation, text, fixture_id)
        results.append(
            RelationFixtureResult(
                fixture_id=fixture_id,
                predicted_relations=predicted_relations,
                latency_ms=latency_ms,
            )
        )

    gold_relations, predicted_relations = _relation_corpus_coordinates(
        fixtures,
        results,
    )
    per_document_relations = _per_document_relations(fixtures, results)
    relation_metrics = compute_relation_metrics_bundle(
        gold_relations,
        predicted_relations,
    )
    relation_intervals = compute_relation_confidence_intervals(
        per_document_relations,
        n_resamples=ci_resamples,
        alpha=ci_alpha,
        seed=ci_seed,
    )
    for key, interval in relation_intervals.items():
        metric = relation_metrics.get(key)
        if isinstance(metric, Mapping):
            relation_metrics[key] = {**metric, "confidence_interval": interval}

    peak_rss_end = _peak_rss_bytes()
    rss_values = [
        value for value in (peak_rss_start, peak_rss_end) if value is not None
    ]
    peak_rss = max(rss_values) if rss_values else None
    metrics: dict[str, Any] = {
        "latency": {
            **compute_latency_summary(
                [result.latency_ms for result in results[1:]]
            ).to_dict(),
            "cold_start_ms": results[0].latency_ms if results else None,
        },
        "relation_extraction": relation_metrics,
        "resources": compute_resource_metrics(peak_rss_bytes=peak_rss).to_dict(),
    }
    metrics["strict_relation_f1"] = relation_metrics["strict"]
    metrics["relaxed_relation_f1"] = relation_metrics["relaxed"]
    metrics["per_relation_type_re_f1"] = relation_metrics["per_relation_type"]

    report_metadata = dict(metadata or {})
    report_metadata.setdefault(
        "fixture_ids",
        [str(getattr(fixture, "fixture_id")) for fixture in fixtures],
    )
    report_metadata.setdefault("task", "relation")
    report_metadata.setdefault(
        "relation_types",
        sorted(
            {
                relation.relation_type
                for relation in normalize_relation_triples(gold_relations)
            }
        ),
    )
    return BenchmarkReport(
        suite=suite,
        model_name=model_name,
        device=device,
        fixture_count=len(fixtures),
        metrics=metrics,
        generated_at=generated_at,
        metadata=report_metadata,
    )


def run_suite(
    fixture_path: str | Path,
    *,
    suite: str,
    model_name: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    confidence_intervals: bool = False,
    ci_resamples: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
    calibration: bool = False,
    calibration_bins: int = 10,
    abstention_thresholds: Any | None = None,
    abstention_thresholds_path: str | Path | None = None,
    abstention_confidence_threshold: float = 0.0,
    abstention_target_risk: float | None = None,
    abstention_confidence_level: float | None = None,
    abstention_bootstrap_resamples: int = 0,
    abstention_seed: int = 0,
    cache_dir: str | Path | None = None,
    cache_code_hash: str | None = None,
) -> BenchmarkReport:
    """Load fixtures, run the benchmark, and optionally write reports."""
    report = run_benchmark(
        load_fixtures(fixture_path),
        suite=suite,
        model_name=model_name,
        device=device,
        runner=runner,
        generated_at=generated_at,
        metadata=metadata,
        confidence_intervals=confidence_intervals,
        ci_resamples=ci_resamples,
        ci_alpha=ci_alpha,
        ci_seed=ci_seed,
        calibration=calibration,
        calibration_bins=calibration_bins,
        abstention_thresholds=abstention_thresholds,
        abstention_thresholds_path=abstention_thresholds_path,
        abstention_confidence_threshold=abstention_confidence_threshold,
        abstention_target_risk=abstention_target_risk,
        abstention_confidence_level=abstention_confidence_level,
        abstention_bootstrap_resamples=abstention_bootstrap_resamples,
        abstention_seed=abstention_seed,
        cache_dir=cache_dir,
        cache_code_hash=cache_code_hash,
    )
    if output_json is not None:
        report.write_json(output_json)
    if output_markdown is not None:
        report.write_markdown(output_markdown)
    return report


def run_cross_lingual_transfer(
    fixtures: Sequence[BenchmarkFixture],
    *,
    suite: str,
    model_name: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
    languages: Sequence[str] | None = None,
    leakage_floors: Mapping[str, float] | None = None,
    ci_resamples: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
) -> Any:
    """Run the cross-lingual transfer matrix over benchmark fixtures.

    The returned report is PHI-free and byte-stable. Its source-language
    calibration context is passed to the runner through fixture metadata.
    """
    from openmed.eval.fairness import cross_lingual_transfer_report

    report = cross_lingual_transfer_report(
        model_name,
        fixtures,
        runner=runner,
        device=device,
        languages=languages,
        leakage_floors=leakage_floors,
        ci_resamples=ci_resamples,
        ci_alpha=ci_alpha,
        ci_seed=ci_seed,
    )
    report = replace(report, suite=suite)
    if output_json is not None:
        report.write_json(output_json)
    if output_markdown is not None:
        report.write_markdown(output_markdown)
    return report


def run_cross_lingual_transfer_suite(
    fixture_path: str | Path,
    *,
    suite: str,
    model_name: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
    languages: Sequence[str] | None = None,
    leakage_floors: Mapping[str, float] | None = None,
    ci_resamples: int = 1000,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
) -> Any:
    """Load fixtures and run a cross-lingual transfer-matrix report."""
    return run_cross_lingual_transfer(
        load_fixtures(fixture_path),
        suite=suite,
        model_name=model_name,
        device=device,
        runner=runner,
        output_json=output_json,
        output_markdown=output_markdown,
        languages=languages,
        leakage_floors=leakage_floors,
        ci_resamples=ci_resamples,
        ci_alpha=ci_alpha,
        ci_seed=ci_seed,
    )


def run_federated_leakage_eval(
    fixtures: Sequence[BenchmarkFixture],
    *,
    detector: FederatedDetectorSpec | str | Path,
    suite: str = "federated",
    detector_name: str | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    max_boundary_leakage_rate: float = 0.0,
    side_channel_threshold_bits: float = 0.30,
    side_channel_min_samples: int = 4,
    signing_key: bytes | str | None = None,
    key_id: str = "federated-eval",
    work_dir: str | Path | None = None,
) -> FederatedEvalReport:
    """Run an untrusted detector out-of-process and gate boundary leakage.

    The detector protocol is intentionally narrow: the child receives a JSON
    fixture path and output path through environment variables, writes
    predictions to that output JSON, and any stdout/stderr/side files are
    treated as monitored egress. Gold spans are never passed to the child.
    """
    from openmed.eval.attacks.reid import probe_span_timing_side_channel

    _validate_unique_fixture_ids(fixtures)
    spec = _coerce_federated_spec(detector)
    script_path = Path(spec.script_path).expanduser().resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"detector script does not exist: {script_path}")

    temp_root = Path(
        tempfile.mkdtemp(
            prefix="openmed-federated-",
            dir=str(work_dir) if work_dir is not None else None,
        )
    )
    wrapper_path = temp_root / "federated_child.py"
    wrapper_path.write_text(_FEDERATED_CHILD_WRAPPER, encoding="utf-8")

    runs: list[_FederatedFixtureRun] = []
    try:
        for fixture in fixtures:
            runs.append(
                _run_federated_fixture(
                    fixture,
                    spec=spec,
                    script_path=script_path,
                    wrapper_path=wrapper_path,
                    root=temp_root,
                )
            )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    leakage = _scan_boundary_egress(fixtures, runs)
    timing_records = [record for run in runs for record in run.timing_records]
    side_channel = probe_span_timing_side_channel(
        fixtures,
        timing_records,
        threshold_bits=side_channel_threshold_bits,
        min_samples=side_channel_min_samples,
    )
    sandbox_violations = tuple(
        violation for run in runs for violation in run.sandbox_violations
    )
    gate_passed = (
        leakage.rate <= max_boundary_leakage_rate
        and not side_channel.flagged
        and not sandbox_violations
    )
    report = FederatedEvalReport(
        suite=suite,
        detector_name=detector_name or script_path.stem,
        fixture_count=len(fixtures),
        boundary_leakage=leakage,
        side_channel=side_channel,
        sandbox_violations=sandbox_violations,
        resource_accounting=_resource_accounting(runs),
        gate_passed=gate_passed,
        generated_at=generated_at,
        metadata={
            **dict(metadata or {}),
            "protocol": "openmed.federated-detector.v1",
            "detector_path_hash": _path_hash(script_path),
        },
    )
    key = signing_key or os.environ.get(
        "OPENMED_FEDERATED_EVAL_KEY",
        _DEFAULT_FEDERATED_SIGNING_KEY,
    )
    return report.sign(key, key_id=key_id)


def _coerce_federated_spec(
    detector: FederatedDetectorSpec | str | Path,
) -> FederatedDetectorSpec:
    if isinstance(detector, FederatedDetectorSpec):
        return detector
    return FederatedDetectorSpec(script_path=detector)


def _run_federated_fixture(
    fixture: BenchmarkFixture,
    *,
    spec: FederatedDetectorSpec,
    script_path: Path,
    wrapper_path: Path,
    root: Path,
) -> _FederatedFixtureRun:
    run_dir = root / _slug_fixture_id(fixture.fixture_id)
    output_dir = run_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "input.json"
    output_path = output_dir / "detector_output.json"
    violation_log = run_dir / "sandbox_violations.jsonl"
    input_path.write_text(
        _canonical_json(
            {
                "fixture_id": fixture.fixture_id,
                "language": fixture.language,
                "metadata": _plain(fixture.metadata),
                "text": fixture.text,
            }
        ),
        encoding="utf-8",
    )

    command = [
        str(spec.python_executable),
        str(wrapper_path),
        str(script_path),
        str(input_path),
        str(output_path),
        str(output_dir),
        str(violation_log),
        _path_list([script_path.parent, *spec.read_roots]),
    ]
    started = time.perf_counter()
    stdout = b""
    stderr = b""
    exit_code = 0
    timeout_violation: SandboxViolation | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=output_dir,
            env=_sandbox_env(spec, input_path, output_path, output_dir),
            capture_output=True,
            timeout=spec.timeout_s,
            check=False,
        )
        stdout = _bytes_output(completed.stdout)
        stderr = _bytes_output(completed.stderr)
        exit_code = int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        stdout = _bytes_output(exc.stdout)
        stderr = _bytes_output(exc.stderr)
        exit_code = -9
        timeout_violation = SandboxViolation(
            fixture_id=fixture.fixture_id,
            kind="timeout",
            event="process.timeout",
            operation="terminated",
            detail=f"detector exceeded {spec.timeout_s:.3f}s timeout",
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    artifacts = list(_collect_artifacts(output_dir, stdout=stdout, stderr=stderr))
    payload = _read_detector_payload(output_path)
    predicted_spans = tuple(
        normalize_eval_spans(
            _extract_detector_spans(payload),
            default_language=fixture.language,
            default_device="federated-subprocess",
            source_text=fixture.text,
        )
    )
    validate_entity_spans([span.to_entity() for span in predicted_spans], fixture.text)
    timing_records = tuple(
        _extract_timing_records(payload, fixture_id=fixture.fixture_id)
    )
    violations = list(_read_sandbox_violations(violation_log, fixture.fixture_id))
    if timeout_violation is not None:
        violations.append(timeout_violation)
    if exit_code != 0 and not violations:
        violations.append(
            SandboxViolation(
                fixture_id=fixture.fixture_id,
                kind="process",
                event="process.exit",
                operation="nonzero_exit",
                detail=f"detector exited with status {exit_code}",
            )
        )
    return _FederatedFixtureRun(
        fixture_id=fixture.fixture_id,
        predicted_spans=predicted_spans,
        timing_records=timing_records,
        artifacts=tuple(artifacts),
        sandbox_violations=tuple(violations),
        elapsed_ms=elapsed_ms,
        exit_code=exit_code,
    )


def _sandbox_env(
    spec: FederatedDetectorSpec,
    input_path: Path,
    output_path: Path,
    output_dir: Path,
) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OPENMED_DETECTOR_INPUT": str(input_path),
        "OPENMED_DETECTOR_OUTPUT": str(output_path),
        "OPENMED_DETECTOR_OUTPUT_DIR": str(output_dir),
    }
    env.update({str(key): str(value) for key, value in spec.env.items()})
    return env


def _collect_artifacts(
    output_dir: Path,
    *,
    stdout: bytes,
    stderr: bytes,
) -> Iterable[_CapturedArtifact]:
    yield _CapturedArtifact("stdout", "stdout", stdout)
    yield _CapturedArtifact("stderr", "stderr", stderr)
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir).as_posix()
        artifact_id = f"file:{_hash_bytes(relative.encode('utf-8'))}"
        yield _CapturedArtifact("file_path", artifact_id, relative.encode("utf-8"))
        try:
            content = path.read_bytes()
        except OSError:
            content = b""
        yield _CapturedArtifact("file", artifact_id, content)


def _read_detector_payload(output_path: Path) -> Any:
    if not output_path.exists():
        return {"spans": [], "timings": []}
    try:
        return json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"spans": [], "timings": []}


def _extract_detector_spans(payload: Any) -> Sequence[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("spans", "predictions", "entities"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(
                value,
                (str, bytes, bytearray),
            ):
                return value
    return ()


def _extract_timing_records(
    payload: Any,
    *,
    fixture_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(payload, Mapping):
        for key in ("timings", "timing_records", "span_timings"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(
                value,
                (str, bytes, bytearray),
            ):
                records.extend(_timing_record(item, fixture_id) for item in value)
    for span in _extract_detector_spans(payload):
        data = span if isinstance(span, Mapping) else vars(span)
        metadata = data.get("metadata") if isinstance(data, Mapping) else None
        if not isinstance(metadata, Mapping) or "duration_ms" not in metadata:
            continue
        records.append(
            _timing_record(
                {
                    "start": data.get("start"),
                    "end": data.get("end"),
                    "duration_ms": metadata.get("duration_ms"),
                },
                fixture_id,
            )
        )
    return [record for record in records if record]


def _timing_record(value: Any, fixture_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    record = {
        "fixture_id": fixture_id,
        "start": value.get("start"),
        "end": value.get("end"),
        "duration_ms": value.get("duration_ms", value.get("elapsed_ms")),
    }
    if value.get("label") is not None:
        record["label"] = str(value["label"])
    return record


def _read_sandbox_violations(
    violation_log: Path,
    fixture_id: str,
) -> Iterable[SandboxViolation]:
    if not violation_log.exists():
        return ()
    violations: list[SandboxViolation] = []
    for line in violation_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        violations.append(
            SandboxViolation(
                fixture_id=fixture_id,
                kind=str(payload.get("kind") or "sandbox"),
                event=str(payload.get("event") or ""),
                operation=str(payload.get("operation") or "blocked"),
                path_hash=(
                    str(payload["path_hash"])
                    if payload.get("path_hash") is not None
                    else None
                ),
                detail=str(payload.get("detail") or ""),
            )
        )
    return tuple(violations)


def _scan_boundary_egress(
    fixtures: Sequence[BenchmarkFixture],
    runs: Sequence[_FederatedFixtureRun],
) -> BoundaryLeakageResult:
    surfaces = _gold_surfaces(fixtures)
    total_phi_bytes = sum(len(surface["bytes"]) for surface in surfaces)
    leaked_keys: set[tuple[str, int, int, str]] = set()
    findings: list[BoundaryLeakageFinding] = []
    emitted_by_sink: dict[str, int] = {}

    for run in runs:
        for artifact in run.artifacts:
            emitted_by_sink[artifact.sink] = emitted_by_sink.get(
                artifact.sink, 0
            ) + len(artifact.content)
            for surface in surfaces:
                offsets = tuple(_find_all(artifact.content, surface["bytes"]))
                if not offsets:
                    continue
                key = (
                    str(surface["fixture_id"]),
                    int(surface["start"]),
                    int(surface["end"]),
                    str(surface["label"]),
                )
                leaked_keys.add(key)
                findings.append(
                    BoundaryLeakageFinding(
                        fixture_id=str(surface["fixture_id"]),
                        sink=artifact.sink,
                        artifact=artifact.artifact,
                        start=int(surface["start"]),
                        end=int(surface["end"]),
                        label=str(surface["label"]),
                        length=int(surface["length"]),
                        text_hash=str(surface["text_hash"]),
                        byte_offsets=offsets,
                    )
                )

    leaked_bytes = sum(
        len(surface["bytes"])
        for surface in surfaces
        if (
            str(surface["fixture_id"]),
            int(surface["start"]),
            int(surface["end"]),
            str(surface["label"]),
        )
        in leaked_keys
    )
    rate = leaked_bytes / total_phi_bytes if total_phi_bytes else 0.0
    return BoundaryLeakageResult(
        rate=rate,
        leaked_bytes=leaked_bytes,
        total_phi_bytes=total_phi_bytes,
        findings=tuple(findings),
        emitted_bytes_by_sink=emitted_by_sink,
    )


def _gold_surfaces(fixtures: Sequence[BenchmarkFixture]) -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    for fixture in fixtures:
        for span in fixture.gold_spans:
            if not (0 <= span.start < span.end <= len(fixture.text)):
                continue
            surface = fixture.text[span.start : span.end]
            if not surface:
                continue
            key = (fixture.fixture_id, span.start, span.end, span.label)
            if key in seen:
                continue
            seen.add(key)
            evidence = hashed_span_surface(
                fixture.text,
                span.start,
                span.end,
                label=span.label,
            )
            surfaces.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "label": span.label,
                    "bytes": surface.encode("utf-8"),
                    **evidence,
                }
            )
    return surfaces


def _find_all(content: bytes, needle: bytes) -> list[int]:
    if not needle:
        return []
    offsets: list[int] = []
    start = 0
    while True:
        index = content.find(needle, start)
        if index < 0:
            break
        offsets.append(index)
        start = index + max(len(needle), 1)
    return offsets


def _resource_accounting(runs: Sequence[_FederatedFixtureRun]) -> dict[str, Any]:
    return {
        "process_count": len(runs),
        "total_elapsed_ms": sum(run.elapsed_ms for run in runs),
        "max_fixture_elapsed_ms": max((run.elapsed_ms for run in runs), default=0.0),
        "exit_codes": [run.exit_code for run in runs],
        "stdout_bytes": sum(
            len(artifact.content)
            for run in runs
            for artifact in run.artifacts
            if artifact.sink == "stdout"
        ),
        "stderr_bytes": sum(
            len(artifact.content)
            for run in runs
            for artifact in run.artifacts
            if artifact.sink == "stderr"
        ),
        "file_bytes": sum(
            len(artifact.content)
            for run in runs
            for artifact in run.artifacts
            if artifact.sink == "file"
        ),
    }


def _bytes_output(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8", errors="replace")


def _path_list(paths: Iterable[str | Path]) -> str:
    return os.pathsep.join(str(Path(path).expanduser().resolve()) for path in paths)


def _path_hash(path: str | Path) -> str:
    return _hash_bytes(str(Path(path).expanduser().resolve()).encode("utf-8"))


def _hash_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _slug_fixture_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return safe[:80] or "fixture"


_FEDERATED_CHILD_WRAPPER = r"""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

detector_path = str(Path(sys.argv[1]).resolve())
input_path = str(Path(sys.argv[2]).resolve())
output_path = str(Path(sys.argv[3]).resolve())
output_dir = str(Path(sys.argv[4]).resolve())
violation_log = str(Path(sys.argv[5]).resolve())
extra_read_roots = [
    str(Path(item).resolve())
    for item in sys.argv[6].split(os.pathsep)
    if item
]
violation_fd = os.open(
    violation_log,
    os.O_CREAT | os.O_WRONLY | os.O_APPEND,
    0o600,
)


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(
    kind: str,
    event: str,
    operation: str,
    path: object | None = None,
    detail: str = "",
) -> None:
    payload = {
        "kind": kind,
        "event": event,
        "operation": operation,
        "detail": detail,
    }
    if path is not None:
        payload["path_hash"] = _hash(str(Path(path).expanduser().resolve()))
    os.write(
        violation_fd,
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        ),
    )


def _roots(values: list[str]) -> list[str]:
    roots: list[str] = []
    for value in values:
        if not value:
            continue
        try:
            roots.append(str(Path(value).expanduser().resolve()))
        except OSError:
            continue
    return roots


READ_ROOTS = _roots(
    [
        detector_path,
        str(Path(detector_path).parent),
        input_path,
        output_path,
        output_dir,
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        sys.base_exec_prefix,
        *extra_read_roots,
    ]
)
WRITE_ROOTS = _roots([output_dir])


def _inside(path: object, roots: list[str]) -> bool:
    if not isinstance(path, (str, bytes, os.PathLike)):
        return True
    try:
        resolved = str(Path(path).expanduser().resolve())
    except OSError:
        resolved = os.path.abspath(os.fspath(path))
    for root in roots:
        try:
            if os.path.commonpath([resolved, root]) == root:
                return True
        except ValueError:
            continue
    return False


def _file_operation(mode: object, flags: object) -> str:
    if isinstance(mode, str):
        if any(marker in mode for marker in ("w", "a", "x", "+")):
            return "write"
        return "read"
    if isinstance(flags, int):
        write_mask = (
            os.O_WRONLY
            | os.O_RDWR
            | os.O_CREAT
            | os.O_TRUNC
            | os.O_APPEND
        )
        if flags & write_mask:
            return "write"
    return "read"


def _audit(event: str, args: tuple[object, ...]) -> None:
    if event.startswith("socket"):
        _record("network", event, "blocked")
        raise PermissionError("OPENMED_SANDBOX_VIOLATION network")
    if event in {
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.spawn",
        "os.fork",
        "os.exec",
    }:
        _record("process", event, "blocked")
        raise PermissionError("OPENMED_SANDBOX_VIOLATION process")
    if event == "open" and args:
        path = args[0]
        operation = _file_operation(
            args[1] if len(args) > 1 else None,
            args[2] if len(args) > 2 else None,
        )
        roots = WRITE_ROOTS if operation == "write" else READ_ROOTS
        if not _inside(path, roots):
            _record("filesystem", event, operation, path)
            raise PermissionError("OPENMED_SANDBOX_VIOLATION filesystem")
    if event in {"os.remove", "os.unlink", "os.rmdir"} and args:
        path = args[0]
        if not _inside(path, WRITE_ROOTS):
            _record("filesystem", event, "write", path)
            raise PermissionError("OPENMED_SANDBOX_VIOLATION filesystem")
    if event == "os.rename" and len(args) >= 2:
        for path in args[:2]:
            if not _inside(path, WRITE_ROOTS):
                _record("filesystem", event, "write", path)
                raise PermissionError("OPENMED_SANDBOX_VIOLATION filesystem")


sys.addaudithook(_audit)
os.environ["OPENMED_DETECTOR_INPUT"] = input_path
os.environ["OPENMED_DETECTOR_OUTPUT"] = output_path
os.environ["OPENMED_DETECTOR_OUTPUT_DIR"] = output_dir
runpy.run_path(detector_path, run_name="__main__")
""".lstrip()


def _shared_default_model_runner() -> ModelRunner:
    shared_loader: Any | None = None
    accepts_loader = _runner_accepts_loader(default_model_runner)

    def run_fixture(
        fixture: BenchmarkFixture,
        model_name: str,
        device: str,
    ) -> Iterable[Any]:
        nonlocal shared_loader
        if not accepts_loader:
            return default_model_runner(fixture, model_name, device)
        if shared_loader is None:
            from openmed.core.models import ModelLoader

            shared_loader = ModelLoader()
        return default_model_runner(
            fixture,
            model_name,
            device,
            loader=shared_loader,
        )

    return run_fixture


def _runner_accepts_loader(runner: Callable[..., Iterable[Any]]) -> bool:
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return True

    return any(
        parameter.name == "loader" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _attach_confidence_intervals(
    metrics: Mapping[str, Any],
    fixtures: Sequence[BenchmarkFixture],
    results: Sequence[FixtureResult],
    *,
    device: str,
    n_resamples: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap per-document CIs and merge them into the metric bundle."""
    result_by_id = {result.fixture_id: result for result in results}
    per_document_spans = [
        (
            fixture.gold_spans,
            getattr(result_by_id.get(fixture.fixture_id), "predicted_spans", ()),
        )
        for fixture in fixtures
    ]
    intervals = compute_confidence_intervals(
        per_document_spans,
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
        default_device=device,
    )
    merged = dict(metrics)
    for key, interval in intervals.items():
        metric = merged.get(key)
        if isinstance(metric, Mapping):
            merged[key] = {**metric, "confidence_interval": interval}
    return merged


def _attach_calibration_metrics(
    metrics: Mapping[str, Any],
    gold_spans: Sequence[EvalSpan],
    predicted_spans: Sequence[EvalSpan],
    *,
    n_bins: int,
) -> dict[str, Any]:
    """Merge reliability diagram data into the metric bundle."""
    bins = reliability_bins(
        _prediction_confidence_records(gold_spans, predicted_spans),
        n_bins=n_bins,
    )
    merged = dict(metrics)
    merged["calibration"] = {
        "expected_calibration_error": expected_calibration_error(bins),
        "reliability_bins": bins,
        "n_bins": n_bins,
    }
    return merged


def _resolve_abstention_thresholds(
    thresholds: Any | None,
    thresholds_path: str | Path | None,
) -> Any | None:
    if thresholds is not None:
        return thresholds
    if thresholds_path is not None:
        return load_calibration_thresholds(thresholds_path)
    return None


def _abstention_cache_hash(
    base_hash: str | None,
    *,
    thresholds: Any | None,
    thresholds_path: str | Path | None,
    confidence_threshold: float,
    target_risk: float | None,
    confidence_level: float | None,
    bootstrap_resamples: int,
    seed: int,
) -> str | None:
    if thresholds is None and thresholds_path is None:
        return base_hash
    digest = hashlib.sha256()
    digest.update(str(base_hash or "").encode("utf-8"))
    digest.update(b"\0abstention\0")
    payload = {
        "confidence_level": confidence_level,
        "confidence_threshold": confidence_threshold,
        "seed": seed,
        "target_risk": target_risk,
        "bootstrap_resamples": bootstrap_resamples,
        "thresholds": _jsonable_thresholds(thresholds),
        "thresholds_path": str(thresholds_path) if thresholds_path else None,
    }
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _jsonable_thresholds(thresholds: Any | None) -> Any:
    if thresholds is None:
        return None
    if hasattr(thresholds, "source_path") and thresholds.source_path:
        return {"source_path": thresholds.source_path}
    if hasattr(thresholds, "thresholds"):
        return {
            ".".join(key): value for key, value in sorted(thresholds.thresholds.items())
        }
    return thresholds


def _prediction_confidence_records(
    gold_spans: Sequence[EvalSpan],
    predicted_spans: Sequence[EvalSpan],
) -> list[dict[str, Any]]:
    matched_gold: set[int] = set()
    records: list[dict[str, Any]] = []
    for predicted in predicted_spans:
        correct = False
        for index, gold in enumerate(gold_spans):
            if index in matched_gold:
                continue
            if _exact_span_match(gold, predicted):
                matched_gold.add(index)
                correct = True
                break
        records.append(
            {
                "confidence": predicted.metadata.get("confidence", 1.0),
                "correct": correct,
            }
        )
    return records


def _exact_span_match(gold_span: EvalSpan, predicted_span: EvalSpan) -> bool:
    return (
        gold_span.label == predicted_span.label
        and gold_span.start == predicted_span.start
        and gold_span.end == predicted_span.end
    )


def _corpus_coordinates(
    fixtures: Sequence[BenchmarkFixture],
    results: Sequence[FixtureResult],
) -> tuple[list[EvalSpan], list[EvalSpan], str]:
    result_by_id = {result.fixture_id: result for result in results}
    gold: list[EvalSpan] = []
    predicted: list[EvalSpan] = []
    text_parts: list[str] = []
    offset = 0
    for fixture in fixtures:
        text_parts.append(fixture.text)
        gold.extend(_shift_spans(fixture.gold_spans, offset))
        result = result_by_id.get(fixture.fixture_id)
        if result is not None:
            predicted.extend(_shift_spans(result.predicted_spans, offset))
        offset += len(fixture.text) + 1
    return gold, predicted, "\n".join(text_parts)


def _relation_corpus_coordinates(
    fixtures: Sequence[Any],
    results: Sequence[RelationFixtureResult],
) -> tuple[list[RelationTriple], list[RelationTriple]]:
    result_by_id = {result.fixture_id: result for result in results}
    gold: list[RelationTriple] = []
    predicted: list[RelationTriple] = []
    offset = 0
    for fixture in fixtures:
        fixture_id = str(getattr(fixture, "fixture_id"))
        text = str(getattr(fixture, "text", ""))
        gold.extend(
            _shift_relation_triples(
                normalize_relation_triples(getattr(fixture, "relations", ())),
                offset,
            )
        )
        result = result_by_id.get(fixture_id)
        if result is not None:
            predicted.extend(
                _shift_relation_triples(result.predicted_relations, offset)
            )
        offset += len(text) + 1
    return gold, predicted


def _per_document_relations(
    fixtures: Sequence[Any],
    results: Sequence[RelationFixtureResult],
) -> list[tuple[tuple[RelationTriple, ...], tuple[RelationTriple, ...]]]:
    result_by_id = {result.fixture_id: result for result in results}
    documents: list[tuple[tuple[RelationTriple, ...], tuple[RelationTriple, ...]]] = []
    for fixture in fixtures:
        fixture_id = str(getattr(fixture, "fixture_id"))
        result = result_by_id.get(fixture_id)
        documents.append(
            (
                normalize_relation_triples(getattr(fixture, "relations", ())),
                result.predicted_relations if result is not None else (),
            )
        )
    return documents


def _shift_relation_triples(
    relations: Iterable[RelationTriple],
    offset: int,
) -> list[RelationTriple]:
    return [relation.shifted(offset) for relation in relations]


def _validate_relation_offsets(
    relation: RelationTriple,
    text: str,
    fixture_id: str,
) -> None:
    for argument_name, argument in (("arg1", relation.arg1), ("arg2", relation.arg2)):
        if (
            argument.start < 0
            or argument.end < argument.start
            or argument.end > len(text)
        ):
            raise ValueError(
                "invalid relation argument offsets "
                f"{fixture_id}:{argument_name} "
                f"{argument.start}:{argument.end} for text length {len(text)}"
            )


def _shift_spans(spans: Iterable[EvalSpan], offset: int) -> list[EvalSpan]:
    return [
        replace(span, start=span.start + offset, end=span.end + offset)
        for span in spans
    ]


def _validate_unique_fixture_ids(fixtures: Sequence[BenchmarkFixture]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for fixture in fixtures:
        if fixture.fixture_id in seen and fixture.fixture_id not in duplicates:
            duplicates.append(fixture.fixture_id)
        seen.add(fixture.fixture_id)
    if duplicates:
        quoted = ", ".join(repr(value) for value in duplicates)
        raise ValueError(f"duplicate benchmark fixture id(s): {quoted}")


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _key_bytes(key: bytes | str) -> bytes:
    if isinstance(key, bytes):
        if not key:
            raise ValueError("signing key must be non-empty")
        return key
    if isinstance(key, str):
        if not key:
            raise ValueError("signing key must be non-empty")
        return key.encode("utf-8")
    raise TypeError("signing key must be bytes or str")


__all__ = [
    "ModelRunner",
    "RelationModelRunner",
    "BenchmarkFixture",
    "BoundaryLeakageFinding",
    "BoundaryLeakageResult",
    "FederatedDetectorSpec",
    "FederatedEvalReport",
    "FixtureResult",
    "RelationFixtureResult",
    "SandboxViolation",
    "load_fixtures",
    "default_model_runner",
    "run_federated_leakage_eval",
    "run_benchmark",
    "run_relation_benchmark",
    "run_cross_lingual_transfer",
    "run_cross_lingual_transfer_suite",
    "run_suite",
]
