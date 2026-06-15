"""Evaluation package for section 4.2.

Intended contents include harness.py, metrics.py, suites/, golden/, report.py,
calibrate.py, and release_gates.py.
"""

from openmed.eval.harness import BenchmarkFixture, FixtureResult, run_benchmark, run_suite
from openmed.eval.metrics import (
    DEVICE_TIERS,
    EvalSpan,
    compute_character_recall,
    compute_clinical_utility_loss,
    compute_date_shift_consistency,
    compute_exact_span_f1,
    compute_latency_summary,
    compute_leakage_rate,
    compute_metrics_bundle,
    compute_over_redaction_loss,
    compute_recall_slices,
    compute_relaxed_span_f1,
    compute_resource_metrics,
    compute_surrogate_consistency,
)
from openmed.eval.report import BenchmarkReport
from openmed.eval.tiers import TIERS


__all__ = [
    "BenchmarkFixture",
    "BenchmarkReport",
    "DEVICE_TIERS",
    "EvalSpan",
    "FixtureResult",
    "TIERS",
    "compute_character_recall",
    "compute_clinical_utility_loss",
    "compute_date_shift_consistency",
    "compute_exact_span_f1",
    "compute_latency_summary",
    "compute_leakage_rate",
    "compute_metrics_bundle",
    "compute_over_redaction_loss",
    "compute_recall_slices",
    "compute_relaxed_span_f1",
    "compute_resource_metrics",
    "compute_surrogate_consistency",
    "run_benchmark",
    "run_suite",
]
