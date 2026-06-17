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
from openmed.eval.calibrate import (
    CalibrationArtifactPaths,
    CalibrationGroupReport,
    CalibrationReport,
    CalibrationSample,
    CalibrationThresholdSet,
    artifact_dir_for,
    build_thresholds_payload,
    coerce_calibration_thresholds,
    default_suite_calibration_samples,
    fit_calibration_thresholds,
    load_calibration_samples,
    load_calibration_thresholds,
    write_calibration_artifacts,
)
from openmed.eval.release_gates import (
    QUARANTINED,
    RELEASABLE,
    GateCheck,
    GateReport,
    ModelStewardConfig,
    ReleaseGate,
)


__all__ = [
    "BenchmarkFixture",
    "BenchmarkReport",
    "CalibrationArtifactPaths",
    "CalibrationGroupReport",
    "CalibrationReport",
    "CalibrationSample",
    "CalibrationThresholdSet",
    "DEVICE_TIERS",
    "EvalSpan",
    "FixtureResult",
    "GateCheck",
    "GateReport",
    "ModelStewardConfig",
    "QUARANTINED",
    "RELEASABLE",
    "ReleaseGate",
    "artifact_dir_for",
    "build_thresholds_payload",
    "coerce_calibration_thresholds",
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
    "default_suite_calibration_samples",
    "fit_calibration_thresholds",
    "load_calibration_samples",
    "load_calibration_thresholds",
    "run_benchmark",
    "run_suite",
    "write_calibration_artifacts",
]
