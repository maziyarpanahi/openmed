"""Clinical-validation harness scaffold.

This package provides a documented validation protocol and a reproducible study
runner over **user-supplied** labeled data. It reuses the :mod:`openmed.eval`
scoring stack (recall/precision/F1, leakage, subgroup fairness) rather than
reinventing scoring, and emits a signed, reproducible validation report as JSON
and Markdown with provenance hashes.

The scaffold supports internal clinical-validation studies. It does not certify
a model for clinical use and is not a medical device.
"""

from openmed.clinical.validation.protocol import (
    CLINICAL_VALIDATION_DISCLAIMER,
    DEFAULT_ACCEPTANCE_THRESHOLDS,
    LOWER_BOUND,
    SUBGROUP_AXES,
    UPPER_BOUND,
    VALIDATION_PROTOCOL_ID,
    VALIDATION_PROTOCOL_SCHEMA_VERSION,
    AcceptanceThreshold,
    coerce_acceptance_thresholds,
    default_thresholds_by_metric,
)
from openmed.clinical.validation.study import (
    VALIDATION_REPORT_SCHEMA_VERSION,
    AcceptanceResult,
    StudyConfig,
    ValidationReport,
    ValidationRunner,
    load_study_dataset,
    run_validation_study,
)

__all__ = [
    "AcceptanceResult",
    "AcceptanceThreshold",
    "CLINICAL_VALIDATION_DISCLAIMER",
    "DEFAULT_ACCEPTANCE_THRESHOLDS",
    "LOWER_BOUND",
    "SUBGROUP_AXES",
    "StudyConfig",
    "UPPER_BOUND",
    "VALIDATION_PROTOCOL_ID",
    "VALIDATION_PROTOCOL_SCHEMA_VERSION",
    "VALIDATION_REPORT_SCHEMA_VERSION",
    "ValidationReport",
    "ValidationRunner",
    "coerce_acceptance_thresholds",
    "default_thresholds_by_metric",
    "load_study_dataset",
    "run_validation_study",
]
