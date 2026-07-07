"""Clinical-validation protocol: metrics, acceptance thresholds, disclaimers.

This module encodes the *documented* validation protocol that
:mod:`openmed.clinical.validation.study` executes. It defines the primary and
secondary metrics, deterministic acceptance thresholds, the subgroup axes the
study reports on, and the medical-device disclaimer that every report carries.

The protocol is intentionally leakage-first: PHI leakage is a gating primary
metric, evaluated with an upper-bound threshold, while detection quality
(recall/precision/F1) is evaluated with lower-bound thresholds.

The scaffold supports clinical-validation studies. It does **not** certify a
model for clinical use and is not a medical device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

VALIDATION_PROTOCOL_SCHEMA_VERSION = "openmed.clinical_validation_protocol.v1"

#: Human-readable protocol identifier recorded in every validation report.
VALIDATION_PROTOCOL_ID = "openmed-deid-clinical-validation-v1"

#: Disclaimer stamped on every validation report and rendered document. The
#: validation scaffold measures de-identification quality against user-supplied
#: labels; it supports but does not certify clinical use.
CLINICAL_VALIDATION_DISCLAIMER = (
    "This validation report measures de-identification quality against "
    "user-supplied labeled data. It supports internal clinical-validation "
    "studies but does not certify a model for clinical use and is not a "
    "medical device. Results do not replace institutional review, regulatory "
    "clearance, or clinical judgment."
)

#: Subgroup axes reported for fairness. ``group`` reads the fixture metadata
#: group key; ``language`` uses the fixture language.
SUBGROUP_AXES: tuple[str, ...] = ("group", "language")

# Direction of the acceptance comparison for each metric.
LOWER_BOUND = "lower_bound"  # observed value must be >= threshold
UPPER_BOUND = "upper_bound"  # observed value must be <= threshold


@dataclass(frozen=True)
class AcceptanceThreshold:
    """One deterministic acceptance criterion for a protocol metric."""

    metric: str
    direction: str
    threshold: float
    primary: bool
    description: str

    def evaluate(self, observed: float) -> bool:
        """Return whether *observed* satisfies this acceptance criterion."""

        if self.direction == LOWER_BOUND:
            return observed >= self.threshold
        if self.direction == UPPER_BOUND:
            return observed <= self.threshold
        raise ValueError(f"unknown acceptance direction: {self.direction}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping for this criterion."""

        return {
            "metric": self.metric,
            "direction": self.direction,
            "threshold": self.threshold,
            "primary": self.primary,
            "description": self.description,
        }


#: Default acceptance thresholds. These are conservative scaffold defaults; a
#: study MAY override them through the runner configuration. Leakage is a
#: gating primary metric with an upper bound; detection quality metrics use
#: lower bounds.
DEFAULT_ACCEPTANCE_THRESHOLDS: tuple[AcceptanceThreshold, ...] = (
    AcceptanceThreshold(
        metric="leakage_rate",
        direction=UPPER_BOUND,
        threshold=0.01,
        primary=True,
        description="Character-weighted PHI leakage rate must not exceed 1%.",
    ),
    AcceptanceThreshold(
        metric="recall",
        direction=LOWER_BOUND,
        threshold=0.95,
        primary=True,
        description="Exact-span detection recall must be at least 0.95.",
    ),
    AcceptanceThreshold(
        metric="precision",
        direction=LOWER_BOUND,
        threshold=0.90,
        primary=False,
        description="Exact-span detection precision must be at least 0.90.",
    ),
    AcceptanceThreshold(
        metric="f1",
        direction=LOWER_BOUND,
        threshold=0.92,
        primary=False,
        description="Exact-span detection F1 must be at least 0.92.",
    ),
    AcceptanceThreshold(
        metric="subgroup_leakage_disparity",
        direction=UPPER_BOUND,
        threshold=0.01,
        primary=False,
        description=(
            "Leakage disparity across subgroups (max minus min group leakage) "
            "must not exceed 1 percentage point."
        ),
    ),
)


def default_thresholds_by_metric() -> dict[str, AcceptanceThreshold]:
    """Return the default acceptance thresholds keyed by metric name."""

    return {threshold.metric: threshold for threshold in DEFAULT_ACCEPTANCE_THRESHOLDS}


def coerce_acceptance_thresholds(
    overrides: Mapping[str, Any] | None,
) -> tuple[AcceptanceThreshold, ...]:
    """Merge caller *overrides* onto the default acceptance thresholds.

    Overrides is a mapping of metric name to a partial mapping of the fields to
    replace (``threshold``, ``direction``, ``primary``, ``description``).
    Unknown metric names are rejected so a study cannot silently define a
    criterion that is never evaluated.
    """

    defaults = default_thresholds_by_metric()
    if not overrides:
        return DEFAULT_ACCEPTANCE_THRESHOLDS

    merged: dict[str, AcceptanceThreshold] = dict(defaults)
    for metric, override in overrides.items():
        if metric not in defaults:
            raise ValueError(f"unknown acceptance metric: {metric}")
        base = defaults[metric]
        if not isinstance(override, Mapping):
            override = {"threshold": override}
        merged[metric] = AcceptanceThreshold(
            metric=metric,
            direction=str(override.get("direction", base.direction)),
            threshold=float(override.get("threshold", base.threshold)),
            primary=bool(override.get("primary", base.primary)),
            description=str(override.get("description", base.description)),
        )
    return tuple(merged[metric] for metric in defaults)


__all__ = [
    "AcceptanceThreshold",
    "CLINICAL_VALIDATION_DISCLAIMER",
    "DEFAULT_ACCEPTANCE_THRESHOLDS",
    "LOWER_BOUND",
    "SUBGROUP_AXES",
    "UPPER_BOUND",
    "VALIDATION_PROTOCOL_ID",
    "VALIDATION_PROTOCOL_SCHEMA_VERSION",
    "coerce_acceptance_thresholds",
    "default_thresholds_by_metric",
]
