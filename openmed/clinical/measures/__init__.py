"""Deterministic clinical measure contracts and native evaluation."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from .contracts import (
    MEASURE_ADVISORY,
    MEASURE_COMPATIBILITY_POLICY,
    MEASURE_SCHEMA_VERSION,
    CalculationTraceStep,
    MeasureConflictError,
    MeasureContractError,
    MeasureDefinition,
    MeasureDriftReport,
    MeasureEngineIdentity,
    MeasureEvidence,
    MeasureLanguage,
    MeasurePopulationDefinition,
    MeasureRunResult,
    MeasureSubjectResult,
    MeasureTimeWindow,
    MeasureUnsupportedError,
    PopulationKind,
    PopulationResult,
    PopulationState,
    ValueSetBinding,
    compare_measure_runs,
)
from .native import (
    NATIVE_MEASURE_ENGINE,
    NativePopulationRule,
    evaluate_native_measure,
)


def load_measure_schema() -> dict[str, Any]:
    """Load the bundled JSON Schema for measure artifacts."""

    text = (
        resources.files("openmed.core.schemas.json")
        .joinpath("clinical_measure.schema.json")
        .read_text(encoding="utf-8")
    )
    value = json.loads(text)
    if not isinstance(value, dict):  # pragma: no cover - packaged invariant
        raise RuntimeError("clinical measure schema must be an object")
    return value


__all__ = [
    "MEASURE_ADVISORY",
    "MEASURE_COMPATIBILITY_POLICY",
    "MEASURE_SCHEMA_VERSION",
    "NATIVE_MEASURE_ENGINE",
    "CalculationTraceStep",
    "MeasureConflictError",
    "MeasureContractError",
    "MeasureDefinition",
    "MeasureDriftReport",
    "MeasureEngineIdentity",
    "MeasureEvidence",
    "MeasureLanguage",
    "MeasurePopulationDefinition",
    "MeasureRunResult",
    "MeasureSubjectResult",
    "MeasureTimeWindow",
    "MeasureUnsupportedError",
    "NativePopulationRule",
    "PopulationKind",
    "PopulationResult",
    "PopulationState",
    "ValueSetBinding",
    "compare_measure_runs",
    "evaluate_native_measure",
    "load_measure_schema",
]
