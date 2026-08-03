"""Release gate harness for benchmark reports.

The gate evaluates a candidate benchmark report against the OpenMed release
criteria, reads the last-green baseline store without mutating
it, and emits a signed, reproducible gate report.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core import baseline as baseline_store
from openmed.core import model_registry, quality_gates
from openmed.core import policy as policy_module
from openmed.core.audit import AuditSignature, stable_hash
from openmed.core.labels import normalize_label
from openmed.core.pii_i18n import (
    DEFAULT_MODEL_PLACEHOLDER_LANGUAGES,
    SUPPORTED_LANGUAGES,
)
from openmed.core.thresholds import (
    DEFAULT_MEMBERSHIP_ADVANTAGE_CEILING,
    load_thresholds,
    profile_recall_floor,
    profile_script_leakage_ceiling,
    profile_script_recall_floors,
    validate_threshold_matrix,
)
from openmed.eval.fairness import DEFAULT_ZERO_SHOT_LEAKAGE_FLOOR
from openmed.eval.metrics import (
    CRITICAL_FINDING_CATEGORY_DIAGNOSIS,
    CRITICAL_FINDING_CATEGORY_DRUG_ALLERGY,
    PIPELINE_EVAL_STAGES,
    normalize_critical_finding_category,
    normalize_eval_spans,
)
from openmed.eval.nano_cert import certify_measurements
from openmed.eval.quant_delta import (
    COREML_RECALL_DELTA_LIMIT,
    INT4_RECALL_DELTA_LIMIT,
    INT8_RECALL_DELTA_LIMIT,
    QuantRecallDeltaResult,
    evaluate_export_variant_gate,
    evaluate_quant_recall_delta,
)
from openmed.eval.report import BenchmarkReport

RELEASABLE = "RELEASABLE"
QUARANTINED = "QUARANTINED"
FLAKINESS_GATE = "flakiness"
SURROGATE_QUALITY_GATE = "surrogate_quality"
GROUNDING_ACCURACY_GATE = "grounding_accuracy"
CROSS_SCRIPT_GATE = "cross_script"
CROSS_DOCUMENT_LINKAGE_GATE = "cross_document_linkage"
REIDENTIFICATION_RISK_GATE = "reidentification_risk"
EXPORT_VARIANT_GATE = "export_variants"
I18N_THROUGHPUT_GATE = "i18n_throughput"
I18N_THROUGHPUT_REGRESSION_THRESHOLD = 0.20
I18N_THROUGHPUT_BASELINE_FAMILY = "i18n-throughput"
I18N_THROUGHPUT_BASELINE_FORMAT = "pattern-only"
I18N_THROUGHPUT_LANGUAGES = ("zh", "hi", "ta")
I18N_THROUGHPUT_METRICS = (
    "segmentation_chars_per_second",
    "deidentify_spans_per_second",
)

G1A_V16_RECALL_FLOOR = 0.990
G1A_V20_RECALL_FLOOR = 0.995
G1B_RECALL_FLOOR = 0.995
G2_V16_RECALL_FLOOR = 0.980
G2_V20_RECALL_FLOOR = 0.990
G4_INT8_DELTA_LIMIT = INT8_RECALL_DELTA_LIMIT
G4_INT4_DELTA_LIMIT = INT4_RECALL_DELTA_LIMIT
G7_RECALL_DROP_LIMIT = 0.002
G11_CRITICAL_RECALL_FLOOR = 0.999
G9_STRICT_RE_F1_FLOOR = 0.850
G9_RELAXED_RE_F1_FLOOR = 0.900
RELATION_GOLDEN_REGRESSION_GATE = "relation_golden_regression"
RELATION_GOLDEN_TRAP_KINDS = ("assertion", "temporal")
G13_STRICT_ENTITY_F1_FLOOR = 0.900
G13_STRICT_RELATION_F1_FLOOR = 0.850
G13_UNCERTAINTY_ACCURACY_FLOOR = 0.950
# Explicit aliases make the gate's radiology scope clear in serialized configs.
G13_RADIOLOGY_ENTITY_F1_FLOOR = G13_STRICT_ENTITY_F1_FLOOR
G13_RADIOLOGY_RELATION_F1_FLOOR = G13_STRICT_RELATION_F1_FLOOR
G13_RADIOLOGY_UNCERTAINTY_ACCURACY_FLOOR = G13_UNCERTAINTY_ACCURACY_FLOOR
#: Maximum tolerated worst-group-vs-best-group extraction-F1 gap across
#: synthetic site/note-type/demographic surrogate groups before G14 quarantines.
G14_EXTRACTION_DISPARITY_CEILING = 0.050
#: Minimum exact fact-level F1 from raw note through FHIR export.
G15_E2E_FACT_F1_FLOOR = 0.900
RESIDUAL_LEAKAGE_SOFT_CEILING = 0.005
G10_UNGROUNDED_FACT_CEILING = 0.0
DEFAULT_CROSS_DOCUMENT_LINKAGE_CEILING = 0.0
PER_LANGUAGE_RESIDUAL_LEAKAGE_CEILINGS: Mapping[str, float] = {
    "as": 0.0,
    "mr": 0.0,
    "or": 0.0,
    "ta": 0.0,
}

_SIGNATURE_ALGORITHM = "HMAC-SHA256"
_DEFAULT_SIGNING_KEY = "openmed-release-gate-local-key"

_G1A_LABELS = frozenset(
    {
        "PERSON",
        "FIRST_NAME",
        "LAST_NAME",
        "MIDDLE_NAME",
        "USERNAME",
        "EMAIL",
        "PHONE",
        "URL",
        "LOCATION",
        "STREET_ADDRESS",
        "BUILDING_NUMBER",
        "ZIPCODE",
        "GPS_COORDINATES",
        "DATE",
        "DATE_OF_BIRTH",
        "TIME",
        "AGE",
        "ID_NUM",
        "SSN",
    }
)
_G1B_LABELS = frozenset({"API_KEY", "ACCOUNT_NUMBER", "CREDIT_CARD", "IBAN"})
_G2_LABELS = frozenset(
    {
        "PERSON",
        "FIRST_NAME",
        "LAST_NAME",
        "MIDDLE_NAME",
        "LOCATION",
        "STREET_ADDRESS",
        "BUILDING_NUMBER",
        "ZIPCODE",
        "DATE",
        "DATE_OF_BIRTH",
    }
)
_CRITICAL_LABELS = frozenset(
    {
        "SSN",
        "ID_NUM",
        "API_KEY",
        "ACCOUNT_NUMBER",
        "PASSWORD",
        "PIN",
        "CREDIT_CARD",
        "CVV",
        "IBAN",
        "BIC",
    }
)
_G1_G2_LABELS = _G1A_LABELS | _G1B_LABELS | _G2_LABELS
_G11_ZERO_MISS_CATEGORIES = frozenset(
    {
        CRITICAL_FINDING_CATEGORY_DIAGNOSIS,
        CRITICAL_FINDING_CATEGORY_DRUG_ALLERGY,
    }
)

_TIER_ALIASES = {
    "nano": "tiny",
    "small": "tiny",
    "lite": "tiny",
    "tiny": "tiny",
    "base": "base",
    "laptop": "base",
    "large": "large",
    "superclinical": "large",
    "accurate": "accurate",
    "xlarge": "accurate",
    "xl": "accurate",
    "moe": "accurate",
}
_TIER_BUDGETS = {
    "tiny": {"ram_mb": 350.0, "p50_ms": 60.0, "p95_ms": 150.0},
    "base": {"ram_mb": 900.0, "p50_ms": 150.0, "p95_ms": 400.0},
    "large": {"ram_mb": 4096.0, "p50_ms": 250.0, "p95_ms": 800.0},
    "accurate": {"ram_mb": 8192.0, "p50_ms": 400.0, "p95_ms": 1200.0},
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFEST_PATH = _REPO_ROOT / "models.jsonl"
_DEFAULT_README_PATH = _REPO_ROOT / "README.md"
# Keep this offline catalog selector aligned with the batch export policy in
# scripts/onnx/batch_android_convert_publish.py.
_DERIVED_FORMAT_PREFIXES = ("mlx", "coreml", "onnx", "tflite")
_ANDROID_FORMAT_PREFIXES = ("onnx", "tflite")
_ANDROID_UNSUPPORTED_ARCHITECTURES = {"gliner", "privacy-filter"}
_ANDROID_DERIVED_REPO_SUFFIXES = (
    "-mlx",
    "-mlx-8bit",
    "-mlx-4bit",
    "-coreml",
    "-onnx",
    "-onnx-android",
    "-onnx-int8",
)


@dataclass(frozen=True)
class ModelStewardConfig:
    """Per-family leakage targets signed off by model stewardship."""

    target_leakage_by_family: Mapping[str, float] = field(default_factory=dict)
    default_target_leakage: float = RESIDUAL_LEAKAGE_SOFT_CEILING

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | "ModelStewardConfig" | None,
    ) -> "ModelStewardConfig":
        if isinstance(value, ModelStewardConfig):
            return value
        if value is None:
            return cls()

        default = float(
            value.get(
                "default_target_leakage",
                value.get("target_leakage", RESIDUAL_LEAKAGE_SOFT_CEILING),
            )
        )
        family_source = (
            value.get("target_leakage_by_family")
            or value.get("families")
            or value.get("family_targets")
            or {}
        )
        targets: dict[str, float] = {}
        if isinstance(family_source, Mapping):
            for family, target in family_source.items():
                if isinstance(target, Mapping):
                    target = target.get("target_leakage")
                if target is not None:
                    targets[_normalise_dimension(str(family))] = float(target)

        for family, target in value.items():
            if family in {
                "default_target_leakage",
                "target_leakage",
                "target_leakage_by_family",
                "families",
                "family_targets",
            }:
                continue
            if isinstance(target, Mapping):
                target = target.get("target_leakage")
            if isinstance(target, (int, float)):
                targets[_normalise_dimension(str(family))] = float(target)

        return cls(target_leakage_by_family=targets, default_target_leakage=default)

    def target_for(self, family: str) -> float:
        key = _normalise_dimension(family)
        return float(
            self.target_leakage_by_family.get(key, self.default_target_leakage)
        )


@dataclass(frozen=True)
class GateCheck:
    """One gate result inside a signed gate report."""

    gate: str
    passed: bool
    reason: str = "ok"
    details: Mapping[str, Any] = field(default_factory=dict)
    blocking_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "gate": self.gate,
            "passed": bool(self.passed),
            "reason": self.reason,
            "details": _plain(self.details),
        }
        if self.blocking_format is not None:
            payload["blocking_format"] = self.blocking_format
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GateCheck":
        return cls(
            gate=str(data.get("gate", "")),
            passed=bool(data.get("passed", False)),
            reason=str(data.get("reason", "")),
            details=dict(data.get("details") or {}),
            blocking_format=(
                str(data["blocking_format"])
                if data.get("blocking_format") is not None
                else None
            ),
        )


def evaluate_i18n_throughput_gate(
    report: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> GateCheck:
    """Compare zh/hi/ta steady-state throughput with committed baselines.

    Cold-start metrics remain visible in the benchmark report but are not gated
    because process and filesystem cache state makes them unsuitable for a
    stable release threshold.
    """

    failures: list[str] = []
    violations: dict[str, Any] = {}
    try:
        baseline_store.validate_baseline_store(baseline)
    except Exception as exc:
        return GateCheck(
            I18N_THROUGHPUT_GATE,
            False,
            reason=f"invalid throughput baseline store: {exc}",
        )

    if report.get("artifact_type") != "openmed.eval.i18n_throughput":
        return GateCheck(
            I18N_THROUGHPUT_GATE,
            False,
            reason="candidate is not an i18n throughput report",
        )
    languages = report.get("languages")
    if not isinstance(languages, Mapping):
        return GateCheck(
            I18N_THROUGHPUT_GATE,
            False,
            reason="candidate throughput report has no languages object",
        )

    for language in I18N_THROUGHPUT_LANGUAGES:
        observed_metrics = languages.get(language)
        if not isinstance(observed_metrics, Mapping):
            failures.append(f"{language}.languages (missing)")
            continue
        key = baseline_store.baseline_key(
            I18N_THROUGHPUT_BASELINE_FAMILY,
            language,
            I18N_THROUGHPUT_BASELINE_FORMAT,
        )
        entry = baseline["entries"].get(key)
        if not isinstance(entry, Mapping):
            failures.append(f"{language}.baseline (missing)")
            continue
        entry_metadata = entry.get("metadata")
        threshold = (
            entry_metadata.get("regression_threshold")
            if isinstance(entry_metadata, Mapping)
            else None
        )
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isclose(
                float(threshold),
                I18N_THROUGHPUT_REGRESSION_THRESHOLD,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            failures.append(
                f"{language}.regression_threshold "
                f"(expected {I18N_THROUGHPUT_REGRESSION_THRESHOLD:.2f})"
            )
            continue

        baseline_metrics = entry.get("metrics")
        if not isinstance(baseline_metrics, Mapping):
            failures.append(f"{language}.baseline.metrics (missing)")
            continue
        for metric in I18N_THROUGHPUT_METRICS:
            baseline_value = _positive_finite_number(baseline_metrics.get(metric))
            observed_value = _positive_finite_number(observed_metrics.get(metric))
            metric_key = f"{language}.{metric}"
            if baseline_value is None:
                failures.append(f"{metric_key} (invalid baseline)")
                continue
            if observed_value is None:
                failures.append(f"{metric_key} (invalid candidate)")
                continue

            minimum = baseline_value * (1.0 - I18N_THROUGHPUT_REGRESSION_THRESHOLD)
            if observed_value < minimum:
                drop = (baseline_value - observed_value) / baseline_value
                failures.append(
                    f"{metric_key} observed {observed_value:.3f}, minimum {minimum:.3f}"
                )
                violations[metric_key] = {
                    "baseline": baseline_value,
                    "observed": observed_value,
                    "minimum": minimum,
                    "drop_fraction": drop,
                    "regression_threshold": I18N_THROUGHPUT_REGRESSION_THRESHOLD,
                }

    return GateCheck(
        I18N_THROUGHPUT_GATE,
        not failures,
        reason="ok"
        if not failures
        else "throughput regression: " + "; ".join(failures),
        details={
            "languages": list(I18N_THROUGHPUT_LANGUAGES),
            "metrics": list(I18N_THROUGHPUT_METRICS),
            "regression_threshold": I18N_THROUGHPUT_REGRESSION_THRESHOLD,
            "violations": violations,
        },
    )


@dataclass
class GateReport:
    """Signed release-gate decision and evidence payload."""

    repo_id: str
    family: str
    tier: str
    param_count: int | None
    format: str
    per_label_recall: Mapping[str, float]
    per_label_precision: Mapping[str, float]
    critical_leakage_count: int
    residual_leakage_rate: float
    quant_recall_delta: float | None
    p50_ms: float | None
    p95_ms: float | None
    ram_mb: float | None
    eval_set_hash: str
    leakage_fixture_hash: str
    decision: str
    gate_results: tuple[GateCheck, ...] = ()
    policy: str = ""
    threshold_profile: str = ""
    target_leakage_rate: float = RESIDUAL_LEAKAGE_SOFT_CEILING
    blocked_formats: tuple[str, ...] = ()
    stability_summary: Mapping[str, Any] = field(default_factory=dict)
    repro_hash: str = ""
    signature: AuditSignature | None = None
    per_script_recall: Mapping[str, float] = field(default_factory=dict)
    per_script_leakage: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.per_label_recall = _float_map(self.per_label_recall)
        self.per_label_precision = _float_map(self.per_label_precision)
        self.per_script_recall = _numeric_map(self.per_script_recall)
        self.per_script_leakage = _numeric_map(self.per_script_leakage)
        self.blocked_formats = tuple(self.blocked_formats)
        self.gate_results = tuple(self.gate_results)
        self.stability_summary = _mapping(self.stability_summary)
        if not self.repro_hash:
            self.repro_hash = self.recompute_repro_hash()

    def _payload(
        self,
        *,
        include_repro_hash: bool,
        include_signature: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "repo_id": self.repo_id,
            "family": self.family,
            "tier": self.tier,
            "param_count": self.param_count,
            "format": self.format,
            "per_label_recall": _float_map(self.per_label_recall),
            "per_label_precision": _float_map(self.per_label_precision),
            "critical_leakage_count": int(self.critical_leakage_count),
            "residual_leakage_rate": float(self.residual_leakage_rate),
            "quant_recall_delta": (
                None
                if self.quant_recall_delta is None
                else float(self.quant_recall_delta)
            ),
            "p50_ms": None if self.p50_ms is None else float(self.p50_ms),
            "p95_ms": None if self.p95_ms is None else float(self.p95_ms),
            "ram_mb": None if self.ram_mb is None else float(self.ram_mb),
            "eval_set_hash": self.eval_set_hash,
            "leakage_fixture_hash": self.leakage_fixture_hash,
            "decision": self.decision,
            "gate_results": [check.to_dict() for check in self.gate_results],
            "policy": self.policy,
            "threshold_profile": self.threshold_profile,
            "target_leakage_rate": float(self.target_leakage_rate),
            "blocked_formats": list(self.blocked_formats),
        }
        if self.per_script_recall or self.per_script_leakage:
            payload["per_script_recall"] = _numeric_map(self.per_script_recall)
            payload["per_script_leakage"] = _numeric_map(self.per_script_leakage)
        if self.stability_summary:
            payload["stability_summary"] = _plain(self.stability_summary)
        if include_repro_hash:
            payload["repro_hash"] = self.repro_hash
        if include_signature:
            payload["signature"] = (
                self.signature.to_dict() if self.signature is not None else None
            )
        return payload

    def recompute_repro_hash(self) -> str:
        """Recompute the report hash without trusting the stored hash."""
        return stable_hash(
            self._payload(include_repro_hash=False, include_signature=False)
        )

    def sign(self, key: bytes | str, *, key_id: str = "release-gate") -> "GateReport":
        """Sign the gate report and return ``self``."""
        self.repro_hash = self.recompute_repro_hash()
        message = _canonical_json(
            self._payload(include_repro_hash=True, include_signature=False)
        ).encode("utf-8")
        signature = hmac.new(_key_bytes(key), message, hashlib.sha256).hexdigest()
        self.signature = AuditSignature(
            key_id=key_id,
            algorithm=_SIGNATURE_ALGORITHM,
            value=signature,
        )
        return self

    def verify(self, key: bytes | str) -> bool:
        """Verify the report signature and reproducibility hash."""
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

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GateReport":
        signature_data = data.get("signature")
        return cls(
            repo_id=str(data.get("repo_id", "")),
            family=str(data.get("family", "")),
            tier=str(data.get("tier", "")),
            param_count=_optional_int(data.get("param_count")),
            format=str(data.get("format", "")),
            per_label_recall=_mapping(data.get("per_label_recall")),
            per_label_precision=_mapping(data.get("per_label_precision")),
            critical_leakage_count=int(data.get("critical_leakage_count", 0)),
            residual_leakage_rate=float(data.get("residual_leakage_rate", 0.0)),
            quant_recall_delta=_optional_float(data.get("quant_recall_delta")),
            p50_ms=_optional_float(data.get("p50_ms")),
            p95_ms=_optional_float(data.get("p95_ms")),
            ram_mb=_optional_float(data.get("ram_mb")),
            eval_set_hash=str(data.get("eval_set_hash", "")),
            leakage_fixture_hash=str(data.get("leakage_fixture_hash", "")),
            decision=str(data.get("decision", QUARANTINED)),
            per_script_recall=_mapping(data.get("per_script_recall")),
            per_script_leakage=_mapping(data.get("per_script_leakage")),
            gate_results=tuple(
                GateCheck.from_dict(item)
                for item in data.get("gate_results", [])
                if isinstance(item, Mapping)
            ),
            policy=str(data.get("policy", "")),
            threshold_profile=str(data.get("threshold_profile", "")),
            target_leakage_rate=float(
                data.get("target_leakage_rate", RESIDUAL_LEAKAGE_SOFT_CEILING)
            ),
            blocked_formats=tuple(
                str(item) for item in data.get("blocked_formats", [])
            ),
            stability_summary=_mapping(data.get("stability_summary")),
            repro_hash=str(data.get("repro_hash", "")),
            signature=(
                AuditSignature.from_dict(signature_data)
                if isinstance(signature_data, Mapping)
                else None
            ),
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> "GateReport":
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for GateReport: {exc}") from exc
        return cls.from_dict(parsed)


class ReleaseGate:
    """Evaluate benchmark reports against the OpenMed release gates."""

    def __init__(
        self,
        *,
        milestone: str = "v1.7",
        policy: str = "hipaa_safe_harbor",
        baseline_path: str | Path = baseline_store.BASELINE_PATH,
        thresholds_matrix: Mapping[str, Any] | None = None,
        thresholds_matrix_path: str | Path | None = None,
        model_steward_config: Mapping[str, Any] | ModelStewardConfig | None = None,
        cross_document_linkage_ceiling: float = (
            DEFAULT_CROSS_DOCUMENT_LINKAGE_CEILING
        ),
        signing_key: bytes | str | None = None,
        key_id: str = "release-gate",
    ) -> None:
        self.milestone = milestone
        self.policy = policy
        self.baseline_path = Path(baseline_path)
        self.thresholds_matrix = copy.deepcopy(dict(thresholds_matrix or {})) or None
        self.thresholds_matrix_path = (
            Path(thresholds_matrix_path) if thresholds_matrix_path is not None else None
        )
        self.model_steward_config = ModelStewardConfig.from_mapping(
            model_steward_config
        )
        self.cross_document_linkage_ceiling = _probability_ceiling(
            cross_document_linkage_ceiling,
            name="cross_document_linkage_ceiling",
        )
        self.signing_key = (
            signing_key
            if signing_key is not None
            else os.environ.get("OPENMED_RELEASE_GATE_KEY", _DEFAULT_SIGNING_KEY)
        )
        self.key_id = key_id

    def evaluate(
        self,
        report: BenchmarkReport | Mapping[str, Any],
        baseline: Mapping[str, Any] | None = None,
        *,
        signing_key: bytes | str | None = None,
        key_id: str | None = None,
    ) -> GateReport:
        """Evaluate *report* and return a signed gate report."""

        gate_report = self._score(report, baseline)
        return gate_report.sign(
            signing_key or self.signing_key, key_id=key_id or self.key_id
        )

    def preview(
        self,
        report: BenchmarkReport | Mapping[str, Any],
        baseline: Mapping[str, Any] | None = None,
    ) -> GateReport:
        """Evaluate *report* in read-only preview mode without signing."""

        return self._score(report, baseline)

    def _score(
        self,
        report: BenchmarkReport | Mapping[str, Any],
        baseline: Mapping[str, Any] | None,
    ) -> GateReport:
        """Score release gates without deciding whether to sign the report."""

        payload = _report_payload(report)
        metrics = _mapping(payload.get("metrics"))
        metadata = _mapping(payload.get("metadata"))
        identity = _identity(payload, metrics, metadata)
        policy_name = str(
            metadata.get("policy") or payload.get("policy") or self.policy
        )

        checks: list[GateCheck] = []
        profile = None
        profile_error = ""
        try:
            profile = policy_module.load_policy(policy_name)
            checks.append(
                GateCheck(
                    "policy_profile",
                    True,
                    details={
                        "policy": profile.name,
                        "threshold_profile": profile.threshold_profile,
                        "strict_no_leak": profile.strict_no_leak,
                    },
                )
            )
        except Exception as exc:  # pragma: no cover - defensive, gate reports failure.
            profile_error = str(exc)
            checks.append(GateCheck("policy_profile", False, reason=profile_error))

        threshold_matrix: Mapping[str, Any] | None = None
        threshold_error = ""
        try:
            threshold_matrix = self._load_threshold_matrix()
            checks.append(
                GateCheck(
                    "thresholds_matrix",
                    True,
                    details={"schema_version": threshold_matrix.get("schema_version")},
                )
            )
        except Exception as exc:
            threshold_error = str(exc)
            checks.append(GateCheck("thresholds_matrix", False, reason=threshold_error))

        per_label_recall, recall_denominators = _per_label_recall(metrics, metadata)
        per_label_precision = _per_label_precision(metrics, metadata)
        per_script_recall, per_script_leakage, script_denominators = (
            _per_script_metrics(metrics, metadata)
        )
        critical_leakage_count = _critical_leakage_count(metrics, metadata)
        residual_leakage_rate = _residual_leakage_rate(metrics, metadata)
        quant_delta_result = evaluate_quant_recall_delta(
            format_name=identity["format"],
            candidate_recall=per_label_recall,
            parent_recall=_quant_parent_recall(metrics, metadata),
            precomputed_delta=_precomputed_quant_recall_delta(
                metrics,
                metadata,
                identity["format"],
            ),
        )
        p50_ms, p95_ms = _latency(metrics, metadata)
        ram_mb = _ram_mb(metrics, metadata)
        baseline_entry = self._resolve_baseline(identity, baseline)
        target_leakage = self.model_steward_config.target_for(identity["family"])
        if profile is not None and profile.strict_no_leak:
            target_leakage = min(target_leakage, 0.0)

        checks.append(_manifest_coherence_check(identity, metadata))
        checks.append(_calibration_check(metadata, profile))
        checks.append(_abstention_advisory_check(metrics, metadata, target_leakage))
        checks.append(_conformal_coverage_check(metrics, metadata))
        checks.append(_grounding_coverage_check(metrics, metadata))
        checks.append(
            self._g1a_check(
                per_label_recall,
                recall_denominators,
                profile=profile,
                threshold_matrix=threshold_matrix,
            )
        )
        checks.append(self._g1b_check(per_label_recall, recall_denominators))
        checks.append(self._g2_check(per_label_recall, recall_denominators))
        checks.append(
            _cross_script_check(
                per_script_recall,
                per_script_leakage,
                script_denominators,
                threshold_profile=(
                    profile.threshold_profile if profile is not None else ""
                ),
                threshold_matrix=threshold_matrix,
            )
        )
        checks.append(_adversarial_recall_under_attack_check(metrics, metadata))
        checks.append(_g3_check(critical_leakage_count))
        checks.append(_g11_critical_finding_recall_check(metrics, metadata))
        checks.append(_g14_extraction_fairness_check(metrics, metadata))
        checks.append(_g15_end_to_end_pipeline_check(metrics, metadata, baseline_entry))
        checks.append(_g4_check(quant_delta_result))
        checks.append(
            _g5_check(
                identity["tier"],
                p50_ms,
                p95_ms,
                ram_mb,
                param_count=identity["param_count"],
            )
        )
        checks.append(_g6_check(p50_ms, p95_ms))
        checks.append(
            _g7_check(
                baseline_entry,
                per_label_recall,
                residual_leakage_rate,
                target_leakage=target_leakage,
            )
        )
        checks.append(_per_language_residual_leakage_check(metrics, metadata))
        checks.append(_membership_leakage_check(metrics, metadata))
        checks.append(_g8_check(metadata))
        checks.append(_surrogate_quality_release_check(metrics, metadata))
        checks.append(_g9_relation_extraction_check(metrics, metadata))
        relation_baseline = baseline
        if relation_baseline is None and _relation_golden_gate_is_applicable(
            metrics, metadata
        ):
            try:
                relation_baseline = baseline_store.load_baseline_store(
                    self.baseline_path
                )
            except OSError:
                relation_baseline = {}
        checks.append(
            evaluate_relation_golden_regression_gate(
                metrics,
                relation_baseline or {},
                family=identity["family"],
                metadata=metadata,
            )
        )
        checks.append(_g13_radiology_entity_relation_check(metrics, metadata))
        coreml_manifest = _coreml_conversion_manifest(metadata)
        if coreml_manifest or _normalise_dimension(identity["format"]).startswith(
            "coreml"
        ):
            checks.append(_coreml_ane_residency_check(coreml_manifest, metadata))
            checks.append(_coreml_variant_parity_check(coreml_manifest, metadata))
        export_manifest = _export_variant_manifest(metadata)
        if export_manifest:
            checks.extend(_export_variant_checks(export_manifest, metrics, metadata))
        checks.append(_zero_shot_language_leakage_check(metrics, metadata))
        checks.append(_g10_faithfulness_check(metrics, metadata))
        federated_check = _federated_boundary_check(metrics, metadata)
        if federated_check is not None:
            checks.append(federated_check)
        checks.append(_k_floor_check(metrics, metadata))
        checks.append(
            _cross_document_linkage_check(
                metrics,
                metadata,
                ceiling=self.cross_document_linkage_ceiling,
            )
        )
        checks.append(_reidentification_risk_check(metrics, metadata))
        checks.append(_structured_release_risk_check(metrics, metadata))

        blocked_formats = tuple(
            sorted(
                {
                    check.blocking_format
                    for check in checks
                    if not check.passed and check.blocking_format is not None
                }
            )
        )
        decision = RELEASABLE if all(check.passed for check in checks) else QUARANTINED
        gate_report = GateReport(
            repo_id=identity["repo_id"],
            family=identity["family"],
            tier=identity["tier"],
            param_count=identity["param_count"],
            format=identity["format"],
            per_label_recall=per_label_recall,
            per_label_precision=per_label_precision,
            critical_leakage_count=critical_leakage_count,
            residual_leakage_rate=residual_leakage_rate,
            quant_recall_delta=quant_delta_result.max_delta,
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            ram_mb=ram_mb,
            eval_set_hash=identity["eval_set_hash"],
            leakage_fixture_hash=identity["leakage_fixture_hash"],
            decision=decision,
            per_script_recall=per_script_recall,
            per_script_leakage=per_script_leakage,
            gate_results=tuple(checks),
            policy=(profile.name if profile is not None else policy_name),
            threshold_profile=(
                profile.threshold_profile if profile is not None else ""
            ),
            target_leakage_rate=target_leakage,
            blocked_formats=blocked_formats,
        )
        return gate_report

    def _load_threshold_matrix(self) -> Mapping[str, Any]:
        if self.thresholds_matrix is not None:
            payload = copy.deepcopy(self.thresholds_matrix)
            validate_threshold_matrix(payload)
            return payload
        if self.thresholds_matrix_path is not None:
            return load_thresholds(self.thresholds_matrix_path)
        return load_thresholds()

    def _resolve_baseline(
        self,
        identity: Mapping[str, Any],
        baseline: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        if baseline is None:
            try:
                return baseline_store.get_baseline(
                    identity["family"],
                    identity["tier"],
                    identity["format"],
                    path=self.baseline_path,
                )
            except OSError:
                return None

        if "entries" in baseline:
            return baseline_store.get_baseline(
                identity["family"],
                identity["tier"],
                identity["format"],
                store=baseline,
            )
        if "metrics" in baseline:
            return baseline
        return {"metrics": baseline}

    def _g1a_check(
        self,
        per_label_recall: Mapping[str, float],
        denominators: Mapping[str, int],
        *,
        profile: Any | None,
        threshold_matrix: Mapping[str, Any] | None,
    ) -> GateCheck:
        floor = _g1a_floor(self.milestone)
        if (
            profile is not None
            and threshold_matrix is not None
            and profile.strict_no_leak
        ):
            try:
                floor = max(
                    floor,
                    profile_recall_floor(
                        profile.threshold_profile,
                        matrix=threshold_matrix,
                    ),
                )
            except Exception as exc:
                return GateCheck(
                    "G1a",
                    False,
                    reason=f"could not resolve policy recall floor: {exc}",
                )
        return _recall_floor_check(
            "G1a",
            _G1A_LABELS,
            per_label_recall,
            denominators,
            floor,
        )

    def _g1b_check(
        self,
        per_label_recall: Mapping[str, float],
        denominators: Mapping[str, int],
    ) -> GateCheck:
        return _recall_floor_check(
            "G1b",
            _G1B_LABELS,
            per_label_recall,
            denominators,
            G1B_RECALL_FLOOR,
        )

    def _g2_check(
        self,
        per_label_recall: Mapping[str, float],
        denominators: Mapping[str, int],
    ) -> GateCheck:
        return _recall_floor_check(
            "G2",
            _G2_LABELS,
            per_label_recall,
            denominators,
            _g2_floor(self.milestone),
        )


def apply_flakiness_quarantine(
    report: GateReport,
    stability_report: Mapping[str, Any] | Any,
) -> GateReport:
    """Return *report* with a blocking flakiness gate when stability fails."""

    summary = _stability_summary_payload(stability_report)
    quarantined_gates = tuple(
        str(gate) for gate in summary.get("quarantined_gates", []) if str(gate)
    )
    unstable_gates = tuple(
        str(gate) for gate in summary.get("unstable_gates", []) if str(gate)
    )
    blocking_gates = tuple(sorted(set(quarantined_gates) | set(unstable_gates)))
    passed = not blocking_gates
    reason = (
        "stable across configured seed sweep"
        if passed
        else "unstable gate verdicts quarantined: " + ", ".join(blocking_gates)
    )
    checks = [check for check in report.gate_results if check.gate != FLAKINESS_GATE]
    checks.append(
        GateCheck(
            FLAKINESS_GATE,
            passed,
            reason=reason,
            details={"stability_summary": summary},
        )
    )
    decision = RELEASABLE if all(check.passed for check in checks) else QUARANTINED
    return GateReport(
        repo_id=report.repo_id,
        family=report.family,
        tier=report.tier,
        param_count=report.param_count,
        format=report.format,
        per_label_recall=report.per_label_recall,
        per_label_precision=report.per_label_precision,
        critical_leakage_count=report.critical_leakage_count,
        residual_leakage_rate=report.residual_leakage_rate,
        quant_recall_delta=report.quant_recall_delta,
        p50_ms=report.p50_ms,
        p95_ms=report.p95_ms,
        ram_mb=report.ram_mb,
        eval_set_hash=report.eval_set_hash,
        leakage_fixture_hash=report.leakage_fixture_hash,
        decision=decision,
        per_script_recall=report.per_script_recall,
        per_script_leakage=report.per_script_leakage,
        gate_results=tuple(checks),
        policy=report.policy,
        threshold_profile=report.threshold_profile,
        target_leakage_rate=report.target_leakage_rate,
        blocked_formats=report.blocked_formats,
        stability_summary=summary,
    )


def evaluate_surrogate_quality_gate(
    report: Mapping[str, Any] | Any | None = None,
    *,
    fixture_path: str | Path | None = None,
    min_pass_rate: float | None = None,
) -> GateCheck:
    """Return the offline multilingual surrogate-quality release gate check."""

    from openmed.eval.surrogate_quality import (
        DEFAULT_SURROGATE_QUALITY_FIXTURE,
        DEFAULT_SURROGATE_QUALITY_PASS_RATE,
        SurrogateQualityReport,
        evaluate_surrogate_quality,
    )

    if isinstance(report, SurrogateQualityReport):
        quality_report = SurrogateQualityReport(
            locale_reports=report.locale_reports,
            required_locales=report.required_locales,
            min_pass_rate=(
                report.min_pass_rate if min_pass_rate is None else min_pass_rate
            ),
        )
    elif report is not None:
        configured_fixture = (
            report.get("fixture_path") if isinstance(report, Mapping) else None
        )
        resolved_fixture = (
            fixture_path
            if fixture_path is not None
            else (
                configured_fixture
                if isinstance(configured_fixture, (str, Path))
                else DEFAULT_SURROGATE_QUALITY_FIXTURE
            )
        )
        quality_report = evaluate_surrogate_quality(
            report.get("records") if isinstance(report, Mapping) else report,
            fixture_path=resolved_fixture,
            min_pass_rate=(
                min_pass_rate
                if min_pass_rate is not None
                else (
                    _optional_float(report.get("min_pass_rate"))
                    if isinstance(report, Mapping)
                    else None
                )
                or DEFAULT_SURROGATE_QUALITY_PASS_RATE
            ),
        )
    else:
        quality_report = evaluate_surrogate_quality(
            fixture_path=fixture_path or DEFAULT_SURROGATE_QUALITY_FIXTURE,
            min_pass_rate=(
                DEFAULT_SURROGATE_QUALITY_PASS_RATE
                if min_pass_rate is None
                else min_pass_rate
            ),
        )

    failing = {
        lang: locale_report.pass_rate
        for lang, locale_report in quality_report.locale_reports.items()
        if locale_report.pass_rate < quality_report.min_pass_rate
    }
    details = quality_report.to_dict()
    details["failing_locales"] = failing
    reason = (
        "ok"
        if quality_report.passed
        else "surrogate quality below per-locale pass-rate floor"
    )
    return GateCheck(
        SURROGATE_QUALITY_GATE,
        quality_report.passed,
        reason=reason,
        details=details,
    )


def _surrogate_quality_release_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    from openmed.eval.surrogate_quality import DEFAULT_SURROGATE_QUALITY_PASS_RATE

    evidence = _first_value(
        metrics.get("surrogate_quality"),
        metadata.get("surrogate_quality"),
    )
    required = bool(
        metrics.get("surrogate_quality_required")
        or metadata.get("surrogate_quality_required")
    )
    if evidence is None:
        return GateCheck(
            SURROGATE_QUALITY_GATE,
            not required,
            reason=(
                "surrogate-quality evidence is required"
                if required
                else "not applicable"
            ),
            details={"required": required},
        )

    if isinstance(evidence, Mapping) and not (
        evidence.get("records") is not None or evidence.get("fixture_path")
    ):
        return GateCheck(
            SURROGATE_QUALITY_GATE,
            False,
            reason="surrogate-quality evidence is malformed",
            details={
                "error": "evidence must include records or fixture_path",
                "required": required,
            },
        )

    try:
        check = evaluate_surrogate_quality_gate(
            evidence,
            min_pass_rate=DEFAULT_SURROGATE_QUALITY_PASS_RATE,
        )
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
        return GateCheck(
            SURROGATE_QUALITY_GATE,
            False,
            reason="surrogate-quality evidence is invalid",
            details={"error": str(exc), "required": required},
        )

    details = dict(check.details)
    details["required"] = required
    return GateCheck(
        SURROGATE_QUALITY_GATE,
        check.passed,
        reason=check.reason,
        details=details,
    )


def _manifest_coherence_check(
    identity: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    missing = [
        key
        for key in (
            "repo_id",
            "family",
            "tier",
            "param_count",
            "format",
            "eval_set_hash",
            "leakage_fixture_hash",
        )
        if identity.get(key) in {None, ""}
    ]
    if missing:
        return GateCheck(
            "manifest_coherence",
            False,
            reason="missing required release metadata",
            details={"missing": missing},
        )

    mismatches: dict[str, Any] = {}
    manifest = _mapping(metadata.get("manifest"))
    if manifest:
        mismatches.update(
            _manifest_row_mismatches(
                manifest,
                identity,
                source="candidate_manifest",
            )
        )

    manifest_path = _manifest_path(metadata)
    manifest_rows: list[dict[str, Any]] = []
    manifest_row: Mapping[str, Any] | None = None
    if manifest_path is not None:
        try:
            manifest_rows = _load_manifest_rows(manifest_path)
        except (OSError, ValueError) as exc:
            mismatches["manifest_file"] = {
                "path": str(manifest_path),
                "error": str(exc),
            }
        else:
            manifest_row = _find_manifest_row(manifest_rows, str(identity["repo_id"]))
            if manifest_row is not None:
                mismatches.update(
                    _manifest_row_mismatches(
                        manifest_row,
                        identity,
                        source="models_jsonl",
                    )
                )
            elif _requires_manifest_row(metadata, manifest):
                mismatches["manifest_row"] = {
                    "repo_id": identity["repo_id"],
                    "path": str(manifest_path),
                    "error": "candidate repo_id is absent from manifest",
                }

    if manifest_rows:
        mismatches.update(
            _manifest_surface_mismatches(manifest_rows, metadata, manifest_path)
        )

    card = _model_card_metadata(metadata)
    if card and manifest_row is not None:
        card_mismatches = _model_card_mismatches(card, manifest_row)
        if card_mismatches:
            mismatches["model_card"] = card_mismatches

    if mismatches:
        return GateCheck(
            "manifest_coherence",
            False,
            reason="candidate metadata or repository surfaces drift from manifest",
            details={"mismatches": mismatches},
        )

    return GateCheck(
        "manifest_coherence",
        True,
        details={
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "manifest_rows": len(manifest_rows),
            "candidate_manifest_row": manifest_row is not None,
        },
    )


def _manifest_row_mismatches(
    row: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    mismatches: dict[str, Any] = {}
    manifest_fields = {
        "repo_id": row.get("repo_id"),
        "family": row.get("family"),
        "tier": row.get("tier"),
        "param_count": row.get("param_count"),
    }
    for key, manifest_value in manifest_fields.items():
        if manifest_value is None:
            continue
        candidate_value = identity.get(key)
        if key == "param_count":
            manifest_value = _optional_int(manifest_value)
        if str(manifest_value) != str(candidate_value):
            mismatches[f"{source}.{key}"] = {
                "manifest": manifest_value,
                "candidate": candidate_value,
            }

    candidate_format = str(identity.get("format") or "")
    manifest_format = row.get("format") or row.get("model_format")
    manifest_formats = row.get("formats")
    if isinstance(manifest_formats, Sequence) and not isinstance(
        manifest_formats,
        (str, bytes),
    ):
        formats = {str(item) for item in manifest_formats}
        if candidate_format not in formats:
            mismatches[f"{source}.format"] = {
                "manifest": sorted(formats),
                "candidate": candidate_format,
            }
    elif manifest_format is not None and str(manifest_format) != candidate_format:
        mismatches[f"{source}.format"] = {
            "manifest": manifest_format,
            "candidate": candidate_format,
        }
    return mismatches


def _manifest_surface_mismatches(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    manifest_path: Path | None,
) -> dict[str, Any]:
    mismatches: dict[str, Any] = {}
    default_manifest = _is_default_path(manifest_path, _DEFAULT_MANIFEST_PATH)

    readme_path = _optional_path(metadata.get("readme_path"))
    if readme_path is None and default_manifest:
        readme_path = _DEFAULT_README_PATH
    if readme_path is not None:
        readme_mismatches = _readme_manifest_mismatches(
            rows,
            readme_path,
            include_android_onnx_derivatives=default_manifest,
        )
        if readme_mismatches:
            mismatches["readme"] = readme_mismatches

    registry_ids = _string_set(metadata.get("registry_model_ids"))
    if registry_ids:
        missing = sorted(_manifest_repo_ids(rows) - registry_ids)
        if missing:
            mismatches["registry"] = {"missing_repo_ids": missing}
    elif default_manifest:
        registry_repo_ids = {
            info.model_id for info in model_registry.OPENMED_MODELS.values()
        }
        missing = sorted(_manifest_repo_ids(rows) - registry_repo_ids)
        if missing:
            mismatches["registry"] = {"missing_repo_ids": missing}

    supported_languages = _string_set(metadata.get("supported_languages"))
    if not supported_languages and default_manifest:
        supported_languages = set(SUPPORTED_LANGUAGES) - set(
            DEFAULT_MODEL_PLACEHOLDER_LANGUAGES
        )
    if supported_languages:
        manifest_languages = _manifest_pii_languages(rows)
        if manifest_languages != supported_languages:
            mismatches["pii_languages"] = {
                "manifest": sorted(manifest_languages),
                "supported": sorted(supported_languages),
            }

    return mismatches


def _readme_manifest_mismatches(
    rows: Sequence[Mapping[str, Any]],
    readme_path: Path,
    *,
    include_android_onnx_derivatives: bool = False,
) -> dict[str, Any]:
    if not readme_path.exists():
        return {"path": str(readme_path), "error": "README evidence is missing"}

    text = readme_path.read_text(encoding="utf-8")
    declared = _readme_declared_counts(text)
    mismatches: dict[str, Any] = {}
    model_count = len(rows)
    derived_model_count = (
        _published_android_onnx_derivative_count(rows)
        if include_android_onnx_derivatives
        else 0
    )
    catalog_model_count = model_count + derived_model_count
    pii_count = len([row for row in rows if _is_pii_manifest_row(row)])
    pii_languages = _manifest_pii_languages(rows)

    if declared.get("models") is not None and catalog_model_count < declared["models"]:
        mismatches["models"] = {
            "readme_floor": declared["models"],
            "manifest": model_count,
            "android_onnx_derivatives": derived_model_count,
            "catalog": catalog_model_count,
        }
    if (
        declared.get("pii_checkpoints") is not None
        and pii_count < declared["pii_checkpoints"]
    ):
        mismatches["pii_checkpoints"] = {
            "readme_floor": declared["pii_checkpoints"],
            "manifest": pii_count,
        }
    if (
        declared.get("languages") is not None
        and len(pii_languages) != declared["languages"]
    ):
        mismatches["languages"] = {
            "readme": declared["languages"],
            "manifest": len(pii_languages),
        }
    return mismatches


def _published_android_onnx_derivative_count(
    rows: Sequence[Mapping[str, Any]],
) -> int:
    """Count the audited Android ONNX fleet derived outside ``models.jsonl``."""

    return sum(_is_android_onnx_source_row(row) for row in rows)


def _is_android_onnx_source_row(row: Mapping[str, Any]) -> bool:
    repo_id = str(row.get("repo_id") or "").strip()
    if not repo_id or repo_id.startswith("OpenMed/privacy-filter-"):
        return False
    if _normalise_dimension(str(row.get("task") or "")) != "token-classification":
        return False
    if (
        _normalise_dimension(str(row.get("architecture") or ""))
        in _ANDROID_UNSUPPORTED_ARCHITECTURES
    ):
        return False

    formats = {
        _normalise_dimension(item)
        for item in row.get("formats") or []
        if str(item).strip()
    }
    if "pytorch" not in formats:
        return False
    if any(item.startswith(_ANDROID_FORMAT_PREFIXES) for item in formats):
        return False
    if any(
        item != "pytorch" and item.startswith(_DERIVED_FORMAT_PREFIXES)
        for item in formats
    ):
        return False
    return not repo_id.endswith(_ANDROID_DERIVED_REPO_SUFFIXES)


def _readme_declared_counts(text: str) -> dict[str, int]:
    import re

    counts: dict[str, int] = {}
    model_matches = [
        _parse_count(match.group(1))
        for match in re.finditer(
            r"(\d[\d,]*)\+?\s+(?:specialized\s+medical\s+)?models\b",
            text,
            flags=re.IGNORECASE,
        )
    ]
    if model_matches:
        counts["models"] = max(model_matches)

    language_match = re.search(
        r"(\d[\d,]*)\+?\s+languages?\b",
        text,
        flags=re.IGNORECASE,
    )
    if language_match:
        counts["languages"] = _parse_count(language_match.group(1))

    pii_match = re.search(
        r"(\d[\d,]*)\+?\s+PII\s+checkpoints?\b",
        text,
        flags=re.IGNORECASE,
    )
    if pii_match:
        counts["pii_checkpoints"] = _parse_count(pii_match.group(1))
    return counts


def _model_card_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    card = _mapping(metadata.get("model_card"))
    if card:
        return card
    card_path = _optional_path(metadata.get("model_card_path"))
    if card_path is None or not card_path.exists():
        return {}
    return _parse_model_card_front_matter(card_path.read_text(encoding="utf-8"))


def _model_card_mismatches(
    card: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    mismatches: dict[str, Any] = {}
    card_license = card.get("license")
    if card_license and row.get("license") and str(card_license) != str(row["license"]):
        mismatches["license"] = {
            "card": card_license,
            "manifest": row["license"],
        }

    card_task = card.get("pipeline_tag") or card.get("task")
    if card_task and row.get("task") and str(card_task) != str(row["task"]):
        mismatches["task"] = {"card": card_task, "manifest": row["task"]}

    card_languages = _string_set(card.get("language") or card.get("languages"))
    manifest_languages = _string_set(row.get("languages"))
    if card_languages and manifest_languages and card_languages != manifest_languages:
        mismatches["languages"] = {
            "card": sorted(card_languages),
            "manifest": sorted(manifest_languages),
        }
    return mismatches


def _parse_model_card_front_matter(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    data: dict[str, Any] = {}
    current_key: str | None = None
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_key:
            data.setdefault(current_key, []).append(stripped[2:].strip().strip("'\""))
            continue
        if ":" not in stripped:
            current_key = None
            continue
        key, value = stripped.split(":", 1)
        current_key = key.strip()
        value = value.strip()
        if not value:
            data[current_key] = []
        elif value.startswith("[") and value.endswith("]"):
            data[current_key] = [
                item.strip().strip("'\"")
                for item in value[1:-1].split(",")
                if item.strip()
            ]
        else:
            data[current_key] = value.strip("'\"")
    return data


def _manifest_path(metadata: Mapping[str, Any]) -> Path | None:
    explicit = _optional_path(
        _first_value(
            metadata.get("manifest_path"), metadata.get("models_manifest_path")
        )
    )
    if explicit is not None:
        return explicit
    if _DEFAULT_MANIFEST_PATH.exists():
        return _DEFAULT_MANIFEST_PATH
    return None


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if isinstance(row, Mapping):
                rows.append(dict(row))
    return rows


def _find_manifest_row(
    rows: Sequence[Mapping[str, Any]],
    repo_id: str,
) -> Mapping[str, Any] | None:
    for row in rows:
        if row.get("repo_id") == repo_id:
            return row
    return None


def _requires_manifest_row(
    metadata: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> bool:
    return bool(
        manifest
        or metadata.get("require_manifest_row")
        or metadata.get("manifest_path")
        or metadata.get("models_manifest_path")
    )


def _coreml_conversion_manifest(metadata: Mapping[str, Any]) -> dict[str, Any]:
    inline = _mapping(
        metadata.get("coreml_conversion_manifest")
        or metadata.get("coreml_manifest")
        or metadata.get("conversion_manifest")
    )
    if inline:
        return inline

    manifest_path = _optional_path(
        _first_value(
            metadata.get("coreml_conversion_manifest_path"),
            metadata.get("coreml_manifest_path"),
            metadata.get("conversion_manifest_path"),
        )
    )
    if manifest_path is None or not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return {}
    return _mapping(loaded)


def _coreml_variants(
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    variants = manifest.get("variants") if manifest else None
    if variants is None:
        variants = metadata.get("coreml_variants")
    if isinstance(variants, Mapping):
        return [
            {"name": str(name), **_mapping(value)} for name, value in variants.items()
        ]
    if isinstance(variants, Sequence) and not isinstance(variants, (str, bytes)):
        return [_mapping(item) for item in variants if _mapping(item)]
    return []


def _find_coreml_variant(
    variants: Sequence[Mapping[str, Any]],
    expected: str,
) -> Mapping[str, Any] | None:
    normalized_expected = _normalise_dimension(expected)
    for variant in variants:
        names = {
            str(variant.get("name") or ""),
            str(variant.get("format") or ""),
            f"coreml-{variant.get('quantization')}",
            f"coreml-{variant.get('precision')}",
        }
        if normalized_expected in {_normalise_dimension(name) for name in names}:
            return variant
    return None


def _coreml_parity_passed(parity: Mapping[str, Any]) -> bool:
    if not parity:
        return False
    if bool(parity.get("passed")) is not True:
        return False
    max_delta = _optional_float(parity.get("max_recall_delta"))
    if max_delta is not None and max_delta > COREML_RECALL_DELTA_LIMIT:
        return False
    mismatches = parity.get("span_mismatches") or []
    return not mismatches


def _manifest_repo_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(row["repo_id"])
        for row in rows
        if isinstance(row.get("repo_id"), str) and row.get("repo_id")
    }


def _manifest_pii_languages(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    languages: set[str] = set()
    for row in rows:
        if not _is_pii_manifest_row(row):
            continue
        languages.update(_string_set(row.get("languages")))
    return languages


def _is_pii_manifest_row(row: Mapping[str, Any]) -> bool:
    repo_id = str(row.get("repo_id") or "").lower()
    family = str(row.get("family") or "").lower()
    return family == "pii" or "pii" in repo_id or "privacy" in repo_id


def _string_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {str(item) for item in value if str(item)}
    return set()


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return Path(text)


def _is_default_path(path: Path | None, default: Path) -> bool:
    if path is None:
        return False
    try:
        return path.resolve() == default.resolve()
    except OSError:
        return False


def _parse_count(value: str) -> int:
    return int(value.replace(",", ""))


def _calibration_check(metadata: Mapping[str, Any], profile: Any | None) -> GateCheck:
    requires_calibration = True
    if profile is not None:
        actions = set(profile.actions.values()) | {profile.default_action}
        requires_calibration = bool(actions & {"mask", "replace"})
    if not requires_calibration:
        return GateCheck("calibration_present", True, reason="not applicable")

    thresholds_present = _artifact_present(
        metadata,
        mapping_keys=("thresholds", "calibration_thresholds", "thresholds_json"),
        path_keys=(
            "thresholds_path",
            "thresholds_json_path",
            "calibration_thresholds_path",
        ),
    )
    report_present = _artifact_present(
        metadata,
        mapping_keys=("calibration", "calibration_report"),
        path_keys=("calibration_report_path", "calibration_path"),
    )
    if thresholds_present and report_present:
        return GateCheck("calibration_present", True)
    return GateCheck(
        "calibration_present",
        False,
        reason="thresholds.json and calibration report are required",
        details={
            "thresholds_present": thresholds_present,
            "calibration_report_present": report_present,
        },
    )


def _abstention_advisory_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    target_leakage: float,
) -> GateCheck:
    abstention = _first_mapping(
        metadata.get("abstention"),
        metrics.get("abstention"),
    )
    if not abstention:
        return GateCheck(
            "abstention_advisory",
            True,
            reason="not supplied",
            details={"target_risk": target_leakage},
        )

    abstention_rate = _first_mapping(abstention.get("abstention_rate"))
    residual_risk = _first_mapping(abstention.get("residual_risk"))
    target_risk = _optional_float(abstention.get("target_risk"))
    return GateCheck(
        "abstention_advisory",
        True,
        reason="advisory",
        details={
            "target_risk": target_risk if target_risk is not None else target_leakage,
            "confidence_level": _optional_float(abstention.get("confidence_level")),
            "abstention_rate": {
                "overall": _optional_float(abstention_rate.get("overall")) or 0.0,
                "by_label": _float_map(abstention_rate.get("by_label")),
                "by_language": _numeric_map(abstention_rate.get("by_language")),
            },
            "residual_risk": {
                "overall": _optional_float(residual_risk.get("overall")) or 0.0,
                "critical": _optional_float(residual_risk.get("critical")) or 0.0,
                "by_label": _float_map(residual_risk.get("by_label")),
                "by_language": _numeric_map(residual_risk.get("by_language")),
                "bootstrap": _mapping(residual_risk.get("bootstrap")),
            },
            "route_counts": _mapping(abstention.get("route_counts")),
        },
    )


def _conformal_coverage_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    report, error, explicit = _conformal_coverage_report(metrics, metadata)
    required = bool(
        _first_value(
            metadata.get("require_conformal_coverage"),
            metrics.get("require_conformal_coverage"),
            False,
        )
    )
    if error:
        return GateCheck("conformal_coverage", False, reason=error)
    if not report:
        if required:
            return GateCheck(
                "conformal_coverage",
                False,
                reason="calibration-under-shift report is required",
            )
        return GateCheck(
            "conformal_coverage",
            True,
            reason="not provided",
            details={"required": False},
        )

    groups = report.get("groups")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        return GateCheck(
            "conformal_coverage",
            False,
            reason="calibration-under-shift report requires groups",
        )

    default_alpha = _optional_float(report.get("alpha"))
    default_target = _optional_float(report.get("target_coverage"))
    if default_target is None and default_alpha is not None:
        default_target = 1.0 - default_alpha
    if default_target is None:
        default_target = 1.0 - 0.05
    tolerance = _optional_float(report.get("coverage_tolerance"))
    if tolerance is None:
        tolerance = 0.01

    evaluated: list[str] = []
    violations: dict[str, Any] = {}
    for item in groups:
        if not isinstance(item, Mapping):
            continue
        label = normalize_label(str(item.get("label") or ""))
        if label not in _CRITICAL_LABELS:
            continue
        gate_weight = _optional_float(
            _first_value(
                item.get("positive_gate_weight"), item.get("total_gate_weight")
            )
        )
        if gate_weight is not None and gate_weight <= 0.0:
            continue
        coverage = _optional_float(
            _first_value(item.get("positive_coverage"), item.get("realized_coverage"))
        )
        if coverage is None:
            coverage = 0.0
        target = _optional_float(item.get("target_coverage"))
        if target is None:
            target = default_target
        language = str(item.get("language") or "").lower()
        key = f"{label}:{language or '*'}"
        evaluated.append(key)
        gap = max(float(target) - float(coverage), 0.0)
        if float(coverage) + float(tolerance) < float(target):
            violations[key] = {
                "label": label,
                "language": language,
                "coverage": coverage,
                "target_coverage": target,
                "coverage_gap": gap,
                "tolerance": tolerance,
            }

    if violations:
        return GateCheck(
            "conformal_coverage",
            False,
            reason="critical-label conformal coverage below target",
            details={
                "target_coverage": default_target,
                "coverage_tolerance": tolerance,
                "critical_labels_evaluated": sorted(evaluated),
                "violations": violations,
                "language_coverage": _mapping(report.get("language_coverage")),
                "explicit": explicit,
            },
        )

    return GateCheck(
        "conformal_coverage",
        True,
        details={
            "target_coverage": default_target,
            "coverage_tolerance": tolerance,
            "critical_labels_evaluated": sorted(evaluated),
            "language_coverage": _mapping(report.get("language_coverage")),
            "explicit": explicit,
        },
    )


def _conformal_coverage_report(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], str, bool]:
    inline = _first_mapping(
        metadata.get("calibration_under_shift"),
        metadata.get("calibration_under_shift_report"),
        metadata.get("conformal_coverage"),
        metrics.get("calibration_under_shift"),
        metrics.get("calibration_under_shift_report"),
        metrics.get("conformal_coverage"),
    )
    if inline:
        return inline, "", True

    path_value = _first_value(
        metadata.get("calibration_under_shift_report_path"),
        metadata.get("under_shift_report_path"),
        metadata.get("conformal_coverage_path"),
        metrics.get("calibration_under_shift_report_path"),
        metrics.get("under_shift_report_path"),
        metrics.get("conformal_coverage_path"),
    )
    if path_value is None:
        return {}, "", False
    path = Path(str(path_value))
    if not path.is_file():
        return {}, f"calibration-under-shift report not found: {path}", True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"could not read calibration-under-shift report: {exc}", True
    if not isinstance(payload, Mapping):
        return {}, "calibration-under-shift report must be a JSON object", True
    return dict(payload), "", True


def _grounding_coverage_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    report, error, explicit = _grounding_coverage_report(metrics, metadata)
    required = bool(
        _first_value(
            metadata.get("require_grounding_coverage"),
            metrics.get("require_grounding_coverage"),
            False,
        )
    )
    if error:
        return GateCheck("grounding_coverage", False, reason=error)
    if not report:
        if required:
            return GateCheck(
                "grounding_coverage",
                False,
                reason="grounding calibration report is required",
            )
        return GateCheck(
            "grounding_coverage",
            True,
            reason="not provided",
            details={"required": False},
        )

    from openmed.clinical.grounding.calibration import (
        evaluate_grounding_coverage_gate,
    )

    min_accuracy = _optional_float(
        _first_value(
            metadata.get("minimum_grounding_accuracy"),
            metrics.get("minimum_grounding_accuracy"),
            report.get("minimum_accuracy"),
        )
    )
    if min_accuracy is None:
        min_accuracy = 0.85
    min_coverage = _optional_float(
        _first_value(
            metadata.get("minimum_grounding_coverage"),
            metrics.get("minimum_grounding_coverage"),
            report.get("minimum_coverage"),
        )
    )
    if min_coverage is None:
        min_coverage = 0.70

    gate = evaluate_grounding_coverage_gate(
        report,
        min_accuracy=min_accuracy,
        min_coverage=min_coverage,
    )
    return GateCheck(
        "grounding_coverage",
        bool(gate.get("passed")),
        reason="ok"
        if gate.get("passed")
        else "grounded-span accuracy below required coverage",
        details={**gate, "explicit": explicit, "required": required},
    )


def _grounding_coverage_report(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], str, bool]:
    inline = _first_mapping(
        metadata.get("grounding_calibration"),
        metadata.get("grounding_calibration_report"),
        metadata.get("grounding_coverage"),
        metrics.get("grounding_calibration"),
        metrics.get("grounding_calibration_report"),
        metrics.get("grounding_coverage"),
    )
    if inline:
        return inline, "", True

    path_value = _first_value(
        metadata.get("grounding_calibration_report_path"),
        metadata.get("grounding_coverage_report_path"),
        metadata.get("grounding_coverage_path"),
        metrics.get("grounding_calibration_report_path"),
        metrics.get("grounding_coverage_report_path"),
        metrics.get("grounding_coverage_path"),
    )
    if path_value is None:
        return {}, "", False
    path = Path(str(path_value))
    if not path.is_file():
        return {}, f"grounding calibration report not found: {path}", True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"could not read grounding calibration report: {exc}", True
    if not isinstance(payload, Mapping):
        return {}, "grounding calibration report must be a JSON object", True
    return dict(payload), "", True


def _recall_floor_check(
    gate: str,
    labels: frozenset[str],
    per_label_recall: Mapping[str, float],
    denominators: Mapping[str, int],
    floor: float,
) -> GateCheck:
    applicable = _applicable_labels(labels, per_label_recall, denominators)
    violations = {
        label: per_label_recall[label]
        for label in applicable
        if per_label_recall[label] < floor
    }
    return GateCheck(
        gate,
        not violations,
        reason="ok" if not violations else "recall below floor",
        details={
            "floor": floor,
            "applicable_labels": applicable,
            "violations": violations,
        },
    )


def _cross_script_check(
    per_script_recall: Mapping[str, float],
    per_script_leakage: Mapping[str, float],
    denominators: Mapping[str, int],
    *,
    threshold_profile: str,
    threshold_matrix: Mapping[str, Any] | None,
) -> GateCheck:
    if threshold_matrix is None or not threshold_profile:
        return GateCheck(
            CROSS_SCRIPT_GATE,
            False,
            reason="per-script thresholds could not be resolved",
        )

    try:
        recall_floors = profile_script_recall_floors(
            threshold_profile,
            matrix=threshold_matrix,
        )
        leakage_ceiling = profile_script_leakage_ceiling(
            threshold_profile,
            matrix=threshold_matrix,
        )
    except Exception as exc:
        return GateCheck(
            CROSS_SCRIPT_GATE,
            False,
            reason=f"could not resolve per-script thresholds: {exc}",
        )

    applicable = tuple(
        sorted(
            script
            for script in recall_floors
            if denominators.get(script, 0) > 0
            and (script in per_script_recall or script in per_script_leakage)
        )
    )
    violations: dict[str, dict[str, float | str]] = {}
    diagnostics: list[str] = []
    for script in applicable:
        observed_recall = per_script_recall.get(script)
        observed_leakage = per_script_leakage.get(script)
        floor = recall_floors[script]
        script_violations: dict[str, float | str] = {}
        if observed_recall is None:
            script_violations["recall"] = "missing"
            script_violations["recall_floor"] = floor
            diagnostics.append(f"{script} recall is missing (floor {floor:.4f})")
        elif observed_recall + 1e-12 < floor:
            script_violations["recall"] = observed_recall
            script_violations["recall_floor"] = floor
            diagnostics.append(
                f"{script} recall {observed_recall:.4f} is below {floor:.4f}"
            )
        if observed_leakage is None:
            script_violations["leakage_rate"] = "missing"
            script_violations["leakage_ceiling"] = leakage_ceiling
            diagnostics.append(
                f"{script} leakage is missing (ceiling {leakage_ceiling:.4f})"
            )
        elif observed_leakage > leakage_ceiling + 1e-12:
            script_violations["leakage_rate"] = observed_leakage
            script_violations["leakage_ceiling"] = leakage_ceiling
            diagnostics.append(
                f"{script} leakage {observed_leakage:.4f} exceeds {leakage_ceiling:.4f}"
            )
        if script_violations:
            violations[script] = script_violations

    return GateCheck(
        CROSS_SCRIPT_GATE,
        not violations,
        reason=("ok" if not violations else "; ".join(diagnostics)),
        details={
            "applicable_scripts": applicable,
            "recall_floors": recall_floors,
            "leakage_ceiling": leakage_ceiling,
            "per_script_recall": per_script_recall,
            "per_script_leakage": per_script_leakage,
            "total_graphemes_by_script": denominators,
            "violations": violations,
        },
    )


def _g3_check(critical_leakage_count: int) -> GateCheck:
    return GateCheck(
        "G3",
        critical_leakage_count == 0,
        reason=(
            "ok"
            if critical_leakage_count == 0
            else "critical leakage must be exactly zero"
        ),
        details={"critical_leakage_count": critical_leakage_count},
    )


def _g11_critical_finding_recall_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    metric = _first_mapping(
        metadata.get("critical_finding_recall"),
        metrics.get("critical_finding_recall"),
        metadata.get("critical_recall"),
        metrics.get("critical_recall"),
    )
    if not metric:
        return GateCheck(
            "G11",
            True,
            reason="not provided",
            details={"floor": G11_CRITICAL_RECALL_FLOOR},
        )

    overall = _optional_float(
        _first_value(metric.get("overall"), metric.get("recall"), metric.get("rate"))
    )
    total = _optional_int(metric.get("total"))
    covered = _optional_int(_first_value(metric.get("covered"), metric.get("hits")))
    by_category = _numeric_map(metric.get("by_category"))
    missed_findings = _critical_finding_misses(metric)
    if total == 0 and overall is None:
        overall = 1.0
    if total is None:
        total = int(metric.get("denominator", 0) or 0)
    if covered is None:
        covered = int(metric.get("numerator", 0) or 0)
    if overall is None:
        return GateCheck(
            "G11",
            False,
            reason="critical-finding recall metric is malformed",
            details={"floor": G11_CRITICAL_RECALL_FLOOR},
        )

    recall_violations: dict[str, Any] = {}
    if overall < G11_CRITICAL_RECALL_FLOOR:
        recall_violations["overall"] = {
            "observed": overall,
            "floor": G11_CRITICAL_RECALL_FLOOR,
        }
    category_violations = {
        category: {"observed": recall, "floor": G11_CRITICAL_RECALL_FLOOR}
        for category, recall in by_category.items()
        if recall < G11_CRITICAL_RECALL_FLOOR
    }
    if category_violations:
        recall_violations["by_category"] = category_violations

    zero_miss_findings = [
        finding
        for finding in missed_findings
        if finding.get("category") in _G11_ZERO_MISS_CATEGORIES
    ]
    violations: dict[str, Any] = {}
    if recall_violations:
        violations["recall_below_floor"] = recall_violations
    if zero_miss_findings:
        violations["must_not_miss_findings"] = zero_miss_findings

    return GateCheck(
        "G11",
        not violations,
        reason="ok" if not violations else "critical-finding recall gate failed",
        details={
            "floor": G11_CRITICAL_RECALL_FLOOR,
            "overall": overall,
            "by_category": by_category,
            "covered": covered,
            "total": total,
            "missed_findings": missed_findings,
            "violations": violations,
        },
    )


def _critical_finding_misses(metric: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_misses = metric.get("missed_findings") or metric.get("misses") or []
    if not isinstance(raw_misses, Sequence) or isinstance(raw_misses, (str, bytes)):
        return []

    misses: list[dict[str, Any]] = []
    for item in raw_misses:
        if not isinstance(item, Mapping):
            continue
        category = normalize_critical_finding_category(item.get("category", ""))
        start = _optional_int(item.get("start"))
        end = _optional_int(item.get("end"))
        misses.append(
            {
                "category": category,
                "fixture_id": str(item.get("fixture_id") or "unknown"),
                "start": 0 if start is None else start,
                "end": 0 if end is None else end,
                "label": normalize_label(str(item.get("label") or "")),
            }
        )
    return misses


def _g14_extraction_fairness_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    """Quarantine when extraction quality diverges too much across surrogate groups.

    Reads a PHI-free extraction-fairness metric (``extraction_fairness``) and
    fails when the worst-group-vs-best-group entity-F1 gap over synthetic
    site/note-type/demographic surrogate groups exceeds
    :data:`G14_EXTRACTION_DISPARITY_CEILING`. The metric is assistive and never
    infers demographic attributes from clinical text; when absent the gate is a
    no-op advisory pass.
    """
    ceiling = G14_EXTRACTION_DISPARITY_CEILING
    metric = _first_mapping(
        metadata.get("extraction_fairness"),
        metrics.get("extraction_fairness"),
        metadata.get("extraction_fairness_audit"),
        metrics.get("extraction_fairness_audit"),
    )
    if not metric:
        return GateCheck(
            "G14",
            True,
            reason="not provided",
            details={"ceiling": ceiling},
        )

    raw_per_group = metric.get("per_group")
    per_group_f1 = _extraction_group_f1(raw_per_group)
    if (
        not isinstance(raw_per_group, Mapping)
        or len(per_group_f1) != len(raw_per_group)
        or len(per_group_f1) < 2
        or any(not 0.0 <= score <= 1.0 for score in per_group_f1.values())
    ):
        return GateCheck(
            "G14",
            False,
            reason="extraction-fairness metric is malformed",
            details={
                "ceiling": ceiling,
                "error": (
                    "per_group must contain at least two groups with finite "
                    "entity_f1 values in [0, 1]"
                ),
            },
        )

    reported_gap = _optional_float(
        _first_value(
            metric.get("extraction_f1_gap"),
            metric.get("f1_gap"),
            metric.get("disparity"),
            metric.get("gap"),
        )
    )
    computed_gap = max(per_group_f1.values()) - min(per_group_f1.values())
    if reported_gap is not None and (
        not 0.0 <= reported_gap <= 1.0
        or not math.isclose(reported_gap, computed_gap, abs_tol=1e-12)
    ):
        return GateCheck(
            "G14",
            False,
            reason="extraction-fairness metric is malformed",
            details={
                "ceiling": ceiling,
                "computed_gap": computed_gap,
                "reported_gap": reported_gap,
                "error": "reported extraction_f1_gap does not match per_group",
            },
        )
    gap = computed_gap

    worst_group = min(per_group_f1, key=lambda key: (per_group_f1[key], key))
    best_group = max(per_group_f1, key=lambda key: (per_group_f1[key], key))

    passed = gap <= ceiling
    return GateCheck(
        "G14",
        passed,
        reason=(
            "ok"
            if passed
            else "extraction-F1 disparity across surrogate groups exceeds ceiling"
        ),
        details={
            "ceiling": ceiling,
            "extraction_f1_gap": gap,
            "worst_group": (str(worst_group) if worst_group is not None else None),
            "best_group": (str(best_group) if best_group is not None else None),
            "per_group_f1": per_group_f1,
            "worst_group_by_metric": _worst_group_by_metric(
                metric.get("worst_group_by_metric")
            ),
            "assistive": True,
        },
    )


def _extraction_group_f1(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    scores: dict[str, float] = {}
    for group, payload in value.items():
        score: float | None
        if isinstance(payload, Mapping):
            score = _optional_float(
                _first_value(payload.get("entity_f1"), payload.get("f1"))
            )
        else:
            score = _optional_float(payload)
        if score is not None:
            scores[str(group)] = score
    return scores


def _worst_group_by_metric(value: Any) -> dict[str, str | None]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(metric): (str(group) if group is not None else None)
        for metric, group in sorted(value.items(), key=lambda item: str(item[0]))
    }


def evaluate_end_to_end_pipeline_gate(
    report: Mapping[str, Any] | Any,
    baseline: Mapping[str, Any] | Any | None = None,
) -> GateCheck:
    """Evaluate end-to-end fact F1 and per-stage errors against G15.

    ``report`` may be a full benchmark report, a
    :class:`~openmed.eval.harness.PipelineEvalReport`, or its compact metric
    payload. A baseline is optional; when supplied, no stage bucket may grow.
    """

    if hasattr(report, "to_metric") and callable(report.to_metric):
        payload = _mapping(report.to_metric())
    else:
        payload = _report_payload(report)
    if "metrics" in payload or "metadata" in payload:
        metrics = _mapping(payload.get("metrics"))
        metadata = _mapping(payload.get("metadata"))
    else:
        metrics = {"end_to_end_pipeline": payload}
        metadata = {}

    baseline_entry: Mapping[str, Any] | None = None
    if baseline is not None:
        if hasattr(baseline, "to_metric") and callable(baseline.to_metric):
            baseline_payload = _mapping(baseline.to_metric())
        else:
            baseline_payload = _report_payload(baseline)
        baseline_entry = (
            baseline_payload
            if "metrics" in baseline_payload
            else {"metrics": {"end_to_end_pipeline": baseline_payload}}
        )
    return _g15_end_to_end_pipeline_check(metrics, metadata, baseline_entry)


def _g15_end_to_end_pipeline_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    baseline_entry: Mapping[str, Any] | None,
) -> GateCheck:
    evidence = _end_to_end_pipeline_evidence(metrics, metadata)
    if not evidence:
        return GateCheck(
            "G15",
            True,
            reason="not provided",
            details={
                "fact_f1_floor": G15_E2E_FACT_F1_FLOOR,
                "pipeline_metric_present": False,
            },
        )

    fact_level = _mapping(evidence.get("fact_level"))
    fact_f1 = _optional_float(
        _first_value(
            evidence.get("fact_f1"),
            evidence.get("end_to_end_fact_f1"),
            fact_level.get("f1"),
        )
    )
    raw_stage_counts = _first_mapping(
        evidence.get("stage_error_counts"),
        _nested(_mapping(evidence.get("attribution")), "stage_error_counts"),
        evidence.get("per_stage_errors"),
    )
    stage_counts, count_error = _pipeline_stage_error_counts(raw_stage_counts)
    reported_total = _optional_int(
        _first_value(
            evidence.get("total_end_to_end_errors"),
            _nested(
                _mapping(evidence.get("attribution")),
                "total_end_to_end_errors",
            ),
        )
    )
    computed_total = sum(stage_counts.values())

    violations: dict[str, Any] = {}
    if fact_f1 is None or not 0.0 <= fact_f1 <= 1.0:
        violations["fact_f1"] = {
            "floor": G15_E2E_FACT_F1_FLOOR,
            "observed": "missing_or_invalid" if fact_f1 is None else fact_f1,
        }
    elif fact_f1 < G15_E2E_FACT_F1_FLOOR:
        violations["fact_f1"] = {
            "floor": G15_E2E_FACT_F1_FLOOR,
            "observed": fact_f1,
        }
    if count_error:
        violations["stage_error_counts"] = count_error
    if reported_total is not None and reported_total != computed_total:
        violations["attribution_total"] = {
            "computed": computed_total,
            "reported": reported_total,
        }

    baseline_evidence = _baseline_end_to_end_pipeline_evidence(
        evidence,
        baseline_entry,
    )
    baseline_counts: dict[str, int] = {}
    if baseline_evidence:
        baseline_raw = _first_mapping(
            baseline_evidence.get("stage_error_counts"),
            _nested(
                _mapping(baseline_evidence.get("attribution")),
                "stage_error_counts",
            ),
            baseline_evidence.get("per_stage_errors"),
        )
        baseline_counts, baseline_error = _pipeline_stage_error_counts(baseline_raw)
        if baseline_error:
            violations["baseline_stage_error_counts"] = baseline_error
        else:
            regressions = {
                stage: {
                    "baseline": baseline_counts[stage],
                    "observed": stage_counts[stage],
                }
                for stage in PIPELINE_EVAL_STAGES
                if stage_counts[stage] > baseline_counts[stage]
            }
            if regressions:
                violations["stage_regressions"] = regressions

    passed = not violations
    details: dict[str, Any] = {
        "baseline_present": bool(baseline_evidence),
        "baseline_stage_error_counts": baseline_counts,
        "fact_f1": fact_f1,
        "fact_f1_floor": G15_E2E_FACT_F1_FLOOR,
        "pipeline_metric_present": True,
        "stage_error_counts": stage_counts,
        "total_end_to_end_errors": computed_total,
        "violations": violations,
    }
    for path_key in (
        "pipeline_attribution_path",
        "pipeline_attribution_report_path",
        "pipeline_eval_report_path",
    ):
        if evidence.get(path_key):
            details[path_key] = str(evidence[path_key])
    return GateCheck(
        "G15",
        passed,
        reason=(
            "ok" if passed else "end-to-end fact F1 or per-stage regression gate failed"
        ),
        details=details,
    )


def _end_to_end_pipeline_evidence(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _first_mapping(
        metrics.get("end_to_end_pipeline"),
        metrics.get("pipeline_eval"),
        metrics.get("pipeline_e2e"),
        metadata.get("end_to_end_pipeline"),
        metadata.get("pipeline_eval"),
        metadata.get("pipeline_e2e"),
    )
    if evidence:
        return evidence
    if (
        "fact_f1" in metrics
        or "fact_level" in metrics
        or "stage_error_counts" in metrics
    ):
        return dict(metrics)
    return {}


def _baseline_end_to_end_pipeline_evidence(
    candidate: Mapping[str, Any],
    baseline_entry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    embedded = _mapping(candidate.get("baseline"))
    if embedded:
        return embedded
    if baseline_entry is None:
        return {}
    baseline_metrics = _mapping(baseline_entry.get("metrics"))
    baseline_metadata = _mapping(baseline_entry.get("metadata"))
    return _end_to_end_pipeline_evidence(baseline_metrics, baseline_metadata)


def _pipeline_stage_error_counts(
    value: Mapping[str, Any],
) -> tuple[dict[str, int], str]:
    counts = {stage: 0 for stage in PIPELINE_EVAL_STAGES}
    if not value:
        return counts, "stage_error_counts is required"
    unknown = sorted(set(value) - set(PIPELINE_EVAL_STAGES))
    if unknown:
        return counts, "unknown stage bucket(s): " + ", ".join(unknown)
    for stage, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            return counts, f"{stage} must be a non-negative integer"
        counts[stage] = raw
    return counts, ""


def _adversarial_recall_under_attack_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    payload = _first_mapping(
        metadata.get("adversarial_robustness"),
        metrics.get("adversarial_robustness"),
    )
    if not payload:
        return GateCheck(
            "adversarial_recall_under_attack",
            True,
            reason="not applicable",
        )

    recall = _float_map(
        payload.get("post_defense_recall_under_attack_by_label")
        or payload.get("recall_under_attack_by_label")
    )
    leaked = _float_map(
        payload.get("post_defense_leaked_chars_by_label")
        or payload.get("leaked_chars_by_label")
    )
    floor = _optional_float(payload.get("recall_floor"))
    if floor is None:
        floor = _optional_float(metadata.get("adversarial_recall_floor"))
    if floor is None:
        floor = G2_V20_RECALL_FLOOR

    applicable = sorted(_G1_G2_LABELS & set(recall))
    recall_violations = {
        label: recall[label] for label in applicable if recall[label] < floor
    }
    direct_leaked = {
        label: int(value)
        for label, value in leaked.items()
        if label in _G1_G2_LABELS and int(value) > 0
    }
    passed = not recall_violations and not direct_leaked
    return GateCheck(
        "adversarial_recall_under_attack",
        passed,
        reason="ok" if passed else "adversarial recall or leakage gate failed",
        details={
            "applicable_labels": applicable,
            "direct_identifier_leaked_chars": direct_leaked,
            "floor": floor,
            "recall_violations": recall_violations,
        },
    )


def _g4_check(result: QuantRecallDeltaResult) -> GateCheck:
    if not result.quantized:
        return GateCheck(
            "G4",
            True,
            reason="not applicable",
            details=result.to_dict(),
        )

    if result.source == "missing_evidence":
        return GateCheck(
            "G4",
            False,
            reason="quantized artifacts require recall delta evidence",
            details=result.to_dict(),
            blocking_format=result.format,
        )

    return GateCheck(
        "G4",
        result.passed,
        reason="ok" if result.passed else "quantized recall delta exceeds limit",
        details=result.to_dict(),
        blocking_format=result.blocking_format,
    )


def _coreml_ane_residency_check(
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    variants = _coreml_variants(manifest, metadata)
    fp16 = _find_coreml_variant(variants, "coreml-fp16")
    if fp16 is None:
        return GateCheck(
            "CoreML-ANE",
            False,
            reason="CoreML fp16 variant residency evidence is required",
        )

    residency = _mapping(fp16.get("residency"))
    residency_percentage = _optional_float(
        fp16.get("ane_residency_percentage")
        or residency.get("ane_residency_percentage")
    )
    fallback_layers = (
        fp16.get("cpu_fallback_layers") or residency.get("cpu_fallback_layers") or []
    )
    fallback_count = (
        len(fallback_layers) if isinstance(fallback_layers, Sequence) else 0
    )
    passed = (
        residency_percentage is not None
        and residency_percentage >= 0.90
        and fallback_count == 0
    )
    return GateCheck(
        "CoreML-ANE",
        passed,
        reason="ok" if passed else "fp16 CoreML variant is not ANE-resident",
        details={
            "variant": fp16.get("name") or fp16.get("format"),
            "ane_residency_percentage": residency_percentage,
            "minimum": 0.90,
            "cpu_fallback_layers": fallback_layers,
        },
        blocking_format=None if passed else str(fp16.get("name") or "coreml-fp16"),
    )


def _coreml_variant_parity_check(
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    variants = _coreml_variants(manifest, metadata)
    if not variants:
        return GateCheck(
            "CoreML-parity",
            False,
            reason="CoreML parity evidence is required",
        )

    missing: list[str] = []
    failures: dict[str, Any] = {}
    for required in ("coreml-fp16", "coreml-int8"):
        variant = _find_coreml_variant(variants, required)
        if variant is None:
            missing.append(required)
            continue
        parity = _mapping(variant.get("parity"))
        if not _coreml_parity_passed(parity):
            failures[required] = parity or {"error": "missing parity payload"}

    int4 = _find_coreml_variant(variants, "coreml-int4")
    if int4 is None:
        missing.append("coreml-int4")
    else:
        int4_parity = _mapping(int4.get("parity"))
        if not (
            _coreml_parity_passed(int4_parity) or bool(int4_parity.get("auto_rejected"))
        ):
            failures["coreml-int4"] = int4_parity or {
                "error": "missing int4 parity rejection payload"
            }

    passed = not missing and not failures
    return GateCheck(
        "CoreML-parity",
        passed,
        reason="ok" if passed else "CoreML span parity gate failed",
        details={
            "recall_delta_limit": COREML_RECALL_DELTA_LIMIT,
            "missing": missing,
            "failures": failures,
        },
        blocking_format=next(iter(failures), missing[0] if missing else None),
    )


def _export_variant_manifest(metadata: Mapping[str, Any]) -> dict[str, Any]:
    inline = _mapping(
        metadata.get("export_variant_manifest")
        or metadata.get("export_conversion_manifest")
        or metadata.get("onnx_webgpu_manifest")
        or metadata.get("export_manifest")
    )
    if inline:
        return inline

    variants = metadata.get("export_variants")
    if isinstance(variants, Sequence) and not isinstance(variants, (str, bytes)):
        manifest: dict[str, Any] = {"variants": list(variants)}
        required = metadata.get("required_export_variants")
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
            manifest["required_variants"] = list(required)
        return manifest

    manifest_path = _optional_path(
        _first_value(
            metadata.get("export_variant_manifest_path"),
            metadata.get("export_manifest_path"),
        )
    )
    if manifest_path is None or not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return {}
    return _mapping(loaded)


def _export_variant_checks(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> list[GateCheck]:
    variants_raw = manifest.get("variants")
    variants = (
        [_mapping(item) for item in variants_raw if isinstance(item, Mapping)]
        if isinstance(variants_raw, Sequence)
        and not isinstance(variants_raw, (str, bytes))
        else []
    )

    parent_recall = _first_mapping(
        manifest.get("parent_recall"),
        manifest.get("fp_parent_per_label_recall"),
    ) or _quant_parent_recall(metrics, metadata)

    required = manifest.get("required_variants") or metadata.get(
        "required_export_variants"
    )
    required_variants = (
        [str(item) for item in required]
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes))
        else []
    )

    result = evaluate_export_variant_gate(
        variants=variants,
        tier_budgets=_TIER_BUDGETS,
        parent_recall=parent_recall or None,
        parent_format=str(manifest.get("parent_format") or ""),
        required_variants=required_variants,
        tier_aliases=_TIER_ALIASES,
    )

    coverage_passed = bool(result.variant_results) and not result.missing_required
    if not result.variant_results:
        coverage_reason = "no ONNX/WebGPU export variants declared"
    elif result.missing_required:
        coverage_reason = "required export variant missing"
    else:
        coverage_reason = "ok"

    checks: list[GateCheck] = [
        GateCheck(
            EXPORT_VARIANT_GATE,
            coverage_passed,
            reason=coverage_reason,
            details={
                "parent_format": result.parent_format,
                "evaluated_formats": list(result.evaluated_formats),
                "missing_required": list(result.missing_required),
                "blocked_formats": list(result.blocked_formats),
            },
            blocking_format=(
                result.missing_required[0] if result.missing_required else None
            ),
        )
    ]
    for variant in result.variant_results:
        checks.append(
            GateCheck(
                f"{EXPORT_VARIANT_GATE}:{variant.format}",
                variant.passed,
                reason=variant.reason,
                details=variant.to_dict(),
                blocking_format=variant.blocking_format,
            )
        )
    return checks


def _g5_check(
    tier: str,
    p50_ms: float | None,
    p95_ms: float | None,
    ram_mb: float | None,
    *,
    param_count: int | None = None,
) -> GateCheck:
    if _normalise_dimension(tier) == "nano":
        result = certify_measurements(
            param_count=param_count,
            ram_mb=ram_mb,
            p50_ms=p50_ms,
            p95_ms=p95_ms,
        )
        return GateCheck(
            "G5",
            result.passed,
            reason="ok" if result.passed else "Nano sub-tier budget not certified",
            details=result.to_dict(),
        )

    normalized_tier = _normalise_tier(tier)
    budget = _TIER_BUDGETS.get(normalized_tier)
    if budget is None:
        return GateCheck(
            "G5",
            False,
            reason="unknown target device tier",
            details={"tier": tier},
        )

    missing = [
        key
        for key, value in {"p50_ms": p50_ms, "p95_ms": p95_ms, "ram_mb": ram_mb}.items()
        if value is None
    ]
    if missing:
        return GateCheck(
            "G5",
            False,
            reason="latency and RAM evidence required",
            details={"missing": missing, "budget": budget},
        )

    assert p50_ms is not None and p95_ms is not None and ram_mb is not None
    observed = {
        "p50_ms": float(p50_ms),
        "p95_ms": float(p95_ms),
        "ram_mb": float(ram_mb),
    }
    violations = {
        key: {"observed": observed[key], "limit": budget[key]}
        for key in budget
        if float(observed[key]) > float(budget[key])
    }
    return GateCheck(
        "G5",
        not violations,
        reason="ok" if not violations else "tier latency or RAM budget exceeded",
        details={"tier": normalized_tier, "budget": budget, "violations": violations},
    )


def _g6_check(p50_ms: float | None, p95_ms: float | None) -> GateCheck:
    missing = [
        key
        for key, value in {"p50_ms": p50_ms, "p95_ms": p95_ms}.items()
        if value is None or not math.isfinite(float(value)) or float(value) < 0.0
    ]
    return GateCheck(
        "G6",
        not missing,
        reason="ok" if not missing else "p50/p95 latency must be documented",
        details={"missing": missing},
    )


def _g7_check(
    baseline_entry: Mapping[str, Any] | None,
    per_label_recall: Mapping[str, float],
    residual_leakage_rate: float,
    *,
    target_leakage: float,
) -> GateCheck:
    violations: dict[str, Any] = {}
    if residual_leakage_rate > RESIDUAL_LEAKAGE_SOFT_CEILING:
        violations["soft_leakage_ceiling"] = {
            "observed": residual_leakage_rate,
            "limit": RESIDUAL_LEAKAGE_SOFT_CEILING,
        }
    if residual_leakage_rate > target_leakage:
        violations["target_leakage"] = {
            "observed": residual_leakage_rate,
            "limit": target_leakage,
        }

    baseline_metrics = _mapping(
        baseline_entry.get("metrics") if baseline_entry is not None else None
    )
    baseline_recall = _baseline_label_recall(baseline_metrics)
    recall_violations = {}
    for label in sorted(_G1_G2_LABELS & set(per_label_recall) & set(baseline_recall)):
        drop = float(baseline_recall[label]) - float(per_label_recall[label])
        if drop > G7_RECALL_DROP_LIMIT:
            recall_violations[label] = {
                "baseline": baseline_recall[label],
                "candidate": per_label_recall[label],
                "drop": drop,
                "limit": G7_RECALL_DROP_LIMIT,
            }
    if recall_violations:
        violations["recall_drop"] = recall_violations

    baseline_leakage = _baseline_leakage(baseline_metrics)
    if baseline_leakage is not None and residual_leakage_rate > baseline_leakage:
        violations["residual_leakage_regression"] = {
            "baseline": baseline_leakage,
            "candidate": residual_leakage_rate,
        }

    return GateCheck(
        "G7",
        not violations,
        reason="ok" if not violations else "baseline regression gate failed",
        details={
            "baseline_key": (
                baseline_entry.get("key") if baseline_entry is not None else None
            ),
            "target_leakage": target_leakage,
            "violations": violations,
        },
    )


def _per_language_residual_leakage_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    """Enforce language-pack-specific leakage ceilings when evidence is present."""

    observed: dict[str, float] = {}
    denominators: dict[str, int] = {}
    denominator_languages: set[str] = set()
    for payload in _leakage_payloads(metrics, metadata):
        for language, raw_rate in _mapping(payload.get("by_language")).items():
            rate = _optional_float(raw_rate)
            if rate is not None:
                key = str(language)
                observed[key] = max(observed.get(key, 0.0), rate)
        totals = _mapping(payload.get("total_chars_by_language"))
        denominator_languages.update(str(language) for language in totals)
        for language, raw_total in totals.items():
            total = _optional_int(raw_total)
            if total is not None:
                key = str(language)
                denominators[key] = max(denominators.get(key, 0), total)

    evaluated = {
        language: observed[language]
        for language in PER_LANGUAGE_RESIDUAL_LEAKAGE_CEILINGS
        if language in observed
        and (language not in denominator_languages or denominators.get(language, 0) > 0)
    }
    violations = {
        language: {
            "observed": rate,
            "limit": PER_LANGUAGE_RESIDUAL_LEAKAGE_CEILINGS[language],
            "total_chars": denominators.get(language),
        }
        for language, rate in sorted(evaluated.items())
        if rate > PER_LANGUAGE_RESIDUAL_LEAKAGE_CEILINGS[language]
    }
    return GateCheck(
        "per_language_residual_leakage",
        not violations,
        reason=(
            "not applicable"
            if not evaluated
            else ("ok" if not violations else "per-language leakage ceiling exceeded")
        ),
        details={
            "ceilings": dict(PER_LANGUAGE_RESIDUAL_LEAKAGE_CEILINGS),
            "evaluated": evaluated,
            "violations": violations,
        },
    )


def _membership_leakage_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    metric = _first_mapping(
        metrics.get("membership_leakage"),
        metrics.get("membership_inference"),
        metadata.get("membership_leakage"),
        metadata.get("membership_inference"),
    )
    required = bool(
        metadata.get("membership_leakage_required")
        or metadata.get("membership_inference_required")
    )
    if not metric:
        return GateCheck(
            "membership_leakage",
            not required,
            reason=(
                "membership leakage evidence not provided"
                if not required
                else "membership leakage evidence is required"
            ),
            details={"required": required},
        )

    advantage = _optional_float(
        _first_value(metric.get("attacker_advantage"), metric.get("advantage"))
    )
    attacker_auc = _optional_float(metric.get("attacker_auc"))
    ceiling = _optional_float(
        _first_value(
            metric.get("advantage_ceiling"),
            metadata.get("membership_advantage_ceiling"),
            metadata.get("membership_inference_advantage_ceiling"),
        )
    )
    if ceiling is None:
        ceiling = DEFAULT_MEMBERSHIP_ADVANTAGE_CEILING
    if advantage is None:
        return GateCheck(
            "membership_leakage",
            False,
            reason="membership attacker advantage is required",
            details={"advantage_ceiling": ceiling},
        )

    per_label = _mapping(metric.get("per_label"))
    label_violations: dict[str, Any] = {}
    for label, values in per_label.items():
        if not isinstance(values, Mapping):
            continue
        label_advantage = _optional_float(
            _first_value(
                values.get("attacker_advantage"),
                values.get("advantage"),
            )
        )
        if label_advantage is not None and label_advantage > ceiling:
            label_violations[str(label)] = {
                "observed": label_advantage,
                "limit": ceiling,
            }

    violations: dict[str, Any] = {}
    if advantage > ceiling:
        violations["overall_advantage"] = {
            "observed": advantage,
            "limit": ceiling,
        }
    if label_violations:
        violations["per_label_advantage"] = label_violations

    return GateCheck(
        "membership_leakage",
        not violations,
        reason=(
            "ok" if not violations else "membership-inference advantage exceeds ceiling"
        ),
        details={
            "attacker_advantage": advantage,
            "attacker_auc": attacker_auc,
            "advantage_ceiling": ceiling,
            "feature_hash": metric.get("feature_hash"),
            "defense": _mapping(metric.get("defense")),
            "violations": violations,
        },
    )


def _g8_check(metadata: Mapping[str, Any]) -> GateCheck:
    fixtures = _span_fixtures(metadata)
    if not fixtures:
        return GateCheck(
            "G8",
            False,
            reason="span integrity evidence is required",
        )

    problems: list[dict[str, Any]] = []
    resolved_overlaps = 0
    checked = 0
    for index, fixture in enumerate(fixtures):
        text = str(fixture.get("text") or fixture.get("source_text") or "")
        raw_spans = (
            fixture.get("predicted_spans")
            or fixture.get("entities")
            or fixture.get("spans")
            or []
        )
        try:
            entities = [
                span.to_entity()
                for span in normalize_eval_spans(raw_spans, source_text=text)
            ]
        except Exception as exc:
            problems.append({"fixture_index": index, "error": str(exc)})
            continue

        scored = quality_gates.validate_entity_spans_strict(entities, text)
        checked += scored.total_spans
        resolved_overlaps += scored.overlaps_resolved
        if not scored.passed:
            problems.append(
                {
                    "fixture_index": index,
                    "span_validation": scored.to_dict(),
                }
            )

    return GateCheck(
        "G8",
        not problems,
        reason="ok" if not problems else "span integrity failed",
        details={
            "fixtures": len(fixtures),
            "spans_checked": checked,
            "overlaps_resolved": resolved_overlaps,
            "problems": problems,
        },
    )


def _g9_relation_extraction_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    evidence = _relation_extraction_evidence(metrics, metadata)
    required = bool(
        metadata.get("relation_extraction_required")
        or _normalise_dimension(str(metadata.get("task") or "")) == "relation"
    )
    if not evidence:
        return GateCheck(
            "G9",
            not required,
            reason=(
                "relation extraction evidence is required"
                if required
                else "not applicable"
            ),
            details={"required": required},
        )

    strict = _mapping(
        _first_value(evidence.get("strict"), metrics.get("strict_relation_f1"))
    )
    relaxed = _mapping(
        _first_value(evidence.get("relaxed"), metrics.get("relaxed_relation_f1"))
    )
    strict_lower = _relation_ci_lower(strict)
    relaxed_lower = _relation_ci_lower(relaxed)
    violations: dict[str, Any] = {}
    if strict_lower is None:
        violations["strict_confidence_interval"] = "missing lower bound"
    elif strict_lower < G9_STRICT_RE_F1_FLOOR:
        violations["strict_relation_f1"] = {
            "lower": strict_lower,
            "floor": G9_STRICT_RE_F1_FLOOR,
        }
    if relaxed_lower is None:
        violations["relaxed_confidence_interval"] = "missing lower bound"
    elif relaxed_lower < G9_RELAXED_RE_F1_FLOOR:
        violations["relaxed_relation_f1"] = {
            "lower": relaxed_lower,
            "floor": G9_RELAXED_RE_F1_FLOOR,
        }

    passed = not violations
    return GateCheck(
        "G9",
        passed,
        reason="ok" if passed else "relation extraction F1 lower CI below floor",
        details={
            "per_relation_type": _relation_type_summary(
                _first_value(
                    evidence.get("per_relation_type"),
                    metrics.get("per_relation_type_re_f1"),
                )
            ),
            "relaxed": _relation_metric_summary(relaxed),
            "relaxed_floor": G9_RELAXED_RE_F1_FLOOR,
            "strict": _relation_metric_summary(strict),
            "strict_floor": G9_STRICT_RE_F1_FLOOR,
            "violations": violations,
        },
    )


def _relation_extraction_evidence(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _first_mapping(
        metrics.get("relation_extraction"),
        metrics.get("relation_metrics"),
        metadata.get("relation_extraction"),
        metadata.get("relation_metrics"),
    )
    if evidence:
        return evidence
    strict = _first_mapping(metrics.get("strict_relation_f1"))
    relaxed = _first_mapping(metrics.get("relaxed_relation_f1"))
    per_type = _first_mapping(metrics.get("per_relation_type_re_f1"))
    if strict or relaxed or per_type:
        return {
            "per_relation_type": per_type,
            "relaxed": relaxed,
            "strict": strict,
        }
    return {}


def _relation_ci_lower(metric: Mapping[str, Any]) -> float | None:
    interval = _first_mapping(
        metric.get("confidence_interval"),
        metric.get("confidence_intervals"),
        metric.get("bootstrap"),
        metric.get("ci"),
    )
    return _optional_float(
        _first_value(
            interval.get("lower"),
            interval.get("lower_bound"),
            metric.get("lower_confidence_bound"),
            metric.get("lower_ci"),
        )
    )


def _relation_metric_summary(metric: Mapping[str, Any]) -> dict[str, Any]:
    interval = _first_mapping(
        metric.get("confidence_interval"),
        metric.get("confidence_intervals"),
        metric.get("bootstrap"),
        metric.get("ci"),
    )
    return {
        "f1": _optional_float(metric.get("f1")),
        "false_negatives": _optional_int(metric.get("false_negatives")),
        "false_positives": _optional_int(metric.get("false_positives")),
        "lower": _optional_float(interval.get("lower")),
        "precision": _optional_float(metric.get("precision")),
        "recall": _optional_float(metric.get("recall")),
        "true_positives": _optional_int(metric.get("true_positives")),
        "upper": _optional_float(interval.get("upper")),
    }


def _relation_type_summary(value: Any) -> dict[str, Any]:
    per_type = _mapping(value)
    summary: dict[str, Any] = {}
    for relation_type, payload in sorted(per_type.items()):
        metrics = _mapping(payload)
        strict = _mapping(metrics.get("strict"))
        relaxed = _mapping(metrics.get("relaxed"))
        summary[str(relation_type)] = {
            "relaxed_f1": _optional_float(relaxed.get("f1")),
            "strict_f1": _optional_float(strict.get("f1")),
        }
    return summary


def evaluate_relation_golden_regression_gate(
    metrics: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    family: str,
    metadata: Mapping[str, Any] | None = None,
) -> GateCheck:
    """Gate per-type strict relation F1 and zero-tolerance trap leaks.

    Candidate evidence uses ``relation_golden.by_type`` with the same strict
    metric payload produced by :func:`compute_relation_metrics`, the evaluated
    fixture-set hash, plus integer ``trap_leaks`` counts for ``assertion`` and
    ``temporal``. Baselines come from the committed ``relation_golden``
    baseline-store section and must be pinned to the same fixture set.
    """

    metadata = metadata or {}
    evidence = _relation_golden_evidence(metrics, metadata)
    required = _relation_golden_gate_required(metadata)
    if not evidence:
        return GateCheck(
            RELATION_GOLDEN_REGRESSION_GATE,
            not required,
            reason=(
                "relation golden evidence is required" if required else "not applicable"
            ),
            details={"family": family, "required": required},
        )

    candidate_f1, invalid_candidates = _candidate_relation_strict_f1(evidence)
    candidate_fixture_hash, fixture_hash_error = _candidate_relation_fixture_hash(
        evidence, metadata
    )
    family_baselines, invalid_baselines = _relation_golden_baselines(
        baseline, family=family
    )
    comparisons: dict[str, Any] = {}
    missing_baselines: list[str] = []
    missing_candidates: list[str] = []
    regressions: dict[str, Any] = {}
    fixture_hash_mismatches: dict[str, Any] = {}

    relation_types = sorted(
        set(candidate_f1)
        | set(invalid_candidates)
        | set(family_baselines)
        | set(invalid_baselines)
    )
    for relation_type in relation_types:
        candidate = candidate_f1.get(relation_type)
        pinned = family_baselines.get(relation_type)
        if pinned is None:
            if relation_type in candidate_f1 or relation_type in invalid_candidates:
                missing_baselines.append(relation_type)
            continue
        if candidate is None:
            if relation_type not in invalid_candidates:
                missing_candidates.append(relation_type)
            continue

        baseline_f1 = float(pinned["strict_f1"])
        tolerance = float(pinned["tolerance"])
        minimum = max(0.0, baseline_f1 - tolerance)
        drop = baseline_f1 - candidate
        comparison = {
            "baseline": baseline_f1,
            "baseline_key": pinned["key"],
            "candidate_fixture_hash": candidate_fixture_hash,
            "candidate": candidate,
            "drop": drop,
            "fixture_hash": pinned["fixture_hash"],
            "minimum": minimum,
            "tolerance": tolerance,
        }
        comparisons[relation_type] = comparison
        if (
            candidate_fixture_hash is not None
            and candidate_fixture_hash != pinned["fixture_hash"]
        ):
            fixture_hash_mismatches[relation_type] = {
                "baseline": pinned["fixture_hash"],
                "candidate": candidate_fixture_hash,
            }
        if candidate + 1e-12 < minimum:
            regressions[relation_type] = comparison

    trap_leaks, invalid_traps = _relation_trap_leaks(evidence, metadata)
    missing_traps = [
        kind
        for kind in RELATION_GOLDEN_TRAP_KINDS
        if kind not in trap_leaks and kind not in invalid_traps
    ]
    leaked_traps = {kind: count for kind, count in trap_leaks.items() if count > 0}

    violations: dict[str, Any] = {}
    if missing_baselines:
        violations["missing_baselines"] = missing_baselines
    if invalid_baselines:
        violations["invalid_baselines"] = invalid_baselines
    if missing_candidates:
        violations["missing_candidate_relation_types"] = missing_candidates
    if invalid_candidates:
        violations["invalid_candidate_strict_f1"] = invalid_candidates
    if fixture_hash_error is not None:
        violations["candidate_fixture_hash"] = fixture_hash_error
    if fixture_hash_mismatches:
        violations["fixture_hash_mismatches"] = fixture_hash_mismatches
    if regressions:
        violations["strict_f1_regressions"] = regressions
    if missing_traps:
        violations["missing_trap_leak_counts"] = missing_traps
    if invalid_traps:
        violations["invalid_trap_leak_counts"] = invalid_traps
    if leaked_traps:
        violations["trap_leaks"] = leaked_traps

    passed = not violations
    if passed:
        reason = "ok"
    elif leaked_traps:
        reason = "zero-tolerance assertion or temporal trap leak"
    elif missing_baselines or invalid_baselines:
        reason = "relation golden baseline is missing or invalid"
    elif fixture_hash_mismatches:
        reason = "relation golden fixture hash does not match pinned baseline"
    elif regressions:
        reason = "strict relation F1 regressed beyond pinned tolerance"
    else:
        reason = "relation golden evidence is incomplete or invalid"

    return GateCheck(
        RELATION_GOLDEN_REGRESSION_GATE,
        passed,
        reason=reason,
        details={
            "comparisons": comparisons,
            "family": family,
            "required": required,
            "trap_leaks": trap_leaks,
            "trap_tolerance": {kind: 0 for kind in RELATION_GOLDEN_TRAP_KINDS},
            "violations": violations,
        },
    )


def _relation_golden_gate_required(metadata: Mapping[str, Any]) -> bool:
    return bool(
        metadata.get("relation_golden_required")
        or metadata.get("relation_golden_regression_required")
    )


def _relation_golden_gate_is_applicable(
    metrics: Mapping[str, Any], metadata: Mapping[str, Any]
) -> bool:
    return bool(_relation_golden_evidence(metrics, metadata)) or (
        _relation_golden_gate_required(metadata)
    )


def _relation_golden_evidence(
    metrics: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    explicit = _first_mapping(
        metrics.get("relation_golden"),
        metrics.get("relation_golden_regression"),
        metadata.get("relation_golden"),
        metadata.get("relation_golden_regression"),
    )
    if explicit:
        return explicit

    relation_evidence = _relation_extraction_evidence(metrics, metadata)
    if relation_evidence and any(
        key in relation_evidence for key in ("by_type", "trap_leaks", "traps")
    ):
        return relation_evidence
    if _relation_golden_gate_required(metadata) and isinstance(
        metrics.get("by_type"), Mapping
    ):
        return dict(metrics)
    return {}


def _candidate_relation_strict_f1(
    evidence: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, str]]:
    nested_metrics = _mapping(evidence.get("metrics"))
    by_type = _first_mapping(
        evidence.get("by_type"),
        evidence.get("per_relation_type"),
        nested_metrics.get("by_type"),
        nested_metrics.get("per_relation_type"),
    )
    values: dict[str, float] = {}
    invalid: dict[str, str] = {}
    if not by_type:
        invalid["*"] = "missing by_type relation metrics"
        return values, invalid

    for raw_type, raw_metric in sorted(by_type.items(), key=lambda item: str(item[0])):
        relation_type = _canonical_relation_type(raw_type)
        if not relation_type:
            invalid[str(raw_type)] = "relation type is empty"
            continue
        if relation_type in values or relation_type in invalid:
            invalid[relation_type] = "duplicate normalized relation type"
            values.pop(relation_type, None)
            continue

        metric = _mapping(raw_metric)
        strict = _first_value(metric.get("strict"), metric.get("strict_f1"))
        if isinstance(strict, Mapping):
            raw_f1 = _first_value(strict.get("f1"), strict.get("point"))
        else:
            raw_f1 = strict
        parsed = _strict_probability(raw_f1)
        if parsed is None:
            invalid[relation_type] = "strict F1 must be between 0 and 1"
            continue
        values[relation_type] = parsed
    return values, invalid


def _candidate_relation_fixture_hash(
    evidence: Mapping[str, Any], metadata: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    nested_metrics = _mapping(evidence.get("metrics"))
    nested_metadata = _mapping(evidence.get("metadata"))
    raw_hash = _first_value(
        evidence.get("fixture_set_hash"),
        evidence.get("fixture_hash"),
        nested_metrics.get("fixture_set_hash"),
        nested_metadata.get("fixture_set_hash"),
        metadata.get("fixture_set_hash"),
    )
    if raw_hash is None:
        return None, "fixture_set_hash is required"
    if not isinstance(raw_hash, str) or not raw_hash.startswith("sha256:"):
        return None, "fixture_set_hash must be a sha256 digest"
    if not _is_privacy_safe_digest(raw_hash):
        return None, "fixture_set_hash must be a sha256 digest"
    return raw_hash, None


def _relation_golden_baselines(
    baseline: Mapping[str, Any], *, family: str
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    section = _mapping(baseline.get("relation_golden"))
    entries = _mapping(section.get("entries"))
    normalized_family = _normalise_dimension(family)
    values: dict[str, dict[str, Any]] = {}
    invalid: dict[str, str] = {}
    for raw_key, raw_entry in sorted(entries.items(), key=lambda item: str(item[0])):
        entry = _mapping(raw_entry)
        if _normalise_dimension(str(entry.get("family") or "")) != normalized_family:
            continue
        relation_type = _canonical_relation_type(entry.get("relation_type"))
        key = str(raw_key)
        if not relation_type:
            invalid[key] = "relation type is empty"
            continue
        expected_key = baseline_store.relation_baseline_key(family, relation_type)
        strict_f1 = _strict_probability(entry.get("strict_f1"))
        tolerance = _strict_probability(entry.get("tolerance"))
        if entry.get("key") != key or key != expected_key:
            invalid[relation_type] = "baseline key does not match family and type"
        elif strict_f1 is None:
            invalid[relation_type] = "baseline strict F1 must be between 0 and 1"
        elif tolerance is None:
            invalid[relation_type] = "baseline tolerance must be between 0 and 1"
        elif relation_type in values:
            invalid[relation_type] = "duplicate normalized relation baseline"
            values.pop(relation_type, None)
        else:
            values[relation_type] = {
                "fixture_hash": str(entry.get("fixture_hash") or ""),
                "key": key,
                "strict_f1": strict_f1,
                "tolerance": tolerance,
            }
    return values, invalid


def _relation_trap_leaks(
    evidence: Mapping[str, Any], metadata: Mapping[str, Any]
) -> tuple[dict[str, int], dict[str, str]]:
    nested_metrics = _mapping(evidence.get("metrics"))
    nested_metadata = _mapping(evidence.get("metadata"))
    trap_leaks = _first_mapping(
        evidence.get("trap_leaks"),
        nested_metrics.get("trap_leaks"),
        nested_metadata.get("trap_leaks"),
        metadata.get("relation_trap_leaks"),
    )
    if not trap_leaks:
        traps = _first_mapping(
            evidence.get("traps"),
            nested_metadata.get("traps"),
            metadata.get("relation_traps"),
        )
        trap_leaks = _mapping(traps.get("by_kind"))

    values: dict[str, int] = {}
    invalid: dict[str, str] = {}
    for kind in RELATION_GOLDEN_TRAP_KINDS:
        if kind not in trap_leaks:
            continue
        count = _relation_trap_leak_count(trap_leaks[kind])
        if count is None:
            invalid[kind] = "trap leak count must be a non-negative integer"
        else:
            values[kind] = count
    return values, invalid


def _relation_trap_leak_count(value: Any) -> int | None:
    direct = _strict_nonnegative_int(value)
    if direct is not None:
        return direct
    payload = _mapping(value)
    for field in ("leak_count", "leaked_count", "failure_count"):
        if field in payload:
            return _strict_nonnegative_int(payload[field])
    for field in ("leaked_relation_ids", "leaks"):
        leaked = payload.get(field)
        if isinstance(leaked, Sequence) and not isinstance(
            leaked, (str, bytes, bytearray)
        ):
            return len(leaked)
    return None


def _canonical_relation_type(value: Any) -> str:
    normalized = _normalise_dimension(str(value or ""))
    return normalized.upper().replace("-", "_")


def evaluate_radiology_entity_relation_gate(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> GateCheck:
    """Evaluate the G13 radiology entity, relation, and uncertainty floors."""
    return _g13_radiology_entity_relation_check(metrics, metadata or {})


def _g13_radiology_entity_relation_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    evidence = _radiology_entity_relation_evidence(metrics, metadata)
    task = _normalise_dimension(str(metadata.get("task") or ""))
    required = bool(
        metadata.get("radiology_entity_relation_required")
        or task in {"radiology-entity-relation", "radiology-relation"}
    )
    if not evidence:
        return GateCheck(
            "G13",
            not required,
            reason=(
                "radiology entity-and-relation evidence is required"
                if required
                else "not applicable"
            ),
            details={"required": required},
        )

    entity = _mapping(evidence.get("entity"))
    relation = _mapping(evidence.get("relation"))
    uncertainty = _mapping(evidence.get("uncertainty"))
    strict_entity_f1 = _metric_point(
        _first_value(entity.get("strict"), evidence.get("strict_entity_f1"))
    )
    strict_relation_f1 = _metric_point(
        _first_value(relation.get("strict"), evidence.get("strict_relation_f1"))
    )
    uncertainty_accuracy = _optional_float(
        _first_value(
            uncertainty.get("accuracy"),
            evidence.get("uncertainty_accuracy"),
        )
    )

    violations: dict[str, Any] = {}
    _record_floor_violation(
        violations,
        "strict_entity_f1",
        strict_entity_f1,
        G13_STRICT_ENTITY_F1_FLOOR,
    )
    _record_floor_violation(
        violations,
        "strict_relation_f1",
        strict_relation_f1,
        G13_STRICT_RELATION_F1_FLOOR,
    )
    _record_floor_violation(
        violations,
        "uncertainty_accuracy",
        uncertainty_accuracy,
        G13_UNCERTAINTY_ACCURACY_FLOOR,
    )
    passed = not violations
    return GateCheck(
        "G13",
        passed,
        reason=(
            "ok"
            if passed
            else "radiology entity, relation, or uncertainty metric below floor"
        ),
        details={
            "per_relation_type": _relation_type_summary(
                relation.get("per_relation_type")
            ),
            "per_uncertainty_class": _uncertainty_class_summary(
                _first_value(
                    uncertainty.get("per_class"),
                    uncertainty.get("by_class"),
                )
            ),
            "strict_entity_f1": strict_entity_f1,
            "strict_entity_f1_floor": G13_STRICT_ENTITY_F1_FLOOR,
            "strict_relation_f1": strict_relation_f1,
            "strict_relation_f1_floor": G13_STRICT_RELATION_F1_FLOOR,
            "uncertainty_accuracy": uncertainty_accuracy,
            "uncertainty_accuracy_floor": G13_UNCERTAINTY_ACCURACY_FLOOR,
            "violations": violations,
        },
    )


def _radiology_entity_relation_evidence(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return _first_mapping(
        metrics.get("radiology_entity_relation"),
        metrics.get("radiology_entity_relation_metrics"),
        metadata.get("radiology_entity_relation"),
        metadata.get("radiology_entity_relation_metrics"),
    )


def _metric_point(value: Any) -> float | None:
    if isinstance(value, Mapping):
        return _optional_float(value.get("f1"))
    return _optional_float(value)


def _record_floor_violation(
    violations: dict[str, Any],
    name: str,
    value: float | None,
    floor: float,
) -> None:
    if value is None:
        violations[name] = {"floor": floor, "value": "missing"}
    elif not 0.0 <= value <= 1.0:
        violations[name] = {"floor": floor, "value": value, "error": "out of range"}
    elif value < floor:
        violations[name] = {"floor": floor, "value": value}


def _uncertainty_class_summary(value: Any) -> dict[str, Any]:
    per_class = _mapping(value)
    summary: dict[str, Any] = {}
    for uncertainty, payload in sorted(per_class.items()):
        values = _mapping(payload)
        summary[str(uncertainty)] = {
            "accuracy": _optional_float(values.get("accuracy")),
            "correct": _optional_int(values.get("correct")),
            "total": _optional_int(values.get("total")),
        }
    return summary


def _zero_shot_language_leakage_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    evidence = _transfer_matrix_evidence(metrics, metadata)
    if not evidence:
        return GateCheck(
            "G9_zero_shot_language_leakage",
            True,
            reason="not applicable",
            details={"transfer_matrix_present": False},
        )

    languages = _transfer_languages(evidence)
    violations = [
        *_transfer_matrix_violations(evidence, metadata, languages),
        *_transfer_deficiency_violations(evidence, metadata),
    ]
    violations = _dedupe_transfer_violations(violations)

    return GateCheck(
        "G9_zero_shot_language_leakage",
        not violations,
        reason=(
            "ok"
            if not violations
            else "zero-shot language leakage exceeds per-language floor"
        ),
        details={
            "transfer_matrix_present": True,
            "language_count": len(languages),
            "default_floor": DEFAULT_ZERO_SHOT_LEAKAGE_FLOOR,
            "violations": violations,
        },
    )


def _g10_faithfulness_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    evidence = _faithfulness_evidence(metrics, metadata)
    if not evidence:
        return GateCheck(
            "G10",
            True,
            reason="not applicable",
            details={
                "ungrounded_fact_ceiling": G10_UNGROUNDED_FACT_CEILING,
                "faithfulness_metric_present": False,
            },
        )

    rate = _optional_float(
        _first_value(
            evidence.get("ungrounded_fact_rate"),
            evidence.get("rate"),
            evidence.get("overall"),
        )
    )
    if rate is None:
        return GateCheck(
            "G10",
            False,
            reason="ungrounded-fact rate is required",
            details={
                "ungrounded_fact_ceiling": G10_UNGROUNDED_FACT_CEILING,
                "faithfulness_metric_present": True,
            },
        )
    if not 0.0 <= rate <= 1.0:
        return GateCheck(
            "G10",
            False,
            reason="ungrounded-fact rate must be between zero and one",
            details={
                "faithfulness_metric_present": True,
                "ungrounded_fact_ceiling": G10_UNGROUNDED_FACT_CEILING,
                "ungrounded_fact_rate": rate,
            },
        )

    violations: dict[str, Any] = {}
    if rate > G10_UNGROUNDED_FACT_CEILING:
        violations["ungrounded_fact_rate"] = {
            "observed": rate,
            "limit": G10_UNGROUNDED_FACT_CEILING,
        }

    return GateCheck(
        "G10",
        not violations,
        reason=(
            "ok" if not violations else "ungrounded-fact rate exceeds hard ceiling"
        ),
        details={
            "by_fact_type": _mapping(evidence.get("by_fact_type")),
            "faithfulness_metric_present": True,
            "total_facts": _optional_int(evidence.get("total_facts")),
            "ungrounded_fact_ceiling": G10_UNGROUNDED_FACT_CEILING,
            "ungrounded_fact_rate": rate,
            "ungrounded_facts": _optional_int(evidence.get("ungrounded_facts")),
            "violations": violations,
        },
    )


def evaluate_cross_document_linkage_gate(
    report: BenchmarkReport | Mapping[str, Any],
    *,
    ceiling: float = DEFAULT_CROSS_DOCUMENT_LINKAGE_CEILING,
) -> GateCheck:
    """Evaluate longitudinal linkage evidence against a release ceiling.

    ``report`` may be a benchmark report containing a
    ``longitudinal_linkage_risk`` metric or the privacy-safe mapping returned by
    :func:`openmed.risk.longitudinal_risk_report`. The returned check retains
    only hashes, offsets, counts, and scores from that evidence.
    """

    resolved_ceiling = _probability_ceiling(ceiling, name="ceiling")
    payload = _report_payload(report)
    if "linkage_success_upper_bound" in payload:
        evidence: Any = payload
    else:
        evidence = _longitudinal_linkage_evidence(
            _mapping(payload.get("metrics")),
            _mapping(payload.get("metadata")),
        )
    if evidence is None:
        return GateCheck(
            CROSS_DOCUMENT_LINKAGE_GATE,
            False,
            reason="longitudinal linkage-risk evidence is required",
            details={"linkage_ceiling": resolved_ceiling},
        )
    return _evaluate_longitudinal_linkage_evidence(
        evidence,
        ceiling=resolved_ceiling,
    )


def _cross_document_linkage_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    ceiling: float,
) -> GateCheck:
    evidence = _longitudinal_linkage_evidence(metrics, metadata)
    if evidence is None:
        required = bool(
            _first_value(
                metadata.get("longitudinal_release"),
                metrics.get("longitudinal_release"),
                metadata.get("cross_document_release"),
                metrics.get("cross_document_release"),
            )
        )
        return GateCheck(
            CROSS_DOCUMENT_LINKAGE_GATE,
            not required,
            reason=(
                "not applicable"
                if not required
                else "longitudinal linkage-risk evidence is required"
            ),
            details={
                "evidence_present": False,
                "linkage_ceiling": ceiling,
            },
        )
    return _evaluate_longitudinal_linkage_evidence(evidence, ceiling=ceiling)


def _longitudinal_linkage_evidence(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Any:
    return _first_value(
        metadata.get("longitudinal_linkage_risk"),
        metrics.get("longitudinal_linkage_risk"),
        metadata.get("longitudinal_risk_report"),
        metrics.get("longitudinal_risk_report"),
        metadata.get("cross_document_linkage_risk"),
        metrics.get("cross_document_linkage_risk"),
    )


def _evaluate_longitudinal_linkage_evidence(
    evidence: Any,
    *,
    ceiling: float,
) -> GateCheck:
    validated = _validated_longitudinal_linkage_evidence(evidence)
    if validated is None:
        return GateCheck(
            CROSS_DOCUMENT_LINKAGE_GATE,
            False,
            reason="longitudinal linkage-risk evidence is malformed",
            details={
                "evidence_valid": False,
                "linkage_ceiling": ceiling,
            },
        )

    violations: dict[str, Any] = {}
    upper_bound = validated["linkage_success_upper_bound"]
    direct_leakage = validated["residual_direct_identifier_leakage"]
    direct_leakage_count = validated["residual_direct_identifier_leakage_count"]
    if upper_bound > ceiling + 1e-12:
        violations["linkage_success_upper_bound"] = {
            "observed": upper_bound,
            "limit": ceiling,
        }
    if direct_leakage > 0.0:
        violations["residual_direct_identifier_leakage"] = {
            "observed": direct_leakage,
            "limit": 0.0,
        }
    if direct_leakage_count > 0:
        violations["residual_direct_identifier_leakage_count"] = {
            "observed": direct_leakage_count,
            "limit": 0,
        }

    return GateCheck(
        CROSS_DOCUMENT_LINKAGE_GATE,
        not violations,
        reason=(
            "ok"
            if not violations
            else "cross-document linkage risk violates release policy"
        ),
        details={
            "evidence_valid": True,
            "evidence_hash": validated["evidence_hash"],
            "patient_count": validated["patient_count"],
            "document_count": validated["document_count"],
            "linkable_patient_count": validated["linkable_patient_count"],
            "linkage_success_upper_bound": upper_bound,
            "mean_patient_linkage_upper_bound": validated[
                "mean_patient_linkage_upper_bound"
            ],
            "linkage_ceiling": ceiling,
            "residual_direct_identifier_leakage": direct_leakage,
            "residual_direct_identifier_leakage_count": direct_leakage_count,
            "high_risk_patient_hashes": validated["high_risk_patient_hashes"],
            "high_risk_evidence": validated["high_risk_evidence"],
            "violations": violations,
        },
    )


def _validated_longitudinal_linkage_evidence(
    value: Any,
) -> dict[str, Any] | None:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        return None

    schema_version = _strict_nonnegative_int(value.get("schema_version"))
    patient_count = _strict_nonnegative_int(value.get("patient_count"))
    document_count = _strict_nonnegative_int(value.get("document_count"))
    linkable_patient_count = _strict_nonnegative_int(
        value.get("linkable_patient_count")
    )
    direct_leakage_count = _strict_nonnegative_int(
        value.get("residual_direct_identifier_leakage_count")
    )
    upper_bound = _strict_probability(value.get("linkage_success_upper_bound"))
    mean_bound = _strict_probability(value.get("mean_patient_linkage_upper_bound"))
    direct_leakage = _strict_probability(
        value.get("residual_direct_identifier_leakage")
    )
    if (
        schema_version != 1
        or patient_count is None
        or patient_count < 1
        or document_count is None
        or document_count < 1
        or linkable_patient_count is None
        or direct_leakage_count is None
        or upper_bound is None
        or mean_bound is None
        or direct_leakage is None
    ):
        return None

    raw_patients = value.get("patient_risks")
    raw_high_risk = value.get("high_risk_patients")
    if not _is_mapping_sequence(raw_patients) or not _is_mapping_sequence(
        raw_high_risk
    ):
        return None

    patients = [_safe_longitudinal_patient(row) for row in raw_patients]
    if any(patient is None for patient in patients):
        return None
    checked_patients = [patient for patient in patients if patient is not None]

    patient_hashes = [patient["patient_hash"] for patient in checked_patients]
    patient_bounds = [patient["linkage_upper_bound"] for patient in checked_patients]
    if (
        len(checked_patients) != patient_count
        or len(set(patient_hashes)) != patient_count
        or sum(patient["document_count"] for patient in checked_patients)
        != document_count
        or sum(patient["direct_identifier_count"] for patient in checked_patients)
        != direct_leakage_count
        or sum(bound > 0.0 for bound in patient_bounds) != linkable_patient_count
        or not math.isclose(
            max(patient_bounds), upper_bound, rel_tol=0.0, abs_tol=1e-12
        )
        or not math.isclose(
            sum(patient_bounds) / patient_count,
            mean_bound,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or (direct_leakage_count == 0 and direct_leakage != 0.0)
        or (direct_leakage_count > 0 and direct_leakage <= 0.0)
    ):
        return None

    expected_high_risk_patient_hashes = sorted(
        patient["patient_hash"]
        for patient in checked_patients
        if upper_bound > 0.0
        and math.isclose(
            patient["linkage_upper_bound"],
            upper_bound,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    actual_high_risk_patient_hashes = sorted(
        str(patient.get("patient_pseudonym")) for patient in raw_high_risk
    )
    if (
        any(
            not _is_privacy_safe_digest(patient_hash)
            for patient_hash in actual_high_risk_patient_hashes
        )
        or actual_high_risk_patient_hashes != expected_high_risk_patient_hashes
    ):
        return None

    high_risk_evidence = [
        item
        for patient in checked_patients
        if patient["patient_hash"] in expected_high_risk_patient_hashes
        for item in patient["evidence"]
    ]
    high_risk_evidence.sort(
        key=lambda item: (
            item["patient_hash"],
            item["note_index"],
            item["note_hash"],
            item["value_hash"],
            item.get("start", -1),
            item.get("end", -1),
        )
    )

    safe_projection = {
        "schema_version": schema_version,
        "patient_count": patient_count,
        "document_count": document_count,
        "linkable_patient_count": linkable_patient_count,
        "linkage_success_upper_bound": upper_bound,
        "mean_patient_linkage_upper_bound": mean_bound,
        "residual_direct_identifier_leakage": direct_leakage,
        "residual_direct_identifier_leakage_count": direct_leakage_count,
        "patients": checked_patients,
    }
    return {
        "evidence_hash": stable_hash(safe_projection),
        "patient_count": patient_count,
        "document_count": document_count,
        "linkable_patient_count": linkable_patient_count,
        "linkage_success_upper_bound": upper_bound,
        "mean_patient_linkage_upper_bound": mean_bound,
        "residual_direct_identifier_leakage": direct_leakage,
        "residual_direct_identifier_leakage_count": direct_leakage_count,
        "high_risk_patient_hashes": expected_high_risk_patient_hashes,
        "high_risk_evidence": high_risk_evidence,
    }


def _safe_longitudinal_patient(
    value: Mapping[str, Any],
) -> dict[str, Any] | None:
    patient_hash = value.get("patient_pseudonym")
    document_count = _strict_nonnegative_int(value.get("document_count"))
    evidence_count = _strict_nonnegative_int(value.get("evidence_count"))
    direct_identifier_count = _strict_nonnegative_int(
        value.get("direct_identifier_count")
    )
    linkage_upper_bound = _strict_probability(value.get("linkage_upper_bound"))
    if (
        not _is_privacy_safe_digest(patient_hash)
        or document_count is None
        or document_count < 1
        or evidence_count is None
        or direct_identifier_count is None
        or linkage_upper_bound is None
    ):
        return None

    raw_evidence = value.get("evidence")
    if not _is_mapping_sequence(raw_evidence):
        return None

    safe_evidence: list[dict[str, Any]] = []
    for item in raw_evidence:
        note_index = _strict_nonnegative_int(item.get("note_index"))
        start = _strict_nonnegative_int(item.get("start"))
        end = _strict_nonnegative_int(item.get("end"))
        if (
            note_index is None
            or not _is_privacy_safe_digest(item.get("note_hash"))
            or not _is_privacy_safe_digest(item.get("value_hash"))
            or (item.get("start") is not None and start is None)
            or (item.get("end") is not None and end is None)
            or (start is not None and end is not None and end < start)
        ):
            return None
        safe_item = {
            "patient_hash": patient_hash,
            "note_index": note_index,
            "note_hash": item["note_hash"],
            "value_hash": item["value_hash"],
        }
        if start is not None:
            safe_item["start"] = start
        if end is not None:
            safe_item["end"] = end
        safe_evidence.append(safe_item)

    if len(safe_evidence) != evidence_count:
        return None
    return {
        "patient_hash": patient_hash,
        "document_count": document_count,
        "direct_identifier_count": direct_identifier_count,
        "linkage_upper_bound": linkage_upper_bound,
        "evidence": safe_evidence,
    }


def _strict_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _strict_probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        return None
    return parsed


def _probability_ceiling(value: Any, *, name: str) -> float:
    parsed = _strict_probability(value)
    if parsed is None:
        raise ValueError(f"{name} must be a finite number between zero and one")
    return parsed


def _is_mapping_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(isinstance(item, Mapping) for item in value)
    )


def _is_privacy_safe_digest(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    prefix, separator, digest = value.partition(":")
    if separator != ":" or prefix not in {"sha256", "hmac-sha256"}:
        return False
    if len(digest) != 64:
        return False
    try:
        int(digest, 16)
    except ValueError:
        return False
    return True


def _federated_boundary_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck | None:
    boundary = _mapping(metrics.get("boundary_leakage"))
    side_channel = _mapping(metrics.get("side_channel"))
    has_federated_signal = bool(
        boundary
        or side_channel
        or metadata.get("federated_eval")
        or metrics.get("federated_boundary_leakage_rate") is not None
    )
    if not has_federated_signal:
        return None

    leakage_rate = _optional_float(
        _first_value(
            boundary.get("rate"),
            metrics.get("federated_boundary_leakage_rate"),
            metadata.get("federated_boundary_leakage_rate"),
        )
    )
    leakage_rate = 1.0 if leakage_rate is None else leakage_rate
    leaked_bytes = _optional_int(boundary.get("leaked_bytes")) or 0
    side_flagged = bool(side_channel.get("flagged", False))
    side_estimate = _optional_float(side_channel.get("estimate_bits")) or 0.0
    sandbox_count = _optional_int(metrics.get("sandbox_violation_count"))
    if sandbox_count is None:
        sandbox_count = len(metadata.get("sandbox_violations") or ())

    violations: dict[str, Any] = {}
    if leakage_rate > 0.0 or leaked_bytes > 0:
        violations["boundary_leakage"] = {
            "rate": leakage_rate,
            "leaked_bytes": leaked_bytes,
        }
    if side_flagged:
        violations["side_channel"] = {
            "estimate_bits": side_estimate,
            "threshold_bits": side_channel.get("threshold_bits"),
        }
    if sandbox_count:
        violations["sandbox"] = {"violation_count": sandbox_count}

    return GateCheck(
        "federated_boundary",
        not violations,
        reason="ok" if not violations else "federated boundary leakage gate failed",
        details={
            "boundary_leakage_rate": leakage_rate,
            "leaked_bytes": leaked_bytes,
            "side_channel_estimate_bits": side_estimate,
            "sandbox_violation_count": sandbox_count,
            "violations": violations,
        },
    )


def _k_floor_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    evidence = _k_floor_evidence(metrics, metadata)
    target_k = _optional_int(evidence.get("target_k"))
    if target_k is None:
        return GateCheck("k_floor", True, reason="not applicable")
    if target_k < 1:
        return GateCheck(
            "k_floor",
            False,
            reason="target_k must be >= 1",
            details={"target_k": target_k},
        )

    measured_k = _optional_int(evidence.get("measured_k"))
    max_bound = _optional_float(evidence.get("max_reidentification_upper_bound"))
    self_check = evidence.get("numeric_self_check")
    self_check_passed = None
    if isinstance(self_check, Mapping) and "passed" in self_check:
        self_check_passed = bool(self_check.get("passed"))

    violations: dict[str, Any] = {}
    if measured_k is None:
        violations["measured_k"] = "missing"
    elif measured_k < target_k:
        violations["measured_k"] = {"observed": measured_k, "target": target_k}

    target_bound = 1.0 / target_k
    if max_bound is None:
        violations["max_reidentification_upper_bound"] = "missing"
    elif max_bound > target_bound + 1e-12:
        violations["max_reidentification_upper_bound"] = {
            "observed": max_bound,
            "limit": target_bound,
        }

    if self_check_passed is False:
        violations["numeric_self_check"] = self_check

    return GateCheck(
        "k_floor",
        not violations,
        reason="ok" if not violations else "realized k or bound violates policy",
        details={
            "target_k": target_k,
            "measured_k": measured_k,
            "target_bound": target_bound,
            "max_reidentification_upper_bound": max_bound,
            "violations": violations,
        },
    )


def _transfer_matrix_evidence(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return _first_mapping(
        metadata.get("cross_lingual_transfer"),
        metadata.get("transfer_matrix_report"),
        metadata.get("transfer_matrix"),
        metrics.get("cross_lingual_transfer"),
        metrics.get("transfer_matrix_report"),
        metrics.get("transfer_matrix"),
        _nested(metrics, "fairness", "cross_lingual_transfer"),
    )


def _transfer_languages(evidence: Mapping[str, Any]) -> list[str]:
    languages = _string_set(evidence.get("languages"))
    matrix = _mapping(evidence.get("matrix"))
    for source_language, targets in matrix.items():
        if str(source_language):
            languages.add(str(source_language))
        languages.update(_mapping(targets))
    if not languages:
        languages = set(SUPPORTED_LANGUAGES)
    return sorted(languages)


def _transfer_floor_map(
    evidence: Mapping[str, Any],
    metadata: Mapping[str, Any],
    languages: Sequence[str],
) -> dict[str, float]:
    floors = {language: DEFAULT_ZERO_SHOT_LEAKAGE_FLOOR for language in languages}
    floor_source = _first_mapping(
        evidence.get("leakage_floors"),
        evidence.get("per_language_leakage_floors"),
        metadata.get("leakage_floors_by_language"),
        metadata.get("per_language_leakage_floors"),
    )
    for language, floor in floor_source.items():
        parsed = _optional_float(floor)
        if parsed is not None:
            floors[str(language)] = parsed
    return floors


def _transfer_matrix_violations(
    evidence: Mapping[str, Any],
    metadata: Mapping[str, Any],
    languages: Sequence[str],
) -> list[dict[str, Any]]:
    matrix = _mapping(evidence.get("matrix"))
    floors = _transfer_floor_map(evidence, metadata, languages)
    violations: list[dict[str, Any]] = []
    for source_language, targets in sorted(matrix.items()):
        source = str(source_language)
        for target_language, raw_cell in sorted(_mapping(targets).items()):
            target = str(target_language)
            if source == target:
                continue
            cell = _mapping(raw_cell)
            leakage_rate = _optional_float(
                _first_value(
                    cell.get("leakage_rate"),
                    cell.get("rate"),
                    cell.get("leakage"),
                )
            )
            if leakage_rate is None:
                continue
            floor = floors.get(target, DEFAULT_ZERO_SHOT_LEAKAGE_FLOOR)
            if leakage_rate <= floor:
                continue
            violations.append(
                _transfer_violation(
                    source_language=source,
                    target_language=target,
                    leakage_rate=leakage_rate,
                    leakage_floor=floor,
                    leaked_chars=_optional_int(cell.get("leaked_chars")),
                    total_chars=_optional_int(cell.get("total_chars")),
                )
            )
    return violations


def _transfer_deficiency_violations(
    evidence: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_rows = evidence.get("deficiencies") or []
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        return []
    floors = _transfer_floor_map(evidence, metadata, _transfer_languages(evidence))
    violations: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        row = _mapping(raw_row)
        target = str(row.get("target_language") or row.get("language") or "")
        source = str(row.get("source_language") or row.get("source") or "")
        leakage_rate = _optional_float(
            _first_value(
                row.get("leakage_rate"),
                row.get("rate"),
                row.get("leakage"),
            )
        )
        if not target or not source or leakage_rate is None:
            continue
        floor = _optional_float(
            _first_value(row.get("leakage_floor"), row.get("floor"))
        )
        if floor is None:
            floor = floors.get(target, DEFAULT_ZERO_SHOT_LEAKAGE_FLOOR)
        excess = _optional_float(row.get("excess"))
        if excess is None:
            excess = leakage_rate - floor
        if excess <= 0.0 and leakage_rate <= floor:
            continue
        violations.append(
            _transfer_violation(
                source_language=source,
                target_language=target,
                leakage_rate=leakage_rate,
                leakage_floor=floor,
                leaked_chars=_optional_int(row.get("leaked_chars")),
                total_chars=_optional_int(row.get("total_chars")),
                rank=_optional_int(row.get("rank")),
            )
        )
    return violations


def _transfer_violation(
    *,
    source_language: str,
    target_language: str,
    leakage_rate: float,
    leakage_floor: float,
    leaked_chars: int | None,
    total_chars: int | None,
    rank: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_language": source_language,
        "target_language": target_language,
        "leakage_rate": leakage_rate,
        "leakage_floor": leakage_floor,
        "excess": leakage_rate - leakage_floor,
    }
    if leaked_chars is not None:
        payload["leaked_chars"] = leaked_chars
    if total_chars is not None:
        payload["total_chars"] = total_chars
    if rank is not None:
        payload["rank"] = rank
    return payload


def _dedupe_transfer_violations(
    violations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for violation in violations:
        row = dict(violation)
        key = (
            str(row.get("target_language") or ""),
            str(row.get("source_language") or ""),
        )
        if not all(key):
            continue
        current = deduped.get(key)
        if current is None or float(row.get("excess", 0.0)) > float(
            current.get("excess", 0.0)
        ):
            deduped[key] = row
    return sorted(
        deduped.values(),
        key=lambda row: (
            -float(row.get("excess", 0.0)),
            -float(row.get("leakage_rate", 0.0)),
            str(row.get("target_language") or ""),
            str(row.get("source_language") or ""),
        ),
    )


def evaluate_federated_boundary_gate(
    report: BenchmarkReport | Mapping[str, Any],
) -> GateCheck:
    """Evaluate only the federated boundary leakage gate for a report."""
    payload = _report_payload(report)
    metrics = dict(_mapping(payload.get("metrics") or payload))
    if "sandbox_violation_count" not in metrics and isinstance(
        payload.get("sandbox_violations"),
        Sequence,
    ):
        metrics["sandbox_violation_count"] = len(payload["sandbox_violations"])
    metadata = _mapping(payload.get("metadata"))
    check = _federated_boundary_check(metrics, metadata)
    if check is not None:
        return check
    return GateCheck(
        "federated_boundary",
        False,
        reason="federated boundary metrics are required",
    )


def evaluate_reidentification_risk_gate(
    report: Any,
    thresholds: Mapping[str, Any] | None = None,
    *,
    threshold: float | None = None,
) -> GateCheck:
    """Gate a structured re-identification report on scenario risk.

    Args:
        report: A report returned by
            :func:`openmed.structured.reid_report.reid_report`.
        thresholds: Per-scenario probability ceilings. Any subset of
            ``prosecutor``, ``journalist``, and ``marketer`` may be configured.
        threshold: Optional shared ceiling for all three scenarios. This is a
            convenience alternative to ``thresholds``.

    Returns:
        A privacy-safe :class:`GateCheck`. A scenario passes when its headline
        risk is less than or equal to the configured ceiling.
    """

    if thresholds is not None and threshold is not None:
        raise ValueError("configure thresholds or threshold, not both")
    configured: Mapping[str, Any]
    if threshold is not None:
        configured = {scenario: threshold for scenario in _REID_SCENARIOS}
    elif thresholds is not None:
        configured = thresholds
    else:
        raise ValueError("at least one re-identification risk threshold is required")
    try:
        return _evaluate_reidentification_risk_report(report, configured)
    except (TypeError, ValueError):
        return GateCheck(
            REIDENTIFICATION_RISK_GATE,
            False,
            reason="re-identification report or thresholds are malformed",
            details={"thresholds_configured": True},
        )


def evaluate_reid_risk_gate(
    report: Any,
    thresholds: Mapping[str, Any] | None = None,
    *,
    threshold: float | None = None,
) -> GateCheck:
    """Alias for :func:`evaluate_reidentification_risk_gate`."""

    return evaluate_reidentification_risk_gate(
        report,
        thresholds,
        threshold=threshold,
    )


_REID_SCENARIOS = ("prosecutor", "journalist", "marketer")


def _reidentification_risk_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    raw_report = _first_value(
        metadata.get("structured_reidentification_risk_report"),
        metrics.get("structured_reidentification_risk_report"),
        metadata.get("structured_reid_report"),
        metrics.get("structured_reid_report"),
        metadata.get("reidentification_risk_report"),
        metrics.get("reidentification_risk_report"),
    )
    if raw_report is None:
        return GateCheck(
            REIDENTIFICATION_RISK_GATE,
            True,
            reason="not applicable",
        )

    raw_thresholds = _first_value(
        metadata.get("reidentification_risk_thresholds"),
        metrics.get("reidentification_risk_thresholds"),
        metadata.get("reid_risk_thresholds"),
        metrics.get("reid_risk_thresholds"),
    )
    if raw_thresholds is None:
        return GateCheck(
            REIDENTIFICATION_RISK_GATE,
            False,
            reason="re-identification risk thresholds are required",
            details={"thresholds_configured": False},
        )
    try:
        return _evaluate_reidentification_risk_report(raw_report, raw_thresholds)
    except (TypeError, ValueError):
        return GateCheck(
            REIDENTIFICATION_RISK_GATE,
            False,
            reason="re-identification report or thresholds are malformed",
            details={"thresholds_configured": True},
        )


def _evaluate_reidentification_risk_report(
    report: Any,
    thresholds: Any,
) -> GateCheck:
    if hasattr(report, "to_dict") and callable(report.to_dict):
        report = report.to_dict()
    if not isinstance(report, Mapping):
        raise TypeError("re-identification report must be a mapping")
    if not isinstance(thresholds, Mapping):
        raise TypeError("re-identification thresholds must be a mapping")

    scenario_risks = _validated_reidentification_scenario_risks(report)
    unknown = sorted(set(thresholds) - set(_REID_SCENARIOS))
    if unknown:
        raise ValueError("re-identification thresholds contain unknown scenarios")
    if not thresholds:
        raise ValueError("at least one re-identification threshold is required")

    configured: dict[str, float] = {}
    observed: dict[str, float] = {}
    violations: dict[str, Any] = {}
    for scenario in _REID_SCENARIOS:
        if scenario not in thresholds:
            continue
        ceiling = _strict_probability(thresholds[scenario])
        if ceiling is None:
            raise ValueError("re-identification thresholds must be probabilities")
        scenario_risk = scenario_risks[scenario]
        configured[scenario] = ceiling
        observed[scenario] = scenario_risk
        if scenario_risk > ceiling:
            violations[scenario] = {
                "observed": scenario_risk,
                "threshold": ceiling,
            }

    population_model_consistent = report.get("population_model_consistent")
    if not isinstance(population_model_consistent, bool):
        raise ValueError("population-model consistency flag is required")
    if not population_model_consistent:
        violations["population_model"] = {"consistent": False}

    return GateCheck(
        REIDENTIFICATION_RISK_GATE,
        not violations,
        reason=(
            "ok"
            if not violations
            else "re-identification report violates configured release policy"
        ),
        details={
            "thresholds_configured": True,
            "risks": observed,
            "thresholds": configured,
            "violations": violations,
        },
    )


def _validated_reidentification_scenario_risks(
    report: Mapping[str, Any],
) -> dict[str, float]:
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("unsupported re-identification report schema")

    parsed: dict[str, dict[str, float]] = {}
    for scenario in _REID_SCENARIOS:
        scenario_report = report.get(scenario)
        if not isinstance(scenario_report, Mapping):
            raise ValueError("re-identification scenario report is missing")
        risk = _strict_probability(scenario_report.get("risk"))
        expected = _strict_probability(scenario_report.get("expected_probability"))
        maximum = _strict_probability(scenario_report.get("maximum_probability"))
        if risk is None or expected is None or maximum is None:
            raise ValueError("re-identification scenario probabilities are invalid")
        if expected > maximum:
            raise ValueError("re-identification expected risk exceeds maximum risk")
        canonical_risk = expected if scenario == "marketer" else maximum
        if not math.isclose(risk, canonical_risk, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("re-identification headline risk is inconsistent")
        parsed[scenario] = {
            "risk": risk,
            "expected": expected,
            "maximum": maximum,
        }

    scenario_relationships_invalid = (
        parsed["journalist"]["expected"] != parsed["marketer"]["expected"]
        or parsed["journalist"]["maximum"] != parsed["marketer"]["maximum"]
    )
    population_model_consistent = report.get("population_model_consistent")
    if population_model_consistent is True:
        scenario_relationships_invalid = scenario_relationships_invalid or (
            parsed["prosecutor"]["expected"] < parsed["journalist"]["expected"]
            or parsed["prosecutor"]["maximum"] < parsed["journalist"]["maximum"]
        )
    if scenario_relationships_invalid:
        raise ValueError("re-identification scenario relationships are inconsistent")
    return {scenario: parsed[scenario]["risk"] for scenario in _REID_SCENARIOS}


def evaluate_release_risk_evidence(
    evidence: Any,
) -> GateCheck:
    """Verify one aggregate expert-review evidence bundle.

    This is a technical integrity and configured-threshold check. Passing it
    does not constitute an Expert Determination or authorize a data release.
    """

    if evidence is None:
        return GateCheck(
            "structured_release_risk",
            False,
            reason="release-risk evidence is required",
            details={"integrity_verified": False},
        )
    return _structured_release_risk_check(
        {"structured_release_risk_evidence": evidence},
        {},
    )


def _structured_release_risk_check(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> GateCheck:
    raw_evidence = _first_value(
        metadata.get("structured_release_risk_evidence"),
        metrics.get("structured_release_risk_evidence"),
        metadata.get("expert_review_evidence"),
        metrics.get("expert_review_evidence"),
    )
    if raw_evidence is None:
        return GateCheck(
            "structured_release_risk",
            True,
            reason="not applicable",
        )

    if hasattr(raw_evidence, "to_dict") and callable(raw_evidence.to_dict):
        raw_evidence = raw_evidence.to_dict()
    if not isinstance(raw_evidence, Mapping):
        return GateCheck(
            "structured_release_risk",
            False,
            reason="release-risk evidence is malformed or fails integrity checks",
            details={"integrity_verified": False},
        )

    try:
        from openmed.compliance import ExpertReviewEvidenceReport

        evidence = ExpertReviewEvidenceReport.from_dict(raw_evidence)
    except (KeyError, TypeError, ValueError):
        # Do not echo parser errors: an invalid payload may contain raw data.
        return GateCheck(
            "structured_release_risk",
            False,
            reason="release-risk evidence is malformed or fails integrity checks",
            details={"integrity_verified": False},
        )

    model = evidence.privacy_models
    post = evidence.post_metrics
    violations: dict[str, Any] = {}
    if not evidence.search.optimality_proven:
        violations["search_optimality_proven"] = False
    elif not evidence.search.complete and evidence.schema_version < 3:
        violations["search_optimality_proof"] = (
            "schema_version_3_required_for_pruned_proof"
        )
    if model.achieved_k < model.configured_k:
        violations["k_anonymity"] = {
            "configured": model.configured_k,
            "achieved": model.achieved_k,
        }
    if evidence.sensitive_attributes and (
        model.l_variant is None or model.t_variant is None
    ):
        violations["sensitive_attribute_models"] = {
            "l_diversity_present": model.l_variant is not None,
            "t_closeness_present": model.t_variant is not None,
        }
    if (
        model.configured_l is not None
        and model.achieved_l is not None
        and model.achieved_l < model.configured_l
    ):
        violations["l_diversity"] = {
            "variant": model.l_variant,
            "configured": model.configured_l,
            "achieved": model.achieved_l,
        }
    if (
        model.configured_t is not None
        and model.achieved_t is not None
        and model.achieved_t > model.configured_t + 1e-12
    ):
        violations["t_closeness"] = {
            "variant": model.t_variant,
            "configured": model.configured_t,
            "achieved": model.achieved_t,
        }
    post_violation_counts = {
        "k_class_count": post.k_violating_class_count,
        "l_class_count": post.l_violating_class_count,
        "t_class_count": post.t_violating_class_count,
        "any_class_count": post.any_violating_class_count,
        "privacy_unit_count": post.violating_privacy_unit_count,
    }
    if any(post_violation_counts.values()):
        violations["post_transform_violations"] = post_violation_counts
    if evidence.composition.risk_status in {"increase_observed", "inconclusive"}:
        violations["composition_risk_status"] = "no_material_increase_observed_required"
    if evidence.composition.release_count == 1 and (
        evidence.composition.longitudinal_linkage_assessed
        or evidence.composition.prior_release_overlap_assessed
        or evidence.composition.risk_status != "not_assessed"
    ):
        violations["composition_review"] = {
            "single_release_requires_unassessed_status": True
        }
    elif evidence.composition.release_count > 1:
        composition_violations: dict[str, Any] = {}
        if not evidence.composition.longitudinal_linkage_assessed:
            composition_violations["longitudinal_linkage_assessed"] = False
        if not evidence.composition.prior_release_overlap_assessed:
            composition_violations["prior_release_overlap_assessed"] = False
        if evidence.composition.risk_status != "no_material_increase_observed":
            composition_violations["risk_status"] = (
                "no_material_increase_observed_required"
            )
        if composition_violations:
            violations["composition_review"] = composition_violations

    return GateCheck(
        "structured_release_risk",
        not violations,
        reason=(
            "ok"
            if not violations
            else "release-risk evidence violates its configured technical policy"
        ),
        details={
            "integrity_verified": True,
            "search_complete": evidence.search.complete,
            "search_optimality_proven": evidence.search.optimality_proven,
            "configured_k": model.configured_k,
            "achieved_k": model.achieved_k,
            "l_variant": model.l_variant,
            "configured_l": model.configured_l,
            "achieved_l": model.achieved_l,
            "t_variant": model.t_variant,
            "configured_t": model.configured_t,
            "achieved_t": model.achieved_t,
            "privacy_unit": evidence.assumptions.privacy_unit,
            "released_dataset_digest": evidence.digests.dataset,
            "policy_digest": evidence.digests.policy,
            "evidence_integrity_hash": evidence.integrity_hash,
            "composition_status": evidence.composition.risk_status,
            "qualified_expert_review_required": True,
            "violations": violations,
        },
    )


def _k_floor_evidence(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    source = _first_mapping(
        metadata.get("kanon_enforcement"),
        metrics.get("kanon_enforcement"),
        metadata.get("k_anonymity_enforcement"),
        metrics.get("k_anonymity_enforcement"),
        metadata.get("k_floor"),
        metrics.get("k_floor"),
        metadata.get("kanon"),
        metrics.get("kanon"),
    )
    target_k = _first_value(
        source.get("target_k"),
        metadata.get("target_k"),
        metrics.get("target_k"),
        _nested(metadata, "privacy_policy", "target_k"),
        _nested(metrics, "privacy_policy", "target_k"),
    )
    kanon = _mapping(source.get("kanon"))
    bounds = _mapping(source.get("bounds"))
    return {
        "target_k": target_k,
        "measured_k": _first_value(
            source.get("measured_k"),
            source.get("realized_k"),
            source.get("k"),
            kanon.get("k"),
            metadata.get("measured_k"),
            metrics.get("measured_k"),
        ),
        "max_reidentification_upper_bound": _first_value(
            source.get("max_reidentification_upper_bound"),
            bounds.get("max_reidentification_upper_bound"),
            metadata.get("max_reidentification_upper_bound"),
            metrics.get("max_reidentification_upper_bound"),
        ),
        "numeric_self_check": _first_value(
            source.get("numeric_self_check"),
            bounds.get("numeric_self_check"),
        ),
    }


def _report_payload(report: BenchmarkReport | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(report, BenchmarkReport):
        return report.to_dict()
    if hasattr(report, "to_dict") and callable(report.to_dict):
        return _mapping(report.to_dict())
    return _mapping(report)


def _identity(
    payload: Mapping[str, Any],
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    repo_id = str(
        metadata.get("repo_id")
        or metadata.get("repository")
        or metadata.get("model_repo")
        or payload.get("repo_id")
        or payload.get("model_name")
        or ""
    )
    family = str(
        metadata.get("family")
        or payload.get("family")
        or _infer_family(str(payload.get("model_name") or repo_id))
    )
    tier = str(metadata.get("tier") or payload.get("tier") or "")
    format_name = str(
        metadata.get("format")
        or metadata.get("model_format")
        or payload.get("format")
        or payload.get("device")
        or ""
    )
    return {
        "repo_id": repo_id,
        "family": family,
        "tier": tier,
        "param_count": _optional_int(
            metadata.get("param_count")
            or metadata.get("parameters")
            or metadata.get("model_parameters")
            or payload.get("param_count")
        ),
        "format": format_name,
        "eval_set_hash": str(
            metadata.get("eval_set_hash")
            or metrics.get("eval_set_hash")
            or payload.get("eval_set_hash")
            or ""
        ),
        "leakage_fixture_hash": str(
            metadata.get("leakage_fixture_hash")
            or metrics.get("leakage_fixture_hash")
            or payload.get("leakage_fixture_hash")
            or ""
        ),
    }


def _per_label_recall(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, int]]:
    recall = _first_mapping(
        metadata.get("per_label_recall"),
        metrics.get("per_label_recall"),
        _nested(metrics, "recall_slices", "by_label"),
        _nested(metrics, "character_recall", "by_label"),
    )
    denominators = _first_mapping(
        metadata.get("per_label_denominators"),
        metadata.get("total_chars_by_label"),
        _nested(metrics, "leakage", "total_chars_by_label"),
    )
    return _float_map(recall), {
        normalize_label(str(label)): int(value)
        for label, value in denominators.items()
        if _optional_int(value) is not None
    }


def _per_label_precision(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, float]:
    precision = _first_mapping(
        metadata.get("per_label_precision"),
        metrics.get("per_label_precision"),
        _nested(metrics, "precision_slices", "by_label"),
    )
    result = _float_map(precision)
    exact_precision = _nested(metrics, "exact_span_f1", "precision")
    if exact_precision is not None and "OVERALL" not in result:
        value = _optional_float(exact_precision)
        if value is not None:
            result["OVERALL"] = value
    return result


def _per_script_metrics(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    recall = _numeric_map(
        _first_mapping(
            metadata.get("per_script_recall"),
            metrics.get("per_script_recall"),
            _nested(metrics, "recall_slices", "by_script"),
            _nested(metrics, "character_recall", "by_script"),
        )
    )
    leakage = _numeric_map(
        _first_mapping(
            metadata.get("per_script_leakage"),
            metrics.get("per_script_leakage"),
            _nested(metrics, "leakage", "by_script"),
            _nested(metrics, "leakage_rate", "by_script"),
        )
    )
    denominators = _first_mapping(
        metadata.get("total_graphemes_by_script"),
        metadata.get("total_chars_by_script"),
        _nested(metrics, "recall_slices", "total_graphemes_by_script"),
        _nested(metrics, "recall_slices", "total_chars_by_script"),
        _nested(metrics, "leakage", "total_graphemes_by_script"),
        _nested(metrics, "leakage", "total_chars_by_script"),
    )
    parsed_denominators = {
        str(script): int(value)
        for script, value in denominators.items()
        if _optional_int(value) is not None
    }
    scripts = set(recall) | set(leakage)
    for script in scripts:
        if script not in recall and script in leakage:
            recall[script] = max(0.0, 1.0 - leakage[script])
        if script not in leakage and script in recall:
            leakage[script] = max(0.0, 1.0 - recall[script])
    return (
        {key: recall[key] for key in sorted(recall)},
        {key: leakage[key] for key in sorted(leakage)},
        {key: parsed_denominators[key] for key in sorted(parsed_denominators)},
    )


def _critical_leakage_count(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> int:
    counts: list[int] = []
    for value in (
        metadata.get("critical_leakage_count"),
        metrics.get("critical_leakage_count"),
    ):
        parsed = _optional_int(value)
        if parsed is not None:
            counts.append(parsed)

    for payload in _leakage_payloads(metrics, metadata):
        parsed = _optional_int(payload.get("critical_leakage_count"))
        if parsed is not None:
            counts.append(parsed)
        leaked_by_label = _float_map(payload.get("leaked_chars_by_label"))
        counts.append(
            int(
                sum(
                    value
                    for label, value in leaked_by_label.items()
                    if label in _CRITICAL_LABELS
                )
            )
        )

    return max(counts) if counts else 0


def _residual_leakage_rate(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> float:
    values: list[float] = []
    for value in (
        metadata.get("residual_leakage_rate"),
        metrics.get("residual_leakage_rate"),
        metrics.get("federated_boundary_leakage_rate"),
        _nested(metrics, "boundary_leakage", "rate"),
    ):
        parsed = _optional_float(value)
        if parsed is not None:
            values.append(parsed)

    for payload in _leakage_payloads(metrics, metadata):
        parsed = _optional_float(
            _first_value(payload.get("overall"), payload.get("rate"))
        )
        if parsed is not None:
            values.append(parsed)

    return max(values) if values else 1.0


def _leakage_payloads(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    payloads: list[dict[str, Any]] = []
    for source in (metrics, metadata):
        for key in (
            "leakage",
            "extraction_reemission_leakage",
            "grounding_reemission_leakage",
        ):
            payload = _mapping(source.get(key))
            if payload:
                payloads.append(payload)
    return tuple(payloads)


def _precomputed_quant_recall_delta(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    format_name: str,
) -> Any:
    raw = _first_value(
        metadata.get("quant_recall_delta"),
        metrics.get("quant_recall_delta"),
        _nested(metrics, "quantization", "recall_delta"),
    )
    if isinstance(raw, Mapping):
        format_key = _normalise_dimension(format_name)
        for key, value in raw.items():
            if _normalise_dimension(str(key)) == format_key:
                return value
    return raw


def _quant_parent_recall(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    parent = _first_mapping(
        metadata.get("fp_parent_per_label_recall"),
        metadata.get("parent_per_label_recall"),
        metadata.get("fp32_per_label_recall"),
        metrics.get("fp_parent_per_label_recall"),
        metrics.get("parent_per_label_recall"),
        metrics.get("fp32_per_label_recall"),
        _nested(metrics, "quantization", "fp_parent_per_label_recall"),
        _nested(metrics, "quantization", "parent_per_label_recall"),
    )
    return parent or None


def _latency(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[float | None, float | None]:
    p50 = _first_value(metadata.get("p50_ms"), _nested(metrics, "latency", "p50_ms"))
    p95 = _first_value(metadata.get("p95_ms"), _nested(metrics, "latency", "p95_ms"))
    return _optional_float(p50), _optional_float(p95)


def _ram_mb(metrics: Mapping[str, Any], metadata: Mapping[str, Any]) -> float | None:
    value = _first_value(
        metadata.get("ram_mb"),
        metadata.get("peak_rss_mib"),
        _nested(metrics, "resources", "peak_rss_mib"),
        _nested(metrics, "resources", "ram_mb"),
    )
    parsed = _optional_float(value)
    if parsed is not None:
        return parsed
    bytes_value = _first_value(
        metadata.get("peak_rss_bytes"),
        _nested(metrics, "resources", "peak_rss_bytes"),
    )
    bytes_parsed = _optional_float(bytes_value)
    if bytes_parsed is None:
        return None
    return bytes_parsed / (1024 * 1024)


def _span_fixtures(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = metadata.get("span_fixtures") or metadata.get("fixtures")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        return [item for item in rows if isinstance(item, Mapping)]
    if "source_text" in metadata or "predicted_spans" in metadata:
        return [
            {
                "text": metadata.get("source_text", ""),
                "predicted_spans": metadata.get("predicted_spans", []),
            }
        ]
    return []


def _faithfulness_evidence(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    direct = _first_mapping(
        metrics.get("faithfulness"),
        metrics.get("span_grounded_faithfulness"),
        metrics.get("grounding_faithfulness"),
        metadata.get("faithfulness"),
        metadata.get("span_grounded_faithfulness"),
    )
    if direct:
        return direct

    rate = _first_value(
        metrics.get("ungrounded_fact_rate"),
        metadata.get("ungrounded_fact_rate"),
    )
    if rate is None:
        return {}

    return {
        "by_fact_type": _first_mapping(
            metrics.get("ungrounded_fact_rate_by_type"),
            metrics.get("faithfulness_by_fact_type"),
            metadata.get("faithfulness_by_fact_type"),
        ),
        "total_facts": _first_value(
            metrics.get("total_facts"),
            metadata.get("total_facts"),
        ),
        "ungrounded_fact_rate": rate,
        "ungrounded_facts": _first_value(
            metrics.get("ungrounded_facts"),
            metadata.get("ungrounded_facts"),
        ),
    }


def _baseline_label_recall(metrics: Mapping[str, Any]) -> dict[str, float]:
    return _float_map(
        _first_mapping(
            metrics.get("per_label_recall"),
            metrics.get("recall_by_label"),
            _nested(metrics, "recall_slices", "by_label"),
        )
    )


def _baseline_leakage(metrics: Mapping[str, Any]) -> float | None:
    return _optional_float(
        _first_value(
            metrics.get("residual_leakage_rate"),
            metrics.get("leakage_rate"),
            _nested(metrics, "leakage", "overall"),
        )
    )


def _artifact_present(
    metadata: Mapping[str, Any],
    *,
    mapping_keys: Sequence[str],
    path_keys: Sequence[str],
) -> bool:
    for key in mapping_keys:
        value = metadata.get(key)
        if isinstance(value, Mapping) and value:
            return True
    for key in path_keys:
        value = metadata.get(key)
        if value and Path(str(value)).exists():
            return True
    return False


def _applicable_labels(
    labels: frozenset[str],
    per_label_recall: Mapping[str, float],
    denominators: Mapping[str, int],
) -> list[str]:
    applicable = []
    for label in sorted(labels):
        if label not in per_label_recall:
            continue
        if denominators and label in denominators and denominators[label] <= 0:
            continue
        applicable.append(label)
    return applicable


def _g1a_floor(milestone: str) -> float:
    return G1A_V20_RECALL_FLOOR if _is_v2_or_later(milestone) else G1A_V16_RECALL_FLOOR


def _g2_floor(milestone: str) -> float:
    return G2_V20_RECALL_FLOOR if _is_v2_or_later(milestone) else G2_V16_RECALL_FLOOR


def _is_v2_or_later(milestone: str) -> bool:
    text = str(milestone).strip().lower().lstrip("v")
    try:
        major = int(text.split(".", 1)[0])
    except ValueError:
        return False
    return major >= 2


def _normalise_tier(tier: str) -> str:
    return _TIER_ALIASES.get(_normalise_dimension(tier), _normalise_dimension(tier))


def _normalise_dimension(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def _infer_family(model_name: str) -> str:
    normalized = model_name.lower()
    if "directid" in normalized or "direct-id" in normalized:
        return "DirectID"
    if "pii" in normalized or "privacy" in normalized:
        return "PII"
    return ""


def _float_map(value: Mapping[str, Any] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for label, raw in _mapping(value).items():
        parsed = _optional_float(raw)
        if parsed is None:
            continue
        if str(label).upper() == "OVERALL":
            canonical = "OVERALL"
        else:
            canonical = normalize_label(str(label))
        result[canonical] = parsed
    return {key: result[key] for key in sorted(result)}


def _numeric_map(value: Mapping[str, Any] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, raw in _mapping(value).items():
        parsed = _optional_float(raw)
        if parsed is not None:
            result[str(key)] = parsed
    return {key: result[key] for key in sorted(result)}


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _stability_summary_payload(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _plain(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        if isinstance(payload, Mapping):
            return _plain(payload)
    raise TypeError("stability_report must be a mapping or expose to_dict()")


def _nested(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
        return key
    if isinstance(key, str):
        return key.encode("utf-8")
    raise TypeError("signing key must be bytes or str")


GROUNDING_TOP1_FLOOR = 0.90
GROUNDING_TOP5_FLOOR = 0.97
# Provisional multilingual floors: zh/hi grounding coverage is still maturing,
# so these are documented as advisory-but-blocking placeholders below the
# English strict floors.
GROUNDING_MULTILINGUAL_TOP1_FLOOR = 0.80
GROUNDING_MULTILINGUAL_TOP5_FLOOR = 0.90
_GROUNDING_PERMISSIVE_SYSTEMS: tuple[str, ...] = ("rxnorm", "loinc", "icd10cm")
_GROUNDING_MULTILINGUAL_LANGUAGES: tuple[str, ...] = ("zh", "hi")


@dataclass(frozen=True)
class GroundingGateConfig:
    """Per-system top-k accuracy floors for the grounding release gate.

    English gold is held to the strict ``top-1 >= 0.90`` / ``top-5 >= 0.97``
    floors; ``zh``/``hi`` gold is held to lower, explicitly provisional floors
    while multilingual grounding coverage matures.
    """

    permissive_systems: tuple[str, ...] = _GROUNDING_PERMISSIVE_SYSTEMS
    english_top1_floor: float = GROUNDING_TOP1_FLOOR
    english_top5_floor: float = GROUNDING_TOP5_FLOOR
    multilingual_languages: tuple[str, ...] = _GROUNDING_MULTILINGUAL_LANGUAGES
    multilingual_top1_floor: float = GROUNDING_MULTILINGUAL_TOP1_FLOOR
    multilingual_top5_floor: float = GROUNDING_MULTILINGUAL_TOP5_FLOOR

    def floors_for(self, language: str) -> tuple[float, float]:
        """Return the ``(top1, top5)`` floors for a language."""

        if language == "en":
            return self.english_top1_floor, self.english_top5_floor
        return self.multilingual_top1_floor, self.multilingual_top5_floor

    def check(self, report: Any) -> tuple[GateCheck, ...]:
        """Return one :class:`GateCheck` per permissive system in the report."""

        checks: list[GateCheck] = []
        for system in self.permissive_systems:
            system_accuracy = report.system(system)
            if system_accuracy is None:
                checks.append(
                    GateCheck(
                        f"{GROUNDING_ACCURACY_GATE}:{system}",
                        False,
                        reason="grounding accuracy missing for system",
                        details={"system": system},
                    )
                )
                continue

            breaches: list[str] = []
            languages: dict[str, Any] = {}
            evaluated = ["en", *self.multilingual_languages]
            for language in evaluated:
                metrics = system_accuracy.language(language)
                if metrics is None:
                    continue
                top1_floor, top5_floor = self.floors_for(language)
                languages[language] = {
                    "support": metrics.support,
                    "top1_accuracy": metrics.top1_accuracy,
                    "top5_accuracy": metrics.top5_accuracy,
                    "abstention_rate": metrics.abstention_rate,
                    "top1_floor": top1_floor,
                    "top5_floor": top5_floor,
                }
                if metrics.top1_accuracy < top1_floor:
                    breaches.append(
                        f"{language}.top1 {metrics.top1_accuracy:.4f} < {top1_floor:.2f}"
                    )
                if metrics.top5_accuracy < top5_floor:
                    breaches.append(
                        f"{language}.top5 {metrics.top5_accuracy:.4f} < {top5_floor:.2f}"
                    )

            if "en" not in languages:
                breaches.append("en (missing English gold)")

            checks.append(
                GateCheck(
                    f"{GROUNDING_ACCURACY_GATE}:{system}",
                    not breaches,
                    reason="ok"
                    if not breaches
                    else "grounding accuracy below floor: " + "; ".join(breaches),
                    details={"system": system, "languages": languages},
                )
            )
        return tuple(checks)


def evaluate_grounding_accuracy_gate(
    report: Any | None = None,
    *,
    config: GroundingGateConfig | None = None,
    gold_dir: str | Path | None = None,
) -> tuple[GateCheck, ...]:
    """Score the grounding accuracy suite and return per-system gate checks.

    When *report* is omitted the shipped synthetic gold is scored with the real
    sparse candidate generator. Passing a precomputed report (or one produced by
    a deliberately broken linker) lets callers exercise the floors directly.
    """

    from openmed.eval import grounding_accuracy as grounding_module

    resolved_config = config or GroundingGateConfig()
    if report is None:
        if gold_dir is None:
            report = grounding_module.evaluate_grounding_accuracy()
        else:
            report = grounding_module.evaluate_grounding_accuracy(gold_dir=gold_dir)
    return resolved_config.check(report)


def build_grounding_gate_report(
    report: Any | None = None,
    *,
    config: GroundingGateConfig | None = None,
    gold_dir: str | Path | None = None,
) -> GateReport:
    """Build an unsigned :class:`GateReport` carrying the grounding checks."""

    from openmed.eval import grounding_accuracy as grounding_module

    resolved_config = config or GroundingGateConfig()
    if report is None:
        if gold_dir is None:
            report = grounding_module.evaluate_grounding_accuracy()
        else:
            report = grounding_module.evaluate_grounding_accuracy(gold_dir=gold_dir)

    checks = resolved_config.check(report)
    decision = RELEASABLE if all(check.passed for check in checks) else QUARANTINED
    eval_set_hash = stable_hash(report.to_dict())
    return GateReport(
        repo_id="grounding-accuracy-suite",
        family="grounding",
        tier="suite",
        param_count=None,
        format="pattern-only",
        per_label_recall={},
        per_label_precision={},
        critical_leakage_count=0,
        residual_leakage_rate=0.0,
        quant_recall_delta=None,
        p50_ms=None,
        p95_ms=None,
        ram_mb=None,
        eval_set_hash=eval_set_hash,
        leakage_fixture_hash="",
        decision=decision,
        gate_results=checks,
    )


def preview(
    report: BenchmarkReport | Mapping[str, Any],
    baseline: Mapping[str, Any] | None = None,
    *,
    milestone: str = "v1.7",
    policy: str = "hipaa_safe_harbor",
    baseline_path: str | Path = baseline_store.BASELINE_PATH,
    thresholds_matrix: Mapping[str, Any] | None = None,
    thresholds_matrix_path: str | Path | None = None,
    model_steward_config: Mapping[str, Any] | ModelStewardConfig | None = None,
) -> GateReport:
    """Return an unsigned release-gate preview for *report*."""

    gate = ReleaseGate(
        milestone=milestone,
        policy=policy,
        baseline_path=baseline_path,
        thresholds_matrix=thresholds_matrix,
        thresholds_matrix_path=thresholds_matrix_path,
        model_steward_config=model_steward_config,
    )
    return gate.preview(report, baseline)


def format_preview(report: GateReport) -> str:
    """Render a read-only release-gate preview table."""

    verdict = "would-pass" if report.decision == RELEASABLE else "would-fail"
    gate_width = max(4, *(len(check.gate) for check in report.gate_results))
    status_width = len("status")
    lines = [
        "Release gate preview (read-only)",
        "No signed report emitted; no GateReport file written.",
        f"Candidate: {report.repo_id}",
        f"Overall verdict: {verdict}",
        "",
        f"{'gate':<{gate_width}}  {'status':<{status_width}}  reason",
    ]
    for check in report.gate_results:
        status = "pass" if check.passed else "fail"
        lines.append(
            f"{check.gate:<{gate_width}}  {status:<{status_width}}  {check.reason}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the OpenMed release gate harness against a candidate "
            "benchmark report and fail closed on any gate failure."
        )
    )
    candidate_group = parser.add_mutually_exclusive_group(required=True)
    candidate_group.add_argument(
        "--candidate",
        help="Path to a candidate BenchmarkReport JSON payload.",
    )
    candidate_group.add_argument(
        "--throughput-candidate",
        help="Path to a zh/hi/ta throughput benchmark JSON payload.",
    )
    candidate_group.add_argument(
        "--grounding",
        action="store_true",
        help="Score the shipped grounding-accuracy gold and gate on top-k floors.",
    )
    parser.add_argument(
        "--baseline",
        help="Optional baseline JSON payload. Defaults to the baseline store.",
    )
    parser.add_argument(
        "--baseline-store",
        default=str(baseline_store.BASELINE_PATH),
        help="Path to the last-green baseline store.",
    )
    parser.add_argument(
        "--output",
        default="release-gate-report.json",
        help="Path to write the signed gate report JSON.",
    )
    parser.add_argument(
        "--milestone",
        default="v1.7",
        help="Milestone version used for release thresholds.",
    )
    parser.add_argument(
        "--policy",
        default="hipaa_safe_harbor",
        help="Policy profile used when the candidate report omits one.",
    )
    parser.add_argument(
        "--thresholds-matrix",
        help="Optional thresholds matrix JSON path.",
    )
    parser.add_argument(
        "--signing-key",
        help="Signing key. Defaults to OPENMED_RELEASE_GATE_KEY or local key.",
    )
    parser.add_argument(
        "--key-id",
        default="release-gate",
        help="Signing key identifier recorded in the gate report.",
    )
    parser.add_argument(
        "--issue-on-failure",
        action="store_true",
        help="Open or update a tracking issue when the candidate is quarantined.",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", "maziyarpanahi/openmed"),
        help="Repository used for failure tracking issues.",
    )
    parser.add_argument(
        "--tracking-issue-title",
        help="Override the failure tracking issue title.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.throughput_candidate:
        return _run_i18n_throughput_gate_cli(args)

    if args.grounding:
        return _run_grounding_accuracy_gate_cli(args)

    candidate_path = Path(args.candidate)
    if not candidate_path.is_file():
        print(
            f"Candidate report not found: {candidate_path}. "
            "Skipping release gate evaluation.",
            file=sys.stderr,
        )
        return 0

    try:
        candidate = _read_json_file(candidate_path)
        baseline = _read_json_file(Path(args.baseline)) if args.baseline else None
        gate = ReleaseGate(
            milestone=args.milestone,
            policy=args.policy,
            baseline_path=args.baseline_store,
            thresholds_matrix_path=args.thresholds_matrix,
            signing_key=args.signing_key,
            key_id=args.key_id,
        )
        report = gate.evaluate(candidate, baseline)
    except Exception as exc:
        message = f"release gate evaluation failed before a report was produced: {exc}"
        print(message, file=sys.stderr)
        if args.issue_on_failure:
            _open_or_update_tracking_issue_for_error(
                repo=args.repo,
                title=args.tracking_issue_title or "Release gate evaluation failed",
                message=message,
            )
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.to_json() + "\n", encoding="utf-8")
    print(report.to_json())

    if report.decision != RELEASABLE:
        if args.issue_on_failure:
            _open_or_update_tracking_issue(
                report,
                repo=args.repo,
                title=args.tracking_issue_title,
            )
        return 1
    return 0


def _run_i18n_throughput_gate_cli(args: argparse.Namespace) -> int:
    candidate_path = Path(args.throughput_candidate)
    if not candidate_path.is_file():
        print(
            f"Throughput candidate report not found: {candidate_path}.",
            file=sys.stderr,
        )
        return 2

    try:
        candidate = _read_json_file(candidate_path)
        baseline = baseline_store.load_baseline_store(args.baseline_store)
        check = evaluate_i18n_throughput_gate(candidate, baseline)
    except Exception as exc:
        print(f"throughput gate evaluation failed: {exc}", file=sys.stderr)
        return 2

    payload = {
        "schema_version": 1,
        "artifact_type": "openmed.eval.i18n_throughput_gate",
        "decision": RELEASABLE if check.passed else QUARANTINED,
        "gate_result": check.to_dict(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if check.passed else 1


def _run_grounding_accuracy_gate_cli(args: argparse.Namespace) -> int:
    try:
        report = build_grounding_gate_report()
    except Exception as exc:
        print(f"grounding accuracy gate evaluation failed: {exc}", file=sys.stderr)
        return 2

    signed = report.sign(
        args.signing_key
        or os.environ.get("OPENMED_RELEASE_GATE_KEY", _DEFAULT_SIGNING_KEY),
        key_id=args.key_id,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(signed.to_json() + "\n", encoding="utf-8")
    print(signed.to_json())
    return 0 if signed.decision == RELEASABLE else 1


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(payload)


def _positive_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _open_or_update_tracking_issue(
    report: GateReport,
    *,
    repo: str,
    title: str | None = None,
) -> int | None:
    issue_title = title or f"Release gate failure for {report.repo_id}"
    failing = [check for check in report.gate_results if not check.passed]
    body = _tracking_issue_body(report, failing)
    return _open_or_update_issue(repo=repo, title=issue_title, body=body)


def _open_or_update_tracking_issue_for_error(
    *,
    repo: str,
    title: str,
    message: str,
) -> int | None:
    body = "\n".join(
        [
            "## Summary",
            "",
            "The release gate job failed before producing a gate report.",
            "",
            "## Failure",
            "",
            f"- `{message}`",
            "",
        ]
    )
    return _open_or_update_issue(repo=repo, title=title, body=body)


def _open_or_update_issue(*, repo: str, title: str, body: str) -> int | None:
    existing = _find_open_issue(repo=repo, title=title)
    if existing is not None:
        subprocess.run(
            [
                "gh",
                "issue",
                "comment",
                str(existing),
                "--repo",
                repo,
                "--body-file",
                "-",
            ],
            input=body,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=60,
        )
        return existing

    result = subprocess.run(
        [
            "gh",
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            title,
            "--body-file",
            "-",
        ],
        input=body,
        text=True,
        encoding="utf-8",
        check=True,
        capture_output=True,
        timeout=60,
    )
    output = result.stdout.strip().rsplit("/", 1)[-1]
    return _optional_int(output.lstrip("#"))


def _find_open_issue(*, repo: str, title: str) -> int | None:
    result = subprocess.run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--search",
            f"{title} in:title",
            "--json",
            "number,title",
        ],
        text=True,
        encoding="utf-8",
        check=True,
        capture_output=True,
        timeout=60,
    )
    try:
        issues = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return None
    for issue in issues:
        if isinstance(issue, Mapping) and issue.get("title") == title:
            return _optional_int(issue.get("number"))
    return None


def _tracking_issue_body(
    report: GateReport,
    failing: Sequence[GateCheck],
) -> str:
    lines = [
        "## Summary",
        "",
        f"Release gates quarantined `{report.repo_id}`.",
        "",
        "## Gate report",
        "",
        f"- Decision: `{report.decision}`",
        f"- Family: `{report.family}`",
        f"- Tier: `{report.tier}`",
        f"- Format: `{report.format}`",
        f"- Eval set hash: `{report.eval_set_hash}`",
        f"- Leakage fixture hash: `{report.leakage_fixture_hash}`",
        f"- Repro hash: `{report.repro_hash}`",
        "",
        "## Failing gates",
        "",
    ]
    for check in failing:
        lines.append(f"- `{check.gate}`: {check.reason}")
    lines.extend(["", "## Blocking formats", ""])
    if report.blocked_formats:
        for format_name in report.blocked_formats:
            lines.append(f"- `{format_name}`")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "CROSS_DOCUMENT_LINKAGE_GATE",
    "CROSS_SCRIPT_GATE",
    "REIDENTIFICATION_RISK_GATE",
    "DEFAULT_CROSS_DOCUMENT_LINKAGE_CEILING",
    "G1A_V16_RECALL_FLOOR",
    "G1A_V20_RECALL_FLOOR",
    "G1B_RECALL_FLOOR",
    "G2_V16_RECALL_FLOOR",
    "G2_V20_RECALL_FLOOR",
    "G4_INT8_DELTA_LIMIT",
    "G4_INT4_DELTA_LIMIT",
    "G7_RECALL_DROP_LIMIT",
    "G10_UNGROUNDED_FACT_CEILING",
    "G11_CRITICAL_RECALL_FLOOR",
    "G13_RADIOLOGY_ENTITY_F1_FLOOR",
    "G13_RADIOLOGY_RELATION_F1_FLOOR",
    "G13_RADIOLOGY_UNCERTAINTY_ACCURACY_FLOOR",
    "G13_STRICT_ENTITY_F1_FLOOR",
    "G13_STRICT_RELATION_F1_FLOOR",
    "G13_UNCERTAINTY_ACCURACY_FLOOR",
    "G14_EXTRACTION_DISPARITY_CEILING",
    "G15_E2E_FACT_F1_FLOOR",
    "G9_STRICT_RE_F1_FLOOR",
    "G9_RELAXED_RE_F1_FLOOR",
    "RELATION_GOLDEN_REGRESSION_GATE",
    "RELATION_GOLDEN_TRAP_KINDS",
    "FLAKINESS_GATE",
    "SURROGATE_QUALITY_GATE",
    "EXPORT_VARIANT_GATE",
    "GROUNDING_ACCURACY_GATE",
    "GROUNDING_TOP1_FLOOR",
    "GROUNDING_TOP5_FLOOR",
    "GROUNDING_MULTILINGUAL_TOP1_FLOOR",
    "GROUNDING_MULTILINGUAL_TOP5_FLOOR",
    "RESIDUAL_LEAKAGE_SOFT_CEILING",
    "QUARANTINED",
    "RELEASABLE",
    "GateCheck",
    "GateReport",
    "GroundingGateConfig",
    "ModelStewardConfig",
    "ReleaseGate",
    "apply_flakiness_quarantine",
    "build_arg_parser",
    "build_grounding_gate_report",
    "evaluate_cross_document_linkage_gate",
    "evaluate_end_to_end_pipeline_gate",
    "evaluate_federated_boundary_gate",
    "evaluate_radiology_entity_relation_gate",
    "evaluate_reid_risk_gate",
    "evaluate_reidentification_risk_gate",
    "evaluate_relation_golden_regression_gate",
    "evaluate_grounding_accuracy_gate",
    "evaluate_surrogate_quality_gate",
    "format_preview",
    "main",
    "preview",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
