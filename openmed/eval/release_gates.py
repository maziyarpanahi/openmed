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
from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.core.thresholds import (
    DEFAULT_MEMBERSHIP_ADVANTAGE_CEILING,
    load_thresholds,
    profile_recall_floor,
    validate_threshold_matrix,
)
from openmed.eval.fairness import DEFAULT_ZERO_SHOT_LEAKAGE_FLOOR
from openmed.eval.metrics import (
    CRITICAL_FINDING_CATEGORY_DIAGNOSIS,
    CRITICAL_FINDING_CATEGORY_DRUG_ALLERGY,
    normalize_critical_finding_category,
    normalize_eval_spans,
)
from openmed.eval.nano_cert import certify_measurements
from openmed.eval.quant_delta import (
    COREML_RECALL_DELTA_LIMIT,
    INT4_RECALL_DELTA_LIMIT,
    INT8_RECALL_DELTA_LIMIT,
    QuantRecallDeltaResult,
    evaluate_quant_recall_delta,
)
from openmed.eval.report import BenchmarkReport

RELEASABLE = "RELEASABLE"
QUARANTINED = "QUARANTINED"
FLAKINESS_GATE = "flakiness"

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
RESIDUAL_LEAKAGE_SOFT_CEILING = 0.005

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

    def __post_init__(self) -> None:
        self.per_label_recall = _float_map(self.per_label_recall)
        self.per_label_precision = _float_map(self.per_label_precision)
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
        checks.append(_adversarial_recall_under_attack_check(metrics, metadata))
        checks.append(_g3_check(critical_leakage_count))
        checks.append(_g11_critical_finding_recall_check(metrics, metadata))
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
        checks.append(_membership_leakage_check(metrics, metadata))
        checks.append(_g8_check(metadata))
        checks.append(_g9_relation_extraction_check(metrics, metadata))
        coreml_manifest = _coreml_conversion_manifest(metadata)
        if coreml_manifest or _normalise_dimension(identity["format"]).startswith(
            "coreml"
        ):
            checks.append(_coreml_ane_residency_check(coreml_manifest, metadata))
            checks.append(_coreml_variant_parity_check(coreml_manifest, metadata))
        checks.append(_zero_shot_language_leakage_check(metrics, metadata))
        federated_check = _federated_boundary_check(metrics, metadata)
        if federated_check is not None:
            checks.append(federated_check)
        checks.append(_k_floor_check(metrics, metadata))

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
        gate_results=tuple(checks),
        policy=report.policy,
        threshold_profile=report.threshold_profile,
        target_leakage_rate=report.target_leakage_rate,
        blocked_formats=report.blocked_formats,
        stability_summary=summary,
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
        readme_mismatches = _readme_manifest_mismatches(rows, readme_path)
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
        supported_languages = set(SUPPORTED_LANGUAGES)
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
) -> dict[str, Any]:
    if not readme_path.exists():
        return {"path": str(readme_path), "error": "README evidence is missing"}

    text = readme_path.read_text(encoding="utf-8")
    declared = _readme_declared_counts(text)
    mismatches: dict[str, Any] = {}
    model_count = len(rows)
    pii_count = len([row for row in rows if _is_pii_manifest_row(row)])
    pii_languages = _manifest_pii_languages(rows)

    if declared.get("models") is not None and model_count < declared["models"]:
        mismatches["models"] = {
            "readme_floor": declared["models"],
            "manifest": model_count,
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
                "bootstrap": dict(
                    residual_risk.get("bootstrap")
                    if isinstance(residual_risk.get("bootstrap"), Mapping)
                    else {}
                ),
            },
            "route_counts": dict(
                abstention.get("route_counts")
                if isinstance(abstention.get("route_counts"), Mapping)
                else {}
            ),
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

    observed = {"p50_ms": p50_ms, "p95_ms": p95_ms, "ram_mb": ram_mb}
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


def _critical_leakage_count(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> int:
    direct = _first_value(
        metadata.get("critical_leakage_count"),
        metrics.get("critical_leakage_count"),
        _nested(metrics, "leakage", "critical_leakage_count"),
    )
    parsed = _optional_int(direct)
    if parsed is not None:
        return parsed

    leaked_by_label = _float_map(_nested(metrics, "leakage", "leaked_chars_by_label"))
    return int(
        sum(
            value
            for label, value in leaked_by_label.items()
            if label in _CRITICAL_LABELS
        )
    )


def _residual_leakage_rate(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> float:
    value = _first_value(
        metadata.get("residual_leakage_rate"),
        metrics.get("residual_leakage_rate"),
        metrics.get("federated_boundary_leakage_rate"),
        _nested(metrics, "boundary_leakage", "rate"),
        _nested(metrics, "leakage", "overall"),
    )
    parsed = _optional_float(value)
    return 1.0 if parsed is None else parsed


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
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to a candidate BenchmarkReport JSON payload.",
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


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(payload)


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
    "G1A_V16_RECALL_FLOOR",
    "G1A_V20_RECALL_FLOOR",
    "G1B_RECALL_FLOOR",
    "G2_V16_RECALL_FLOOR",
    "G2_V20_RECALL_FLOOR",
    "G4_INT8_DELTA_LIMIT",
    "G4_INT4_DELTA_LIMIT",
    "G7_RECALL_DROP_LIMIT",
    "G11_CRITICAL_RECALL_FLOOR",
    "G9_STRICT_RE_F1_FLOOR",
    "G9_RELAXED_RE_F1_FLOOR",
    "FLAKINESS_GATE",
    "RESIDUAL_LEAKAGE_SOFT_CEILING",
    "QUARANTINED",
    "RELEASABLE",
    "GateCheck",
    "GateReport",
    "ModelStewardConfig",
    "ReleaseGate",
    "apply_flakiness_quarantine",
    "build_arg_parser",
    "evaluate_federated_boundary_gate",
    "format_preview",
    "main",
    "preview",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
