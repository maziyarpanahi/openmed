"""Release gate harness for benchmark reports.

The gate evaluates a candidate benchmark report against the section 6.4
G1a-G8 release criteria, reads the last-green baseline store without mutating
it, and emits a signed, reproducible gate report.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core import baseline as baseline_store
from openmed.core import policy as policy_module
from openmed.core import quality_gates
from openmed.core.audit import AuditSignature, stable_hash
from openmed.core.labels import normalize_label
from openmed.core.thresholds import (
    load_thresholds,
    profile_recall_floor,
    validate_threshold_matrix,
)
from openmed.eval.metrics import normalize_eval_spans
from openmed.eval.report import BenchmarkReport


RELEASABLE = "RELEASABLE"
QUARANTINED = "QUARANTINED"

G1A_V16_RECALL_FLOOR = 0.990
G1A_V20_RECALL_FLOOR = 0.995
G1B_RECALL_FLOOR = 0.995
G2_V16_RECALL_FLOOR = 0.980
G2_V20_RECALL_FLOOR = 0.990
G4_INT8_DELTA_LIMIT = 0.005
G4_INT4_DELTA_LIMIT = 0.010
G7_RECALL_DROP_LIMIT = 0.002
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
    repro_hash: str = ""
    signature: AuditSignature | None = None

    def __post_init__(self) -> None:
        self.per_label_recall = _float_map(self.per_label_recall)
        self.per_label_precision = _float_map(self.per_label_precision)
        self.blocked_formats = tuple(self.blocked_formats)
        self.gate_results = tuple(self.gate_results)
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
            blocked_formats=tuple(str(item) for item in data.get("blocked_formats", [])),
            repro_hash=str(data.get("repro_hash", "")),
            signature=(
                AuditSignature.from_dict(signature_data)
                if isinstance(signature_data, Mapping)
                else None
            ),
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> "GateReport":
        return cls.from_dict(json.loads(data))


class ReleaseGate:
    """Evaluate benchmark reports against the G1a-G8 release gates."""

    def __init__(
        self,
        *,
        milestone: str = "v1.6",
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
        self.model_steward_config = ModelStewardConfig.from_mapping(model_steward_config)
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

        payload = _report_payload(report)
        metrics = _mapping(payload.get("metrics"))
        metadata = _mapping(payload.get("metadata"))
        identity = _identity(payload, metrics, metadata)
        policy_name = str(metadata.get("policy") or payload.get("policy") or self.policy)

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
        quant_delta = _quant_recall_delta(metrics, metadata, identity["format"])
        p50_ms, p95_ms = _latency(metrics, metadata)
        ram_mb = _ram_mb(metrics, metadata)
        baseline_entry = self._resolve_baseline(identity, baseline)
        target_leakage = self.model_steward_config.target_for(identity["family"])
        if profile is not None and profile.strict_no_leak:
            target_leakage = min(target_leakage, 0.0)

        checks.append(_manifest_coherence_check(identity, metadata))
        checks.append(_calibration_check(metadata, profile))
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
        checks.append(_g3_check(critical_leakage_count))
        checks.append(_g4_check(identity["format"], quant_delta))
        checks.append(_g5_check(identity["tier"], p50_ms, p95_ms, ram_mb))
        checks.append(_g6_check(p50_ms, p95_ms))
        checks.append(
            _g7_check(
                baseline_entry,
                per_label_recall,
                residual_leakage_rate,
                target_leakage=target_leakage,
            )
        )
        checks.append(_g8_check(metadata))

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
            quant_recall_delta=quant_delta,
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
        return gate_report.sign(signing_key or self.signing_key, key_id=key_id or self.key_id)

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
        if profile is not None and threshold_matrix is not None and profile.strict_no_leak:
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

    manifest = _mapping(metadata.get("manifest"))
    mismatches: dict[str, dict[str, Any]] = {}
    if manifest:
        manifest_fields = {
            "repo_id": manifest.get("repo_id"),
            "family": manifest.get("family"),
            "tier": manifest.get("tier"),
            "param_count": manifest.get("param_count"),
            "format": manifest.get("format") or manifest.get("model_format"),
        }
        for key, manifest_value in manifest_fields.items():
            if manifest_value is None:
                continue
            candidate_value = identity.get(key)
            if key == "param_count":
                manifest_value = _optional_int(manifest_value)
            if str(manifest_value) != str(candidate_value):
                mismatches[key] = {
                    "manifest": manifest_value,
                    "candidate": candidate_value,
                }

    if mismatches:
        return GateCheck(
            "manifest_coherence",
            False,
            reason="candidate metadata does not match manifest",
            details={"mismatches": mismatches},
        )

    return GateCheck("manifest_coherence", True)


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


def _g4_check(format_name: str, quant_delta: float | None) -> GateCheck:
    normalized = _normalise_dimension(format_name)
    if "int8" in normalized or "8bit" in normalized or "8-bit" in normalized:
        limit = G4_INT8_DELTA_LIMIT
    elif "int4" in normalized or "4bit" in normalized or "4-bit" in normalized:
        limit = G4_INT4_DELTA_LIMIT
    else:
        return GateCheck("G4", True, reason="not applicable")

    if quant_delta is None:
        return GateCheck(
            "G4",
            False,
            reason="quantized artifacts require recall delta evidence",
            blocking_format=format_name,
        )
    delta = _normalise_delta(quant_delta)
    return GateCheck(
        "G4",
        delta < limit,
        reason="ok" if delta < limit else "quantized recall delta exceeds limit",
        details={"delta": delta, "limit": limit},
        blocking_format=None if delta < limit else format_name,
    )


def _g5_check(
    tier: str,
    p50_ms: float | None,
    p95_ms: float | None,
    ram_mb: float | None,
) -> GateCheck:
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

        checked += len(entities)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            quality_gates.validate_entity_spans(entities, text)
        invalid_entities = [
            entity
            for entity in entities
            if isinstance(entity.metadata, Mapping)
            and entity.metadata.get("span_valid") is False
        ]
        if caught or invalid_entities:
            problems.append(
                {
                    "fixture_index": index,
                    "span_warnings": [str(item.message) for item in caught],
                    "invalid_spans": len(invalid_entities),
                }
            )

        resolved = quality_gates.resolve_overlapping_entities(entities)
        resolved_overlaps += max(len(entities) - len(resolved), 0)
        residual_overlaps = quality_gates.detect_overlapping_entities(resolved)
        if residual_overlaps:
            problems.append(
                {
                    "fixture_index": index,
                    "residual_overlaps": len(residual_overlaps),
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
        sum(value for label, value in leaked_by_label.items() if label in _CRITICAL_LABELS)
    )


def _residual_leakage_rate(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> float:
    value = _first_value(
        metadata.get("residual_leakage_rate"),
        metrics.get("residual_leakage_rate"),
        _nested(metrics, "leakage", "overall"),
    )
    parsed = _optional_float(value)
    return 1.0 if parsed is None else parsed


def _quant_recall_delta(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    format_name: str,
) -> float | None:
    raw = _first_value(
        metadata.get("quant_recall_delta"),
        metrics.get("quant_recall_delta"),
        _nested(metrics, "quantization", "recall_delta"),
    )
    if isinstance(raw, Mapping):
        format_key = _normalise_dimension(format_name)
        for key, value in raw.items():
            if _normalise_dimension(str(key)) == format_key:
                return _optional_float(value)
        return None
    return _optional_float(raw)


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


def _normalise_delta(value: float) -> float:
    delta = abs(float(value))
    if delta > 0.05:
        return delta / 100.0
    return delta


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


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


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


__all__ = [
    "G1A_V16_RECALL_FLOOR",
    "G1A_V20_RECALL_FLOOR",
    "G1B_RECALL_FLOOR",
    "G2_V16_RECALL_FLOOR",
    "G2_V20_RECALL_FLOOR",
    "G4_INT8_DELTA_LIMIT",
    "G4_INT4_DELTA_LIMIT",
    "G7_RECALL_DROP_LIMIT",
    "RESIDUAL_LEAKAGE_SOFT_CEILING",
    "QUARANTINED",
    "RELEASABLE",
    "GateCheck",
    "GateReport",
    "ModelStewardConfig",
    "ReleaseGate",
]
