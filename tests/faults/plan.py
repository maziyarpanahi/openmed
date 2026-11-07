"""Versioned, seeded fault plans for the service fault-injection matrix.

A plan names cases. Each case selects a component adapter, a fault category and
kind, a named injection point, the phase relative to the component's declared
commit point, a trigger (an explicit call number or a seeded call range) and
the one documented outcome the component must produce. Plans and everything
derived from them hold identifiers, enums, numbers and digests only.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "openmed" / "core" / "schemas" / "json"
PLAN_SCHEMA_PATH = SCHEMA_DIR / "service_fault_plan.schema.json"
MANIFEST_SCHEMA_PATH = SCHEMA_DIR / "service_fault_manifest.schema.json"
SERVICE_PLAN_PATH = Path(__file__).with_name("service_fault_plan.json")

SCHEMA_VERSION = "1.0.0"

FAULTS_BY_CATEGORY: Mapping[str, tuple[str, ...]] = {
    "disk": ("enospc", "eio", "partial_write", "fsync_failure"),
    "clock": (
        "clock_skew",
        "clock_jump_forward",
        "clock_jump_backward",
        "monotonic_stall",
    ),
    "process": ("process_kill",),
    "network": ("connection_refused", "timeout", "connection_reset", "egress_denied"),
    "cancellation": ("cancellation",),
    "corrupted_state": ("truncated_state", "garbled_state"),
}
CATEGORIES = tuple(FAULTS_BY_CATEGORY)
PHASES = ("before_commit", "during_commit", "after_commit")
OUTCOMES = (
    "no_effect",
    "retryable",
    "quarantined",
    "rolled_back",
    "committed_exactly_once",
)
AMBIGUOUS = "ambiguous"
OBSERVED_OUTCOMES = (*OUTCOMES, AMBIGUOUS)


class FaultPlanError(ValueError):
    """Value-free plan rejection carrying a fixed reason code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def canonical_json(value: Any) -> bytes:
    """Serialize ``value`` deterministically for digests and manifests.

    Args:
        value: JSON-compatible value.

    Returns:
        Sorted, compact, ASCII-only JSON bytes.
    """
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def sha256_digest(payload: bytes) -> str:
    """Return a prefixed SHA-256 digest for ``payload``."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def load_json_schema(path: Path) -> dict[str, Any]:
    """Load one bundled fault-harness JSON Schema."""
    with path.open("r", encoding="utf-8") as handle:
        schema: dict[str, Any] = json.load(handle)
    return schema


def _derived_seed(seed: int, *parts: str) -> int:
    material = ":".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


@dataclass(frozen=True)
class FaultTrigger:
    """When a case fires: one explicit call or one seeded call in a range."""

    at_call: int | None = None
    min_call: int | None = None
    max_call: int | None = None
    persistent: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FaultTrigger":
        """Build a trigger from a schema-valid mapping."""
        seeded = data.get("seeded")
        if seeded is not None:
            low, high = int(seeded["min_call"]), int(seeded["max_call"])
            if low > high:
                raise FaultPlanError("fault_trigger_range_invalid")
            return cls(
                min_call=low,
                max_call=high,
                persistent=bool(data.get("persistent", False)),
            )
        return cls(
            at_call=int(data["at_call"]),
            persistent=bool(data.get("persistent", False)),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the declared (unresolved) trigger mapping."""
        payload: dict[str, Any] = {}
        if self.at_call is not None:
            payload["at_call"] = self.at_call
        else:
            payload["seeded"] = {"min_call": self.min_call, "max_call": self.max_call}
        if self.persistent:
            payload["persistent"] = True
        return payload

    def resolve(self, rng: random.Random) -> int:
        """Return the 1-based call number on which the fault first fires."""
        if self.at_call is not None:
            return self.at_call
        assert self.min_call is not None and self.max_call is not None
        return rng.randint(self.min_call, self.max_call)


@dataclass(frozen=True)
class KnownGap:
    """A documented master behaviour that misses the expected outcome."""

    issue: int
    reason_code: str


@dataclass(frozen=True)
class FaultCase:
    """One declared fault case of a plan."""

    case_id: str
    component: str
    category: str
    fault: str
    injection_point: str
    phase: str
    trigger: FaultTrigger
    expected_outcome: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    require_no_residue: bool = False
    known_gap: KnownGap | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FaultCase":
        """Build a case from a schema-valid mapping."""
        known_gap = data.get("known_gap")
        return cls(
            case_id=str(data["case_id"]),
            component=str(data["component"]),
            category=str(data["category"]),
            fault=str(data["fault"]),
            injection_point=str(data["injection_point"]),
            phase=str(data["phase"]),
            trigger=FaultTrigger.from_mapping(data["trigger"]),
            expected_outcome=str(data["expected_outcome"]),
            parameters=dict(data.get("parameters", {})),
            require_no_residue=bool(data.get("require_no_residue", False)),
            known_gap=(
                None
                if known_gap is None
                else KnownGap(
                    issue=int(known_gap["issue"]),
                    reason_code=str(known_gap["reason_code"]),
                )
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the declared case mapping."""
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "component": self.component,
            "category": self.category,
            "fault": self.fault,
            "injection_point": self.injection_point,
            "phase": self.phase,
            "trigger": self.trigger.to_mapping(),
            "expected_outcome": self.expected_outcome,
        }
        if self.parameters:
            payload["parameters"] = dict(self.parameters)
        if self.require_no_residue:
            payload["require_no_residue"] = True
        if self.known_gap is not None:
            payload["known_gap"] = {
                "issue": self.known_gap.issue,
                "reason_code": self.known_gap.reason_code,
            }
        return payload

    def number(self, name: str, default: float) -> float:
        """Return a numeric case parameter."""
        value = self.parameters.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FaultPlanError("fault_parameter_invalid")
        return float(value)

    def token(self, name: str, default: str) -> str:
        """Return a controlled-token case parameter."""
        value = self.parameters.get(name, default)
        if not isinstance(value, str):
            raise FaultPlanError("fault_parameter_invalid")
        return value


@dataclass(frozen=True)
class ResolvedCase:
    """A case bound to its plan seed, with a reproducible identity."""

    plan_id: str
    plan_seed: int
    case: FaultCase
    fault_call: int
    case_digest: str

    def rng(self, purpose: str) -> random.Random:
        """Return a deterministic RNG for one named purpose of this case."""
        return random.Random(_derived_seed(self.plan_seed, self.case.case_id, purpose))

    def fires_on(self, call_number: int) -> bool:
        """Return whether the fault fires on ``call_number`` (1-based)."""
        if self.case.trigger.persistent:
            return call_number >= self.fault_call
        return call_number == self.fault_call


@dataclass(frozen=True)
class FaultPlan:
    """A validated, seeded fault plan."""

    plan_id: str
    seed: int
    cases: tuple[FaultCase, ...]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FaultPlan":
        """Validate ``data`` against the plan schema and build a plan.

        Args:
            data: Parsed plan document.

        Returns:
            The validated plan.

        Raises:
            FaultPlanError: With a fixed code when the plan is invalid.
        """
        validator = Draft202012Validator(load_json_schema(PLAN_SCHEMA_PATH))
        if next(iter(validator.iter_errors(data)), None) is not None:
            raise FaultPlanError("fault_plan_schema_invalid")
        cases = tuple(FaultCase.from_mapping(item) for item in data["cases"])
        identifiers = [case.case_id for case in cases]
        if len(set(identifiers)) != len(identifiers):
            raise FaultPlanError("fault_case_duplicate")
        for case in cases:
            if case.fault not in FAULTS_BY_CATEGORY[case.category]:
                raise FaultPlanError("fault_category_mismatch")
        return cls(plan_id=str(data["plan_id"]), seed=int(data["seed"]), cases=cases)

    @classmethod
    def load(cls, path: Path = SERVICE_PLAN_PATH) -> "FaultPlan":
        """Load and validate a plan document from ``path``."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise FaultPlanError("fault_plan_unreadable") from None
        return cls.from_mapping(data)

    def to_mapping(self) -> dict[str, Any]:
        """Return the declared plan document."""
        return {
            "schema_version": SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "seed": self.seed,
            "cases": [case.to_mapping() for case in self.cases],
        }

    def digest(self) -> str:
        """Return the digest of the declared plan."""
        return sha256_digest(canonical_json(self.to_mapping()))

    def with_seed(self, seed: int) -> "FaultPlan":
        """Return the same cases bound to another seed."""
        return FaultPlan.from_mapping({**self.to_mapping(), "seed": seed})

    def case(self, case_id: str) -> FaultCase:
        """Return one declared case by identifier."""
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise FaultPlanError("fault_case_unknown")

    def resolve(self, case: FaultCase) -> ResolvedCase:
        """Bind one case to the plan seed and compute its identity."""
        rng = random.Random(_derived_seed(self.seed, case.case_id, "trigger"))
        fault_call = case.trigger.resolve(rng)
        identity = {
            "schema_version": SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "seed": self.seed,
            "case": case.to_mapping(),
            "fault_call": fault_call,
        }
        return ResolvedCase(
            plan_id=self.plan_id,
            plan_seed=self.seed,
            case=case,
            fault_call=fault_call,
            case_digest=sha256_digest(canonical_json(identity)),
        )

    def resolved(self) -> tuple[ResolvedCase, ...]:
        """Return every case bound to the plan seed, in plan order."""
        return tuple(self.resolve(case) for case in self.cases)


__all__ = [
    "AMBIGUOUS",
    "CATEGORIES",
    "FAULTS_BY_CATEGORY",
    "FaultCase",
    "FaultPlan",
    "FaultPlanError",
    "FaultTrigger",
    "KnownGap",
    "MANIFEST_SCHEMA_PATH",
    "OBSERVED_OUTCOMES",
    "OUTCOMES",
    "PHASES",
    "PLAN_SCHEMA_PATH",
    "ResolvedCase",
    "SCHEMA_VERSION",
    "SERVICE_PLAN_PATH",
    "canonical_json",
    "load_json_schema",
    "sha256_digest",
]
