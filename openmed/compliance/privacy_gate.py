"""Deterministic, PHI-safe aggregation of privacy release gates.

Gate producers provide a stable name, an explicit state, an aggregate finding
count, and (optionally) a SHA-256 fingerprint. The aggregate record contains
only those safe summaries. Finding details, free-text reasons, and arbitrary
metadata are deliberately not part of this API, so a release decision can be
logged or persisted without copying sensitive values across the boundary.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sized
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from openmed.core.audit import stable_hash

PRIVACY_GATE_SCHEMA_VERSION: Final = 1
PRIVACY_GATE_REPORT_TYPE: Final = "privacy_release_gate"

_DIGEST_PATTERN: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_PATTERN: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_STATUS_KEYS: Final = ("status", "state")
_COUNT_KEYS: Final = ("finding_count", "findings_count", "count")


class GateStatus(str, Enum):
    """Explicit state reported by one privacy gate."""

    BLOCKING = "blocking"
    WARNING = "warning"
    WAIVED = "waived"


class ReleaseDecision(str, Enum):
    """Deterministic aggregate decision derived from gate states."""

    BLOCKED = "blocked"
    WARNING = "warning"
    RELEASED = "released"


@dataclass(frozen=True)
class PrivacyGateResult:
    """A PHI-safe result from one privacy release gate.

    ``finding_count`` is an aggregate only. A producer may supply a
    ``fingerprint`` over its private finding set, but the finding set itself is
    intentionally not accepted as a stored field. ``from_mapping`` and
    ``from_findings`` are convenience boundaries for callers that still hold
    findings locally; they retain only the count and a safe fingerprint.
    """

    gate: str
    status: GateStatus | str
    finding_count: int = 0
    fingerprint: str | None = None
    waiver_code: str | None = None

    def __post_init__(self) -> None:
        gate = _safe_identifier(self.gate, field_name="gate")
        status = _coerce_status(self.status)
        finding_count = _coerce_count(self.finding_count)
        waiver_code = self.waiver_code
        if waiver_code is not None:
            waiver_code = _safe_identifier(waiver_code, field_name="waiver_code")
            if status is not GateStatus.WAIVED:
                raise ValueError("waiver_code is valid only for a waived gate")

        fingerprint = self.fingerprint
        if fingerprint is None:
            fingerprint = stable_hash(
                {
                    "kind": "openmed-privacy-gate-result",
                    "gate": gate,
                    "status": status.value,
                    "finding_count": finding_count,
                    "waiver_code": waiver_code,
                }
            )
        else:
            _require_digest(fingerprint, field_name="fingerprint")

        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "finding_count", finding_count)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "waiver_code", waiver_code)

    @classmethod
    def from_findings(
        cls,
        gate: str,
        status: GateStatus | str,
        findings: Sized,
        *,
        fingerprint: str | None = None,
        waiver_code: str | None = None,
    ) -> "PrivacyGateResult":
        """Create a result while discarding local finding values.

        ``findings`` must be a sized, non-text collection. The values are
        never inspected, copied, or included in an exception or report.
        """

        finding_count = _count_findings(findings)
        return cls(
            gate=gate,
            status=status,
            finding_count=finding_count,
            fingerprint=fingerprint,
            waiver_code=waiver_code,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PrivacyGateResult":
        """Create a result from an allow-listed mapping.

        Accepted keys are ``gate``/``name``, ``status``/``state``, a count,
        ``fingerprint``/``gate_fingerprint``, and ``waiver_code``. If no count
        is supplied, a sized ``findings`` collection is counted without
        retaining its values. Other keys, including free-text details, are
        ignored rather than copied into the result.
        """

        if not isinstance(value, Mapping):
            raise TypeError("gate result must be a mapping")

        gate = value.get("gate", value.get("name"))
        status = _first_present(value, _STATUS_KEYS)
        count = _first_present(value, _COUNT_KEYS)
        if count is None:
            findings = value.get("findings")
            count = 0 if findings is None else _count_findings(findings)

        return cls(
            gate=gate,
            status=status,
            finding_count=count,
            fingerprint=value.get("fingerprint", value.get("gate_fingerprint")),
            waiver_code=value.get("waiver_code"),
        )

    @property
    def state(self) -> GateStatus:
        """Alias for callers that use ``state`` rather than ``status``."""

        return self.status

    @property
    def name(self) -> str:
        """Alias for callers that use ``name`` rather than ``gate``."""

        return self.gate

    @property
    def count(self) -> int:
        """Return the aggregate number of findings."""

        return self.finding_count

    def to_dict(self) -> dict[str, Any]:
        """Return the counts-only, JSON-compatible gate summary."""

        result: dict[str, Any] = {
            "gate": self.gate,
            "status": self.status.value,
            "finding_count": self.finding_count,
            "fingerprint": self.fingerprint,
        }
        if self.waiver_code is not None:
            result["waiver_code"] = self.waiver_code
        return result


@dataclass(frozen=True)
class PrivacyReleaseGateRecord:
    """A stable counts-only decision record for a set of privacy gates."""

    gates: tuple[PrivacyGateResult, ...]
    schema_version: int = PRIVACY_GATE_SCHEMA_VERSION
    report_type: str = PRIVACY_GATE_REPORT_TYPE

    def __post_init__(self) -> None:
        if self.schema_version != PRIVACY_GATE_SCHEMA_VERSION:
            raise ValueError("unsupported privacy gate schema version")
        if self.report_type != PRIVACY_GATE_REPORT_TYPE:
            raise ValueError("unsupported privacy gate report type")

        try:
            normalized = tuple(_coerce_gate_result(item) for item in self.gates)
        except TypeError:
            raise TypeError("gates must be an iterable of gate results") from None
        if not normalized:
            raise ValueError("at least one privacy gate result is required")

        ordered = tuple(sorted(normalized, key=lambda item: item.gate))
        names = tuple(item.gate for item in ordered)
        if len(names) != len(set(names)):
            raise ValueError("privacy gate names must be unique")
        object.__setattr__(self, "gates", ordered)

    @property
    def counts(self) -> Mapping[str, int]:
        """Return counts for every explicit gate state in stable order."""

        return MappingProxyType(
            {
                status.value: sum(gate.status is status for gate in self.gates)
                for status in GateStatus
            }
        )

    @property
    def status_counts(self) -> Mapping[str, int]:
        """Alias for :attr:`counts`."""

        return self.counts

    @property
    def decision(self) -> ReleaseDecision:
        """Return the fail-closed precedence decision.

        A blocking gate wins over warnings. Waived gates remain visible in the
        counts but do not block release. With no blocking or warning gates, the
        decision is ``released``.
        """

        if self.counts[GateStatus.BLOCKING.value]:
            return ReleaseDecision.BLOCKED
        if self.counts[GateStatus.WARNING.value]:
            return ReleaseDecision.WARNING
        return ReleaseDecision.RELEASED

    @property
    def released(self) -> bool:
        """Whether no blocking or warning gate remains."""

        return self.decision is ReleaseDecision.RELEASED

    @property
    def gate_fingerprints(self) -> Mapping[str, str]:
        """Return gate fingerprints keyed by their stable gate names."""

        return MappingProxyType({gate.gate: gate.fingerprint for gate in self.gates})

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_type": self.report_type,
            "decision": self.decision.value,
            "counts": dict(self.counts),
            "gates": [gate.to_dict() for gate in self.gates],
        }

    @property
    def fingerprint(self) -> str:
        """Return the stable fingerprint of the decision record payload."""

        return stable_hash(self._payload())

    @property
    def decision_fingerprint(self) -> str:
        """Alias for the aggregate record fingerprint."""

        return self.fingerprint

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible decision record."""

        return {**self._payload(), "fingerprint": self.fingerprint}

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the decision record using canonical JSON options."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PrivacyReleaseGateRecord":
        """Rebuild a record from its counts-only representation."""

        if not isinstance(value, Mapping):
            raise TypeError("privacy gate record must be a mapping")
        if value.get("schema_version") != PRIVACY_GATE_SCHEMA_VERSION:
            raise ValueError("unsupported privacy gate schema version")
        if value.get("report_type") != PRIVACY_GATE_REPORT_TYPE:
            raise ValueError("unsupported privacy gate report type")
        gates = value.get("gates")
        if isinstance(gates, (str, bytes, bytearray)) or not isinstance(
            gates, Iterable
        ):
            raise TypeError("privacy gate record gates must be an iterable")
        return cls(tuple(PrivacyGateResult.from_mapping(item) for item in gates))

    @classmethod
    def from_json(cls, value: str) -> "PrivacyReleaseGateRecord":
        """Rebuild a record from JSON without trusting supplied fingerprints."""

        try:
            payload = json.loads(value)
        except (TypeError, ValueError):
            raise ValueError("privacy gate record JSON is invalid") from None
        return cls.from_dict(payload)


def aggregate_privacy_gates(
    gates: Iterable[PrivacyGateResult | Mapping[str, Any]],
) -> PrivacyReleaseGateRecord:
    """Aggregate typed privacy gate results into one deterministic record."""

    if isinstance(gates, (str, bytes, bytearray)):
        raise TypeError("gates must be an iterable of gate results")
    try:
        results = tuple(_coerce_gate_result(gate) for gate in gates)
    except TypeError:
        raise TypeError("gates must be an iterable of gate results") from None
    return PrivacyReleaseGateRecord(results)


def render_privacy_release_gate(
    gates: PrivacyReleaseGateRecord | Iterable[PrivacyGateResult | Mapping[str, Any]],
    *,
    indent: int | None = 2,
) -> str:
    """Render a record or gate results as stable counts-only JSON."""

    record = (
        gates
        if isinstance(gates, PrivacyReleaseGateRecord)
        else aggregate_privacy_gates(gates)
    )
    return record.to_json(indent=indent)


class PrivacyGateAggregator:
    """Stateless object facade for dependency-injected release workflows."""

    @staticmethod
    def aggregate(
        gates: Iterable[PrivacyGateResult | Mapping[str, Any]],
    ) -> PrivacyReleaseGateRecord:
        """Aggregate gate results without network or external state."""

        return aggregate_privacy_gates(gates)


def _coerce_gate_result(value: Any) -> PrivacyGateResult:
    if isinstance(value, PrivacyGateResult):
        return value
    if isinstance(value, Mapping):
        return PrivacyGateResult.from_mapping(value)
    raise TypeError("gate results must be PrivacyGateResult values or mappings")


def _coerce_status(value: Any) -> GateStatus:
    if isinstance(value, GateStatus):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower().replace("_", "-")
        for status in GateStatus:
            if normalized == status.value:
                return status
    raise ValueError("gate status must be blocking, warning, or waived")


def _safe_identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a stable identifier")
    return value


def _coerce_count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("finding_count must be a non-negative integer")
    return value


def _count_findings(value: Any) -> int:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sized
    ):
        raise TypeError("findings must be a non-text sized collection")
    try:
        return _coerce_count(len(value))
    except Exception:
        raise TypeError("findings must be a countable collection") from None


def _require_digest(value: Any, *, field_name: str) -> None:
    if not isinstance(value, str) or not _DIGEST_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a sha256 digest")


def _first_present(value: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


# Descriptive aliases keep the public surface discoverable for callers that
# use "state" or "gate result" terminology.
GateState = GateStatus
PrivacyGateState = GateStatus
GateResult = PrivacyGateResult
PrivacyGateDecisionRecord = PrivacyReleaseGateRecord
aggregate_privacy_gate_results = aggregate_privacy_gates
build_privacy_release_gate = aggregate_privacy_gates


__all__ = [
    "GateResult",
    "GateState",
    "GateStatus",
    "PRIVACY_GATE_REPORT_TYPE",
    "PRIVACY_GATE_SCHEMA_VERSION",
    "PrivacyGateAggregator",
    "PrivacyGateDecisionRecord",
    "PrivacyGateResult",
    "PrivacyGateState",
    "PrivacyReleaseGateRecord",
    "ReleaseDecision",
    "aggregate_privacy_gate_results",
    "aggregate_privacy_gates",
    "build_privacy_release_gate",
    "render_privacy_release_gate",
]
