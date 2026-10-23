"""Offline private-learning self-check over bounded synthetic metadata contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import islice

from jsonschema import Draft202012Validator
from referencing import Registry

from openmed.training.federated_metrics import (
    FEDERATED_METRIC_SCHEMA_VERSION,
    FederatedMetricEnvelope,
    FederatedMetricKind,
    FederatedParticipantCountBand,
    FederatedPrivacyMechanism,
    build_federated_metric_envelope,
)
from openmed.training.federated_round import (
    FEDERATED_ROUND_SCHEMA_VERSION,
    FederatedRoundLifecycle,
    FederatedRoundState,
)
from openmed.training.federated_schedule import (
    FEDERATED_SCHEDULE_SCHEMA_VERSION,
    FederatedRoundSchedule,
    FederatedSchedulePhase,
)
from openmed.training.federated_schema_fingerprint import (
    fingerprint_update_schema,
    same_update_schema,
)
from openmed.training.federated_status import (
    FEDERATED_ROUND_STATUS_SCHEMA_VERSION,
    build_federated_round_status,
)
from openmed.training.federated_update_metadata import (
    FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)
from openmed.training.private_training_schemas import (
    PRIVATE_TRAINING_SCHEMA_DIALECT,
    build_private_training_schemas,
)

PRIVATE_LEARNING_CHECK_VERSION = "1.0.0"
_CONTRACT_VERSIONS = {
    "lifecycle": FEDERATED_ROUND_SCHEMA_VERSION.rsplit(".", 1)[-1],
    "schedule": FEDERATED_SCHEDULE_SCHEMA_VERSION.rsplit(".", 1)[-1],
    "status": FEDERATED_ROUND_STATUS_SCHEMA_VERSION.rsplit(".", 1)[-1],
    "update": FEDERATED_UPDATE_METADATA_SCHEMA_VERSION.rsplit(".", 1)[-1],
    "aggregate": FEDERATED_METRIC_SCHEMA_VERSION.rsplit(".", 1)[-1],
}
_START = datetime(2000, 1, 1, tzinfo=timezone.utc)
_MODEL = "sha256:" + "a" * 64
_UPDATE = "sha256:" + "b" * 64
_NAMES = (
    "schemas",
    "schedule",
    "lifecycle",
    "update",
    "suppression",
    "forbidden_fields",
    "fingerprint",
)


def _schedule() -> FederatedRoundSchedule:
    return FederatedRoundSchedule(
        _START,
        _START + timedelta(hours=1),
        _START + timedelta(hours=2),
        _START + timedelta(hours=3),
        _START + timedelta(hours=4),
    )


def _update(
    *, shape: tuple[int, ...] = (2, 3), digest: str = _UPDATE
) -> tuple[FederatedUpdateMetadata, FederatedUpdatePolicy]:
    parameter = FederatedParameterMetadata("adapter.lora_A.weight", shape, "float32")
    policy = FederatedUpdatePolicy(_MODEL, (parameter,), max_total_elements=6)
    metadata = FederatedUpdateMetadata(
        model_digest=_MODEL,
        adapter_format="dense",
        parameters=(parameter,),
        total_elements=6,
        update_digest=digest,
        clipped=True,
        policy=policy,
    )
    return metadata, policy


def _metric(participants: int) -> FederatedMetricEnvelope:
    return build_federated_metric_envelope(
        metric_id="synthetic.loss",
        metric_kind=FederatedMetricKind.BOUNDED_MEAN,
        aggregate_value=0.25,
        clipping_lower_bound=0.0,
        clipping_upper_bound=1.0,
        privacy_mechanism=FederatedPrivacyMechanism.LAPLACE,
        privacy_mechanism_version="v1",
        participant_count=participants,
    )


def _reject_remote(_uri: str):
    raise ValueError("remote schema resolution is disabled")


def _schema_case() -> bool:
    catalog = build_private_training_schemas()
    payloads = {
        "federated_round_lifecycle": FederatedRoundLifecycle().to_dict(),
        "federated_round_schedule": _schedule().to_dict(),
        "federated_round_status": build_federated_round_status(
            state=FederatedRoundState.COLLECTING,
            participant_count=10,
            completed_participant_count=6,
            required_quorum=5,
        ).to_dict(),
        "federated_update_metadata": _update()[0].to_dict(),
        "federated_aggregate_metric": _metric(100).to_dict(),
    }
    if set(catalog) != set(payloads):
        return False
    for name in sorted(payloads):
        schema = catalog[name]
        if schema.get("$schema") != PRIVATE_TRAINING_SCHEMA_DIALECT:
            return False
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(
            schema,
            registry=Registry(retrieve=_reject_remote),
            format_checker=Draft202012Validator.FORMAT_CHECKER,
        )
        if not validator.is_valid(payloads[name]):
            return False
        invalid = {**payloads[name], "forbidden_synthetic_field": True}
        if validator.is_valid(invalid):
            return False
    return True


def _schedule_case() -> bool:
    schedule = _schedule()
    if FederatedRoundSchedule.from_json(schedule.to_json()) != schedule:
        return False
    if schedule.active_phase_at(_START) is not FederatedSchedulePhase.ENROLLMENT:
        return False
    if (
        schedule.active_phase_at(_START + timedelta(hours=1))
        is not FederatedSchedulePhase.UPDATE_SUBMISSION
    ):
        return False
    if schedule.active_phase_at(_START + timedelta(hours=4)) is not None:
        return False
    bad = schedule.to_dict()
    bad["boundaries"]["finishes_at"] = bad["boundaries"]["enrollment_starts_at"]
    return _rejects(lambda: FederatedRoundSchedule.from_dict(bad))


def _lifecycle_case() -> bool:
    lifecycle = FederatedRoundLifecycle()
    updated = lifecycle.transition_to(FederatedRoundState.PREFLIGHT)
    if (
        lifecycle.state is not FederatedRoundState.PLANNED
        or updated.state is not FederatedRoundState.PREFLIGHT
    ):
        return False
    if FederatedRoundLifecycle.from_json(updated.to_json()) != updated:
        return False
    if not _rejects(lambda: lifecycle.transition_to(FederatedRoundState.PROMOTED)):
        return False
    terminal = FederatedRoundLifecycle(FederatedRoundState.ABORTED)
    return terminal.is_terminal and not terminal.can_transition_to(
        FederatedRoundState.COLLECTING
    )


def _update_case() -> bool:
    metadata, policy = _update()
    if FederatedUpdateMetadata.from_dict(metadata.to_dict(), policy=policy) != metadata:
        return False
    wrong_count = {**metadata.to_dict(), "total_elements": 5}
    unclipped = {**metadata.to_dict(), "clipped": False}
    return _rejects(
        lambda: FederatedUpdateMetadata.from_dict(wrong_count, policy=policy)
    ) and _rejects(lambda: FederatedUpdateMetadata.from_dict(unclipped, policy=policy))


def _suppression_case() -> bool:
    suppressed = _metric(1)
    released = _metric(100)
    return (
        suppressed.aggregate_value is None
        and suppressed.participant_count_band
        is FederatedParticipantCountBand.SUPPRESSED
        and released.aggregate_value == 0.25
        and FederatedMetricEnvelope.from_dict(suppressed.to_dict()) == suppressed
    )


def _forbidden_fields_case() -> bool:
    metadata, policy = _update()
    return all(
        (
            _rejects(
                lambda: FederatedRoundLifecycle.from_dict(
                    {**FederatedRoundLifecycle().to_dict(), "patient_count": 99}
                )
            ),
            _rejects(
                lambda: FederatedUpdateMetadata.from_dict(
                    {**metadata.to_dict(), "gradients": [0]}, policy=policy
                )
            ),
            _rejects(
                lambda: FederatedMetricEnvelope.from_dict(
                    {**_metric(100).to_dict(), "endpoint": "synthetic_forbidden"}
                )
            ),
        )
    )


def _fingerprint_case() -> bool:
    first, _policy = _update()
    content_changed, _policy = _update(digest="sha256:" + "c" * 64)
    shape_changed, _policy = _update(shape=(3, 2))
    fingerprint = fingerprint_update_schema(first)
    return (
        # Independently recorded v1 golden identity for the fixed synthetic _update().
        fingerprint
        == "sha256:db17cbd61d6189843f75f975c6c780849d5a9a2e769ef9a707c5c4349d2d22d1"
        and same_update_schema(first, content_changed)
        and not same_update_schema(first, shape_changed)
    )


def _rejects(action: Callable[[], object]) -> bool:
    try:
        action()
    except (ValueError, TypeError):
        return True
    return False


@dataclass(frozen=True, slots=True)
class PrivateLearningCheckResult:
    """One closed-category result without exception text or source payloads."""

    name: str
    passed: bool
    case_count: int
    digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or self.name not in _NAMES
            or type(self.passed) is not bool
            or type(self.case_count) is not int
            or not 1 <= self.case_count <= 10
            or not isinstance(self.digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.digest) is None
        ):
            raise ValueError("invalid private-learning check result")

    def to_dict(self) -> dict[str, object]:
        """Return controlled status, code, check count and deterministic receipt digest."""
        return {
            "name": self.name,
            "passed": self.passed,
            "fixture_case_count": self.case_count,
            "digest": self.digest,
            "reason_code": "passed" if self.passed else "check_failed",
        }


@dataclass(frozen=True, slots=True)
class PrivateLearningCheckReport:
    """Bounded synthetic self-check report; not release or clinical qualification."""

    checks: tuple[PrivateLearningCheckResult, ...]
    version: str = PRIVATE_LEARNING_CHECK_VERSION

    def __post_init__(self) -> None:
        checks = tuple(islice(self.checks, len(_NAMES) + 1))
        if (
            not isinstance(self.version, str)
            or self.version != PRIVATE_LEARNING_CHECK_VERSION
            or len(checks) != len(_NAMES)
            or any(
                not isinstance(check, PrivateLearningCheckResult) for check in checks
            )
            or tuple(check.name for check in checks) != _NAMES
        ):
            raise ValueError("invalid private-learning check report")
        object.__setattr__(self, "checks", checks)

    @property
    def passed(self) -> bool:
        """Require every mandatory independent synthetic contract check to pass."""
        return len(self.checks) == len(_NAMES) and all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, object]:
        """Export versions, statuses and synthetic check counts without source values."""
        return {
            "version": self.version,
            "contract_versions": dict(_CONTRACT_VERSIONS),
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self) -> str:
        """Render byte-stable JSON without clocks, IDs or input-dependent diagnostics."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"

    def to_text(self) -> str:
        """Render closed check categories and statuses for an offline operator."""
        return (
            "Private-learning self-check "
            + self.version
            + "\n"
            + "\n".join(
                c.name + ": " + ("passed" if c.passed else "check_failed")
                for c in self.checks
            )
            + "\n"
        )


def run_private_learning_check() -> PrivateLearningCheckReport:
    """Run all metadata-only synthetic cases without models, credentials or network.

    Every check runs independently even if another raises. Reports contain only
    closed statuses, code categories, versions, fixture case counts and receipt
    digests. Runtime exception text and underlying envelopes are never exported.
    """
    definitions = (
        ("schemas", _schema_case, 10),
        ("schedule", _schedule_case, 5),
        ("lifecycle", _lifecycle_case, 4),
        ("update", _update_case, 3),
        ("suppression", _suppression_case, 3),
        ("forbidden_fields", _forbidden_fields_case, 3),
        ("fingerprint", _fingerprint_case, 3),
    )
    results = []
    for name, case, count in definitions:
        try:
            passed = case() is True
        except Exception:
            passed = False
        canonical = json.dumps(
            {
                "name": name,
                "passed": passed,
                "case_count": count,
                "version": PRIVATE_LEARNING_CHECK_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
        results.append(PrivateLearningCheckResult(name, passed, count, digest))
    return PrivateLearningCheckReport(tuple(results))
