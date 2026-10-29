"""Deterministic, wholly synthetic adapters for the governed workflow proof."""

from __future__ import annotations

from collections import Counter
from typing import Any

from openmed.agent.correlation import ActionId, RunId
from openmed.agent.identifiers import ToolId, WorkflowId
from openmed.agent.permissions.grants import CapabilityGrantConstraint
from openmed.agent.workflows.recovery import (
    CompensationLimit,
    EffectKind,
    EffectObservation,
    EffectRecord,
    ObservationState,
)

RUN_ID = RunId("run_" + "a" * 32)
ACTION_ID = ActionId("act_" + "b" * 32)
WORKFLOW_ID = WorkflowId("workflow:org.openmed/governed-reference@1.0.0")
REVIEWER_ROLE = "role:org.openmed/clinical-reviewer@1.0.0"
ACTOR_ROLE = "role:openmed.local/operator"
NOW = 1_800_000_000
EXPIRES_AT = NOW + 300
KEY = b"synthetic-governed-workflow-key-32-bytes"
PRIVATE_MARKER = "synthetic-private-clinical-value"
PURPOSE = "purpose:org.openmed/governed-review@1.0.0"
DATA_CLASS = "data:org.openmed/clinical-status@1.0.0"
GRANT_CONSTRAINT = CapabilityGrantConstraint(
    tool="tool:org.openmed/governed-dispatch@1.0.0",
    resource="resource:org.openmed/synthetic-observation@1.0.0",
    action="action:org.openmed/reviewed-write@1.0.0",
    policy_profile="policy:org.openmed/minimum-data@1.0.0",
)


def reviewed_tool_schema() -> dict[str, Any]:
    """Return a minimum-data schema with no payload-derived field names."""

    return {
        "type": "object",
        "x-openmed-purpose": PURPOSE,
        "properties": {
            "status": {
                "type": "string",
                "x-openmed-purpose": PURPOSE,
                "x-openmed-minimum-data": "required",
                "x-openmed-data-class": DATA_CLASS,
            }
        },
        "required": ["status"],
        "additionalProperties": False,
    }


def synthetic_effects() -> tuple[EffectRecord, ...]:
    """Create stable local, FHIR, and OMOP effect identities."""

    return tuple(
        EffectRecord.create(
            ordinal=index,
            run_id=RUN_ID,
            action_id=ActionId(f"act_{index + 1:032x}"),
            tool_id=ToolId(
                f"tool:org.openmed/governed-{kind.value.replace('_', '-')}@1.0.0"
            ),
            kind=kind,
            operation_digest=f"sha256:{index + 1:064x}",
            approval_required=True,
            compensation_limit=(
                CompensationLimit.NONE
                if kind is EffectKind.LOCAL_TOOL
                else CompensationLimit.PROPOSE_ONLY
            ),
        )
        for index, kind in enumerate(EffectKind)
    )


class SyntheticEffectSink:
    """Idempotent stand-in that persists digests, never clinical payloads."""

    def __init__(self) -> None:
        self.commits: dict[str, str] = {}
        self.commit_counts: Counter[str] = Counter()
        self.omop_key: str | None = None

    def apply(self, key: str, evidence_digest: str) -> None:
        """Commit one effect once for a stable cross-adapter key."""

        if key not in self.commits:
            self.commits[key] = evidence_digest
            self.commit_counts[key] += 1

    def observe(
        self, effects: tuple[EffectRecord, ...]
    ) -> tuple[EffectObservation, ...]:
        """Return content-free commit observations for recovery."""

        return tuple(
            EffectObservation(
                action_id=effect.action_id,
                operation_digest=effect.operation_digest,
                idempotency_key=effect.idempotency_key,
                state=(
                    ObservationState.COMMITTED
                    if effect.idempotency_key in self.commits
                    else ObservationState.ABSENT
                ),
                commit_evidence_digest=self.commits.get(effect.idempotency_key),
            )
            for effect in effects
        )

    def commit_batch(
        self, mutations: tuple[Any, ...], *, batch_digest: str, approval: Any
    ) -> None:
        """Record an approved OMOP batch without keeping row values."""

        if self.omop_key is None or not mutations or approval is None:
            raise ValueError("omop_commit_not_prepared")
        self.apply(self.omop_key, batch_digest)
