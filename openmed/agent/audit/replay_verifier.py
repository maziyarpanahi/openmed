"""Offline, metadata-only verification of frozen clinical agent run evidence.

The verifier never invokes a model or tool. Clinical bytes remain in caller-owned
memory and are reduced to digests before any comparison or report is built.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from typing import Any, Final

from openmed.agent.audit.action_ledger import (
    ActionEntry,
    ActionState,
    verify_action_ledger,
)
from openmed.agent.correlation import ActionId, RunId
from openmed.agent.permissions.grants import (
    CapabilityGrantManifest,
    CapabilityGrantRequest,
    CapabilityGrantVerifier,
)

REPLAY_SCHEMA_VERSION: Final = "openmed.agent.deterministic_replay.v1"
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_SIGNATURE_RE = re.compile(r"hmac-sha256:[0-9a-f]{64}")
_CONTENT_FIELDS = ("policy_digest", "response_digest", "model_digest")
_FIELDS = (
    "grant_digest",
    "tool_digest",
    "policy_digest",
    "response_digest",
    "model_digest",
    "artifact_digest",
)


class ReplayError(ValueError):
    """Value-free refusal to replay incomplete or unauthenticated evidence."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _commit(key: bytes, domain: str, value: bytes) -> str:
    return (
        "hmac-sha256:"
        + hmac.new(
            key, domain.encode("ascii") + b"\x00" + value, hashlib.sha256
        ).hexdigest()
    )


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _valid_digest(value: object) -> bool:
    return type(value) is str and _DIGEST_RE.fullmatch(value) is not None


def _valid_key(value: object) -> bool:
    return type(value) is bytes and len(value) >= 32


@dataclass(frozen=True, slots=True, repr=False)
class ReplayStep:
    """Signed commitment to one planned action and its frozen inputs."""

    action_id: ActionId
    captured_at: int
    grant_digest: str
    tool_digest: str
    policy_digest: str
    response_digest: str
    model_digest: str
    artifact_digest: str

    def __post_init__(self) -> None:
        if type(self.action_id) is not ActionId:
            raise ReplayError("invalid_action_id")
        if type(self.captured_at) is not int or not 0 <= self.captured_at < 2**63:
            raise ReplayError("invalid_capture_time")
        if any(
            not _valid_digest(getattr(self, name))
            for name in ("grant_digest", "tool_digest", "artifact_digest")
        ) or any(
            type(getattr(self, name)) is not str
            or _SIGNATURE_RE.fullmatch(getattr(self, name)) is None
            for name in _CONTENT_FIELDS
        ):
            raise ReplayError("invalid_step_digest")

    def to_dict(self) -> dict[str, str | int]:
        """Return a metadata-only canonical step commitment."""
        return {
            "action_id": self.action_id.serialize(),
            "captured_at": self.captured_at,
            **{name: getattr(self, name) for name in _FIELDS},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReplayStep:
        """Restore only the exact signed step fields."""
        if type(payload) is not dict or set(payload) != {
            "action_id",
            "captured_at",
            *_FIELDS,
        }:
            raise ReplayError("invalid_step_fields")
        try:
            return cls(
                action_id=ActionId.parse(payload["action_id"]),
                captured_at=payload["captured_at"],
                **{name: payload[name] for name in _FIELDS},
            )
        except ValueError:
            raise ReplayError("invalid_step_fields") from None


@dataclass(frozen=True, slots=True, repr=False)
class FrozenReplayEvidence:
    """Caller-held inputs for one step; byte fields are never serialized."""

    grant: CapabilityGrantManifest
    request: CapabilityGrantRequest
    tool_contract: bytes = field(repr=False)
    policy_snapshot: bytes = field(repr=False)
    tool_response: bytes = field(repr=False)
    model_configuration: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.grant) is not CapabilityGrantManifest
            or type(self.request) is not CapabilityGrantRequest
        ):
            raise ReplayError("invalid_frozen_evidence")
        if any(
            type(getattr(self, name)) is not bytes
            for name in (
                "tool_contract",
                "policy_snapshot",
                "tool_response",
                "model_configuration",
            )
        ):
            raise ReplayError("incomplete_frozen_evidence")
        if (
            not self.tool_contract
            or not self.policy_snapshot
            or not self.model_configuration
        ):
            raise ReplayError("incomplete_frozen_evidence")


def _artifact_digest(
    action_id: ActionId, captured_at: int, digests: dict[str, str], previous: str | None
) -> str:
    return _digest(
        _canonical(
            {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "action_id": action_id.serialize(),
                "captured_at": captured_at,
                "previous_artifact_digest": previous,
                **digests,
            }
        )
    )


def capture_replay_step(
    action_id: ActionId,
    captured_at: int,
    evidence: FrozenReplayEvidence,
    *,
    commitment_key: bytes,
    previous_artifact_digest: str | None = None,
) -> ReplayStep:
    """Commit frozen inputs to a deterministic, chained semantic artifact hash.

    Call this when recording a run, then sign the complete manifest. The bytes
    remain with the caller and are never written by this module.
    """
    if type(action_id) is not ActionId or type(evidence) is not FrozenReplayEvidence:
        raise ReplayError("invalid_frozen_evidence")
    if not _valid_key(commitment_key):
        raise ReplayError("invalid_commitment_key")
    if type(captured_at) is not int or not 0 <= captured_at < 2**63:
        raise ReplayError("invalid_capture_time")
    if previous_artifact_digest is not None and not _valid_digest(
        previous_artifact_digest
    ):
        raise ReplayError("invalid_previous_artifact")
    digests = {
        "grant_digest": _digest(evidence.grant.to_json().encode("ascii")),
        "tool_digest": _digest(evidence.tool_contract),
        "policy_digest": _commit(commitment_key, "policy", evidence.policy_snapshot),
        "response_digest": _commit(commitment_key, "response", evidence.tool_response),
        "model_digest": _commit(commitment_key, "model", evidence.model_configuration),
    }
    return ReplayStep(
        action_id=action_id,
        captured_at=captured_at,
        **digests,
        artifact_digest=_artifact_digest(
            action_id, captured_at, digests, previous_artifact_digest
        ),
    )


@dataclass(frozen=True, slots=True, repr=False)
class SignedReplayManifest:
    """Authenticated plan, ledger anchor, and expected semantic hashes."""

    run_id: RunId
    ledger_head_digest: str
    steps: tuple[ReplayStep, ...]
    signature: str
    schema_version: str = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ReplayError("unsupported_schema_version")
        if type(self.run_id) is not RunId or not _valid_digest(self.ledger_head_digest):
            raise ReplayError("invalid_manifest_anchor")
        if (
            type(self.steps) is not tuple
            or not 0 < len(self.steps) <= 1024
            or any(type(step) is not ReplayStep for step in self.steps)
            or len({step.action_id for step in self.steps}) != len(self.steps)
        ):
            raise ReplayError("invalid_manifest_steps")
        if (
            type(self.signature) is not str
            or _SIGNATURE_RE.fullmatch(self.signature) is None
        ):
            raise ReplayError("invalid_manifest_signature")

    def signing_payload(self) -> dict[str, Any]:
        """Return every authenticated field except the signature."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id.serialize(),
            "ledger_head_digest": self.ledger_head_digest,
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return signed metadata without frozen clinical inputs."""
        return {**self.signing_payload(), "signature": self.signature}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SignedReplayManifest:
        """Restore an exact signed manifest for offline verification."""
        if type(payload) is not dict or set(payload) != {
            "schema_version",
            "run_id",
            "ledger_head_digest",
            "steps",
            "signature",
        }:
            raise ReplayError("invalid_manifest_fields")
        if type(payload["steps"]) is not list:
            raise ReplayError("invalid_manifest_steps")
        try:
            return cls(
                run_id=RunId.parse(payload["run_id"]),
                ledger_head_digest=payload["ledger_head_digest"],
                steps=tuple(ReplayStep.from_dict(step) for step in payload["steps"]),
                signature=payload["signature"],
                schema_version=payload["schema_version"],
            )
        except ValueError:
            raise ReplayError("invalid_manifest_fields") from None

    @classmethod
    def sign(
        cls,
        *,
        run_id: RunId,
        ledger_head_digest: str,
        steps: tuple[ReplayStep, ...],
        key: bytes,
    ) -> SignedReplayManifest:
        """Sign an already captured, metadata-only plan with a local key."""
        if not _valid_key(key):
            raise ReplayError("invalid_signing_key")
        payload = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "run_id": run_id.serialize() if type(run_id) is RunId else None,
            "ledger_head_digest": ledger_head_digest,
            "steps": [step.to_dict() for step in steps]
            if type(steps) is tuple and all(type(s) is ReplayStep for s in steps)
            else None,
        }
        signature = (
            "hmac-sha256:"
            + hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()
        )
        return cls(run_id, ledger_head_digest, steps, signature)


@dataclass(frozen=True, slots=True, repr=False)
class ReplayReport:
    """First semantic divergence, expressed only as a step and hashes."""

    matched: bool
    step: int | None
    artifact: str | None
    expected_digest: str | None
    actual_digest: str | None

    def to_dict(self) -> dict[str, bool | int | str | None]:
        """Return a value-free report suitable for audit storage."""
        return {
            "matched": self.matched,
            "step": self.step,
            "artifact": self.artifact,
            "expected_digest": self.expected_digest,
            "actual_digest": self.actual_digest,
        }


def verify_replay(
    manifest: SignedReplayManifest,
    ledger: tuple[ActionEntry, ...],
    evidence: tuple[FrozenReplayEvidence, ...],
    *,
    signing_key: bytes,
    grant_verifier: CapabilityGrantVerifier,
) -> ReplayReport:
    """Replay frozen hashes without tool, model, clock, disk, or network access.

    Refuse incomplete or unauthenticated evidence before comparing any step.
    The first differing field in ledger order is returned as a digest pair.
    """
    if type(manifest) is not SignedReplayManifest or not _valid_key(signing_key):
        raise ReplayError("invalid_manifest")
    manifest.__post_init__()
    expected_signature = (
        "hmac-sha256:"
        + hmac.new(
            signing_key, _canonical(manifest.signing_payload()), hashlib.sha256
        ).hexdigest()
    )
    if not hmac.compare_digest(expected_signature, manifest.signature):
        raise ReplayError("manifest_signature_mismatch")
    if type(ledger) is not tuple or not ledger:
        raise ReplayError("incomplete_ledger")
    verify_action_ledger(ledger)
    proposals = tuple(entry for entry in ledger if entry.state is ActionState.PROPOSED)
    if (
        ledger[0].run_id != manifest.run_id
        or ledger[-1].entry_digest != manifest.ledger_head_digest
        or len(proposals) != len(manifest.steps)
        or any(
            proposal.action_id != step.action_id
            for proposal, step in zip(proposals, manifest.steps)
        )
    ):
        raise ReplayError("manifest_ledger_mismatch")
    if (
        type(evidence) is not tuple
        or len(evidence) != len(manifest.steps)
        or any(type(item) is not FrozenReplayEvidence for item in evidence)
    ):
        raise ReplayError("incomplete_frozen_evidence")
    if any(
        step.grant_digest != proposal.grant_digest
        or step.tool_digest != proposal.tool_digest
        for step, proposal in zip(manifest.steps, proposals)
    ):
        raise ReplayError("manifest_ledger_mismatch")
    # Authenticate every grant before returning even an early divergence.
    for step, item in zip(manifest.steps, evidence):
        grant_verifier.verify(item.grant, item.request, now=step.captured_at)
    previous: str | None = None
    for index, (step, item) in enumerate(zip(manifest.steps, evidence)):
        actual = capture_replay_step(
            step.action_id,
            step.captured_at,
            item,
            commitment_key=signing_key,
            previous_artifact_digest=previous,
        )
        for name in _FIELDS:
            expected_digest = getattr(step, name)
            actual_digest = getattr(actual, name)
            if expected_digest != actual_digest:
                return ReplayReport(False, index, name, expected_digest, actual_digest)
        previous = actual.artifact_digest
    return ReplayReport(True, None, None, None, None)
