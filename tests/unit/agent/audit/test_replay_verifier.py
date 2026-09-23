"""Synthetic offline checks for deterministic, side-effect-free replay."""

from dataclasses import replace

import pytest

from openmed.agent.audit.action_ledger import (
    ActionEntry,
    ActionState,
)
from openmed.agent.audit.replay_verifier import (
    FrozenReplayEvidence,
    ReplayError,
    SignedReplayManifest,
    capture_replay_step,
    verify_replay,
)
from openmed.agent.correlation import ActionId, RunId
from openmed.agent.permissions.grants import (
    CapabilityGrantConstraint,
    CapabilityGrantRequest,
    CapabilityGrantSignatureError,
    CapabilityGrantSigner,
    CapabilityGrantVerifier,
)

KEY = b"synthetic-local-test-signing-key-32-bytes"
RUN = RunId("run_" + "1" * 32)
ACTIONS = (ActionId("act_" + "2" * 32), ActionId("act_" + "3" * 32))
CONSTRAINT = CapabilityGrantConstraint(
    tool="tool:org.example/redact@1.0.0",
    resource="resource:org.example/document@1.0.0",
    action="action:org.example/read@1.0.0",
    policy_profile="policy:org.example/local@1.0.0",
)


def fixture():
    grant = CapabilityGrantSigner(KEY).issue([CONSTRAINT], expires_at=2000)
    request = CapabilityGrantRequest(**CONSTRAINT.to_dict())
    frozen = (
        FrozenReplayEvidence(
            grant, request, b"tool-v1", b"policy-v1", b"response-a", b"model-v1"
        ),
        FrozenReplayEvidence(
            grant, request, b"tool-v1", b"policy-v1", b"response-b", b"model-v1"
        ),
    )
    steps = []
    ledger = []
    previous_artifact = None
    for index, (action, item) in enumerate(zip(ACTIONS, frozen)):
        step = capture_replay_step(
            action,
            1000 + index,
            item,
            commitment_key=KEY,
            previous_artifact_digest=previous_artifact,
        )
        steps.append(step)
        previous_artifact = step.artifact_digest
        ledger.append(
            ActionEntry.create(
                run_id=RUN,
                action_id=action,
                sequence=index,
                state=ActionState.PROPOSED,
                actor_role="role:openmed.local/operator",
                grant_digest=step.grant_digest,
                tool_digest=step.tool_digest,
                resource_refs=(),
                previous_digest=ledger[-1].entry_digest if ledger else None,
            )
        )
    manifest = SignedReplayManifest.sign(
        run_id=RUN,
        ledger_head_digest=ledger[-1].entry_digest,
        steps=tuple(steps),
        key=KEY,
    )
    return manifest, tuple(ledger), frozen


def replay(manifest, ledger, frozen):
    return verify_replay(
        manifest,
        ledger,
        frozen,
        signing_key=KEY,
        grant_verifier=CapabilityGrantVerifier(KEY),
    )


def test_replay_is_deterministic_and_read_only():
    manifest, ledger, frozen = fixture()
    assert SignedReplayManifest.from_dict(manifest.to_dict()) == manifest
    assert replay(manifest, ledger, frozen).to_dict() == {
        "matched": True,
        "step": None,
        "artifact": None,
        "expected_digest": None,
        "actual_digest": None,
    }
    assert replay(manifest, ledger, frozen) == replay(manifest, ledger, frozen)
    assert ledger == fixture()[1]


@pytest.mark.parametrize(
    "field,expected_artifact",
    [
        ("policy_snapshot", "policy_digest"),
        ("tool_response", "response_digest"),
        ("model_configuration", "model_digest"),
        ("tool_contract", "tool_digest"),
    ],
)
def test_first_semantic_divergence_is_hash_only(field, expected_artifact):
    manifest, ledger, frozen = fixture()
    changed = replace(frozen[0], **{field: b"PRIVATE_VALUE"})
    report = replay(manifest, ledger, (changed, frozen[1]))
    assert not report.matched
    assert report.step == 0
    assert report.artifact == expected_artifact
    assert report.expected_digest != report.actual_digest
    assert "PRIVATE_VALUE" not in repr(report.to_dict())


def test_later_divergence_reports_later_step():
    manifest, ledger, frozen = fixture()
    report = replay(
        manifest,
        ledger,
        (frozen[0], replace(frozen[1], tool_response=b"changed")),
    )
    assert (report.step, report.artifact) == (1, "response_digest")


def test_missing_frozen_evidence_and_ledger_are_refused():
    manifest, ledger, frozen = fixture()
    with pytest.raises(ReplayError, match="incomplete_frozen_evidence"):
        replay(manifest, ledger, frozen[:1])
    with pytest.raises(ReplayError, match="manifest_ledger_mismatch"):
        replay(manifest, ledger[:1], frozen)
    with pytest.raises(ReplayError, match="incomplete_frozen_evidence"):
        replace(frozen[0], policy_snapshot=b"")


def test_malformed_manifest_fields_never_echo_values():
    manifest, _, _ = fixture()
    payload = manifest.to_dict()
    payload["steps"][0]["action_id"] = "PRIVATE_VALUE"
    with pytest.raises(ReplayError) as caught:
        SignedReplayManifest.from_dict(payload)
    assert "PRIVATE_VALUE" not in str(caught.value)


def test_manifest_tampering_and_wrong_key_are_refused():
    manifest, ledger, frozen = fixture()
    altered = replace(
        manifest,
        steps=(replace(manifest.steps[0], captured_at=1002), manifest.steps[1]),
    )
    with pytest.raises(ReplayError, match="manifest_signature_mismatch"):
        replay(altered, ledger, frozen)
    with pytest.raises(ReplayError, match="manifest_signature_mismatch"):
        verify_replay(
            manifest,
            ledger,
            frozen,
            signing_key=b"other-local-test-signing-key-32-bytes",
            grant_verifier=CapabilityGrantVerifier(KEY),
        )


def test_signed_artifact_mismatch_is_reported_at_its_step():
    manifest, ledger, frozen = fixture()
    steps = (
        replace(manifest.steps[0], artifact_digest="sha256:" + "f" * 64),
        manifest.steps[1],
    )
    altered = SignedReplayManifest.sign(
        run_id=RUN,
        ledger_head_digest=manifest.ledger_head_digest,
        steps=steps,
        key=KEY,
    )
    report = replay(altered, ledger, frozen)
    assert (report.step, report.artifact) == (0, "artifact_digest")


def test_invalid_grant_is_refused_even_when_response_differs():
    manifest, ledger, frozen = fixture()
    other = CapabilityGrantSigner(b"other-local-test-signing-key-32-bytes").issue(
        [CONSTRAINT], expires_at=2000
    )
    with pytest.raises(CapabilityGrantSignatureError) as caught:
        replay(
            manifest,
            ledger,
            (
                replace(frozen[0], grant=other, tool_response=b"PRIVATE_VALUE"),
                frozen[1],
            ),
        )
    assert "PRIVATE_VALUE" not in str(caught.value)
