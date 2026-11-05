"""Synthetic tests for the local service recovery operator CLI."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import replace

import pytest

from openmed.agent.approvals.tokens import (
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
)
from openmed.cli import main
from openmed.cli.main import build_parser
from openmed.cli.service_recovery import (
    SERVICE_RECOVERY_CLI_SCHEMA_VERSION,
    SERVICE_RECOVERY_EXIT_CODES,
    RecoveryAction,
    RecoveryCLIStatus,
    RecoveryOutcome,
    RecoveryPlan,
    RecoveryPoint,
    RecoveryProposal,
    RecoveryStateSnapshot,
    ServiceRecoveryError,
    ServiceRecoveryServices,
    recovery_approval_digest,
    run_service_recovery_cli,
)

pytestmark = pytest.mark.skipif(
    not hasattr(os, "O_NOFOLLOW"), reason="Protected CLI files require O_NOFOLLOW"
)

STATE = "sha256:" + "b" * 64
NEXT_STATE = "sha256:" + "c" * 64
POINT_DIGEST = "sha256:" + "d" * 64
OTHER_DIGEST = "sha256:" + "e" * 64
POINT_REF = "rp_" + "1" * 32
OTHER_REF = "rp_" + "2" * 32
KEY = b"synthetic-recovery-approval-key-0001"
ROLE = "role:org.example/recovery-operator@1.0.0"
NOW = 1_800_000_000
SENTINEL = "PRIVATE PATIENT https://private.example.test secret-token"


class _State:
    def __init__(self, generation=3, digest=STATE):
        self.snapshot = RecoveryStateSnapshot(generation, digest)
        self.calls = []

    def inspect(self):
        self.calls.append("inspect")
        return self.snapshot


class _Component:
    def __init__(self, state, *, steps=4):
        self.state = state
        self.point = RecoveryPoint(POINT_REF, POINT_DIGEST, True)
        self.steps = steps
        self.hold_code = None
        self.outcome_status = RecoveryCLIStatus.COMPLETED
        self.applied = None
        self.calls = []

    def recovery_points(self, snapshot):
        self.calls.append("recovery_points")
        return (self.point,)

    def propose(self, snapshot, recovery_point_ref):
        self.calls.append("propose")
        if recovery_point_ref != self.point.reference:
            raise ServiceRecoveryError("recovery_point_unknown", 6)
        return RecoveryProposal(self.point, self.steps, self.hold_code)

    def apply(self, plan, receipt):
        self.calls.append("apply")
        current = self.state.snapshot
        if (
            current.active_generation != plan.active_generation
            or current.state_digest != plan.state_digest
        ):
            return RecoveryOutcome(
                plan.plan_digest,
                RecoveryCLIStatus.CONFLICT,
                current.active_generation,
                current.state_digest,
                0,
            )
        status = self.outcome_status
        if status is RecoveryCLIStatus.COMPLETED:
            self.state.snapshot = RecoveryStateSnapshot(
                plan.active_generation + 1, NEXT_STATE
            )
            applied = plan.step_count
        elif status is RecoveryCLIStatus.PARTIAL:
            self.state.snapshot = RecoveryStateSnapshot(
                plan.active_generation + 1, NEXT_STATE
            )
            applied = 1
        else:
            applied = 0
        if self.applied is not None:
            applied = self.applied
        snapshot = self.state.snapshot
        return RecoveryOutcome(
            plan.plan_digest,
            status,
            snapshot.active_generation,
            snapshot.state_digest,
            applied,
        )


class _Harness:
    def __init__(self, tmp_path, *, key=KEY):
        self.tmp_path = tmp_path
        self.state = _State()
        self.component = _Component(self.state)
        self.nonces = InMemoryApprovalNonceStore()
        self.claims = []
        original = self.nonces.claim

        def claim(nonce_digest, *, expires_at, now):
            self.claims.append(nonce_digest)
            return original(nonce_digest, expires_at=expires_at, now=now)

        self.nonces.claim = claim
        self.verifier = ApprovalTokenVerifier(
            key, self.nonces, clock=lambda: NOW, max_lifetime_seconds=3600
        )
        self.services = ServiceRecoveryServices(
            state=self.state,
            components={RecoveryAction.RESTORE_BACKUP: self.component},
            approval_verifier=self.verifier,
            reviewer_role=ROLE,
        )

    def plan(self, **changes):
        snapshot = self.state.snapshot
        values = {
            "action": RecoveryAction.RESTORE_BACKUP,
            "active_generation": snapshot.active_generation,
            "state_digest": snapshot.state_digest,
            "recovery_point_ref": POINT_REF,
            "recovery_point_digest": POINT_DIGEST,
            "step_count": self.component.steps,
        }
        values.update(changes)
        return RecoveryPlan(**values)

    def private(self, name, payload, mode=0o600):
        path = self.tmp_path / name
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload if type(payload) is str else json.dumps(payload))
        os.chmod(path, mode)
        return path

    def token(self, digest, *, expires_at=NOW + 600, key=KEY):
        signer = ApprovalTokenSigner(key, clock=lambda: NOW, max_lifetime_seconds=3600)
        token = signer.issue(
            action_digest=digest, reviewer_role=ROLE, expires_at=expires_at
        )
        return token.to_dict()


def _run(argv, capsys, *, services=None, clock=lambda: NOW):
    code = run_service_recovery_cli(argv, services=services, clock=clock)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert SENTINEL not in captured.out
    result = json.loads(captured.out)
    assert result["schema_version"] == SERVICE_RECOVERY_CLI_SCHEMA_VERSION
    assert result["command"].startswith("service-recovery")
    return code, result


def _files(harness, plan=None, token=None):
    plan = plan or harness.plan()
    plan_path = harness.private("plan.json", plan.to_dict())
    token_path = harness.private(
        "approval.json", token or harness.token(plan.approval_digest)
    )
    return plan, plan_path, token_path


def _apply(harness, capsys, plan_path, token_path, action="restore-backup"):
    argv = ["apply", action, "--plan", str(plan_path), "--approval", str(token_path)]
    return _run(argv, capsys, services=harness.services)


def _listing(path):
    return sorted(entry.name for entry in path.iterdir())


def test_exit_codes_distinguish_every_recovery_outcome():
    codes = {
        status: SERVICE_RECOVERY_EXIT_CODES[status]
        for status in (
            RecoveryCLIStatus.COMPLETED,
            RecoveryCLIStatus.HELD,
            RecoveryCLIStatus.CONFLICT,
            RecoveryCLIStatus.UNAVAILABLE,
            RecoveryCLIStatus.PARTIAL,
        )
    }
    assert len(set(codes.values())) == len(codes)
    assert codes[RecoveryCLIStatus.COMPLETED] == 0
    assert 2 not in set(SERVICE_RECOVERY_EXIT_CODES.values())
    assert set(SERVICE_RECOVERY_EXIT_CODES) == set(RecoveryCLIStatus)


def test_inspect_is_read_only(tmp_path, capsys):
    harness = _Harness(tmp_path)
    before = harness.state.snapshot
    code, result = _run(["inspect"], capsys, services=harness.services)
    assert code == 0
    assert result["ok"] is True
    assert result["status"] == "ready"
    data = result["data"]
    assert data["active_generation"] == 3
    assert data["state_digest"] == STATE
    components = {entry["action"]: entry for entry in data["components"]}
    assert components["restore-backup"]["available"] is True
    assert components["restore-backup"]["recovery_points"] == [
        {"reference": POINT_REF, "digest": POINT_DIGEST, "compatible": True}
    ]
    assert components["rollback-model"]["available"] is False
    assert harness.state.snapshot is before
    assert "apply" not in harness.component.calls
    assert harness.claims == []
    assert _listing(tmp_path) == []


def test_plan_is_read_only_and_binds_generation_and_state(tmp_path, capsys):
    harness = _Harness(tmp_path)
    before = harness.state.snapshot
    argv = [
        "plan",
        "--action",
        "restore-backup",
        "--recovery-point",
        POINT_REF,
    ]
    code, result = _run(argv, capsys, services=harness.services)
    assert code == 0
    assert result["status"] == "ready"
    plan = RecoveryPlan.from_dict(result["data"]["plan"])
    assert plan == harness.plan()
    assert result["data"]["approval_digest"] == plan.approval_digest
    assert plan.approval_digest == recovery_approval_digest(plan)
    assert harness.state.snapshot is before
    assert "apply" not in harness.component.calls
    assert harness.claims == []
    assert _listing(tmp_path) == []
    other_generation = harness.plan(active_generation=4)
    other_state = harness.plan(state_digest=OTHER_DIGEST)
    digests = {
        plan.approval_digest,
        other_generation.approval_digest,
        other_state.approval_digest,
    }
    assert len(digests) == 3


def test_plan_output_is_private_and_never_overwritten(tmp_path, capsys):
    harness = _Harness(tmp_path)
    output = tmp_path / "plan.json"
    argv = [
        "plan",
        "--action",
        "restore-backup",
        "--recovery-point",
        POINT_REF,
        "--output",
        str(output),
    ]
    code, result = _run(argv, capsys, services=harness.services)
    assert code == 0
    assert str(tmp_path) not in json.dumps(result)
    mode = stat.S_IMODE(os.stat(output).st_mode)
    assert mode == 0o600
    written = json.loads(output.read_text(encoding="utf-8"))
    assert RecoveryPlan.from_dict(written) == harness.plan()
    original = output.read_bytes()
    code, result = _run(argv, capsys, services=harness.services)
    assert code == 2
    assert result["error"]["code"] == "output_exists"
    assert output.read_bytes() == original


def test_plan_rejects_incompatible_recovery_point(tmp_path, capsys):
    harness = _Harness(tmp_path)
    harness.component.point = RecoveryPoint(POINT_REF, POINT_DIGEST, False)
    output = tmp_path / "plan.json"
    argv = [
        "plan",
        "--action",
        "restore-backup",
        "--recovery-point",
        POINT_REF,
        "--output",
        str(output),
    ]
    code, result = _run(argv, capsys, services=harness.services)
    assert code == SERVICE_RECOVERY_EXIT_CODES[RecoveryCLIStatus.CONFLICT]
    assert result["status"] == "conflict"
    assert result["error"]["code"] == "recovery_point_incompatible"
    assert not output.exists()


def test_plan_reports_component_hold_without_writing(tmp_path, capsys):
    harness = _Harness(tmp_path)
    harness.component.hold_code = "writes_active"
    output = tmp_path / "plan.json"
    argv = [
        "plan",
        "--action",
        "restore-backup",
        "--recovery-point",
        POINT_REF,
        "--output",
        str(output),
    ]
    code, result = _run(argv, capsys, services=harness.services)
    assert code == SERVICE_RECOVERY_EXIT_CODES[RecoveryCLIStatus.HELD]
    assert result["status"] == "held"
    assert result["data"]["hold_code"] == "writes_active"
    assert not output.exists()


def test_plan_unknown_recovery_point_is_a_conflict(tmp_path, capsys):
    harness = _Harness(tmp_path)
    argv = ["plan", "--action", "restore-backup", "--recovery-point", OTHER_REF]
    code, result = _run(argv, capsys, services=harness.services)
    assert code == 6
    assert result["error"]["code"] == "recovery_point_unknown"


def test_verify_is_read_only_and_does_not_consume_approval(tmp_path, capsys):
    harness = _Harness(tmp_path)
    plan, plan_path, token_path = _files(harness)
    before = harness.state.snapshot
    listing = _listing(tmp_path)
    argv = ["verify", "--plan", str(plan_path), "--approval", str(token_path)]
    code, result = _run(argv, capsys, services=harness.services)
    assert code == 0
    assert result["status"] == "ready"
    assert result["data"]["plan_digest"] == plan.plan_digest
    assert result["data"]["approval_bound"] is True
    assert harness.state.snapshot is before
    assert "apply" not in harness.component.calls
    assert harness.claims == []
    assert _listing(tmp_path) == listing
    code, result = _apply(harness, capsys, plan_path, token_path)
    assert code == 0
    assert result["status"] == "completed"


def test_verify_without_approval_reports_unbound(tmp_path, capsys):
    harness = _Harness(tmp_path)
    _plan, plan_path, _token_path = _files(harness)
    code, result = _run(
        ["verify", "--plan", str(plan_path)], capsys, services=harness.services
    )
    assert code == 0
    assert result["data"]["approval_bound"] is False


def test_verify_detects_changed_state(tmp_path, capsys):
    harness = _Harness(tmp_path)
    _plan, plan_path, _token = _files(harness)
    harness.state.snapshot = RecoveryStateSnapshot(3, NEXT_STATE)
    code, result = _run(
        ["verify", "--plan", str(plan_path)], capsys, services=harness.services
    )
    assert code == 6
    assert result["error"]["code"] == "state_changed"
    assert "apply" not in harness.component.calls


def test_apply_completes_under_exact_approval(tmp_path, capsys):
    harness = _Harness(tmp_path)
    plan, plan_path, token_path = _files(harness)
    code, result = _apply(harness, capsys, plan_path, token_path)
    assert code == SERVICE_RECOVERY_EXIT_CODES[RecoveryCLIStatus.COMPLETED] == 0
    assert result["ok"] is True
    data = result["data"]
    assert data["plan_digest"] == plan.plan_digest
    assert data["previous_generation"] == 3
    assert data["active_generation"] == 4
    assert data["state_digest"] == NEXT_STATE
    assert data["applied_step_count"] == data["step_count"] == 4
    assert data["receipt_digest"].startswith("sha256:")
    assert harness.component.calls.count("apply") == 1
    assert len(harness.claims) == 1
    assert "signature" not in json.dumps(result)


def test_apply_rejects_changed_state_before_application(tmp_path, capsys):
    harness = _Harness(tmp_path)
    _plan, plan_path, token_path = _files(harness)
    harness.state.snapshot = RecoveryStateSnapshot(4, STATE)
    code, result = _apply(harness, capsys, plan_path, token_path)
    assert code == 6
    assert result["error"]["code"] == "state_changed"
    assert "apply" not in harness.component.calls
    assert harness.claims == []


def test_apply_rejects_stale_plan_approval(tmp_path, capsys):
    harness = _Harness(tmp_path)
    stale = harness.plan(active_generation=2)
    plan, plan_path, token_path = _files(
        harness, token=harness.token(stale.approval_digest)
    )
    code, result = _apply(harness, capsys, plan_path, token_path)
    assert code == 6
    assert result["error"]["code"] == "approval_stale"
    assert "apply" not in harness.component.calls
    assert harness.claims == []


def test_apply_rejects_recovery_point_that_became_incompatible(tmp_path, capsys):
    harness = _Harness(tmp_path)
    _plan, plan_path, token_path = _files(harness)
    harness.component.point = RecoveryPoint(POINT_REF, POINT_DIGEST, False)
    code, result = _apply(harness, capsys, plan_path, token_path)
    assert code == 6
    assert result["error"]["code"] == "recovery_point_incompatible"
    assert "apply" not in harness.component.calls
    assert harness.claims == []


def test_apply_rejects_stale_plan_contents(tmp_path, capsys):
    harness = _Harness(tmp_path)
    _plan, plan_path, token_path = _files(harness)
    harness.component.point = RecoveryPoint(POINT_REF, OTHER_DIGEST, True)
    code, result = _apply(harness, capsys, plan_path, token_path)
    assert code == 6
    assert result["error"]["code"] == "plan_stale"
    assert "apply" not in harness.component.calls


def test_apply_rejects_mismatched_action(tmp_path, capsys):
    harness = _Harness(tmp_path)
    _plan, plan_path, token_path = _files(harness)
    code, result = _apply(
        harness, capsys, plan_path, token_path, action="rollback-model"
    )
    assert code == 6
    assert result["error"]["code"] == "plan_action_mismatch"
    assert "apply" not in harness.component.calls


def test_apply_holds_expired_approval(tmp_path, capsys):
    harness = _Harness(tmp_path)
    plan = harness.plan()
    _plan, plan_path, token_path = _files(
        harness, token=_expired_token(plan.approval_digest)
    )
    code, result = _apply(harness, capsys, plan_path, token_path)
    assert code == SERVICE_RECOVERY_EXIT_CODES[RecoveryCLIStatus.HELD]
    assert result["error"]["code"] == "approval_expired"
    assert "apply" not in harness.component.calls


def _expired_token(digest):
    signer = ApprovalTokenSigner(
        KEY, clock=lambda: NOW - 1000, max_lifetime_seconds=3600
    )
    return signer.issue(
        action_digest=digest, reviewer_role=ROLE, expires_at=NOW - 10
    ).to_dict()


def test_apply_denies_unverified_signature(tmp_path, capsys):
    harness = _Harness(tmp_path)
    plan = harness.plan()
    _plan, plan_path, token_path = _files(
        harness,
        token=harness.token(
            plan.approval_digest, key=b"synthetic-other-approval-key-000002"
        ),
    )
    code, result = _apply(harness, capsys, plan_path, token_path)
    assert code == 3
    assert result["error"]["code"] == "approval_denied"
    assert "apply" not in harness.component.calls


def test_apply_rejects_replayed_approval(tmp_path, capsys):
    harness = _Harness(tmp_path)
    _plan, plan_path, token_path = _files(harness)
    harness.component.outcome_status = RecoveryCLIStatus.HELD
    code, result = _apply(harness, capsys, plan_path, token_path)
    assert code == SERVICE_RECOVERY_EXIT_CODES[RecoveryCLIStatus.HELD]
    assert result["status"] == "held"
    code, result = _apply(harness, capsys, plan_path, token_path)
    assert code == 3
    assert result["error"]["code"] == "approval_replayed"
    assert harness.component.calls.count("apply") == 1


def test_apply_reports_partial_recovery_without_retry(tmp_path, capsys):
    harness = _Harness(tmp_path)
    _plan, plan_path, token_path = _files(harness)
    harness.component.outcome_status = RecoveryCLIStatus.PARTIAL
    code, result = _apply(harness, capsys, plan_path, token_path)
    assert code == SERVICE_RECOVERY_EXIT_CODES[RecoveryCLIStatus.PARTIAL]
    assert result["ok"] is False
    assert result["status"] == "partial"
    assert result["data"]["applied_step_count"] == 1
    assert harness.component.calls.count("apply") == 1


@pytest.mark.parametrize(
    ("status", "name"),
    [
        (RecoveryCLIStatus.HELD, "held"),
        (RecoveryCLIStatus.UNAVAILABLE, "unavailable"),
        (RecoveryCLIStatus.FAILED, "failed"),
    ],
)
def test_apply_reports_component_outcomes(tmp_path, capsys, status, name):
    harness = _Harness(tmp_path)
    _plan, plan_path, token_path = _files(harness)
    harness.component.outcome_status = status
    before = harness.state.snapshot
    code, result = _apply(harness, capsys, plan_path, token_path)
    assert code == SERVICE_RECOVERY_EXIT_CODES[status]
    assert result["status"] == name
    assert harness.state.snapshot is before


def test_apply_reports_lost_compare_and_set_as_conflict(tmp_path, capsys):
    harness = _Harness(tmp_path)
    _plan, plan_path, token_path = _files(harness)
    original = harness.component.apply

    def racing_apply(plan, receipt):
        harness.state.snapshot = RecoveryStateSnapshot(9, OTHER_DIGEST)
        return original(plan, receipt)

    harness.component.apply = racing_apply
    code, result = _apply(harness, capsys, plan_path, token_path)
    assert code == SERVICE_RECOVERY_EXIT_CODES[RecoveryCLIStatus.CONFLICT]
    assert result["status"] == "conflict"
    assert harness.state.snapshot.active_generation == 9


def test_apply_refuses_completion_that_did_not_advance_generation(tmp_path, capsys):
    harness = _Harness(tmp_path)
    plan, plan_path, token_path = _files(harness)

    def silent_overwrite(plan, receipt):
        return RecoveryOutcome(
            plan.plan_digest,
            RecoveryCLIStatus.COMPLETED,
            plan.active_generation,
            NEXT_STATE,
            plan.step_count,
        )

    harness.component.apply = silent_overwrite
    code, result = _apply(harness, capsys, plan_path, token_path)
    assert code == 1
    assert result["error"]["code"] == "service_result_invalid"


def test_outcome_rejects_inconsistent_step_counts():
    with pytest.raises(ServiceRecoveryError):
        RecoveryOutcome(OTHER_DIGEST, RecoveryCLIStatus.HELD, 3, STATE, 2)
    with pytest.raises(ServiceRecoveryError):
        RecoveryOutcome(OTHER_DIGEST, RecoveryCLIStatus.PARTIAL, 3, STATE, 0)
    with pytest.raises(ServiceRecoveryError):
        RecoveryOutcome(OTHER_DIGEST, RecoveryCLIStatus.READY, 3, STATE, 0)


def test_missing_services_and_components_are_unavailable(tmp_path, capsys):
    harness = _Harness(tmp_path)
    _plan, plan_path, token_path = _files(harness)
    commands = [
        ["inspect"],
        ["plan", "--action", "restore-backup", "--recovery-point", POINT_REF],
        ["verify", "--plan", str(plan_path)],
        [
            "apply",
            "restore-backup",
            "--plan",
            str(plan_path),
            "--approval",
            str(token_path),
        ],
    ]
    for argv in commands:
        code, result = _run(argv, capsys, services=None)
        assert code == SERVICE_RECOVERY_EXIT_CODES[RecoveryCLIStatus.UNAVAILABLE]
        assert result["status"] == "unavailable"
        assert result["error"]["code"] == "adapter_unavailable"
    rollback = harness.plan(action=RecoveryAction.ROLLBACK_MODEL)
    path = harness.private("rollback.json", rollback.to_dict())
    code, result = _run(
        ["verify", "--plan", str(path)], capsys, services=harness.services
    )
    assert code == 7
    assert result["error"]["code"] == "adapter_unavailable"


def test_apply_without_verifier_is_unavailable(tmp_path, capsys):
    harness = _Harness(tmp_path)
    _plan, plan_path, token_path = _files(harness)
    harness.services = replace(
        harness.services, approval_verifier=None, reviewer_role=None
    )
    code, result = _apply(harness, capsys, plan_path, token_path)
    assert code == 7
    assert "apply" not in harness.component.calls


def test_protected_inputs_are_required(tmp_path, capsys):
    harness = _Harness(tmp_path)
    plan = harness.plan()
    loose = harness.private("loose.json", plan.to_dict(), mode=0o644)
    code, result = _run(
        ["verify", "--plan", str(loose)], capsys, services=harness.services
    )
    assert code == 2
    assert result["error"]["code"] == "input_unprotected"
    target = harness.private("target.json", plan.to_dict())
    link = tmp_path / "link.json"
    link.symlink_to(target)
    code, result = _run(
        ["verify", "--plan", str(link)], capsys, services=harness.services
    )
    assert code == 2
    assert result["error"]["code"] in {"input_unprotected", "input_unreadable"}
    token = harness.private(
        "loose-token.json", harness.token(plan.approval_digest), mode=0o640
    )
    code, result = _apply(harness, capsys, target, token)
    assert code == 2
    assert result["error"]["code"] == "input_unprotected"
    assert "apply" not in harness.component.calls
    assert harness.claims == []


def test_tampered_plan_is_rejected(tmp_path, capsys):
    harness = _Harness(tmp_path)
    document = harness.plan().to_dict()
    document["step_count"] = 1
    path = harness.private("plan.json", document)
    code, result = _run(
        ["verify", "--plan", str(path)], capsys, services=harness.services
    )
    assert code == 2
    assert result["error"]["code"] == "input_invalid"
    document = harness.plan().to_dict()
    document["extra"] = SENTINEL
    path = harness.private("extra.json", document)
    code, result = _run(
        ["verify", "--plan", str(path)], capsys, services=harness.services
    )
    assert code == 2


def test_service_failures_are_value_free(tmp_path, capsys):
    harness = _Harness(tmp_path)

    def noisy(*_args, **_kwargs):
        print(SENTINEL)
        raise RuntimeError(SENTINEL)

    harness.state.inspect = noisy
    code, result = _run(["inspect"], capsys, services=harness.services)
    assert code == 1
    assert result["error"]["code"] == "service_failed"


def test_untrusted_component_error_codes_are_closed(tmp_path, capsys):
    harness = _Harness(tmp_path)

    def propose(_snapshot, _ref):
        raise ServiceRecoveryError(SENTINEL, 99)

    harness.component.propose = propose
    argv = ["plan", "--action", "restore-backup", "--recovery-point", POINT_REF]
    code, result = _run(argv, capsys, services=harness.services)
    assert code == 2
    assert result["error"]["code"] == "input_invalid"


def test_invalid_arguments_never_echo_values(capsys):
    for argv in (
        ["plan", "--action", SENTINEL, "--recovery-point", POINT_REF],
        ["plan", "--action", "restore-backup", "--recovery-point", SENTINEL],
        ["apply", SENTINEL],
        [SENTINEL],
    ):
        code, result = _run(argv, capsys)
        assert code == 2
        assert result["error"]["code"] == "input_invalid"


def test_reprs_and_errors_are_value_free():
    plan = RecoveryPlan(
        RecoveryAction.RESTORE_BACKUP, 3, STATE, POINT_REF, POINT_DIGEST, 4
    )
    services = ServiceRecoveryServices(state=_State(), components={})
    for value in (plan, services, RecoveryStateSnapshot(3, STATE)):
        assert POINT_REF not in repr(value)
        assert STATE not in repr(value)
    error = ServiceRecoveryError(SENTINEL)
    assert SENTINEL not in str(error)
    assert error.code == "input_invalid"


def test_console_surface_registers_help_and_is_unavailable_by_default(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["service-recovery", "apply", "restore-backup", "--help"])
    assert exc_info.value.code == 0
    capsys.readouterr()
    code = main(["service-recovery", "inspect"])
    captured = capsys.readouterr()
    assert code == 7
    result = json.loads(captured.out)
    assert result["error"]["code"] == "adapter_unavailable"
    assert result["command"] == "service-recovery inspect"


def test_console_usage_errors_never_echo_values(capsys):
    for argv in (
        ["service-recovery", SENTINEL],
        ["service-recovery", "plan", "--action", SENTINEL],
        ["service-recovery", "apply", "restore-backup", "--plan", SENTINEL],
    ):
        with pytest.raises(SystemExit) as exc_info:
            main(argv)
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert SENTINEL not in captured.out + captured.err
        assert json.loads(captured.out)["error"]["code"] == "input_invalid"


def test_documented_embedding_example_runs(capsys):
    from pathlib import Path

    page = Path(__file__).resolve().parents[3] / "docs" / "cli" / "service-recovery.md"
    text = page.read_text(encoding="utf-8")
    source = text.split("```python\n", 1)[1].split("\n```", 1)[0]
    exec(compile(source, str(page), "exec"), {"__name__": "service_recovery_doc"})
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "verify: 0 ready",
        "apply: 0 completed 4",
        "repeat apply: 6 state_changed",
    ]
