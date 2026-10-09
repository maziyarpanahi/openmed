"""Synthetic command routing, protected inputs and exact resume authority tests."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import replace

import pytest
from typer.testing import CliRunner

from openmed.agent.action_phases import ActionPhase
from openmed.agent.approvals.tokens import ApprovalReceipt
from openmed.agent.correlation import RunId
from openmed.agent.identifiers import WorkflowId
from openmed.cli import main
from openmed.cli.governed_workflows import (
    WORKFLOW_CLI_EXIT_CODES,
    WORKFLOW_CLI_SCHEMA_VERSION,
    WorkflowCLIReceiptVerification,
    WorkflowCLIRequest,
    WorkflowCLIStatus,
    WorkflowCLIView,
    run_governed_workflow_cli,
    workflow_cli_receipt_digest,
)
from openmed.cli.typer_app import build_app

pytestmark = pytest.mark.skipif(
    not hasattr(os, "O_NOFOLLOW"), reason="Protected CLI files require O_NOFOLLOW"
)
ACTION = "sha256:" + "a" * 64
STATE = "sha256:" + "b" * 64
NEXT_STATE = "sha256:" + "c" * 64
OTHER = "sha256:" + "d" * 64
SENTINEL = "PRIVATE PATIENT https://private.example.test secret-token"


def _request(**changes):
    return replace(
        WorkflowCLIRequest(
            RunId("run_" + "a" * 32),
            WorkflowId("workflow:org.example/synthetic@1.0.0"),
            ACTION,
            STATE,
        ),
        **changes,
    )


def _receipt(**changes):
    return replace(
        ApprovalReceipt(
            ACTION, "role:org.example/reviewer", "sha256:" + "e" * 64, 10, 30
        ),
        **changes,
    )


def _private(tmp_path, name, payload):
    path = tmp_path / name
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(payload if type(payload) is str else json.dumps(payload))
    return path


def _files(tmp_path, *, request=None, receipt=None):
    req = _private(tmp_path, "request.json", (request or _request()).to_dict())
    rec = _private(tmp_path, "receipt.json", (receipt or _receipt()).to_dict())
    return req, rec


class _Service:
    def __init__(self):
        self.view = WorkflowCLIView(
            _request().run_id,
            _request().workflow_id,
            ACTION,
            STATE,
            ActionPhase.WAITING_REVIEW,
            WorkflowCLIStatus.REVIEW_REQUIRED,
            1,
        )
        self.calls = []
        self.effects = 0
        self.stored_receipt = workflow_cli_receipt_digest(_receipt())
        self.verified = True
        self.race = False

    def preview(self, request):
        self.calls.append("preview")
        return replace(
            self.view, phase=ActionPhase.READY, status=WorkflowCLIStatus.READY
        )

    def inspect(self, request):
        self.calls.append("inspect")
        return self.view

    def submit_review(self, request):
        self.calls.append("submit_review")
        if request.expected_state_digest != self.view.state_digest:
            return replace(self.view, status=WorkflowCLIStatus.CONFLICT)
        self.view = replace(
            self.view,
            state_digest=NEXT_STATE,
            phase=ActionPhase.WAITING_REVIEW,
            status=WorkflowCLIStatus.REVIEW_REQUIRED,
            review_request_digest=OTHER,
        )
        return self.view

    def cancel(self, request):
        self.calls.append("cancel")
        if request.expected_state_digest != self.view.state_digest:
            return replace(self.view, status=WorkflowCLIStatus.CONFLICT)
        self.view = replace(
            self.view,
            state_digest=NEXT_STATE,
            phase=ActionPhase.ABORTED,
            status=WorkflowCLIStatus.CANCELLED,
        )
        return self.view

    def verify_receipt(self, request, receipt, *, now):
        self.calls.append("verify_receipt")
        proof = WorkflowCLIReceiptVerification(
            request.action_digest,
            workflow_cli_receipt_digest(receipt),
            self.view.state_digest,
            self.verified
            and workflow_cli_receipt_digest(receipt) == self.stored_receipt,
        )
        if self.race:
            self.view = replace(self.view, state_digest=NEXT_STATE)
        return proof

    def resume(self, request, receipt, *, now):
        self.calls.append("resume")
        if request.expected_state_digest != self.view.state_digest:
            return replace(self.view, status=WorkflowCLIStatus.CONFLICT)
        if (
            now >= receipt.expires_at
            or workflow_cli_receipt_digest(receipt) != self.stored_receipt
        ):
            return replace(self.view, status=WorkflowCLIStatus.DENIED)
        self.effects += 1
        self.view = replace(
            self.view,
            phase=ActionPhase.COMPLETED,
            status=WorkflowCLIStatus.COMPLETED,
            state_digest=NEXT_STATE,
            committed_effect_count=1,
            receipt_digest=self.stored_receipt,
        )
        return self.view


def _run(
    tmp_path,
    operation,
    capsys,
    *,
    service=None,
    request=None,
    receipt=None,
    clock=lambda: 20,
):
    req, rec = _files(tmp_path, request=request, receipt=receipt)
    argv = ["workflow", operation, "--request", str(req)]
    if operation == "resume":
        argv += ["--receipt", str(rec)]
    code = run_governed_workflow_cli(argv, service=service, clock=clock)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert SENTINEL not in captured.out
    assert str(tmp_path) not in captured.out
    result = json.loads(captured.out)
    assert result["schema_version"] == WORKFLOW_CLI_SCHEMA_VERSION
    return code, result


@pytest.mark.parametrize("operation", ["plan", "preview", "inspect"])
def test_read_only_commands_never_invoke_mutations(tmp_path, capsys, operation):
    service = _Service()
    before = service.view
    code, result = _run(tmp_path, operation, capsys, service=service)
    assert code == (4 if operation == "inspect" else 0)
    assert service.calls == ["inspect" if operation == "inspect" else "preview"]
    assert service.view == before and service.effects == 0
    assert result["command"] == f"agents workflow {operation}"


def test_submit_review_only_requests_a_human_decision(tmp_path, capsys):
    service = _Service()
    code, result = _run(tmp_path, "submit-review", capsys, service=service)
    assert code == 4 and not result["ok"]
    assert result["data"]["review_request_digest"] == OTHER
    assert service.calls == ["inspect", "submit_review"]
    assert service.effects == 0
    assert result["data"]["receipt_digest"] is None


def test_cancel_has_a_distinct_exit_status_and_never_executes_effects(tmp_path, capsys):
    service = _Service()
    code, result = _run(tmp_path, "cancel", capsys, service=service)
    assert code == 5 and result["ok"]
    assert result["data"]["phase"] == "aborted"
    assert service.calls == ["inspect", "cancel"]
    assert service.effects == 0


def test_resume_requires_service_verification_and_exact_action_state_receipt(
    tmp_path, capsys
):
    service = _Service()
    code, result = _run(tmp_path, "resume", capsys, service=service)
    assert code == 0 and result["ok"]
    assert service.calls == ["inspect", "verify_receipt", "resume"]
    assert service.effects == 1
    assert result["data"]["receipt_digest"] == service.stored_receipt
    assert "reviewer_role" not in json.dumps(result)


def test_receipt_file_is_not_authority_without_durable_service_verification(
    tmp_path, capsys
):
    service = _Service()
    service.verified = False
    code, result = _run(tmp_path, "resume", capsys, service=service)
    assert code == 3 and result["error"]["code"] == "receipt_unverified"
    assert service.calls == ["inspect", "verify_receipt"] and service.effects == 0


def test_edited_unsigned_receipt_cannot_resume(tmp_path, capsys):
    service = _Service()
    code, result = _run(
        tmp_path,
        "resume",
        capsys,
        service=service,
        receipt=_receipt(token_digest=OTHER),
    )
    assert code == 3 and result["error"]["code"] == "receipt_unverified"
    assert service.effects == 0 and "resume" not in service.calls


@pytest.mark.parametrize(
    "request_data,receipt,code,reason",
    [
        (_request(action_digest=OTHER), _receipt(), 6, "receipt_conflict"),
        (_request(expected_state_digest=OTHER), _receipt(), 6, "state_conflict"),
        (_request(expected_state_digest=None), _receipt(), 6, "state_conflict"),
        (_request(), _receipt(expires_at=20), 4, "receipt_expired"),
        (_request(), _receipt(consumed_at=21), 4, "receipt_future"),
    ],
)
def test_stale_mismatched_or_expired_custody_never_dispatches(
    tmp_path, capsys, request_data, receipt, code, reason
):
    service = _Service()
    actual, result = _run(
        tmp_path,
        "resume",
        capsys,
        service=service,
        request=request_data,
        receipt=receipt,
    )
    assert actual == code and result["error"]["code"] == reason
    assert service.effects == 0 and "resume" not in service.calls


def test_expiry_is_rechecked_after_service_verification(tmp_path, capsys):
    service = _Service()
    times = iter((20, 20, 30))
    code, result = _run(
        tmp_path, "resume", capsys, service=service, clock=lambda: next(times)
    )
    assert code == 4 and result["error"]["code"] == "receipt_expired"
    assert service.calls == ["inspect", "verify_receipt"] and service.effects == 0


@pytest.mark.parametrize("field", ["action_digest", "state_digest", "receipt_digest"])
def test_verification_result_must_bind_every_requested_digest(tmp_path, capsys, field):
    service = _Service()
    original = service.verify_receipt

    def substituted(*args, **kwargs):
        return replace(original(*args, **kwargs), **{field: OTHER})

    service.verify_receipt = substituted
    code, result = _run(tmp_path, "resume", capsys, service=service)
    assert code == 6 and result["error"]["code"] == "receipt_conflict"
    assert service.effects == 0 and "resume" not in service.calls


def test_adapter_compare_and_set_prevents_a_race_after_verification(tmp_path, capsys):
    service = _Service()
    service.race = True
    code, result = _run(tmp_path, "resume", capsys, service=service)
    assert code == 6 and result["data"]["status"] == "conflict"
    assert service.calls == ["inspect", "verify_receipt", "resume"]
    assert service.effects == 0


def test_replaying_same_request_after_success_never_repeats_an_effect(tmp_path, capsys):
    req, rec = _files(tmp_path)
    service = _Service()
    argv = ["workflow", "resume", "--request", str(req), "--receipt", str(rec)]
    assert run_governed_workflow_cli(argv, service=service, clock=lambda: 20) == 0
    capsys.readouterr()
    assert run_governed_workflow_cli(argv, service=service, clock=lambda: 20) == 6
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "state_conflict"
    assert service.effects == 1 and service.calls.count("resume") == 1


@pytest.mark.parametrize("operation", ["submit-review", "cancel", "resume"])
def test_other_run_or_action_cannot_mutate_a_service_snapshot(
    tmp_path, capsys, operation
):
    service = _Service()
    service.view = replace(service.view, run_id=RunId("run_" + "f" * 32))
    code, result = _run(tmp_path, operation, capsys, service=service)
    assert code == 6 and result["error"]["code"] == "action_conflict"
    assert service.calls == ["inspect"] and service.effects == 0


@pytest.mark.parametrize(
    "status,phase,expected",
    [
        (WorkflowCLIStatus.DENIED, ActionPhase.PREFLIGHT, 3),
        (WorkflowCLIStatus.UNAVAILABLE, ActionPhase.PREFLIGHT, 7),
        (WorkflowCLIStatus.CONFLICT, ActionPhase.PREFLIGHT, 6),
        (WorkflowCLIStatus.FAILED, ActionPhase.PREFLIGHT, 1),
    ],
)
def test_service_refusals_have_stable_distinct_exit_codes(
    tmp_path, capsys, status, phase, expected
):
    service = _Service()
    service.view = replace(service.view, status=status, phase=phase)
    code, result = _run(tmp_path, "resume", capsys, service=service)
    assert code == expected and result["data"]["status"] == status.value
    assert service.calls == ["inspect"] and service.effects == 0


@pytest.mark.parametrize(
    "phase,status",
    [
        (ActionPhase.COMPLETED, WorkflowCLIStatus.COMPLETED),
        (ActionPhase.ABORTED, WorkflowCLIStatus.CANCELLED),
    ],
)
def test_terminal_runs_cannot_be_resumed(tmp_path, capsys, phase, status):
    service = _Service()
    service.view = replace(service.view, phase=phase, status=status)
    code, result = _run(tmp_path, "resume", capsys, service=service)
    assert code == 6 and result["error"]["code"] == "terminal_state"
    assert service.effects == 0 and service.calls == ["inspect"]


def test_cancel_of_already_cancelled_run_is_idempotent(tmp_path, capsys):
    service = _Service()
    service.view = replace(
        service.view, phase=ActionPhase.ABORTED, status=WorkflowCLIStatus.CANCELLED
    )
    code, result = _run(tmp_path, "cancel", capsys, service=service)
    assert code == 5 and result["data"]["status"] == "cancelled"
    assert service.calls == ["inspect"]


@pytest.mark.parametrize(
    "operation", ["plan", "preview", "inspect", "submit-review", "cancel", "resume"]
)
def test_no_default_adapter_is_enabled(tmp_path, capsys, operation):
    code, result = _run(tmp_path, operation, capsys)
    assert code == 7 and result["error"]["code"] == "adapter_unavailable"


@pytest.mark.parametrize(
    "field",
    [
        "approval_token",
        "credential",
        "reviewer_identity",
        "reviewer_role",
        "payload",
        "patient_id",
        "endpoint",
    ],
)
def test_protected_request_rejects_content_or_inline_authority_fields(
    tmp_path, capsys, field
):
    payload = {**_request().to_dict(), field: SENTINEL}
    path = _private(tmp_path, "input.json", payload)
    service = _Service()
    code = run_governed_workflow_cli(
        ["workflow", "preview", "--request", str(path)], service=service
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 2 and result["error"]["code"] == "input_invalid"
    assert SENTINEL not in json.dumps(result) and service.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        '{"run_id":"PRIVATE PATIENT", "run_id":"OTHER"}',
        '{"secret":NaN}',
        '{"secret":Infinity}',
        '{"secret":',
        "[]",
        "x" * 65_537,
        '{"secret":' + "[" * 12 + '"PRIVATE PATIENT"' + "]" * 12 + "}",
    ],
)
def test_invalid_json_and_bounds_fail_before_adapter_calls(tmp_path, capsys, payload):
    path = _private(tmp_path, "input.json", payload)
    service = _Service()
    assert (
        run_governed_workflow_cli(
            ["workflow", "preview", "--request", str(path)], service=service
        )
        == 2
    )
    out = capsys.readouterr().out
    assert "PRIVATE" not in out and str(path) not in out and service.calls == []


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604])
def test_unprotected_input_permissions_are_refused(tmp_path, capsys, mode):
    req, _ = _files(tmp_path)
    req.chmod(mode)
    service = _Service()
    assert (
        run_governed_workflow_cli(
            ["workflow", "preview", "--request", str(req)], service=service
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "input_unprotected"
    assert service.calls == []


def test_symlink_and_fifo_inputs_are_not_followed_or_blocked(tmp_path, capsys):
    req, _ = _files(tmp_path)
    link = tmp_path / "link.json"
    link.symlink_to(req)
    fifo = tmp_path / "input.fifo"
    os.mkfifo(fifo, 0o600)
    for path in (link, fifo):
        service = _Service()
        assert (
            run_governed_workflow_cli(
                ["workflow", "preview", "--request", str(path)], service=service
            )
            == 2
        )
        out = capsys.readouterr().out
        assert str(path) not in out and service.calls == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--approve", SENTINEL],
        ["--receipt", SENTINEL],
        ["--req", SENTINEL],
        ["--request", SENTINEL],
        ["--request=" + SENTINEL],
        ["--json", "--json"],
        ["--reviewed"],
        ["--now", "20"],
    ],
)
def test_command_line_secrets_flags_and_ambiguous_options_are_refused(
    tmp_path, capsys, extra
):
    req, _ = _files(tmp_path)
    service = _Service()
    code = main(
        ["agents", "workflow", "preview", "--request", str(req), *extra],
        governance_service=service,
    )
    captured = capsys.readouterr()
    assert code == 2 and captured.err == ""
    assert SENTINEL not in captured.out and str(req) not in captured.out
    assert json.loads(captured.out)["error"]["code"] == "input_invalid"
    assert service.calls == []


@pytest.mark.parametrize(
    "prefix",
    [
        ["--config-path", SENTINEL],
        ["--config-path=" + SENTINEL],
        ["--config-path", SENTINEL, "--config-path=" + SENTINEL],
    ],
)
def test_global_config_does_not_become_ambient_workflow_authority(
    tmp_path, capsys, prefix
):
    service = _Service()
    code = main(
        [
            *prefix,
            "agents",
            "workflow",
            "resume",
            "--approval-token",
            SENTINEL,
        ],
        governance_service=service,
    )
    captured = capsys.readouterr()
    assert code == 2 and captured.err == "" and SENTINEL not in captured.out
    assert service.calls == []


@pytest.mark.parametrize(
    "prefix",
    [
        [],
        ["--config-path", "agents"],
        ["--config-path", "workflow"],
        ["--config-path=agents"],
        ["--config-path", "agents", "--config-path=workflow"],
    ],
)
def test_other_command_data_does_not_route_to_governed_workflows(monkeypatch, prefix):
    import importlib

    cli_module = importlib.import_module("openmed.cli.main")
    calls = []

    def benchmark_handler(args):
        calls.append(args.models)
        return 0

    monkeypatch.setattr(cli_module, "_handle_benchmark_pii", benchmark_handler)
    service = _Service()
    assert (
        main(
            [*prefix, "benchmark", "pii", "--models", "agents", "workflow"],
            governance_service=service,
        )
        == 0
    )
    assert calls == [["agents", "workflow"]] and service.calls == []


@pytest.mark.parametrize(
    "operation",
    ["preview", "inspect", "submit_review", "cancel", "verify_receipt", "resume"],
)
def test_adapter_output_and_exception_text_are_not_cli_diagnostics(
    tmp_path, capsys, operation
):
    service = _Service()

    def broken(*_args, **_kwargs):
        print(SENTINEL)
        print(SENTINEL, file=__import__("sys").stderr)
        raise RuntimeError(SENTINEL)

    setattr(service, operation, broken)
    command = {"submit_review": "submit-review", "verify_receipt": "resume"}.get(
        operation, operation
    )
    code, result = _run(tmp_path, command, capsys, service=service)
    assert code == 1 and result["error"]["code"] == "service_failed"
    assert service.effects == 0


@pytest.mark.parametrize("returned", [{"payload": SENTINEL}, True, None, SENTINEL])
def test_untyped_service_results_cannot_reach_output(tmp_path, capsys, returned):
    service = _Service()
    service.preview = lambda *_: returned
    code, result = _run(tmp_path, "preview", capsys, service=service)
    assert code == 1 and result["error"]["code"] == "service_result_invalid"


def test_fabricated_boolean_verification_cannot_authorize_resume(tmp_path, capsys):
    service = _Service()
    service.verify_receipt = lambda *_args, **_kwargs: True
    code, result = _run(tmp_path, "resume", capsys, service=service)
    assert code == 1 and result["error"]["code"] == "service_result_invalid"
    assert service.effects == 0


def test_actual_console_entry_and_typer_share_the_value_free_boundary(tmp_path, capsys):
    req, _ = _files(tmp_path)
    argv = ["agents", "workflow", "preview", "--request", str(req)]
    first_service = _Service()
    assert main(argv, governance_service=first_service) == 0
    first = json.loads(capsys.readouterr().out)
    second_service = _Service()
    result = CliRunner().invoke(build_app(governance_service=second_service), argv)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == first
    assert first_service.calls == second_service.calls == ["preview"]


def test_commands_do_not_open_network_connections(tmp_path, capsys, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("synthetic CLI attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    service = _Service()
    code, _ = _run(tmp_path, "resume", capsys, service=service)
    assert code == 0 and service.effects == 1


def test_exit_status_contract_cannot_be_mutated():
    with pytest.raises(TypeError):
        WORKFLOW_CLI_EXIT_CODES[WorkflowCLIStatus.DENIED] = 0


@pytest.mark.parametrize(
    "change",
    [
        {"signature": SENTINEL},
        {"nonce": SENTINEL},
        {"reviewer_identity": SENTINEL},
        {"reviewer_role": SENTINEL},
        {"action_digest": SENTINEL},
        {"expires_at": "30"},
        {"consumed_at": True},
        {"schema_version": "unknown"},
    ],
)
def test_malformed_receipt_or_token_file_never_reaches_a_service(
    tmp_path, capsys, change
):
    req, _ = _files(tmp_path)
    receipt_path = _private(
        tmp_path, "bad-receipt.json", {**_receipt().to_dict(), **change}
    )
    service = _Service()
    code = run_governed_workflow_cli(
        ["workflow", "resume", "--request", str(req), "--receipt", str(receipt_path)],
        service=service,
        clock=lambda: 20,
    )
    captured = capsys.readouterr()
    assert code == 2 and captured.err == "" and SENTINEL not in captured.out
    assert json.loads(captured.out)["error"]["code"] == "receipt_invalid"
    assert service.calls == [] and service.effects == 0


@pytest.mark.parametrize(
    "operation",
    ["preview", "inspect", "submit_review", "cancel", "verify_receipt", "resume"],
)
def test_missing_adapter_capability_is_distinct_from_adapter_failure(
    tmp_path, capsys, operation
):
    service = _Service()

    def unavailable(*_args, **_kwargs):
        raise NotImplementedError(SENTINEL)

    setattr(service, operation, unavailable)
    command = {"submit_review": "submit-review", "verify_receipt": "resume"}.get(
        operation, operation
    )
    code, result = _run(tmp_path, command, capsys, service=service)
    assert code == 7 and result["error"]["code"] == "adapter_unavailable"
    assert service.effects == 0


def test_partial_preview_only_service_has_no_effectful_adapter(tmp_path, capsys):
    class PreviewOnly:
        preview = _Service().preview
        inspect = _Service().inspect

    code, result = _run(tmp_path, "resume", capsys, service=PreviewOnly())
    assert code == 7 and result["error"]["code"] == "adapter_unavailable"


@pytest.mark.parametrize(
    "result",
    [
        {"receipt_digest": None},
        {"receipt_digest": OTHER},
        {"run_id": RunId("run_" + "f" * 32)},
    ],
)
def test_resume_acknowledgement_must_match_actual_bound_receipt(
    tmp_path, capsys, result
):
    service = _Service()
    original = service.resume

    def wrong(*args, **kwargs):
        return replace(original(*args, **kwargs), **result)

    service.resume = wrong
    code, payload = _run(tmp_path, "resume", capsys, service=service)
    assert code in (1, 6) and not payload["ok"]
    # A malformed acknowledgement cannot prove that the effect did not commit.
    assert service.effects == 1 and service.calls.count("resume") == 1


@pytest.mark.parametrize(
    "clock", [lambda: True, lambda: 1.2, lambda: -1, lambda: "PRIVATE PATIENT"]
)
def test_invalid_clock_fails_before_authority_verification(tmp_path, capsys, clock):
    service = _Service()
    code, result = _run(tmp_path, "resume", capsys, service=service, clock=clock)
    assert code == 1 and result["error"]["code"] == "service_failed"
    assert service.calls == [] and service.effects == 0


def test_same_action_cannot_silently_change_proposed_effect_counts(tmp_path, capsys):
    service = _Service()
    original = service.resume

    def changed(*args, **kwargs):
        return replace(original(*args, **kwargs), proposed_effect_count=2)

    service.resume = changed
    code, result = _run(tmp_path, "resume", capsys, service=service)
    assert code == 1 and result["error"]["code"] == "service_result_invalid"
    assert service.effects == 1 and service.calls.count("resume") == 1


def test_cancellation_acknowledgement_cannot_erase_committed_effect_evidence(
    tmp_path, capsys
):
    service = _Service()
    service.view = replace(service.view, committed_effect_count=1)
    original = service.cancel

    def changed(*args, **kwargs):
        return replace(original(*args, **kwargs), committed_effect_count=0)

    service.cancel = changed
    code, result = _run(tmp_path, "cancel", capsys, service=service)
    assert code == 1 and result["error"]["code"] == "service_result_invalid"
    assert service.effects == 0 and service.calls.count("cancel") == 1


def test_platform_without_secure_open_fails_closed(tmp_path, capsys, monkeypatch):
    req, _ = _files(tmp_path)
    monkeypatch.delattr(os, "O_NOFOLLOW")
    service = _Service()
    code = run_governed_workflow_cli(
        ["workflow", "preview", "--request", str(req)], service=service
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "input_unprotected"
    assert service.calls == []


def test_typer_parser_errors_also_omit_inline_values(tmp_path):
    req, _ = _files(tmp_path)
    service = _Service()
    result = CliRunner().invoke(
        build_app(governance_service=service),
        [
            "agents",
            "workflow",
            "preview",
            "--request",
            str(req),
            "--approval-token",
            SENTINEL,
        ],
    )
    assert result.exit_code == 2 and SENTINEL not in result.output
    assert str(req) not in result.output
    assert json.loads(result.stdout)["error"]["code"] == "input_invalid"
    assert service.calls == []
