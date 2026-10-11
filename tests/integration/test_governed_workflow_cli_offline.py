"""Real console routing and explicit synthetic effects under offline injection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openmed.agent.approvals.tokens import ApprovalReceipt
from openmed.agent.correlation import RunId
from openmed.agent.identifiers import WorkflowId
from openmed.cli.governed_workflows import WorkflowCLIRequest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not hasattr(os, "O_NOFOLLOW"), reason="Protected CLI files require O_NOFOLLOW"
    ),
]
ROOT = Path(__file__).resolve().parents[2]


def _private(path, content):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(content))


def test_console_entry_offline_preview_review_and_explicit_resume(tmp_path):
    action = "sha256:" + "a" * 64
    state = "sha256:" + "b" * 64
    request_path = tmp_path / "request.json"
    receipt_path = tmp_path / "receipt.json"
    marker = tmp_path / "synthetic-effect"
    request = WorkflowCLIRequest(
        RunId("run_" + "a" * 32),
        WorkflowId("workflow:org.example/synthetic@1.0.0"),
        action,
        state,
    )
    receipt = ApprovalReceipt(action, "sha256:" + "c" * 64)
    _private(request_path, request.to_dict())
    _private(receipt_path, receipt.to_dict())
    code = r"""
import json, os, socket, sys
from dataclasses import replace
from pathlib import Path
from openmed.cli import main
from openmed.cli.governed_workflows import WorkflowCLIRequest, WorkflowCLIView, WorkflowCLIStatus, WorkflowCLIReceiptVerification, workflow_cli_receipt_digest
from openmed.agent.action_phases import ActionPhase
from openmed.agent.approvals.tokens import ApprovalReceipt
def forbidden(*args, **kwargs):
    raise AssertionError("Network forbidden")
socket.create_connection = forbidden
socket.socket.connect = forbidden
request_path, receipt_path, marker_path = map(Path, sys.argv[1:])
request = WorkflowCLIRequest.from_dict(json.loads(request_path.read_text()))
receipt = ApprovalReceipt.from_dict(json.loads(receipt_path.read_text()))
class Service:
    def __init__(self):
        self.view = WorkflowCLIView(request.run_id, request.workflow_id, request.action_digest, request.expected_state_digest, ActionPhase.WAITING_REVIEW, WorkflowCLIStatus.REVIEW_REQUIRED, 1)
        self.calls = []
        self.receipt_expiry = 4_102_444_800
    def preview(self, bound):
        self.calls.append("preview")
        return replace(self.view, status=WorkflowCLIStatus.READY, phase=ActionPhase.READY)
    def inspect(self, bound):
        self.calls.append("inspect")
        return self.view
    def submit_review(self, bound):
        raise AssertionError("Review was not requested")
    def cancel(self, bound):
        raise AssertionError("Cancellation was not requested")
    def verify_receipt(self, bound, given, *, now):
        self.calls.append("verify_receipt")
        assert now < self.receipt_expiry
        return WorkflowCLIReceiptVerification(bound.action_digest, workflow_cli_receipt_digest(given), self.view.state_digest, given == receipt)
    def resume(self, bound, given, *, now):
        self.calls.append("resume")
        assert bound.expected_state_digest == self.view.state_digest and given == receipt and now < self.receipt_expiry
        fd = os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write("synthetic effect only")
        self.view = replace(self.view, status=WorkflowCLIStatus.COMPLETED, phase=ActionPhase.COMPLETED, state_digest="sha256:" + "d" * 64, committed_effect_count=1, receipt_digest=workflow_cli_receipt_digest(given))
        return self.view
service = Service()
prefix = ["agents", "workflow"]
assert main(prefix + ["preview", "--request", str(request_path)], governance_service=service) == 0
assert not marker_path.exists() and service.calls == ["preview"]
assert main(prefix + ["resume", "--request", str(request_path), "--receipt", str(receipt_path)], governance_service=service) == 0
assert marker_path.read_text() == "synthetic effect only"
assert main(prefix + ["resume", "--request", str(request_path), "--receipt", str(receipt_path)], governance_service=service) == 6
assert service.calls.count("resume") == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(request_path), str(receipt_path), str(marker)],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    documents = [json.loads(row) for row in result.stdout.splitlines()]
    assert len(documents) == 3
    assert documents[0]["data"]["status"] == "ready"
    assert documents[1]["data"]["status"] == "completed"
    assert documents[2]["error"]["code"] == "state_conflict"
    assert str(tmp_path) not in result.stdout


def test_installed_console_entry_has_offline_help_and_no_default_adapter(tmp_path):
    request = WorkflowCLIRequest(
        RunId("run_" + "a" * 32),
        WorkflowId("workflow:org.example/synthetic"),
        "sha256:" + "a" * 64,
    )
    path = tmp_path / "request.json"
    _private(path, request.to_dict())
    code = "import socket,sys; from importlib import metadata; entry=next(e for e in metadata.entry_points(group='console_scripts') if e.name=='openmed'); assert entry.value=='openmed.cli:main'; main=entry.load(); socket.create_connection=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Network forbidden')); sys.exit(main(sys.argv[1:]))"
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    help_result = subprocess.run(
        [sys.executable, "-c", code, "agents", "workflow", "resume", "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert help_result.returncode == 0 and "--receipt" in help_result.stdout
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "agents",
            "workflow",
            "preview",
            "--request",
            str(path),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 7 and result.stderr == ""
    assert json.loads(result.stdout)["error"]["code"] == "adapter_unavailable"
    assert str(path) not in result.stdout
