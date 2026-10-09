"""Synthetic command safety, lifecycle, permissions and leakage controls."""

import hashlib
import json
import os
import signal
from pathlib import Path

import pytest

from openmed.cli import main as cli_main
from openmed.cli import multimodal_inference as cli
from openmed.cli.main import build_parser
from openmed.multimodal.provider_result import ProviderResultEnvelope

SENSITIVE = "Ada Synthetic, ID 000-12-3456, Lucía 合成"


class FakeSession:
    def __init__(self, request, *, error=None, close_error=False, outcome="success"):
        self.request = request
        self.error = error
        self.close_error = close_error
        self.outcome = outcome
        self.calls = []
        self.content = json.dumps({"text": SENSITIVE, "offsets": [0, 2]}).encode()
        self.result = cli.LocalSessionResult(
            self.content if outcome == "success" else None,
            ProviderResultEnvelope(
                provider_id="synthetic",
                model_id="fixture",
                input_digest=hashlib.sha256(request.media).hexdigest(),
                output_digest=(
                    hashlib.sha256(self.content).hexdigest()
                    if outcome == "success"
                    else None
                ),
                outcome=outcome,
                abstention_code="malformed_media" if outcome == "abstention" else None,
                duration_ms=0,
                count_metadata={"input_bytes": len(request.media)},
            ),
        )

    def run(self):
        self.calls.append("run")
        print(SENSITIVE)
        if self.error:
            raise self.error
        return self.result

    def cancel(self):
        self.calls.append("cancel")
        self.request = None

    def close(self):
        self.calls.append("close")
        self.request = None
        if self.close_error:
            raise RuntimeError(SENSITIVE)


class FakeService:
    local_only = True
    tasks = frozenset(cli.LocalTask)

    def __init__(self, **options):
        self.options = options
        self.sessions = []
        self.open_error = None

    def open(self, request):
        if self.open_error:
            raise self.open_error
        session = FakeSession(request, **self.options)
        self.sessions.append(session)
        return session


@pytest.fixture
def arguments(tmp_path, monkeypatch):
    # POSIX positive cases run natively on macOS/Linux. Windows refusal is
    # tested separately; mocking permission support is not a Windows ACL test.
    if os.name != "posix":
        pytest.skip("Protected POSIX output is unsupported on this platform")
    media = tmp_path / "private-source.txt"
    media.write_bytes(b"synthetic media\n")
    artifact = tmp_path / "local-model"
    artifact.mkdir()
    consent = tmp_path / "private-consent.txt"
    consent.write_bytes(b"synthetic consent\n")
    return build_parser().parse_args(
        [
            "multimodal",
            "draft",
            str(media),
            "--model-artifact",
            str(artifact),
            "--consent-evidence",
            str(consent),
            "--confirm-review",
            "--output-directory",
            str(tmp_path / "outputs"),
            "--json",
        ]
    )


@pytest.mark.parametrize("task", list(cli.LocalTask))
def test_success_separates_payload_and_evidence(arguments, capsys, task):
    arguments.local_task = task
    service = FakeService()
    assert cli.handle_multimodal_inference(arguments, service=service) == 0
    assert service.sessions[0].calls == ["run", "close"]
    assert service.sessions[0].request is None
    directory = arguments.output_directory
    protected = (directory / "result.json").read_bytes()
    result = json.loads(protected)
    assert result["result"]["text"] == SENSITIVE
    assert result["result"]["offsets"] == [0, 2]
    assert result["requires_clinical_review"] is True
    assert result["reviewer_confirmed"] is True
    assert result["notice"] == cli.NON_DIAGNOSTIC_NOTICE
    evidence = json.loads((directory / "evidence.json").read_bytes())
    assert evidence["result_file_digest"] == hashlib.sha256(protected).hexdigest()
    assert (
        evidence["provider"]["input_digest"]
        == hashlib.sha256(arguments.media.read_bytes()).hexdigest()
    )
    assert (directory / "COMPLETE").read_bytes() == b"complete\n"
    assert directory.stat().st_mode & 0o777 == 0o700
    for path in directory.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
    captured = capsys.readouterr()
    assert json.loads(captured.out)["data"] == evidence
    for output in (captured.out, captured.err, json.dumps(evidence)):
        assert SENSITIVE not in output
        assert str(arguments.media) not in output
        assert str(arguments.model_artifact) not in output
        assert str(arguments.consent_evidence) not in output
        assert "private-source" not in output


@pytest.mark.parametrize(
    ("error", "code", "exit_code"),
    [
        (cli.SessionRefusal("provider_unavailable"), "provider_unavailable", 1),
        (cli.SessionRefusal("consent_required"), "consent_required", 2),
        (cli.SessionRefusal("malformed_media"), "malformed_media", 2),
        (RuntimeError(SENSITIVE), "session_failed", 1),
        (KeyboardInterrupt(), "cancelled", 130),
        (SystemExit(SENSITIVE), "session_failed", 1),
        (cli.SessionRefusal(SENSITIVE), "session_failed", 1),
    ],
)
def test_refusal_cancels_closes_without_outputs(
    arguments, capsys, error, code, exit_code
):
    service = FakeService(error=error)
    with pytest.raises(cli.CliError) as caught:
        cli.handle_multimodal_inference(arguments, service=service)
    assert caught.value.code == code
    assert caught.value.exit_code == exit_code
    assert service.sessions[0].calls == ["run", "cancel", "close"]
    assert not arguments.output_directory.exists()
    assert SENSITIVE not in str(caught.value)
    assert SENSITIVE not in repr(service.sessions[0].result)
    assert SENSITIVE not in capsys.readouterr().out


@pytest.mark.parametrize(
    "field,code",
    [("consent_evidence", "consent_required"), ("confirm_review", "review_required")],
)
def test_missing_permission_does_not_open_service(arguments, field, code):
    setattr(arguments, field, None)
    service = FakeService()
    with pytest.raises(cli.CliError, match=code):
        cli.handle_multimodal_inference(arguments, service=service)
    assert service.sessions == []
    assert not arguments.output_directory.exists()


def test_document_does_not_require_ambient_consent(arguments):
    arguments.local_task = cli.LocalTask.PROCESS_DOCUMENT
    arguments.consent_evidence = None
    assert cli.handle_multimodal_inference(arguments, service=FakeService()) == 0


@pytest.mark.parametrize(
    "local_only,tasks,code",
    [
        (False, frozenset(cli.LocalTask), "offline_required"),
        (True, frozenset(), "provider_unavailable"),
    ],
)
def test_declaration_refuses_before_open(arguments, local_only, tasks, code):
    service = FakeService()
    service.local_only = local_only
    service.tasks = tasks
    with pytest.raises(cli.CliError, match=code):
        cli.handle_multimodal_inference(arguments, service=service)
    assert service.sessions == []


@pytest.mark.parametrize(
    "outcome,code",
    [
        ("abstention", "malformed_media"),
        ("provider_unavailable", "provider_unavailable"),
        ("validation_failure", "session_refused"),
    ],
)
def test_provider_envelope_refusals(arguments, outcome, code):
    service = FakeService(outcome=outcome)
    with pytest.raises(cli.CliError, match=code):
        cli.handle_multimodal_inference(arguments, service=service)
    assert service.sessions[0].calls == ["run", "cancel", "close"]
    assert not arguments.output_directory.exists()


def test_unverified_cleanup_never_finalizes(arguments):
    service = FakeService(close_error=True)
    with pytest.raises(cli.CliError, match="cleanup_failed"):
        cli.handle_multimodal_inference(arguments, service=service)
    assert service.sessions[0].calls == ["run", "close", "cancel", "close"]
    assert not arguments.output_directory.exists()


@pytest.mark.parametrize("break_at", ["result.json", "evidence.json", "COMPLETE"])
@pytest.mark.parametrize(
    "error,code",
    [(OSError(SENSITIVE), "output_failed"), (KeyboardInterrupt(), "cancelled")],
)
def test_write_failure_removes_pair(arguments, monkeypatch, break_at, error, code):
    original = cli._write_private

    def write(path, content):
        original(path, content)
        if path.name == break_at:
            raise error

    monkeypatch.setattr(cli, "_write_private", write)
    service = FakeService()
    with pytest.raises(cli.CliError, match=code):
        cli.handle_multimodal_inference(arguments, service=service)
    assert not arguments.output_directory.exists()
    assert service.sessions[0].calls == ["run", "close"]


def test_existing_destination_preserved(arguments):
    arguments.output_directory.mkdir()
    sentinel = arguments.output_directory / "result.json"
    sentinel.write_text("existing")
    service = FakeService()
    with pytest.raises(cli.CliError, match="output_failed"):
        cli.handle_multimodal_inference(arguments, service=service)
    assert sentinel.read_text() == "existing"
    assert service.sessions == []


def test_windows_refuses_before_factory(tmp_path, monkeypatch):
    args = build_parser().parse_args(
        [
            "multimodal",
            "transcribe",
            "source",
            "--model-artifact",
            "model",
            "--output-directory",
            str(tmp_path / "outputs"),
        ]
    )
    monkeypatch.setattr(cli, "_platform", lambda: "nt")
    service = FakeService()
    with pytest.raises(cli.CliError, match="unsupported_permissions"):
        cli.handle_multimodal_inference(args, service=service)
    assert service.sessions == []
    assert not args.output_directory.exists()


@pytest.mark.parametrize("field", ["media", "model_artifact", "consent_evidence"])
def test_symlink_inputs_refused(arguments, field):
    original = getattr(arguments, field)
    link = original.with_name("link")
    link.symlink_to(original)
    setattr(arguments, field, link)
    service = FakeService()
    with pytest.raises(cli.CliError, match="invalid_artifact"):
        cli.handle_multimodal_inference(arguments, service=service)
    assert service.sessions == []


def test_empty_and_over_limit_inputs(arguments, monkeypatch):
    arguments.media.write_bytes(b"")
    with pytest.raises(cli.CliError, match="malformed_media"):
        cli.handle_multimodal_inference(arguments, service=FakeService())
    arguments.media.write_bytes(b"1234")
    monkeypatch.setattr(cli, "MAX_INPUT_BYTES", 3)
    with pytest.raises(cli.CliError, match="resource_limit"):
        cli.handle_multimodal_inference(arguments, service=FakeService())


@pytest.mark.parametrize(
    "content", [b"not json", b"[]", b'{"text":NaN}', b'{"text":"tampered"}']
)
def test_invalid_results_fail_closed(arguments, monkeypatch, content):
    service = FakeService()
    original = service.open

    def open_session(request):
        session = original(request)
        metadata = session.result.evidence.to_dict()
        if content != b'{"text":"tampered"}':
            metadata["output_digest"] = hashlib.sha256(content).hexdigest()
        session.result = cli.LocalSessionResult(
            content, ProviderResultEnvelope.from_dict(metadata)
        )
        return session

    monkeypatch.setattr(service, "open", open_session)
    with pytest.raises(cli.CliError, match="invalid_result"):
        cli.handle_multimodal_inference(arguments, service=service)
    assert not arguments.output_directory.exists()


def test_no_default_provider_or_network(arguments, capsys, monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        pytest.fail("Network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    argv = [
        "multimodal",
        "draft",
        str(arguments.media),
        "--model-artifact",
        str(arguments.model_artifact),
        "--consent-evidence",
        str(arguments.consent_evidence),
        "--confirm-review",
        "--output-directory",
        str(arguments.output_directory),
        "--json",
    ]
    assert cli_main(argv) == 1
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "provider_unavailable"
    assert str(arguments.media) not in error["message"]
    assert not arguments.output_directory.exists()


def test_signals_restore_host_handlers(arguments, monkeypatch):
    original = signal.getsignal(signal.SIGTERM)
    service = FakeService()

    def interrupted(self):
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(FakeSession, "run", interrupted)
    with pytest.raises(cli.CliError, match="cancelled"):
        cli.handle_multimodal_inference(arguments, service=service)
    assert signal.getsignal(signal.SIGTERM) == original
    assert service.sessions[0].calls == ["cancel", "close"]


@pytest.mark.parametrize(
    "argv",
    [
        ["multimodal", SENSITIVE, "--json"],
        ["multimodal", "draft", SENSITIVE, "--json"],
        ["multimodal", "--private", SENSITIVE, "--json"],
    ],
)
def test_parse_errors_never_echo_submitted_values(argv, capsys):
    with pytest.raises(SystemExit) as caught:
        cli_main(argv)
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "invalid_arguments"
    assert SENSITIVE not in captured.out + captured.err


def test_cancel_failure_still_attempts_close(arguments, monkeypatch):
    def fail_cancel(self):
        self.calls.append("cancel")
        raise RuntimeError(SENSITIVE)

    monkeypatch.setattr(FakeSession, "cancel", fail_cancel)
    service = FakeService(error=KeyboardInterrupt())
    with pytest.raises(cli.CliError, match="cleanup_failed"):
        cli.handle_multimodal_inference(arguments, service=service)
    assert service.sessions[0].calls == ["run", "cancel", "close"]
    assert not arguments.output_directory.exists()


def test_output_cleanup_failure_is_not_success(arguments, monkeypatch):
    original = cli._write_private

    def fail_write(path, content):
        original(path, content)
        raise OSError(SENSITIVE)

    def fail_unlink(*args, **kwargs):
        raise PermissionError(SENSITIVE)

    monkeypatch.setattr(cli, "_write_private", fail_write)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(cli.CliError, match="cleanup_failed"):
        cli.handle_multimodal_inference(arguments, service=FakeService())
    assert not (arguments.output_directory / "COMPLETE").exists()
    assert arguments.output_directory.stat().st_mode & 0o777 == 0o700


def test_consent_refused_during_open(arguments):
    service = FakeService()
    service.open_error = cli.SessionRefusal("consent_required")
    with pytest.raises(cli.CliError, match="consent_required"):
        cli.handle_multimodal_inference(arguments, service=service)
    assert not arguments.output_directory.exists()


def test_bound_source_digest_required(arguments, monkeypatch):
    service = FakeService()
    original = service.open

    def open_session(request):
        session = original(request)
        metadata = session.result.evidence.to_dict()
        metadata["input_digest"] = "0" * 64
        session.result = cli.LocalSessionResult(
            session.content, ProviderResultEnvelope.from_dict(metadata)
        )
        return session

    monkeypatch.setattr(service, "open", open_session)
    with pytest.raises(cli.CliError, match="invalid_result"):
        cli.handle_multimodal_inference(arguments, service=service)
    assert service.sessions[0].calls == ["run", "cancel", "close"]


def test_output_symlink_never_followed(arguments):
    target = arguments.output_directory.with_name("existing")
    target.mkdir()
    arguments.output_directory.symlink_to(target)
    with pytest.raises(cli.CliError, match="output_failed"):
        cli.handle_multimodal_inference(arguments, service=FakeService())
    assert arguments.output_directory.is_symlink()
    assert list(target.iterdir()) == []
