"""Offline command adapters over a trusted, caller-injected session service.

This is a CLI-owned protocol, not the ambient library's session state machine.
No default provider, recorder, downloader or clinical-write API is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import signal
import stat
import threading
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import NoReturn, Protocol, Sequence

from openmed.multimodal.provider_result import (
    ProviderAbstentionCode,
    ProviderResultEnvelope,
    ProviderResultOutcome,
)

from ._output import CliError, emit, emit_error

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_CONSENT_BYTES = 64 * 1024
MAX_RESULT_BYTES = 8 * 1024 * 1024
NON_DIAGNOSTIC_NOTICE = (
    "Non-diagnostic draft output. Clinical review is required before downstream use."
)


class LocalTask(str, Enum):
    """Tasks explicitly selectable at the command boundary."""

    PROCESS_DOCUMENT = "process-document"
    TRANSCRIBE = "transcribe"
    DRAFT = "draft"


@dataclass(frozen=True, repr=False)
class LocalSessionRequest:
    """Explicit local inputs passed privately to a trusted application service.

    Args:
        task: Requested declared task; never capture or clinical submission.
        media: Bounded source bytes; service validates media and applies PHI policy.
        model_artifact: Existing user-selected artifact; never a download request.
        consent_evidence: Opaque evidence, validated by the service, not a boolean.
        reviewer_confirmed: Explicit permission to produce a protected draft.
    """

    task: LocalTask
    media: bytes = field(repr=False)
    model_artifact: Path = field(repr=False)
    consent_evidence: bytes | None = field(repr=False)
    reviewer_confirmed: bool


@dataclass(frozen=True, repr=False)
class LocalSessionResult:
    """Separate protected JSON bytes from the existing value-free envelope.

    Args:
        protected_json: Complete JSON object with service-owned citations/provenance.
            None on refusal. No partial transcripts or unfinished drafts are accepted.
        evidence: Existing provider envelope binding source and result digests.
    """

    protected_json: bytes | None = field(repr=False)
    evidence: ProviderResultEnvelope


class LocalSession(Protocol):
    """Service owns consent, media validation, policy, provenance and retention.

    Implementations must stay offline, log no payloads, and perform no capture,
    downloads, final clinical export or record writes. Results must obey the
    library's refusal/review rules. Cleanup methods must be idempotent.
    """

    def run(self) -> LocalSessionResult:
        """Return one complete result or a refusal, never partial finalization."""
        ...

    def cancel(self) -> None:
        """Invalidate pending work and destroy service-owned retained inputs."""
        ...

    def close(self) -> None:
        """Finish verified policy cleanup; raise if destruction cannot be verified."""
        ...


class LocalSessionService(Protocol):
    """Narrow application injection boundary, independent of inference algorithms.

    ``local_only`` must be exactly True. ``tasks`` declares supported tasks.
    ``open`` must validate consent before processing and release any allocations
    itself if opening fails. The CLI never starts or resumes capture.
    """

    local_only: bool
    tasks: frozenset[LocalTask]

    def open(self, request: LocalSessionRequest) -> LocalSession:
        """Open a bounded file-processing session under application policy."""
        ...


class SessionRefusal(Exception):
    """Controlled refusal from an injected service; arbitrary text is suppressed.

    Args:
        code: One of the documented command refusal codes.
    """

    def __init__(self, code: str) -> None:
        super().__init__("Local session refused.")
        self.code = code if type(code) is str and code in _ERRORS else "session_failed"


_ERRORS = {
    "invalid_arguments": 2,
    "provider_unavailable": 1,
    "offline_required": 1,
    "consent_required": 2,
    "review_required": 2,
    "malformed_media": 2,
    "resource_limit": 2,
    "invalid_artifact": 2,
    "invalid_result": 1,
    "session_refused": 1,
    "session_failed": 1,
    "output_failed": 1,
    "cleanup_failed": 1,
    "unsupported_permissions": 1,
    "cancelled": 130,
}


class _Discard:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        pass


def add_multimodal_inference_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register explicit document, transcription and draft file commands."""
    group = subparsers.add_parser("multimodal", help="Explicit local media tasks.")
    tasks = group.add_subparsers(dest="multimodal_task", required=True)
    for task in LocalTask:
        parser = tasks.add_parser(task.value, help=f"Run local {task.value}.")
        parser.add_argument("media", type=Path, help="Explicit local source file.")
        parser.add_argument("--model-artifact", required=True, type=Path)
        parser.add_argument(
            "--session-factory", help="Trusted installed module:function."
        )
        parser.add_argument("--consent-evidence", type=Path)
        parser.add_argument("--confirm-review", action="store_true")
        parser.add_argument("--output-directory", required=True, type=Path)
        parser.set_defaults(handler=handle_multimodal_inference, local_task=task)


def protect_multimodal_diagnostics(
    parser: argparse.ArgumentParser, argv: Sequence[str]
) -> None:
    """Suppress submitted values in argparse failures for media invocations.

    Args:
        parser: Top-level parser; only this invocation's error handlers are changed.
        argv: Explicit arguments, used only to select controlled command/JSON output.
    """
    if "multimodal" not in argv:
        return
    safe_args = argparse.Namespace(command="multimodal", json_output="--json" in argv)

    def error(message: str) -> NoReturn:
        raise SystemExit(emit_error(safe_args, _error("invalid_arguments")))

    setattr(parser, "error", error)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            group = action.choices["multimodal"]
            setattr(group, "error", error)
            for task_action in group._actions:
                if isinstance(task_action, argparse._SubParsersAction):
                    for leaf in task_action.choices.values():
                        setattr(leaf, "error", error)


def _error(code: str) -> CliError:
    return CliError(
        f"Local media command refused ({code}).", code=code, exit_code=_ERRORS[code]
    )


def _read_local(path: Path, limit: int) -> bytes:
    # Reject special files before opening (no blocking FIFO/device reads).
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise SessionRefusal("invalid_artifact")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise SessionRefusal("invalid_artifact")
        data = handle.read(limit + 1)
    if not data:
        raise SessionRefusal("malformed_media")
    if len(data) > limit:
        raise SessionRefusal("resource_limit")
    return data


def _load_service(factory: str | None) -> LocalSessionService:
    if not factory:
        raise SessionRefusal("provider_unavailable")
    module, separator, name = factory.partition(":")
    if (
        not separator
        or not all(part.isidentifier() for part in module.split("."))
        or not name.isidentifier()
    ):
        raise SessionRefusal("provider_unavailable")
    return getattr(importlib.import_module(module), name)()


@contextmanager
def _interruptions():
    # Signal hooks are process-global; embedded worker threads use cancellation
    # from their injected service rather than altering the host's signal policy.
    previous = {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, interrupt)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _platform() -> str:
    return os.name


def _validated_result(
    result: LocalSessionResult, media: bytes, task: LocalTask
) -> tuple[bytes, bytes, dict]:
    if type(result) is not LocalSessionResult:
        raise SessionRefusal("invalid_result")
    evidence = ProviderResultEnvelope.from_dict(result.evidence.to_dict())
    if evidence.input_digest != hashlib.sha256(media).hexdigest():
        raise SessionRefusal("invalid_result")
    if evidence.outcome is not ProviderResultOutcome.SUCCESS:
        if result.protected_json is not None:
            raise SessionRefusal("invalid_result")
        if evidence.outcome is ProviderResultOutcome.PROVIDER_UNAVAILABLE:
            raise SessionRefusal("provider_unavailable")
        if evidence.abstention_code is ProviderAbstentionCode.MALFORMED_MEDIA:
            raise SessionRefusal("malformed_media")
        if evidence.abstention_code is ProviderAbstentionCode.RESOURCE_LIMIT:
            raise SessionRefusal("resource_limit")
        raise SessionRefusal("session_refused")
    content = result.protected_json
    if (
        type(content) is not bytes
        or len(content) > MAX_RESULT_BYTES
        or hashlib.sha256(content).hexdigest() != evidence.output_digest
    ):
        raise SessionRefusal("invalid_result")
    payload = json.loads(content)
    if type(payload) is not dict:
        raise SessionRefusal("invalid_result")
    protected = {
        "schema_version": "openmed.cli.multimodal.protected.v1",
        "task": task.value,
        "notice": NON_DIAGNOSTIC_NOTICE,
        "reviewer_confirmed": True,
        "requires_clinical_review": True,
        "result": payload,
    }
    protected_bytes = (
        json.dumps(protected, allow_nan=False, ensure_ascii=True, sort_keys=True) + "\n"
    ).encode("utf-8")
    report = {
        "schema_version": "openmed.cli.multimodal.evidence.v1",
        "task": task.value,
        "notice": NON_DIAGNOSTIC_NOTICE,
        "reviewer_confirmed": True,
        "requires_clinical_review": True,
        "provider": evidence.to_dict(),
        "result_file_digest": hashlib.sha256(protected_bytes).hexdigest(),
    }
    evidence_bytes = (json.dumps(report, sort_keys=True) + "\n").encode("utf-8")
    return protected_bytes, evidence_bytes, report


def _write_private(path: Path, content: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def handle_multimodal_inference(
    args: argparse.Namespace, *, service: LocalSessionService | None = None
) -> int:
    """Execute one injected task; clean owned outputs on failure or interruption.

    Args:
        args: Parsed arguments selecting media, model, consent and new output directory.
        service: Trusted local service for embedding/tests; otherwise load explicit factory.

    Returns:
        Zero after verified service cleanup, protected writes and safe evidence output.

    Raises:
        CliError: Controlled refusal, including cancellation (130) and cleanup failure.
    """
    session = None
    created = False
    closed = False
    failure = None
    report = None
    directory = args.output_directory
    # Python stream suppression is defensive, not a sandbox for trusted plugins.
    with _interruptions(), redirect_stdout(_Discard()), redirect_stderr(_Discard()):
        try:
            if _platform() != "posix":
                raise SessionRefusal("unsupported_permissions")
            if not args.confirm_review:
                raise SessionRefusal("review_required")
            if (
                args.local_task is not LocalTask.PROCESS_DOCUMENT
                and not args.consent_evidence
            ):
                raise SessionRefusal("consent_required")
            if directory.exists() or directory.is_symlink():
                raise SessionRefusal("output_failed")
            artifact = args.model_artifact
            if (
                artifact.is_symlink()
                or not artifact.exists()
                or not (artifact.is_file() or artifact.is_dir())
            ):
                raise SessionRefusal("invalid_artifact")
            media = _read_local(args.media, MAX_INPUT_BYTES)
            consent = (
                _read_local(args.consent_evidence, MAX_CONSENT_BYTES)
                if args.consent_evidence
                else None
            )
            service = (
                service if service is not None else _load_service(args.session_factory)
            )
            if service.local_only is not True:
                raise SessionRefusal("offline_required")
            if args.local_task not in service.tasks:
                raise SessionRefusal("provider_unavailable")
            session = service.open(
                LocalSessionRequest(args.local_task, media, artifact, consent, True)
            )
            result = session.run()
            try:
                protected, evidence, report = _validated_result(
                    result, media, args.local_task
                )
            except SessionRefusal:
                raise
            except Exception:
                raise SessionRefusal("invalid_result") from None
            # Complete service retention/destruction before publishing any result.
            try:
                session.close()
            except Exception:
                raise SessionRefusal("cleanup_failed") from None
            closed = True
            try:
                os.mkdir(directory, mode=0o700)
            except OSError:
                raise SessionRefusal("output_failed") from None
            created = True
            if stat.S_IMODE(directory.stat().st_mode) & 0o077:
                raise SessionRefusal("unsupported_permissions")
            _write_private(directory / "result.json", protected)
            _write_private(directory / "evidence.json", evidence)
            # Only a receipt written after both fsynced files marks a complete pair.
            _write_private(directory / "COMPLETE", b"complete\n")
        except KeyboardInterrupt:
            failure = "cancelled"
        except SessionRefusal as exc:
            failure = (
                exc.code
                if type(exc.code) is str and exc.code in _ERRORS
                else "session_failed"
            )
        except (OSError, UnicodeError):
            failure = "output_failed" if created else "invalid_artifact"
        except (Exception, SystemExit):
            failure = "session_failed"
        finally:
            if failure and session is not None and not closed:
                try:
                    session.cancel()
                except BaseException:
                    failure = "cleanup_failed"
                try:
                    session.close()
                except BaseException:
                    failure = "cleanup_failed"
            if failure and created:
                try:
                    for name in ("COMPLETE", "evidence.json", "result.json"):
                        (directory / name).unlink(missing_ok=True)
                    directory.rmdir()
                except OSError:
                    failure = "cleanup_failed"
    if failure:
        raise _error(failure)
    return emit(args, report, human=NON_DIAGNOSTIC_NOTICE)
