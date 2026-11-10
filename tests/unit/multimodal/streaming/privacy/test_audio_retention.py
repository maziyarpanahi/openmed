"""Synthetic owned-buffer, file-identity and crash-accounting regressions."""

import json
import os
import socket
from pathlib import Path

import pytest

from openmed.multimodal.streaming.privacy import audio_retention as module
from openmed.multimodal.streaming.privacy.audio_retention import (
    ArtifactKind,
    AudioRetentionError,
    AudioRetentionPolicy,
    AudioRetentionSession,
    AudioStage,
    DestructionAction,
    ExpectedAudioArtifact,
    SessionEnd,
)

SESSION, FIRST, SECOND, THIRD = (f"{value:032x}" for value in range(4))
SYNTHETIC = b"SYNTHETIC-AUDIO-PRIVATE-VALUE"
POSIX = pytest.mark.skipif(os.name != "posix", reason="POSIX file enrollment contract")


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def expected(ref=FIRST, kind=ArtifactKind.BUFFER):
    return ExpectedAudioArtifact(ref, kind)


def session(*items, clock=None, directory=None, **limits):
    return AudioRetentionSession(
        SESSION,
        items or (expected(),),
        policy=AudioRetentionPolicy(**limits),
        clock=clock or Clock(),
        artifact_directory=directory,
        automatic_expiry=False,
    )


def file_fixture(tmp_path, ref=FIRST):
    root = tmp_path / "private-audio"
    root.mkdir(mode=0o700, exist_ok=True)
    path = root / (ref + ".audio")
    path.write_bytes(SYNTHETIC)
    path.chmod(0o600)
    return root, path


def test_buffer_lifecycle_overwrites_owned_views_and_preserves_caller_ownership():
    source = bytearray(SYNTHETIC)
    target = session()
    target.capture(FIRST, source)
    owned = target._artifacts[FIRST].buffer
    view = memoryview(owned)
    target.advance(FIRST, AudioStage.BUFFERING)
    seen = []

    def provider(audio):
        assert audio.readonly
        seen.append(bytes(audio))
        return "protected-result"

    assert target.infer(FIRST, provider) == "protected-result"
    report = target.finish()
    assert report.passed and seen == [SYNTHETIC]
    assert bytes(view) == bytes(len(SYNTHETIC))
    assert source == SYNTHETIC
    assert report.receipts[0].action is DestructionAction.BUFFER_OVERWRITTEN
    assert report.receipts[0].stage_history == (
        AudioStage.CAPTURE,
        AudioStage.BUFFERING,
        AudioStage.INFERENCE,
    )
    assert target.finish() == report
    view.release()


@pytest.mark.parametrize("reason", list(SessionEnd))
def test_all_terminal_reasons_destroy_and_prevent_reuse(reason):
    target = session()
    target.capture(FIRST, SYNTHETIC)
    owned = target._artifacts[FIRST].buffer
    assert target.finish(reason).passed
    assert not any(owned)
    with pytest.raises(AudioRetentionError, match="closed"):
        target.capture(FIRST, SYNTHETIC)
    assert target.finish(SessionEnd.CANCELLED).session_end is reason


def test_context_failure_destroys_even_when_caller_raises():
    target = session()
    with pytest.raises(RuntimeError), target:
        target.capture(FIRST, SYNTHETIC)
        owned = target._artifacts[FIRST].buffer
        raise RuntimeError("synthetic caller failure")
    assert not any(owned)
    assert target.audit().session_end is SessionEnd.FAILURE
    assert target.audit().passed


def test_normal_context_exit_finalizes():
    with session() as target:
        target.capture(FIRST, SYNTHETIC)
    assert target.audit().passed
    assert target.audit().session_end is SessionEnd.FINALIZED


def test_empty_unenrolled_and_external_artifacts_fail_audit():
    target = session(expected(), expected(SECOND, ArtifactKind.EXTERNAL))
    assert not target.audit().passed
    report = target.finish()
    assert not report.passed
    assert len(report.receipts) == 2
    assert all(row.action is DestructionAction.UNACCOUNTED for row in report.receipts)


def test_partial_inventory_and_unknown_capture_fail_without_losing_owned_audio():
    target = session()
    target.capture(FIRST, SYNTHETIC)
    owned = target._artifacts[FIRST].buffer
    with pytest.raises(AudioRetentionError, match="unexpected_audio"):
        target.capture(SECOND, SYNTHETIC)
    assert not any(owned)
    assert "unexpected_artifact" in target.audit().blockers


def test_overflow_destroys_prior_buffers_and_accounts_for_missing_capture():
    target = session(expected(), expected(SECOND), max_bytes=len(SYNTHETIC))
    target.capture(FIRST, SYNTHETIC)
    owned = target._artifacts[FIRST].buffer
    with pytest.raises(AudioRetentionError, match="byte_limit"):
        target.capture(SECOND, SYNTHETIC)
    assert not any(owned)
    assert set(target.audit().blockers) == {"byte_limit", "unaccounted_artifacts"}


def test_operation_limit_is_terminal_and_cannot_be_reset():
    target = session(max_operations=1)
    target.capture(FIRST, SYNTHETIC)
    with pytest.raises(AudioRetentionError, match="operation_limit"):
        target.advance(FIRST, AudioStage.BUFFERING)
    assert "operation_limit" in target.audit().blockers
    assert target.audit().receipts[0].action is DestructionAction.BUFFER_OVERWRITTEN


def test_deadline_not_extended_by_stages_and_finalize_honors_expiry():
    clock = Clock()
    target = session(clock=clock, max_lifetime_ms=100)
    target.capture(FIRST, SYNTHETIC)
    clock.value = 0.05
    target.advance(FIRST, AudioStage.BUFFERING)
    clock.value = 0.1
    assert target.finish().session_end is SessionEnd.EXPIRED
    assert target.audit().passed


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), "private", True])
def test_invalid_clock_closes_and_cleans(value):
    clock = Clock()
    target = session(clock=clock)
    target.capture(FIRST, SYNTHETIC)
    clock.value = value
    assert target.expire()
    assert target.audit().session_end is SessionEnd.CLOCK_INVALID
    assert target.audit().passed


def test_clock_rollback_and_clock_exception_are_fail_closed():
    clock = Clock()
    target = session(clock=clock)
    target.capture(FIRST, SYNTHETIC)
    clock.value = 10.0
    target.expire()
    clock.value = 9.0
    assert target.expire()
    assert target.audit().session_end is SessionEnd.CLOCK_INVALID

    def broken_clock():
        raise RuntimeError(SYNTHETIC.decode())

    with pytest.raises(AudioRetentionError) as caught:
        session(clock=broken_clock)
    assert caught.value.__context__ is None
    assert SYNTHETIC.decode() not in str(caught.value)


def test_automatic_timer_expires_idle_buffers_without_real_sleep(monkeypatch):
    timers = []

    class Timer:
        def __init__(self, delay, callback):
            self.delay, self.callback, self.cancelled = delay, callback, False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(module.threading, "Timer", Timer)
    clock = Clock()
    target = AudioRetentionSession(
        SESSION,
        (expected(),),
        clock=clock,
        policy=AudioRetentionPolicy(max_lifetime_ms=100),
    )
    target.capture(FIRST, SYNTHETIC)
    owned = target._artifacts[FIRST].buffer
    clock.value = 0.05
    timers[-1].callback()
    assert target._end is None and len(timers) == 2
    clock.value = 0.1
    timers[-1].callback()
    assert target.audit().session_end is SessionEnd.EXPIRED
    assert not any(owned) and timers[-1].cancelled


def test_cancel_during_inference_zeroes_borrowed_view_and_discards_result():
    target = session()
    target.capture(FIRST, SYNTHETIC)

    def provider(audio):
        target.finish(SessionEnd.CANCELLED)
        assert bytes(audio) == bytes(len(SYNTHETIC))
        return "must-not-release"

    with pytest.raises(AudioRetentionError, match="discarded"):
        target.infer(FIRST, provider)
    assert target.audit().passed


def test_expiry_during_inference_discards_result():
    clock = Clock()
    target = session(clock=clock, max_lifetime_ms=1)
    target.capture(FIRST, SYNTHETIC)

    def provider(audio):
        clock.value = 1.0
        return bytes(audio)

    with pytest.raises(AudioRetentionError, match="discarded"):
        target.infer(FIRST, provider)
    assert target.audit().session_end is SessionEnd.EXPIRED


def test_provider_errors_are_sanitized_without_original_exception_context():
    target = session()
    target.capture(FIRST, SYNTHETIC)

    def provider(audio):
        raise RuntimeError(bytes(audio).decode())

    with pytest.raises(AudioRetentionError) as caught:
        target.infer(FIRST, provider)
    assert str(caught.value) == "audio_provider_failed"
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert target.audit().passed


def test_failed_overwrite_is_recorded_and_retry_preserves_failed_receipt(monkeypatch):
    target = session()
    target.capture(FIRST, SYNTHETIC)
    original = module._overwrite
    monkeypatch.setattr(module, "_overwrite", lambda buffer: False)
    first = target.finish()
    assert not first.passed and first.receipts[0].action is DestructionAction.FAILED
    for _ in range(5):
        assert target.finish() == first
    monkeypatch.setattr(module, "_overwrite", original)
    second = target.finish()
    assert second.passed and len(second.receipts) == 2
    assert second.receipts[0] == first.receipts[0]


def test_reports_are_deterministic_value_free_and_offline(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("network call")

    monkeypatch.setattr(socket, "socket", no_network)
    reports = []
    for payload in (SYNTHETIC, b"different synthetic audio"):
        target = session()
        target.capture(FIRST, payload)
        assert SYNTHETIC.decode() not in repr(target)
        assert SYNTHETIC.decode() not in target.checkpoint()
        reports.append(target.finish().to_json())
    assert reports[0] == reports[1]
    assert SYNTHETIC.decode() not in reports[0]


@pytest.mark.parametrize(
    "limits",
    [
        {"max_artifacts": 0},
        {"max_bytes": True},
        {"max_lifetime_ms": 0},
        {"max_operations": -1},
    ],
)
def test_invalid_policy_rejected(limits):
    with pytest.raises(AudioRetentionError):
        AudioRetentionPolicy(**limits)


def test_inventory_validation_duplicate_wrong_type_and_cross_session_refs():
    for items in ((expected(), expected()), ("private",), ()):
        with pytest.raises(AudioRetentionError):
            AudioRetentionSession(SESSION, items, automatic_expiry=False)
    target = session()
    with pytest.raises(AudioRetentionError):
        target.capture("../private", SYNTHETIC)
    assert SYNTHETIC.decode() not in target.finish().to_json()
    a, b = (
        session(),
        AudioRetentionSession(SECOND, (expected(),), automatic_expiry=False),
    )
    assert a.audit().binding_digest != b.audit().binding_digest
    a.close()
    b.close()


@POSIX
def test_file_enrollment_removes_exact_artifact_and_reports_unlink(tmp_path):
    root, path = file_fixture(tmp_path)
    target = session(expected(kind=ArtifactKind.FILE), directory=root)
    target.enroll_file(FIRST)
    target.advance(FIRST, AudioStage.CACHE)
    report = target.finish()
    assert report.passed and not path.exists()
    assert root.is_dir() and list(root.iterdir()) == []
    assert report.receipts[0].action is DestructionAction.FILE_UNLINKED
    assert str(root) not in target.checkpoint() + report.to_json() + repr(target)
    assert target.close().passed


@POSIX
def test_unexpected_file_is_preserved_and_blocks_audit(tmp_path):
    root, path = file_fixture(tmp_path)
    unexpected = root / "unrelated-file"
    unexpected.write_bytes(b"synthetic unrelated content")
    target = session(expected(kind=ArtifactKind.FILE), directory=root)
    target.enroll_file(FIRST)
    report = target.finish()
    assert not report.passed and not path.exists() and unexpected.exists()
    assert "remaining_directory_entries" in report.blockers
    assert not target.close().passed
    assert not target.audit().passed


@POSIX
@pytest.mark.parametrize(
    "change", ["replace", "symlink", "hardlink", "edit", "chmod", "missing"]
)
def test_changed_enrolled_file_is_never_deleted_or_counted_as_destroyed(
    tmp_path, change
):
    root, path = file_fixture(tmp_path)
    target = session(expected(kind=ArtifactKind.FILE), directory=root)
    target.enroll_file(FIRST)
    outside = tmp_path / "outside"
    outside.write_bytes(b"synthetic outside data")
    if change in {"replace", "symlink", "missing"}:
        path.rename(tmp_path / "original")
    if change == "replace":
        path.write_bytes(b"replacement")
        path.chmod(0o600)
    elif change == "symlink":
        path.symlink_to(outside)
    elif change == "hardlink":
        os.link(path, tmp_path / "extra-link")
    elif change == "edit":
        path.write_bytes(b"changed")
    elif change == "chmod":
        path.chmod(0o644)
    report = target.finish()
    assert not report.passed
    assert report.receipts[0].action is DestructionAction.FAILED
    assert outside.read_bytes() == b"synthetic outside data"
    if change != "missing":
        assert path.exists()
    target.close()


@POSIX
def test_unlink_failure_can_be_retried_after_quarantine(tmp_path, monkeypatch):
    root, path = file_fixture(tmp_path)
    target = session(expected(kind=ArtifactKind.FILE), directory=root)
    target.enroll_file(FIRST)
    original = module.os.unlink

    def deny(*args, **kwargs):
        raise OSError(str(root) + SYNTHETIC.decode())

    monkeypatch.setattr(module.os, "unlink", deny)
    report = target.finish()
    assert not report.passed and not path.exists()
    assert (root / (FIRST + ".retiring")).exists()
    assert SYNTHETIC.decode() not in report.to_json()
    monkeypatch.setattr(module.os, "unlink", original)
    assert target.finish().passed
    assert len(target.audit().receipts) == 2
    target.close()


@POSIX
def test_preexisting_quarantine_is_never_overwritten(tmp_path):
    root, path = file_fixture(tmp_path)
    target = session(expected(kind=ArtifactKind.FILE), directory=root)
    target.enroll_file(FIRST)
    quarantine = root / (FIRST + ".retiring")
    quarantine.write_bytes(b"unrelated")
    assert not target.finish().passed
    assert path.exists() and quarantine.read_bytes() == b"unrelated"
    target.close()


@POSIX
def test_insecure_symlinked_and_foreign_directories_rejected(tmp_path):
    root, _ = file_fixture(tmp_path)
    link = tmp_path / "linked"
    link.symlink_to(root, target_is_directory=True)
    for path in (link, tmp_path / "missing"):
        with pytest.raises(AudioRetentionError) as caught:
            session(directory=path)
        assert str(path) not in str(caught.value)
        assert caught.value.__context__ is None
    root.chmod(0o755)
    with pytest.raises(AudioRetentionError):
        session(directory=root)


@POSIX
def test_directory_permission_change_blocks_removal(tmp_path):
    root, path = file_fixture(tmp_path)
    target = session(expected(kind=ArtifactKind.FILE), directory=root)
    target.enroll_file(FIRST)
    root.chmod(0o755)
    assert not target.finish().passed and path.exists()
    root.chmod(0o700)
    assert target.finish().passed
    target.close()


@POSIX
def test_invalid_file_enrollment_fails_closed_with_sanitized_error(tmp_path):
    root, path = file_fixture(tmp_path)
    path.chmod(0o644)
    target = session(expected(kind=ArtifactKind.FILE), directory=root)
    with pytest.raises(AudioRetentionError) as caught:
        target.enroll_file(FIRST)
    assert caught.value.__context__ is None
    assert path.exists() and not target.audit().passed
    target.close()


@POSIX
def test_crash_recovery_destroys_original_file_but_cannot_certify_lost_memory(tmp_path):
    root, path = file_fixture(tmp_path)
    target = session(expected(kind=ArtifactKind.FILE), expected(SECOND), directory=root)
    target.enroll_file(FIRST)
    target.capture(SECOND, SYNTHETIC)
    checkpoint = target.checkpoint()
    # Simulate process loss: the recovery object has no reference to old memory.
    recovered = AudioRetentionSession.recover(checkpoint, artifact_directory=root)
    assert not path.exists()
    report = recovered.audit()
    assert not report.passed and report.session_end is SessionEnd.RECOVERED_CRASH
    actions = {row.artifact_ref: row.action for row in report.receipts}
    assert actions[FIRST] is DestructionAction.FILE_UNLINKED
    assert actions[SECOND] is DestructionAction.UNACCOUNTED
    recovered.close()
    target.close()


@POSIX
def test_file_only_recovery_and_quarantine_recovery(tmp_path, monkeypatch):
    root, path = file_fixture(tmp_path)
    target = session(expected(kind=ArtifactKind.FILE), directory=root)
    target.enroll_file(FIRST)
    original = module.os.unlink
    monkeypatch.setattr(
        module.os, "unlink", lambda *a, **k: (_ for _ in ()).throw(OSError())
    )
    assert not target.finish().passed
    checkpoint = target.checkpoint()
    monkeypatch.setattr(module.os, "unlink", original)
    recovered = AudioRetentionSession.recover(checkpoint, artifact_directory=root)
    assert recovered.audit().passed and not path.exists()
    recovered.close()
    target.close()


@POSIX
@pytest.mark.parametrize(
    "fault", ["directory", "identity", "kind", "budget", "extra", "invalid_json"]
)
def test_invalid_recovery_never_deletes_partially_parsed_files(tmp_path, fault):
    root, path = file_fixture(tmp_path)
    target = session(expected(kind=ArtifactKind.FILE), directory=root)
    target.enroll_file(FIRST)
    data = json.loads(target.checkpoint())
    if fault == "directory":
        data["directory_identity"][1] += 1
    elif fault == "identity":
        data["files"][FIRST]["identity"] = [1]
    elif fault == "kind":
        data["expected"][0]["kind"] = "buffer"
    elif fault == "budget":
        data["policy"]["max_bytes"] = 1
    elif fault == "extra":
        data["files"][SECOND] = data["files"][FIRST]
    checkpoint = "{" if fault == "invalid_json" else json.dumps(data)
    with pytest.raises(AudioRetentionError) as caught:
        AudioRetentionSession.recover(checkpoint, artifact_directory=root)
    assert caught.value.__context__ is None
    assert path.read_bytes() == SYNTHETIC
    target.close()


def test_recovery_never_accepts_crashed_memory_as_destroyed():
    target = session()
    target.capture(FIRST, SYNTHETIC)
    recovered = AudioRetentionSession.recover(target.checkpoint())
    assert not recovered.audit().passed
    assert recovered.audit().receipts[0].action is DestructionAction.UNACCOUNTED
    with pytest.raises(AudioRetentionError, match="closed"):
        recovered.capture(FIRST, SYNTHETIC)
    recovered.close()
    target.close()


def test_non_posix_file_mode_fails_closed_but_memory_works(monkeypatch, tmp_path):
    monkeypatch.setattr(module.os, "supports_dir_fd", set())
    with pytest.raises(AudioRetentionError):
        session(directory=tmp_path)
    with session() as target:
        target.capture(FIRST, SYNTHETIC)
    assert target.audit().passed


@POSIX
def test_recovery_preserves_lifecycle_history_and_known_inventory_violations(
    tmp_path, monkeypatch
):
    root, path = file_fixture(tmp_path)
    target = session(expected(kind=ArtifactKind.FILE), directory=root)
    target.enroll_file(FIRST)
    target.advance(FIRST, AudioStage.BUFFERING)
    target.advance(FIRST, AudioStage.CACHE)
    original = module.os.unlink
    monkeypatch.setattr(
        module.os, "unlink", lambda *a, **k: (_ for _ in ()).throw(OSError())
    )
    with pytest.raises(AudioRetentionError):
        target.capture(SECOND, SYNTHETIC)
    checkpoint = target.checkpoint()
    monkeypatch.setattr(module.os, "unlink", original)
    recovered = AudioRetentionSession.recover(checkpoint, artifact_directory=root)
    report = recovered.audit()
    assert not path.exists() and not report.passed
    assert "unexpected_artifact" in report.blockers
    assert report.receipts[0].stage_history == (
        AudioStage.CAPTURE,
        AudioStage.BUFFERING,
        AudioStage.CACHE,
        AudioStage.CRASH,
    )
    recovered.close()
    target.close()


@pytest.mark.parametrize("field", ["violations", "histories"])
def test_recovery_rejects_uncontrolled_codes_without_sensitive_exception_context(field):
    target = session()
    data = json.loads(target.checkpoint())
    data[field] = (
        [SYNTHETIC.decode()] if field == "violations" else {FIRST: [SYNTHETIC.decode()]}
    )
    with pytest.raises(AudioRetentionError) as caught:
        AudioRetentionSession.recover(json.dumps(data))
    assert caught.value.__context__ is None
    assert SYNTHETIC.decode() not in str(caught.value)
    target.close()
