"""Consent, reference revocation and owned-memory cleanup with synthetic bytes."""

import json
from pathlib import Path

import pytest

from openmed.multimodal.streaming.session import (
    AmbientArtifactKind,
    AmbientReference,
    AmbientSession,
    AmbientSessionError,
    AmbientSessionPolicy,
)

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "fixtures/multimodal/ambient_session_transitions.json"
)
PARTICIPANT = "1" * 32


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


def session(clock=None, **policy):
    clock = clock or Clock()
    result = AmbientSession(
        "0" * 32,
        (PARTICIPANT,),
        policy=AmbientSessionPolicy(revision="a" * 64, **policy),
        clock=clock,
        automatic_expiry=False,
    )
    result.consent(PARTICIPANT)
    result.transition("start")
    return result, clock


def test_shared_transition_fixture():
    fixture = json.loads(FIXTURE.read_text())
    for scenario in fixture["scenarios"]:
        clock = Clock()
        stream = AmbientSession(
            fixture["session_ref"],
            tuple(fixture["participants"]),
            policy=AmbientSessionPolicy(
                revision=fixture["policy_revision"],
                max_lifetime_ms=10_000,
                allow_review_export=True,
            ),
            clock=clock,
            automatic_expiry=False,
        )
        expected = {
            "schema": "openmed.ambient.session.v1",
            "session_ref": fixture["session_ref"],
            "policy_revision": fixture["policy_revision"],
            "state": "awaiting_consent",
            "generation": 0,
            "participants": 2,
            "consents": 0,
            "owned_artifacts": 0,
            "owned_bytes": 0,
            "unaccounted_external_copies": 0,
            "deletion_verified": False,
            "reason": "none",
            "draft_only": True,
            "requires_review": True,
        }
        reference = None
        for index, step in enumerate(scenario["steps"]):
            error = None
            try:
                match step["action"]:
                    case "consent":
                        stream.consent(fixture["participants"][step["participant"]])
                    case "retain":
                        reference = stream.retain(
                            f"{index + 100:032x}",
                            AmbientArtifactKind(step["kind"]),
                            b"\1\2\3",
                        )
                    case "use":
                        stream.use(reference, len)
                    case "participants":
                        stream.change_participants((fixture["participants"][0],))
                    case "expire":
                        clock.value = 10
                        stream.expire()
                    case "withdraw":
                        stream.withdraw()
                    case "delete":
                        stream.delete()
                    case "export":
                        copy = stream.export_for_review(reference)
                        assert copy.draft_only and copy.requires_review
                    case event:
                        stream.transition(event)
            except AmbientSessionError as exc:
                error = str(exc)
            assert error == step.get("error"), (scenario["name"], index)
            expected.update(step["expect"])
            assert json.loads(stream.to_json()) == expected, (scenario["name"], index)
        stream.delete()


@pytest.mark.parametrize("ending", ["withdraw", "delete", "participants", "expire"])
def test_all_content_kinds_overwritten_and_original_references_revoked(ending):
    stream, clock = session(max_lifetime_ms=1000)
    buffers, references = [], []
    for index, kind in enumerate(AmbientArtifactKind):
        reference = stream.retain(f"{index:032x}", kind, b"synthetic private content")
        references.append(reference)
        buffers.append(stream.use(reference, lambda view: view.obj))
    if ending == "participants":
        stream.change_participants(("2" * 32,))
    elif ending == "expire":
        clock.value = 1
        stream.expire()
    else:
        getattr(stream, ending)()
    assert all(not any(buffer) for buffer in buffers)
    for reference in references:
        with pytest.raises(AmbientSessionError):
            stream.use(reference, len)
    assert "synthetic" not in stream.to_json() + repr(stream)


def test_forged_and_cross_session_capabilities_are_rejected():
    stream, _ = session()
    reference = stream.retain("f" * 32, AmbientArtifactKind.DRAFT, b"abc")
    forged = AmbientReference(
        reference.artifact_ref, reference.kind, reference.generation
    )
    other, _ = session()
    foreign = other.retain("f" * 32, AmbientArtifactKind.DRAFT, b"def")
    for invalid in (forged, foreign):
        with pytest.raises(AmbientSessionError, match="stale_reference"):
            stream.use(invalid, len)
    assert stream.use(reference, len) == 3
    stream.delete()
    other.delete()


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
def test_invalid_or_backwards_clock_fails_closed(value):
    stream, clock = session()
    reference = stream.retain("f" * 32, AmbientArtifactKind.AUDIO, b"abc")
    buffer = stream.use(reference, lambda view: view.obj)
    clock.value = value
    assert stream.expire()
    assert not any(buffer)
    assert stream.snapshot()["reason"] == "clock_invalid"


def test_idle_timer_keeps_original_deadline(monkeypatch):
    scheduled = []

    class Timer:
        daemon = False

        def __init__(self, delay, callback):
            self.delay, self.callback = delay, callback
            self.cancelled = False
            scheduled.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr("openmed.multimodal.streaming.session.threading.Timer", Timer)
    clock = Clock()
    stream = AmbientSession(
        "0" * 32,
        (PARTICIPANT,),
        policy=AmbientSessionPolicy(revision="a" * 64, max_lifetime_ms=1000),
        clock=clock,
    )
    clock.value = 0.5
    stream.consent(PARTICIPANT)
    stream.transition("start")
    reference = stream.retain("f" * 32, AmbientArtifactKind.CACHE, b"abc")
    buffer = stream.use(reference, lambda view: view.obj)
    assert len(scheduled) == 1 and scheduled[0].delay == 1
    clock.value = 1
    scheduled[0].callback()
    assert not any(buffer)
    assert stream.snapshot()["state"] == "expired"

    external = AmbientSession(
        "0" * 32,
        (PARTICIPANT,),
        policy=AmbientSessionPolicy(revision="a" * 64),
        clock=clock,
    )
    external.declare_external_copy()
    external.change_participants(("2" * 32,))
    assert external.snapshot()["state"] == "deletion_incomplete"
    assert scheduled[-1].cancelled


def test_export_requires_stop_opt_in_and_never_claims_external_deletion():
    stream, _ = session()
    reference = stream.retain("f" * 32, AmbientArtifactKind.DRAFT, b"draft")
    stream.transition("stop")
    with pytest.raises(AmbientSessionError, match="export_not_permitted"):
        stream.export_for_review(reference)
    stream.delete()
    enabled, _ = session(allow_review_export=True, max_bytes=8)
    audio = enabled.retain("e" * 32, AmbientArtifactKind.AUDIO, b"123")
    note = enabled.retain("f" * 32, AmbientArtifactKind.DRAFT, b"note!")
    enabled.transition("stop")
    with pytest.raises(AmbientSessionError, match="export_not_permitted"):
        enabled.export_for_review(audio)
    copy = enabled.export_for_review(note)
    assert copy.draft_only and copy.requires_review and "note!" not in repr(copy)
    with pytest.raises(AmbientSessionError, match="export_limit"):
        enabled.export_for_review(note)
    enabled.delete()
    assert copy.content == b"note!"
    assert enabled.snapshot()["state"] == "deletion_incomplete"
    assert enabled.snapshot()["deletion_verified"] is False


def test_retention_and_operation_limits_destroy_pending_data():
    stream, _ = session(max_bytes=3)
    reference = stream.retain("f" * 32, AmbientArtifactKind.CACHE, b"abc")
    buffer = stream.use(reference, lambda view: view.obj)
    with pytest.raises(AmbientSessionError, match="retention_limit"):
        stream.retain("e" * 32, AmbientArtifactKind.DRAFT, b"d")
    assert not any(buffer)
    limited, _ = session(max_operations=3)
    limited.retain("f" * 32, AmbientArtifactKind.CACHE, b"abc")
    with pytest.raises(AmbientSessionError, match="operation_limit"):
        limited.transition("pause")
    assert limited.snapshot()["owned_bytes"] == 0


def test_callback_failure_and_expiry_never_return_late_result():
    for failure in (True, False):
        stream, clock = session(max_lifetime_ms=1000)
        reference = stream.retain("f" * 32, AmbientArtifactKind.CACHE, b"abc")

        def reader(view):
            if failure:
                raise RuntimeError("synthetic private text")
            clock.value = 1
            return "late output"

        with pytest.raises(AmbientSessionError) as raised:
            stream.use(reference, reader)
        assert "synthetic" not in str(raised.value)
        assert stream.snapshot()["deletion_verified"] is True


def test_external_copies_hold_participant_change_and_repeated_delete():
    stream, _ = session()
    stream.declare_external_copy()
    stream.change_participants(("2" * 32,))
    before = stream.snapshot()
    assert before["state"] == "deletion_incomplete"
    assert not before["deletion_verified"]
    stream.delete()
    assert stream.snapshot() == before
    with pytest.raises(AmbientSessionError, match="session_closed"):
        stream.consent("2" * 32)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"revision": "private"},
        {"max_bytes": 0},
        {"max_lifetime_ms": True},
        {"max_artifacts": 257},
        {"allow_review_export": 1},
    ],
)
def test_malformed_policy_is_rejected(kwargs):
    with pytest.raises(AmbientSessionError, match="invalid_session_contract"):
        AmbientSessionPolicy(**{"revision": "a" * 64, **kwargs})
