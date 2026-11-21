"""Confirm local session withdrawal over both concrete owner types."""

import json

import pytest

from openmed.compliance.withdrawal import (
    AmbientWithdrawalParticipant,
    ParticipantCode,
    WithdrawalCoordinator,
)
from openmed.multimodal.streaming.session import (
    AmbientArtifactKind,
    AmbientSession,
    AmbientSessionPolicy,
)
from openmed.multimodal.voice_embeddings import (
    VoiceEmbeddingError,
    VoiceEmbeddingSession,
)

SUBJECT = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64


def live_session(number):
    participant = "f" * 32
    session = AmbientSession(
        f"{number:032x}",
        (participant,),
        policy=AmbientSessionPolicy(revision="a" * 64),
        automatic_expiry=False,
    )
    session.consent(participant)
    session.transition("start")
    session.retain("d" * 32, AmbientArtifactKind.AUDIO, b"private-synthetic-audio")
    return session


@pytest.mark.parametrize("external", [True, False])
def test_withdraws_only_matching_objects_and_preserves_external_status(external):
    sessions = AmbientWithdrawalParticipant(ParticipantCode.AMBIENT_SESSIONS)
    voices = AmbientWithdrawalParticipant(ParticipantCode.VOICE_EMBEDDINGS)
    coordinator = WithdrawalCoordinator({sessions.code: sessions, voices.code: voices})
    first, second, other = (live_session(i) for i in range(3))
    if external:
        first.declare_external_copy()
    for obj in (first, second, second):
        sessions.register(SUBJECT, obj)
    sessions.register(OTHER, other)
    voice = VoiceEmbeddingSession()
    handle = voice.add([1.0, 2.0])
    voices.register(SUBJECT, voice)
    plan = coordinator.plan(SUBJECT)
    receipt = coordinator.withdraw(SUBJECT, confirmation=plan.confirmation_token)
    assert receipt.state == ("partial" if external else "success")
    assert first.snapshot()["state"] == (
        "deletion_incomplete" if external else "withdrawn"
    )
    assert second.snapshot()["owned_bytes"] == first.snapshot()["owned_bytes"] == 0
    assert other.snapshot()["state"] == "capturing"
    assert [p.applied for p in receipt.participants] == [2, 1]
    with pytest.raises(VoiceEmbeddingError):
        handle.similarity(handle)
    assert voices.receipts(SUBJECT) == (
        {"state": "withdrawal", "destroyed_count": 1, "verified": True},
    )
    assert sessions.receipts(SUBJECT)[0]["external_copies"] == int(external)
    assert "private-" not in json.dumps([receipt.to_dict(), sessions.receipts(SUBJECT)])
    with pytest.raises(ValueError, match="subject_withdrawn"):
        sessions.register(SUBJECT, other)
    assert (
        coordinator.withdraw(
            SUBJECT, confirmation=coordinator.plan(SUBJECT).confirmation_token
        ).state
        == receipt.state
    )


def test_changed_selection_refuses_stale_plan():
    sessions = AmbientWithdrawalParticipant(ParticipantCode.AMBIENT_SESSIONS)
    session = live_session(1)
    plan = sessions.plan(SUBJECT)
    sessions.register(SUBJECT, session)
    with pytest.raises(ValueError, match="ambient_selection_changed"):
        sessions.apply(SUBJECT, plan)
    assert session.snapshot()["state"] == "capturing"
