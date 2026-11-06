"""Synthetic observations test the contract without requiring model weights."""

from dataclasses import replace

import pytest

from openmed.multimodal.streaming.speech.diarization import (
    MAX_AUDIO_BYTES,
    AcousticTurn,
    ClusterEvidence,
    DiarizationBatch,
    DiarizationError,
    DiarizationRequest,
    DiarizationSession,
)
from openmed.multimodal.streaming.speech.speaker_aliases import (
    LineageEvent,
    LineageKind,
)

A, B, C, D = (str(n) * 32 for n in range(1, 5))


def request(revision=0):
    return DiarizationRequest(
        pcm=b"\x00\x00" * 16000, revision=revision, offset_samples=16000, request_ref=A
    )


def batch(req, turns=(), lineage=()):
    return DiarizationBatch(
        request_binding=req.binding,
        provenance_digest="a" * 64,
        turns=turns,
        lineage=lineage,
    )


def turn(cluster=A, start=0, end=8000, evidence=ClusterEvidence.UNREVIEWED):
    return AcousticTurn(
        start_sample=start, end_sample=end, cluster_ref=cluster, evidence=evidence
    )


def test_overlap_and_insufficient_evidence_do_not_become_roles():
    session = DiarizationSession(A)
    req = request()
    result = session.apply(
        req, batch(req, (turn(), turn(B, 4000, 16000, ClusterEvidence.UNSTABLE)))
    )
    assert {t["overlap"] for t in result["turns"]} == {"observed"}
    assert {t["alias"] for t in result["turns"]} == {"speaker_0001", "speaker_0002"}
    assert result["turns"][0]["start_sample"] == 16000
    assert result["turns"][-1]["end_sample"] == 32000
    for item in result["turns"]:
        assert item["clinical_role"] is item["confidence"] is item["identity"] is None
    assert all(a["role"] is None for a in result["aliases"]["active"])
    req = request(1)
    empty = session.apply(req, batch(req))
    assert empty["status"] == "insufficient_evidence"
    assert empty["turns"] == []
    assert empty["aliases"]["active"] == []


def test_disjoint_turns_and_same_cluster_do_not_prove_no_overlap():
    req = request()
    result = DiarizationSession(A).apply(
        req, batch(req, (turn(), turn(A, 4000), turn(B, 8000, 16000)))
    )
    assert {t["overlap"] for t in result["turns"]} == {"unknown"}


def test_explicit_relabel_merge_split_preserve_alias_contract_without_roles():
    session = DiarizationSession(A)
    req = request()
    first = session.apply(req, batch(req, (turn(A), turn(B, 8000, 16000))))
    req = request(1)
    relabel = LineageEvent(kind=LineageKind.RELABEL, parents=(A,), children=(C,))
    second = session.apply(req, batch(req, (turn(C), turn(B, 8000, 16000)), (relabel,)))
    assert second["turns"][0]["alias"] == first["turns"][0]["alias"]
    req = request(2)
    merge = LineageEvent(kind=LineageKind.MERGE, parents=(C, B), children=(D,))
    third = session.apply(req, batch(req, (turn(D),), (merge,)))
    assert third["turns"][0]["alias"] == "speaker_0001"
    req = request(3)
    e, f = "e" * 32, "f" * 32
    split = LineageEvent(kind=LineageKind.SPLIT, parents=(D,), children=(e, f))
    fourth = session.apply(req, batch(req, (turn(e), turn(f)), (split,)))
    assert {t["alias"] for t in fourth["turns"]} == {"speaker_0003", "speaker_0004"}
    assert all(t["clinical_role"] is None for t in fourth["turns"])


def test_failed_binding_or_transition_does_not_consume_revision():
    req = request()
    session = DiarizationSession(A)
    with pytest.raises(DiarizationError, match="invalid_diarization"):
        session.apply(req, batch(replace(req, pcm=b"\x01\x00" * 16000), (turn(),)))
    with pytest.raises(DiarizationError):
        session.apply(req, batch(req, (turn(end=16001),)))
    invalid = LineageEvent(kind=LineageKind.RELABEL, parents=(B,), children=(A,))
    with pytest.raises(DiarizationError):
        session.apply(req, batch(req, (turn(),), (invalid,)))
    assert (
        session.apply(req, batch(req, (turn(),)))["turns"][0]["alias"] == "speaker_0001"
    )
    with pytest.raises(DiarizationError):
        session.apply(req, batch(req))
    session.close()
    with pytest.raises(DiarizationError):
        session.apply(request(1), batch(request(1)))


@pytest.mark.parametrize(
    "changes",
    [
        {"pcm": b""},
        {"pcm": b"x"},
        {"pcm": b"xx" * (MAX_AUDIO_BYTES // 2 + 1)},
        {"revision": True},
        {"offset_samples": -1},
        {"request_ref": "person@example.test"},
    ],
)
def test_input_bounds_and_private_repr(changes):
    with pytest.raises(DiarizationError) as exc:
        replace(request(), **changes)
    assert str(exc.value) == "invalid_diarization"
    assert "pcm=" not in repr(request())


def test_complete_batch_bounds():
    req = request()
    with pytest.raises(DiarizationError):
        batch(req, (turn(),) * 1025)
    with pytest.raises(DiarizationError):
        batch(req, tuple(turn(f"{i:032x}") for i in range(33)))
    with pytest.raises(DiarizationError):
        replace(turn(), evidence="confirmed")
    with pytest.raises(DiarizationError):
        replace(turn(), start_sample=True)
