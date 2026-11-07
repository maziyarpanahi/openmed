"""Offline seam-to-ledger-to-PHI-buffer controls with fixed provider outputs."""

import json
import socket

import pytest

from openmed.multimodal.streaming.transcript import (
    RevisionLedger,
    RevisionLedgerError,
    TranscriptToken,
)
from tests.fixtures.multimodal.asr_overlap import (
    PRIVATE,
    SESSION,
    SyntheticScreener,
    reconcile,
    window,
)

pytestmark = pytest.mark.integration


def test_identifier_crossing_overlap_is_screened_only_after_stable_reconciliation(
    monkeypatch,
):
    def deny(*args, **kwargs):
        raise AssertionError("network is outside the overlap contract")

    monkeypatch.setattr(socket, "create_connection", deny)
    result = reconcile(
        (
            window(0, (("Name", 100, 300), ("SYNTHETIC", 1500, 1700))),
            window(1, (("SYNTHETIC", 1500, 1700), ("PATIENT", 2200, 2500))),
        )
    )
    screener = SyntheticScreener()
    released = result.release_with(screener, clock=lambda: 0)
    assert len(released) == 1 and screener.calls == 1
    assert released[0].text == "Name [REDACTED]"
    assert PRIVATE not in result.to_json() + released[0].lineage_json + repr(
        result
    ) + repr(released)
    lineage = json.loads(released[0].lineage_json)
    assert lineage["finalized"] and lineage["revision"] == 3
    assert [op["kind"] for op in lineage["operations"]] == [
        "insert",
        "insert",
        "finalize",
    ]
    assert [item["start_ms"] for item in lineage["tokens"]] == [100, 1500, 2200]
    assert released[0].mappings[-1].redacted


def test_emitted_operations_replay_through_existing_ledger_and_seal_citations():
    result = reconcile()
    ledger = RevisionLedger(SESSION, result.segment_ref, 0)
    for operation in result.operations[:-1]:
        ledger.insert(
            operation.position,
            len(operation.text),
            operation.tokens,
            expected_revision=operation.expected_revision,
        )
    citation = ledger.capture_span("c" * 32, 0, 2)
    final = result.operations[-1]
    ledger.finalize(expected_revision=final.expected_revision)
    mapped = ledger.remap_span(citation)
    assert mapped.valid and mapped.finalized and mapped.start_ms == 100
    with pytest.raises(RevisionLedgerError):
        ledger.insert(
            0,
            3,
            (
                TranscriptToken(
                    token_ref="d" * 32, start=0, end=3, start_ms=0, end_ms=10
                ),
            ),
            expected_revision=ledger.revision,
        )
    released = result.release_with(SyntheticScreener(), clock=lambda: 0)
    assert released[0].text == "No allergy to penicillin reported"
    assert json.loads(released[0].lineage_json)["tokens"] == ledger.to_dict()["tokens"]
