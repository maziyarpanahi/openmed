"""Offline durable access evidence across synthetic tool adapters and restart."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from openmed.agent import RunId
from openmed.agent.permissions import (
    AccessTicket,
    AccessTicketDeniedError,
    AccessTicketRequest,
    AccessTicketVerifier,
    LocalAccessEventSink,
    RecordSelector,
    ToolAction,
    dispatch_with_access_ticket,
)

pytestmark = pytest.mark.integration


def test_durable_read_events_and_aggregate_counts(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    canary = "Synthetic patient Jane Example; MRN-123; synthetic credential"
    selector = RecordSelector.from_value(
        kind="selector:org.example/record@1.0.0",
        value=canary,
        key=b"k" * 32,
    )
    run = RunId("run_" + "a" * 32)
    purpose = "purpose:org.example/review@1.0.0"
    data = "data:org.example/medications@1.0.0"
    calls = []

    def read(tool: str) -> None:
        action = ToolAction(
            f"tool:org.example/{tool}@1.0.0", "action:org.example/read@1.0.0"
        )
        ticket = AccessTicket(run, purpose, (data,), (selector,), (action,), 100)
        request = AccessTicketRequest(run, purpose, (data,), (selector,), action)
        verifier = AccessTicketVerifier(
            sink=LocalAccessEventSink(path), clock=lambda: 99
        )
        assert (
            dispatch_with_access_ticket(
                ticket, request, verifier, lambda: calls.append(tool) or canary
            )
            == canary
        )
        with pytest.raises(AccessTicketDeniedError):
            dispatch_with_access_ticket(
                None, request, verifier, lambda: calls.append("denied")
            )

    # Independent instances serialize appends through the file lock.
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(read, ("fhir-search", "omop-query", "journey-read")))
    restarted = LocalAccessEventSink(path)
    assert len(calls) == 3
    head = restarted.verify_chain()
    assert LocalAccessEventSink(path).verify_chain(expected_head=head) == head
    counts = restarted.export_counts()
    assert counts["event_count"] == counts["selector_count"] == 6
    assert counts["outcomes"] == {"allow": 3, "deny": 3}
    assert counts["denial_reasons"] == {"missing_ticket": 3}
    assert counts["purposes"] == {purpose: 6}
    assert counts["data_classes"] == {data: 6}
    assert [item["count"] for item in counts["tool_actions"]] == [2, 2, 2]
    assert path.stat().st_mode & 0o777 == 0o600
    serialized = path.read_text() + json.dumps(counts)
    for forbidden in (
        canary,
        selector.kind,
        selector.digest,
        selector.digest.split(":")[1],
    ):
        assert forbidden not in serialized
    assert run.value not in json.dumps(counts)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["sequence"] for line in lines] == list(range(1, 7))
