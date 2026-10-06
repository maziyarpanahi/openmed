"""Synthetic metadata canaries and negative controls for access decisions."""

import json
import traceback
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.agent import RunId
from openmed.agent.permissions import (
    AccessEvent,
    AccessEventError,
    AccessTicket,
    AccessTicketError,
    AccessTicketRequest,
    AccessTicketVerifier,
    LocalAccessEventSink,
    MemoryAccessEventSink,
    RecordSelector,
    ToolAction,
    dispatch_with_access_ticket,
)

RUN = RunId("run_" + "a" * 32)
PURPOSE = "purpose:org.example/read-review@1.0.0"
DATA = "data:org.example/medications@1.0.0"
CANARY = "Synthetic María Example; MRN 123456; 患者テスト; secret-token"
SELECTOR = RecordSelector.from_value(
    kind="selector:org.example/record@1.0.0",
    value=CANARY,
    key=b"k" * 32,
)
ACTION = ToolAction("tool:org.example/search@1.0.0", "action:org.example/read@1.0.0")


def request() -> AccessTicketRequest:
    return AccessTicketRequest(RUN, PURPOSE, (DATA,), (SELECTOR,), ACTION)


def ticket() -> AccessTicket:
    return AccessTicket(RUN, PURPOSE, (DATA,), (SELECTOR,), (ACTION,), 100)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("missing", "missing_ticket"),
        ("invalid", "invalid_ticket"),
        ("expired", "expired"),
        ("run", "run_mismatch"),
        ("purpose", "purpose_mismatch"),
        ("projection", "projection_denied"),
        ("selector", "record_selector_denied"),
        ("action", "tool_action_denied"),
        ("request", "invalid_request"),
        ("time", "invalid_timestamp"),
    ],
)
def test_each_denial_records_once_without_dispatch(change: str, reason: str) -> None:
    presented = ticket()
    proposed = request()
    now = 99
    if change == "missing":
        presented = None
    elif change == "invalid":
        presented = CANARY
    elif change == "expired":
        now = 100
    elif change == "run":
        proposed = replace(proposed, run_id=RunId("run_" + "b" * 32))
    elif change == "purpose":
        proposed = replace(proposed, purpose="purpose:org.example/research@1.0.0")
    elif change == "projection":
        proposed = replace(proposed, projection=("data:org.example/other@1.0.0",))
    elif change == "selector":
        proposed = replace(
            proposed,
            record_selectors=(
                RecordSelector.from_value(
                    kind=SELECTOR.kind,
                    value="another synthetic record",
                    key=b"k" * 32,
                ),
            ),
        )
    elif change == "action":
        proposed = replace(
            proposed,
            tool_action=ToolAction(
                ACTION.tool,
                "action:org.example/export@1.0.0",
            ),
        )
    elif change == "request":
        proposed = {"patient": CANARY}
    elif change == "time":
        now = True
    sink = MemoryAccessEventSink()
    calls = []
    with pytest.raises(AccessTicketError) as caught:
        dispatch_with_access_ticket(
            presented,
            proposed,
            AccessTicketVerifier(sink=sink),
            lambda: calls.append(True),
            now=now,
        )
    assert caught.value.code == reason
    assert calls == []
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.outcome == "deny"
    assert event.reason_code == reason
    assert CANARY not in json.dumps(event.to_dict())


def test_allow_records_once_before_dispatch_and_contains_no_selector_metadata() -> None:
    sink = MemoryAccessEventSink()
    verifier = AccessTicketVerifier(sink=sink)

    def dispatch() -> str:
        assert len(sink.events) == 1
        return CANARY

    assert (
        dispatch_with_access_ticket(ticket(), request(), verifier, dispatch, now=99)
        == CANARY
    )
    value = sink.events[0].to_dict()
    assert value == {
        "schema_version": "openmed.agent.access_event.v1",
        "run_id": RUN.value,
        "purpose": PURPOSE,
        "data_classes": [DATA],
        "selector_count": 1,
        "tool": ACTION.tool,
        "action": ACTION.action,
        "outcome": "allow",
        "reason_code": None,
        "timestamp": 99,
    }
    for forbidden in (
        CANARY,
        SELECTOR.kind,
        SELECTOR.digest,
        SELECTOR.digest.split(":")[1],
    ):
        assert forbidden not in json.dumps(value)
        assert forbidden not in repr(sink.events[0])
    # No broad ticket scope is copied to the evidence.
    verifier.verify(
        replace(
            ticket(), permitted_data_classes=(DATA, "data:org.example/other@1.0.0")
        ),
        request(),
        now=99,
    )
    assert sink.events[-1].data_classes == (DATA,)


def test_default_sink_retains_one_event_per_check() -> None:
    verifier = AccessTicketVerifier(clock=lambda: 99)
    verifier.verify(ticket(), request())
    assert len(verifier.sink.events) == 1


def test_clock_failure_records_denial_without_provider_details() -> None:
    def clock() -> int:
        raise RuntimeError(CANARY)

    sink = MemoryAccessEventSink()
    with pytest.raises(AccessTicketError) as caught:
        AccessTicketVerifier(sink=sink, clock=clock).verify(ticket(), request())
    assert len(sink.events) == 1
    assert sink.events[0].timestamp is None
    assert sink.events[0].reason_code == "clock_unavailable"
    assert CANARY not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("denied", [False, True])
def test_sink_failure_blocks_dispatch_and_does_not_retry(denied: bool) -> None:
    class FailingSink:
        calls = 0

        def emit(self, event: AccessEvent) -> None:
            self.calls += 1
            raise RuntimeError(CANARY)

    sink = FailingSink()
    calls = []
    with pytest.raises(AccessEventError) as caught:
        dispatch_with_access_ticket(
            None if denied else ticket(),
            request(),
            AccessTicketVerifier(sink=sink),
            lambda: calls.append(True),
            now=99,
        )
    assert sink.calls == 1
    assert calls == []
    assert caught.value.code == "sink_unavailable"
    assert caught.value.__context__ is None
    assert CANARY not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    "tamper",
    ["value", "delete", "reorder", "duplicate", "truncate", "unknown", "partial"],
)
def test_tampering_breaks_chain(tmp_path: Path, tamper: str) -> None:
    path = tmp_path / "events.jsonl"
    sink = LocalAccessEventSink(path)
    verifier = AccessTicketVerifier(sink=sink)
    for now in (97, 98, 99):
        verifier.verify(ticket(), request(), now=now)
    head = sink.verify_chain()
    lines = path.read_bytes().splitlines(keepends=True)
    if tamper == "value":
        lines[1] = lines[1].replace(b'"timestamp":98', b'"timestamp":80')
    elif tamper == "delete":
        del lines[1]
    elif tamper == "reorder":
        lines.reverse()
    elif tamper == "duplicate":
        lines.insert(1, lines[0])
    elif tamper == "truncate":
        lines.pop()
    elif tamper == "unknown":
        lines[0] = lines[0].replace(b'"event":{', b'"event":{"patient":"synthetic",')
    elif tamper == "partial":
        lines[-1] = lines[-1][:-8]
    path.write_bytes(b"".join(lines))
    with pytest.raises(AccessEventError):
        sink.verify_chain(expected_head=head)
    if tamper != "truncate":
        with pytest.raises(AccessEventError):
            sink.export_counts()
        before = path.read_bytes()
        with pytest.raises(AccessEventError):
            verifier.verify(ticket(), request(), now=99)
        assert path.read_bytes() == before


def test_private_path_and_symlink_failures_have_value_free_errors(
    tmp_path: Path,
) -> None:
    private = tmp_path / "synthetic-private-patient" / "events"
    sink = LocalAccessEventSink(private)
    with pytest.raises(AccessEventError) as caught:
        sink.verify_chain()
    assert caught.value.__context__ is None
    assert str(private) not in repr(sink)
    assert str(private) not in "".join(traceback.format_exception(caught.value))
    target = tmp_path / "target"
    target.write_text("do not change")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(AccessEventError):
        LocalAccessEventSink(link).verify_chain()
    assert target.read_text() == "do not change"
