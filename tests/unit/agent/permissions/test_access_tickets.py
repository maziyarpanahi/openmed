"""Offline tests for purpose-bound, single-run data-access tickets."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from openmed.agent.correlation import RunId
from openmed.agent.permissions.access_tickets import (
    ACCESS_TICKET_DENIAL_SCHEMA_VERSION,
    AccessTicket,
    AccessTicketExpiredError,
    AccessTicketProjectionError,
    AccessTicketPurposeMismatchError,
    AccessTicketRequest,
    AccessTicketRequiredError,
    AccessTicketRunMismatchError,
    AccessTicketSelectorError,
    AccessTicketToolActionError,
    AccessTicketValidationError,
    AccessTicketVerifier,
    RecordSelector,
    ToolAction,
    dispatch_with_access_ticket,
)

SELECTOR_KEY = b"local-selector-key-material-32-bytes"
EXPIRES_AT = 2_000_000_000
RUN_ID = RunId.parse(f"run_{'1' * 32}")
OTHER_RUN_ID = RunId.parse(f"run_{'2' * 32}")
PURPOSE = "purpose:org.example/care-summary@1.0.0"
DATA_CLASS = "data:org.example/medications@1.0.0"
OTHER_DATA_CLASS = "data:org.example/demographics@1.0.0"


def _selector(
    value: str = "synthetic-record-001",
    *,
    kind: str = "selector:org.example/record-id@1.0.0",
) -> RecordSelector:
    return RecordSelector.from_value(kind=kind, value=value, key=SELECTOR_KEY)


def _tool_action(
    *,
    tool: str = "tool:org.example/summarize@1.0.0",
    action: str = "action:org.example/read@1.0.0",
) -> ToolAction:
    return ToolAction(tool=tool, action=action)


def _ticket(**overrides: Any) -> AccessTicket:
    values: dict[str, Any] = {
        "run_id": RUN_ID,
        "purpose": PURPOSE,
        "permitted_data_classes": (DATA_CLASS,),
        "record_selectors": (_selector(),),
        "permitted_tool_actions": (_tool_action(),),
        "expires_at": EXPIRES_AT,
    }
    values.update(overrides)
    return AccessTicket(**values)


def _request(**overrides: Any) -> AccessTicketRequest:
    values: dict[str, Any] = {
        "run_id": RUN_ID,
        "purpose": PURPOSE,
        "projection": (DATA_CLASS,),
        "record_selectors": (_selector(),),
        "tool_action": _tool_action(),
    }
    values.update(overrides)
    return AccessTicketRequest(**values)


def _assert_not_dispatched(
    ticket: AccessTicket | None,
    request: AccessTicketRequest,
    error_type: type[Exception],
    *,
    now: int = EXPIRES_AT - 1,
) -> Exception:
    calls = 0

    def dispatch() -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(error_type) as caught:
        dispatch_with_access_ticket(
            ticket,
            request,
            AccessTicketVerifier(),
            dispatch,
            now=now,
        )

    assert calls == 0
    return caught.value


def test_complete_in_scope_request_is_verified_before_dispatch() -> None:
    events: list[str] = []
    verifier = AccessTicketVerifier()
    original_verify = verifier.verify

    def recording_verify(*args: Any, **kwargs: Any) -> AccessTicket:
        events.append("verified")
        return original_verify(*args, **kwargs)

    verifier.verify = recording_verify  # type: ignore[method-assign]

    result = dispatch_with_access_ticket(
        _ticket(),
        _request(),
        verifier,
        lambda: events.append("dispatched") or "ok",
        now=EXPIRES_AT - 1,
    )

    assert result == "ok"
    assert events == ["verified", "dispatched"]


def test_ticket_scope_is_canonical_and_deterministic() -> None:
    first_selector = _selector("synthetic-record-001")
    second_selector = _selector("synthetic-record-002")
    first_action = _tool_action()
    second_action = _tool_action(
        tool="tool:org.example/redact@1.0.0",
        action="action:org.example/transform@1.0.0",
    )

    forward = _ticket(
        permitted_data_classes=(DATA_CLASS, OTHER_DATA_CLASS),
        record_selectors=(first_selector, second_selector),
        permitted_tool_actions=(first_action, second_action),
    )
    reverse = _ticket(
        permitted_data_classes=(OTHER_DATA_CLASS, DATA_CLASS),
        record_selectors=(second_selector, first_selector),
        permitted_tool_actions=(second_action, first_action),
    )

    assert forward == reverse
    assert forward.permitted_data_classes == tuple(
        sorted((DATA_CLASS, OTHER_DATA_CLASS))
    )


def test_projection_may_be_a_strict_subset_of_ticket_scope() -> None:
    ticket = _ticket(permitted_data_classes=(DATA_CLASS, OTHER_DATA_CLASS))

    assert (
        AccessTicketVerifier().verify(
            ticket,
            _request(projection=(DATA_CLASS,)),
            now=EXPIRES_AT - 1,
        )
        == ticket
    )


def test_missing_or_expired_ticket_fails_closed() -> None:
    _assert_not_dispatched(None, _request(), AccessTicketRequiredError)
    for now in (EXPIRES_AT, EXPIRES_AT + 1):
        _assert_not_dispatched(
            _ticket(),
            _request(),
            AccessTicketExpiredError,
            now=now,
        )


def test_ticket_is_non_transferable_between_runs() -> None:
    _assert_not_dispatched(
        _ticket(),
        _request(run_id=OTHER_RUN_ID),
        AccessTicketRunMismatchError,
    )


def test_ticket_cannot_be_repurposed() -> None:
    _assert_not_dispatched(
        _ticket(),
        _request(purpose="purpose:org.example/research-export@1.0.0"),
        AccessTicketPurposeMismatchError,
    )


@pytest.mark.parametrize(
    ("request_overrides", "error_type", "code"),
    [
        (
            {"projection": (OTHER_DATA_CLASS,)},
            AccessTicketProjectionError,
            "projection_denied",
        ),
        (
            {"record_selectors": (_selector("synthetic-record-999"),)},
            AccessTicketSelectorError,
            "record_selector_denied",
        ),
        (
            {"tool_action": _tool_action(action="action:org.example/export@1.0.0")},
            AccessTicketToolActionError,
            "tool_action_denied",
        ),
    ],
)
def test_projection_selector_and_tool_action_fail_closed_outside_ticket(
    request_overrides: dict[str, Any],
    error_type: type[Exception],
    code: str,
) -> None:
    error = _assert_not_dispatched(
        _ticket(),
        _request(**request_overrides),
        error_type,
    )

    assert getattr(error, "code") == code


def test_record_selectors_are_keyed_deterministic_and_opaque() -> None:
    first = _selector("synthetic-record-001")
    repeated = _selector("synthetic-record-001")
    different_key = RecordSelector.from_value(
        kind=first.kind,
        value="synthetic-record-001",
        key=b"different-local-selector-key-32bytes",
    )

    assert first == repeated
    assert first != different_key
    assert first.digest.startswith("hmac-sha256:")
    assert "synthetic-record-001" not in first.digest
    assert first.digest not in repr(first)


def test_denial_evidence_contains_only_fixed_metadata() -> None:
    rejected_selector = _selector("synthetic-rejected-record")
    rejected_action = _tool_action(action="action:org.example/export@1.0.0")
    request = _request(
        record_selectors=(rejected_selector,),
        tool_action=rejected_action,
    )

    error = _assert_not_dispatched(_ticket(), request, AccessTicketSelectorError)
    evidence = getattr(error, "evidence")
    serialized = evidence.to_json()

    assert json.loads(serialized) == {
        "field_name": "record_selectors",
        "reason_code": "record_selector_denied",
        "schema_version": ACCESS_TICKET_DENIAL_SCHEMA_VERSION,
    }
    for rejected_value in (
        rejected_selector.kind,
        rejected_selector.digest,
        rejected_action.tool,
        rejected_action.action,
        request.purpose,
        request.run_id.value,
    ):
        assert rejected_value not in str(error)
        assert rejected_value not in repr(error)
        assert rejected_value not in serialized


def test_ticket_and_request_reprs_do_not_expose_scope() -> None:
    ticket = _ticket()
    request = _request()

    for value in (
        ticket.purpose,
        ticket.run_id.value,
        ticket.permitted_data_classes[0],
        ticket.record_selectors[0].digest,
        request.tool_action.action,
    ):
        assert value not in repr(ticket)
        assert value not in repr(request)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: _ticket(run_id="run_invalid"),
        lambda: _ticket(purpose="free-text-purpose"),
        lambda: _ticket(permitted_data_classes=()),
        lambda: _ticket(permitted_data_classes=(DATA_CLASS, DATA_CLASS)),
        lambda: _ticket(record_selectors=()),
        lambda: _ticket(record_selectors=(_selector(), _selector())),
        lambda: _ticket(permitted_tool_actions=()),
        lambda: _ticket(permitted_tool_actions=(_tool_action(), _tool_action())),
        lambda: _ticket(expires_at=True),
        lambda: _request(projection=()),
        lambda: _request(tool_action="not-a-tool-action"),
        lambda: RecordSelector.from_value(
            kind="selector:org.example/record-id@1.0.0",
            value="synthetic-record-001",
            key=b"too-short",
        ),
    ],
)
def test_malformed_ticket_metadata_fails_closed(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(AccessTicketValidationError):
        factory()


def test_invalid_dispatch_is_rejected_without_invocation() -> None:
    with pytest.raises(AccessTicketValidationError) as caught:
        dispatch_with_access_ticket(
            _ticket(),
            _request(),
            AccessTicketVerifier(),
            None,  # type: ignore[arg-type]
            now=EXPIRES_AT - 1,
        )

    assert caught.value.code == "invalid_dispatch"
    assert caught.value.evidence.to_dict()["field_name"] == "dispatch"


def test_clock_failure_does_not_echo_provider_exception() -> None:
    rejected = "sensitive-clock-provider-detail"

    def failing_clock() -> int:
        raise RuntimeError(rejected)

    with pytest.raises(AccessTicketValidationError) as caught:
        AccessTicketVerifier(clock=failing_clock).verify(_ticket(), _request())

    assert caught.value.code == "clock_unavailable"
    assert rejected not in str(caught.value)
    assert rejected not in repr(caught.value)
    assert rejected not in caught.value.evidence.to_json()


def test_invalid_unicode_selector_value_fails_without_echoing_value() -> None:
    rejected = "synthetic-surrogate-\ud800"

    with pytest.raises(AccessTicketValidationError) as caught:
        RecordSelector.from_value(
            kind="selector:org.example/record-id@1.0.0",
            value=rejected,
            key=SELECTOR_KEY,
        )

    assert caught.value.code == "invalid_selector_value"
    assert rejected not in str(caught.value)
    assert rejected not in repr(caught.value)
