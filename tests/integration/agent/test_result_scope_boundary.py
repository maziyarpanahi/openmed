"""Synthetic tool-to-next-step authorization and independent privacy gates."""

from dataclasses import replace

import pytest

from openmed.agent.permissions import (
    AccessTicketVerifier,
    ResultQuarantinedError,
    ToolResultPage,
    dispatch_with_authorized_results,
)
from openmed.guard import PreflightBlockedError, preflight_context
from tests.fixtures.agent.result_scope import binding, pages, record, schema, selector

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "attack", ["patient", "encounter", "later_page", "nested", "extra"]
)
def test_hostile_synthetic_tool_cannot_feed_next_step_or_trace(attack):
    ticket, request, scope = binding()
    next_steps = []
    trace = []

    def read():
        output = pages(scope)
        if attack in {"patient", "encounter"}:
            wrong = replace(scope, **{attack: selector(attack, "synthetic-other")})
            return (ToolResultPage(scope, (record(wrong),)),)
        if attack == "later_page":
            return (output[0], replace(output[1], scope=None))
        if attack == "nested":
            parent = replace(
                record(scope), children=(replace(record(scope, 2), scope=None),)
            )
            return (ToolResultPage(scope, (parent,)),)
        output[1].records[0].fields["observations"][0]["extra"] = "synthetic secret"
        return output

    with pytest.raises(ResultQuarantinedError) as caught:
        dispatch_with_authorized_results(
            ticket,
            request,
            AccessTicketVerifier(clock=lambda: 1),
            scope=scope,
            schema=schema(),
            read=read,
            consume=lambda output: next_steps.append(output),
        )
    trace.append(caught.value.to_dict())
    assert next_steps == []
    assert trace == [
        {
            "schema_version": "openmed.agent.result_quarantine.v1",
            "reason_code": caught.value.code.value,
        }
    ]
    assert "synthetic secret" not in repr(trace)


def test_benign_control_preserves_original_evidence_and_reaches_next_step():
    ticket, request, scope = binding()
    source = pages(scope)
    received = []

    def consume(output):
        received.extend(output)
        return "accepted"

    assert (
        dispatch_with_authorized_results(
            ticket,
            request,
            AccessTicketVerifier(clock=lambda: 1),
            scope=scope,
            schema=schema(),
            read=lambda: source,
            consume=consume,
        )
        == "accepted"
    )
    assert len(received) == 2
    assert received[1].records[0].evidence is source[1].records[0].evidence
    assert received[0].records[0].children[0].evidence is (
        source[0].records[0].children[0].evidence
    )


def test_authorized_content_still_requires_independent_privacy_scanning():
    ticket, request, scope = binding()
    source = record(scope)
    email = "synthetic-subject" + "@example.invalid"
    source.fields["observations"][0]["text"] = email
    scanner_calls = []

    def scanner(text):
        scanner_calls.append(text)
        if email in text:
            start = text.index(email)
            return [{"category": "EMAIL", "start": start, "end": start + len(email)}]
        return []

    def consume(output):
        assert output[0].records[0].evidence is source.evidence
        return preflight_context(
            {}, tool_outputs=output[0].records[0].fields, scanner=scanner
        )

    with pytest.raises(PreflightBlockedError) as caught:
        dispatch_with_authorized_results(
            ticket,
            request,
            AccessTicketVerifier(clock=lambda: 1),
            scope=scope,
            schema=schema(),
            read=lambda: (ToolResultPage(scope, (source,)),),
            consume=consume,
        )
    assert email in scanner_calls
    assert email not in repr(caught.value.report.to_dict())
    assert caught.value.result.tool_outputs is None


def test_wrong_scope_fails_even_when_privacy_scanner_would_find_nothing():
    ticket, request, scope = binding()
    wrong = replace(scope, patient=selector("patient", "synthetic-other"))
    calls = []

    def consume(output):
        calls.append("scan")
        return preflight_context(
            {}, tool_outputs=output[0].records[0].fields, scanner=lambda _: []
        )

    with pytest.raises(ResultQuarantinedError):
        dispatch_with_authorized_results(
            ticket,
            request,
            AccessTicketVerifier(clock=lambda: 1),
            scope=scope,
            schema=schema(),
            read=lambda: (ToolResultPage(scope, (record(wrong),)),),
            consume=consume,
        )
    assert calls == []
