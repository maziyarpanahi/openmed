"""Focused tests for the deterministic agent-context preflight gate."""

from __future__ import annotations

import json

import pytest

from openmed.guard import (
    REDACT_THEN_CONTINUE_POLICY,
    PreflightBlockedError,
    PreflightScanError,
    inspect_context,
    preflight_context,
)


def _synthetic_email() -> str:
    return "subject-" + "example" + "@example.invalid"


def _synthetic_mrn() -> str:
    return "MRN: " + "1234567"


def test_fail_closed_returns_safe_offsets_and_does_not_echo_source() -> None:
    email = _synthetic_email()
    context = {"messages": ["Please contact " + email]}

    with pytest.raises(PreflightBlockedError) as raised:
        preflight_context(context)

    error = raised.value
    finding = error.result.findings[0]
    report = error.report.to_dict()

    assert error.result.blocked is True
    assert error.result.context is None
    assert finding.category == "EMAIL"
    assert finding.offsets == (
        context["messages"][0].index(email),
        context["messages"][0].index(email) + len(email),
    )
    assert finding.channel == "context"
    assert finding.payload_index == 0
    assert email not in str(error)
    assert email not in json.dumps(report, sort_keys=True)
    assert report["finding_categories"] == ["EMAIL"]


def test_inspect_context_can_observe_a_block_without_raising() -> None:
    email = _synthetic_email()

    result = inspect_context("value=" + email)

    assert result.allowed is False
    assert result.blocked is True
    assert result.report.redacted is False
    assert result.to_dict() == result.report.to_dict()


def test_redact_then_continue_handles_context_and_tool_output_payloads() -> None:
    email = _synthetic_email()
    mrn = _synthetic_mrn()
    context = {"messages": ["Contact " + email]}
    tool_outputs = [{"content": "Record " + mrn}]

    result = preflight_context(
        context,
        tool_outputs,
        policy=REDACT_THEN_CONTINUE_POLICY,
    )

    assert result.allowed is True
    assert result.report.redacted is True
    assert result.context["messages"][0] == "Contact [OPENMED_REDACTED_EMAIL]"
    assert result.tool_outputs[0]["content"] == (
        "Record [OPENMED_REDACTED_MEDICAL_RECORD_NUMBER]"
    )
    assert email not in json.dumps(result.context, sort_keys=True)
    assert mrn not in json.dumps(result.tool_outputs, sort_keys=True)
    assert {finding.channel for finding in result.findings} == {
        "context",
        "tool_output",
    }
    assert result.finding_categories == ("EMAIL", "MEDICAL_RECORD_NUMBER")
    assert email not in json.dumps(result.to_dict(), sort_keys=True)
    assert mrn not in json.dumps(result.to_dict(), sort_keys=True)


def test_redaction_replaces_multiple_findings_in_one_string_leaf() -> None:
    first_email = _synthetic_email()
    second_email = "second-" + "example" + "@example.invalid"
    context = first_email + " and " + second_email

    result = preflight_context(context, policy="redact")

    assert result.context == ("[OPENMED_REDACTED_EMAIL] and [OPENMED_REDACTED_EMAIL]")
    assert len(result.findings) == 2


def test_custom_scanner_is_local_and_deterministic() -> None:
    marker = "SYNTHETIC_IDENTIFIER"

    def scanner(text: str):
        start = text.find(marker)
        if start < 0:
            return ()
        return (
            {
                "category": "LOCAL_IDENTIFIER",
                "start": start,
                "end": start + len(marker),
            },
        )

    first = preflight_context(
        {"content": "prefix " + marker},
        policy="redact",
        scanner=scanner,
    )
    second = preflight_context(
        {"content": "prefix " + marker},
        policy="redact",
        scanner=scanner,
    )

    assert first.to_dict() == second.to_dict()
    assert first.context == {"content": "prefix [OPENMED_REDACTED_LOCAL_IDENTIFIER]"}
    assert first.findings[0].offsets == (7, 7 + len(marker))
    assert marker not in json.dumps(first.to_dict(), sort_keys=True)


def test_scanner_failures_are_generic_and_do_not_echo_input() -> None:
    marker = "SYNTHETIC_IDENTIFIER"

    def scanner(_: str):
        raise RuntimeError("scanner saw " + marker)

    with pytest.raises(PreflightScanError) as raised:
        preflight_context(marker, scanner=scanner)

    assert marker not in str(raised.value)
