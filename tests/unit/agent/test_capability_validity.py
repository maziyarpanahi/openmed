"""Synthetic table tests for agent capability grant validity."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from openmed.agent.capability_validity import (
    CAPABILITY_VALIDITY_SCHEMA_VERSION,
    DEFAULT_CLOCK_SKEW_SECONDS,
    GRANT_REASON_CODES,
    MAX_CLOCK_SKEW_SECONDS,
    MAX_EPOCH_SECONDS,
    MAX_GRANT_LIFETIME_SECONDS,
    CapabilityGrant,
    CapabilityValidityError,
    GrantStatus,
    GrantValidityReport,
    check_capability_validity,
)

CAPABILITY = "capability:openmed.agent/summarize"
AUDIENCE = "local-agent"
ISSUED = 1_700_000_000
EXPIRES = ISSUED + 3_600
INSIDE = ISSUED + 1_800

SENTINEL = "JANE DOE MRN 8675309 Bearer abc.def /var/phi/grant.json"

GRANT = CapabilityGrant(
    capability_id=CAPABILITY,
    audience=AUDIENCE,
    issued_at=ISSUED,
    expires_at=EXPIRES,
)


def check(grant=GRANT, *, now=INSIDE, audience=AUDIENCE, **kwargs):
    return check_capability_validity(
        grant, now=now, expected_audience=audience, **kwargs
    )


@pytest.mark.parametrize(
    "now,audience,status,reasons",
    [
        (INSIDE, AUDIENCE, GrantStatus.VALID, ()),
        (ISSUED, AUDIENCE, GrantStatus.VALID, ()),
        (EXPIRES, AUDIENCE, GrantStatus.VALID, ()),
        (EXPIRES + 10_000, AUDIENCE, GrantStatus.EXPIRED, ("grant_expired",)),
        (
            ISSUED - 10_000,
            AUDIENCE,
            GrantStatus.REJECTED,
            ("clock_skew_exceeded", "grant_not_yet_valid"),
        ),
        (INSIDE, "other-agent", GrantStatus.REJECTED, ("audience_mismatch",)),
    ],
)
def test_grant_validity_table(now, audience, status, reasons) -> None:
    report = check(now=now, audience=audience)
    assert report.status is status
    assert report.reason_codes == reasons
    assert report.is_valid is (status is GrantStatus.VALID)
    assert report.capability_id == CAPABILITY
    assert report.schema_version == CAPABILITY_VALIDITY_SCHEMA_VERSION


def test_seconds_until_expiry_is_signed() -> None:
    assert check(now=EXPIRES - 5).seconds_until_expiry == 5
    assert check(now=EXPIRES + 5).seconds_until_expiry == -5
    assert check(now=EXPIRES).seconds_until_expiry == 0


def test_clock_skew_widens_both_window_edges() -> None:
    assert check(now=EXPIRES + DEFAULT_CLOCK_SKEW_SECONDS).is_valid
    assert not check(now=EXPIRES + DEFAULT_CLOCK_SKEW_SECONDS + 1).is_valid
    assert check(now=ISSUED - DEFAULT_CLOCK_SKEW_SECONDS).is_valid
    early = check(now=ISSUED - DEFAULT_CLOCK_SKEW_SECONDS - 1)
    assert early.status is GrantStatus.REJECTED
    assert early.reason_codes == ("clock_skew_exceeded", "grant_not_yet_valid")


def test_zero_skew_makes_the_window_exact() -> None:
    assert check(now=EXPIRES, max_clock_skew_seconds=0).is_valid
    assert not check(now=EXPIRES + 1, max_clock_skew_seconds=0).is_valid
    assert check(now=ISSUED, max_clock_skew_seconds=0).is_valid
    assert not check(now=ISSUED - 1, max_clock_skew_seconds=0).is_valid


def test_a_grant_issued_far_in_the_future_is_rejected_not_merely_early() -> None:
    report = check(now=ISSUED - MAX_CLOCK_SKEW_SECONDS - 1)
    assert "clock_skew_exceeded" in report.reason_codes
    assert report.status is GrantStatus.REJECTED


def test_not_before_delays_activation_without_changing_expiry() -> None:
    delayed = replace(GRANT, not_before=ISSUED + 1_000)
    assert delayed.effective_not_before == ISSUED + 1_000
    early = check(delayed, now=ISSUED + 100, max_clock_skew_seconds=0)
    assert early.status is GrantStatus.NOT_YET_VALID
    assert early.reason_codes == ("grant_not_yet_valid",)
    assert check(delayed, now=ISSUED + 1_000, max_clock_skew_seconds=0).is_valid
    assert check(delayed, now=EXPIRES + 10_000).status is GrantStatus.EXPIRED


def test_absent_not_before_defaults_to_issued_at() -> None:
    assert GRANT.not_before is None
    assert GRANT.effective_not_before == ISSUED
    assert GRANT.lifetime_seconds == 3_600


def test_lifetime_ceiling_rejects_a_long_grant() -> None:
    within = check(max_lifetime_seconds=3_600)
    assert within.is_valid
    over = check(max_lifetime_seconds=3_599)
    assert over.status is GrantStatus.REJECTED
    assert over.reason_codes == ("lifetime_exceeded",)


def test_the_worst_status_wins() -> None:
    report = check(now=EXPIRES + 10_000, audience="other-agent")
    assert report.status is GrantStatus.REJECTED
    assert report.reason_codes == ("audience_mismatch", "grant_expired")


def test_reason_codes_follow_the_declared_order() -> None:
    report = check(now=ISSUED - 10_000, audience="other-agent", max_lifetime_seconds=60)
    assert report.reason_codes == (
        "audience_mismatch",
        "clock_skew_exceeded",
        "lifetime_exceeded",
        "grant_not_yet_valid",
    )
    positions = [GRANT_REASON_CODES.index(code) for code in report.reason_codes]
    assert positions == sorted(positions)


def test_report_serialization_is_byte_stable_and_field_ordered() -> None:
    report = check(now=EXPIRES + 10_000)
    assert list(report.to_dict()) == [
        "schema_version",
        "capability_id",
        "audience",
        "status",
        "reason_codes",
        "evaluated_at",
        "seconds_until_expiry",
    ]
    assert json.loads(report.to_json()) == report.to_dict()
    assert report.to_json() == check(now=EXPIRES + 10_000).to_json()


def test_grant_serialization_is_metadata_only() -> None:
    assert GRANT.to_dict() == {
        "capability_id": CAPABILITY,
        "audience": AUDIENCE,
        "issued_at": ISSUED,
        "not_before": None,
        "expires_at": EXPIRES,
    }


def test_reports_carry_only_identifiers_and_integers() -> None:
    payload = json.loads(check().to_json())
    assert isinstance(payload["reason_codes"], list)
    for key in ("evaluated_at", "seconds_until_expiry"):
        assert isinstance(payload[key], int)
    assert set(payload) == {
        "schema_version",
        "capability_id",
        "audience",
        "status",
        "reason_codes",
        "evaluated_at",
        "seconds_until_expiry",
    }


def test_grants_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        GRANT.expires_at = EXPIRES + 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "value",
    ["", "summarize", "tool:openmed.agent/summarize", "capability:bad/NAME", 7, None],
)
def test_invalid_capability_identifiers_fail_closed(value) -> None:
    with pytest.raises(
        CapabilityValidityError, match="^capability_id: invalid_capability_id$"
    ):
        replace(GRANT, capability_id=value)


@pytest.mark.parametrize(
    "value", ["", "Local-Agent", "-lead", "trail-", "a" * 129, "svc/one", 7, None]
)
def test_invalid_audiences_fail_closed(value) -> None:
    with pytest.raises(CapabilityValidityError, match="^audience: invalid_audience$"):
        replace(GRANT, audience=value)
    with pytest.raises(
        CapabilityValidityError, match="^expected_audience: invalid_audience$"
    ):
        check(audience=value)


@pytest.mark.parametrize("field", ["issued_at", "expires_at"])
@pytest.mark.parametrize("value", [True, False, 1.0, "1700000000", None])
def test_malformed_timestamps_fail_closed(field, value) -> None:
    with pytest.raises(CapabilityValidityError) as excinfo:
        replace(GRANT, **{field: value})
    assert (excinfo.value.code, excinfo.value.field_name) == (
        "invalid_timestamp",
        field,
    )


@pytest.mark.parametrize("value", [-1, MAX_EPOCH_SECONDS + 1, 2**70])
def test_out_of_range_timestamps_fail_closed(value) -> None:
    with pytest.raises(CapabilityValidityError, match="^issued_at: "):
        replace(GRANT, issued_at=value)
    with pytest.raises(CapabilityValidityError, match="^now: "):
        check(now=value)


@pytest.mark.parametrize("value", [True, 1.0, "now", None])
def test_a_malformed_evaluation_instant_fails_closed(value) -> None:
    with pytest.raises(CapabilityValidityError, match="^now: invalid_timestamp$"):
        check(now=value)


@pytest.mark.parametrize("expires", [ISSUED, ISSUED - 1, 0])
def test_expiry_must_follow_issue(expires) -> None:
    with pytest.raises(
        CapabilityValidityError, match="^expires_at: expiry_not_after_issue$"
    ):
        replace(GRANT, expires_at=expires)


@pytest.mark.parametrize("value", [ISSUED - 1, EXPIRES + 1])
def test_not_before_must_sit_inside_the_window(value) -> None:
    with pytest.raises(
        CapabilityValidityError, match="^not_before: not_before_out_of_window$"
    ):
        replace(GRANT, not_before=value)
    assert replace(GRANT, not_before=ISSUED).not_before == ISSUED
    assert replace(GRANT, not_before=EXPIRES).not_before == EXPIRES


@pytest.mark.parametrize("value", [True, 1.5, "60", None])
def test_malformed_bounds_fail_closed(value) -> None:
    with pytest.raises(
        CapabilityValidityError, match="^max_clock_skew_seconds: invalid_bound$"
    ):
        check(max_clock_skew_seconds=value)
    if value is not None:
        with pytest.raises(
            CapabilityValidityError, match="^max_lifetime_seconds: invalid_bound$"
        ):
            check(max_lifetime_seconds=value)


@pytest.mark.parametrize("value", [-1, MAX_CLOCK_SKEW_SECONDS + 1])
def test_out_of_range_skew_fails_closed(value) -> None:
    with pytest.raises(CapabilityValidityError, match="^max_clock_skew_seconds: "):
        check(max_clock_skew_seconds=value)


@pytest.mark.parametrize("value", [-1, MAX_GRANT_LIFETIME_SECONDS + 1])
def test_out_of_range_lifetime_bounds_fail_closed(value) -> None:
    with pytest.raises(CapabilityValidityError, match="^max_lifetime_seconds: "):
        check(max_lifetime_seconds=value)


def test_a_zero_lifetime_ceiling_fails_closed() -> None:
    with pytest.raises(
        CapabilityValidityError, match="^max_lifetime_seconds: lifetime_bound_invalid$"
    ):
        check(max_lifetime_seconds=0)


@pytest.mark.parametrize("value", [None, "grant", 7, GRANT.to_dict()])
def test_invalid_grant_arguments_fail_closed(value) -> None:
    with pytest.raises(CapabilityValidityError, match="^grant: invalid_grant_type$"):
        check_capability_validity(value, now=INSIDE, expected_audience=AUDIENCE)


def test_invalid_schema_version_fails_closed() -> None:
    with pytest.raises(CapabilityValidityError, match="^schema_version: "):
        GrantValidityReport(
            capability_id=CAPABILITY,
            audience=AUDIENCE,
            status=GrantStatus.VALID,
            reason_codes=(),
            evaluated_at=INSIDE,
            seconds_until_expiry=0,
            schema_version="openmed.agent.capability_validity.v0",
        )


def test_status_values_are_a_closed_vocabulary() -> None:
    assert [status.value for status in GrantStatus] == [
        "valid",
        "not_yet_valid",
        "expired",
        "rejected",
    ]
    assert len(set(GRANT_REASON_CODES)) == len(GRANT_REASON_CODES)


def test_no_grant_payload_can_reach_reports_or_errors() -> None:
    assert SENTINEL not in check().to_json()
    with pytest.raises(CapabilityValidityError) as excinfo:
        replace(GRANT, capability_id=SENTINEL)
    assert SENTINEL not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    with pytest.raises(CapabilityValidityError) as audience_error:
        replace(GRANT, audience=SENTINEL)
    assert SENTINEL not in str(audience_error.value)


def test_capability_validity_is_available_from_the_public_agent_api() -> None:
    import openmed.agent as agent

    assert agent.CapabilityGrant is CapabilityGrant
    assert agent.GrantValidityReport is GrantValidityReport
    assert agent.check_capability_validity is check_capability_validity
    exported = {
        "CAPABILITY_VALIDITY_SCHEMA_VERSION",
        "CapabilityGrant",
        "CapabilityValidityError",
        "DEFAULT_CLOCK_SKEW_SECONDS",
        "GRANT_REASON_CODES",
        "GrantStatus",
        "GrantValidityReport",
        "MAX_CLOCK_SKEW_SECONDS",
        "MAX_EPOCH_SECONDS",
        "MAX_GRANT_LIFETIME_SECONDS",
        "check_capability_validity",
    }
    assert exported.issubset(set(agent.__all__))
