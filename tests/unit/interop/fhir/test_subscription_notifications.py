"""Synthetic offline intake coverage for the three notification formats."""

from __future__ import annotations

import copy
import json
import re
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.interop.fhir.subscription_notifications import (
    MAX_EVENT_NUMBER,
    MAX_NOTIFICATION_BYTES,
    MAX_NOTIFICATION_ENTRIES,
    MAX_NOTIFICATION_EVENTS,
    SubscriptionContentMode,
    SubscriptionFindingCode,
    SubscriptionNotificationError,
    SubscriptionNotificationType,
    parse_subscription_notification,
)

VERSIONS = ("R4", "R4B", "R5")
MODES = ("empty", "id-only", "full-resource")
TYPES = tuple(kind.value for kind in SubscriptionNotificationType)
SENTINEL = "synthetic-private-notification-marker"


def notification(
    version="R5", kind="event-notification", mode="empty", numbers=(1,), since="1"
):
    status = {
        "resourceType": "SubscriptionStatus",
        "status": "active",
        "type": kind,
        "subscription": {"reference": "Subscription/synthetic-subscription"},
    }
    if since is not None:
        status["eventsSinceSubscriptionStart"] = since
    events = []
    for number in numbers:
        event = {"eventNumber": str(number)}
        if mode != "empty":
            event["focus"] = {"reference": "Observation/synthetic-observation"}
        events.append(event)
    if kind != "query-status":
        status["notificationEvent"] = events
    if version == "R4":
        parameters = [
            {"name": "subscription", "valueReference": status["subscription"]},
            {"name": "status", "valueCode": status["status"]},
            {"name": "type", "valueCode": kind},
        ]
        if since is not None:
            parameters.append(
                {"name": "events-since-subscription-start", "valueString": since}
            )
        for event in status.get("notificationEvent", []):
            parts = [{"name": "event-number", "valueString": event["eventNumber"]}]
            if "focus" in event:
                parts.append({"name": "focus", "valueReference": event["focus"]})
            parameters.append({"name": "notification-event", "part": parts})
        status = {"resourceType": "Parameters", "parameter": parameters}
    entries = [{"resource": status}]
    if mode == "full-resource" and kind != "query-status":
        entries.append(
            {
                "fullUrl": "https://synthetic.invalid/Observation/synthetic-observation",
                "resource": {
                    "resourceType": "Observation",
                    "id": "synthetic-observation",
                    "valueString": SENTINEL,
                },
            }
        )
    return {
        "resourceType": "Bundle",
        "type": "subscription-notification" if version == "R5" else "history",
        "entry": entries,
    }


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("kind", TYPES)
@pytest.mark.parametrize("mode", MODES)
def test_formats_types_and_content_modes(version, kind, mode):
    source = notification(version, kind, mode)
    before = copy.deepcopy(source)
    report = parse_subscription_notification(
        source, version=version, previous_event_number=0
    )
    assert (
        report.to_json()
        == parse_subscription_notification(
            json.dumps(source), version=version, previous_event_number=0
        ).to_json()
    )
    assert source == before
    assert report.version.value == version
    assert report.notification_type.value == kind
    if kind == "query-status":
        assert report.events == ()
    else:
        assert report.events[0].content_mode.value == mode
        assert report.events[0].requires_authorized_read == (mode != "full-resource")
    assert bool(report.workflow_events) == (kind == "event-notification")
    rendered = report.to_json() + repr(report)
    for marker in (
        SENTINEL,
        "synthetic-subscription",
        "synthetic-observation",
        "https://synthetic.invalid",
    ):
        assert marker not in rendered


@pytest.mark.parametrize("version", VERSIONS)
def test_gaps_duplicates_order_and_checkpoint_history_are_explicit(version):
    report = parse_subscription_notification(
        notification(version, numbers=(4, 2, 2), since="6"),
        version=version,
        previous_event_number=1,
    )
    codes = [finding.code for finding in report.findings]
    assert SubscriptionFindingCode.OUT_OF_ORDER in codes
    assert SubscriptionFindingCode.DUPLICATE in codes
    gaps = [f for f in report.findings if f.code is SubscriptionFindingCode.GAP]
    assert [(f.first_missing, f.last_missing, f.count) for f in gaps] == [
        (3, 3, 1),
        (5, 6, 2),
    ]
    assert report.workflow_events == ()
    assert report.requires_reconciliation


def test_unknown_history_does_not_invent_a_missing_prefix():
    report = parse_subscription_notification(
        notification(numbers=(100,), since="100"), version="R5"
    )
    assert [f.code for f in report.findings] == [
        SubscriptionFindingCode.HISTORY_UNKNOWN
    ]
    known = parse_subscription_notification(
        notification(numbers=(100,), since="100"),
        version="R5",
        previous_event_number=99,
    )
    assert not known.requires_reconciliation
    assert len(known.workflow_events) == 1


def test_guaranteed_delivery_relative_numbers_do_not_compare_global_counter():
    report = parse_subscription_notification(
        notification(numbers=(1, 2), since="100"),
        version="R5",
        global_event_numbers=False,
    )
    assert not report.findings
    assert len(report.workflow_events) == 2


def test_relative_number_gaps_require_reconciliation_without_global_tail():
    report = parse_subscription_notification(
        notification(numbers=(2, 4), since="100"),
        version="R5",
        global_event_numbers=False,
    )
    assert [(f.first_missing, f.last_missing) for f in report.findings] == [
        (1, 1),
        (3, 3),
    ]
    assert report.workflow_events == ()


def test_two_names_for_the_same_entry_are_not_ambiguous():
    source = notification(mode="full-resource")
    source["entry"][1]["fullUrl"] = "Observation/synthetic-observation"
    event = parse_subscription_notification(source, version="R5").events[0]
    assert event.content_mode is SubscriptionContentMode.FULL_RESOURCE


def test_query_events_and_control_history_cannot_become_new_workflows():
    for kind in ("query-event", "handshake", "heartbeat"):
        report = parse_subscription_notification(
            notification(kind=kind, numbers=(2,), since="100"), version="R5"
        )
        assert len(report.events) == 1
        assert report.workflow_events == ()


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("kind", ("heartbeat", "handshake", "query-status"))
def test_control_counter_ahead_of_checkpoint_reports_missing_delivery(version, kind):
    source = notification(version, kind, numbers=(), since="6")
    report = parse_subscription_notification(
        source, version=version, previous_event_number=2
    )
    gaps = [f for f in report.findings if f.code is SubscriptionFindingCode.GAP]
    assert [(f.first_missing, f.last_missing, f.count) for f in gaps] == [(3, 6, 4)]
    assert report.workflow_events == ()
    unknown = parse_subscription_notification(source, version=version)
    assert [f.code for f in unknown.findings] == [
        SubscriptionFindingCode.HISTORY_UNKNOWN
    ]


def test_query_event_history_does_not_report_missing_newer_deliveries():
    report = parse_subscription_notification(
        notification(kind="query-event", numbers=(3,), since="100"),
        version="R5",
        previous_event_number=2,
    )
    assert report.findings == ()
    assert report.workflow_events == ()


@pytest.mark.parametrize(
    "location", ("bundle", "entry", "status", "event", "reference", "parameter")
)
@pytest.mark.parametrize(
    "field", ("extension", "modifierExtension", "implicitRules", "_type")
)
@pytest.mark.parametrize("value", (None, False, 0, "", [], {}))
def test_unhandled_extensions_cannot_hide_behind_falsey_values(location, field, value):
    source = notification("R4" if location == "parameter" else "R5")
    status = source["entry"][0]["resource"]
    target = {
        "bundle": source,
        "entry": source["entry"][0],
        "status": status,
    }.get(location)
    if location == "event":
        target = status["notificationEvent"][0]
    elif location == "reference":
        target = status["subscription"]
    elif location == "parameter":
        target = status["parameter"][0]
    target[field] = value
    with pytest.raises(SubscriptionNotificationError) as error:
        parse_subscription_notification(
            source, version="R4" if location == "parameter" else "R5"
        )
    assert error.value.code == "unsupported_modifier"


@pytest.mark.parametrize("release", ("R4", "R5"))
def test_recognized_unsuffixed_r4b_bundle_profile_rejects_other_releases(release):
    source = notification(release)
    source["meta"] = {
        "profile": [
            "http://hl7.org/fhir/uv/subscriptions-backport/StructureDefinition/backport-subscription-notification|1.1.0"
        ]
    }
    with pytest.raises(SubscriptionNotificationError, match="mixed_version"):
        parse_subscription_notification(source, version=release)
    source = notification("R4B")
    source["meta"] = {
        "profile": [
            "http://hl7.org/fhir/uv/subscriptions-backport/StructureDefinition/backport-subscription-notification|1.1.0"
        ]
    }
    assert parse_subscription_notification(source, version="R4B").version.value == "R4B"


@pytest.mark.parametrize(
    "number", [0, -1, MAX_EVENT_NUMBER + 1, True, None, {}, "01", "+1", "1.0"]
)
def test_bad_counter_values_fail_with_closed_codes(number):
    source = notification()
    source["entry"][0]["resource"]["notificationEvent"][0]["eventNumber"] = number
    with pytest.raises(SubscriptionNotificationError, match="invalid_counter"):
        parse_subscription_notification(source, version="R5")


@pytest.mark.parametrize("version", ["R3", "R6", SENTINEL, 5, None, []])
def test_unsupported_versions_do_not_echo_the_value(version):
    with pytest.raises(SubscriptionNotificationError) as error:
        parse_subscription_notification(notification(), version=version)
    assert error.value.code == "unsupported_version"
    assert SENTINEL not in str(error.value)


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda b: b.update(type="transaction"), "invalid_bundle"),
        (lambda b: b.update(entry=[]), "invalid_bundle"),
        (lambda b: b["entry"][0]["resource"].update(type=[]), "invalid_status"),
        (lambda b: b["entry"][0]["resource"].update(status={}), "invalid_status"),
        (
            lambda b: b["entry"][0]["resource"].update(
                subscription={"identifier": {"value": SENTINEL}}
            ),
            "invalid_reference",
        ),
        (
            lambda b: b["entry"][0]["resource"].update(notificationEvent=[]),
            "invalid_event",
        ),
        (
            lambda b: b["entry"][0]["resource"].update(
                modifierExtension=[{"url": SENTINEL}]
            ),
            "unsupported_modifier",
        ),
        (
            lambda b: b["entry"].append(
                {"resource": {"resourceType": "SubscriptionStatus"}}
            ),
            "mixed_version",
        ),
        (
            lambda b: b["entry"][0]["resource"].update(
                meta={
                    "profile": [
                        "http://hl7.org/fhir/StructureDefinition/SubscriptionStatus|4.3.0"
                    ]
                }
            ),
            "mixed_version",
        ),
    ],
)
def test_malformed_and_mixed_bundles_are_refused(mutation, code):
    source = notification()
    mutation(source)
    with pytest.raises(SubscriptionNotificationError) as error:
        parse_subscription_notification(source, version="R5")
    assert error.value.code == code
    assert SENTINEL not in str(error.value)


@pytest.mark.parametrize(
    "field",
    [
        "type",
        "status",
        "subscription",
        "notificationEvent",
        "eventsSinceSubscriptionStart",
    ],
)
@pytest.mark.parametrize("value", [None, True, 0, [], {}, SENTINEL])
def test_malformed_required_values_never_escape_as_python_type_errors(field, value):
    source = notification()
    source["entry"][0]["resource"][field] = value
    with pytest.raises(SubscriptionNotificationError) as error:
        parse_subscription_notification(source, version="R5")
    assert SENTINEL not in str(error.value)


@pytest.mark.parametrize(
    "source,code",
    [
        (b'{"resourceType":"Bundle","resourceType":"Bundle"}', "duplicate_json_key"),
        (b'{"resourceType":', "invalid_json"),
        (b"\xff", "invalid_json"),
        (json.dumps(notification()).encode("utf-16"), "invalid_json"),
        ({"invalid": float("nan")}, "invalid_json"),
        ({"invalid": "\ud800"}, "invalid_json"),
        ({"\ud800": "invalid"}, "invalid_json"),
        ({"invalid": 2**1000}, "invalid_json"),
        (" " * (MAX_NOTIFICATION_BYTES + 1), "payload_too_large"),
        (
            '{"text":"' + "\u4e00" * (MAX_NOTIFICATION_BYTES // 2) + '"}',
            "payload_too_large",
        ),
    ],
)
def test_json_refusals_have_no_raw_decoder_context(source, code):
    with pytest.raises(SubscriptionNotificationError) as error:
        parse_subscription_notification(source, version="R5")
    assert error.value.code == code
    assert error.value.__context__ is None


def test_entry_event_depth_and_node_bounds():
    source = notification()
    source["entry"] *= MAX_NOTIFICATION_ENTRIES + 1
    with pytest.raises(SubscriptionNotificationError, match="invalid_bundle"):
        parse_subscription_notification(source, version="R5")
    source = notification(
        numbers=tuple(range(1, MAX_NOTIFICATION_EVENTS + 2)),
        since=str(MAX_NOTIFICATION_EVENTS + 1),
    )
    with pytest.raises(SubscriptionNotificationError, match="invalid_event"):
        parse_subscription_notification(source, version="R5")
    nested = 1
    for _ in range(40):
        nested = [nested]
    with pytest.raises(SubscriptionNotificationError, match="payload_too_deep"):
        parse_subscription_notification({"nested": nested}, version="R5")
    with pytest.raises(SubscriptionNotificationError, match="payload_too_complex"):
        parse_subscription_notification({"wide": [0] * 40_000}, version="R5")


def test_partial_additional_context_requires_authorized_read():
    source = notification(mode="full-resource")
    source["entry"][0]["resource"]["notificationEvent"][0]["additionalContext"] = [
        {"reference": "Patient/synthetic-unincluded"}
    ]
    event = parse_subscription_notification(source, version="R5").events[0]
    assert event.content_mode is SubscriptionContentMode.ID_ONLY
    assert event.requires_authorized_read
    assert len(event.additional_reference_digests) == 1


def test_replay_and_server_error_hold_new_workflow_events():
    source = notification()
    source["entry"][0]["resource"]["error"] = [{"text": SENTINEL}]
    report = parse_subscription_notification(
        source, version="R5", previous_event_number=1
    )
    assert {f.code for f in report.findings} == {
        SubscriptionFindingCode.SERVER_ERROR,
        SubscriptionFindingCode.DUPLICATE,
    }
    assert not report.workflow_events
    assert SENTINEL not in report.to_json()


def test_inconsistent_typed_metadata_cannot_bypass_read_requirement():
    event = parse_subscription_notification(notification(), version="R5").events[0]
    with pytest.raises(SubscriptionNotificationError, match="invalid_metadata"):
        replace(event, content_mode=SubscriptionContentMode.FULL_RESOURCE)


def test_parser_performs_no_network_or_persistence(monkeypatch, tmp_path):
    def deny(*args, **kwargs):
        raise AssertionError("unexpected external effect")

    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.chdir(tmp_path)
    parse_subscription_notification(notification(mode="full-resource"), version="R5")
    assert list(tmp_path.iterdir()) == []


def test_documented_example_runs_without_a_resource_read():
    path = (
        Path(__file__).resolve().parents[4]
        / "docs/interop/fhir-subscription-notifications.md"
    )
    blocks = re.findall(
        r"^```python\n(.*?)^```", path.read_text(encoding="utf-8"), re.M | re.S
    )
    assert len(blocks) == 1
    namespace = {}
    exec(compile(blocks[0], "fhir-subscription-notifications.md", "exec"), namespace)
    assert namespace["event"].requires_authorized_read


def test_parser_is_exported_from_fhir_without_expanding_the_converter():
    from openmed.interop.fhir import parse_subscription_notification as public
    from openmed.interop.fhir.versions import FHIRVersion

    assert public is parse_subscription_notification
    assert [version.value for version in FHIRVersion] == ["R4", "R5"]
