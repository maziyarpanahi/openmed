from __future__ import annotations

import json

import pytest

from openmed.clinical.review_packet_privacy import (
    ReviewPacketPrivacyBlocked,
    ReviewPacketPrivacyScanError,
    enforce_review_packet_privacy,
    export_review_packet,
    persist_review_packet,
    scan_review_packet_privacy,
)
from openmed.core.audit import hash_text


def test_scan_uses_exact_rendered_content_and_emits_only_safe_finding_fields() -> None:
    rendered = "Review packet: SYNTHETIC_IDENTIFIER_001"
    start = rendered.index("SYNTHETIC_IDENTIFIER_001")
    seen: list[str] = []

    def detector(text: str):
        seen.append(text)
        return [
            {
                "label": "name",
                "start": start,
                "end": len(rendered),
                "text": "SYNTHETIC_IDENTIFIER_001",
                "severity": "critical",
            }
        ]

    report = scan_review_packet_privacy(rendered, detector)
    payload = report.to_dict()

    assert seen == [rendered]
    assert report.blocked is True
    assert payload["findings"] == [
        {
            "entity_class": "NAME",
            "start": start,
            "end": len(rendered),
            "text_hash": hash_text("SYNTHETIC_IDENTIFIER_001"),
        }
    ]
    serialized = json.dumps(payload, sort_keys=True)
    assert "SYNTHETIC_IDENTIFIER_001" not in serialized
    assert 'text"' not in serialized


def test_critical_leak_blocks_exporter_and_exception_is_value_free() -> None:
    rendered = "Packet IDENTIFIER_002"
    called = False

    def exporter(_: str) -> None:
        nonlocal called
        called = True

    with pytest.raises(ReviewPacketPrivacyBlocked) as error:
        export_review_packet(
            rendered,
            lambda _: [{"label": "SSN", "start": 7, "end": len(rendered)}],
            exporter,
            critical_entity_classes={"SSN"},
        )

    assert called is False
    assert error.value.report.critical_count == 1
    assert "IDENTIFIER_002" not in str(error.value)
    assert "IDENTIFIER_002" not in repr(error.value)


def test_critical_leak_does_not_create_persistence_target(tmp_path) -> None:
    destination = tmp_path / "blocked-packet.txt"

    with pytest.raises(ReviewPacketPrivacyBlocked):
        persist_review_packet(
            destination,
            "IDENTIFIER_003",
            lambda _: [
                {"entity_class": "NAME", "start": 0, "end": 14, "critical": True}
            ],
        )

    assert not destination.exists()


def test_noncritical_packet_can_be_exported_and_persisted(tmp_path) -> None:
    rendered = "De-identified synthetic clinical review packet"
    detector = lambda _: [  # noqa: E731
        {"label": "CLINICAL_TERM", "start": 0, "end": 2, "severity": "low"}
    ]

    exported, report = export_review_packet(rendered, detector, str.upper)
    destination = tmp_path / "packet.txt"
    persisted_report = persist_review_packet(destination, rendered, detector)

    assert exported == rendered.upper()
    assert report.blocked is False
    assert persisted_report.to_dict() == report.to_dict()
    assert destination.read_text(encoding="utf-8") == rendered


def test_findings_are_sorted_deduplicated_and_deterministic() -> None:
    rendered = "abcd"
    finding = {"label": "name", "start": 1, "end": 3, "critical": False}
    detector = lambda _: [  # noqa: E731
        {"label": "phone", "start": 3, "end": 4, "critical": False},
        finding,
        dict(finding),
    ]

    first = scan_review_packet_privacy(rendered, detector)
    second = scan_review_packet_privacy(rendered, detector)

    assert first.to_dict() == second.to_dict()
    assert [(item.start, item.end) for item in first.findings] == [(1, 3), (3, 4)]


def test_detector_and_offset_failures_do_not_echo_sensitive_values() -> None:
    rendered = "SYNTHETIC_IDENTIFIER_004"

    def failing_detector(_: str):
        raise RuntimeError(rendered)

    with pytest.raises(ReviewPacketPrivacyScanError) as detector_error:
        enforce_review_packet_privacy(rendered, failing_detector)
    assert rendered not in str(detector_error.value)

    with pytest.raises(ReviewPacketPrivacyScanError) as offset_error:
        scan_review_packet_privacy(
            rendered,
            lambda _: [{"label": rendered, "start": -1, "end": 5}],
        )
    assert rendered not in str(offset_error.value)
