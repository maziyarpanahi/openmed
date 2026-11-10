"""Tests for vCard and iCalendar PHI redaction."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from openmed.multimodal import (
    ExtractedDocument,
    redact_contacts_calendar,
    redact_document,
)

FIXTURES = Path(__file__).parent / "fixtures"
URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
MAILTO_URI_RE = re.compile(r"^mailto:[^@\s\[\]]+@[^@\s\[\]]+$", re.IGNORECASE)
TEL_URI_RE = re.compile(r"^tel:\+?[0-9][0-9(). -]*$", re.IGNORECASE)


def _fake_redactor(text: str) -> str:
    replacements = {
        "John Doe": "[PERSON]",
        "Jane Roe": "[PERSON]",
        "555-0101": "[PHONE]",
        "john.doe@example.com": "[EMAIL]",
        "jane.roe@example.com": "[EMAIL]",
    }
    redacted = text
    for source, replacement in replacements.items():
        redacted = redacted.replace(source, replacement)
    return redacted


def _unfold(text: str) -> str:
    logical_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith((" ", "\t")) and logical_lines:
            logical_lines[-1] += line[1:]
        else:
            logical_lines.append(line)
    return "\n".join(logical_lines)


def _assert_folded_width(text: str) -> None:
    for line in text.splitlines():
        assert len(line.encode("utf-8")) <= 75


def _parameter_value(prefix: str, name: str) -> str | None:
    for token in prefix.split(";")[1:]:
        key, separator, value = token.partition("=")
        if separator and key.upper() == name.upper():
            if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
                return value[1:-1]
            return value
    return None


def _assert_content_lines_are_valid(text: str) -> None:
    for line in _unfold(text).splitlines():
        assert ":" in line
        prefix, value = line.split(":", 1)
        value_type = _parameter_value(prefix, "VALUE")
        if value_type is not None and value_type.lower() == "uri":
            assert URI_SCHEME_RE.match(value)
        if value.lower().startswith("mailto:"):
            assert MAILTO_URI_RE.match(value)
        if value.lower().startswith("tel:"):
            assert TEL_URI_RE.match(value)


def test_vcard_redacts_direct_properties_and_free_text_note():
    result = redact_contacts_calendar(
        FIXTURES / "synthetic_phi.vcf",
        models={"text_redactor": _fake_redactor},
    )

    text = _unfold(result.text)

    assert result.metadata["format"] == "vcard"
    assert "VERSION:3.0" in text
    assert "UID:contact-safe-001" in text
    assert "FN:[PERSON]" in text
    assert "N:[PERSON];[PERSON];;[PERSON];" in text
    assert "TEL;TYPE=CELL:[PHONE]" in text
    assert "EMAIL;TYPE=HOME:[EMAIL]" in text
    assert (
        "ADR;TYPE=HOME:;;[STREET_ADDRESS];[STREET_ADDRESS];"
        "[STREET_ADDRESS];[STREET_ADDRESS];[STREET_ADDRESS]"
    ) in text
    assert "ORG:[ORGANIZATION]" in text
    assert "NOTE:Patient [PERSON] can be reached at [PHONE]" in text
    assert "john.doe@example.com" not in text
    assert "John Doe" not in text
    assert "555-0101" not in text
    _assert_folded_width(result.text)
    assert any(line.startswith(" ") for line in result.text.splitlines())


def test_vcard_uri_values_keep_valid_uri_shape():
    source = (
        "BEGIN:VCARD\n"
        "VERSION:4.0\n"
        "TEL;VALUE=uri:tel:+1-555-0101\n"
        "EMAIL;VALUE=uri:mailto:john.doe@example.com\n"
        "END:VCARD\n"
    )

    result = redact_contacts_calendar(source)
    text = _unfold(result.text)

    assert "TEL;VALUE=uri:tel:+10000000000" in text
    assert "EMAIL;VALUE=uri:mailto:redacted@example.invalid" in text
    assert "+1-555-0101" not in text
    assert "john.doe@example.com" not in text
    _assert_content_lines_are_valid(result.text)

    hashed = redact_contacts_calendar(
        source,
        policy={"action_overrides": {"TEL": "hash", "EMAIL": "hash"}},
    )
    hashed_text = _unfold(hashed.text)

    assert re.search(r"TEL;VALUE=uri:tel:\+1\d{10}", hashed_text)
    assert re.search(
        r"EMAIL;VALUE=uri:mailto:email-[0-9a-f]{8}@example\.invalid",
        hashed_text,
    )
    assert "+1-555-0101" not in hashed_text
    assert "john.doe@example.com" not in hashed_text
    _assert_content_lines_are_valid(hashed.text)


def test_icalendar_redacts_event_fields_values_and_cn_parameters():
    result = redact_contacts_calendar(
        FIXTURES / "synthetic_phi.ics",
        models={"text_redactor": _fake_redactor},
    )

    text = _unfold(result.text)

    assert result.metadata["format"] == "icalendar"
    assert "VERSION:2.0" in text
    assert "PRODID:-//OpenMed//Synthetic Calendar//EN" in text
    assert "UID:appointment-safe-001" in text
    assert "DTSTART:20260401T090000Z" in text
    assert "SUMMARY:Visit for [PERSON]" in text
    assert "DESCRIPTION:[PERSON] reported symptoms" in text
    assert "LOCATION:[LOCATION]" in text
    assert (
        "ATTENDEE;CN=[PERSON];ROLE=REQ-PARTICIPANT:mailto:redacted@example.invalid"
    ) in text
    assert 'ORGANIZER;CN="[PERSON]":mailto:redacted@example.invalid' in text
    assert "Jane Roe" not in text
    assert "jane.roe@example.com" not in text
    assert "Dr Alice Smith" not in text
    assert "alice.smith@clinic.example" not in text
    _assert_folded_width(result.text)
    assert any(line.startswith(" ") for line in result.text.splitlines())


def test_redacted_output_keeps_valid_content_lines():
    vcard = redact_contacts_calendar(
        (
            "BEGIN:VCARD\n"
            "VERSION:4.0\n"
            "TEL;VALUE=uri:tel:+1-555-0101\n"
            "EMAIL;VALUE=uri:mailto:john.doe@example.com\n"
            "END:VCARD\n"
        )
    )
    calendar = redact_contacts_calendar(
        FIXTURES / "synthetic_phi.ics",
        models={"text_redactor": _fake_redactor},
    )

    _assert_content_lines_are_valid(vcard.text)
    _assert_content_lines_are_valid(calendar.text)


def test_default_free_text_redactor_calls_openmed_pii(monkeypatch):
    calls = []

    def fake_deidentify(text: str, **kwargs):
        calls.append((text, kwargs))
        return SimpleNamespace(deidentified_text=text.replace("John Doe", "[PERSON]"))

    monkeypatch.setattr("openmed.core.pii.deidentify", fake_deidentify)

    result = redact_contacts_calendar(
        "BEGIN:VCALENDAR\nSUMMARY:Visit John Doe\nEND:VCALENDAR\n",
        policy={"deidentify_policy": "hipaa_safe_harbor"},
        lang="fr",
    )

    assert "SUMMARY:Visit [PERSON]" in result.text
    assert calls == [
        (
            "Visit John Doe",
            {
                "method": "mask",
                "lang": "fr",
                "policy": "hipaa_safe_harbor",
            },
        )
    ]


def test_action_overrides_can_keep_selected_properties():
    result = redact_contacts_calendar(
        "BEGIN:VCARD\nFN:John Doe\nNOTE:John Doe\nEND:VCARD\n",
        policy={"action_overrides": {"FN": "keep", "NOTE": "keep"}},
        models={"text_redactor": _fake_redactor},
    )

    assert "FN:John Doe" in result.text
    assert "NOTE:John Doe" in result.text


def test_redact_document_dispatches_vcard_and_icalendar_handlers():
    vcard = redact_document(
        str(FIXTURES / "synthetic_phi.vcf"),
        models={"text_redactor": _fake_redactor},
    )
    calendar = redact_document(
        str(FIXTURES / "synthetic_phi.ics"),
        models={"text_redactor": _fake_redactor},
    )

    assert isinstance(vcard, ExtractedDocument)
    assert isinstance(calendar, ExtractedDocument)
    assert vcard.metadata["format"] == "vcard"
    assert calendar.metadata["format"] == "icalendar"
    assert "FN:[PERSON]" in _unfold(vcard.text)
    assert "SUMMARY:Visit for [PERSON]" in _unfold(calendar.text)
