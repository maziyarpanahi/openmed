from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.interop import adapter_spec, available_adapters, get_adapter
from openmed.interop.hl7v2 import (
    DEFAULT_FIELD_MAP,
    HL7FieldRule,
    parse_hl7v2,
    redact_hl7v2,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("message_type", ["ADT^A40", "ORU^R01"])
def test_default_map_covers_relatives_aliases_prior_ids_and_visit_ids(message_type):
    import json

    source_fields = {
        6: "MaternalSynthetic",
        9: "AliasSynthetic",
        14: "555-0139",
        20: "LicenseSynthetic",
        21: "MotherIDSynthetic",
    }
    message = "\r".join(
        [
            f"MSH|^~\\&|TEST|TEST|TEST|TEST|202610031200||{message_type}|SYNTHETIC|P|2.5",
            segment("PID", 21, source_fields),
            segment("MRG", 7, {i: f"PriorSynthetic{i}" for i in range(1, 8)}),
            segment("PV1", 19, {19: "VisitSynthetic"}),
            "ZPX|CustomPrivateValue|otherSynthetic",
            "OBX|1|NM|LAB^SYNTHETIC||7.1|mg/dL",
        ]
    )
    report = {}
    result = redact_hl7v2(message, date_shift_days=1, coverage_report=report)
    for value in (
        *source_fields.values(),
        *(f"PriorSynthetic{i}" for i in range(1, 8)),
        "VisitSynthetic",
    ):
        assert value not in result
    assert parse_hl7v2(result).encoding == parse_hl7v2(message).encoding
    assert "7.1|mg/dL" in result
    assert "ZPX|CustomPrivateValue|otherSynthetic" in result
    assert "CustomPrivateValue" not in json.dumps(report)
    assert {
        row["field"] for row in report["unmapped_fields"] if row["segment"] == "ZPX"
    } == {1, 2}
    assert all(
        set(row) == {"segment_index", "segment", "field", "length"}
        for row in report["unmapped_fields"]
    )


@pytest.mark.parametrize(
    ("segment_name", "position"), [("PID", 11), ("NK1", 4), ("GT1", 5), ("IN1", 19)]
)
def test_xad_components_keep_type_shape_and_repetitions(segment_name, position):
    import re

    address = "123 Synthetic Street^Apartment 24^SourceCity^CA^94105-1234^US^H"
    message = (
        "MSH|^~\\&|TEST|TEST|TEST|TEST|202610031200||ADT^A40|SYNTHETIC|P|2.5\r"
        + segment(segment_name, position, {position: address + "~" + address})
    )
    output = redact_hl7v2(message, date_shift_days=1, seed=32)
    value = parse_hl7v2(output).segments[1].get_field(position)
    first, second = value.split("~")
    assert first == second
    parts = first.split("^")
    assert len(parts) == 7
    assert all(a != b for a, b in zip(parts[:6], address.split("^")[:6]))
    assert not any(char.isdigit() for char in parts[2])
    assert re.fullmatch(r"[A-Z]{2}", parts[3])
    assert re.fullmatch(r"[0-9]{5}-[0-9]{4}", parts[4])
    assert re.fullmatch(r"[A-Z]{2}", parts[5])
    assert parts[6] == "H"


def fake_deidentifier(text: str, **kwargs):
    assert kwargs["method"] == "mask"
    redacted = (
        text.replace("Jane Roe", "[PERSON]")
        .replace("John Doe", "[PERSON]")
        .replace("jane.roe@example.com", "[EMAIL]")
        .replace("555-0199", "[PHONE]")
        .replace("555-0101", "[PHONE]")
        .replace("MRN67890", "[ID_NUM]")
        .replace("MRN12345", "[ID_NUM]")
    )
    return SimpleNamespace(deidentified_text=redacted)


def segment(name: str, field_count: int, values: dict[int, str]) -> str:
    fields = [""] * field_count
    for position, value in values.items():
        fields[position - 1] = value
    return "|".join([name, *fields])


def test_registry_loads_hl7v2_adapter_lazily():
    adapter = get_adapter("hl7v2")

    assert adapter is get_adapter("hl7v2")
    assert "hl7v2" in available_adapters()
    assert adapter_spec("hl7v2").description.startswith("HL7 v2")
    assert hasattr(adapter, "redact_hl7v2")


@pytest.mark.parametrize("message", ["MSH|", "MSH|^", "MSH|^~\\&X|"])
def test_parser_rejects_invalid_msh_2_length(message: str) -> None:
    with pytest.raises(ValueError, match="exactly four encoding characters"):
        parse_hl7v2(message)


def test_redacts_synthetic_adt_pid_and_nk1_fields():
    source = FIXTURES / "synthetic_phi_adt.hl7"

    redacted = redact_hl7v2(
        source,
        deidentifier=fake_deidentifier,
        date_shift_days=30,
        seed=1,
    )
    parsed = parse_hl7v2(redacted)
    pid = next(segment for segment in parsed.segments if segment.name == "PID")
    nk1 = next(segment for segment in parsed.segments if segment.name == "NK1")

    assert parsed.segment_names() == ("MSH", "EVN", "PID", "NK1", "PV1")
    assert pid.get_field(7) == "19800131"
    assert "MRN12345" not in redacted
    assert "DOE^JOHN^A" not in redacted
    assert "123 MAIN ST" not in redacted
    assert "555-0101" not in redacted
    assert "123456789" not in redacted
    assert "[ID_NUM_HASH_" in pid.get_field(3)
    assert "^" in pid.get_field(5)
    assert "^" in nk1.get_field(2)


def test_redacts_synthetic_oru_obx_and_nte_free_text():
    source = FIXTURES / "synthetic_phi_oru.hl7"

    redacted = redact_hl7v2(
        source,
        deidentifier=fake_deidentifier,
        date_shift_days=7,
        seed=1,
    )
    parsed = parse_hl7v2(redacted)
    obx_segments = [segment for segment in parsed.segments if segment.name == "OBX"]
    nte = next(segment for segment in parsed.segments if segment.name == "NTE")

    assert parsed.segment_names() == ("MSH", "PID", "OBR", "OBX", "OBX", "NTE")
    assert obx_segments[0].get_field(5) == (
        "Patient [PERSON] called from [PHONE] about [ID_NUM]."
    )
    assert obx_segments[1].get_field(5) == "7.1"
    assert nte.get_field(3) == "Follow-up email [EMAIL] belongs to [PERSON]."
    assert "Jane Roe" not in redacted
    assert "jane.roe@example.com" not in redacted
    assert "GLU^Glucose^L" in redacted


def test_free_text_deidentifier_receives_lang_for_obx_and_nte():
    calls: list[tuple[str, str]] = []

    def deidentifier(text: str, **kwargs):
        calls.append((text, kwargs["lang"]))
        return text.replace("Maria Garcia", "[PERSON]")

    message = "\r".join(
        [
            "MSH|^~\\&|LAB|GOOD HOSPITAL|OPENMED|LOCAL|202401011300||ORU^R01|"
            "MSG00006|P|2.5",
            "OBX|1|FT|NOTE^Clinical note^L||Patient Maria Garcia called.",
            "NTE|1||Follow-up with Maria Garcia.",
        ]
    )

    redacted = redact_hl7v2(
        message,
        deidentifier=deidentifier,
        lang="es",
        date_shift_days=1,
    )

    assert "Maria Garcia" not in redacted
    assert calls == [
        ("Patient Maria Garcia called.", "es"),
        ("Follow-up with Maria Garcia.", "es"),
    ]


def test_date_fields_shift_consistently_within_message():
    message = "\r".join(
        [
            "MSH|^~\\&|ADTAPP|GOOD HOSPITAL|OPENMED|LOCAL|202401011200||ADT^A01|"
            "MSG00003|P|2.5",
            segment(
                "PID",
                19,
                {
                    3: "MRN111^^^GOOD HOSPITAL^MR",
                    5: "DOE^JOHN",
                    7: "19800101",
                },
            ),
            segment(
                "IN1",
                36,
                {
                    16: "DOE^JOHN",
                    18: "19800101",
                    36: "POLICY123",
                },
            ),
        ]
    )

    redacted = redact_hl7v2(
        message,
        deidentifier=fake_deidentifier,
        date_shift_days=45,
        seed=1,
    )
    parsed = parse_hl7v2(redacted)
    pid = next(segment for segment in parsed.segments if segment.name == "PID")
    in1 = next(segment for segment in parsed.segments if segment.name == "IN1")

    assert pid.get_field(7) == "19800215"
    assert in1.get_field(18) == "19800215"
    assert "19800101" not in redacted


def test_preserves_msh_encoding_characters_and_custom_delimiters():
    message = "\r".join(
        [
            "MSH*$%?@*ADTAPP*GOOD HOSPITAL*OPENMED*LOCAL*202401011200**ADT$A01*"
            "MSG00004*P*2.5",
            "PID*1**MRN222$$$GOOD HOSPITAL$MR**DOE$JOHN**19800101",
        ]
    )

    redacted = redact_hl7v2(
        message,
        deidentifier=fake_deidentifier,
        date_shift_days=1,
        seed=1,
    )
    parsed = parse_hl7v2(redacted)
    pid = next(segment for segment in parsed.segments if segment.name == "PID")

    assert redacted.startswith("MSH*$%?@*")
    assert parsed.encoding.field == "*"
    assert parsed.encoding.component == "$"
    assert parsed.segment_names() == ("MSH", "PID")
    assert pid.get_field(7) == "19800102"
    assert "$" in pid.get_field(3)
    assert "$" in pid.get_field(5)
    assert "MRN222" not in redacted
    assert "DOE$JOHN" not in redacted


def test_unknown_segments_pass_through_unless_configured():
    message = "\r".join(
        [
            "MSH|^~\\&|APP|FAC|OPENMED|LOCAL|202401011200||ORU^R01|MSG00005|P|2.5",
            "ZZZ|1|Leave Jane Roe unchanged",
            "ZNT|1|Call Jane Roe at 555-0199",
        ]
    )
    field_map = {
        **DEFAULT_FIELD_MAP,
        ("ZNT", 2): HL7FieldRule("redact_text"),
    }

    redacted = redact_hl7v2(
        message,
        field_map=field_map,
        deidentifier=fake_deidentifier,
        date_shift_days=1,
    )
    parsed = parse_hl7v2(redacted)
    zzz = next(segment for segment in parsed.segments if segment.name == "ZZZ")
    znt = next(segment for segment in parsed.segments if segment.name == "ZNT")

    assert zzz.get_field(2) == "Leave Jane Roe unchanged"
    assert znt.get_field(2) == "Call [PERSON] at [PHONE]"


BLANK_LINE_BASE = "\n".join(
    [
        "MSH|^~\\&|SYNTHETIC|TEST|OPENMED|LOCAL|202401011200||ADT^A01|MSG00001|P|2.5",
        "PID|1||MRN12345^^^TEST^MR||Jane Roe||19800101|F",
        "PV1|1|O",
        "",
    ]
)


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ("no blank lines", BLANK_LINE_BASE),
        ("leading blank line", "\n" + BLANK_LINE_BASE),
        ("interior blank line", BLANK_LINE_BASE.replace("|F\n", "|F\n\n")),
        ("extra trailing blank line", BLANK_LINE_BASE + "\n"),
        (
            "blank lines in every position",
            "\n\n" + BLANK_LINE_BASE.replace("|F\n", "|F\n\n\n") + "\n",
        ),
        ("no trailing separator", BLANK_LINE_BASE.rstrip("\n")),
        ("carriage-return separators", BLANK_LINE_BASE.replace("\n", "\r")),
        ("CRLF separators", BLANK_LINE_BASE.replace("\n", "\r\n")),
    ],
)
def test_serialize_round_trips_messages_containing_blank_lines(
    label: str, message: str
) -> None:
    assert parse_hl7v2(message).serialize() == message, label


def test_blank_lines_are_not_parsed_as_segments():
    parsed = parse_hl7v2("\n" + BLANK_LINE_BASE.replace("|F\n", "|F\n\n"))

    assert parsed.segment_names() == ("MSH", "PID", "PV1")
    assert parsed.blank_line_positions == (0, 3)


def test_redaction_preserves_blank_lines():
    message = BLANK_LINE_BASE.replace("|F\n", "|F\n\n")

    redacted = redact_hl7v2(
        message,
        deidentifier=fake_deidentifier,
        date_shift_days=1,
    )

    assert redacted.count("\n") == message.count("\n")
    assert "\n\n" in redacted
    assert parse_hl7v2(redacted).segment_names() == ("MSH", "PID", "PV1")


def test_serialize_ignores_blank_line_positions_beyond_the_segment_list():
    parsed = parse_hl7v2(BLANK_LINE_BASE)
    parsed.blank_line_positions = (99,)

    assert parsed.serialize() == BLANK_LINE_BASE
