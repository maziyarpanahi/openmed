from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from openmed.interop import adapter_spec, available_adapters, get_adapter
from openmed.interop.hl7v2 import (
    DEFAULT_FIELD_MAP,
    HL7FieldRule,
    parse_hl7v2,
    redact_hl7v2,
)

FIXTURES = Path(__file__).parent / "fixtures"


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
