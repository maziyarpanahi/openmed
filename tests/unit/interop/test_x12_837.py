from __future__ import annotations

from pathlib import Path

import pytest

from openmed.interop.x12_837 import parse_x12_837, redact_x12_837

# This fixture is entirely synthetic. Names, identifiers, addresses, dates,
# providers, and claim values do not describe a real person or claim.
SYNTHETIC_837 = "\n".join(
    [
        "ISA*00*          *00*          *ZZ*SYNTHSUBMIT001 *ZZ*SYNTHRECEIVE01 "
        "*240101*1200*^*00501*000000001*1*T*:~",
        "GS*HC*SYNTHSUBMIT*SYNTHRECV*20240101*1200*1*X*005010X222A1~",
        "ST*837*0001*005010X222A1~",
        "BHT*0019*00*SYNTHCLAIM*20240101*1200*CH~",
        "NM1*41*2*SYNTHETIC SUBMITTER*****46*SUBMIT001~",
        "NM1*40*2*SYNTHETIC RECEIVER*****46*RECV001~",
        "HL*1**20*1~",
        "NM1*85*2*SYNTHETIC CLINIC*****XX*1234567893~",
        "N3*100 SYNTHETIC WAY~",
        "N4*TESTVILLE*CA*90000~",
        "REF*EI*999999999~",
        "HL*2*1*22*1~",
        "SBR*P*18*SYNTHGROUP******CI~",
        "NM1*IL*1*SYNTHLAST*SYNTHFIRST*SYNTHMIDDLE***MI*SYNTHSUB123~",
        "N3*200 SAMPLE STREET*APT 3~",
        "N4*EXAMPLE CITY*CA*90210~",
        "DMG*D8*19800101*F~",
        "REF*SY*SYNTHSUB999~",
        "HL*3*2*23*0~",
        "PAT*19~",
        "NM1*QC*1*SYNTHPATIENT*SYNTHGIVEN****MI*SYNTHPAT456~",
        "N3*300 FIXTURE ROAD~",
        "N4*DEMO TOWN*NY*10001~",
        "DMG*D8*20100102*M~",
        "REF*1W*SYNTHPAT999~",
        "CLM*SYNTHCLAIM01*125***11:B:1*Y*A*Y*I~",
        "NM1*82*1*SYNTHDOCTOR*SYNTHDOC****XX*1111111111~",
        "REF*0B*SYNTHLICENSE~",
        "LX*1~",
        "SV1*HC:99213*125*UN*1***1~",
        "DTP*472*D8*20240115~",
        "SE*30*0001~",
        "GE*1*1~",
        "IEA*1*000000001~",
        "",
    ]
)


def _values(message, tag: str) -> list[tuple[str, ...]]:
    return [
        tuple(element.value for element in segment.elements)
        for segment in message.segments
        if segment.tag == tag
    ]


def test_synthetic_837_parses_and_reserializes_with_exact_framing():
    parsed = parse_x12_837(SYNTHETIC_837)

    assert parsed.serialize() == SYNTHETIC_837
    assert parsed.delimiters.element == "*"
    assert parsed.delimiters.repetition == "^"
    assert parsed.delimiters.component == ":"
    assert parsed.delimiters.segment == "~"
    assert parsed.segment_names()[:4] == ("ISA", "GS", "ST", "BHT")
    assert parsed.segment_names()[-3:] == ("SE", "GE", "IEA")
    assert parsed.segments[1].leading == "\n"
    assert parsed.suffix == "\n"


def test_redacts_patient_subscriber_and_provider_elements_only():
    original = parse_x12_837(SYNTHETIC_837)
    result = redact_x12_837(SYNTHETIC_837)
    redacted = parse_x12_837(result.deidentified_text)

    for value in (
        "SYNTHETIC CLINIC",
        "1234567893",
        "SYNTHLAST",
        "SYNTHFIRST",
        "SYNTHSUB123",
        "200 SAMPLE STREET",
        "19800101",
        "SYNTHSUB999",
        "SYNTHPATIENT",
        "SYNTHGIVEN",
        "SYNTHPAT456",
        "300 FIXTURE ROAD",
        "20100102",
        "SYNTHPAT999",
        "SYNTHDOCTOR",
        "SYNTHLICENSE",
    ):
        assert value not in result.deidentified_text

    assert "SYNTHETIC SUBMITTER" in result.deidentified_text
    assert "SYNTHETIC RECEIVER" in result.deidentified_text
    assert _values(original, "ISA") == _values(redacted, "ISA")
    assert _values(original, "GS") == _values(redacted, "GS")
    assert _values(original, "ST") == _values(redacted, "ST")
    assert _values(original, "BHT") == _values(redacted, "BHT")
    assert _values(original, "HL") == _values(redacted, "HL")
    assert _values(original, "SBR") == _values(redacted, "SBR")
    assert _values(original, "CLM") == _values(redacted, "CLM")
    assert _values(original, "LX") == _values(redacted, "LX")
    assert _values(original, "SV1") == _values(redacted, "SV1")
    assert _values(original, "DTP") == _values(redacted, "DTP")
    assert _values(original, "SE") == _values(redacted, "SE")
    assert _values(original, "GE") == _values(redacted, "GE")
    assert _values(original, "IEA") == _values(redacted, "IEA")

    subscriber = next(
        segment
        for segment in redacted.segments
        if segment.tag == "NM1" and segment.get_value(1) == "IL"
    )
    assert subscriber.get_value(2) == "1"
    assert subscriber.get_value(3) == "REDACTED"
    assert subscriber.get_value(8) == "MI"
    assert subscriber.get_value(9) == "REDACTED"
    assert result.deidentified_text.endswith("IEA*1*000000001~\n")


def test_offset_map_round_trips_raw_audit_provenance():
    result = redact_x12_837(SYNTHETIC_837)
    redaction = next(
        item
        for item in result.redactions
        if item.entity_code == "IL"
        and item.segment_tag == "NM1"
        and item.element_position == 9
    )

    assert SYNTHETIC_837[redaction.source_start : redaction.source_end] == (
        "SYNTHSUB123"
    )
    assert (
        result.deidentified_text[redaction.output_start : redaction.output_end]
        == "REDACTED"
    )
    assert result.offset_map.source_to_output(*redaction.source_span) == (
        redaction.output_span
    )
    assert result.offset_map.output_to_source(*redaction.output_span) == (
        redaction.source_span
    )
    assert len(redaction.original_sha256) == 64
    assert "SYNTHSUB123" not in repr(redaction)

    st_control = result.offset_map.for_element(2, 2)
    assert SYNTHETIC_837[st_control.source_start : st_control.source_end] == "0001"
    assert (
        result.deidentified_text[st_control.output_start : st_control.output_end]
        == "0001"
    )


def test_redaction_preserves_composite_and_repetition_structure():
    source = SYNTHETIC_837.replace("SYNTHSUB999", "SYNTHSUB999:PART2^SYNTHALT999")

    result = redact_x12_837(source, replacement="MASKED")

    assert "REF*SY*MASKED:MASKED^MASKED~" in result.deidentified_text
    ref = next(
        item
        for item in result.redactions
        if item.entity_code == "IL" and item.segment_tag == "REF"
    )
    assert ref.replacement == "MASKED:MASKED^MASKED"


def test_reads_utf8_path_without_normalizing_intersegment_newlines(tmp_path: Path):
    path = tmp_path / "synthetic-claim.837"
    path.write_text(SYNTHETIC_837.replace("\n", "\r\n"), encoding="utf-8", newline="")

    result = redact_x12_837(path)

    assert "~\r\nGS" in result.deidentified_text
    assert result.deidentified_text.endswith("IEA*1*000000001~\r\n")


def test_rejects_non_837_and_separator_breaking_replacements():
    with pytest.raises(ValueError, match="only X12 837"):
        parse_x12_837(SYNTHETIC_837.replace("ST*837", "ST*835", 1))

    with pytest.raises(ValueError, match="must not contain X12 separators"):
        redact_x12_837(SYNTHETIC_837, replacement="NOT*SAFE")
