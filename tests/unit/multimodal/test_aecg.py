"""Synthetic decoding and fail-closed aECG parser tests."""

import dataclasses
import hashlib
import json
import traceback
from pathlib import Path

import pytest

from openmed.multimodal.aecg import AECG_NOTICE, AecgError, AecgLimits, read_aecg

FIXTURE = Path(__file__).parents[2] / "fixtures/multimodal/aecg.xml"


def xml() -> bytes:
    return FIXTURE.read_bytes()


def mutate(old: bytes, new: bytes) -> bytes:
    assert old in xml()
    return xml().replace(old, new, 1)


def assert_rejected(data: bytes, reason: str, **kwargs) -> None:
    with pytest.raises(AecgError) as caught:
        read_aecg(data, **kwargs)
    assert caught.value.reason_code == reason
    assert str(caught.value) == reason
    assert caught.value.__context__ is None
    assert "SYNTHETIC-ID" not in "".join(traceback.format_exception(caught.value))
    assert "架空" not in repr(caught.value)


def test_hand_checked_scaled_leads_and_separate_representative_beat():
    record = read_aecg(xml())
    assert len(record.rhythm) == len(record.derived) == 1
    assert record.rhythm[0].kind == "rhythm"
    assert record.derived[0].kind == "derived_representative_beat"
    assert record.rhythm[0].sample_interval_seconds == 0.002
    assert record.rhythm[0].leads[0].code == "MDC_ECG_LEAD_I"
    assert record.rhythm[0].leads[0].samples == (-4.0, 1.0, 11.0)
    assert record.rhythm[0].leads[1].samples == (-3.0, -2.0, 0.0)
    assert record.derived[0].leads[0].samples == (1.0, 6.0)
    assert record.derived[0].leads[1].samples == (-2.0, -1.0)
    assert record.source_digest == hashlib.sha256(xml()).hexdigest()
    assert record == read_aecg(xml())
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.annotation_count = 0


def test_all_public_outputs_withhold_identifiers_and_free_text():
    record = read_aecg(xml())
    assert record.demographics_present and record.demographic_element_count == 4
    assert record.annotations_present and record.annotation_count == 2
    public = (
        repr(record)
        + json.dumps(dataclasses.asdict(record))
        + json.dumps(record.report())
    )
    for sentinel in (
        "SYNTHETIC-ID-3839",
        "架空",
        "Patient 3839",
        "19000101",
        "20000101000000",
        "SYNTHETIC-NOTE-3839",
        "patient@example.invalid",
    ):
        assert sentinel not in public
    assert "samples" not in json.dumps(record.report())
    assert record.report()["review_required"] is True
    assert record.notice == record.report()["notice"] == AECG_NOTICE
    assert record.confirm_review(reviewer_confirmed=True) is record
    for unconfirmed in (False, None, 1, "yes"):
        with pytest.raises(AecgError, match="^review_required$"):
            record.confirm_review(reviewer_confirmed=unconfirmed)


@pytest.mark.parametrize(
    "declaration",
    [
        b'<!DOCTYPE AnnotatedECG [<!ENTITY phi "SYNTHETIC-ID-3839">]>',
        b'<!DOCTYPE AnnotatedECG SYSTEM "file:///private/SYNTHETIC-ID-3839">',
        b'<!DOCTYPE AnnotatedECG SYSTEM "https://example.invalid/external.dtd">',
        b'<!DOCTYPE AnnotatedECG [<!ENTITY a "ha"><!ENTITY b "&a;&a;"><!ENTITY c "&b;&b;">]>',
        b'<!ENTITY secret SYSTEM "https://example.invalid">',
    ],
)
def test_entities_and_external_dtds_fail_closed(declaration, monkeypatch):
    import socket

    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network access"))
    assert_rejected(
        xml().replace(b"<AnnotatedECG ", declaration + b"<AnnotatedECG "), "unsafe_xml"
    )


@pytest.mark.parametrize(
    "field,maximum,reason",
    [
        ("max_bytes", 100, "byte_limit"),
        ("max_elements", 10, "element_limit"),
        ("max_depth", 3, "depth_limit"),
        ("max_series", 1, "series_limit"),
        ("max_leads_per_series", 1, "lead_limit"),
        ("max_samples_per_lead", 2, "sample_limit"),
        ("max_total_samples", 9, "sample_limit"),
    ],
)
def test_each_resource_bound(field, maximum, reason):
    assert_rejected(xml(), reason, limits=AecgLimits(**{field: maximum}))


def test_sample_bounds_are_inclusive_and_total_includes_derived():
    record = read_aecg(
        xml(),
        limits=AecgLimits(
            max_bytes=len(xml()),
            max_series=2,
            max_leads_per_series=2,
            max_samples_per_lead=3,
            max_total_samples=10,
        ),
    )
    assert len(record.derived[0].leads) == 2


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10"])
def test_invalid_limit_values(value):
    with pytest.raises(AecgError, match="^invalid_limits$"):
        AecgLimits(max_elements=value)


@pytest.mark.parametrize(
    "old,new,reason",
    [
        (b"<digits>-2 0 4</digits>", b"<digits>-2 0</digits>", "inconsistent_lengths"),
        (b"<digits>-2 0 4</digits>", b"<digits/>", "unsupported_layout"),
        (
            b"<digits>-2 0 4</digits>",
            b"<digits>NaN SYNTHETIC-ID-3839</digits>",
            "invalid_numeric",
        ),
        (b"<digits>-2 0 4</digits>", b"<digits>1.5 0 4</digits>", "invalid_numeric"),
        (
            b"<digits>-2 0 4</digits>",
            b"<digits>2147483648 0 4</digits>",
            "invalid_numeric",
        ),
        (
            b"<digits>-2 0 4</digits>",
            b"<digits><text>1</text></digits>",
            "unsupported_encoding",
        ),
        (b'xsi:type="SLIST_PQ"', b'xsi:type="B64"', "unsupported_encoding"),
        (
            b"<digits>-2 0 4</digits>",
            b'<digits encoding="base64">123</digits>',
            "unsupported_encoding",
        ),
        (
            b'xsi:type="SLIST_PQ"',
            b'xsi:type="SLIST_PQ" nullFlavor="UNK"',
            "unsupported_encoding",
        ),
        (
            b"<sequenceSet>",
            b"<sequenceSet><component><unknown/></component>",
            "unsupported_layout",
        ),
        (b'<origin value="1"', b'<origin value="NaN"', "invalid_numeric"),
        (b'<scale value="2.5"', b'<scale value="0"', "invalid_numeric"),
        (b'<scale value="2.5"', b'<scale value="1e309"', "invalid_numeric"),
        (b'<scale value="2.5"', b'<scale value="1e308"', "invalid_numeric"),
        (b'<increment value="0.002"', b'<increment value="-1"', "invalid_numeric"),
        (b'unit="s"', b'unit="ms"', "unsupported_encoding"),
        (b'code="RHYTHM"', b'code="SYNTHETIC-ID-3839"', "unsupported_layout"),
        (b'code="REPRESENTATIVE_BEAT"', b'code="RHYTHM"', "unsupported_layout"),
        (b'code="MDC_ECG_LEAD_I"', b'code="SYNTHETIC-ID-3839"', "unsupported_lead"),
        (b'code="MDC_ECG_LEAD_II"', b'code="MDC_ECG_LEAD_I"', "unsupported_layout"),
        (b"<sequenceSet>", b"<sequenceSet><sequence/>", "unsupported_layout"),
        (
            b'<origin value="1" unit="uV"/>',
            b'<origin value="1" unit="mV"/>',
            "unsupported_encoding",
        ),
        (
            b'<origin value="1" unit="uV"/>',
            b'<origin value="1" unit="uV"/><origin value="2" unit="uV"/>',
            "unsupported_layout",
        ),
        (
            b"<sequenceSet>",
            b'<sequenceSet><reference value="file:///private/SYNTHETIC-ID-3839"/>',
            "external_reference",
        ),
        (
            b"<sequenceSet>",
            b'<sequenceSet><xi:include xmlns:xi="http://www.w3.org/2001/XInclude" href="https://example.invalid"/>',
            "external_reference",
        ),
    ],
)
def test_unsupported_and_malformed_series(old, new, reason):
    assert_rejected(mutate(old, new), reason)


@pytest.mark.parametrize(
    "data,reason",
    [
        (b"<AnnotatedECG>", "malformed_xml"),
        (b'<AnnotatedECG xmlns="urn:wrong"/>', "unsupported_layout"),
        (b'<AnnotatedECG xmlns="urn:hl7-org:v3"/>', "unsupported_layout"),
        (b"\xff", "unsupported_encoding"),
        (b"SYNTHETIC-ID-3839\xff", "unsupported_encoding"),
        ("private.xml", "invalid_input"),
    ],
)
def test_document_and_input_rejections(data, reason):
    assert_rejected(data, reason)


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32", "iso-8859-1"])
def test_unsupported_xml_encodings(encoding):
    assert_rejected(
        xml().decode().replace("Patient", "Patiént").encode(encoding, errors="replace"),
        "unsupported_encoding",
    )
    assert_rejected(
        mutate(b'encoding="UTF-8"', ('encoding="' + encoding + '"').encode()),
        "unsupported_encoding",
    )


def test_namespace_prefixes_relative_time_and_negative_scale():
    data = (
        xml()
        .replace(b"xmlns:xsi=", b"xmlns:types=")
        .replace(b"xsi:type", b"types:type")
    )
    data = data.replace(b"TIME_ABSOLUTE", b"TIME_RELATIVE").replace(
        b"GLIST_TS", b"GLIST_PQ"
    )
    data = data.replace(b'scale value="2.5"', b'scale value="-2.5"')
    assert read_aecg(data).rhythm[0].leads[0].samples == (6.0, 1.0, -9.0)


def test_duplicate_time_and_nested_series_rejected():
    import copy
    from xml.etree import ElementTree as ET

    ns = "{urn:hl7-org:v3}"
    root = ET.fromstring(xml())
    sequence_set = next(root.iter(ns + "sequenceSet"))
    sequence_set.append(copy.deepcopy(sequence_set[0]))
    assert_rejected(ET.tostring(root), "unsupported_layout")
    root = ET.fromstring(xml())
    series = next(root.iter(ns + "series"))
    series.append(copy.deepcopy(series))
    assert_rejected(ET.tostring(root), "unsupported_layout")


def test_withheld_metadata_sweep_and_absent_presence_flags():
    from xml.etree import ElementTree as ET

    ns = "{urn:hl7-org:v3}"
    root = ET.fromstring(xml())
    root.remove(root.find(ns + "componentOf"))
    series = next(root.iter(ns + "series"))
    series.remove(series.find(ns + "subjectOf"))
    data = ET.tostring(root)
    record = read_aecg(data)
    assert not record.demographics_present and not record.annotations_present
    for sentinel in (
        "SYNTHETIC-ID-3839",
        "架空患者3839",
        "patient@example.invalid",
        "06-12-34-56-78",
        "19000101",
    ):
        mutated = data.replace(
            b"<ns0:series>",
            (
                '<ns0:series><ns0:id extension="'
                + sentinel
                + '"/><ns0:text>'
                + sentinel
                + "</ns0:text>"
            ).encode(),
        )
        assert mutated != data
        result = read_aecg(mutated)
        assert result.rhythm == record.rhythm and result.derived == record.derived
        assert sentinel not in repr(result) + json.dumps(
            dataclasses.asdict(result)
        ) + json.dumps(result.report(), ensure_ascii=False)


def test_utf8_bom_and_all_source_voltage_units():
    assert read_aecg(b"\xef\xbb\xbf" + xml()).rhythm[0].leads[0].samples == (
        -4.0,
        1.0,
        11.0,
    )
    for unit in (b"uV", b"mV", b"V"):
        result = read_aecg(xml().replace(b'unit="uV"', b'unit="' + unit + b'"'))
        assert result.rhythm[0].leads[0].unit == unit.decode()
        assert result.rhythm[0].leads[0].samples == (-4.0, 1.0, 11.0)
