from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

import openmed.multimodal.base as multimodal_base
from openmed.interop import cda, get_adapter
from openmed.interop.cda import CDA_NAMESPACE, PhiElementRule
from openmed.multimodal import redact_document
from openmed.multimodal.exceptions import UnsupportedDocumentError

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_phi_ccda.xml"
HL7 = {"hl7": CDA_NAMESPACE}


def _parse(xml: str) -> ET.Element:
    return ET.fromstring(xml.encode())


def test_registry_loads_cda_adapter_lazily():
    adapter = get_adapter("cda")

    assert adapter is cda
    assert hasattr(adapter, "redact_cda")


def test_redact_cda_redacts_header_and_narrative_phi():
    redacted = cda.redact_cda(FIXTURE, date_shift_days=30, hash_salt="unit-test")
    root = _parse(str(redacted))

    assert "Jane" not in redacted
    assert "Doe" not in redacted
    assert "MRN12345" not in redacted
    assert "123 Main Street" not in redacted
    assert "+1-555-010-2222" not in redacted
    assert "jane.doe@example.test" not in redacted

    patient_name = root.find(
        ".//hl7:recordTarget/hl7:patientRole/hl7:patient/hl7:name",
        HL7,
    )
    assert patient_name is not None
    assert patient_name.get("nullFlavor") == "UNK"
    assert list(patient_name) == []

    patient_id = root.find(".//hl7:recordTarget/hl7:patientRole/hl7:id", HL7)
    assert patient_id is not None
    assert patient_id.get("root") == "2.16.840.1.113883.19.5"
    assert patient_id.get("extension", "").startswith("h")

    addr = root.find(".//hl7:recordTarget/hl7:patientRole/hl7:addr", HL7)
    telecom = root.find(".//hl7:recordTarget/hl7:patientRole/hl7:telecom", HL7)
    assert addr is not None
    assert telecom is not None
    assert addr.get("nullFlavor") == "UNK"
    assert telecom.get("nullFlavor") == "UNK"
    assert telecom.get("value") is None


def test_redact_cda_preserves_namespace_markup_and_structured_entries():
    redacted = str(cda.redact_cda(FIXTURE, date_shift_days=30))
    root = _parse(redacted)

    assert root.tag == f"{{{CDA_NAMESPACE}}}ClinicalDocument"
    assert f'xmlns="{CDA_NAMESPACE}"' in redacted
    assert "<paragraph>" in redacted
    assert "<content>[ADDRESS]</content>" in redacted

    code = root.find(".//hl7:entry/hl7:observation/hl7:code", HL7)
    value = root.find(".//hl7:entry/hl7:observation/hl7:value", HL7)
    assert code is not None
    assert value is not None
    assert code.get("code") == "ASSERTION"
    assert value.get("displayName") == "No known drug allergy"


def test_redact_cda_sweeps_narrative_only_structured_phi():
    xml = """
    <ClinicalDocument xmlns="urn:hl7-org:v3">
      <templateId root="2.16.840.1.113883.10.20.22.1.1"/>
      <component>
        <structuredBody>
          <component>
            <section>
              <text>
                <paragraph>Seen at <content>99 Cedar Lane</content> on January 15, 2024.</paragraph>
                <paragraph>Call 212-555-0171. MRN ABC123456.</paragraph>
              </text>
            </section>
          </component>
        </structuredBody>
      </component>
    </ClinicalDocument>
    """

    redacted = str(cda.redact_cda(xml, date_shift_days=30))

    assert "99 Cedar Lane" not in redacted
    assert "January 15, 2024" not in redacted
    assert "212-555-0171" not in redacted
    assert "ABC123456" not in redacted
    assert "[ADDRESS]" in redacted
    assert "[DATE]" in redacted


def test_redact_cda_redacts_known_surface_spanning_narrative_nodes():
    xml = """
    <ClinicalDocument xmlns="urn:hl7-org:v3">
      <templateId root="2.16.840.1.113883.10.20.22.1.1"/>
      <recordTarget>
        <patientRole>
          <patient><name><given>Jane</given><family>Doe</family></name></patient>
        </patientRole>
      </recordTarget>
      <component>
        <structuredBody>
          <component>
            <section>
              <text>
                <paragraph><content>Jane</content> <content>Doe</content> arrived.</paragraph>
              </text>
            </section>
          </component>
        </structuredBody>
      </component>
    </ClinicalDocument>
    """

    redacted = str(cda.redact_cda(xml, date_shift_days=30))

    assert "Jane" not in redacted
    assert "Doe" not in redacted
    assert "[PERSON]" in redacted
    _parse(redacted)


def test_redact_cda_shifts_birth_time_and_effective_times_consistently():
    root = _parse(str(cda.redact_cda(FIXTURE, date_shift_days=30)))

    birth_time = root.find(
        ".//hl7:recordTarget/hl7:patientRole/hl7:patient/hl7:birthTime",
        HL7,
    )
    effective_times = root.findall(".//hl7:effectiveTime", HL7)

    assert birth_time is not None
    assert birth_time.get("value") == "19700214"
    assert [item.get("value") for item in effective_times] == [
        "20240204",
        "20240204",
    ]


def test_phi_element_map_is_extensible_without_parser_changes():
    xml = """
    <ClinicalDocument xmlns="urn:hl7-org:v3">
      <templateId root="2.16.840.1.113883.10.20.22.1.1"/>
      <guardian>
        <guardianPerson><name><given>Greg</given><family>Guardian</family></name></guardianPerson>
      </guardian>
    </ClinicalDocument>
    """
    rules = (
        *cda.DEFAULT_PHI_ELEMENT_MAP,
        PhiElementRule(
            ".//hl7:guardian/hl7:guardianPerson/hl7:name",
            "null_flavor",
            label="PERSON",
        ),
    )

    redacted = str(cda.redact_cda(xml, element_map=rules, date_shift_days=30))
    root = _parse(redacted)
    guardian_name = root.find(".//hl7:guardian/hl7:guardianPerson/hl7:name", HL7)

    assert "Greg" not in redacted
    assert "Guardian" not in redacted
    assert guardian_name is not None
    assert guardian_name.get("nullFlavor") == "UNK"


def test_cda_detection_does_not_match_generic_xml(tmp_path):
    generic = tmp_path / "generic.xml"
    generic.write_text("<root><ClinicalDocument>not CDA</ClinicalDocument></root>")

    assert cda.is_cda_document(FIXTURE)
    assert not cda.is_cda_document(generic)


def test_cda_parser_rejects_doctype_and_entities():
    xml = """
    <!DOCTYPE ClinicalDocument [
      <!ENTITY leak SYSTEM "file:///etc/passwd">
    ]>
    <ClinicalDocument xmlns="urn:hl7-org:v3">
      <templateId root="2.16.840.1.113883.10.20.22.1.1"/>
      <component>&leak;</component>
    </ClinicalDocument>
    """

    assert not cda.is_cda_document(xml)
    with pytest.raises(ValueError, match="DOCTYPE or ENTITY"):
        cda.redact_cda(xml)


def test_namespace_less_cda_detection_and_redaction_agree():
    xml = """
    <ClinicalDocument>
      <templateId root="2.16.840.1.113883.10.20.22.1.1"/>
      <recordTarget>
        <patientRole>
          <id extension="MRN5555"/>
          <patient><name><given>Sam</given><family>Sample</family></name></patient>
        </patientRole>
      </recordTarget>
    </ClinicalDocument>
    """

    redacted = str(cda.redact_cda(xml, date_shift_days=30))

    assert cda.is_cda_document(xml)
    assert "Sam" not in redacted
    assert "Sample" not in redacted
    assert "MRN5555" not in redacted


def test_registered_xml_handler_does_not_hijack_non_cda_xml(tmp_path, monkeypatch):
    generic = tmp_path / "generic.xml"
    generic.write_text("<root><patient>Jane Doe</patient></root>")

    monkeypatch.setattr(multimodal_base, "_missing_multimodal_dependencies", lambda: [])

    with pytest.raises(UnsupportedDocumentError, match="matched content"):
        redact_document(generic)


def test_registered_xml_handler_routes_cda_xml_without_multimodal_extra(monkeypatch):
    monkeypatch.setattr(
        multimodal_base,
        "_missing_multimodal_dependencies",
        lambda: ["pdfplumber", "python-docx"],
    )

    doc = redact_document(FIXTURE, models=lambda text: text.replace("Patient", "Pt"))

    assert doc.metadata["format"] == "cda"
    assert "Jane" not in doc.text
    assert "Pt [PERSON]" in doc.text
