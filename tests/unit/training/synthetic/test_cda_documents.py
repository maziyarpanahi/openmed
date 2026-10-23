"""Synthetic unit tests for seeded C-CDA documents and the leakage harness."""

from __future__ import annotations

import dataclasses
import json
import re
from xml.etree import ElementTree as ET
from xml.parsers import expat

import pytest

from openmed.interop.cda import (
    CDA_NAMESPACE,
    DEFAULT_PHI_ELEMENT_MAP,
    PhiElementRule,
    is_cda_document,
    redact_cda,
)
from openmed.training.synthetic.cda_documents import (
    CDA_PLACEHOLDER_CODE_SYSTEM,
    CDA_SYNTHETIC_DOCUMENT_TYPES,
    CDA_SYNTHETIC_SCHEMA_VERSION,
    MAX_CDA_DOCUMENTS,
    CdaDocumentGenerator,
    CdaLeakageError,
    CdaSynthesisConfig,
    CdaSynthesisError,
    SyntheticCdaDocument,
    evaluate_cda_leakage,
    generate_cda_documents,
    resolve_cda_locator,
)

HL7 = {"hl7": CDA_NAMESPACE}
_SEGMENT = re.compile(r"^(?P<name>[A-Za-z]+)(?:\[(?P<index>\d+)\])?$")
_RESTRICTED_CODE_SYSTEMS = (
    "2.16.840.1.113883.6.1",  # LOINC
    "2.16.840.1.113883.6.96",  # SNOMED CT
    "2.16.840.1.113883.6.12",  # CPT
    "2.16.840.1.113883.6.88",  # RxNorm
    "2.16.840.1.113883.6.90",  # ICD-10-CM
)


@pytest.fixture(scope="module")
def documents() -> tuple[SyntheticCdaDocument, ...]:
    return generate_cda_documents(9, seed=3885)


def _independent_resolve(xml: str, path: str, node: str, attribute: str | None):
    """Resolve an indexed truth path without using the module under test."""

    root = ET.fromstring(xml.encode("utf-8"))
    segments = path.strip("/").split("/")
    assert segments[0] == "ClinicalDocument"
    element = root
    for segment in segments[1:]:
        match = _SEGMENT.match(segment)
        assert match is not None, segment
        tag = f"{{{CDA_NAMESPACE}}}{match.group('name')}"
        matches = [child for child in element if child.tag == tag]
        element = matches[int(match.group("index") or 1) - 1]
    if node == "attribute":
        return element.get(attribute)
    if node == "text":
        return element.text
    assert node == "tail"
    return element.tail


def _parse_without_dtd(data: bytes) -> None:
    parser = expat.ParserCreate(namespace_separator=" ")
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)

    def refuse(*_args):
        raise AssertionError("DTD or entity declaration present")

    parser.StartDoctypeDeclHandler = refuse
    parser.EntityDeclHandler = refuse
    parser.ExternalEntityRefHandler = refuse
    parser.Parse(data, True)


# Documents ---------------------------------------------------------------


def test_every_document_type_passes_is_cda_document(documents) -> None:
    assert {doc.document_type for doc in documents} == set(CDA_SYNTHETIC_DOCUMENT_TYPES)
    for doc in documents:
        assert is_cda_document(doc.xml)
        assert is_cda_document(doc.to_bytes())
        root = ET.fromstring(doc.to_bytes())
        assert root.tag == f"{{{CDA_NAMESPACE}}}ClinicalDocument"


def test_documents_parse_without_dtd_or_entity_expansion(documents) -> None:
    for doc in documents:
        data = doc.to_bytes()
        assert b"<!DOCTYPE" not in data.upper()
        assert b"<!ENTITY" not in data.upper()
        assert b"SYSTEM" not in data
        assert b"schemaLocation" not in data
        assert b"&" not in data
        _parse_without_dtd(data)


def test_documents_are_ascii_and_contain_no_remote_references(documents) -> None:
    for doc in documents:
        assert doc.xml.isascii()
        assert "://" not in doc.xml


def test_every_truth_locator_resolves_to_its_value(documents) -> None:
    total = 0
    for doc in documents:
        assert doc.truth, doc.document_type
        for truth in doc.truth:
            node_value = _independent_resolve(
                doc.xml, truth.path, truth.node, truth.attribute
            )
            assert node_value is not None, truth.truth_id
            assert node_value[truth.start : truth.end] == truth.value
            assert doc.xml[truth.document_start : truth.document_end] == truth.value
            resolved = resolve_cda_locator(doc.xml, truth)
            assert resolved == node_value
            assert resolve_cda_locator(ET.fromstring(doc.to_bytes()), truth) == (
                node_value
            )
            total += 1
    assert total > 200


def test_truth_records_carry_locator_offsets_and_labels(documents) -> None:
    doc = documents[0]
    ids = [truth.truth_id for truth in doc.truth]
    assert len(ids) == len(set(ids))
    starts = [truth.document_start for truth in doc.truth]
    assert starts == sorted(starts)
    nodes = {truth.node for truth in doc.truth}
    assert {"attribute", "text", "tail"} <= {
        truth.node for item in documents for truth in item.truth
    }
    assert nodes <= {"attribute", "text", "tail"}
    for truth in doc.truth:
        assert (truth.attribute is not None) == (truth.node == "attribute")
        assert "[" not in truth.element_path
        assert truth.locator.startswith(truth.path)
        record = truth.to_dict()
        assert list(record) == [
            "truth_id",
            "label",
            "role",
            "context",
            "path",
            "node",
            "attribute",
            "start",
            "end",
            "document_start",
            "document_end",
            "value",
        ]


def test_telecom_truth_offsets_skip_the_uri_scheme(documents) -> None:
    telecoms = [
        truth
        for doc in documents
        for truth in doc.truth
        if truth.node == "attribute" and truth.path.endswith("telecom[1]")
    ]
    assert telecoms
    for truth in telecoms:
        assert truth.attribute == "value"
        assert truth.start == len("tel:")
        assert truth.label == "PHONE"
    emails = [
        truth for doc in documents for truth in doc.truth if truth.label == "EMAIL"
    ]
    assert emails
    assert all(truth.start == len("mailto:") for truth in emails)


def test_headers_cover_required_participants(documents) -> None:
    required = {
        "patient",
        "guardian",
        "birthplace",
        "next_of_kin",
        "informant",
        "author",
        "custodian",
        "legal_authenticator",
    }
    for doc in documents:
        header_roles = {t.role for t in doc.truth if t.context == "header"}
        assert required <= header_roles, doc.document_type
    paths = {t.element_path for doc in documents for t in doc.truth}
    for expected in (
        "/ClinicalDocument/recordTarget/patientRole/patient/guardian/guardianPerson"
        "/name/given/text()",
        "/ClinicalDocument/recordTarget/patientRole/patient/birthplace/place/addr"
        "/city/text()",
        "/ClinicalDocument/participant/associatedEntity/associatedPerson/name"
        "/family/text()",
        "/ClinicalDocument/informant/relatedEntity/relatedPerson/name/given/text()",
        "/ClinicalDocument/author/assignedAuthor/assignedPerson/name/family/text()",
        "/ClinicalDocument/custodian/assignedCustodian"
        "/representedCustodianOrganization/name/text()",
    ):
        assert expected in paths


def test_document_types_use_distinct_templates(documents) -> None:
    templates = {}
    for doc in documents:
        root = ET.fromstring(doc.to_bytes())
        roots = {item.get("root") for item in root.findall("hl7:templateId", HL7)}
        templates.setdefault(doc.document_type, set()).update(roots)
    assert "2.16.840.1.113883.10.20.22.1.2" in templates["ccd"]
    assert "2.16.840.1.113883.10.20.22.1.8" in templates["discharge_summary"]
    assert "2.16.840.1.113883.10.20.22.1.9" in templates["progress_note"]


def test_narrative_uses_paragraphs_tables_lists_and_content_ids(documents) -> None:
    for doc in documents:
        root = ET.fromstring(doc.to_bytes())
        narrative = root.findall(".//hl7:section/hl7:text", HL7)
        assert narrative
        tags = {
            child.tag.rsplit("}", 1)[1] for block in narrative for child in block.iter()
        }
        assert {"paragraph", "table", "list", "content"} <= tags, doc.document_type
        assert doc.content_references, doc.document_type


def test_entry_references_resolve_to_narrative_content_ids(documents) -> None:
    for doc in documents:
        root = ET.fromstring(doc.to_bytes())
        ids = {
            node.get("ID"): node
            for node in root.iter(f"{{{CDA_NAMESPACE}}}content")
            if node.get("ID")
        }
        references = [
            node.get("value")
            for node in root.findall(".//hl7:entry//hl7:reference", HL7)
        ]
        assert references
        assert all(value.startswith("#") for value in references)
        assert {value[1:] for value in references} <= set(ids)
        for reference in doc.content_references:
            content = _independent_resolve(
                doc.xml, reference.content_path, "attribute", "ID"
            )
            pointer = _independent_resolve(
                doc.xml, reference.reference_path, "attribute", "value"
            )
            assert content == reference.content_id
            assert pointer == "#" + reference.content_id
            assert "/entry[" in reference.reference_path
            assert "/text[1]/" in reference.content_path


def test_narrative_mentions_link_to_structured_values(documents) -> None:
    linked_roles = set()
    for doc in documents:
        by_id = {truth.truth_id: truth for truth in doc.truth}
        assert doc.links, doc.document_type
        for link in doc.links:
            mention = by_id[link.mention_id]
            values = [by_id[item] for item in link.value_ids]
            assert mention.context == "narrative"
            assert values
            assert all(value.context in {"header", "entry"} for value in values)
            assert all(value.role == mention.role for value in values)
            if mention.label == "PERSON":
                assert mention.value == " ".join(value.value for value in values)
            elif mention.label == "PHONE":
                assert [value.value for value in values] == [mention.value]
            elif mention.label == "DATE":
                digits = re.sub(r"\D", "", mention.value)
                if len(digits) == 8:
                    assert values[0].value.startswith(digits)
            if link.content_id is not None:
                assert mention.path.rsplit("/", 1)[-1].startswith("content[")
            linked_roles.add(mention.role)
        narrative_ids = {t.truth_id for t in doc.truth if t.context == "narrative"}
        assert {link.mention_id for link in doc.links} <= narrative_ids
    assert {
        "patient",
        "guardian",
        "next_of_kin",
        "informant",
        "author",
        "custodian",
        "performer",
    } <= linked_roles


def test_manifest_is_json_and_value_bearing_only_in_truth(documents) -> None:
    manifest = documents[0].manifest()
    assert manifest["schema_version"] == CDA_SYNTHETIC_SCHEMA_VERSION
    assert manifest["synthetic"] is True
    assert manifest["contains_real_phi"] is False
    assert manifest["xml_sha256"] == documents[0].xml_sha256
    assert manifest["code_system"] == CDA_PLACEHOLDER_CODE_SYSTEM
    json.dumps(manifest)
    assert len(documents[0].manifest_sha256) == 64


# Reproducibility and terminology --------------------------------------------


def test_same_seed_is_byte_identical_and_different_seed_differs() -> None:
    first = generate_cda_documents(4, seed=11)
    second = generate_cda_documents(4, seed=11)
    other = generate_cda_documents(4, seed=12)
    assert [doc.to_bytes() for doc in first] == [doc.to_bytes() for doc in second]
    assert [doc.manifest_sha256 for doc in first] == [
        doc.manifest_sha256 for doc in second
    ]
    assert all(a.to_bytes() != b.to_bytes() for a, b in zip(first, other, strict=True))
    assert len({doc.to_bytes() for doc in first}) == len(first)


def test_generator_rotates_types_and_honors_override() -> None:
    generator = CdaDocumentGenerator(seed=5)
    types = [generator.generate().document_type for _ in range(3)]
    assert types == list(CDA_SYNTHETIC_DOCUMENT_TYPES)
    assert generator.generate("progress_note").document_type == "progress_note"
    batch = CdaDocumentGenerator(seed=5).generate_batch(3)
    assert [doc.index for doc in batch] == [0, 1, 2]


def test_terminology_codes_are_placeholders(documents) -> None:
    for doc in documents:
        root = ET.fromstring(doc.to_bytes())
        coded = [node for node in root.iter() if node.get("code") is not None]
        assert coded
        for node in coded:
            assert node.get("codeSystem") == CDA_PLACEHOLDER_CODE_SYSTEM
            assert re.fullmatch(r"SYN-[A-Z]+-\d{3}", node.get("code"))
        for oid in _RESTRICTED_CODE_SYSTEMS:
            assert f'"{oid}"' not in doc.xml


# Options and negative controls ----------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "http://example.test/CDA.xsd",
        "https://example.test/CDA.xsd",
        "file:///etc/CDA.xsd",
        "//example.test/CDA.xsd",
        "/abs/CDA.xsd",
        "../CDA.xsd",
        "urn:hl7-org:v3 http://example.test/CDA.xsd",
        'CDA.xsd"?><!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/x">]><x a="',
        "CDA.xsd&ext;",
        "%remote;",
        "CDA.dtd",
        "",
        b"CDA.xsd",
    ],
)
def test_remote_or_entity_bearing_schema_locations_are_rejected(value) -> None:
    with pytest.raises(CdaSynthesisError) as error:
        CdaSynthesisConfig(schema_location=value)
    assert error.value.code == "unsafe_schema_location"
    if isinstance(value, str) and value:
        assert value not in str(error.value)


@pytest.mark.parametrize(
    "value",
    [
        "https://example.test/cda.xsl",
        "file:///tmp/cda.xsl",
        '<!ENTITY x SYSTEM "http://example.test/">',
        "cda.xsl&x;",
        "cda.xsl?><!DOCTYPE x>",
    ],
)
def test_remote_or_entity_bearing_stylesheets_are_rejected(value) -> None:
    with pytest.raises(CdaSynthesisError) as error:
        CdaSynthesisConfig(stylesheet=value)
    assert error.value.code == "unsafe_stylesheet"
    assert value not in str(error.value)


def test_local_schema_location_and_stylesheet_stay_safe() -> None:
    config = CdaSynthesisConfig(schema_location="CDA.xsd", stylesheet="CDA.xsl")
    doc = CdaDocumentGenerator(config, seed=1).generate()
    data = doc.to_bytes()
    assert b'xsi:schemaLocation="urn:hl7-org:v3 CDA.xsd"' in data
    assert b'<?xml-stylesheet type="text/xsl" href="CDA.xsl"?>' in data
    _parse_without_dtd(data)
    assert is_cda_document(data)
    for truth in doc.truth:
        assert resolve_cda_locator(doc.xml, truth)[truth.start : truth.end] == (
            truth.value
        )


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"document_types": ("ccd", "referral")}, "unknown_document_type"),
        ({"document_types": ()}, "unknown_document_type"),
        ({"document_types": "ccd"}, "unknown_document_type"),
        ({"max_authors": 0}, "count_out_of_bounds"),
        ({"max_informants": 4}, "count_out_of_bounds"),
        ({"max_next_of_kin": -1}, "count_out_of_bounds"),
        ({"max_problems": 0}, "count_out_of_bounds"),
        ({"max_allergies": 6}, "count_out_of_bounds"),
        ({"max_procedures": True}, "count_out_of_bounds"),
        ({"guardian": 1}, "invalid_flag"),
        ({"birthplace": "yes"}, "invalid_flag"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, code) -> None:
    with pytest.raises(CdaSynthesisError) as error:
        CdaSynthesisConfig(**kwargs)
    assert error.value.code == code


def test_invalid_seed_count_and_type_are_rejected() -> None:
    with pytest.raises(CdaSynthesisError, match="invalid_seed"):
        CdaDocumentGenerator(seed="1")
    with pytest.raises(CdaSynthesisError, match="invalid_config"):
        CdaDocumentGenerator(config={"guardian": True})
    with pytest.raises(CdaSynthesisError, match="count_out_of_bounds"):
        generate_cda_documents(0)
    with pytest.raises(CdaSynthesisError, match="count_out_of_bounds"):
        generate_cda_documents(MAX_CDA_DOCUMENTS + 1)
    with pytest.raises(CdaSynthesisError, match="unknown_document_type"):
        CdaDocumentGenerator().generate("referral")


def test_optional_participants_can_be_disabled() -> None:
    config = CdaSynthesisConfig(
        guardian=False,
        birthplace=False,
        max_informants=0,
        max_next_of_kin=0,
    )
    for doc in generate_cda_documents(3, seed=2, config=config):
        roles = {truth.role for truth in doc.truth}
        assert not roles & {"guardian", "birthplace", "informant", "next_of_kin"}
        assert is_cda_document(doc.xml)
        for truth in doc.truth:
            node_value = resolve_cda_locator(doc.xml, truth)
            assert node_value[truth.start : truth.end] == truth.value


# Leakage harness ---------------------------------------------------------------

_KNOWN_DEFAULT_RESIDUALS = {
    ("/ClinicalDocument/author/time/@value", "DATE"),
    ("/ClinicalDocument/author/assignedAuthor/id/@extension", "ID_NUM"),
    ("/ClinicalDocument/author/assignedAuthor/telecom/@value", "PHONE"),
    (
        "/ClinicalDocument/author/assignedAuthor/addr/streetAddressLine/text()",
        "STREET_ADDRESS",
    ),
    (
        "/ClinicalDocument/custodian/assignedCustodian"
        "/representedCustodianOrganization/name/text()",
        "ORGANIZATION",
    ),
    (
        "/ClinicalDocument/custodian/assignedCustodian"
        "/representedCustodianOrganization/telecom/@value",
        "PHONE",
    ),
    ("/ClinicalDocument/legalAuthenticator/time/@value", "DATE"),
    ("/ClinicalDocument/legalAuthenticator/assignedEntity/id/@extension", "ID_NUM"),
    (
        "/ClinicalDocument/component/structuredBody/component/section/entry"
        "/procedure/performer/assignedEntity/assignedPerson/name/given/text()",
        "FIRST_NAME",
    ),
}
_COVERED_PREFIXES = (
    "/ClinicalDocument/recordTarget/",
    "/ClinicalDocument/participant/",
    "/ClinicalDocument/informant/",
    "/ClinicalDocument/effectiveTime/",
    "/ClinicalDocument/author/assignedAuthor/assignedPerson/",
    "/ClinicalDocument/legalAuthenticator/assignedEntity/assignedPerson/",
)


def test_harness_reports_residuals_by_element_path_and_label(documents) -> None:
    report = evaluate_cda_leakage(documents, date_shift_days=17)
    assert report.document_count == len(documents)
    assert report.well_formed_count == len(documents)
    assert report.truth_count == sum(len(doc.truth) for doc in documents)
    assert 0 < report.residual_count < report.truth_count
    residual = {(item.path, item.label) for item in report.residual_positions}
    assert _KNOWN_DEFAULT_RESIDUALS <= residual
    for item in report.positions:
        assert "[" not in item.path
        if item.path.startswith(_COVERED_PREFIXES):
            assert item.residual_count == 0, item.path
        assert 0.0 <= item.residual_rate <= 1.0
    payload = report.to_dict()
    assert set(payload) == {
        "schema_version",
        "document_count",
        "well_formed_count",
        "truth_count",
        "residual_count",
        "partial_count",
        "positions",
    }
    for row in payload["positions"]:
        assert set(row) == {
            "path",
            "label",
            "truth_count",
            "residual_count",
            "partial_count",
        }
    assert report.residual_by_path_and_label == {
        (item.path, item.label): item.residual_count
        for item in report.residual_positions
    }


def test_harness_flags_narrative_mentions_of_unmapped_values(documents) -> None:
    report = evaluate_cda_leakage(documents, date_shift_days=17)
    narrative = {
        (item.path, item.label)
        for item in report.residual_positions
        if "/section/text/" in item.path
    }
    labels = {label for _path, label in narrative}
    assert {"PERSON", "ORGANIZATION"} <= labels
    clean_narrative = [
        item
        for item in report.positions
        if "/section/text/" in item.path and item.label in {"DATE", "PHONE"}
    ]
    assert clean_narrative
    assert all(item.residual_count == 0 for item in clean_narrative)


def test_harness_counts_partial_leaks_of_multi_token_values(documents) -> None:
    def drop_leading_word(value: str) -> str:
        return re.sub(r"^[A-Z][a-z]+ (?=[A-Z][a-z]+$)", "[X] ", value)

    report = evaluate_cda_leakage(
        documents, text_redactor=drop_leading_word, date_shift_days=17
    )
    performer = (
        "/ClinicalDocument/component/structuredBody/component/section/text/table"
        "/tbody/tr/td/content/text()",
        "PERSON",
    )
    rows = {(item.path, item.label): item for item in report.positions}
    assert rows[performer].partial_count > 0
    assert rows[performer].residual_count == 0
    assert report.partial_count == sum(item.partial_count for item in rows.values())
    baseline = evaluate_cda_leakage(documents, date_shift_days=17)
    assert baseline.partial_count >= 0
    assert all(
        item.partial_count == 0
        for item in baseline.positions
        if item.label not in {"PERSON", "ORGANIZATION", "STREET_ADDRESS", "DATE"}
    )


def test_harness_report_never_contains_truth_values(documents) -> None:
    sparse_map = (DEFAULT_PHI_ELEMENT_MAP[0],)
    for report in (
        evaluate_cda_leakage(documents, date_shift_days=17),
        evaluate_cda_leakage(documents, element_map=sparse_map, date_shift_days=17),
    ):
        rendered = report.to_json() + repr(report) + str(report.to_dict())
        for doc in documents:
            for truth in doc.truth:
                assert truth.value not in rendered, truth.label
    sparse = evaluate_cda_leakage(documents, element_map=sparse_map)
    assert sparse.residual_count > (
        evaluate_cda_leakage(documents, date_shift_days=17).residual_count
    )


def test_extended_element_map_clears_structured_gaps(documents) -> None:
    extra = tuple(
        PhiElementRule(xpath, "null_flavor", attribute=attribute, label=label)
        for xpath, attribute, label in (
            (".//hl7:author/hl7:time", "value", "DATE"),
            (".//hl7:legalAuthenticator/hl7:time", "value", "DATE"),
            (".//hl7:assignedAuthor/hl7:id", "extension", "ID"),
            (".//hl7:assignedAuthor/hl7:addr", None, "ADDRESS"),
            (".//hl7:assignedAuthor/hl7:telecom", "value", "PHONE"),
            (".//hl7:legalAuthenticator/hl7:assignedEntity/hl7:id", "extension", "ID"),
            (".//hl7:representedCustodianOrganization/hl7:name", None, "ORG"),
            (".//hl7:representedCustodianOrganization/hl7:addr", None, "ADDRESS"),
            (
                ".//hl7:representedCustodianOrganization/hl7:telecom",
                "value",
                "PHONE",
            ),
            (".//hl7:performer//hl7:assignedPerson/hl7:name", None, "PERSON"),
        )
    )
    report = evaluate_cda_leakage(
        documents,
        element_map=DEFAULT_PHI_ELEMENT_MAP + extra,
        date_shift_days=17,
    )
    assert report.well_formed_count == len(documents)
    assert report.residual_count == 0, report.residual_by_path_and_label


def test_harness_is_deterministic_and_accepts_text_redactor(documents) -> None:
    first = evaluate_cda_leakage(documents, date_shift_days=-9)
    second = evaluate_cda_leakage(documents, date_shift_days=-9)
    assert first.to_json() == second.to_json()

    def scrub(value: str) -> str:
        return re.sub(r"[A-Z][a-z]+", "[NAME]", value)

    scrubbed = evaluate_cda_leakage(documents, text_redactor=scrub, date_shift_days=9)
    assert not any(
        "/section/text/" in item.path for item in scrubbed.residual_positions
    )


def test_harness_errors_are_controlled_and_value_free(documents) -> None:
    broken = dataclasses.replace(documents[0], xml="<ClinicalDocument")
    with pytest.raises(CdaLeakageError) as error:
        evaluate_cda_leakage([documents[1], broken])
    assert error.value.code == "redaction_failed"
    assert error.value.document_index == 1
    assert "ClinicalDocument" not in str(error.value)

    with pytest.raises(CdaLeakageError) as error:
        evaluate_cda_leakage([object()])
    assert error.value.code == "invalid_document"

    with pytest.raises(CdaLeakageError) as error:
        evaluate_cda_leakage([])
    assert error.value.code == "count_out_of_bounds"

    def explode(value: str) -> str:
        raise RuntimeError(value)

    with pytest.raises(CdaLeakageError) as error:
        evaluate_cda_leakage(documents[:1], text_redactor=explode)
    assert error.value.code == "redaction_failed"
    for truth in documents[0].truth:
        assert truth.value not in str(error.value)


def test_harness_counts_malformed_output_as_residual(monkeypatch, documents) -> None:
    import openmed.training.synthetic.cda_documents as module

    monkeypatch.setattr(module, "redact_cda", lambda *_a, **_k: "<broken")
    report = evaluate_cda_leakage(documents[:2])
    assert report.well_formed_count == 0
    assert report.residual_count == report.truth_count


def test_redacted_output_stays_parseable_and_redaction_is_unmodified(
    documents,
) -> None:
    for doc in documents[:3]:
        redacted = redact_cda(doc.xml, date_shift_days=30)
        assert is_cda_document(redacted)
        _parse_without_dtd(redacted.encode("utf-8"))
