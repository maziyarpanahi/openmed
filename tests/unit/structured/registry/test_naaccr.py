"""Synthetic NAACCR projection, custody, XML and protected-file controls."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import xml.etree.ElementTree as ET
from dataclasses import replace

import pytest

from openmed.clinical.journey_contracts import ClinicalFact, canonical_digest
from openmed.structured.registry import (
    RegistryCase,
    RegistryCaseState,
    RegistryDefinition,
    RegistryExportAuthorization,
    RegistryFieldEvidence,
    RegistryFieldResult,
    RegistryFieldRule,
    RegistryFieldState,
    RegistryWorkflowPolicy,
    build_registry_export,
    version_registry_definition,
)
from openmed.structured.registry.naaccr import (
    NAACCR_NAMESPACE,
    NAACCRDictionaryError,
    NAACCRFieldMapping,
    NAACCRItemDefinition,
    NAACCRProjectionError,
    NAACCRValueError,
    parse_naaccr_dictionary,
    write_naaccr_xml,
)

CREATED = "2026-01-02T03:04:05Z"
SECRET = b"synthetic-test-secret-32-bytes-only"
NS = {"n": NAACCR_NAMESPACE}


def _dictionary(items=(), *, version="1.8", key_type="text", key_length=64):
    root = ET.Element(
        "NaaccrDictionary",
        {
            "xmlns": NAACCR_NAMESPACE,
            "dictionaryUri": "urn:synthetic:naaccr:test",
            "specificationVersion": version,
        },
    )
    defs = ET.SubElement(root, "ItemDefs")
    for index, attrs in enumerate(
        (
            {
                "naaccrId": "opaquePatient",
                "parentXmlElement": "Patient",
                "length": str(key_length),
                "dataType": key_type,
            },
            *items,
        )
    ):
        ET.SubElement(defs, "ItemDef", {"naaccrNum": str(9500 + index), **attrs})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _item(name="code", *, parent="Tumor", kind="text", length=128, **attrs):
    return {
        "naaccrId": name,
        "parentXmlElement": parent,
        "dataType": kind,
        "length": str(length),
        **attrs,
    }


def _batch(rows=({"code": "SYNTHETIC"},), *, states=None, subjects=None):
    names = sorted(rows[0])
    definition = version_registry_definition(
        RegistryDefinition(
            registry_id="registry_syntheticnaaccr01",
            cohort_definition_version_id="cohortdefinition_syntheticnaaccr01",
            cohort_definition_digest="sha256:" + "1" * 64,
            fields=tuple(
                RegistryFieldRule(
                    field_id=name,
                    fact_type=name,
                    required=False,
                    allowed_statuses=("active", "corrected", "unknown", "conflict"),
                )
                for name in names
            ),
            workflow=RegistryWorkflowPolicy(
                policy_id="synthetic_naaccr",
                version="1.0.0",
                owner_scope_id="ownerscope_syntheticnaaccr01",
                # This explicit synthetic policy allows unresolved optional fields;
                # the projector must still omit them individually.
                review_field_states=(),
                adjudication_field_states=(),
                privacy_policy_digest="sha256:" + "2" * 64,
                export_policy_digest="sha256:" + "3" * 64,
            ),
            definition_version="1.0.0",
        )
    )
    facts = {}
    cases = []
    for index, row in enumerate(rows):
        subject = (subjects or [f"patient_{i:016x}" for i in range(len(rows))])[index]
        fields = []
        for offset, name in enumerate(names):
            fact = ClinicalFact(
                fact_id=f"fact_{index * 256 + offset:016x}",
                subject_id=subject,
                fact_type=name,
                value=row[name],
                status="active",
                evidence_ids=(f"evidence_{index * 256 + offset:016x}",),
                derivation_hash=canonical_digest({"synthetic": [index, offset]}),
            )
            facts[fact.fact_id] = fact
            state = (states or {}).get(name, RegistryFieldState.PRESENT)
            fields.append(
                RegistryFieldResult(
                    name,
                    state,
                    RegistryFieldEvidence(
                        (fact.fact_id,),
                        fact.evidence_ids,
                        (canonical_digest(fact.value),),
                        (fact.derivation_hash,),
                        ("fact_prior00000000001",)
                        if state is RegistryFieldState.CORRECTED
                        else (),
                    ),
                    "synthetic",
                )
            )
        cases.append(
            RegistryCase(
                case_id=f"registrycase_{index:016x}",
                definition_version_id=definition.version_id,
                definition_digest=definition.definition_digest,
                subject_id=subject,
                cohort_execution_id="cohortexecution_syntheticnaaccr01",
                cohort_execution_digest="sha256:" + "4" * 64,
                source_snapshot_id="snapshot_syntheticnaaccr01",
                source_snapshot_digest="sha256:" + "5" * 64,
                created_at=CREATED,
                fields=tuple(fields),
                origin_state=RegistryCaseState.EXPORT_READY,
                state=RegistryCaseState.EXPORT_READY,
                workflow_policy_digest=definition.definition.workflow.digest,
            )
        )
    workflow = definition.definition.workflow
    authorization = RegistryExportAuthorization(
        "authorization_syntheticnaaccr01",
        definition.version_id,
        workflow.privacy_policy_digest,
        workflow.export_policy_digest,
        True,
    )
    cases = tuple(cases)
    envelope = build_registry_export(
        definition, cases, authorization=authorization, created_at=CREATED
    ).value
    assert envelope is not None
    return {
        "cases": cases,
        "definition_version": definition,
        "authorization": authorization,
        "export_envelope": envelope,
        "resolver": lambda _case, _field, fact_id: facts.get(fact_id),
        "patient_key_item": "opaquePatient",
        "patient_key_secret": SECRET,
    }, facts


def _write(tmp_path, batch=None, *, items=None, mappings=None, **changes):
    if batch is None:
        batch, _ = _batch()
    options = {
        **batch,
        "dictionary": parse_naaccr_dictionary(_dictionary(items or (_item(),))),
        "field_map": mappings
        if mappings is not None
        else (NAACCRFieldMapping("code", "code"),),
        "output_path": tmp_path / "protected.xml",
        **changes,
    }
    return write_naaccr_xml(**options)


def _items(node):
    children = node.findall("n:Item", NS)
    assert len({n.attrib["naaccrId"] for n in children}) == len(children)
    return {n.attrib["naaccrId"]: n.text for n in children}


def test_multiple_patients_tumors_and_parents_have_exact_nonduplicated_values(tmp_path):
    batch, _ = _batch(
        (
            {"root": "ROOT", "patient": "FIRST", "code": "TUMOR1"},
            {"root": "ROOT", "patient": "FIRST", "code": "TUMOR2"},
            {"root": "ROOT", "patient": "SECOND", "code": "TUMOR3"},
        ),
        subjects=(
            "patient_0000000000000001",
            "patient_0000000000000001",
            "patient_0000000000000002",
        ),
    )
    report = _write(
        tmp_path,
        batch,
        items=(
            _item("root", parent="NaaccrData"),
            _item("patient", parent="Patient"),
            _item(),
        ),
        mappings=tuple(NAACCRFieldMapping(n, n) for n in ("code", "patient", "root")),
    )
    payload = (tmp_path / "protected.xml").read_bytes()
    root = ET.fromstring(payload)
    assert root.tag == f"{{{NAACCR_NAMESPACE}}}NaaccrData"
    assert root.attrib == {
        "baseDictionaryUri": "urn:synthetic:naaccr:test",
        "recordType": "I",
        "specificationVersion": "1.8",
    }
    assert _items(root) == {"root": "ROOT"}
    patients = root.findall("n:Patient", NS)
    observed = {}
    for patient in patients:
        assert patient.attrib == {}
        pitems = _items(patient)
        key = pitems.pop("opaquePatient")
        assert len(key) == 64
        tumors = patient.findall("n:Tumor", NS)
        assert all(t.attrib == {} for t in tumors)
        observed[pitems["patient"]] = sorted(_items(t)["code"] for t in tumors)
    assert observed == {"FIRST": ["TUMOR1", "TUMOR2"], "SECOND": ["TUMOR3"]}
    assert (
        report.case_count,
        report.patient_count,
        report.tumor_count,
        report.item_count,
    ) == (3, 2, 3, 8)
    assert report.losses == ()
    assert report.xml_digest == "sha256:" + hashlib.sha256(payload).hexdigest()
    assert stat.S_IMODE(os.stat(tmp_path / "protected.xml").st_mode) == 0o600
    safe = json.dumps(report.to_dict()) + repr(report)
    assert all(
        v not in safe
        for v in (
            "TUMOR1",
            "FIRST",
            "ROOT",
            "patient_000",
            str(tmp_path),
            SECRET.decode(),
        )
    )
    assert report.to_dict()["submitted"] is False
    assert report.to_dict()["edits_validated"] is False


def test_output_bytes_and_digest_are_stable_across_input_order(tmp_path):
    batch, _ = _batch(({"code": "FIRST"}, {"code": "SECOND"}))
    first = _write(tmp_path, batch)
    second = _write(
        tmp_path,
        {**batch, "cases": tuple(reversed(batch["cases"]))},
        output_path=tmp_path / "second.xml",
    )
    assert first.xml_digest == second.xml_digest
    assert (tmp_path / "protected.xml").read_bytes() == (
        tmp_path / "second.xml"
    ).read_bytes()


@pytest.mark.parametrize("value", ["A&B <tag> \"quoted\" 'value'", "élève\t中\n𝄞\rEND"])
def test_xml_escaping_preserves_exact_reviewed_text(tmp_path, value):
    batch, _ = _batch(({"code": value},))
    _write(tmp_path, batch)
    root = ET.parse(tmp_path / "protected.xml").getroot()
    assert root.find("n:Patient/n:Tumor/n:Item", NS).text == value


@pytest.mark.parametrize(
    "state",
    [
        s
        for s in RegistryFieldState
        if s not in (RegistryFieldState.PRESENT, RegistryFieldState.CORRECTED)
    ],
)
def test_unresolved_fields_never_call_resolver_or_write_values(tmp_path, state):
    batch, _ = _batch(states={"code": state})
    batch["resolver"] = lambda *_: pytest.fail("unresolved fact was resolved")
    report = _write(tmp_path, batch)
    assert report.to_dict()["loss_counts"] == {state.value: 1}
    assert b"SYNTHETIC" not in (tmp_path / "protected.xml").read_bytes()
    assert report.tumor_count == 0


def test_unmapped_field_is_a_loss_and_is_not_resolved(tmp_path):
    batch, _ = _batch()
    batch["resolver"] = lambda *_: pytest.fail("unmapped fact was resolved")
    report = _write(tmp_path, batch, mappings=())
    assert report.to_dict()["loss_counts"] == {"unmapped": 1}
    assert b"SYNTHETIC" not in (tmp_path / "protected.xml").read_bytes()


@pytest.mark.parametrize(
    "attribute,value,loss",
    [
        ("value", "SUBSTITUTED", "fact_custody_mismatch"),
        ("subject_id", "patient_ffffffffffffffff", "fact_custody_mismatch"),
        ("fact_id", "fact_other00000000001", "fact_custody_mismatch"),
        ("fact_type", "other", "fact_custody_mismatch"),
        ("derivation_hash", "sha256:" + "f" * 64, "fact_custody_mismatch"),
        ("evidence_ids", ("evidence_ffffffffffffffff",), "fact_custody_mismatch"),
        ("status", "unknown", "fact_state_refused"),
        ("status", "conflict", "fact_state_refused"),
        ("status", "unsupported", "fact_state_refused"),
    ],
)
def test_substituted_facts_and_refused_statuses_cannot_reach_xml(
    tmp_path, attribute, value, loss
):
    batch, facts = _batch()
    original = next(iter(facts.values()))
    batch["resolver"] = lambda *_: replace(original, **{attribute: value})
    report = _write(tmp_path, batch)
    assert report.to_dict()["loss_counts"] == {loss: 1}
    assert b"SYNTHETIC" not in (tmp_path / "protected.xml").read_bytes()
    assert b"SUBSTITUTED" not in (tmp_path / "protected.xml").read_bytes()


@pytest.mark.parametrize(
    "resolved,loss",
    [(None, "fact_missing"), ({"value": "SECRET"}, "fact_custody_mismatch")],
)
def test_missing_and_untyped_facts_are_controlled_losses(tmp_path, resolved, loss):
    batch, _ = _batch()
    batch["resolver"] = lambda *_: resolved
    assert _write(tmp_path, batch).to_dict()["loss_counts"] == {loss: 1}


def test_dropped_evidence_custody_is_not_accepted(tmp_path):
    batch, facts = _batch()
    fact = next(iter(facts.values()))
    case = batch["cases"][0]
    field = case.fields[0]
    field = replace(
        field,
        evidence=replace(
            field.evidence,
            evidence_ids=(*fact.evidence_ids, "evidence_eeeeeeeeeeeeeeee"),
        ),
    )
    case = replace(case, fields=(field,))
    batch["cases"] = (case,)
    batch["export_envelope"] = build_registry_export(
        batch["definition_version"],
        (case,),
        authorization=batch["authorization"],
        created_at=CREATED,
    ).value
    report = _write(tmp_path, batch)
    assert report.to_dict()["loss_counts"] == {"fact_custody_mismatch": 1}


@pytest.mark.parametrize("parent", ["Patient", "NaaccrData"])
def test_contradictory_shared_items_are_omitted_for_all_sources(tmp_path, parent):
    batch, _ = _batch(
        ({"code": "FIRST"}, {"code": "SECOND"}),
        subjects=("patient_aaaaaaaaaaaaaaaa", "patient_aaaaaaaaaaaaaaaa"),
    )
    report = _write(tmp_path, batch, items=(_item(parent=parent),))
    assert report.to_dict()["loss_counts"] == {"value_conflict": 2}
    payload = (tmp_path / "protected.xml").read_bytes()
    assert b"FIRST" not in payload and b"SECOND" not in payload
    assert report.item_count == 1


def test_static_path_selects_a_value_without_exporting_the_surrounding_object(tmp_path):
    batch, _ = _batch(
        ({"code": {"coding": [{"code": "S01", "display": "PRIVATE SURFACE"}]}},)
    )
    _write(
        tmp_path,
        batch,
        mappings=(NAACCRFieldMapping("code", "code", ("coding", 0, "code")),),
    )
    payload = (tmp_path / "protected.xml").read_bytes()
    assert b">S01<" in payload and b"PRIVATE SURFACE" not in payload


@pytest.mark.parametrize(
    "kind,length,value",
    [
        ("digits", 3, "007"),
        ("alpha", 2, "AB"),
        ("alphanumeric", 3, "A07"),
        ("numeric", 10, "12.5"),
        ("numeric", 10, 12.5),
        ("numeric", 10, 12),
        ("date", 8, "2024"),
        ("date", 8, "202402"),
        ("date", 8, "20240229"),
        ("dateTime", 25, "0001"),
        ("dateTime", 25, "2024-02"),
        ("dateTime", 25, "2024-02-29"),
        ("dateTime", 25, "2024-02-29T23:59:60Z"),
        ("dateTime", 25, "2024-02-29T00:00:00+14:00"),
        ("text", 3, "中文"),
    ],
)
def test_declared_data_types_accept_only_exact_valid_scalars(
    tmp_path, kind, length, value
):
    batch, _ = _batch(({"code": value},))
    report = _write(tmp_path, batch, items=(_item(kind=kind, length=length),))
    assert report.losses == ()
    root = ET.parse(tmp_path / "protected.xml").getroot()
    assert root.find("n:Patient/n:Tumor/n:Item", NS).text == str(value)


@pytest.mark.parametrize(
    "kind,length,value,code",
    [
        ("digits", 3, "07", "data_type"),
        ("digits", 3, "0007", "length"),
        ("digits", 3, "０07", "data_type"),
        ("alpha", 2, "Ab", "data_type"),
        ("alphanumeric", 3, "A-7", "data_type"),
        ("numeric", 20, "-1", "data_type"),
        ("numeric", 20, "1e2", "data_type"),
        ("numeric", 20, "1,000", "data_type"),
        ("numeric", 20, ".5", "data_type"),
        ("numeric", 20, True, "data_type"),
        ("text", 128, {"raw": "PRIVATE SURFACE"}, "data_type"),
        ("text", 128, ["PRIVATE SURFACE"], "data_type"),
        ("text", 128, None, "data_type"),
        ("text", 128, "", "length"),
        ("text", 128, " \t\n", "length"),
        ("text", 128, "PRIVATE\x01SURFACE", "xml_character"),
        ("text", 4, "PRIVATE SURFACE", "length"),
        ("date", 8, "20230229", "data_type"),
        ("date", 8, "202413", "data_type"),
        ("date", 8, "1799", "data_type"),
        ("date", 8, "2100", "data_type"),
        ("date", 8, "20240000", "data_type"),
        ("dateTime", 25, "0000", "data_type"),
        ("dateTime", 25, "2023-02-29", "data_type"),
        ("dateTime", 25, "2024-02-29T00:00:00", "data_type"),
        ("dateTime", 25, "2024-02-29T24:00:00Z", "data_type"),
        ("dateTime", 25, "2024-02-29T00:00:00+14:01", "data_type"),
        ("dateTime", 25, "2024-02-29T00:00:00.1Z", "data_type"),
    ],
)
def test_type_length_and_xml_failures_are_typed_and_create_no_file(
    tmp_path, kind, length, value, code
):
    batch, _ = _batch(({"code": value},))
    with pytest.raises(NAACCRValueError) as exc:
        _write(tmp_path, batch, items=(_item(kind=kind, length=length),))
    assert exc.value.to_dict() == {"code": code, "case_index": 0, "field_index": 0}
    assert str(exc.value) == "NAACCR projection refused."
    assert exc.value.__suppress_context__
    assert not (tmp_path / "protected.xml").exists()


def test_unready_or_stale_case_refuses_entire_batch_before_resolving(tmp_path):
    batch, _ = _batch()
    case = batch["cases"][0]
    batch["cases"] = (
        replace(
            case,
            state=RegistryCaseState.REVIEW_REQUIRED,
            origin_state=RegistryCaseState.REVIEW_REQUIRED,
        ),
    )
    batch["resolver"] = lambda *_: pytest.fail("unreviewed batch resolved values")
    report = _write(tmp_path, batch)
    assert report.to_dict()["loss_counts"] == {"case_not_export_ready": 1}
    assert not (tmp_path / "protected.xml").exists()
    batch["cases"] = (replace(case, created_at="2026-01-02T03:04:06Z"),)
    assert _write(tmp_path, batch).to_dict()["loss_counts"] == {
        "case_custody_mismatch": 1
    }
    assert not (tmp_path / "protected.xml").exists()


@pytest.mark.parametrize(
    "change,code",
    [
        ({"export_approved": False}, "authorization_refused"),
        ({"privacy_policy_digest": "sha256:" + "f" * 64}, "authorization_refused"),
        ({"export_policy_digest": "sha256:" + "f" * 64}, "authorization_refused"),
        ({"authorization_id": "authorization_ffffffffffffffff"}, "custody_invalid"),
    ],
)
def test_authorization_must_match_approved_policy_and_envelope(tmp_path, change, code):
    batch, _ = _batch()
    batch["authorization"] = replace(batch["authorization"], **change)
    with pytest.raises(NAACCRProjectionError) as exc:
        _write(tmp_path, batch)
    assert exc.value.code == code
    assert not (tmp_path / "protected.xml").exists()


def test_resolver_prints_and_exception_text_do_not_escape(tmp_path, capsys):
    batch, _ = _batch()

    def broken(*_):
        print("PRIVATE SURFACE")
        raise RuntimeError("PRIVATE SURFACE")

    batch["resolver"] = broken
    with pytest.raises(NAACCRProjectionError) as exc:
        _write(tmp_path, batch)
    assert exc.value.code == "resolver_failed"
    assert "PRIVATE" not in str(exc.value)
    assert capsys.readouterr() == ("", "")
    assert not (tmp_path / "protected.xml").exists()


def test_existing_file_and_symlink_are_never_overwritten(tmp_path):
    existing = tmp_path / "existing.xml"
    existing.write_bytes(b"EXISTING PRIVATE BYTES")
    link = tmp_path / "link.xml"
    link.symlink_to(existing)
    for path in (existing, link):
        with pytest.raises(NAACCRProjectionError) as exc:
            _write(tmp_path, output_path=path)
        assert exc.value.code == "output_refused"
        assert str(path) not in str(exc.value)
        assert existing.read_bytes() == b"EXISTING PRIVATE BYTES"
    assert link.is_symlink()


def test_partial_file_is_removed_on_fsync_failure(tmp_path, monkeypatch):
    def failed(*_):
        raise OSError("PRIVATE PATH")

    monkeypatch.setattr(os, "fsync", failed)
    with pytest.raises(NAACCRProjectionError) as exc:
        _write(tmp_path)
    assert exc.value.code == "output_refused"
    assert not (tmp_path / "protected.xml").exists()


@pytest.mark.parametrize(
    "kind,length",
    [("digits", 8), ("alpha", 8), ("alphanumeric", 8), ("numeric", 16), ("text", 64)],
)
def test_patient_key_matches_declared_item_without_using_source_identifier(
    tmp_path, kind, length
):
    dictionary = parse_naaccr_dictionary(
        _dictionary((_item(),), key_type=kind, key_length=length)
    )
    _write(tmp_path, dictionary=dictionary)
    payload = (tmp_path / "protected.xml").read_bytes()
    assert b"patient_" not in payload
    key = ET.fromstring(payload).find("n:Patient/n:Item", NS).text
    assert len(key) == length


def test_dictionary_is_immutable_and_digest_binds_exact_supplied_bytes():
    raw = _dictionary((_item(),))
    dictionary = parse_naaccr_dictionary(raw)
    assert dictionary.source_digest == "sha256:" + hashlib.sha256(raw).hexdigest()
    with pytest.raises(TypeError):
        dictionary.items["code"] = dictionary.items["code"]
    assert (
        parse_naaccr_dictionary(raw.decode()).source_digest == dictionary.source_digest
    )


@pytest.mark.parametrize(
    "attrs",
    [
        {"naaccrId": "unknown-id"},
        {"naaccrNum": "-1"},
        {"naaccrNum": "0"},
        {"length": "0"},
        {"length": "1.0"},
        {"length": "1048577"},
        {"parentXmlElement": "Unknown"},
        {"dataType": "boolean"},
        {"recordTypes": "I,I"},
        {"recordTypes": "I, C"},
        {"dataType": "date", "length": "7"},
        {"dataType": "dateTime", "length": "24"},
        {"allowUnlimitedText": "true"},
        {"allowUnlimitedText": "false"},
    ],
)
def test_invalid_dictionary_constraints_are_fixed_source_free_errors(attrs):
    raw = _dictionary((_item(**attrs),))
    with pytest.raises(NAACCRDictionaryError) as exc:
        parse_naaccr_dictionary(raw)
    assert str(exc.value) == "NAACCR projection refused."
    assert exc.value.__suppress_context__


@pytest.mark.parametrize(
    "raw",
    [
        b"<PRIVATE-SURFACE",
        b"<NaaccrDictionary/>",
        b"\xff",
        b"\0",
        b"x" * 4_194_305,
        b'<!DOCTYPE x [<!ENTITY secret "PRIVATE SURFACE">]><x>&secret;</x>',
        b'<!DOCTYPE x SYSTEM "file:///PRIVATE-PATH"><x/>',
        b'<?xml version="1.0" encoding="iso-8859-1"?><x/>',
    ],
)
def test_malformed_oversized_or_entity_dictionary_is_refused_without_source_echo(raw):
    with pytest.raises(NAACCRDictionaryError) as exc:
        parse_naaccr_dictionary(raw)
    assert "PRIVATE" not in str(exc.value)
    assert "PRIVATE" not in json.dumps(exc.value.to_dict())


@pytest.mark.parametrize(
    "mappings",
    [
        (NAACCRFieldMapping("missing", "code"),),
        (NAACCRFieldMapping("code", "missing"),),
        (NAACCRFieldMapping("code", "opaquePatient"),),
        (NAACCRFieldMapping("code", "code"), NAACCRFieldMapping("code", "code")),
    ],
)
def test_mapping_unknown_items_and_duplicate_bindings_fail_before_file(
    tmp_path, mappings
):
    with pytest.raises(NAACCRProjectionError) as exc:
        _write(tmp_path, mappings=mappings)
    assert exc.value.code == "mapping_invalid"
    assert not (tmp_path / "protected.xml").exists()


@pytest.mark.parametrize(
    "path", [(True,), (-1,), (4097,), ("private path",), tuple("x" for _ in range(9))]
)
def test_value_paths_are_static_bounded_selectors(path):
    with pytest.raises(NAACCRProjectionError):
        NAACCRFieldMapping("code", "code", path)


def test_public_error_cannot_echo_untrusted_codes_or_indices():
    error = NAACCRProjectionError(["PRIVATE SURFACE"], "PRIVATE SURFACE", True)
    assert error.to_dict() == {
        "code": "input_invalid",
        "case_index": None,
        "field_index": None,
    }
    with pytest.raises(NAACCRDictionaryError):
        NAACCRItemDefinition("code", 9500, 8, ["PRIVATE SURFACE"])


def test_truncated_patient_key_collision_refuses_the_batch(tmp_path):
    # Eleven subjects cannot have unique keys in a single decimal digit.
    batch, _ = _batch(tuple({"code": "SYNTHETIC"} for _ in range(11)))
    dictionary = parse_naaccr_dictionary(
        _dictionary((_item(),), key_type="digits", key_length=1)
    )
    with pytest.raises(NAACCRProjectionError) as exc:
        _write(tmp_path, batch, dictionary=dictionary)
    assert exc.value.code == "patient_key_collision"
    assert not (tmp_path / "protected.xml").exists()


@pytest.mark.parametrize("different", [False, True])
def test_multiple_bound_facts_require_one_exact_scalar_and_complete_evidence(
    tmp_path, different
):
    batch, facts = _batch()
    first = next(iter(facts.values()))
    second = replace(
        first,
        fact_id="fact_ffffffffffffffff",
        value="OTHER" if different else first.value,
        evidence_ids=("evidence_ffffffffffffffff",),
        derivation_hash=canonical_digest({"synthetic": "second"}),
    )
    facts[second.fact_id] = second
    case = batch["cases"][0]
    field = replace(
        case.fields[0],
        evidence=RegistryFieldEvidence(
            (first.fact_id, second.fact_id),
            (*first.evidence_ids, *second.evidence_ids),
            (canonical_digest(first.value), canonical_digest(second.value)),
            (first.derivation_hash, second.derivation_hash),
        ),
    )
    case = replace(case, fields=(field,))
    batch["cases"] = (case,)
    batch["export_envelope"] = build_registry_export(
        batch["definition_version"],
        (case,),
        authorization=batch["authorization"],
        created_at=CREATED,
    ).value
    report = _write(tmp_path, batch)
    assert report.to_dict()["loss_counts"] == (
        {"value_conflict": 1} if different else {}
    )
    assert report.tumor_count == (0 if different else 1)
    assert b"OTHER" not in (tmp_path / "protected.xml").read_bytes()


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_id",
        "duplicate_number",
        "missing_parent",
        "deep",
        "missing_uri",
        "future_version",
        "dateTime_old",
    ],
)
def test_dictionary_identity_and_structure_are_validated(change):
    root = ET.fromstring(_dictionary((_item(),)))
    defs = root.find("n:ItemDefs", NS)
    item = defs[1]
    if change == "duplicate_id":
        item.set("naaccrId", "opaquePatient")
    elif change == "duplicate_number":
        item.set("naaccrNum", "9500")
    elif change == "missing_parent":
        del item.attrib["parentXmlElement"]
    elif change == "missing_uri":
        del root.attrib["dictionaryUri"]
    elif change == "future_version":
        root.set("specificationVersion", "9.0")
    elif change == "dateTime_old":
        root.set("specificationVersion", "1.7")
        item.set("dataType", "dateTime")
        item.set("length", "25")
    else:
        node = root
        for _ in range(34):
            node = ET.SubElement(node, "Extra")
    with pytest.raises(NAACCRDictionaryError):
        parse_naaccr_dictionary(ET.tostring(root))


def test_legacy_unlimited_text_is_explicitly_version_gated(tmp_path):
    dictionary = parse_naaccr_dictionary(
        _dictionary((_item(length=1, allowUnlimitedText="true"),), version="1.5")
    )
    report = _write(tmp_path, dictionary=dictionary)
    assert report.file_written
    assert b"SYNTHETIC" in (tmp_path / "protected.xml").read_bytes()


def test_dictionary_record_type_filters_are_enforced_before_resolution(tmp_path):
    with pytest.raises(NAACCRProjectionError) as exc:
        _write(tmp_path, items=(_item(recordTypes="C"),))
    assert exc.value.code == "mapping_invalid"
    assert not (tmp_path / "protected.xml").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"patient_key_secret": b"short"},
        {"record_type": "UNKNOWN"},
        {"patient_key_item": "code"},
        {"cases": ()},
    ],
)
def test_invalid_configuration_cannot_create_a_file(tmp_path, changes):
    with pytest.raises(NAACCRProjectionError):
        _write(tmp_path, **changes)
    assert not (tmp_path / "protected.xml").exists()


def test_case_count_bound_is_checked_before_resolution(tmp_path):
    batch, _ = _batch(tuple({"code": "SYNTHETIC"} for _ in range(129)))
    batch["resolver"] = lambda *_: pytest.fail("oversized batch resolved values")
    with pytest.raises(NAACCRProjectionError):
        _write(tmp_path, batch)
    assert not (tmp_path / "protected.xml").exists()


def test_caller_supplied_mapping_cannot_select_an_absent_path(tmp_path):
    with pytest.raises(NAACCRValueError) as exc:
        _write(tmp_path, mappings=(NAACCRFieldMapping("code", "code", ("missing",)),))
    assert exc.value.code == "data_type"
    assert not (tmp_path / "protected.xml").exists()


def test_aggregate_values_are_bounded_before_large_xml_allocation(tmp_path):
    batch, _ = _batch(tuple({"code": "A" * 800_000} for _ in range(6)))
    with pytest.raises(NAACCRProjectionError) as exc:
        _write(tmp_path, batch, items=(_item(length=1_048_576),))
    assert exc.value.code == "input_limit"
    assert not (tmp_path / "protected.xml").exists()
