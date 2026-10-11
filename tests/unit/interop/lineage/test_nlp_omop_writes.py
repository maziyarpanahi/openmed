"""Synthetic loader-to-batch lineage, replacement and vocabulary controls."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from itertools import repeat

import pytest

from openmed.interop.lineage.nlp_omop_writes import (
    NlpOmopLineageRecord,
    NlpOmopStagingError,
    NlpOmopWriteLineage,
    stage_nlp_omop_tables,
)
from openmed.interop.omop import (
    OmopRowKey,
    VocabularyConcept,
    VocabularyMappingProvenance,
    VocabularySnapshot,
    load_grounded_notes,
)

PIPELINE = "sha256:" + "a" * 64
OLD_PIPELINE = "sha256:" + "b" * 64
RECEIPT = "sha256:" + "c" * 64
CANARY = "PRIVATE-LEXICAL-SNIPPET-CANARY"


def notes(*, stale=False, person="private-person", origin="1" * 64):
    text = f"Synthetic {CANARY} and beta findings."
    entities = []
    for surface, concept in ((CANARY, 101), ("beta", 102)):
        if concept == 102 and not stale:
            continue
        start = text.index(surface)
        entities.append(
            {
                "text": surface,
                "start": start,
                "end": start + len(surface),
                "domain_id": "Condition",
                "concept_id": concept,
                "vocabulary_id": "SYNTHETIC",
                "code": f"C-{concept}",
                "concept_name": "Synthetic concept",
            }
        )
    return [
        {
            "note_id": "private-note",
            "person_id": person,
            "visit_id": "private-visit",
            "source_note_hash": origin,
            "note_text": text,
            "note_date": "2026-01-01",
            "entities": entities,
        }
    ]


def tables(*, stale=False, mode="append", person="private-person", origin="1" * 64):
    result = load_grounded_notes(
        notes(stale=stale, person=person, origin=origin),
        vocabulary_version="release-current",
        mode=mode,
    )
    copied = copy.deepcopy(result.tables)
    for row in copied["note_nlp"]:
        row["snippet"] = CANARY
    return replace(result, tables=copied)


def snapshot(*, retired=False, version="release-current"):
    return VocabularySnapshot(
        {"SYNTHETIC": version},
        (
            VocabularyConcept(
                101,
                "SYNTHETIC",
                standard_concept="S",
                invalid_reason="D" if retired else None,
            ),
            VocabularyConcept(102, "SYNTHETIC", standard_concept="S"),
        ),
    )


def mappings(*, stale=False, version="release-current"):
    return tuple(
        VocabularyMappingProvenance(n, "SYNTHETIC", version)
        for n in ((101, 102) if stale else (101,))
    )


def staged(**kwargs):
    input_tables = kwargs.pop("tables", tables())
    return stage_nlp_omop_tables(
        input_tables,
        pipeline_digest=kwargs.pop("pipeline_digest", PIPELINE),
        vocabulary_snapshot=kwargs.pop("vocabulary_snapshot", snapshot()),
        vocabulary_mappings=kwargs.pop("vocabulary_mappings", mappings()),
        **kwargs,
    )


def preview(stage):
    manifest = stage.build_rollback_manifest(stage.rollback_requirements)
    return stage.preview(manifest), manifest


def old_pipelines(old):
    return {
        OmopRowKey("note_nlp", {"note_nlp_id": r["note_nlp_id"]}).digest: OLD_PIPELINE
        for r in old.table("note_nlp")
    }


def test_loader_stages_all_rows_without_writes_and_binds_evidence():
    stage = staged()
    packet, manifest = preview(stage)
    assert packet.is_approvable
    assert len(stage.lineage.records) == 3
    assert len(stage.batch.mutations) == sum(
        len(rows) for rows in tables().tables.values()
    )
    assert packet.lineage_report.expected_count == 3
    assert stage.lineage.lineage_digest in packet.batch_preview.evidence_digests
    assert snapshot().snapshot_digest in packet.batch_preview.evidence_digests
    assert len(packet.batch_preview.evidence_digests) == 4
    approval = stage.bind_approval(
        packet,
        manifest,
        approved_preview_digest=packet.preview_digest,
        approval_receipt_digest=RECEIPT,
    )
    assert approval.batch_digest == stage.batch.batch_digest
    assert approval.approval_receipt_digest == RECEIPT


def test_canaries_stay_inside_private_rows_and_rollback_material(caplog):
    stage = staged()
    packet, manifest = preview(stage)
    safe = (
        json.dumps(packet.to_dict())
        + json.dumps(stage.lineage.to_dict())
        + json.dumps(manifest.to_dict())
        + repr(stage)
        + repr(stage.lineage)
        + caplog.text
    )
    for secret in (CANARY, "private-person", "private-note", "private-visit"):
        assert secret not in safe
    assert any(CANARY in json.dumps(m.values) for m in stage.batch.mutations)
    source = stage.lineage.records[0]
    assert source.source_note_digest == "sha256:" + "1" * 64
    assert source.note_content_digest != source.source_note_digest
    assert source.pipeline_digest == PIPELINE


@pytest.mark.parametrize(
    "change", ["missing", "extra", "duplicate", "wrong_digest", "wrong_table"]
)
def test_lineage_coverage_changes_block_approval(change):
    stage = staged()
    records = list(stage.lineage.records)
    if change == "missing":
        records.pop()
    elif change == "extra":
        records.append(replace(records[0], mutation_ordinal=0))
    elif change == "duplicate":
        records.append(records[0])
    elif change == "wrong_digest":
        records[0] = replace(records[0], row_digest="sha256:" + "d" * 64)
    else:
        records[0] = replace(records[0], table="observation")
    changed = replace(stage, lineage=NlpOmopWriteLineage(records))
    packet, manifest = preview(changed)
    assert not packet.is_approvable
    with pytest.raises(NlpOmopStagingError, match="preflight_failed"):
        changed.bind_approval(
            packet,
            manifest,
            approved_preview_digest=packet.preview_digest,
            approval_receipt_digest=RECEIPT,
        )


def test_replace_by_note_tombstones_stale_rows_and_preserves_old_pipeline():
    old, incoming = tables(stale=True), tables(mode="replace_by_note")
    stage = staged(
        tables=incoming,
        existing_tables=old,
        existing_pipeline_digests=old_pipelines(old),
    )
    packet, manifest = preview(stage)
    assert packet.is_approvable
    assert dict(packet.batch_preview.operation_counts) == {"insert": 4, "tombstone": 7}
    assert sum(r.pipeline_digest == OLD_PIPELINE for r in stage.lineage.records) == 6
    assert sum(r.pipeline_digest == PIPELINE for r in stage.lineage.records) == 3
    assert any(e.strategy.value == "reinsert_tombstoned_row" for e in manifest.entries)
    for i, mutation in enumerate(stage.batch.mutations):
        material = stage.rollback_material(i)
        if mutation.operation.value == "tombstone":
            assert material["before_image"] is not None
            assert material["key"] == mutation.key.values
        else:
            assert material["before_image"] is None
    assert CANARY not in json.dumps(packet.to_dict()) + json.dumps(manifest.to_dict())


def test_replace_requires_explicit_snapshot_and_old_pipeline_evidence():
    with pytest.raises(NlpOmopStagingError, match="replacement_snapshot_required"):
        staged(tables=tables(mode="replace_by_note"))
    with pytest.raises(NlpOmopStagingError, match="missing_old_pipeline_evidence"):
        staged(
            tables=tables(mode="replace_by_note"), existing_tables=tables(stale=True)
        )
    with pytest.raises(NlpOmopStagingError, match="extra_old_pipeline_evidence"):
        staged(existing_pipeline_digests={"sha256:" + "f" * 64: OLD_PIPELINE})


@pytest.mark.parametrize(
    "kind", ["missing", "extra", "stale", "retired", "snapshot_changed"]
)
def test_vocabulary_gate_and_coverage_fail_closed(kind):
    kwargs = {}
    if kind == "missing":
        kwargs["vocabulary_mappings"] = ()
    elif kind == "extra":
        kwargs["vocabulary_mappings"] = mappings(stale=True)
    elif kind == "stale":
        kwargs["vocabulary_mappings"] = mappings(version="release-stale")
    elif kind == "retired":
        kwargs["vocabulary_snapshot"] = snapshot(retired=True)
    else:
        kwargs["vocabulary_snapshot"] = snapshot(version="release-changed")
    stage = staged(**kwargs)
    packet, manifest = preview(stage)
    assert not packet.is_approvable
    with pytest.raises(NlpOmopStagingError, match="preflight_failed"):
        stage.bind_approval(
            packet,
            manifest,
            approved_preview_digest=packet.preview_digest,
            approval_receipt_digest=RECEIPT,
        )


def test_approval_packet_change_requires_new_human_review():
    first = staged()
    packet, manifest = preview(first)
    changed = staged(pipeline_digest=OLD_PIPELINE)
    changed_packet, changed_manifest = preview(changed)
    assert first.batch.batch_digest != changed.batch.batch_digest
    with pytest.raises(NlpOmopStagingError, match="approval_mismatch"):
        changed.bind_approval(
            changed_packet,
            changed_manifest,
            approved_preview_digest=packet.preview_digest,
            approval_receipt_digest=RECEIPT,
        )
    with pytest.raises(NlpOmopStagingError, match="invalid_preview"):
        first.bind_approval(
            replace(packet, rejected_span_count=1),
            manifest,
            approved_preview_digest=packet.preview_digest,
            approval_receipt_digest=RECEIPT,
        )


@pytest.mark.parametrize(
    "kind", ["missing", "extra", "wrong_strategy", "wrong_payload"]
)
def test_rollback_coverage_must_match_private_material(kind):
    stage = staged()
    instructions = list(stage.rollback_requirements)
    if kind == "missing":
        instructions.pop()
    elif kind == "extra":
        instructions.append(
            replace(instructions[0], mutation_ordinal=len(instructions))
        )
    elif kind == "wrong_strategy":
        from openmed.interop.omop_rollback_manifest import RollbackStrategy

        instructions[0] = replace(
            instructions[0], strategy=RollbackStrategy.RESTORE_BEFORE_IMAGE
        )
    else:
        instructions[0] = replace(instructions[0], rollback_artifact_digest=RECEIPT)
    with pytest.raises(NlpOmopStagingError, match="invalid_rollback"):
        stage.build_rollback_manifest(instructions)


def test_private_row_inputs_are_frozen_and_material_returns_copies():
    source = tables()
    stage = staged(tables=source)
    digest = stage.batch.batch_digest
    source.tables["note_nlp"][0]["snippet"] = "edited-private-value"
    assert stage.batch.batch_digest == digest
    material = stage.rollback_material(0)
    material["key"].clear()
    assert stage.rollback_material(0)["key"]


@pytest.mark.parametrize(
    "change",
    [
        "offset",
        "missing_note",
        "nlp_system",
        "source_digest",
        "unknown_column",
        "duplicate_row",
    ],
)
def test_malformed_source_rows_have_value_free_errors(change):
    source = tables()
    rows = copy.deepcopy(source.tables)
    if change == "offset":
        rows["note_nlp"][0]["offset_end"] = 10_000
    elif change == "missing_note":
        rows["note_nlp"][0]["note_id"] = 123
    elif change == "nlp_system":
        rows["note_nlp"][0]["nlp_system"] = ""
    elif change == "source_digest":
        rows["note"][0]["source_note_hash"] = CANARY
    elif change == "unknown_column":
        rows["note_nlp"][0][CANARY] = CANARY
    else:
        rows["note_nlp"] = rows["note_nlp"] + rows["note_nlp"]
    with pytest.raises(NlpOmopStagingError) as exc:
        staged(tables=replace(source, tables=rows))
    assert CANARY not in str(exc.value)


def test_lineage_record_rejects_free_text_system_and_pipeline_values():
    record = staged().lineage.records[0]
    with pytest.raises(NlpOmopStagingError, match="invalid_digest"):
        replace(record, pipeline_digest=CANARY)
    with pytest.raises(NlpOmopStagingError, match="invalid_offsets"):
        replace(record, start=True)
    with pytest.raises(NlpOmopStagingError, match="invalid_lineage"):
        NlpOmopWriteLineage((CANARY,))


def with_rows(source, rows):
    return replace(
        source,
        tables=rows,
        summary=replace(
            source.summary, row_counts={n: len(r) for n, r in rows.items()}
        ),
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing_domain",
        "missing_column",
        "event",
        "person",
        "visit",
        "concept",
        "source_hash",
        "missing_mapping_row",
    ],
)
def test_source_relationships_are_verified_in_both_directions(change):
    source = tables()
    rows = copy.deepcopy(source.tables)
    if change == "missing_domain":
        rows["condition_occurrence"] = ()
    elif change == "missing_column":
        rows["note_nlp"][0].pop("note_nlp_event_id")
    elif change == "event":
        rows["note_nlp"][0]["note_nlp_event_id"] += 1
    elif change in {"person", "visit"}:
        field = "person_id" if change == "person" else "visit_occurrence_id"
        rows["condition_occurrence"][0][field] += 1
    elif change == "concept":
        rows["condition_occurrence"][0]["condition_concept_id"] = 0
    elif change == "source_hash":
        rows["condition_occurrence"][0]["source_note_hash"] = "2" * 64
    else:
        rows["source_to_concept_map"] = ()
        # Target provenance supplied separately is still required by the gate.
        # The loader lineage contract requires its span mapping row as well.
    with pytest.raises(NlpOmopStagingError):
        staged(tables=with_rows(source, rows))


def test_replace_is_scoped_to_patient_and_note_not_hash_alone():
    first = notes(stale=True)
    other = notes(stale=True, person="other-synthetic-person")
    old = load_grounded_notes(first + other, vocabulary_version="release-current")
    own_note = tables().table("note")[0]["note_id"]
    removed_nlps = [r for r in old.table("note_nlp") if r["note_id"] == own_note]
    proof = {
        OmopRowKey("note_nlp", {"note_nlp_id": r["note_nlp_id"]}).digest: OLD_PIPELINE
        for r in removed_nlps
    }
    stage = staged(
        tables=tables(mode="replace_by_note"),
        existing_tables=old,
        existing_pipeline_digests=proof,
    )
    packet, _ = preview(stage)
    assert packet.is_approvable
    assert dict(packet.batch_preview.operation_counts) == {"insert": 4, "tombstone": 7}
    untouched = {
        OmopRowKey("note_nlp", {"note_nlp_id": r["note_nlp_id"]}).digest
        for r in old.table("note_nlp")
        if r["note_id"] != own_note
    }
    assert not untouched.intersection(m.key.digest for m in stage.batch.mutations)


def test_old_pipeline_aliases_cannot_hide_duplicate_evidence():
    old = tables()
    proof = old_pipelines(old)
    key = next(iter(proof))
    proof[key.removeprefix("sha256:")] = OLD_PIPELINE
    with pytest.raises(NlpOmopStagingError, match="invalid_lineage"):
        staged(
            tables=tables(mode="replace_by_note"),
            existing_tables=old,
            existing_pipeline_digests=proof,
        )


def test_append_duplicate_and_parent_changes_never_become_implicit_upserts():
    old = tables()
    stage = staged(existing_tables=old)
    packet, _ = preview(stage)
    assert not packet.is_approvable
    assert {i.code for i in packet.batch_preview.issues} == {"duplicate_insert"}
    rows = copy.deepcopy(old.tables)
    rows["person"][0]["person_source_value"] = CANARY
    with pytest.raises(NlpOmopStagingError, match="parent_conflict"):
        staged(tables=replace(old, tables=rows), existing_tables=old)


def test_all_five_domains_have_exact_character_lineage_with_normalized_lexical_text():
    example = notes()[0]
    text = "α condition drug measurement procedure observation"
    domains = ("Condition", "Drug", "Measurement", "Procedure", "Observation")
    example["note_text"] = text
    example["entities"] = [
        {
            "text": "NORMALIZED-" + surface,
            "start": text.index(surface),
            "end": text.index(surface) + len(surface),
            "domain_id": domain,
            "concept_id": 101 + i,
            "vocabulary_id": "SYNTHETIC",
            "code": f"C-{101 + i}",
            "concept_name": "Synthetic",
        }
        for i, (domain, surface) in enumerate(zip(domains, text.split()[1:]))
    ]
    loaded = load_grounded_notes([example], vocabulary_version="release-current")
    vocab = VocabularySnapshot(
        {"SYNTHETIC": "release-current"},
        tuple(
            VocabularyConcept(101 + i, "SYNTHETIC", standard_concept="S")
            for i in range(5)
        ),
    )
    provenance = tuple(
        VocabularyMappingProvenance(101 + i, "SYNTHETIC", "release-current")
        for i in range(5)
    )
    stage = staged(
        tables=loaded, vocabulary_snapshot=vocab, vocabulary_mappings=provenance
    )
    packet, _ = preview(stage)
    assert packet.is_approvable
    assert len(stage.lineage.records) == 15
    assert {r.table for r in stage.lineage.records} == {
        "note_nlp",
        "source_to_concept_map",
        "condition_occurrence",
        "drug_exposure",
        "measurement",
        "procedure_occurrence",
        "observation",
    }
    assert all(
        r.note_content_digest == "sha256:" + hashlib.sha256(text.encode()).hexdigest()
        for r in stage.lineage.records
    )
    assert all(text[r.start : r.end] in text.split()[1:] for r in stage.lineage.records)


def test_vocabulary_provenance_cannot_be_swapped_for_a_matching_target_id():
    stage = staged(
        vocabulary_mappings=(
            VocabularyMappingProvenance(101, "OTHER", "release-current"),
        )
    )
    packet, _ = preview(stage)
    assert packet.provenance_mismatch_count == 1
    assert not packet.is_approvable


def test_unmapped_zero_target_cannot_pass_the_existing_vocabulary_gate():
    source = notes()
    source[0]["entities"][0].update(concept_id=0, vocabulary_id="UNMAPPED")
    loaded = load_grounded_notes(source, vocabulary_version="release-current")
    stage = staged(
        tables=loaded,
        vocabulary_mappings=(
            VocabularyMappingProvenance(0, "UNMAPPED", "release-current"),
        ),
    )
    packet, _ = preview(stage)
    assert not packet.vocabulary_compatible and not packet.is_approvable


def test_bounded_inputs_and_adapter_exceptions_never_echo_values():
    with pytest.raises(NlpOmopStagingError, match="input_too_large"):
        staged(vocabulary_mappings=repeat(mappings()[0]))

    def failing():
        raise RuntimeError(CANARY)
        yield

    with pytest.raises(NlpOmopStagingError, match="invalid_input") as exc:
        staged(vocabulary_mappings=failing())
    assert CANARY not in str(exc.value)


def test_private_rollback_material_hashes_match_exact_expected_artifacts():
    old = tables(stale=True)
    stage = staged(
        tables=tables(mode="replace_by_note"),
        existing_tables=old,
        existing_pipeline_digests=old_pipelines(old),
    )
    fields = {
        "note": "note_id",
        "note_nlp": "note_nlp_id",
        "condition_occurrence": "condition_occurrence_id",
        "source_to_concept_map": "source_to_concept_map_id",
    }
    old_rows = {
        OmopRowKey(table, {field: row[field]}).digest: row
        for table, field in fields.items()
        for row in old.table(table)
    }
    for instruction in stage.rollback_requirements:
        material = stage.rollback_material(instruction.mutation_ordinal)
        canonical = json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        assert (
            instruction.rollback_artifact_digest
            == "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
        )
        mutation = stage.batch.mutations[instruction.mutation_ordinal]
        if mutation.operation.value == "tombstone":
            assert material["before_image"] == old_rows[mutation.key.digest]


@pytest.mark.parametrize(
    "kind", ["table", "field", "digest", "schema", "counts", "issue"]
)
def test_forged_public_preview_cannot_echo_private_metadata(kind):
    from openmed.interop.omop import OmopReferenceIssue

    packet, _ = preview(staged())
    inner = packet.batch_preview
    if kind == "schema":
        changed = replace(inner, schema=CANARY)
    elif kind == "counts":
        changed = replace(inner, operation_counts=((CANARY, 1),))
    elif kind == "issue":
        changed = replace(inner, issues=(OmopReferenceIssue(0, CANARY, "note"),))
    else:
        item = inner.mutations[0]
        kwargs = (
            {"table": CANARY}
            if kind == "table"
            else {"field_names": (CANARY,)}
            if kind == "field"
            else {"row_digest": CANARY}
        )
        changed = replace(
            inner, mutations=(replace(item, **kwargs), *inner.mutations[1:])
        )
    with pytest.raises(NlpOmopStagingError, match="invalid_preview") as exc:
        replace(packet, batch_preview=changed)
    assert CANARY not in str(exc.value)


@pytest.mark.parametrize("boundary", ["staging", "lineage", "rollback", "evidence"])
def test_private_iterator_exceptions_do_not_remain_in_public_context(boundary):
    from openmed.interop.omop import OmopMutationBatch, OmopMutationError

    def failing():
        raise RuntimeError(CANARY)
        yield

    stage = staged()
    with pytest.raises((NlpOmopStagingError, OmopMutationError)) as caught:
        if boundary == "staging":
            staged(vocabulary_mappings=failing())
        elif boundary == "lineage":
            NlpOmopWriteLineage(failing())
        elif boundary == "rollback":
            stage.build_rollback_manifest(failing())
        else:
            OmopMutationBatch(stage.batch.mutations, evidence_digests=failing())
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert CANARY not in str(caught.value)


@pytest.mark.parametrize("boundary", ["preview", "binding"])
def test_private_preflight_failures_do_not_remain_in_public_context(
    monkeypatch, boundary
):
    import openmed.interop.lineage.nlp_omop_writes as module
    from openmed.interop.omop import OmopMutationBatch

    stage = staged()
    packet, manifest = preview(stage)

    def failing(*args, **kwargs):
        raise RuntimeError(CANARY)

    if boundary == "preview":
        monkeypatch.setattr(module, "validate_omop_rollback_manifest", failing)
    else:
        monkeypatch.setattr(OmopMutationBatch, "bind_approval", failing)
    with pytest.raises(NlpOmopStagingError) as caught:
        if boundary == "preview":
            stage.preview(manifest)
        else:
            stage.bind_approval(
                packet,
                manifest,
                approved_preview_digest=packet.preview_digest,
                approval_receipt_digest=RECEIPT,
            )
    assert caught.value.__context__ is None
    assert CANARY not in str(caught.value)


def test_private_invalid_rollback_json_has_no_public_exception_context():
    stage = staged()
    changed = replace(
        stage, _rollback_materials=(CANARY, *stage._rollback_materials[1:])
    )
    with pytest.raises(NlpOmopStagingError) as caught:
        changed.rollback_material(0)
    assert caught.value.__context__ is None


def test_private_note_unicode_failure_has_no_public_exception_context():
    source = tables()
    rows = copy.deepcopy(source.tables)
    rows["note"][0]["note_text"] += "\ud800" + CANARY
    with pytest.raises(NlpOmopStagingError) as caught:
        staged(tables=replace(source, tables=rows))
    assert caught.value.__context__ is None


def test_foreign_staging_error_is_not_forwarded_or_inspected():
    getter_calls = []

    class ForeignError(NlpOmopStagingError):
        def __init__(self):
            ValueError.__init__(self, CANARY)

        @property
        def code(self):
            getter_calls.append(True)
            return CANARY

    def failing():
        raise ForeignError()
        yield

    source = tables()
    rows = dict(source.tables)
    rows["note"] = failing()
    with pytest.raises(NlpOmopStagingError) as caught:
        staged(tables=replace(source, tables=rows))
    assert type(caught.value) is NlpOmopStagingError
    assert caught.value.__context__ is None
    assert not getter_calls
    assert CANARY not in str(caught.value)


def test_operation_count_string_subclasses_cannot_impersonate_closed_metadata():
    class Impersonator(str):
        def __eq__(self, other):
            return other == "insert"

        def __hash__(self):
            return hash("insert")

    packet, _ = preview(staged())
    counts = ((Impersonator(CANARY), packet.batch_preview.operation_counts[0][1]),)
    with pytest.raises(NlpOmopStagingError):
        replace(
            packet, batch_preview=replace(packet.batch_preview, operation_counts=counts)
        )
