"""Tests for the Doccano JSONL annotation interchange adapter."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from openmed.cli._output import EXIT_ERROR, CliError
from openmed.cli.annotation_interchange import add_annotation_interchange_command
from openmed.clinical.journey_contracts import derived_opaque_id, sha256_digest
from openmed.core.labels import normalize_label
from openmed.core.schemas import hmac_text_hash
from openmed.eval.annotation import (
    AnnotationEnvelope,
    AnnotationInterchangeError,
    AnnotationLossKind,
    AnnotationRecord,
    AnnotationState,
    AnnotationType,
    AnnotationValidationError,
    CoordinateConvention,
    build_annotation_envelope,
    export_doccano,
    import_doccano,
)

_SECRET = "synthetic-doccano-key"
_TEXT = "Ada Lovelace founded Analytical Engines in London."
_DOC_ID = "doc_synthetic01"
_TEXT_DIGEST = sha256_digest(_TEXT)
_KEY_ENV = "OPENMED_ANNOTATION_KEY"


def _entity(identifier: int, surface: str, label: str) -> dict[str, Any]:
    start = _TEXT.index(surface)
    return {
        "id": identifier,
        "start_offset": start,
        "end_offset": start + len(surface),
        "label": label,
    }


def _document(
    *,
    entities: list[dict[str, Any]] | None = None,
    label_spans: list[list[Any]] | None = None,
    relations: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> str:
    payload: dict[str, Any] = {"text": _TEXT}
    if entities is not None:
        payload["entities"] = entities
    if label_spans is not None:
        payload["label"] = label_spans
    if relations is not None:
        payload["relations"] = relations
    payload.update(extra)
    return json.dumps(payload)


def _canonical_document() -> str:
    """Doccano line whose labels already match the OpenMed taxonomy."""

    return _document(
        entities=[
            _entity(0, "Ada Lovelace", "person"),
            _entity(1, "Analytical Engines", "organization"),
            _entity(2, "London", "location"),
        ],
        relations=[{"id": 0, "from_id": 0, "to_id": 1, "type": "foundedat"}],
    )


def _upstream_document() -> str:
    """Doccano line using upstream example labels and a mixed-case type."""

    return _document(
        entities=[
            _entity(0, "Ada Lovelace", "PERSON"),
            _entity(1, "Analytical Engines", "ORG"),
            _entity(2, "London", "LOCATION"),
        ],
        relations=[{"id": 0, "from_id": 0, "to_id": 1, "type": "foundedAt"}],
    )


def _import(line: str, **overrides: Any):
    options: dict[str, Any] = {
        "text_digest": _TEXT_DIGEST,
        "doc_id": _DOC_ID,
        "hash_secret": _SECRET,
    }
    options.update(overrides)
    return import_doccano(line, **options)


def _entity_record(
    *,
    start: int,
    end: int,
    label: str = "person",
    document_id: str = _DOC_ID,
    state: AnnotationState = AnnotationState.SUCCESS,
    embedding: tuple[float, ...] | None = None,
) -> AnnotationRecord:
    return AnnotationRecord(
        annotation_id=derived_opaque_id(
            "synthetic_span", document_id, str(start), str(end)
        ),
        document_id=document_id,
        namespace="default",
        annotation_type=AnnotationType.ENTITY,
        coordinate_convention=CoordinateConvention.UNICODE_CODEPOINT,
        start=start,
        end=end,
        state=state,
        result={"label": label, "surface_hash": sha256_digest(_TEXT[start:end])},
        embedding=embedding,
    )


def _relation_record(
    source: AnnotationRecord,
    target: AnnotationRecord,
    *,
    relation: str = "foundedat",
    state: AnnotationState = AnnotationState.SUCCESS,
) -> AnnotationRecord:
    return AnnotationRecord(
        annotation_id=derived_opaque_id(
            "synthetic_relation", source.annotation_id, target.annotation_id
        ),
        document_id=source.document_id,
        namespace="default",
        annotation_type=AnnotationType.RELATION,
        coordinate_convention=CoordinateConvention.NONE,
        start=None,
        end=None,
        state=state,
        result={
            "relation": relation,
            "source_annotation_id": source.annotation_id,
            "target_annotation_id": target.annotation_id,
        },
    )


def _fact_correction_record() -> AnnotationRecord:
    fact_id = derived_opaque_id("synthetic_fact", "fact-1")
    return AnnotationRecord(
        annotation_id=fact_id,
        document_id=_DOC_ID,
        namespace="default",
        annotation_type=AnnotationType.FACT_CORRECTION,
        coordinate_convention=CoordinateConvention.NONE,
        start=None,
        end=None,
        state=AnnotationState.SUCCESS,
        result={
            "fact_id": fact_id,
            "field": "code",
            "reason_code": "stale",
            "replacement_code": "current",
        },
    )


def _entities(envelope: AnnotationEnvelope) -> list[AnnotationRecord]:
    return [
        record
        for record in envelope.records
        if record.annotation_type is AnnotationType.ENTITY
    ]


def _relations(envelope: AnnotationEnvelope) -> list[AnnotationRecord]:
    return [
        record
        for record in envelope.records
        if record.annotation_type is AnnotationType.RELATION
    ]


def test_import_and_export_round_trip_preserves_offsets_and_relation_pairs() -> None:
    imported = _import(_canonical_document())

    exported = export_doccano(imported.envelope, text=_TEXT, text_digest=_TEXT_DIGEST)

    assert json.loads(exported.text) == json.loads(_canonical_document())
    assert exported.report.state is AnnotationState.SUCCESS
    assert exported.report.entries == ()
    assert [record.result["label"] for record in _entities(imported.envelope)] == [
        "person",
        "organization",
        "location",
    ]


def test_import_canonicalizes_upstream_labels_and_declares_the_rewrite() -> None:
    imported = _import(_upstream_document())

    assert [record.result["label"] for record in _entities(imported.envelope)] == [
        normalize_label("PERSON").lower(),
        normalize_label("ORG").lower(),
        normalize_label("LOCATION").lower(),
    ]
    relations = _relations(imported.envelope)
    assert len(relations) == 1
    assert relations[0].result["relation"] == "foundedat"
    assert imported.report.state is AnnotationState.PARTIAL
    assert [entry.code for entry in imported.report.entries] == [
        "doccano_relation_type_normalized"
    ]


def test_import_report_binds_the_payload_and_the_envelope_digests() -> None:
    line = _canonical_document()

    imported = _import(line)

    assert imported.report.operation == "import_doccano"
    assert imported.report.input_digest == sha256_digest(line)
    assert imported.report.output_digest == imported.envelope.envelope_digest


def test_import_hashes_surfaces_without_storing_source_text() -> None:
    imported = _import(_canonical_document())

    entities = _entities(imported.envelope)
    assert entities[0].result["surface_hash"] == sha256_digest(
        hmac_text_hash(_TEXT[0:12], _SECRET)
    )
    serialized = imported.envelope.to_json()
    assert _TEXT not in serialized
    assert "Ada Lovelace" not in serialized
    assert '"text"' not in serialized


def test_import_accepts_the_sequence_labeling_dialect() -> None:
    imported = _import(_document(label_spans=[[0, 12, "PERSON"], [21, 39, "ORG"]]))

    assert [
        (record.start, record.end, record.result["label"])
        for record in _entities(imported.envelope)
    ] == [(0, 12, "person"), (21, 39, "organization")]


def test_import_accepts_utf8_bytes_and_requires_annotations() -> None:
    imported = _import(_canonical_document().encode("utf-8"))

    assert len(imported.envelope.records) == 4
    empty = _import(_document(entities=[]))
    assert empty.envelope.records == ()
    assert empty.report.state is AnnotationState.SUCCESS


def test_import_rejects_a_changed_document() -> None:
    with pytest.raises(AnnotationValidationError, match="de-identification digest"):
        _import(_canonical_document(), text_digest=sha256_digest("some other text"))


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (
            _document(entities=[_entity(0, "Ada Lovelace", "NOT_A_LABEL")]),
            "unknown label",
        ),
        (
            _document(
                entities=[
                    {"id": 0, "start_offset": 0, "end_offset": 500, "label": "PERSON"}
                ]
            ),
            "outside the document",
        ),
        (
            _document(
                entities=[
                    {"id": 0, "start_offset": 30, "end_offset": 4, "label": "PERSON"}
                ]
            ),
            "outside the document",
        ),
        (
            _document(
                entities=[
                    {"id": 0, "start_offset": 0.5, "end_offset": 4, "label": "PERSON"}
                ]
            ),
            "must be integers",
        ),
        (
            _document(entities=[{"start_offset": 0, "end_offset": 12}]),
            "is missing label",
        ),
        (
            _document(
                entities=[
                    _entity(0, "Ada Lovelace", "PERSON"),
                    _entity(0, "London", "LOCATION"),
                ]
            ),
            "invalid id",
        ),
        (
            _document(
                entities=[_entity(0, "Ada Lovelace", "PERSON")],
                relations=[{"id": 0, "from_id": 0, "to_id": 7, "type": "foundedat"}],
            ),
            "entity that was not imported",
        ),
        (
            _document(
                entities=[_entity(0, "Ada Lovelace", "PERSON")],
                relations=[{"id": 0, "from_id": 0, "to_id": 0}],
            ),
            "is missing type",
        ),
    ],
)
def test_import_rejects_invalid_annotations(line: str, message: str) -> None:
    with pytest.raises(AnnotationValidationError, match=message) as excinfo:
        _import(line)

    assert excinfo.value.format_name == "Doccano JSONL"


def test_import_declares_unsupported_entity_and_relation_fields() -> None:
    line = _document(
        entities=[
            {
                "id": 0,
                "start_offset": 0,
                "end_offset": 12,
                "label": "PERSON",
                "confidence": 0.9,
            }
        ],
        relations=[],
        comments=["synthetic"],
    )

    imported = _import(line)

    assert sorted({entry.code for entry in imported.report.entries}) == [
        "doccano_unsupported_field"
    ]
    assert len(imported.report.entries) == 2
    assert "confidence" not in imported.envelope.to_json()
    assert "comments" not in imported.envelope.to_json()


def test_import_rejects_unrepresentable_relation_types() -> None:
    line = _document(
        entities=[
            _entity(0, "Ada Lovelace", "PERSON"),
            _entity(1, "Analytical Engines", "ORG"),
        ],
        relations=[{"id": 0, "from_id": 0, "to_id": 1, "type": "###"}],
    )

    with pytest.raises(AnnotationInterchangeError, match="controlled identifier"):
        _import(line)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (_document(), "must carry Doccano entities or label spans"),
        (
            _document(
                entities=[_entity(0, "Ada Lovelace", "PERSON")],
                label_spans=[[0, 12, "PERSON"]],
            ),
            "must not mix",
        ),
        (_document(entities={}), "entities must be a JSON array"),
        (_document(label_spans={"a": 1}), "label spans must be a JSON array"),
        (_document(label_spans=[[0, 12]]), "triple"),
        (_document(label_spans=[["a", "b", "PERSON"]]), "must be integers"),
    ],
)
def test_import_rejects_malformed_annotation_shapes(line: str, message: str) -> None:
    with pytest.raises(AnnotationValidationError, match=message):
        _import(line)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{not json", "valid JSON object"),
        ('{"text": "synthetic", "label": [], "x": NaN}', "non-finite number"),
        ('{"text": "synthetic", "label": [], "y": Infinity}', "non-finite number"),
        ('{"text": "synthetic", "text": "other", "label": []}', "duplicate keys"),
        (
            '{"text": "synthetic", "label": [], "deep": '
            + "[" * 20000
            + "]" * 20000
            + "}",
            "valid JSON object",
        ),
        (
            '{"text": "synthetic", "entities": [{"id": '
            + "9" * 5000
            + ', "start_offset": 0, "end_offset": 9, "label": "PERSON"}]}',
            "valid JSON object",
        ),
        (b"\xff\xfe\x00", "UTF-8 encoded"),
        (42, "must be text or UTF-8 bytes"),
        ("", "import one Doccano document per call"),
        (
            _canonical_document() + "\n" + _canonical_document(),
            "import one Doccano document per call",
        ),
        (
            json.dumps({"text": "a" * 65537, "label": []}),
            "exceeds the annotation cell limit",
        ),
    ],
    ids=[
        "invalid-json",
        "nan",
        "infinity",
        "duplicate-keys",
        "deep-nesting",
        "huge-integer",
        "not-utf8",
        "not-text",
        "empty-payload",
        "two-documents",
        "oversized-cell",
    ],
)
def test_import_fails_closed_on_hostile_payloads(payload: Any, message: str) -> None:
    with pytest.raises(
        (AnnotationValidationError, AnnotationInterchangeError), match=message
    ):
        import_doccano(
            payload, text_digest=_TEXT_DIGEST, doc_id=_DOC_ID, hash_secret=_SECRET
        )


def test_import_requires_an_opaque_document_identifier() -> None:
    with pytest.raises(AnnotationInterchangeError, match="opaque identifier"):
        _import(_canonical_document(), doc_id="doc-1")

    with pytest.raises(AnnotationValidationError, match="doc_id must be non-empty"):
        _import(_canonical_document(), doc_id="")


def test_import_requires_a_hash_secret() -> None:
    with pytest.raises((AnnotationValidationError, ValueError)):
        _import(_canonical_document(), hash_secret="")


def test_export_refuses_text_without_a_matching_de_identification_digest() -> None:
    envelope = build_annotation_envelope([_entity_record(start=0, end=12)])

    with pytest.raises(AnnotationInterchangeError, match="SHA-256 digest"):
        export_doccano(envelope, text=_TEXT, text_digest="not-a-digest")
    with pytest.raises(AnnotationInterchangeError, match="de-identification digest"):
        export_doccano(envelope, text=_TEXT, text_digest=sha256_digest("other"))
    with pytest.raises(AnnotationInterchangeError, match="requires the de-identified"):
        export_doccano(envelope, text="", text_digest=sha256_digest(""))


def test_export_declares_losses_for_records_doccano_cannot_express() -> None:
    plain = _entity_record(start=0, end=12)
    embedded = _entity_record(
        start=21, end=39, label="organization", embedding=(0.5, 0.25)
    )
    review = _entity_record(
        start=43, end=49, label="location", state=AnnotationState.UNKNOWN
    )
    envelope = build_annotation_envelope(
        [plain, embedded, review, _fact_correction_record()]
    )

    exported = export_doccano(envelope, text=_TEXT, text_digest=_TEXT_DIGEST)

    payload = json.loads(exported.text)
    assert payload["entities"] == [
        {"end_offset": 12, "id": 0, "label": "person", "start_offset": 0},
        {"end_offset": 39, "id": 1, "label": "organization", "start_offset": 21},
    ]
    assert "embedding" not in exported.text
    assert payload["relations"] == []
    assert exported.report.state is AnnotationState.PARTIAL
    kinds = {entry.code: entry.kind for entry in exported.report.entries}
    assert kinds == {
        "doccano_embedding_not_exported": AnnotationLossKind.EMBEDDING_OMITTED,
        "doccano_state_not_exported": AnnotationLossKind.REVIEW_REQUIRED,
        "doccano_annotation_type_not_exported": (
            AnnotationLossKind.UNSUPPORTED_ANNOTATION
        ),
    }


def test_export_skips_relations_whose_entities_were_not_exported() -> None:
    plain = _entity_record(start=0, end=12)
    review = _entity_record(
        start=21, end=39, label="organization", state=AnnotationState.UNKNOWN
    )
    envelope = build_annotation_envelope(
        [plain, review, _relation_record(plain, review)]
    )

    with pytest.raises(AnnotationInterchangeError, match="point at exported entities"):
        export_doccano(envelope, text=_TEXT, text_digest=_TEXT_DIGEST)


def test_export_rejects_offsets_outside_the_exported_text() -> None:
    envelope = build_annotation_envelope([_entity_record(start=0, end=len(_TEXT) + 5)])

    with pytest.raises(AnnotationInterchangeError, match="outside the exported text"):
        export_doccano(envelope, text=_TEXT, text_digest=_TEXT_DIGEST)


def test_export_selects_one_document_explicitly() -> None:
    first = _entity_record(start=0, end=12)
    second = _entity_record(
        start=21, end=39, label="organization", document_id="doc_synthetic02"
    )
    envelope = build_annotation_envelope([first, second])

    with pytest.raises(AnnotationInterchangeError, match="explicit document_id"):
        export_doccano(envelope, text=_TEXT, text_digest=_TEXT_DIGEST)

    exported = export_doccano(
        envelope, text=_TEXT, text_digest=_TEXT_DIGEST, document_id=_DOC_ID
    )

    assert [entity["id"] for entity in json.loads(exported.text)["entities"]] == [0]
    with pytest.raises(AnnotationInterchangeError, match="not present"):
        export_doccano(
            envelope, text=_TEXT, text_digest=_TEXT_DIGEST, document_id="doc_missing1"
        )


def test_adapter_imports_no_network_or_doccano_client_modules() -> None:
    import openmed.eval.annotation.doccano_io as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    roots = {
        line.split()[1].split(".")[0]
        for line in source.splitlines()
        if line.startswith(("import ", "from "))
    }
    forbidden = {"requests", "httpx", "urllib", "urllib3", "socket", "aiohttp"}
    assert roots.isdisjoint(forbidden)
    assert "doccano" not in roots


def test_adapter_never_opens_a_network_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the Doccano adapter must not use the network")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    imported = _import(_canonical_document())
    exported = export_doccano(imported.envelope, text=_TEXT, text_digest=_TEXT_DIGEST)

    assert json.loads(exported.text)["text"] == _TEXT


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openmed")
    add_annotation_interchange_command(parser.add_subparsers(dest="command"))
    return parser


def test_cli_doccano_import_and_export_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "synthetic.jsonl"
    source.write_text(_canonical_document() + "\n", encoding="utf-8")
    text_path = tmp_path / "synthetic.txt"
    text_path.write_text(_TEXT, encoding="utf-8")
    envelope_path = tmp_path / "annotations.json"
    loss_path = tmp_path / "import-loss.json"
    exported_path = tmp_path / "exported.jsonl"
    monkeypatch.setenv(_KEY_ENV, _SECRET)

    parser = _cli_parser()
    imported = parser.parse_args(
        [
            "annotation",
            "import",
            "--input",
            str(source),
            "--output",
            str(envelope_path),
            "--format",
            "doccano",
            "--doc-id",
            _DOC_ID,
            "--text-digest",
            _TEXT_DIGEST,
            "--loss-report",
            str(loss_path),
        ]
    )
    assert imported.handler(imported) == 0
    assert AnnotationEnvelope.from_json(envelope_path.read_bytes())
    assert (
        json.loads(loss_path.read_text(encoding="utf-8"))["operation"]
        == "import_doccano"
    )

    exported = parser.parse_args(
        [
            "annotation",
            "export",
            "--input",
            str(envelope_path),
            "--output",
            str(exported_path),
            "--format",
            "doccano",
            "--text",
            str(text_path),
            "--text-digest",
            _TEXT_DIGEST,
        ]
    )
    assert exported.handler(exported) == 0
    assert json.loads(exported_path.read_text(encoding="utf-8")) == json.loads(
        _canonical_document()
    )


def test_cli_annotation_commands_default_to_canonical_tsv() -> None:
    parser = _cli_parser()

    imported = parser.parse_args(
        ["annotation", "import", "--input", "in.tsv", "--output", "out.json"]
    )
    exported = parser.parse_args(
        ["annotation", "export", "--input", "in.json", "--output", "out.tsv"]
    )

    assert imported.format == "tsv"
    assert exported.format == "tsv"


@pytest.mark.parametrize(
    ("argv", "code", "message", "with_key"),
    [
        (
            ["annotation", "import", "--format", "doccano", "--doc-id", _DOC_ID],
            "annotation_import_failed",
            "--text-digest is required",
            True,
        ),
        (
            [
                "annotation",
                "import",
                "--format",
                "doccano",
                "--text-digest",
                _TEXT_DIGEST,
            ],
            "annotation_import_failed",
            "--doc-id is required",
            True,
        ),
        (
            [
                "annotation",
                "import",
                "--format",
                "doccano",
                "--doc-id",
                _DOC_ID,
                "--text-digest",
                _TEXT_DIGEST,
            ],
            "annotation_import_failed",
            "non-empty HMAC key",
            False,
        ),
    ],
)
def test_cli_doccano_import_requires_the_binding_inputs(
    argv: list[str],
    code: str,
    message: str,
    with_key: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "synthetic.jsonl"
    source.write_text(_canonical_document() + "\n", encoding="utf-8")
    if with_key:
        monkeypatch.setenv(_KEY_ENV, _SECRET)
    else:
        monkeypatch.delenv(_KEY_ENV, raising=False)

    parsed = _cli_parser().parse_args(
        [*argv, "--input", str(source), "--output", str(tmp_path / "out.json")]
    )

    with pytest.raises(CliError, match=message) as excinfo:
        parsed.handler(parsed)

    assert excinfo.value.code == code
    assert excinfo.value.exit_code == EXIT_ERROR


def test_cli_doccano_export_requires_the_de_identified_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = build_annotation_envelope([_entity_record(start=0, end=12)])
    envelope_path = tmp_path / "annotations.json"
    envelope_path.write_text(envelope.to_json() + "\n", encoding="utf-8")
    monkeypatch.setenv(_KEY_ENV, _SECRET)

    parsed = _cli_parser().parse_args(
        [
            "annotation",
            "export",
            "--input",
            str(envelope_path),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--format",
            "doccano",
            "--text-digest",
            _TEXT_DIGEST,
        ]
    )

    with pytest.raises(CliError, match="--text is required") as excinfo:
        parsed.handler(parsed)

    assert excinfo.value.code == "annotation_export_failed"
