"""Eval-only loader for the MADE 1.0 medication/ADE corpus.

MADE 1.0 is distributed as a credentialed local BioC corpus.  The loader also
accepts paired BRAT files so synthetic tests can exercise the same entity and
relation contract as the source data without bundling corpus rows.  It never
downloads, caches, or reads a default path inside the repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.labels import (
    CANONICAL_LABELS,
    CONDITION,
    DOSAGE,
    DURATION,
    FREQUENCY,
    INDICATION,
    MEDICATION,
    ROUTE,
    SEVERITY,
)
from openmed.eval.datasets.drugprot import (
    DrugProtEntity,
    DrugProtRelation,
    DrugProtRelationFixture,
)
from openmed.eval.datasets.dua_stubs import DUACredentialRequired
from openmed.eval.datasets.licenses import license_for
from openmed.eval.harness import BenchmarkFixture

MADE = "made"
MADE_1_0 = MADE
MADE_VERSION = "1.0"
MADE_DUA_NAME = "MADE 1.0/UMass DUA"
MADE_PATH_ENV = "OPENMED_MADE_PATH"

MADE_ENTITY_TYPES: tuple[str, ...] = (
    "ADE",
    "Indication",
    "Other SSD",
    "Severity",
    "Drugname",
    "Dosage",
    "Duration",
    "Frequency",
    "Route",
)
MADE_ENTITY_TO_CANONICAL: Mapping[str, str] = {
    "ADE": CONDITION,
    "Indication": INDICATION,
    "Other SSD": CONDITION,
    "Severity": SEVERITY,
    "Drugname": MEDICATION,
    "Dosage": DOSAGE,
    "Duration": DURATION,
    "Frequency": FREQUENCY,
    "Route": ROUTE,
}

MADE_RELATION_TYPES: tuple[str, ...] = (
    "ADE-Drugname",
    "Indication-Drugname",
    "Drugname-Dosage",
    "Drugname-Route",
    "Drugname-Frequency",
    "Drugname-Duration",
    "SSD-Severity",
)
MADE_RELATION_TO_CANONICAL: Mapping[str, str] = {
    "ADE-Drugname": "DRUG_TO_ADE",
    "Indication-Drugname": "DRUG_TO_INDICATION",
    "Drugname-Dosage": "DRUG_TO_DOSE",
    "Drugname-Route": "DRUG_TO_ROUTE",
    "Drugname-Frequency": "DRUG_TO_FREQUENCY",
    "Drugname-Duration": "DRUG_TO_DURATION",
    "SSD-Severity": "SSD_TO_SEVERITY",
}

MADE_SUITE_METADATA: Mapping[str, Any] = {
    "access": (f"credentialed local path only; pass path=... or set {MADE_PATH_ENV}"),
    "annotation_format": "BioC JSON/XML or BRAT standoff",
    "cache_corpus_rows": False,
    "dataset": MADE,
    "dua": MADE_DUA_NAME,
    "entity_label_mapping": dict(sorted(MADE_ENTITY_TO_CANONICAL.items())),
    "eval_only": True,
    "license": license_for(MADE).to_dict(),
    "network_fetch": False,
    "redistribution": "credentialed eval-only; never redistributed",
    "relation_type_mapping": dict(sorted(MADE_RELATION_TO_CANONICAL.items())),
    "suite": MADE,
    "task": "relation",
    "version": MADE_VERSION,
}

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BIOC_SUFFIXES = frozenset({".bioc", ".json", ".jsonl", ".xml"})


class MADECredentialRequired(DUACredentialRequired):
    """Raised when the MADE 1.0 DUA path is not configured."""


def map_made_entity_label(label: str) -> str:
    """Map a MADE 1.0 entity type onto an OpenMed canonical label."""

    aliases = {
        "ade": "ADE",
        "adverse_drug_event": "ADE",
        "drug": "Drugname",
        "drug_name": "Drugname",
        "medication": "Drugname",
        "other_ssd": "Other SSD",
        "otherssd": "Other SSD",
        "ssd": "Other SSD",
    }
    key = _mapping_key(label)
    source_label = aliases.get(key, label)
    canonical = _lookup_mapping(
        source_label,
        MADE_ENTITY_TO_CANONICAL,
        kind="entity label",
    )
    _ensure_canonical(canonical, source_label)
    return canonical


def map_made_relation_type(relation_type: str) -> str:
    """Map a MADE relation type onto the shared relation schema."""

    aliases = {
        "adedrug": "ADE-Drugname",
        "indicationdrug": "Indication-Drugname",
        "drugdosage": "Drugname-Dosage",
        "drugroute": "Drugname-Route",
        "drugfrequency": "Drugname-Frequency",
        "drugduration": "Drugname-Duration",
        "ssdseverity": "SSD-Severity",
    }
    key = _mapping_key(relation_type)
    source_type = aliases.get(key, relation_type)
    return _lookup_mapping(
        source_type,
        MADE_RELATION_TO_CANONICAL,
        kind="relation type",
    )


def made_suite_metadata() -> dict[str, Any]:
    """Return row-free metadata for MADE 1.0."""

    return dict(MADE_SUITE_METADATA)


def load_made_relation_fixtures(
    path: str | Path | None = None,
) -> list[DrugProtRelationFixture]:
    """Load MADE relation fixtures from a credentialed local path only."""

    source = _credentialed_path(path)
    sources = _source_files(source)
    fixtures: list[DrugProtRelationFixture] = []
    for source_kind, source_path, annotation_path in sources:
        if source_kind == "brat":
            assert annotation_path is not None
            fixtures.append(
                _brat_fixture_from_pair(
                    source_path,
                    annotation_path,
                    root=source,
                )
            )
            continue
        for document in _bioc_documents(source_path):
            fixtures.append(
                _bioc_fixture_from_document(
                    document,
                    source=source_path,
                    root=source,
                )
            )
    if not fixtures:
        raise MADECredentialRequired(
            f"{MADE_DUA_NAME} path contains no supported MADE fixtures; "
            "no corpus rows were loaded"
        )
    _validate_unique_fixture_ids(fixtures)
    return fixtures


def load_made_ner_fixtures(
    path: str | Path | None = None,
) -> list[BenchmarkFixture]:
    """Load the MADE entity view as benchmark NER fixtures."""

    relation_fixtures = load_made_relation_fixtures(path)
    return [_ner_fixture(fixture) for fixture in relation_fixtures]


def load_made_fixtures(
    path: str | Path | None = None,
    *,
    task: str = "relation",
) -> list[BenchmarkFixture | DrugProtRelationFixture]:
    """Load the requested MADE NER or relation view."""

    normalized_task = str(task).strip().casefold().replace("-", "_")
    if normalized_task in {"ner", "entity", "entities"}:
        return load_made_ner_fixtures(path)
    if normalized_task in {"relation", "relations", "re"}:
        return load_made_relation_fixtures(path)
    raise ValueError("MADE task must be 'ner' or 'relation'")


def load_made(
    path: str | Path | None = None,
    *,
    task: str = "relation",
) -> list[BenchmarkFixture | DrugProtRelationFixture]:
    """Load the requested MADE 1.0 view."""

    return load_made_fixtures(path, task=task)


load_made_1_0 = load_made
load_made_1_0_ner_fixtures = load_made_ner_fixtures
load_made_1_0_relation_fixtures = load_made_relation_fixtures


def _credentialed_path(path: str | Path | None) -> Path:
    raw_path = path if path is not None else os.environ.get(MADE_PATH_ENV)
    if raw_path is None or not str(raw_path).strip():
        raise MADECredentialRequired(
            f"{MADE_DUA_NAME} credentialed local path is required; pass path=... "
            f"or set {MADE_PATH_ENV}. No corpus rows were loaded."
        )
    candidate = Path(raw_path).expanduser().resolve(strict=False)
    if _is_relative_to(candidate, _REPO_ROOT):
        raise MADECredentialRequired(
            f"{MADE_DUA_NAME} data must stay outside the repository tree; "
            f"refusing to read {candidate}. No corpus rows were loaded."
        )
    if not candidate.exists():
        raise MADECredentialRequired(
            f"{MADE_DUA_NAME} credentialed path does not exist: {candidate}. "
            "No corpus rows were loaded."
        )
    if not candidate.is_file() and not candidate.is_dir():
        raise MADECredentialRequired(
            f"{MADE_DUA_NAME} path must be a file or directory: {candidate}. "
            "No corpus rows were loaded."
        )
    return candidate


def _source_files(
    root: Path,
) -> tuple[tuple[str, Path, Path | None], ...]:
    if root.is_file():
        if root.suffix.casefold() == ".txt":
            annotation_path = root.with_suffix(".ann")
            if not annotation_path.is_file():
                raise MADECredentialRequired(
                    f"{MADE_DUA_NAME} BRAT text requires paired .ann file; "
                    f"no corpus rows were loaded: {root.name}"
                )
            return (("brat", root, annotation_path),)
        if root.suffix.casefold() == ".ann":
            text_path = root.with_suffix(".txt")
            if not text_path.is_file():
                raise MADECredentialRequired(
                    f"{MADE_DUA_NAME} BRAT annotation requires paired .txt file; "
                    f"no corpus rows were loaded: {root.name}"
                )
            return (("brat", text_path, root),)
        if _is_bioc_source(root):
            return (("bioc", root, None),)
        raise MADECredentialRequired(
            f"{MADE_DUA_NAME} path has no supported BioC or BRAT extension; "
            f"no corpus rows were loaded: {root.name}"
        )

    sources: list[tuple[str, Path, Path | None]] = []
    annotation_paths = sorted(
        candidate for candidate in root.rglob("*.ann") if candidate.is_file()
    )
    for annotation_path in annotation_paths:
        text_path = annotation_path.with_suffix(".txt")
        if not text_path.is_file():
            raise ValueError(
                f"MADE BRAT input requires paired .ann and .txt files: "
                f"{annotation_path.name}"
            )
        sources.append(("brat", text_path, annotation_path))
    sources.extend(
        ("bioc", candidate, None)
        for candidate in sorted(root.rglob("*"))
        if candidate.is_file() and _is_bioc_source(candidate)
    )
    if not sources:
        raise MADECredentialRequired(
            f"{MADE_DUA_NAME} path contains no BioC or paired BRAT files; "
            "no corpus rows were loaded"
        )
    return tuple(sources)


def _is_bioc_source(path: Path) -> bool:
    name = path.name.casefold()
    return path.suffix.casefold() in _BIOC_SUFFIXES or name.endswith(
        (".bioc.json", ".bioc.jsonl", ".bioc.xml")
    )


def _bioc_documents(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        documents: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(_read_exact(path).splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid MADE BioC JSONL row {line_number}: {exc}"
                ) from exc
            documents.extend(_bioc_json_documents(payload))
        return documents

    contents = _read_exact(path)
    if path.suffix.casefold() == ".xml" or contents.lstrip().startswith("<"):
        return _bioc_xml_documents(contents, source_name=path.name)
    try:
        payload = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid MADE BioC JSON {path.name}: {exc}") from exc
    return _bioc_json_documents(payload)


def _bioc_json_documents(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        documents: list[Mapping[str, Any]] = []
        for value in payload:
            documents.extend(_bioc_json_documents(value))
        return documents
    if not isinstance(payload, Mapping):
        raise ValueError("MADE BioC JSON must be a mapping or list")
    collection = payload.get("collection")
    if isinstance(collection, Mapping):
        return _bioc_json_documents(collection)
    documents = payload.get("documents")
    if isinstance(documents, list):
        return [_require_mapping(value, "MADE BioC document") for value in documents]
    if payload.get("id") is not None and (
        isinstance(payload.get("passages"), list) or "text" in payload
    ):
        return [payload]
    raise ValueError("MADE BioC JSON contains no documents")


def _bioc_xml_documents(
    contents: str,
    *,
    source_name: str,
) -> list[Mapping[str, Any]]:
    try:
        root = ET.fromstring(contents)
    except ET.ParseError as exc:
        raise ValueError(f"failed to parse MADE BioC XML {source_name}: {exc}") from exc
    documents = [
        _bioc_xml_document(element)
        for element in root.iter()
        if _local_name(element.tag).casefold() == "document"
    ]
    if not documents:
        raise ValueError(f"MADE BioC XML {source_name} contains no documents")
    return documents


def _bioc_xml_document(element: ET.Element) -> Mapping[str, Any]:
    passages: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for child in element:
        name = _local_name(child.tag).casefold()
        if name == "passage":
            passages.append(_bioc_xml_passage(child))
        elif name == "relation":
            relations.append(_bioc_xml_relation(child))
    return {
        "id": _bioc_child_text(element, "id"),
        "passages": passages,
        "relations": relations,
    }


def _bioc_xml_passage(element: ET.Element) -> dict[str, Any]:
    annotations: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for child in element:
        name = _local_name(child.tag).casefold()
        if name == "annotation":
            annotations.append(_bioc_xml_annotation(child))
        elif name == "relation":
            relations.append(_bioc_xml_relation(child))
    return {
        "offset": _parse_int(_bioc_child_text(element, "offset"), "BioC offset"),
        "text": _bioc_child_text(element, "text"),
        "annotations": annotations,
        "relations": relations,
    }


def _bioc_xml_annotation(element: ET.Element) -> dict[str, Any]:
    return {
        "id": str(element.attrib.get("id") or ""),
        "infons": _bioc_xml_infons(element),
        "locations": [
            {
                "offset": _parse_int(child.attrib.get("offset"), "BioC offset"),
                "length": _parse_int(child.attrib.get("length"), "BioC length"),
            }
            for child in element
            if _local_name(child.tag).casefold() == "location"
        ],
        "text": _bioc_child_text(element, "text"),
    }


def _bioc_xml_relation(element: ET.Element) -> dict[str, Any]:
    return {
        "id": str(element.attrib.get("id") or ""),
        "infons": _bioc_xml_infons(element),
        "nodes": [
            {
                "refid": str(child.attrib.get("refid") or ""),
                "role": str(child.attrib.get("role") or ""),
            }
            for child in element
            if _local_name(child.tag).casefold() == "node"
        ],
    }


def _bioc_xml_infons(element: ET.Element) -> dict[str, str]:
    return {
        str(child.attrib.get("key") or ""): str(child.text or "")
        for child in element
        if _local_name(child.tag).casefold() == "infon"
    }


def _bioc_child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if _local_name(child.tag).casefold() == name.casefold():
            return str(child.text or "")
    return ""


def _bioc_fixture_from_document(
    document: Mapping[str, Any],
    *,
    source: Path,
    root: Path,
) -> DrugProtRelationFixture:
    document_id = str(document.get("id") or document.get("document_id") or "").strip()
    if not document_id:
        raise ValueError(f"MADE document in {source.name} is missing an id")
    raw_passages = document.get("passages")
    if not isinstance(raw_passages, list) or not raw_passages:
        text_value = str(document.get("text") or "")
        if not text_value:
            raise ValueError(f"MADE document {document_id!r} contains no passages")
        passages: list[Mapping[str, Any]] = [
            {
                "offset": 0,
                "text": text_value,
                "annotations": document.get("annotations")
                or document.get("entities")
                or [],
                "relations": document.get("relations") or [],
            }
        ]
    else:
        passages = [_require_mapping(value, "MADE passage") for value in raw_passages]
    text = _bioc_document_text(passages)
    fixture_id = _fixture_id(source, root, document_id)
    entities_by_id: dict[str, DrugProtEntity] = {}
    for annotation in _bioc_annotation_rows(document, passages):
        entity = _made_entity_from_mapping(
            annotation,
            fixture_id=fixture_id,
            text=text,
        )
        if entity.entity_id in entities_by_id:
            raise ValueError(f"duplicate MADE entity id: {entity.entity_id}")
        entities_by_id[entity.entity_id] = entity

    relations = tuple(
        _made_relation_from_mapping(
            relation,
            fixture_id=fixture_id,
            entities_by_id=entities_by_id,
            relation_index=index,
        )
        for index, relation in enumerate(
            _bioc_relation_rows(document, passages),
            start=1,
        )
    )
    metadata = {
        **MADE_SUITE_METADATA,
        "source_path_hash": _source_path_hash(source, root),
        "task": "relation",
    }
    return DrugProtRelationFixture(
        fixture_id=fixture_id,
        text=text,
        entities=tuple(
            sorted(
                entities_by_id.values(),
                key=lambda entity: (entity.start, entity.end, entity.entity_id),
            )
        ),
        relations=relations,
        metadata=metadata,
    )


def _bioc_document_text(passages: Sequence[Mapping[str, Any]]) -> str:
    positioned: list[tuple[int, str]] = []
    next_offset = 0
    for passage in passages:
        offset_value = passage.get("offset")
        offset = (
            _parse_int(offset_value, "BioC passage offset")
            if offset_value is not None
            else next_offset
        )
        passage_text = str(passage.get("text") or "")
        if offset < 0:
            raise ValueError("BioC passage offset must be non-negative")
        positioned.append((offset, passage_text))
        next_offset = max(next_offset, offset + len(passage_text))
    if not positioned:
        raise ValueError("MADE document contains no passages")
    text_length = max(offset + len(value) for offset, value in positioned)
    characters = [" "] * text_length
    for offset, value in positioned:
        for index, character in enumerate(value, start=offset):
            current = characters[index]
            if current != " " and current != character:
                raise ValueError("overlapping MADE passages contain conflicting text")
            characters[index] = character
    return "".join(characters)


def _bioc_annotation_rows(
    document: Mapping[str, Any],
    passages: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    values: list[Any] = list(document.get("annotations") or [])
    values.extend(document.get("entities") or [])
    for passage in passages:
        values.extend(passage.get("annotations") or passage.get("entities") or [])
    return [_require_mapping(value, "MADE annotation") for value in values]


def _bioc_relation_rows(
    document: Mapping[str, Any],
    passages: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    values: list[Any] = list(document.get("relations") or [])
    for passage in passages:
        values.extend(passage.get("relations") or [])
    return [_require_mapping(value, "MADE relation") for value in values]


def _made_entity_from_mapping(
    annotation: Mapping[str, Any],
    *,
    fixture_id: str,
    text: str,
) -> DrugProtEntity:
    entity_id = str(annotation.get("id") or annotation.get("entity_id") or "").strip()
    if not entity_id:
        raise ValueError("MADE annotation id is required")
    infons = _mapping_or_empty(annotation.get("infons"))
    source_label = str(
        infons.get("type")
        or infons.get("entity_type")
        or annotation.get("type")
        or annotation.get("label")
        or ""
    )
    canonical_label = map_made_entity_label(source_label)
    locations_value = annotation.get("locations") or annotation.get("location")
    if locations_value is None:
        if annotation.get("offset") is None or annotation.get("length") is None:
            raise ValueError(f"MADE entity {entity_id!r} has no location")
        locations_value = [annotation]
    if isinstance(locations_value, Mapping):
        locations_value = [locations_value]
    if not isinstance(locations_value, list) or not locations_value:
        raise ValueError(f"MADE entity {entity_id!r} has no location")
    locations = [_require_mapping(value, "MADE location") for value in locations_value]
    starts = [
        _parse_int(value.get("offset"), "BioC entity offset") for value in locations
    ]
    ends = [
        start + _parse_int(value.get("length"), "BioC entity length")
        for start, value in zip(starts, locations, strict=True)
    ]
    start, end = min(starts), max(ends)
    if start < 0 or end <= start or end > len(text):
        raise ValueError(
            f"invalid MADE span offsets {start}:{end} for text length {len(text)}"
        )
    span_text = text[start:end]
    annotated_text = str(annotation.get("text") or "")
    if len(locations) == 1 and annotated_text:
        if _surface(annotated_text) != _surface(span_text):
            raise ValueError(f"MADE span text mismatch for entity {entity_id!r}")
    return DrugProtEntity(
        pmid=fixture_id,
        entity_id=entity_id,
        source_label=source_label,
        start=start,
        end=end,
        text=span_text,
        canonical_label=canonical_label,
    )


def _made_relation_from_mapping(
    relation: Mapping[str, Any],
    *,
    fixture_id: str,
    entities_by_id: Mapping[str, DrugProtEntity],
    relation_index: int,
) -> DrugProtRelation:
    infons = _mapping_or_empty(relation.get("infons"))
    source_type = str(
        infons.get("type")
        or infons.get("relation")
        or relation.get("type")
        or relation.get("label")
        or ""
    )
    relation_type = map_made_relation_type(source_type)
    node_ids = _relation_node_ids(relation)
    try:
        arguments = [entities_by_id[node_id] for node_id in node_ids]
    except KeyError as exc:
        raise ValueError(
            f"MADE relation references unknown entity {exc.args[0]!r}"
        ) from exc
    arg1, arg2 = _orient_relation(relation_type, arguments)
    relation_id = str(relation.get("id") or f"R{relation_index}")
    return DrugProtRelation(
        pmid=fixture_id,
        relation_type=relation_type,
        arg1_id=arg1.entity_id,
        arg2_id=arg2.entity_id,
        arg1=arg1,
        arg2=arg2,
        scope="document",
        relation_id=relation_id,
        metadata={
            "canonical_relation_type": relation_type,
            "source_relation_type": source_type,
        },
    )


def _relation_node_ids(relation: Mapping[str, Any]) -> list[str]:
    nodes_value = relation.get("nodes") or relation.get("arguments")
    if isinstance(nodes_value, Mapping):
        nodes_value = list(nodes_value.values())
    if isinstance(nodes_value, list):
        rows = [_require_mapping(value, "MADE relation node") for value in nodes_value]
        if len(rows) != 2:
            raise ValueError("MADE relation must contain exactly two nodes")
        node_ids = [
            str(row.get("refid") or row.get("id") or row.get("entity_id") or "").strip()
            for row in rows
        ]
        if any(not node_id for node_id in node_ids):
            raise ValueError("MADE relation nodes require entity references")
        role_indexes = {
            "arg1": 0,
            "argument1": 0,
            "entity1": 0,
            "head": 0,
            "source": 0,
            "subject": 0,
            "arg2": 1,
            "argument2": 1,
            "entity2": 1,
            "object": 1,
            "tail": 1,
            "target": 1,
        }
        ordered: list[str | None] = [None, None]
        for node_id, row in zip(node_ids, rows, strict=True):
            index = role_indexes.get(_mapping_key(str(row.get("role") or "")))
            if index is None or ordered[index] is not None:
                ordered = [None, None]
                break
            ordered[index] = node_id
        if all(node_id is not None for node_id in ordered):
            return [node_id for node_id in ordered if node_id is not None]
        return node_ids

    flat_ids = [
        relation.get("arg1_id") or relation.get("arg1") or relation.get("source"),
        relation.get("arg2_id") or relation.get("arg2") or relation.get("target"),
    ]
    node_ids = [str(value or "").strip() for value in flat_ids]
    if any(not node_id for node_id in node_ids):
        raise ValueError("MADE relation requires two entity references")
    return node_ids


def _orient_relation(
    relation_type: str,
    arguments: Sequence[DrugProtEntity],
) -> tuple[DrugProtEntity, DrugProtEntity]:
    if len(arguments) != 2:
        raise ValueError("MADE relations must be binary")
    if relation_type.startswith("DRUG_TO_"):
        medication_indexes = [
            index
            for index, entity in enumerate(arguments)
            if entity.canonical_label == MEDICATION
        ]
        if len(medication_indexes) != 1:
            raise ValueError(
                "MADE medication relations must connect exactly one medication"
            )
        medication = arguments[medication_indexes[0]]
        other = arguments[1 - medication_indexes[0]]
        return medication, other
    severity_indexes = [
        index
        for index, entity in enumerate(arguments)
        if entity.canonical_label == SEVERITY
    ]
    if len(severity_indexes) != 1:
        raise ValueError("MADE SSD-Severity relations require one severity entity")
    severity = arguments[severity_indexes[0]]
    other = arguments[1 - severity_indexes[0]]
    return other, severity


def _brat_fixture_from_pair(
    text_path: Path,
    annotation_path: Path,
    *,
    root: Path,
) -> DrugProtRelationFixture:
    text = _read_exact(text_path)
    fixture_id = _fixture_id(text_path, root, text_path.stem)
    entities_by_id: dict[str, DrugProtEntity] = {}
    lines = _read_exact(annotation_path).splitlines()
    for line in lines:
        if not line.startswith("T"):
            continue
        entity = _brat_entity_from_line(
            line,
            fixture_id=fixture_id,
            text=text,
        )
        if entity.entity_id in entities_by_id:
            raise ValueError(f"duplicate MADE entity id: {entity.entity_id}")
        entities_by_id[entity.entity_id] = entity

    relations = tuple(
        _brat_relation_from_line(
            line,
            fixture_id=fixture_id,
            entities_by_id=entities_by_id,
        )
        for line in lines
        if line.startswith("R")
    )
    return DrugProtRelationFixture(
        fixture_id=fixture_id,
        text=text,
        entities=tuple(
            sorted(
                entities_by_id.values(),
                key=lambda entity: (entity.start, entity.end, entity.entity_id),
            )
        ),
        relations=relations,
        metadata={
            **MADE_SUITE_METADATA,
            "source_path_hash": _source_path_hash(text_path, root),
            "task": "relation",
        },
    )


def _brat_entity_from_line(
    line: str,
    *,
    fixture_id: str,
    text: str,
) -> DrugProtEntity:
    columns = line.split("\t", 2)
    if len(columns) != 3:
        raise ValueError("malformed MADE BRAT entity line")
    entity_id = columns[0].strip()
    label_and_offsets = columns[1].split(maxsplit=1)
    if len(label_and_offsets) != 2:
        raise ValueError("malformed MADE BRAT entity offsets")
    source_label, offset_spec = label_and_offsets
    spans: list[tuple[int, int]] = []
    for segment in offset_spec.split(";"):
        values = segment.split()
        if len(values) != 2:
            raise ValueError("malformed MADE discontinuous BRAT span")
        start, end = (_parse_int(value, "BRAT span offset") for value in values)
        spans.append((start, end))
    start, end = min(value[0] for value in spans), max(value[1] for value in spans)
    if start < 0 or end <= start or end > len(text):
        raise ValueError(
            f"invalid MADE span offsets {start}:{end} for text length {len(text)}"
        )
    span_text = text[start:end]
    if len(spans) == 1 and _surface(columns[2]) != _surface(span_text):
        raise ValueError(f"MADE span text mismatch for entity {entity_id!r}")
    return DrugProtEntity(
        pmid=fixture_id,
        entity_id=entity_id,
        source_label=source_label,
        start=start,
        end=end,
        text=span_text,
        canonical_label=map_made_entity_label(source_label),
    )


def _brat_relation_from_line(
    line: str,
    *,
    fixture_id: str,
    entities_by_id: Mapping[str, DrugProtEntity],
) -> DrugProtRelation:
    columns = line.split("\t", 1)
    if len(columns) != 2:
        raise ValueError("malformed MADE BRAT relation line")
    relation_id = columns[0].strip()
    values = columns[1].split()
    if len(values) != 3:
        raise ValueError("malformed MADE BRAT relation arguments")
    source_type = values[0]
    node_ids = [_brat_argument_id(value) for value in values[1:]]
    try:
        arguments = [entities_by_id[node_id] for node_id in node_ids]
    except KeyError as exc:
        raise ValueError(
            f"MADE relation references unknown entity {exc.args[0]!r}"
        ) from exc
    relation_type = map_made_relation_type(source_type)
    arg1, arg2 = _orient_relation(relation_type, arguments)
    return DrugProtRelation(
        pmid=fixture_id,
        relation_type=relation_type,
        arg1_id=arg1.entity_id,
        arg2_id=arg2.entity_id,
        arg1=arg1,
        arg2=arg2,
        scope="document",
        relation_id=relation_id,
        metadata={
            "canonical_relation_type": relation_type,
            "source_relation_type": source_type,
        },
    )


def _brat_argument_id(value: str) -> str:
    _, separator, entity_id = value.partition(":")
    if separator != ":" or not entity_id:
        raise ValueError(f"malformed MADE BRAT relation argument: {value!r}")
    return entity_id


def _ner_fixture(fixture: DrugProtRelationFixture) -> BenchmarkFixture:
    return BenchmarkFixture(
        fixture_id=fixture.fixture_id,
        text=fixture.text,
        gold_spans=tuple(
            entity.to_eval_span(fixture.text) for entity in fixture.entities
        ),
        language=fixture.language,
        metadata={
            **dict(fixture.metadata or {}),
            "relation_count": len(fixture.relations),
            "task": "ner",
        },
    )


def _lookup_mapping(
    value: str,
    mapping: Mapping[str, str],
    *,
    kind: str,
) -> str:
    key = _mapping_key(value)
    for source, canonical in mapping.items():
        if _mapping_key(source) == key:
            return canonical
    allowed = ", ".join(mapping)
    raise ValueError(f"unknown MADE {kind} {value!r}; expected one of: {allowed}")


def _ensure_canonical(canonical: str, source_label: str) -> None:
    if canonical not in CANONICAL_LABELS:
        raise RuntimeError(
            f"MADE mapping for {source_label!r} is not canonical: {canonical!r}"
        )


def _mapping_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return value


def _parse_int(value: Any, description: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be an integer") from exc


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _read_exact(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _surface(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ")


def _fixture_id(source: Path, root: Path, document_id: str) -> str:
    relative_source = _relative_source_path(source, root)
    digest = hashlib.sha256(
        f"{MADE}:{relative_source}:{document_id}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{MADE}-{digest}"


def _source_path_hash(source: Path, root: Path) -> str:
    return hashlib.sha256(
        _relative_source_path(source, root).encode("utf-8")
    ).hexdigest()


def _relative_source_path(source: Path, root: Path) -> str:
    base = root if root.is_dir() else root.parent
    try:
        return source.relative_to(base).as_posix()
    except ValueError:
        return source.name


def _validate_unique_fixture_ids(
    fixtures: Sequence[DrugProtRelationFixture],
) -> None:
    ids = [fixture.fixture_id for fixture in fixtures]
    duplicates = sorted({fixture_id for fixture_id in ids if ids.count(fixture_id) > 1})
    if duplicates:
        raise ValueError(
            "duplicate MADE relation fixture ids: " + ", ".join(duplicates)
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


_missing_entity_mappings = sorted(
    set(MADE_ENTITY_TYPES) - set(MADE_ENTITY_TO_CANONICAL)
)
_extra_entity_mappings = sorted(set(MADE_ENTITY_TO_CANONICAL) - set(MADE_ENTITY_TYPES))
_missing_relation_mappings = sorted(
    set(MADE_RELATION_TYPES) - set(MADE_RELATION_TO_CANONICAL)
)
_extra_relation_mappings = sorted(
    set(MADE_RELATION_TO_CANONICAL) - set(MADE_RELATION_TYPES)
)
if (
    _missing_entity_mappings
    or _extra_entity_mappings
    or _missing_relation_mappings
    or _extra_relation_mappings
):
    raise RuntimeError(
        "MADE mappings must cover the source tables exactly; "
        f"missing_entities={_missing_entity_mappings}, "
        f"extra_entities={_extra_entity_mappings}, "
        f"missing_relations={_missing_relation_mappings}, "
        f"extra_relations={_extra_relation_mappings}"
    )
for _source_label, _canonical_label in MADE_ENTITY_TO_CANONICAL.items():
    _ensure_canonical(_canonical_label, _source_label)


__all__ = [
    "MADE",
    "MADE_1_0",
    "MADE_VERSION",
    "MADE_DUA_NAME",
    "MADE_PATH_ENV",
    "MADE_ENTITY_TYPES",
    "MADE_ENTITY_TO_CANONICAL",
    "MADE_RELATION_TYPES",
    "MADE_RELATION_TO_CANONICAL",
    "MADE_SUITE_METADATA",
    "MADECredentialRequired",
    "load_made",
    "load_made_1_0",
    "load_made_1_0_ner_fixtures",
    "load_made_1_0_relation_fixtures",
    "load_made_fixtures",
    "load_made_ner_fixtures",
    "load_made_relation_fixtures",
    "made_suite_metadata",
    "map_made_entity_label",
    "map_made_relation_type",
]
