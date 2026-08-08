"""Eval-only loader for the credentialed Mayo-THYME temporal corpus."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

from openmed.core.labels import (
    CANONICAL_LABELS,
    CONDITION,
    DATE,
    DURATION,
    FINDING,
    MEDICATION,
    OTHER,
    PROBLEM,
    PROCEDURE,
    TIME,
    normalize_label,
)
from openmed.eval.metrics import EvalSpan
from openmed.eval.relation_metrics import EvalRelation

from ._dua import (
    fixture_id,
    load_json_rows,
    require_credentialed_path,
    source_path_hash,
)
from ._task_fixtures import RelationTaskFixture
from .dua_stubs import DUACredentialRequired
from .licenses import license_for

THYME = "thyme"
THYME_PATH_ENV = "OPENMED_THYME_PATH"
THYME_AUTHORITY = "Mayo-THYME"
THYME_DUA = "Mayo-THYME data-use terms"
THYME_TASK = "temporal_event_relation"
THYME_RELATION_TYPES: tuple[str, ...] = (
    "BEFORE",
    "AFTER",
    "OVERLAP",
    "CONTAINS",
    "BEGINS_ON",
    "ENDS_ON",
)
THYME_ENTITY_TO_CANONICAL: Mapping[str, str] = {
    "condition": CONDITION,
    "date": DATE,
    "diagnosis": CONDITION,
    "duration": DURATION,
    "event": OTHER,
    "event_mention": OTHER,
    "finding": FINDING,
    "medication": MEDICATION,
    "problem": PROBLEM,
    "procedure": PROCEDURE,
    "symptom": PROBLEM,
    "time": TIME,
    "timex": DATE,
    "timex2": DATE,
    "timex3": DATE,
    "timex_2": DATE,
    "timex_3": DATE,
    "temporal_expression": DATE,
}

_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})
_THYME_SPAN_NAMES = {"EVENT", "TIMEX", "TIMEX2", "TIMEX3", "ENTITY"}
_THYME_RELATION_NAMES = {"TLINK", "RELATION", "TEMPORAL_RELATION"}
_RELATION_ALIASES = {
    "before": "BEFORE",
    "after": "AFTER",
    "begins": "BEGINS_ON",
    "begins_on": "BEGINS_ON",
    "contains": "CONTAINS",
    "ends": "ENDS_ON",
    "ends_on": "ENDS_ON",
    "overlap": "OVERLAP",
}


def load_thyme(path: str | Path | None = None) -> list[RelationTaskFixture]:
    """Load THYME JSON/JSONL, XML, or paired BRAT temporal annotations."""

    root = require_credentialed_path(
        path,
        dataset=THYME,
        authority=THYME_AUTHORITY,
        env_var=THYME_PATH_ENV,
    )
    fixtures: list[RelationTaskFixture] = []
    for source in _files(root, _JSON_SUFFIXES):
        for row_number, row in enumerate(
            load_json_rows(source, dataset=THYME, authority=THYME_AUTHORITY),
            start=1,
        ):
            fixtures.append(
                _fixture_from_row(row, source=source, root=root, row_number=row_number)
            )
    for source in _files(root, {".xml"}):
        fixtures.append(_fixture_from_xml(source, root=root))
    for annotation_path in _paired_annotation_files(root):
        text_path = annotation_path.with_suffix(".txt")
        if not text_path.exists():
            raise ValueError(
                f"THYME BRAT annotation {annotation_path.name} requires a paired .txt file"
            )
        fixtures.append(_fixture_from_brat(text_path, annotation_path, root=root))

    if not fixtures:
        raise DUACredentialRequired(
            f"{THYME_AUTHORITY} credentialed {THYME} path contains no supported "
            "fixtures; no corpus rows were loaded"
        )
    _validate_unique_fixture_ids(fixtures)
    return fixtures


load_thyme_fixtures = load_thyme
load_thyme_relation_fixtures = load_thyme


def map_thyme_entity_label(label: str, *, role: str = "") -> str:
    """Map a THYME EVENT/TIMEX label to a canonical span label."""

    key = _mapping_key(label)
    role_key = _mapping_key(role)
    if role_key in {"timex", "timex2", "timex3", "temporal_expression"}:
        key = role_key
    canonical = THYME_ENTITY_TO_CANONICAL.get(key)
    if canonical is None:
        normalized = normalize_label(str(label).strip())
        if normalized in CANONICAL_LABELS:
            canonical = normalized
    if canonical is None:
        allowed = ", ".join(sorted(THYME_ENTITY_TO_CANONICAL))
        raise ValueError(
            f"unknown THYME entity label {label!r}; expected one of: {allowed}"
        )
    return canonical


def map_thyme_relation_type(relation_type: str) -> str:
    """Normalize a THYME TLINK relation type."""

    key = _mapping_key(relation_type)
    try:
        return _RELATION_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(THYME_RELATION_TYPES)
        raise ValueError(
            f"unknown THYME temporal relation {relation_type!r}; expected one of: "
            f"{allowed}"
        ) from exc


def thyme_suite_metadata() -> dict[str, Any]:
    """Return row-free metadata for the THYME temporal relation view."""

    return {
        "access": (
            f"credentialed local path only; pass path=... or set {THYME_PATH_ENV}"
        ),
        "dataset": THYME,
        "dua": THYME_DUA,
        "eval_only": True,
        "entity_label_mapping": dict(sorted(THYME_ENTITY_TO_CANONICAL.items())),
        "license": license_for(THYME).to_dict(),
        "network_fetch": False,
        "redistribution": "never; read-only from user-supplied credentialed path",
        "relation_type_mapping": dict(sorted(_RELATION_ALIASES.items())),
        "suite": THYME,
        "task": "relation",
        "task_view": THYME_TASK,
    }


def _fixture_from_row(
    row: Mapping[str, Any],
    *,
    source: Path,
    root: Path,
    row_number: int,
) -> RelationTaskFixture:
    text = _row_text(row)
    record_id = _record_id(row, fallback=f"row-{row_number}")
    raw_entities = []
    for key in ("entities", "spans", "annotations", "events", "timexes", "timex"):
        value = row.get(key)
        if isinstance(value, list):
            raw_entities.extend(value)
        elif isinstance(value, Mapping):
            raw_entities.extend(
                {"id": item_id, **dict(item)}
                for item_id, item in value.items()
                if isinstance(item, Mapping)
            )
    entities = _entities_from_rows(raw_entities, text=text, source_name=source.name)
    resolved_id = fixture_id(THYME, source, root, record_id)
    relations = _relations_from_rows(
        row.get("relations") or row.get("tlinks") or row.get("gold_relations") or [],
        entities=entities,
        fixture_id=resolved_id,
    )
    return RelationTaskFixture(
        fixture_id=resolved_id,
        text=text,
        entities=entities,
        gold_relations=tuple(relations),
        language=str(row.get("language") or row.get("lang") or "en"),
        metadata=_metadata(source, root),
    )


def _fixture_from_brat(
    text_path: Path,
    annotation_path: Path,
    *,
    root: Path,
) -> RelationTaskFixture:
    with text_path.open("r", encoding="utf-8", newline="") as handle:
        text = handle.read()
    entities: dict[str, EvalSpan] = {}
    relation_rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        annotation_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line.startswith("T"):
            columns = line.split("\t", 2)
            if len(columns) != 3:
                raise ValueError(f"malformed THYME BRAT entity at line {line_number}")
            label, (start, end) = _brat_offsets(columns[1], text=text)
            entities[columns[0]] = _span_from_values(
                start=start,
                end=end,
                label=label,
                role=label,
                text=text,
                supplied_text=columns[2],
                source_name=annotation_path.name,
            )
        elif line.startswith("R"):
            columns = line.split("\t", 1)
            if len(columns) != 2:
                raise ValueError(f"malformed THYME BRAT relation at line {line_number}")
            values = columns[1].split()
            if len(values) < 3:
                raise ValueError(f"malformed THYME BRAT relation at line {line_number}")
            relation_rows.append(
                {
                    "id": columns[0],
                    "type": values[0],
                    "head": values[1].split(":", 1)[-1],
                    "tail": values[2].split(":", 1)[-1],
                    "scope": "document",
                }
            )

    resolved_id = fixture_id(THYME, text_path, root, text_path.stem)
    return RelationTaskFixture(
        fixture_id=resolved_id,
        text=text,
        entities=entities,
        gold_relations=tuple(
            _relations_from_rows(
                relation_rows, entities=entities, fixture_id=resolved_id
            )
        ),
        metadata=_metadata(text_path, root),
    )


def _fixture_from_xml(path: Path, *, root: Path) -> RelationTaskFixture:
    try:
        document = ET.parse(path)
    except ET.ParseError as exc:
        raise ValueError(f"failed to parse THYME XML {path.name}: {exc}") from exc
    xml_root = document.getroot()
    text_element = next(
        (
            element
            for element in xml_root.iter()
            if _local_name(element.tag).upper() in {"TEXT", "RAW_TEXT", "NOTE"}
        ),
        None,
    )
    if text_element is None:
        raise ValueError(f"THYME XML {path.name} is missing a TEXT element")
    text = "".join(text_element.itertext())
    entities: dict[str, EvalSpan] = {}
    relation_rows: list[Mapping[str, Any]] = []
    for element in xml_root.iter():
        local_name = _local_name(element.tag).upper()
        if local_name in _THYME_RELATION_NAMES:
            relation_rows.append(_xml_relation_row(element))
            continue
        if local_name not in _THYME_SPAN_NAMES:
            continue
        attributes = {
            str(key).casefold(): value for key, value in element.attrib.items()
        }
        entity_id = str(
            _xml_value(element, attributes, "id")
            or _xml_value(element, attributes, "eventid")
            or _xml_value(element, attributes, "timexid")
            or f"T{len(entities) + 1}"
        )
        start, end = _xml_offsets(element, attributes)
        source_label = str(
            _xml_value(element, attributes, "type")
            or _xml_value(element, attributes, "label")
            or local_name
        )
        entities[entity_id] = _span_from_values(
            start=start,
            end=end,
            label=source_label,
            role=local_name,
            text=text,
            supplied_text=str(_xml_value(element, attributes, "text") or ""),
            source_name=path.name,
        )
    resolved_id = fixture_id(THYME, path, root, path.stem)
    return RelationTaskFixture(
        fixture_id=resolved_id,
        text=text,
        entities=entities,
        gold_relations=tuple(
            _relations_from_rows(
                relation_rows, entities=entities, fixture_id=resolved_id
            )
        ),
        metadata=_metadata(path, root),
    )


def _relations_from_rows(
    rows: Any,
    *,
    entities: Mapping[str, EvalSpan],
    fixture_id: str,
) -> list[EvalRelation]:
    if not isinstance(rows, list):
        raise ValueError("THYME temporal links must be a list")
    relations: list[EvalRelation] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("THYME temporal links must be objects")
        head_id = str(
            row.get("head")
            or row.get("source")
            or row.get("from")
            or row.get("arg1")
            or row.get("source_id")
            or ""
        )
        tail_id = str(
            row.get("tail")
            or row.get("target")
            or row.get("to")
            or row.get("arg2")
            or row.get("target_id")
            or ""
        )
        try:
            head = entities[head_id]
            tail = entities[tail_id]
        except KeyError as exc:
            raise ValueError(
                f"THYME temporal link references unknown entity {exc.args[0]!r}"
            ) from exc
        source_type = str(
            row.get("type")
            or row.get("relation_type")
            or row.get("relType")
            or row.get("predicate")
            or ""
        )
        relation_type = map_thyme_relation_type(source_type)
        relations.append(
            EvalRelation(
                relation_type=relation_type,
                head=head,
                tail=tail,
                scope=str(row.get("scope") or "document"),
                relation_id=str(row.get("id") or row.get("relation_id") or f"R{index}"),
                fixture_id=fixture_id,
                metadata={"source_relation_type": source_type},
            )
        )
    return relations


def _entities_from_rows(
    rows: list[Any],
    *,
    text: str,
    source_name: str,
) -> dict[str, EvalSpan]:
    entities: dict[str, EvalSpan] = {}
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError("THYME entity rows must be objects")
        entity_id = str(raw.get("id") or raw.get("entity_id") or f"T{index}")
        if entity_id in entities:
            raise ValueError(f"duplicate THYME entity id: {entity_id}")
        entities[entity_id] = _span_from_values(
            start=_mapping_offsets(raw)[0],
            end=_mapping_offsets(raw)[1],
            label=str(
                raw.get("label")
                or raw.get("type")
                or raw.get("entity_type")
                or raw.get("role")
                or "EVENT"
            ),
            role=str(raw.get("role") or raw.get("type") or ""),
            text=text,
            supplied_text=str(raw.get("text") or raw.get("surface") or ""),
            source_name=source_name,
        )
    return entities


def _span_from_values(
    *,
    start: int,
    end: int,
    label: str,
    role: str,
    text: str,
    supplied_text: str,
    source_name: str,
) -> EvalSpan:
    if start < 0 or end <= start or end > len(text):
        raise ValueError(f"invalid THYME span offsets {start}:{end} in {source_name}")
    span_text = text[start:end]
    if supplied_text and supplied_text != span_text:
        raise ValueError(f"THYME span text mismatch in {source_name}")
    source_label = label.strip()
    role_key = _mapping_key(role)
    if role_key in {"timex", "timex2", "timex3"}:
        source_label = role_key
    return EvalSpan(
        start=start,
        end=end,
        label=map_thyme_entity_label(source_label, role=role),
        text=span_text,
        metadata={
            "source_label": label.strip(),
            "temporal_role": "TIMEX" if role_key.startswith("timex") else "EVENT",
        },
    )


def _xml_relation_row(element: ET.Element) -> Mapping[str, Any]:
    attributes = {str(key).casefold(): value for key, value in element.attrib.items()}
    return {
        "id": _xml_value(element, attributes, "id") or "",
        "type": (
            _xml_value(element, attributes, "type")
            or _xml_value(element, attributes, "reltype")
            or _xml_value(element, attributes, "relation")
            or ""
        ),
        "head": (
            _xml_value(element, attributes, "source")
            or _xml_value(element, attributes, "from")
            or _xml_value(element, attributes, "fromid")
            or _xml_value(element, attributes, "arg1")
            or _xml_value(element, attributes, "eventinstanceid")
            or ""
        ),
        "tail": (
            _xml_value(element, attributes, "target")
            or _xml_value(element, attributes, "to")
            or _xml_value(element, attributes, "toid")
            or _xml_value(element, attributes, "arg2")
            or _xml_value(element, attributes, "relatedtoeventinstance")
            or ""
        ),
        "scope": "document",
    }


def _xml_offsets(element: ET.Element, attributes: Mapping[str, Any]) -> tuple[int, int]:
    start = (
        _xml_value(element, attributes, "start")
        or _xml_value(element, attributes, "begin")
        or _xml_value(element, attributes, "offset")
    )
    end = _xml_value(element, attributes, "end")
    if start is not None and end is not None:
        return _integer(start), _integer(end)
    length = _xml_value(element, attributes, "length")
    if start is not None and length is not None:
        parsed_start = _integer(start)
        return parsed_start, parsed_start + _integer(length)
    raise ValueError("THYME XML spans require start/end or offset/length")


def _xml_value(element: ET.Element, attributes: Mapping[str, Any], name: str) -> Any:
    if name.casefold() in attributes:
        return attributes[name.casefold()]
    wanted = name.casefold()
    for child in element:
        if _local_name(child.tag).casefold() == wanted:
            return child.text
    return None


def _mapping_offsets(item: Mapping[str, Any]) -> tuple[int, int]:
    if item.get("start") is not None and item.get("end") is not None:
        return _integer(item["start"]), _integer(item["end"])
    if item.get("offset") is not None and item.get("length") is not None:
        start = _integer(item["offset"])
        return start, start + _integer(item["length"])
    raise ValueError("THYME spans require start/end or offset/length")


def _brat_offsets(specification: str, *, text: str) -> tuple[str, tuple[int, int]]:
    values = specification.split(maxsplit=1)
    if len(values) != 2:
        raise ValueError("malformed THYME BRAT span offsets")
    label, offset_spec = values
    spans = []
    for segment in offset_spec.split(";"):
        offsets = segment.split()
        if len(offsets) != 2:
            raise ValueError("malformed THYME discontinuous BRAT span")
        spans.append((_integer(offsets[0]), _integer(offsets[1])))
    start = min(item[0] for item in spans)
    end = max(item[1] for item in spans)
    if start < 0 or end > len(text):
        raise ValueError("THYME BRAT span is outside document text")
    return label, (start, end)


def _metadata(source: Path, root: Path) -> dict[str, Any]:
    return {
        "cache_corpus_rows": False,
        "dataset": THYME,
        "dua": THYME_DUA,
        "eval_only": True,
        "license": license_for(THYME).to_dict(),
        "network_fetch": False,
        "redistribution": "never; read-only from user-supplied credentialed path",
        "source_path_hash": source_path_hash(source, root),
        "suite": THYME,
        "task": "relation",
        "task_view": THYME_TASK,
    }


def _row_text(row: Mapping[str, Any]) -> str:
    for key in ("text", "note", "document"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("THYME rows require non-empty text")


def _record_id(row: Mapping[str, Any], *, fallback: str) -> str:
    for key in ("id", "record_id", "document_id", "doc_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return fallback


def _files(root: Path, suffixes: set[str] | frozenset[str]) -> tuple[Path, ...]:
    wanted = {suffix.casefold() for suffix in suffixes}
    if root.is_file():
        return (root,) if root.suffix.casefold() in wanted else tuple()
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.casefold() in wanted
    )


def _paired_annotation_files(root: Path) -> tuple[Path, ...]:
    if root.is_file() and root.suffix.casefold() == ".txt":
        annotation_path = root.with_suffix(".ann")
        return (annotation_path,) if annotation_path.exists() else tuple()
    return _files(root, {".ann"})


def _mapping_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def _integer(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("THYME span offsets must be integers")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("THYME span offsets must be integers") from exc


def _local_name(value: str) -> str:
    return value.rsplit("}", maxsplit=1)[-1]


def _validate_unique_fixture_ids(fixtures: list[RelationTaskFixture]) -> None:
    ids = [fixture.fixture_id for fixture in fixtures]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {THYME} fixture ids")


__all__ = [
    "THYME",
    "THYME_AUTHORITY",
    "THYME_DUA",
    "THYME_ENTITY_TO_CANONICAL",
    "THYME_PATH_ENV",
    "THYME_RELATION_TYPES",
    "THYME_TASK",
    "RelationTaskFixture",
    "load_thyme",
    "load_thyme_fixtures",
    "load_thyme_relation_fixtures",
    "map_thyme_entity_label",
    "map_thyme_relation_type",
    "thyme_suite_metadata",
]
