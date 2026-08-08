"""Eval-only loader for the credentialed SHAC SDOH relation corpus."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from openmed.core.labels import CANONICAL_LABELS, OCCUPATION, OTHER, normalize_label
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

SHAC = "shac"
SHAC_PATH_ENV = "OPENMED_SHAC_PATH"
SHAC_AUTHORITY = "DBMI / n2c2 SHAC"
SHAC_DUA = "DBMI / n2c2 SHAC DUA"
SHAC_TASK = "sdoh_event_relation"
SHAC_DETERMINANTS: tuple[str, ...] = (
    "alcohol",
    "drug",
    "tobacco",
    "employment",
    "living",
)

SHAC_ENTITY_TO_CANONICAL: Mapping[str, str] = {
    "alcohol": OTHER,
    "alcohol_use": OTHER,
    "amount": OTHER,
    "drug": OTHER,
    "drug_use": OTHER,
    "duration": OTHER,
    "employment": OCCUPATION,
    "employment_status": OTHER,
    "extent": OTHER,
    "frequency": OTHER,
    "history": OTHER,
    "living": OTHER,
    "living_situation": OTHER,
    "living_status": OTHER,
    "status": OTHER,
    "temporality": OTHER,
    "tobacco": OTHER,
    "tobacco_status": OTHER,
    "tobacco_use": OTHER,
    "substance": OTHER,
    "substance_use": OTHER,
    "type": OTHER,
}
SHAC_RELATION_TO_CANONICAL: Mapping[str, str] = {
    "amount": "HAS_AMOUNT",
    "duration": "HAS_DURATION",
    "extent": "HAS_EXTENT",
    "frequency": "HAS_FREQUENCY",
    "history": "HAS_HISTORY",
    "status": "HAS_STATUS",
    "temporality": "HAS_TEMPORALITY",
    "type": "HAS_TYPE",
}

_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})
_DETERMINANT_ALIASES = {
    "alcohol": "alcohol",
    "alcohol_use": "alcohol",
    "drugs": "drug",
    "drug": "drug",
    "drug_use": "drug",
    "employment": "employment",
    "job": "employment",
    "living": "living",
    "living_status": "living",
    "housing": "living",
    "tobacco": "tobacco",
    "tobacco_use": "tobacco",
    "smoking": "tobacco",
}


def load_shac(path: str | Path | None = None) -> list[RelationTaskFixture]:
    """Load SHAC JSON/JSONL or paired BRAT files from a credentialed path."""

    root = require_credentialed_path(
        path,
        dataset=SHAC,
        authority=SHAC_AUTHORITY,
        env_var=SHAC_PATH_ENV,
    )
    fixtures: list[RelationTaskFixture] = []
    for source in _files(root, _JSON_SUFFIXES):
        for row_number, row in enumerate(
            load_json_rows(source, dataset=SHAC, authority=SHAC_AUTHORITY),
            start=1,
        ):
            fixtures.append(
                _fixture_from_row(row, source=source, root=root, row_number=row_number)
            )
    for annotation_path in _paired_annotation_files(root):
        text_path = annotation_path.with_suffix(".txt")
        if not text_path.exists():
            raise ValueError(
                f"SHAC BRAT annotation {annotation_path.name} requires a paired .txt file"
            )
        fixtures.append(_fixture_from_brat(text_path, annotation_path, root=root))

    if not fixtures:
        raise DUACredentialRequired(
            f"{SHAC_AUTHORITY} credentialed {SHAC} path contains no supported "
            "fixtures; no corpus rows were loaded"
        )
    _validate_unique_fixture_ids(fixtures)
    return fixtures


load_shac_fixtures = load_shac
load_shac_relation_fixtures = load_shac


def map_shac_entity_label(label: str) -> str:
    """Map a SHAC event or argument label to a canonical span label."""

    key = _mapping_key(label)
    canonical = SHAC_ENTITY_TO_CANONICAL.get(key)
    if canonical is None:
        normalized = normalize_label(str(label).strip())
        if normalized in CANONICAL_LABELS:
            canonical = normalized
    if canonical is None:
        allowed = ", ".join(sorted(SHAC_ENTITY_TO_CANONICAL))
        raise ValueError(
            f"unknown SHAC entity label {label!r}; expected one of: {allowed}"
        )
    return canonical


def map_shac_relation_type(relation_type: str) -> str:
    """Map a SHAC role to a stable SDOH relation type."""

    key = _mapping_key(relation_type)
    if key in SHAC_RELATION_TO_CANONICAL:
        return SHAC_RELATION_TO_CANONICAL[key]
    determinant = _DETERMINANT_ALIASES.get(key)
    if determinant is not None:
        return f"SDOH_{determinant.upper()}"
    if key.startswith("has_"):
        return key.upper()
    if not key:
        raise ValueError("SHAC relation type is required")
    return f"SDOH_{key.upper()}"


def shac_suite_metadata() -> dict[str, Any]:
    """Return row-free metadata for the SHAC SDOH relation view."""

    return {
        "access": (
            f"credentialed local path only; pass path=... or set {SHAC_PATH_ENV}"
        ),
        "dataset": SHAC,
        "determinants": list(SHAC_DETERMINANTS),
        "dua": SHAC_DUA,
        "eval_only": True,
        "entity_label_mapping": dict(sorted(SHAC_ENTITY_TO_CANONICAL.items())),
        "license": license_for(SHAC).to_dict(),
        "network_fetch": False,
        "redistribution": "never; read-only from user-supplied credentialed path",
        "relation_type_mapping": dict(sorted(SHAC_RELATION_TO_CANONICAL.items())),
        "suite": SHAC,
        "task": "relation",
        "task_view": SHAC_TASK,
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
    entity_rows = _entity_rows(
        row.get("entities") or row.get("spans") or row.get("annotations")
    )
    event_rows = row.get("events") or []
    if not isinstance(event_rows, list):
        raise ValueError("SHAC events must be a list")
    entity_rows = _materialize_event_spans(entity_rows, event_rows)
    entities = _entities_from_rows(entity_rows, text=text, source_name=source.name)
    relations = _relations_from_rows(
        row.get("relations") or row.get("gold_relations") or [],
        entities=entities,
        text=text,
        fixture_id=fixture_id(SHAC, source, root, record_id),
    )
    relations.extend(
        _event_relations(
            event_rows,
            entities=entities,
            text=text,
            fixture_id=fixture_id(SHAC, source, root, record_id),
        )
    )
    return RelationTaskFixture(
        fixture_id=fixture_id(SHAC, source, root, record_id),
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
    event_rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        annotation_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line.startswith("T"):
            columns = line.split("\t", 2)
            if len(columns) != 3:
                raise ValueError(f"malformed SHAC BRAT entity at line {line_number}")
            label, (start, end) = _brat_offsets(columns[1], text=text)
            entities[columns[0]] = _span_from_mapping(
                {
                    "start": start,
                    "end": end,
                    "label": label,
                    "text": columns[2],
                },
                text=text,
                source_name=annotation_path.name,
                fallback_label=label,
            )
        elif line.startswith("R"):
            columns = line.split("\t", 1)
            if len(columns) != 2:
                raise ValueError(f"malformed SHAC BRAT relation at line {line_number}")
            values = columns[1].split()
            if len(values) < 3:
                raise ValueError(f"malformed SHAC BRAT relation at line {line_number}")
            relation_rows.append(
                {
                    "id": columns[0],
                    "type": values[0],
                    "head": values[1].split(":", 1)[-1],
                    "tail": values[2].split(":", 1)[-1],
                    "scope": "document",
                }
            )
        elif line.startswith("E"):
            event_rows.append(_brat_event_row(line, line_number=line_number))

    relations = _relations_from_rows(
        relation_rows,
        entities=entities,
        text=text,
        fixture_id=fixture_id(SHAC, text_path, root, text_path.stem),
    )
    relations.extend(
        _event_relations(
            event_rows,
            entities=entities,
            text=text,
            fixture_id=fixture_id(SHAC, text_path, root, text_path.stem),
        )
    )
    return RelationTaskFixture(
        fixture_id=fixture_id(SHAC, text_path, root, text_path.stem),
        text=text,
        entities=entities,
        gold_relations=tuple(relations),
        metadata=_metadata(text_path, root),
    )


def _relations_from_rows(
    rows: Any,
    *,
    entities: dict[str, EvalSpan],
    text: str,
    fixture_id: str,
) -> list[EvalRelation]:
    if not isinstance(rows, list):
        raise ValueError("SHAC relations must be a list")
    relations: list[EvalRelation] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("SHAC relation rows must be objects")
        relation_id = str(row.get("id") or row.get("relation_id") or f"R{index}")
        head = _resolve_entity(
            row.get("head") or row.get("source") or row.get("arg1"), entities, text
        )
        tail = _resolve_entity(
            row.get("tail") or row.get("target") or row.get("arg2"), entities, text
        )
        relation_type = map_shac_relation_type(
            str(row.get("type") or row.get("relation_type") or row.get("role") or "")
        )
        relations.append(
            EvalRelation(
                relation_type=relation_type,
                head=head,
                tail=tail,
                scope=str(row.get("scope") or "document"),
                relation_id=relation_id,
                fixture_id=fixture_id,
                metadata={
                    "sdoh_category": _determinant_category(
                        str(row.get("category") or row.get("event_type") or "")
                    )
                    if row.get("category") or row.get("event_type")
                    else "",
                    "source_relation_type": str(
                        row.get("type")
                        or row.get("relation_type")
                        or row.get("role")
                        or ""
                    ),
                },
            )
        )
    return relations


def _event_relations(
    events: list[Any],
    *,
    entities: dict[str, EvalSpan],
    text: str,
    fixture_id: str,
) -> list[EvalRelation]:
    relations: list[EvalRelation] = []
    for event_index, raw_event in enumerate(events, start=1):
        if not isinstance(raw_event, Mapping):
            raise ValueError("SHAC event rows must be objects")
        event_id = str(
            raw_event.get("id") or raw_event.get("event_id") or f"E{event_index}"
        )
        category = _determinant_category(
            str(
                raw_event.get("type")
                or raw_event.get("event_type")
                or raw_event.get("category")
                or ""
            )
        )
        trigger_value = raw_event.get("trigger") or raw_event.get("head")
        if isinstance(trigger_value, Mapping):
            trigger_id = str(trigger_value.get("id") or f"{event_id}-trigger")
            entities.setdefault(
                trigger_id,
                _span_from_mapping(
                    trigger_value,
                    text=text,
                    source_name="SHAC event trigger",
                    fallback_label=category,
                ),
            )
        else:
            trigger_id = str(trigger_value or raw_event.get("trigger_id") or "")
        if not trigger_id:
            raise ValueError(f"SHAC event {event_id!r} requires a trigger")
        head = _resolve_entity(trigger_id, entities, text)
        arguments = raw_event.get("arguments") or raw_event.get("args") or []
        if not isinstance(arguments, list):
            raise ValueError(f"SHAC event {event_id!r} arguments must be a list")
        for argument_index, raw_argument in enumerate(arguments, start=1):
            if not isinstance(raw_argument, Mapping):
                raise ValueError("SHAC event arguments must be objects")
            argument_value = (
                raw_argument.get("target")
                or raw_argument.get("tail")
                or raw_argument.get("argument")
                or raw_argument.get("id")
            )
            if isinstance(argument_value, Mapping):
                argument_id = str(
                    argument_value.get("id") or f"{event_id}-argument-{argument_index}"
                )
                entities.setdefault(
                    argument_id,
                    _span_from_mapping(
                        argument_value,
                        text=text,
                        source_name="SHAC event argument",
                        fallback_label=str(raw_argument.get("role") or "type"),
                    ),
                )
            else:
                argument_id = str(argument_value or "")
            if not argument_id:
                raise ValueError(
                    f"SHAC event {event_id!r} has an argument without a target"
                )
            tail = _resolve_entity(argument_id, entities, text)
            source_role = str(
                raw_argument.get("role")
                or raw_argument.get("type")
                or raw_argument.get("label")
                or ""
            )
            relations.append(
                EvalRelation(
                    relation_type=map_shac_relation_type(source_role),
                    head=head,
                    tail=tail,
                    scope="document",
                    relation_id=f"{event_id}-{argument_index}",
                    fixture_id=fixture_id,
                    metadata={
                        "event_type": str(raw_event.get("type") or category),
                        "sdoh_category": category,
                        "source_relation_type": source_role,
                    },
                )
            )
    return relations


def _entities_from_rows(
    rows: list[Mapping[str, Any]],
    *,
    text: str,
    source_name: str,
) -> dict[str, EvalSpan]:
    entities: dict[str, EvalSpan] = {}
    for index, row in enumerate(rows, start=1):
        entity_id = str(row.get("id") or row.get("entity_id") or f"T{index}")
        if entity_id in entities:
            raise ValueError(f"duplicate SHAC entity id: {entity_id}")
        entities[entity_id] = _span_from_mapping(
            row,
            text=text,
            source_name=source_name,
            fallback_label=str(row.get("category") or row.get("type") or ""),
        )
    return entities


def _materialize_event_spans(
    rows: list[Mapping[str, Any]], events: list[Any]
) -> list[Mapping[str, Any]]:
    materialized = list(rows)
    known = {str(row.get("id")) for row in materialized if row.get("id") is not None}
    for event_index, event in enumerate(events, start=1):
        if not isinstance(event, Mapping):
            continue
        trigger = event.get("trigger")
        if isinstance(trigger, Mapping) and trigger.get("id") not in known:
            materialized.append(
                {"id": trigger.get("id") or f"E{event_index}-trigger", **dict(trigger)}
            )
            known.add(str(materialized[-1]["id"]))
        arguments = event.get("arguments") or event.get("args") or []
        if isinstance(arguments, list):
            for argument_index, argument in enumerate(arguments, start=1):
                if not isinstance(argument, Mapping):
                    continue
                target = argument.get("target") or argument.get("argument")
                if isinstance(target, Mapping) and target.get("id") not in known:
                    materialized.append(
                        {
                            "id": target.get("id")
                            or f"E{event_index}-argument-{argument_index}",
                            **dict(target),
                            "label": target.get("label")
                            or argument.get("role")
                            or "type",
                        }
                    )
                    known.add(str(materialized[-1]["id"]))
    return materialized


def _span_from_mapping(
    item: Mapping[str, Any],
    *,
    text: str,
    source_name: str,
    fallback_label: str,
) -> EvalSpan:
    start, end = _mapping_offsets(item)
    label = str(
        item.get("label")
        or item.get("type")
        or item.get("entity_type")
        or item.get("category")
        or fallback_label
        or ""
    )
    if start < 0 or end <= start or end > len(text):
        raise ValueError(f"invalid SHAC span offsets {start}:{end} in {source_name}")
    span_text = text[start:end]
    supplied_text = str(item.get("text") or item.get("surface") or "")
    if supplied_text and supplied_text != span_text:
        raise ValueError(f"SHAC span text mismatch in {source_name}")
    source_label = label.strip()
    return EvalSpan(
        start=start,
        end=end,
        label=map_shac_entity_label(source_label),
        text=span_text,
        metadata={"source_label": source_label},
    )


def _resolve_entity(
    value: Any, entities: Mapping[str, EvalSpan], text: str
) -> EvalSpan:
    if isinstance(value, Mapping):
        entity_id = str(value.get("id") or "")
    else:
        entity_id = str(value or "")
    try:
        return entities[entity_id]
    except KeyError as exc:
        raise ValueError(
            f"SHAC relation references unknown entity {entity_id!r}"
        ) from exc


def _brat_event_row(line: str, *, line_number: int) -> Mapping[str, Any]:
    columns = line.split("\t", 1)
    if len(columns) != 2:
        raise ValueError(f"malformed SHAC BRAT event at line {line_number}")
    values = columns[1].split()
    if len(values) < 2 or ":" not in values[0]:
        raise ValueError(f"malformed SHAC BRAT event at line {line_number}")
    event_type, trigger_id = values[0].split(":", 1)
    arguments = [
        {"role": role, "target": target}
        for value in values[1:]
        for role, separator, target in (value.partition(":"),)
        if separator
    ]
    if len(arguments) != len(values) - 1:
        raise ValueError(f"malformed SHAC BRAT event at line {line_number}")
    return {
        "id": columns[0],
        "type": event_type,
        "trigger": trigger_id,
        "arguments": arguments,
    }


def _mapping_offsets(item: Mapping[str, Any]) -> tuple[int, int]:
    if item.get("start") is not None and item.get("end") is not None:
        return _integer(item["start"]), _integer(item["end"])
    if item.get("offset") is not None and item.get("length") is not None:
        start = _integer(item["offset"])
        return start, start + _integer(item["length"])
    raise ValueError("SHAC spans require start/end or offset/length")


def _brat_offsets(specification: str, *, text: str) -> tuple[str, tuple[int, int]]:
    values = specification.split(maxsplit=1)
    if len(values) != 2:
        raise ValueError("malformed SHAC BRAT span offsets")
    label, offset_spec = values
    spans = []
    for segment in offset_spec.split(";"):
        offsets = segment.split()
        if len(offsets) != 2:
            raise ValueError("malformed SHAC discontinuous BRAT span")
        spans.append((_integer(offsets[0]), _integer(offsets[1])))
    start = min(item[0] for item in spans)
    end = max(item[1] for item in spans)
    if start < 0 or end > len(text):
        raise ValueError("SHAC BRAT span is outside document text")
    return label, (start, end)


def _determinant_category(value: str) -> str:
    key = _mapping_key(value)
    for alias, category in _DETERMINANT_ALIASES.items():
        if key == alias or key.startswith(f"{alias}_"):
            return category
    raise ValueError(
        f"unknown SHAC determinant {value!r}; expected one of: "
        + ", ".join(SHAC_DETERMINANTS)
    )


def _metadata(source: Path, root: Path) -> dict[str, Any]:
    return {
        "cache_corpus_rows": False,
        "dataset": SHAC,
        "determinants": list(SHAC_DETERMINANTS),
        "dua": SHAC_DUA,
        "eval_only": True,
        "license": license_for(SHAC).to_dict(),
        "network_fetch": False,
        "redistribution": "never; read-only from user-supplied credentialed path",
        "source_path_hash": source_path_hash(source, root),
        "suite": SHAC,
        "task": "relation",
        "task_view": SHAC_TASK,
    }


def _row_text(row: Mapping[str, Any]) -> str:
    for key in ("text", "note", "document"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("SHAC rows require non-empty text")


def _record_id(row: Mapping[str, Any], *, fallback: str) -> str:
    for key in ("id", "record_id", "document_id", "doc_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return fallback


def _entity_rows(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [
            {"id": key, **dict(item)}
            for key, item in value.items()
            if isinstance(item, Mapping)
        ]
    if not isinstance(value, list):
        raise ValueError("SHAC entities must be a list or mapping")
    return [item for item in value if isinstance(item, Mapping)]


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
        raise ValueError("SHAC span offsets must be integers")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("SHAC span offsets must be integers") from exc


def _validate_unique_fixture_ids(fixtures: list[RelationTaskFixture]) -> None:
    ids = [fixture.fixture_id for fixture in fixtures]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {SHAC} fixture ids")


__all__ = [
    "SHAC",
    "SHAC_AUTHORITY",
    "SHAC_DETERMINANTS",
    "SHAC_DUA",
    "SHAC_ENTITY_TO_CANONICAL",
    "SHAC_PATH_ENV",
    "SHAC_RELATION_TO_CANONICAL",
    "SHAC_TASK",
    "RelationTaskFixture",
    "load_shac",
    "load_shac_fixtures",
    "load_shac_relation_fixtures",
    "map_shac_entity_label",
    "map_shac_relation_type",
    "shac_suite_metadata",
]
