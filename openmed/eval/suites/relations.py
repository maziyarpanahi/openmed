"""Synthetic relation extraction suite loader."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from openmed.eval.metrics import EvalSpan, normalize_eval_span
from openmed.eval.relation_metrics import (
    EvalRelation,
    compute_relation_metrics,
    normalize_eval_relations,
)

RELATIONS = "relations"
RELATION_GOLD_SCHEMA_VERSION = 1
TRAP_ASSERTION = "assertion"
TRAP_TEMPORAL = "temporal"
RELATION_TRAP_KINDS: tuple[str, ...] = (TRAP_ASSERTION, TRAP_TEMPORAL)

DEFAULT_RELATION_GOLD_PATH = (
    Path(__file__).resolve().parents[1] / "golden" / "fixtures" / "relation_gold.jsonl"
)


@dataclass(frozen=True)
class RelationTrap:
    """One zero-tolerance trap attached to relation fixture rows."""

    trap_id: str
    kind: str
    relation_ids: tuple[str, ...]
    description: str = ""
    zero_tolerance: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        known_relation_ids: set[str],
    ) -> "RelationTrap":
        """Build and validate a trap record from fixture metadata."""
        trap_id = str(data.get("id") or data.get("trap_id") or "")
        if not trap_id:
            raise ValueError("relation trap id is required")

        kind = str(data.get("kind") or data.get("type") or "").strip().lower()
        if kind not in RELATION_TRAP_KINDS:
            allowed = ", ".join(RELATION_TRAP_KINDS)
            raise ValueError(f"relation trap kind {kind!r} must be one of: {allowed}")

        raw_relation_ids = data.get("relation_ids") or data.get("relations") or []
        if not isinstance(raw_relation_ids, list) or not raw_relation_ids:
            raise ValueError("relation trap relation_ids must be a non-empty list")
        relation_ids = tuple(str(value) for value in raw_relation_ids)
        unknown_ids = sorted(set(relation_ids) - known_relation_ids)
        if unknown_ids:
            raise ValueError(
                "relation trap references unknown relation ids: "
                + ", ".join(unknown_ids)
            )

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {"value": metadata}
        return cls(
            trap_id=trap_id,
            kind=kind,
            relation_ids=relation_ids,
            description=str(data.get("description") or ""),
            zero_tolerance=bool(data.get("zero_tolerance", True)),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready trap record."""
        return {
            "id": self.trap_id,
            "kind": self.kind,
            "relation_ids": list(self.relation_ids),
            "description": self.description,
            "zero_tolerance": self.zero_tolerance,
            "metadata": _plain_mapping(self.metadata),
        }


@dataclass(frozen=True)
class RelationFixture:
    """One synthetic relation-gold document."""

    fixture_id: str
    text: str
    entities: Mapping[str, EvalSpan]
    gold_relations: tuple[EvalRelation, ...]
    language: str = "en"
    traps: tuple[RelationTrap, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RelationFixture":
        """Build and validate a relation fixture from a JSON-ready mapping."""
        if not isinstance(data, Mapping):
            raise ValueError("relation fixture must be a mapping")

        fixture_id = str(data.get("id") or data.get("fixture_id") or "")
        if not fixture_id:
            raise ValueError("relation fixture id is required")

        text = str(data.get("text") or "")
        if not text:
            raise ValueError("relation fixture text is required")

        language = str(data.get("language") or data.get("lang") or "en")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("relation fixture metadata must be a mapping")
        metadata = dict(metadata)
        if metadata.get("synthetic") is not True:
            raise ValueError("relation fixture metadata.synthetic must be true")

        schema_version = data.get("schema_version", metadata.get("schema_version"))
        if schema_version != RELATION_GOLD_SCHEMA_VERSION:
            raise ValueError(
                "relation fixture schema_version must be "
                f"{RELATION_GOLD_SCHEMA_VERSION}"
            )

        entities = _entities_from_rows(
            data.get("entities") or data.get("gold_spans") or [],
            language=language,
            text=text,
        )
        relations = tuple(
            _relations_from_rows(
                data.get("relations") or data.get("gold_relations") or [],
                entities=entities,
                fixture_id=fixture_id,
                language=language,
                text=text,
            )
        )
        relation_ids = _relation_ids(relations)
        traps = tuple(
            RelationTrap.from_mapping(row, known_relation_ids=relation_ids)
            for row in _trap_rows(data, metadata)
        )
        return cls(
            fixture_id=fixture_id,
            text=text,
            entities=entities,
            gold_relations=relations,
            language=language,
            traps=traps,
            metadata=metadata,
        )

    @property
    def gold_spans(self) -> tuple[EvalSpan, ...]:
        """Return relation argument spans in deterministic entity-id order."""
        return tuple(span for _, span in sorted(self.entities.items()))

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready fixture payload."""
        return {
            "id": self.fixture_id,
            "schema_version": RELATION_GOLD_SCHEMA_VERSION,
            "language": self.language,
            "text": self.text,
            "entities": [
                {"id": span_id, **_span_to_dict(span)}
                for span_id, span in sorted(self.entities.items())
            ],
            "relations": [relation.to_dict() for relation in self.gold_relations],
            "traps": [trap.to_dict() for trap in self.traps],
            "metadata": _plain_mapping(self.metadata),
        }


def load_relation_fixtures(path: str | Path | None = None) -> list[RelationFixture]:
    """Load and validate synthetic relation-gold fixtures."""
    fixture_path = Path(path) if path is not None else DEFAULT_RELATION_GOLD_PATH
    rows = _load_rows(fixture_path)
    fixtures = [RelationFixture.from_mapping(row) for row in rows]
    _validate_unique_fixture_ids(fixtures)
    return fixtures


def relation_suite_metadata() -> dict[str, Any]:
    """Return metadata for the synthetic relation gold suite."""
    return {
        "suite": RELATIONS,
        "schema_version": RELATION_GOLD_SCHEMA_VERSION,
        "source": "synthetic committed fixtures",
        "redistribution": "safe; no DUA or production data",
        "trap_kinds": list(RELATION_TRAP_KINDS),
    }


def relation_trap_summary(fixtures: list[RelationFixture]) -> dict[str, Any]:
    """Summarize zero-tolerance trap metadata for gate integration."""
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in RELATION_TRAP_KINDS:
        traps = [
            trap for fixture in fixtures for trap in fixture.traps if trap.kind == kind
        ]
        by_kind[kind] = {
            "count": len(traps),
            "zero_tolerance": all(trap.zero_tolerance for trap in traps),
            "relation_ids": sorted(
                {relation_id for trap in traps for relation_id in trap.relation_ids}
            ),
        }
    total = sum(value["count"] for value in by_kind.values())
    return {"total": total, "by_kind": by_kind}


def score_relation_fixtures(
    fixtures: list[RelationFixture],
    predictions_by_fixture: Mapping[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Score predicted relations against relation-gold fixtures."""
    predictions_by_fixture = predictions_by_fixture or {}
    gold: list[EvalRelation] = []
    predicted: list[EvalRelation] = []
    for fixture in fixtures:
        gold.extend(fixture.gold_relations)
        predicted.extend(
            normalize_eval_relations(
                predictions_by_fixture.get(fixture.fixture_id, []),
                entity_spans=fixture.entities,
                fixture_id=fixture.fixture_id,
                default_language=fixture.language,
                source_text=fixture.text,
            )
        )

    return {
        "suite": RELATIONS,
        "fixture_count": len(fixtures),
        "relation_count": len(gold),
        "metrics": compute_relation_metrics(gold, predicted),
        "metadata": {
            **relation_suite_metadata(),
            "fixture_ids": [fixture.fixture_id for fixture in fixtures],
            "traps": relation_trap_summary(fixtures),
        },
    }


def _load_rows(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        if raw.get("schema_version") not in (None, RELATION_GOLD_SCHEMA_VERSION):
            raise ValueError("relation fixture file has unsupported schema_version")
        rows = raw.get("fixtures")
    else:
        rows = raw
    if not isinstance(rows, list):
        raise ValueError("relation fixture JSON must be a list or fixtures mapping")
    return rows


def _entities_from_rows(
    raw_entities: Any,
    *,
    language: str,
    text: str,
) -> dict[str, EvalSpan]:
    if not isinstance(raw_entities, list) or not raw_entities:
        raise ValueError("relation fixture entities must be a non-empty list")

    entities: dict[str, EvalSpan] = {}
    for row in raw_entities:
        if not isinstance(row, Mapping):
            raise ValueError("relation fixture entity must be a mapping")
        span_id = str(row.get("id") or row.get("span_id") or row.get("entity_id") or "")
        if not span_id:
            raise ValueError("relation fixture entity id is required")
        if span_id in entities:
            raise ValueError(f"duplicate relation fixture entity id: {span_id}")
        span = normalize_eval_span(row, default_language=language, source_text=text)
        _validate_span_offsets(span, text)
        entities[span_id] = span
    return entities


def _relations_from_rows(
    raw_relations: Any,
    *,
    entities: Mapping[str, EvalSpan],
    fixture_id: str,
    language: str,
    text: str,
) -> list[EvalRelation]:
    if not isinstance(raw_relations, list) or not raw_relations:
        raise ValueError("relation fixture relations must be a non-empty list")

    relations: list[EvalRelation] = []
    seen_ids: set[str] = set()
    for row in raw_relations:
        if not isinstance(row, Mapping):
            raise ValueError("relation fixture relation must be a mapping")
        relation_id = str(row.get("id") or row.get("relation_id") or "")
        if not relation_id:
            raise ValueError("relation fixture relation id is required")
        if relation_id in seen_ids:
            raise ValueError(f"duplicate relation fixture relation id: {relation_id}")
        seen_ids.add(relation_id)
        relations.extend(
            normalize_eval_relations(
                [row],
                entity_spans=entities,
                fixture_id=fixture_id,
                default_language=language,
                source_text=text,
            )
        )
    return relations


def _relation_ids(relations: tuple[EvalRelation, ...]) -> set[str]:
    return {relation.relation_id for relation in relations}


def _trap_rows(
    data: Mapping[str, Any], metadata: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    traps = data.get("traps") or metadata.get("traps") or []
    if not isinstance(traps, list):
        raise ValueError("relation fixture traps must be a list")
    for trap in traps:
        if not isinstance(trap, Mapping):
            raise ValueError("relation fixture trap must be a mapping")
    return traps


def _validate_unique_fixture_ids(fixtures: list[RelationFixture]) -> None:
    counts: defaultdict[str, int] = defaultdict(int)
    for fixture in fixtures:
        counts[fixture.fixture_id] += 1
    duplicates = sorted(fixture_id for fixture_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError("duplicate relation fixture ids: " + ", ".join(duplicates))


def _validate_span_offsets(span: EvalSpan, text: str) -> None:
    if span.start < 0 or span.end < span.start or span.end > len(text):
        raise ValueError(
            f"invalid relation entity offsets {span.start}:{span.end} "
            f"for text length {len(text)}"
        )
    if span.text and text[span.start : span.end] != span.text:
        raise ValueError(
            f"relation entity text mismatch at offsets {span.start}:{span.end}"
        )


def _span_to_dict(span: EvalSpan) -> dict[str, Any]:
    return {
        "start": span.start,
        "end": span.end,
        "label": span.label,
        "text": span.text,
        "language": span.language,
        "metadata": _plain_mapping(span.metadata),
    }


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    plain: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            plain[str(key)] = _plain_mapping(item)
        elif isinstance(item, (list, tuple)):
            plain[str(key)] = [
                _plain_mapping(entry) if isinstance(entry, Mapping) else entry
                for entry in item
            ]
        else:
            plain[str(key)] = item
    return plain
