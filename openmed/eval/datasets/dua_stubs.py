"""Eval-only stubs for corpora that require a local data-use agreement."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from openmed.eval.metrics import EvalSpan, normalize_radiology_entity
from openmed.eval.relation_metrics import EvalRelation, normalize_eval_relations

from .public import DatasetLoadResult

RADGRAPH = "radgraph"
RADGRAPH_STYLE_SCHEMA_VERSION = 1
DEFAULT_SYNTHETIC_RADIOLOGY_PATH = (
    Path(__file__).parents[1]
    / "golden"
    / "fixtures"
    / "radiology_entity_relations.jsonl"
)

DUA_GATED_CORPORA: tuple[str, ...] = (
    "biored",
    "i2b2",
    "n2c2",
    "n2c2-2018",
    "n2c2-2022",
    "shac",
    "thyme",
    "mednli",
    "made",
    "mimic",
    RADGRAPH,
)

DUA_PATH_REMEDIATION: Mapping[str, str] = {
    "biored": "pass path=... or set OPENMED_BIORED_PATH",
    "made": "pass path=... or set OPENMED_MADE_PATH",
    "n2c2-2018": "pass path=... or set OPENMED_N2C2_2018_PATH",
    "n2c2-2022": "pass path=... or set OPENMED_N2C2_2022_PATH",
}


class DUACredentialRequired(PermissionError):
    """Raised when a gated corpus is requested without a credentialed path."""


@dataclass(frozen=True)
class RadiologyEntityRelationFixture:
    """One RadGraph-style text fixture held in memory for evaluation only."""

    fixture_id: str
    text: str
    entities: Mapping[str, EvalSpan]
    gold_relations: tuple[EvalRelation, ...]
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def gold_spans(self) -> tuple[EvalSpan, ...]:
        """Return entities in deterministic identifier order."""
        return tuple(span for _, span in sorted(self.entities.items()))

    @property
    def relations(self) -> tuple[EvalRelation, ...]:
        """Return the relation-engine-compatible gold relation sequence."""
        return self.gold_relations

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        fixture_id: str | None = None,
        require_synthetic: bool = False,
    ) -> "RadiologyEntityRelationFixture":
        """Build a validated fixture from canonical or RadGraph-style JSON."""
        if not isinstance(data, Mapping):
            raise ValueError("RadGraph-style fixture must be a mapping")
        resolved_id = str(
            fixture_id or data.get("id") or data.get("fixture_id") or ""
        ).strip()
        if not resolved_id:
            raise ValueError("RadGraph-style fixture id is required")
        text = str(data.get("text") or "")
        if not text:
            raise ValueError("RadGraph-style fixture text is required")
        language = str(data.get("language") or data.get("lang") or "en")
        metadata_value = data.get("metadata") or {}
        if not isinstance(metadata_value, Mapping):
            raise ValueError("RadGraph-style fixture metadata must be a mapping")
        metadata = dict(metadata_value)
        if require_synthetic:
            schema_version = data.get(
                "schema_version",
                metadata.get("schema_version", RADGRAPH_STYLE_SCHEMA_VERSION),
            )
            if schema_version != RADGRAPH_STYLE_SCHEMA_VERSION:
                raise ValueError(
                    "synthetic radiology fixture schema_version must be "
                    f"{RADGRAPH_STYLE_SCHEMA_VERSION}"
                )
            _validate_synthetic_radiology_metadata(metadata)

        entities, embedded_relations = _radgraph_entities(
            data.get("entities"),
            text=text,
            language=language,
        )
        relation_rows = _radgraph_relation_rows(
            data.get("relations") or data.get("gold_relations") or [],
            embedded_relations=embedded_relations,
        )
        relations = tuple(
            normalize_eval_relations(
                relation_rows,
                entity_spans=entities,
                fixture_id=resolved_id,
                default_language=language,
                source_text=text,
            )
        )
        return cls(
            fixture_id=resolved_id,
            text=text,
            entities=entities,
            gold_relations=relations,
            language=language,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready fixture representation."""
        return {
            "id": self.fixture_id,
            "schema_version": RADGRAPH_STYLE_SCHEMA_VERSION,
            "language": self.language,
            "text": self.text,
            "entities": [
                {
                    "id": entity_id,
                    "start": entity.start,
                    "end": entity.end,
                    "label": entity.label,
                    "text": entity.text,
                    "uncertainty": entity.metadata.get("uncertainty"),
                }
                for entity_id, entity in sorted(self.entities.items())
            ],
            "relations": [relation.to_dict() for relation in self.gold_relations],
            "metadata": dict(self.metadata),
        }


RadGraphFixture = RadiologyEntityRelationFixture


@dataclass(frozen=True)
class DUACorpusStub:
    name: str
    eval_only: bool = True

    def load(self, credentialed_path: str | Path | None = None) -> DatasetLoadResult:
        if credentialed_path is None:
            remediation = DUA_PATH_REMEDIATION.get(
                self.name,
                "supply an approved credentialed local path",
            )
            raise DUACredentialRequired(
                f"{self.name} requires a credentialed local path and cannot be "
                f"bundled; {remediation}. No corpus rows were loaded."
            )
        path = Path(credentialed_path)
        if not path.exists():
            raise DUACredentialRequired(
                f"{self.name} credentialed path does not exist: {path}"
            )
        return DatasetLoadResult(
            dataset=self.name,
            records=(),
            skipped=True,
            reason="eval-only gated corpus stub; local loader is intentionally not bundled",
        )


def dua_stub_for(name: str) -> DUACorpusStub:
    key = name.lower()
    if key not in DUA_GATED_CORPORA:
        raise ValueError(f"unknown gated corpus: {name}")
    return DUACorpusStub(key)


def load_dua_corpus(
    name: str, credentialed_path: str | Path | None = None
) -> DatasetLoadResult:
    return dua_stub_for(name).load(credentialed_path)


def all_dua_stubs() -> Mapping[str, DUACorpusStub]:
    return {name: DUACorpusStub(name) for name in DUA_GATED_CORPORA}


def load_radgraph_fixtures(
    credentialed_path: str | Path | None = None,
) -> tuple[RadiologyEntityRelationFixture, ...]:
    """Load a user-supplied RadGraph-style corpus without caching corpus rows.

    The loader performs read-only parsing from an explicit local path. It never
    downloads, copies, caches, logs, or writes source report text.

    Args:
        credentialed_path: Authorized local JSON/JSONL file or directory.

    Returns:
        In-memory radiology entity-and-relation fixtures.

    Raises:
        DUACredentialRequired: If no existing credentialed path is supplied.
    """
    if credentialed_path is None:
        raise DUACredentialRequired(
            "radgraph requires a credentialed local path and cannot be bundled"
        )
    path = Path(credentialed_path)
    if not path.exists():
        raise DUACredentialRequired(
            f"radgraph credentialed path does not exist: {path}"
        )
    return _load_radgraph_paths(_radgraph_source_files(path), require_synthetic=False)


def load_synthetic_radiology_fixtures(
    path: str | Path | None = None,
) -> tuple[RadiologyEntityRelationFixture, ...]:
    """Load committed synthetic-only radiology fixtures for offline CI."""
    fixture_path = Path(path) if path is not None else DEFAULT_SYNTHETIC_RADIOLOGY_PATH
    return _load_radgraph_paths((fixture_path,), require_synthetic=True)


def _radgraph_source_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    files = tuple(
        sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file()
            and candidate.suffix.casefold() in {".json", ".jsonl"}
        )
    )
    if not files:
        raise ValueError("credentialed RadGraph path contains no JSON or JSONL files")
    return files


def _load_radgraph_paths(
    paths: tuple[Path, ...],
    *,
    require_synthetic: bool,
) -> tuple[RadiologyEntityRelationFixture, ...]:
    fixtures: list[RadiologyEntityRelationFixture] = []
    for path in paths:
        for fixture_id, row in _radgraph_rows(path):
            fixtures.append(
                RadiologyEntityRelationFixture.from_mapping(
                    row,
                    fixture_id=fixture_id,
                    require_synthetic=require_synthetic,
                )
            )
    fixture_ids = [fixture.fixture_id for fixture in fixtures]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("duplicate RadGraph-style fixture ids")
    if not fixtures:
        raise ValueError("RadGraph-style corpus contains no fixtures")
    return tuple(fixtures)


def _radgraph_rows(path: Path) -> list[tuple[str | None, Mapping[str, Any]]]:
    if path.suffix.casefold() == ".jsonl":
        rows: list[tuple[str | None, Mapping[str, Any]]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"RadGraph JSONL row {line_number} must be a mapping")
            rows.append((None, row))
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [(None, _require_mapping(row)) for row in payload]
    if not isinstance(payload, Mapping):
        raise ValueError("RadGraph JSON must be a mapping or list")
    fixtures = payload.get("fixtures")
    if isinstance(fixtures, list):
        return [(None, _require_mapping(row)) for row in fixtures]
    if "text" in payload and "entities" in payload:
        return [(None, payload)]
    return [
        (str(fixture_id), _require_mapping(row)) for fixture_id, row in payload.items()
    ]


def _require_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("RadGraph fixture must be a mapping")
    return value


def _radgraph_entities(
    raw_entities: Any,
    *,
    text: str,
    language: str,
) -> tuple[dict[str, EvalSpan], list[dict[str, Any]]]:
    if isinstance(raw_entities, Mapping):
        rows = [
            (str(entity_id), _require_mapping(row))
            for entity_id, row in raw_entities.items()
        ]
    elif isinstance(raw_entities, list):
        rows = [
            (
                str(row.get("id") or row.get("entity_id") or row.get("span_id") or ""),
                _require_mapping(row),
            )
            for row in raw_entities
        ]
    else:
        raise ValueError("RadGraph-style entities must be a mapping or list")
    if not rows:
        raise ValueError("RadGraph-style fixture must include entities")

    token_offsets = tuple(match.span() for match in re.finditer(r"\w+|[^\w\s]", text))
    entities: dict[str, EvalSpan] = {}
    embedded_relations: list[dict[str, Any]] = []
    for entity_id, row in rows:
        if not entity_id:
            raise ValueError("RadGraph-style entity id is required")
        if entity_id in entities:
            raise ValueError(f"duplicate RadGraph-style entity id: {entity_id}")
        entity_row = dict(row)
        if "start" not in entity_row or "end" not in entity_row:
            start_index = _integer(entity_row.get("start_ix"), "start_ix")
            end_index = _integer(entity_row.get("end_ix"), "end_ix")
            if not 0 <= start_index <= end_index < len(token_offsets):
                raise ValueError(
                    f"invalid RadGraph token offsets {start_index}:{end_index}"
                )
            entity_row["start"] = token_offsets[start_index][0]
            entity_row["end"] = token_offsets[end_index][1]
        start = _integer(entity_row.get("start"), "start")
        end = _integer(entity_row.get("end"), "end")
        entity_row["text"] = text[start:end]
        entities[entity_id] = normalize_radiology_entity(
            entity_row,
            default_language=language,
            source_text=text,
        )
        for relation_index, relation in enumerate(row.get("relations") or [], start=1):
            if isinstance(relation, Mapping):
                relation_type = relation.get("type") or relation.get("relation_type")
                target = relation.get("tail") or relation.get("target")
            elif isinstance(relation, list | tuple) and len(relation) == 2:
                relation_type, target = relation
            else:
                raise ValueError(
                    "embedded RadGraph relation must contain type and target"
                )
            embedded_relations.append(
                {
                    "id": f"{entity_id}-relation-{relation_index}",
                    "type": relation_type,
                    "head": entity_id,
                    "tail": str(target),
                }
            )
    return entities, embedded_relations


def _radgraph_relation_rows(
    raw_relations: Any,
    *,
    embedded_relations: list[dict[str, Any]],
) -> list[Mapping[str, Any]]:
    if not isinstance(raw_relations, list):
        raise ValueError("RadGraph-style relations must be a list")
    rows = [_require_mapping(row) for row in raw_relations]
    return [*rows, *embedded_relations]


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"RadGraph entity {field_name} must be an integer")
    return value


def _validate_synthetic_radiology_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("synthetic") is not True:
        raise ValueError("committed radiology fixture must be synthetic-only")
    disclaimer = str(metadata.get("medical_device_disclaimer") or "").casefold()
    normalized = disclaimer.replace("-", " ")
    if "synthetic" not in normalized or "not a medical device" not in normalized:
        raise ValueError(
            "committed radiology fixture requires a synthetic medical-device disclaimer"
        )


__all__ = [
    "DEFAULT_SYNTHETIC_RADIOLOGY_PATH",
    "DUA_GATED_CORPORA",
    "DUA_PATH_REMEDIATION",
    "DUACorpusStub",
    "DUACredentialRequired",
    "RADGRAPH",
    "RADGRAPH_STYLE_SCHEMA_VERSION",
    "RadGraphFixture",
    "RadiologyEntityRelationFixture",
    "all_dua_stubs",
    "dua_stub_for",
    "load_dua_corpus",
    "load_radgraph_fixtures",
    "load_synthetic_radiology_fixtures",
]
