"""Loader for synthetic golden de-identification fixtures."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.labels import CANONICAL_LABELS, normalize_label
from openmed.core.pii_i18n import (
    INDIC_NER_LANGUAGES,
    NATIONAL_ID_ONLY_LANGUAGES,
    SUPPORTED_LANGUAGES,
)
from openmed.eval.golden.hard_negatives import HARD_NEGATIVE_CATEGORY
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import (
    CRITICAL_FINDING_CATEGORIES,
    EvalSpan,
    critical_finding_category,
    normalize_critical_finding_category,
    normalize_eval_spans,
)

CRITICAL_FINDINGS_CATEGORY = "critical_findings"
GOLDEN_CATEGORIES: tuple[str, ...] = (
    "nested_overlapping",
    "chunk_boundary",
    "multilingual",
    "checksum_ids",
    "financial_ids",
    "india_health_ids",
    "date_arithmetic",
    "policy_profile_actions",
    HARD_NEGATIVE_CATEGORY,
    CRITICAL_FINDINGS_CATEGORY,
)

_FIXTURE_VERSION = 1
_GOLDEN_DIR = Path(__file__).resolve().parent
_FIXTURE_DIR = _GOLDEN_DIR / "fixtures"
_TOP_LEVEL_FIXTURES: tuple[Path, ...] = tuple(
    _GOLDEN_DIR / name
    for name in (
        "be.jsonl",
        "ch.jsonl",
        "es_mx.jsonl",
        "ie.jsonl",
        "financial_ids.jsonl",
    )
)
_SPECIALIZED_FIXTURE_NAMES = frozenset(
    {
        "code_mixed_hinglish.jsonl",
        "context_multilingual.jsonl",
        "dicom_sr_content.jsonl",
        "doclevel_relations.jsonl",
        "event_coref.jsonl",
        "fhir_roundtrip.jsonl",
        "code_mixed_deidentification.jsonl",
        "grounding_crosslingual.jsonl",
        "grounded_codeable_concepts.jsonl",
        "grounding_export.jsonl",
        "grounding_vocab_synthetic.jsonl",
        "india_clinical.jsonl",
        "indic_name_variants.json",
        "joint_entity_relation.jsonl",
        "relation_calibration.jsonl",
        "relation_assertion.jsonl",
        "relation_gold.jsonl",
        "relations_indic.jsonl",
        "relations_zh.jsonl",
        "surrogate_multilingual.jsonl",
        "consensus_corpus.jsonl",
        # Domain eval fixtures that are not PII de-identification gold spans and
        # must not be loaded as such by load_golden_fixtures().
        "biomarker_result.jsonl",
        "radiology_finding.jsonl",
        "radiology_report.jsonl",
        "radiology_entity_relations.jsonl",
        "hgvs_parse.jsonl",
        "measurement_trend.jsonl",
        "norm_multilingual.jsonl",
        "temporal_tlinks.jsonl",
        "tnm_stage.jsonl",
        "oncotree_map.jsonl",
    }
)

#: Committed synthetic multi-annotator consensus corpus.
_CONSENSUS_CORPUS = _FIXTURE_DIR / "consensus_corpus.jsonl"


@dataclass(frozen=True)
class GoldenFixture:
    """One validated golden fixture with expected post-action output."""

    fixture_id: str
    category: str
    language: str
    text: str
    gold_spans: tuple[EvalSpan, ...]
    expected_output: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GoldenFixture":
        """Build and validate a golden fixture from a JSON-ready mapping."""
        if not isinstance(data, Mapping):
            raise ValueError("golden fixture must be a mapping")

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("golden fixture metadata must be a mapping")
        metadata = dict(metadata)

        if metadata.get("synthetic") is not True:
            raise ValueError("golden fixture metadata.synthetic must be true")

        category = str(metadata.get("category", ""))
        if category not in GOLDEN_CATEGORIES:
            raise ValueError(f"unknown golden fixture category: {category!r}")

        expected_output = metadata.get("expected_output")
        if not isinstance(expected_output, Mapping):
            raise ValueError(
                "golden fixture metadata.expected_output must be a mapping"
            )
        if not str(expected_output.get("method", "")):
            raise ValueError("golden fixture expected_output.method is required")
        if not isinstance(expected_output.get("text"), str):
            raise ValueError("golden fixture expected_output.text is required")

        language = str(data.get("language") or data.get("lang") or "en")
        fixture_languages = (
            SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES | INDIC_NER_LANGUAGES
        )
        if language not in fixture_languages:
            raise ValueError(f"unsupported golden fixture language: {language!r}")

        text = str(data.get("text", ""))
        if not text:
            raise ValueError("golden fixture text is required")

        fixture_id = str(data.get("id") or data.get("fixture_id") or "")
        if not fixture_id:
            raise ValueError("golden fixture id is required")

        raw_spans = data.get("gold_spans") or []
        if not isinstance(raw_spans, list):
            raise ValueError("golden fixture gold_spans must be a list")
        if not raw_spans and category != HARD_NEGATIVE_CATEGORY:
            raise ValueError("golden fixture must include at least one gold span")
        _validate_raw_span_labels(raw_spans, language)

        gold_spans = tuple(
            normalize_eval_spans(raw_spans, default_language=language, source_text=text)
        )
        _validate_offsets(text, gold_spans)
        if category == HARD_NEGATIVE_CATEGORY:
            _validate_hard_negative_fixture(text, metadata, language)
        if category == CRITICAL_FINDINGS_CATEGORY:
            gold_spans = _validate_critical_finding_fixture(
                fixture_id,
                text,
                metadata,
                gold_spans,
            )

        return cls(
            fixture_id=fixture_id,
            category=category,
            language=language,
            text=text,
            gold_spans=gold_spans,
            expected_output=dict(expected_output),
            metadata=metadata,
        )

    def to_benchmark_fixture(self) -> BenchmarkFixture:
        """Return the harness-compatible fixture view."""
        return BenchmarkFixture(
            fixture_id=self.fixture_id,
            text=self.text,
            gold_spans=self.gold_spans,
            language=self.language,
            metadata=dict(self.metadata),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a stable JSON-ready mapping."""
        return {
            "id": self.fixture_id,
            "language": self.language,
            "text": self.text,
            "gold_spans": [_span_to_mapping(span) for span in self.gold_spans],
            "metadata": _plain_mapping(self.metadata),
        }


def list_fixture_paths(path: str | Path | None = None) -> tuple[Path, ...]:
    """Return fixture paths in deterministic order."""
    fixture_path = Path(path) if path is not None else _FIXTURE_DIR
    if fixture_path.is_file():
        return (fixture_path,)
    paths = [
        *(
            path
            for path in fixture_path.glob("*.json")
            if path.name not in _SPECIALIZED_FIXTURE_NAMES
        ),
        *(
            path
            for path in fixture_path.glob("**/*.jsonl")
            if path.name not in _SPECIALIZED_FIXTURE_NAMES
        ),
    ]
    if path is None:
        paths.extend(fixture for fixture in _TOP_LEVEL_FIXTURES if fixture.exists())
    return tuple(sorted(paths))


def load_golden_fixtures(path: str | Path | None = None) -> list[GoldenFixture]:
    """Load and validate all golden fixtures under *path*."""
    fixtures: list[GoldenFixture] = []
    fixture_paths: dict[str, Path] = {}
    for fixture_path in list_fixture_paths(path):
        if fixture_path.suffix.lower() == ".jsonl":
            rows = [
                json.loads(line)
                for line in fixture_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            raw = json.loads(fixture_path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise ValueError(f"{fixture_path} must contain a mapping")
            if raw.get("version") != _FIXTURE_VERSION:
                raise ValueError(f"{fixture_path} has unsupported fixture version")
            if raw.get("synthetic") is not True:
                raise ValueError(f"{fixture_path} must be marked synthetic")
            rows = raw.get("fixtures")
            if not isinstance(rows, list):
                raise ValueError(f"{fixture_path} must contain a fixtures list")

        for row in rows:
            fixture = GoldenFixture.from_mapping(row)
            first_path = fixture_paths.get(fixture.fixture_id)
            if first_path is not None:
                raise ValueError(
                    f"duplicate golden fixture id {fixture.fixture_id!r}: "
                    f"{first_path} and {fixture_path}"
                )
            fixture_paths[fixture.fixture_id] = fixture_path
            fixtures.append(fixture)
    return fixtures


def load_benchmark_fixtures(path: str | Path | None = None) -> list[BenchmarkFixture]:
    """Load golden fixtures as eval harness benchmark fixtures."""
    return [fixture.to_benchmark_fixture() for fixture in load_golden_fixtures(path)]


def benchmark_fixtures_by_language(
    fixtures: list[BenchmarkFixture] | None = None,
    *,
    category: str | None = None,
) -> dict[str, list[BenchmarkFixture]]:
    """Group benchmark fixtures by language in deterministic order."""
    source = fixtures if fixtures is not None else load_benchmark_fixtures()
    grouped: defaultdict[str, list[BenchmarkFixture]] = defaultdict(list)
    for fixture in source:
        if category is None or fixture.metadata.get("category") == category:
            grouped[fixture.language].append(fixture)
    return {
        language: sorted(rows, key=lambda fixture: fixture.fixture_id)
        for language, rows in sorted(grouped.items())
    }


def benchmark_fixture_languages(
    fixtures: list[BenchmarkFixture] | None = None,
    *,
    category: str | None = None,
) -> set[str]:
    """Return languages covered by benchmark fixtures."""
    return set(benchmark_fixtures_by_language(fixtures, category=category))


def fixtures_by_category(
    fixtures: list[GoldenFixture] | None = None,
) -> dict[str, list[GoldenFixture]]:
    """Group fixtures by golden category."""
    source = fixtures if fixtures is not None else load_golden_fixtures()
    grouped: defaultdict[str, list[GoldenFixture]] = defaultdict(list)
    for fixture in source:
        grouped[fixture.category].append(fixture)
    return dict(grouped)


def fixtures_by_language(
    fixtures: list[GoldenFixture] | None = None,
    *,
    category: str | None = None,
) -> dict[str, list[GoldenFixture]]:
    """Group fixtures by language, optionally restricted to one category."""
    source = fixtures if fixtures is not None else load_golden_fixtures()
    grouped: defaultdict[str, list[GoldenFixture]] = defaultdict(list)
    for fixture in source:
        if category is None or fixture.category == category:
            grouped[fixture.language].append(fixture)
    return dict(grouped)


def fixture_languages(
    fixtures: list[GoldenFixture] | None = None,
    *,
    category: str | None = None,
) -> set[str]:
    """Return languages covered by loaded fixtures."""
    source = fixtures if fixtures is not None else load_golden_fixtures()
    return {
        fixture.language
        for fixture in source
        if category is None or fixture.category == category
    }


def non_latin_golden_fixtures(
    fixtures: list[GoldenFixture] | None = None,
) -> list[GoldenFixture]:
    """Return synthetic golden fixtures containing non-Latin PHI spans."""
    source = fixtures if fixtures is not None else load_golden_fixtures()
    return sorted(
        (
            fixture
            for fixture in source
            if any(_has_non_latin_alpha(span.text) for span in fixture.gold_spans)
        ),
        key=lambda fixture: fixture.fixture_id,
    )


# ---------------------------------------------------------------------------
# Multi-annotator annotation imports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnnotationRelation:
    """A directed relation imported from one annotator's export."""

    document_id: str
    annotator_id: str
    relation_id: str
    relation_type: str
    source_id: str
    target_id: str
    source_span: EvalSpan
    target_span: EvalSpan
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def head(self) -> EvalSpan:
        """Return the source endpoint as a relation head."""
        return self.source_span

    @property
    def tail(self) -> EvalSpan:
        """Return the target endpoint as a relation tail."""
        return self.target_span

    @property
    def arg1_id(self) -> str:
        """Return the source endpoint annotation id."""
        return self.source_id

    @property
    def arg2_id(self) -> str:
        """Return the target endpoint annotation id."""
        return self.target_id

    @property
    def label(self) -> str:
        """Return the normalized relation type."""
        return self.relation_type

    def to_tuple(self) -> tuple[str, str, str]:
        """Return the compact relation triple used by adapter consumers."""
        return (self.relation_type, self.source_id, self.target_id)

    def to_eval_relation(self) -> Any:
        """Return the standard relation-metrics representation."""
        from openmed.eval.relation_metrics import EvalRelation

        return EvalRelation(
            relation_type=self.relation_type,
            head=self.source_span,
            tail=self.target_span,
            relation_id=self.relation_id,
            fixture_id=self.document_id,
            metadata={
                **dict(self.metadata),
                "annotator_id": self.annotator_id,
                "document_id": self.document_id,
                "source_id": self.source_id,
                "target_id": self.target_id,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready relation with both ids and endpoint spans."""
        return {
            "annotator_id": self.annotator_id,
            "document_id": self.document_id,
            "relation_id": self.relation_id,
            "relation_type": self.relation_type,
            "source_id": self.source_id,
            "source_span": _span_to_mapping(self.source_span),
            "target_id": self.target_id,
            "target_span": _span_to_mapping(self.target_span),
            "metadata": _plain_mapping(self.metadata),
        }


@dataclass(frozen=True)
class MultiAnnotatorGoldDocument:
    """A source document containing annotations from multiple reviewers."""

    document_id: str
    text: str
    spans: tuple[EvalSpan, ...]
    relations: tuple[AnnotationRelation, ...]
    annotators: tuple[str, ...]
    language: str = "en"
    source_format: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def spans_for(self, annotator_id: str) -> tuple[EvalSpan, ...]:
        """Return spans belonging to one annotator in stable import order."""
        identifier = str(annotator_id)
        return tuple(
            span
            for span in self.spans
            if str(span.metadata.get("annotator_id", "")) == identifier
        )

    def relations_for(self, annotator_id: str) -> tuple[AnnotationRelation, ...]:
        """Return relations belonging to one annotator in stable import order."""
        identifier = str(annotator_id)
        return tuple(
            relation
            for relation in self.relations
            if relation.annotator_id == identifier
        )

    @property
    def relation_triples(self) -> tuple[tuple[str, str, str], ...]:
        """Return all imported relation triples in document order."""
        return tuple(relation.to_tuple() for relation in self.relations)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready document without source file paths."""
        return {
            "annotators": list(self.annotators),
            "document_id": self.document_id,
            "language": self.language,
            "metadata": _plain_mapping(self.metadata),
            "relations": [relation.to_dict() for relation in self.relations],
            "source_format": self.source_format,
            "spans": [_span_to_mapping(span) for span in self.spans],
            "text": self.text,
        }


def load_brat_multi_annotator_document(
    text_path: str | Path,
    annotation_paths: Mapping[str, str | Path],
    document_id: str | None = None,
    language: str = "en",
) -> MultiAnnotatorGoldDocument:
    """Load a BRAT text file and one standoff file per annotator.

    annotation_paths maps an annotator id to that annotator's .ann file.
    Only the source text and annotation records are loaded; no network access
    or generated artifact is involved.
    """
    source_path = Path(text_path)
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read BRAT text file {source_path}") from exc
    resolved_document_id = document_id or source_path.stem
    return parse_brat_multi_annotator(
        text,
        {annotator: Path(path) for annotator, path in annotation_paths.items()},
        document_id=resolved_document_id,
        language=language,
    )


def parse_brat_multi_annotator(
    document_text: str,
    annotations: Mapping[str, str | bytes | Path],
    document_id: str = "document",
    language: str = "en",
) -> MultiAnnotatorGoldDocument:
    """Parse BRAT standoff content for multiple annotators.

    Text-bound records and directed R relation records are supported. BRAT
    attributes, notes, normalizations, events, and equivalence records are
    ignored because they do not map to the span/relation schema. All
    text-bound offsets and relation endpoints are validated.
    """
    if not isinstance(document_text, str) or not document_text:
        raise ValueError("BRAT source text must be a non-empty string")
    resolved_document_id = _require_import_identifier(document_id, "BRAT document")
    if not isinstance(annotations, Mapping) or not annotations:
        raise ValueError("BRAT annotations must map at least one annotator")

    spans: list[EvalSpan] = []
    relations: list[AnnotationRelation] = []
    annotator_ids: list[str] = []
    seen_annotators: set[str] = set()
    for raw_annotator_id, annotation_source in sorted(
        annotations.items(), key=lambda item: str(item[0])
    ):
        annotator_id = _require_import_identifier(
            raw_annotator_id,
            "BRAT annotator",
        )
        if annotator_id in seen_annotators:
            raise ValueError(f"duplicate BRAT annotator id: {annotator_id!r}")
        seen_annotators.add(annotator_id)
        annotator_ids.append(annotator_id)
        content = _read_annotation_source(
            annotation_source,
            f"BRAT annotations for annotator {annotator_id}",
        )
        annotator_spans, annotator_relations = _parse_brat_annotation_content(
            content,
            document_text,
            document_id=resolved_document_id,
            annotator_id=annotator_id,
            language=language,
        )
        spans.extend(annotator_spans)
        relations.extend(annotator_relations)

    spans.sort(key=_import_span_sort_key)
    relations.sort(key=_import_relation_sort_key)
    return MultiAnnotatorGoldDocument(
        document_id=resolved_document_id,
        text=document_text,
        spans=tuple(spans),
        relations=tuple(relations),
        annotators=tuple(sorted(annotator_ids)),
        language=str(language or "en"),
        source_format="brat",
        metadata={
            "annotator_count": len(annotator_ids),
            "document_id": resolved_document_id,
            "source_format": "brat",
        },
    )


def load_label_studio_multi_annotator_export(
    path: str | Path,
    default_language: str = "en",
) -> list[MultiAnnotatorGoldDocument]:
    """Load a Label Studio JSON export from a local file."""
    export_path = Path(path)
    try:
        payload = json.loads(export_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"could not read Label Studio JSON export {export_path}"
        ) from exc
    return parse_label_studio_multi_annotator_export(
        payload,
        default_language=default_language,
    )


def parse_label_studio_multi_annotator_export(
    payload: Any,
    default_language: str = "en",
) -> list[MultiAnnotatorGoldDocument]:
    """Parse Label Studio tasks and their annotator completions.

    Both current annotations and legacy completions task keys are accepted.
    Label Studio span results and directed relation results are normalized into
    the same document view used by the BRAT adapter.
    """
    payload = _load_label_studio_payload(payload)
    tasks = _label_studio_tasks(payload)
    documents: list[MultiAnnotatorGoldDocument] = []
    for task_index, task in enumerate(tasks, start=1):
        documents.append(
            _parse_label_studio_task(
                task,
                task_index=task_index,
                default_language=default_language,
            )
        )
    return documents


def _parse_brat_annotation_content(
    content: str,
    document_text: str,
    *,
    document_id: str,
    annotator_id: str,
    language: str,
) -> tuple[list[EvalSpan], list[AnnotationRelation]]:
    spans: list[EvalSpan] = []
    span_by_id: dict[str, EvalSpan] = {}
    raw_relations: list[tuple[int, str, str, str, str]] = []
    seen_ids: set[str] = set()

    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t", 2)
        annotation_id = fields[0].strip() if fields else ""
        if not annotation_id:
            raise ValueError(
                f"BRAT annotator {annotator_id} line {line_number} "
                "is missing an annotation id"
            )
        record_kind = annotation_id[:1].upper()
        if record_kind in {"#", "A", "M", "N", "E", "*"}:
            continue
        if record_kind not in {"T", "R"}:
            raise ValueError(
                f"BRAT annotator {annotator_id} line {line_number} "
                f"has unsupported record id {annotation_id!r}"
            )
        if annotation_id in seen_ids:
            raise ValueError(
                f"BRAT annotator {annotator_id} line {line_number} "
                f"duplicates annotation id {annotation_id!r}"
            )
        seen_ids.add(annotation_id)

        if record_kind == "T":
            if len(fields) != 3:
                raise ValueError(
                    f"BRAT annotator {annotator_id} line {line_number} "
                    "text-bound records require three tab-separated fields"
                )
            spec = fields[1].split()
            if len(spec) != 3:
                if any(";" in token for token in spec):
                    reason = (
                        "discontinuous text-bound spans are not supported by EvalSpan"
                    )
                else:
                    reason = "expected '<label> <start> <end>' offsets"
                raise ValueError(
                    f"BRAT annotator {annotator_id} line {line_number} "
                    f"text-bound {annotation_id}: {reason}"
                )
            raw_label, raw_start, raw_end = spec
            start = _parse_import_int(
                raw_start,
                f"BRAT annotator {annotator_id} line {line_number} "
                f"{annotation_id} start",
            )
            end = _parse_import_int(
                raw_end,
                f"BRAT annotator {annotator_id} line {line_number} {annotation_id} end",
            )
            _validate_import_offsets(
                document_text,
                start,
                end,
                fields[2],
                f"BRAT annotator {annotator_id} line {line_number} "
                f"text-bound {annotation_id}",
            )
            span = EvalSpan(
                start=start,
                end=end,
                label=normalize_label(raw_label, language),
                text=fields[2],
                language=str(language or "en"),
                metadata={
                    "annotator_id": annotator_id,
                    "document_id": document_id,
                    "source_annotation_id": annotation_id,
                    "source_format": "brat",
                    "source_label": raw_label,
                },
            )
            span_by_id[annotation_id] = span
            spans.append(span)
            continue

        if len(fields) != 2:
            raise ValueError(
                f"BRAT annotator {annotator_id} line {line_number} "
                f"relation {annotation_id} requires two tab-separated fields"
            )
        relation_parts = fields[1].split()
        if len(relation_parts) < 2 or not relation_parts[0].strip():
            raise ValueError(
                f"BRAT annotator {annotator_id} line {line_number} "
                f"relation {annotation_id} requires a type and Arg1/Arg2"
            )
        raw_relation_type = relation_parts[0].strip()
        endpoints: dict[str, str] = {}
        for argument in relation_parts[1:]:
            role, separator, endpoint = argument.partition(":")
            if not separator:
                continue
            role_key = role.casefold()
            if role_key not in {"arg1", "arg2"}:
                continue
            if role_key in endpoints:
                raise ValueError(
                    f"BRAT annotator {annotator_id} line {line_number} "
                    f"relation {annotation_id} repeats {role}"
                )
            endpoints[role_key] = endpoint.strip()
        raw_relations.append(
            (
                line_number,
                annotation_id,
                raw_relation_type,
                endpoints.get("arg1", ""),
                endpoints.get("arg2", ""),
            )
        )

    relations = [
        _build_annotation_relation(
            document_id=document_id,
            annotator_id=annotator_id,
            relation_id=relation_id,
            relation_type=raw_relation_type,
            source_id=source_id,
            target_id=target_id,
            span_by_id=span_by_id,
            source_format="brat",
            line_number=line_number,
        )
        for line_number, relation_id, raw_relation_type, source_id, target_id in raw_relations
    ]
    return spans, relations


def _parse_label_studio_task(
    task: Mapping[str, Any],
    *,
    task_index: int,
    default_language: str,
) -> MultiAnnotatorGoldDocument:
    if not isinstance(task, Mapping):
        raise ValueError(f"Label Studio task {task_index} must be a mapping")
    raw_data = task.get("data") or {}
    if not isinstance(raw_data, Mapping):
        raise ValueError(f"Label Studio task {task_index} data must be a mapping")
    text = _label_studio_text(raw_data, task_index)
    document_id = _label_studio_document_id(task, raw_data, task_index)
    language = str(
        raw_data.get("language") or task.get("language") or default_language or "en"
    )
    raw_annotations = task.get("annotations")
    if raw_annotations is None:
        raw_annotations = task.get("completions")
    if raw_annotations is None:
        raw_annotations = []
    if isinstance(raw_annotations, Mapping):
        raw_annotations = [raw_annotations]
    if not isinstance(raw_annotations, Sequence) or isinstance(
        raw_annotations, (str, bytes)
    ):
        raise ValueError(f"Label Studio task {task_index} annotations must be a list")

    spans: list[EvalSpan] = []
    relations: list[AnnotationRelation] = []
    annotator_ids: list[str] = []
    for annotation_index, raw_annotation in enumerate(raw_annotations, start=1):
        if not isinstance(raw_annotation, Mapping):
            raise ValueError(
                f"Label Studio task {task_index} annotation "
                f"{annotation_index} must be a mapping"
            )
        if raw_annotation.get("was_cancelled") is True:
            continue
        annotator_id = _label_studio_annotator_id(
            raw_annotation,
            annotation_index,
        )
        annotator_ids.append(annotator_id)
        raw_results = raw_annotation.get("result") or []
        if not isinstance(raw_results, Sequence) or isinstance(
            raw_results, (str, bytes)
        ):
            raise ValueError(
                f"Label Studio task {task_index} annotation "
                f"{annotation_index} result must be a list"
            )

        span_by_id: dict[str, EvalSpan] = {}
        relation_results: list[tuple[int, Mapping[str, Any]]] = []
        seen_result_ids: set[str] = set()
        for result_index, result in enumerate(raw_results, start=1):
            if not isinstance(result, Mapping):
                raise ValueError(
                    f"Label Studio task {task_index} annotation "
                    f"{annotation_index} result {result_index} must be a mapping"
                )
            result_id = str(result.get("id") or "").strip()
            if result_id and result_id in seen_result_ids:
                raise ValueError(
                    f"duplicate Label Studio result id {result_id!r} for {annotator_id}"
                )
            if result_id:
                seen_result_ids.add(result_id)
            result_type = str(result.get("type") or "").casefold()
            if result_type == "relation":
                relation_results.append((result_index, result))
                continue
            if _is_label_studio_span_result(result):
                source_id, span = _parse_label_studio_span_result(
                    result,
                    text,
                    document_id=document_id,
                    annotator_id=annotator_id,
                    language=language,
                    task_index=task_index,
                    annotation_index=annotation_index,
                    result_index=result_index,
                )
                if source_id in span_by_id:
                    raise ValueError(
                        f"duplicate Label Studio span id {source_id!r} for "
                        f"{annotator_id}"
                    )
                span_by_id[source_id] = span

        spans.extend(span_by_id.values())
        for result_index, result in relation_results:
            relations.append(
                _parse_label_studio_relation_result(
                    result,
                    span_by_id,
                    document_id=document_id,
                    annotator_id=annotator_id,
                    task_index=task_index,
                    annotation_index=annotation_index,
                    result_index=result_index,
                )
            )

    spans.sort(key=_import_span_sort_key)
    relations.sort(key=_import_relation_sort_key)
    return MultiAnnotatorGoldDocument(
        document_id=document_id,
        text=text,
        spans=tuple(spans),
        relations=tuple(relations),
        annotators=tuple(sorted(set(annotator_ids))),
        language=language,
        source_format="label_studio",
        metadata=_label_studio_task_metadata(
            task,
            raw_data,
            document_id=document_id,
            annotation_count=len(annotator_ids),
        ),
    )


def _parse_label_studio_span_result(
    result: Mapping[str, Any],
    document_text: str,
    *,
    document_id: str,
    annotator_id: str,
    language: str,
    task_index: int,
    annotation_index: int,
    result_index: int,
) -> tuple[str, EvalSpan]:
    source_id = str(result.get("id") or "").strip()
    context = (
        f"Label Studio task {task_index} annotation {annotation_index} "
        f"result {result_index}"
    )
    if not source_id:
        raise ValueError(f"{context} span result id is required")
    value = result.get("value") or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} span value must be a mapping")
    start = _parse_import_int(value.get("start"), f"{context} span start")
    end = _parse_import_int(value.get("end"), f"{context} span end")
    surface = _label_studio_surface_text(
        value.get("text"),
        document_text,
        start,
        end,
    )
    raw_label = _first_label(
        value.get("labels")
        or value.get("paragraphlabels")
        or value.get("hypertextlabels")
        or value.get("label")
        or result.get("labels")
        or result.get("label"),
    )
    _validate_import_offsets(document_text, start, end, surface, context)
    metadata: dict[str, Any] = {
        "annotator_id": annotator_id,
        "document_id": document_id,
        "source_annotation_id": source_id,
        "source_format": "label_studio",
        "source_label": raw_label,
    }
    for key in ("from_name", "to_name", "type"):
        if result.get(key) is not None:
            metadata[key] = result[key]
    return (
        source_id,
        EvalSpan(
            start=start,
            end=end,
            label=normalize_label(raw_label, language),
            text=surface,
            language=language,
            metadata=metadata,
        ),
    )


def _parse_label_studio_relation_result(
    result: Mapping[str, Any],
    span_by_id: Mapping[str, EvalSpan],
    *,
    document_id: str,
    annotator_id: str,
    task_index: int,
    annotation_index: int,
    result_index: int,
) -> AnnotationRelation:
    relation_id = str(result.get("id") or "").strip()
    context = (
        f"Label Studio task {task_index} annotation {annotation_index} "
        f"result {result_index}"
    )
    if not relation_id:
        raise ValueError(f"{context} relation result id is required")
    value = result.get("value") or {}
    if value is not None and not isinstance(value, Mapping):
        raise ValueError(f"{context} relation value must be a mapping")
    value_mapping = value if isinstance(value, Mapping) else {}
    source_id = str(
        result.get("from_id")
        or result.get("source")
        or value_mapping.get("from_id")
        or value_mapping.get("source")
        or ""
    ).strip()
    target_id = str(
        result.get("to_id")
        or result.get("target")
        or value_mapping.get("to_id")
        or value_mapping.get("target")
        or ""
    ).strip()
    raw_relation_type = _first_label(
        value_mapping.get("labels")
        or value_mapping.get("label")
        or value_mapping.get("relation")
        or result.get("labels")
        or result.get("label")
        or result.get("relation"),
        default="RELATED_TO",
    )
    metadata = {
        key: result[key]
        for key in ("direction", "from_name", "to_name")
        if result.get(key) is not None
    }
    return _build_annotation_relation(
        document_id=document_id,
        annotator_id=annotator_id,
        relation_id=relation_id,
        relation_type=raw_relation_type,
        source_id=source_id,
        target_id=target_id,
        span_by_id=span_by_id,
        source_format="label_studio",
        context=context,
        metadata=metadata,
    )


def _build_annotation_relation(
    *,
    document_id: str,
    annotator_id: str,
    relation_id: str,
    relation_type: str,
    source_id: str,
    target_id: str,
    span_by_id: Mapping[str, EvalSpan],
    source_format: str,
    context: str = "",
    line_number: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AnnotationRelation:
    missing: list[str] = []
    if not source_id:
        missing.append("Arg1/source_id")
    elif source_id not in span_by_id:
        missing.append(source_id)
    if not target_id:
        missing.append("Arg2/target_id")
    elif target_id not in span_by_id:
        missing.append(target_id)
    if missing:
        location = context or (f"line {line_number}" if line_number else "")
        location_suffix = f" ({location})" if location else ""
        raise ValueError(
            f"missing {source_format} relation endpoint(s) for "
            f"annotator {annotator_id} relation {relation_id}{location_suffix}: "
            f"{', '.join(missing)}"
        )
    normalized_type = _normalize_import_relation_type(relation_type)
    relation_metadata = {
        **dict(metadata or {}),
        "annotator_id": annotator_id,
        "document_id": document_id,
        "source_annotation_id": relation_id,
        "source_format": source_format,
        "source_id": source_id,
        "source_relation_type": str(relation_type),
        "target_id": target_id,
    }
    return AnnotationRelation(
        document_id=document_id,
        annotator_id=annotator_id,
        relation_id=relation_id,
        relation_type=normalized_type,
        source_id=source_id,
        target_id=target_id,
        source_span=span_by_id[source_id],
        target_span=span_by_id[target_id],
        metadata=relation_metadata,
    )


def _label_studio_tasks(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        raw_tasks = payload.get("tasks")
        if raw_tasks is None:
            raw_tasks = [payload]
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        raw_tasks = payload
    else:
        raise ValueError("Label Studio export must be a task mapping or list")
    if isinstance(raw_tasks, Mapping):
        raw_tasks = [raw_tasks]
    if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes)):
        raise ValueError("Label Studio export tasks must be a list")
    tasks: list[Mapping[str, Any]] = []
    for index, task in enumerate(raw_tasks, start=1):
        if not isinstance(task, Mapping):
            raise ValueError(f"Label Studio task {index} must be a mapping")
        tasks.append(task)
    if not tasks:
        raise ValueError("Label Studio export must contain at least one task")
    return tasks


def _load_label_studio_payload(payload: Any) -> Any:
    if isinstance(payload, Path):
        try:
            return json.loads(payload.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read Label Studio payload {payload}") from exc
    if isinstance(payload, bytes):
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Label Studio payload must contain valid JSON") from exc
    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped[:1] in {"[", "{"}:
            try:
                return json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Label Studio payload must contain valid JSON"
                ) from exc
        try:
            candidate = Path(payload)
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read Label Studio payload {payload}") from exc
        raise ValueError("Label Studio string payload must be JSON or an existing file")
    return payload


def _label_studio_text(data: Mapping[str, Any], task_index: int) -> str:
    for key in ("text", "source_text", "document_text", "document", "raw_text"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    candidates = [
        value
        for key, value in data.items()
        if key not in {"document_id", "doc_id", "id", "language"}
        and isinstance(value, str)
        and value
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Label Studio task {task_index} data must include source text")


def _label_studio_document_id(
    task: Mapping[str, Any],
    data: Mapping[str, Any],
    task_index: int,
) -> str:
    for source in (data, task):
        for key in ("document_id", "doc_id", "id"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    raise ValueError(f"Label Studio task {task_index} document id is required")


def _label_studio_task_metadata(
    task: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    document_id: str,
    annotation_count: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "annotation_count": annotation_count,
        "document_id": document_id,
        "source_format": "label_studio",
    }
    for source in (task, data):
        for key in ("synthetic", "contains_real_phi", "source", "source_dataset"):
            if key in source:
                metadata[key] = source[key]
    return metadata


def _label_studio_annotator_id(
    annotation: Mapping[str, Any],
    fallback_index: int,
) -> str:
    completed_by = annotation.get("completed_by") or annotation.get("created_by")
    if isinstance(completed_by, Mapping):
        for key in ("username", "email", "id"):
            value = completed_by.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    if completed_by is not None and str(completed_by).strip():
        return str(completed_by).strip()
    annotation_id = annotation.get("id")
    if annotation_id is not None and str(annotation_id).strip():
        return str(annotation_id).strip()
    return f"annotator-{fallback_index}"


def _is_label_studio_span_result(result: Mapping[str, Any]) -> bool:
    result_type = str(result.get("type") or "").casefold()
    if result_type in {"labels", "hypertextlabels", "paragraphlabels"}:
        return True
    value = result.get("value")
    return (
        isinstance(value, Mapping)
        and "start" in value
        and "end" in value
        and any(
            key in value or key in result
            for key in (
                "labels",
                "paragraphlabels",
                "hypertextlabels",
                "label",
            )
        )
    )


def _label_studio_surface_text(
    value: Any,
    document_text: str,
    start: int,
    end: int,
) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        strings = [item for item in value if isinstance(item, str)]
        if len(strings) == 1:
            return strings[0]
    return document_text[start:end] if 0 <= start <= end <= len(document_text) else ""


def _first_label(value: Any, *, default: str | None = None) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    if default is not None:
        return default
    raise ValueError("annotation label is required")


def _read_annotation_source(source: str | bytes | Path, context: str) -> str:
    if isinstance(source, Path):
        try:
            return source.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"could not read {context} file {source}") from exc
    if isinstance(source, bytes):
        try:
            return source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{context} must be valid UTF-8") from exc
    if isinstance(source, str):
        if "\n" not in source and "\r" not in source and "\t" not in source:
            try:
                candidate = Path(source)
                if candidate.is_file():
                    return candidate.read_text(encoding="utf-8")
            except OSError:
                pass
        return source
    raise ValueError(f"{context} must be annotation text or a file path")


def _require_import_identifier(value: Any, context: str) -> str:
    identifier = str(value or "").strip()
    if not identifier:
        raise ValueError(f"{context} id is required")
    return identifier


def _normalize_import_relation_type(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("relation type is required")
    return "_".join(
        part
        for part in raw.replace("-", "_").replace(" ", "_").upper().split("_")
        if part
    )


def _parse_import_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _validate_import_offsets(
    document_text: str,
    start: int,
    end: int,
    expected_text: str,
    context: str,
) -> None:
    if start < 0 or end <= start or end > len(document_text):
        raise ValueError(
            f"{context} has invalid offsets {start}:{end}; "
            f"source text length is {len(document_text)}"
        )
    actual_text = document_text[start:end]
    if actual_text != expected_text:
        raise ValueError(f"{context} span text mismatch at offsets {start}:{end}")


def _import_span_sort_key(span: EvalSpan) -> tuple[str, int, int, str, str]:
    return (
        str(span.metadata.get("annotator_id", "")),
        span.start,
        span.end,
        span.label,
        str(span.metadata.get("source_annotation_id", "")),
    )


def _import_relation_sort_key(
    relation: AnnotationRelation,
) -> tuple[str, str, str, str, str]:
    return (
        relation.annotator_id,
        relation.relation_id,
        relation.relation_type,
        relation.source_id,
        relation.target_id,
    )


def _validate_raw_span_labels(raw_spans: list[Any], language: str) -> None:
    for raw_span in raw_spans:
        if not isinstance(raw_span, Mapping):
            raise ValueError("gold span must be a mapping")
        raw_label = raw_span.get("label") or raw_span.get("canonical_label")
        if not isinstance(raw_label, str):
            raise ValueError("gold span label is required")
        canonical = normalize_label(raw_label, language)
        if canonical != raw_label or canonical not in CANONICAL_LABELS:
            raise ValueError(f"gold span label must be canonical: {raw_label!r}")


def _validate_critical_finding_fixture(
    fixture_id: str,
    text: str,
    metadata: Mapping[str, Any],
    spans: tuple[EvalSpan, ...],
) -> tuple[EvalSpan, ...]:
    disclaimer = str(metadata.get("medical_device_disclaimer") or "")
    normalized_disclaimer = disclaimer.lower()
    if (
        "assistive safety probe" not in normalized_disclaimer
        or "not clinical ground truth" not in normalized_disclaimer
    ):
        raise ValueError(
            "critical finding fixtures require a medical_device_disclaimer "
            "noting the set is an assistive safety probe, not clinical ground truth"
        )

    source = str(metadata.get("source") or metadata.get("source_dataset") or "")
    if _is_dua_source_marker(source):
        raise ValueError("critical finding fixtures must not reference DUA sources")

    validated: list[EvalSpan] = []
    for span in spans:
        category = critical_finding_category(span)
        if category is None:
            raise ValueError(
                "critical finding gold spans require critical_finding_category"
            )
        category = normalize_critical_finding_category(category)
        if category not in CRITICAL_FINDING_CATEGORIES:
            raise ValueError(f"unknown critical finding category: {category!r}")
        span_fixture_id = span.metadata.get("fixture_id")
        if span_fixture_id is not None and str(span_fixture_id) != fixture_id:
            raise ValueError("critical finding span fixture_id must match fixture id")
        span_metadata = dict(span.metadata)
        span_metadata["critical_finding"] = True
        span_metadata["critical_finding_category"] = category
        span_metadata["fixture_id"] = fixture_id
        validated.append(replace(span, metadata=span_metadata))

    if not validated:
        raise ValueError("critical finding fixture must include critical gold spans")
    _validate_offsets(text, tuple(validated))
    return tuple(validated)


def _validate_hard_negative_fixture(
    text: str,
    metadata: Mapping[str, Any],
    language: str,
) -> None:
    candidates = metadata.get("hard_negative_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(
            "hard negative fixture metadata.hard_negative_candidates is required"
        )
    source = str(metadata.get("source") or metadata.get("source_dataset") or "")
    if _is_dua_source_marker(source):
        raise ValueError("hard negative fixtures must not reference DUA sources")

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("hard negative candidate must be a mapping")
        start = _int_field(candidate, "start")
        end = _int_field(candidate, "end")
        if start < 0 or end <= start or end > len(text):
            raise ValueError("hard negative candidate has invalid offsets")
        candidate_text = str(candidate.get("text", ""))
        if text[start:end] != candidate_text:
            raise ValueError("hard negative candidate text must match offsets")
        raw_label = candidate.get("label")
        if not isinstance(raw_label, str):
            raise ValueError("hard negative candidate label is required")
        canonical = normalize_label(raw_label, language)
        if canonical != raw_label or canonical not in CANONICAL_LABELS:
            raise ValueError(
                f"hard negative candidate label must be canonical: {raw_label!r}"
            )
        if candidate.get("synthetic") is not True:
            raise ValueError("hard negative candidate synthetic must be true")
        candidate_source = str(
            candidate.get("source_dataset")
            or candidate.get("source")
            or candidate.get("source_shard_id")
            or ""
        )
        if _is_dua_source_marker(candidate_source):
            raise ValueError("hard negative candidates must not reference DUA sources")
        difficulty = candidate.get("difficulty_score")
        if difficulty is not None:
            try:
                difficulty_value = float(difficulty)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "hard negative candidate difficulty_score must be numeric"
                ) from exc
            if not 0.0 <= difficulty_value <= 1.0:
                raise ValueError(
                    "hard negative candidate difficulty_score must be in [0, 1]"
                )


# ---------------------------------------------------------------------------
# Multi-annotator consensus corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsensusRelation:
    """An adjudicated relation between two consensus spans."""

    relation_type: str
    head: EvalSpan
    tail: EvalSpan
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_type": self.relation_type,
            "label": self.label,
            "head": _span_to_mapping(self.head),
            "tail": _span_to_mapping(self.tail),
        }


@dataclass(frozen=True)
class ConsensusDocument:
    """A synthetic document with per-annotator exports and a consensus view."""

    doc_id: str
    text: str
    annotators: Mapping[str, tuple[EvalSpan, ...]]
    annotator_relations: Mapping[str, tuple[ConsensusRelation, ...]]
    consensus_spans: tuple[EvalSpan, ...]
    consensus_relations: tuple[ConsensusRelation, ...]


def _require_synthetic(payload: Mapping[str, Any], where: str) -> None:
    if payload.get("synthetic") is not True:
        raise ValueError(f"{where} must be explicitly marked synthetic")


def _consensus_spans(raw_spans: Any, text: str) -> tuple[EvalSpan, ...]:
    spans = tuple(normalize_eval_spans(raw_spans or [], source_text=text))
    _validate_offsets(text, spans)
    return spans


def _consensus_relation(raw: Mapping[str, Any], text: str) -> ConsensusRelation:
    if not isinstance(raw, Mapping):
        raise ValueError("consensus relation must be a mapping")
    head = _consensus_spans([raw["head"]], text)[0]
    tail = _consensus_spans([raw["tail"]], text)[0]
    relation_type = str(raw.get("relation_type", "")).strip()
    if not relation_type:
        raise ValueError("consensus relation type is required")
    return ConsensusRelation(
        relation_type=relation_type,
        head=head,
        tail=tail,
        label=str(raw.get("label", "")),
    )


def _span_key(span: EvalSpan) -> tuple[int, int, str]:
    return span.start, span.end, span.label


def _consensus_relations(
    raw_relations: Any,
    text: str,
    spans: tuple[EvalSpan, ...],
    where: str,
) -> tuple[ConsensusRelation, ...]:
    if raw_relations is None:
        return ()
    if not isinstance(raw_relations, list):
        raise ValueError(f"{where} relations must be a list")

    relations = tuple(_consensus_relation(relation, text) for relation in raw_relations)
    span_keys = {_span_key(span) for span in spans}
    for relation in relations:
        if _span_key(relation.head) not in span_keys:
            raise ValueError(f"{where} relation head must reference one of its spans")
        if _span_key(relation.tail) not in span_keys:
            raise ValueError(f"{where} relation tail must reference one of its spans")
    return relations


def _consensus_document(data: Mapping[str, Any]) -> ConsensusDocument:
    if not isinstance(data, Mapping):
        raise ValueError("consensus record must be a mapping")
    _require_synthetic(data, "consensus document")

    text = str(data.get("text", ""))
    if not text:
        raise ValueError("consensus document text is required")

    raw_annotators = data.get("annotators") or {}
    if not isinstance(raw_annotators, Mapping) or len(raw_annotators) < 2:
        raise ValueError("consensus document requires at least two annotators")

    annotators: dict[str, tuple[EvalSpan, ...]] = {}
    annotator_relations: dict[str, tuple[ConsensusRelation, ...]] = {}
    for name, export in raw_annotators.items():
        if not isinstance(export, Mapping):
            raise ValueError("annotator export must be a mapping")
        _require_synthetic(export, f"annotator {name!r} export")
        annotator_name = str(name)
        annotator_spans = _consensus_spans(export.get("spans"), text)
        annotators[annotator_name] = annotator_spans
        annotator_relations[annotator_name] = _consensus_relations(
            export.get("relations"),
            text,
            annotator_spans,
            f"annotator {name!r}",
        )

    consensus = data.get("consensus") or {}
    if not isinstance(consensus, Mapping):
        raise ValueError("consensus view must be a mapping")
    consensus_spans = _consensus_spans(consensus.get("spans"), text)
    consensus_relations = _consensus_relations(
        consensus.get("relations"),
        text,
        consensus_spans,
        "consensus",
    )

    doc_id = str(data.get("id") or data.get("doc_id") or "")
    if not doc_id:
        raise ValueError("consensus document id is required")

    return ConsensusDocument(
        doc_id=doc_id,
        text=text,
        annotators=annotators,
        annotator_relations=annotator_relations,
        consensus_spans=consensus_spans,
        consensus_relations=consensus_relations,
    )


def load_consensus_corpus(
    path: str | Path | None = None,
) -> list[ConsensusDocument]:
    """Load the synthetic multi-annotator consensus corpus.

    Each record carries source text, at least two synthetic annotator exports,
    and an adjudicated consensus view of spans and relations. Every document and
    annotation must be explicitly marked synthetic, and all span offsets are
    validated against the document text.
    """

    corpus_path = Path(path) if path is not None else _CONSENSUS_CORPUS
    documents: list[ConsensusDocument] = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            documents.append(_consensus_document(json.loads(line)))
    return documents


def _validate_offsets(text: str, spans: tuple[EvalSpan, ...]) -> None:
    for span in spans:
        if span.start < 0 or span.end <= span.start or span.end > len(text):
            raise ValueError(f"gold span has invalid offsets: {span!r}")
        actual_text = text[span.start : span.end]
        if span.text and actual_text != span.text:
            raise ValueError(
                f"gold span text mismatch for {span.label}: "
                f"{span.text!r} != {actual_text!r}"
            )


def _span_to_mapping(span: EvalSpan) -> dict[str, Any]:
    row: dict[str, Any] = {
        "start": span.start,
        "end": span.end,
        "label": span.label,
        "text": span.text,
    }
    metadata = dict(span.metadata)
    group = metadata.pop("group", None)
    if group is not None and str(group).strip():
        row["group"] = str(group).strip()
    if metadata:
        row["metadata"] = _plain_mapping(metadata)
    return row


def _int_field(payload: Mapping[str, Any], field: str) -> int:
    try:
        return int(payload[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _is_dua_source_marker(value: str) -> bool:
    markers = {"dua", "i2b2", "n2c2", "mimic"}
    parts = {
        part.strip().lower()
        for part in value.replace("_", "-").replace(".", "-").split("-")
    }
    return bool(parts & markers)


def _has_non_latin_alpha(value: str) -> bool:
    return any(ord(char) > 127 and char.isalpha() for char in value)


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain(value[key]) for key in sorted(value, key=str)}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "AnnotationRelation",
    "CRITICAL_FINDINGS_CATEGORY",
    "GOLDEN_CATEGORIES",
    "HARD_NEGATIVE_CATEGORY",
    "ConsensusDocument",
    "ConsensusRelation",
    "GoldenFixture",
    "MultiAnnotatorGoldDocument",
    "benchmark_fixture_languages",
    "benchmark_fixtures_by_language",
    "fixture_languages",
    "fixtures_by_category",
    "fixtures_by_language",
    "list_fixture_paths",
    "load_brat_multi_annotator_document",
    "load_benchmark_fixtures",
    "load_consensus_corpus",
    "load_golden_fixtures",
    "load_label_studio_multi_annotator_export",
    "non_latin_golden_fixtures",
    "parse_brat_multi_annotator",
    "parse_label_studio_multi_annotator_export",
]
